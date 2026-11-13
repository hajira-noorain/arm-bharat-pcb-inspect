# PCB Inspection Station: offline defect inspection with on-device Llama advice

Arm Bharat AI-SoC Challenge 2026-27, Problem Statement 5 (real-time industrial defect detection and quality-control assistant).

A Raspberry Pi 5 with a camera inspects printed circuit boards. Everything runs on the Pi's Arm CPU through ExecuTorch, with no cloud. For each board it:

- **finds and names defects** on bare boards (8 types), and flags anything unusual on assembled boards;
- **estimates each defect's size in millimetres** by comparing with a known-good image of the same design. Every value carries a ±1 pixel uncertainty, and when a limit falls inside that band the station says so instead of guessing;
- **gives a disposition for the customer's IPC class**: ACCEPT, REWORK, SCRAP, or REVIEW by a person;
- **returns a verdict**: PASS, FAIL, or CHECK (a person decides). The pass threshold is set by conformal calibration, so that *on boards like the calibration set* the share of defective boards passed is at most the chosen rate, on average;
- **asks Llama 3.2 (on ExecuTorch) for advice**: Llama picks a likely cause and next action from a defect knowledge base and writes a short summary. A checker rejects any answer that is not supported by the findings;
- **logs everything** on the device and lights a green, amber or red stack light;
- **learns on the floor**:
  - a new design is taught from good boards;
  - operator corrections train a small head on top of the detector. It is switched on only if checks on the operators' own labels and on calibration boards show it does not hide defects.

## How it covers PS5

| PS5 requirement | What the station does | Where |
|---|---|---|
| Detect surface defects, cracks, anomalies | YOLO26n (8 bare-board defect types), plus a patch-memory anomaly flag for assembled boards | `pcbinspect/detect_bare.py`, `pcbinspect/anomaly.py` |
| Classify type and severity | Defect type from the detector. Size from golden-board comparison against IPC-style class limits gives a disposition for the customer's class | `pcbinspect/golden.py`, `pcbinspect/severity.py`, `config/severity.yaml` |
| Alert operators with actionable recommendations | Verdict band, GPIO stack light and buzzer, and Llama advice grounded in `kb/defects.yaml` | `pcbinspect/app.py`, `pcbinspect/advisor.py`, `pcbinspect/grounding.py` |
| Log inspection results | SQLite on the device (images, defects, dispositions, operator decisions, advice, model versions), plus CSV export | `pcbinspect/store.py` |
| Real time at the edge, no cloud | All models run as ExecuTorch `.pte` files with the XNNPACK delegate on the Pi 5 CPU. Measured timings are in `docs/RESULTS.md` | `bench/bench_device.py`, `bench/soak.py` |
| Llama via ExecuTorch for reasoning | Llama 3.2 Instruct (Meta's SpinQuant INT4 checkpoint) exported with ExecuTorch `export_llm` and run through `TextLLMRunner` | `pcbinspect/llm.py`, `llm/` |

## Two pipelines

```
VISION (ExecuTorch + XNNPACK, Arm CPU)                 REASONING (ExecuTorch, Llama 3.2, Arm CPU)
camera -> ArUco scale + QR board ID                    structured defect record (no pixels)
  -> YOLO26n on tiles (.pte) -> named defects            + knowledge-base options for each defect
  -> ResNet patch memory (.pte) -> unusual areas         + recent history on this line
  -> golden-board comparison -> size in mm (+/-1 px)   -> Llama picks a cause/action for each defect,
  -> IPC-style class limits -> disposition                then writes a 1-2 sentence summary
  -> conformal gate -> PASS / FAIL / CHECK             -> checker: options valid, numbers and defect
  -> SQLite log, stack light                              types must come from the record
                                                       -> operator reads knowledge-base text verbatim
```

The verdict never waits for Llama. Advice arrives on a background thread a few seconds later.

## What is and is not validated

| Measured | Not yet validated |
|---|---|
| Detector accuracy on held-out boards from three public datasets, split by physical board | Accuracy on a live camera with our own boards |
| Conformal gate behaviour on held-out boards (where the calibration set is large enough) | Copper segmentation and mm measurement on solder-masked boards under shop lighting |
| Anomaly memory on VisA pcb1-4 (official split), including a few-shot curve | Component presence/polarity and solder-joint judgement on assembled boards (anomaly flag only) |
| Llama advice: grounding pass rate, option-position bias and latency, for 1B and 3B | Llama choosing a cause better than chance (measured: it does not, see RESULTS) |
| Raspberry Pi 5 timings, memory and a 30-minute soak (see RESULTS once run) | Long-term drift on a real line |

The public datasets are scans or close-up photos, and their defects are partly synthetic: PKU-Market-PCB defects were drawn in with an image editor, and DeepPCB's were partly added by hand. Dataset results and real-camera results are therefore reported separately.

## Run it

**On the Raspberry Pi 5** (Raspberry Pi OS 64-bit, with the active cooler fitted):

```bash
tar xf pcb-station.tar && cd pcb-station
bash deploy/pi_setup.sh          # once: apt packages + Python env with the aarch64 ExecuTorch wheel
python deploy/arm_check.py       # loads every model and runs it once
bash deploy/run_on_pi.sh         # camera auto-detect (Pi Camera, else USB webcam)
bash deploy/run_on_pi.sh replay  # or cycle through held-out demo boards
bash deploy/hotspot.sh           # optional: a phone joins "PCB-Station" and opens http://10.42.0.1:8000
```

Press **Inspect**, the space bar or a USB foot pedal. Print `deploy/assets/scale_marker_sheet.pdf` at 100%, check its ruler, and place a marker beside the board.

**Measurements:** `python -m bench.bench_device` and `python -m bench.soak --minutes 30`.

**Rebuilding the models** (on a PC):

- `training/prepare_bare.py` builds the dataset, and `train_bare.py` trains on a GPU.
- `eval_bare.py` evaluates and calibrates the detector.
- `build_visa_banks.py` builds the anomaly memories.
- `export_vision.py` and `llm/` produce the ExecuTorch exports. These run in `deploy/Dockerfile.export`.
- `deploy/Dockerfile.pi` is an arm64 Debian image that installs exactly what the Pi installs. It is used to check the Arm build without hardware.

**Results:** `docs/RESULTS.md` is generated from the result JSON files by `docs/make_results.py`. Research and design decisions are in `docs/RESEARCH_AND_DECISIONS.md`. The full write-up is in `docs/TECHNICAL_REPORT.md`.

## Data and licences

| Dataset | Use | Licence |
|---|---|---|
| [DeepPCB](https://github.com/tangsanli5201/DeepPCB) | Bare-board defects (binarised, partly hand-added) plus defect-free templates | Research use (per README) |
| [PKU-Market-PCB / HRIPCB](https://github.com/Ironbrotherstyle/PCB-DATASET) | Bare-board defects drawn in with an image editor, split by physical board | Research use |
| [PCB-Defect (Mendeley, 2025)](https://data.mendeley.com/datasets/vdj74sngvn/1) | Real etched boards, flatbed scans | See the dataset page |
| [VisA pcb1-4](https://github.com/amazon-science/spot-diff) | Assembled boards, anomaly detection | CC BY 4.0 |

Llama 3.2 is used under the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/). The IPC-style limits in `config/severity.yaml` come from a cited secondary source and are editable. They do not replace the licensed IPC standards or a customer's own specification.
