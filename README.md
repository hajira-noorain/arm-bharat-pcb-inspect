# PCB Inspection Station: an offline inspector that knows its own limits

Arm Bharat AI-SoC Challenge 2026-27, Problem Statement 5 (real-time industrial defect detection and quality-control assistant).

Every inspection system claims a high accuracy figure. This one **measures the smallest defect it can find on your camera, says what it cannot see, and refuses customer rules it cannot guarantee.** It runs on a Raspberry Pi 5's Arm CPU with no cloud: vision through ExecuTorch (or Arm's optimised YOLO26 runtime), Llama 3.2 through ExecuTorch.

For each board it:

- **finds and names defects** on bare boards (11 types) and flags anything unusual on assembled boards;
- **compares against a known-good board first**: this is the station's main evidence. It sees any change in the copper, checks connectivity (a short joins two conductors, an open splits one), **measures each defect in millimetres**, and tells the detector which parts of the picture to look at. If a board does not line up with its reference picture, it is never passed: a person looks;
- **grades severity** against IPC-style limits for the customer's class;
- **returns PASS, FAIL or CHECK** (a person decides). The operator picks a target for how many defects may slip through (1%, 5%, 10%); the threshold is set by conformal calibration, and what it achieved on held-out boards is published (4.0% missed at the 5% setting with the known-good board; 6.5% with the detector alone, see `docs/RESULTS.md`);
- **explains each finding with Llama 3.2 1B**: the causes and actions the operator reads are knowledge-base text, word for word; Llama selects among them and writes the summary, and a checker rejects any answer that states something not in the findings. (Measured: its choice of cause is no better than chance, so the knowledge base orders the options and Llama is not credited with diagnosis.);
- **logs everything** on the device and lights a green, amber or red stack light;
- **learns on the floor**:
  - a new bare board is taught in minutes from two or more pictures of a good board. The first becomes the reference picture; the station then draws test defects of known size onto the pictures to set that board's pass and fail thresholds and to state what size of defect it finds on it;
  - a new assembled board is taught from 5-10 good boards;
  - operator corrections train a small head that is switched on only if checks on held-out labels show it does not hide defects.

And it proves what it can do:

- **Eye test.** Photograph a printed card of 252 defects of known size. The station states, per defect type, the smallest size it finds 90% of the time with 95% confidence (MIL-HDBK-1823A a90/95), and which sizes it cannot certify.
- **Customer rules.** Plain-English acceptance clauses are compiled by **Llama 3.2 3B on ExecuTorch** into executable rules. Llama can only choose answers the clause supports (constrained decoding on the exported model's next-token scores), the result is verified by deterministic checks, and it is compared with the measured limit. A rule asking for smaller defects than the station can see is flagged, not silently accepted. On 40 held-out clauses: 62.5% exactly right, 1 silently wrong, the rest sent to the operator to confirm.

## How it covers PS5

| PS5 requirement | What the station does | Where |
|---|---|---|
| Detect surface defects, cracks, anomalies | YOLO26n (11 bare-board defect types), golden-board comparison with a connectivity check, and a patch-memory anomaly flag for assembled boards | `pcbinspect/detect_bare.py`, `golden_diff.py`, `fusion.py`, `anomaly.py` |
| Classify type and severity | Defect type from the detector; size in mm from the golden comparison against IPC-style class limits | `pcbinspect/golden.py`, `severity.py`, `config/severity.yaml` |
| Alert operators with actionable recommendations | PASS / FAIL / CHECK, GPIO stack light and buzzer, and Llama advice grounded in `kb/defects.yaml` | `pcbinspect/app.py`, `advisor.py`, `grounding.py` |
| Log inspection results | SQLite on the device (images, defects, dispositions, operator decisions, advice, eye tests, rules, model versions), plus CSV export | `pcbinspect/store.py` |
| Real time at the edge, no cloud | Every model runs on the Pi 5 CPU; timings, memory and a 30-minute soak are measured on the device | `bench/`, `deploy/measure_on_pi.sh` |
| Llama via ExecuTorch for reasoning | Llama 3.2 Instruct (Meta's SpinQuant INT4 checkpoints) exported with ExecuTorch `export_llm`, run through `TextLLMRunner`: 1B for advice, 3B for compiling customer rules | `pcbinspect/llm.py`, `spec_llm.py`, `llm/` |

## Two pipelines

```
VISION (ExecuTorch / Arm runtime, Arm CPU)            REASONING (ExecuTorch, Llama 3.2, Arm CPU)
camera (focus + exposure locked) -> ArUco scale, QR   structured defect record (no pixels)
  -> YOLO26n on tiles -> named defects                  + knowledge-base options for each defect
  -> golden-board comparison + connectivity check     -> 1B picks causes / actions, writes a summary
  -> fused evidence, size in mm                       -> checker: options valid, numbers and defect
  -> ResNet-18 patch memory -> unusual areas             types must come from the record
  -> IPC-style class limits -> disposition
  -> per-defect conformal gate -> PASS / FAIL / CHECK  customer clause in plain English
  -> customer rules -> SQLite log, stack light        -> 3B fills a fixed rule form, field by field
                                                       -> schema, numbers, defect and second-reader checks
eye-test card -> hit/miss per size -> a90/95 limit ---> capability check: can the station keep this rule?
```

The verdict never waits for Llama. Advice arrives on a background thread a few seconds later.

## What is and is not validated

| Measured | Not yet validated |
|---|---|
| Detector accuracy on held-out boards from five public datasets, split by physical board where the data allows | Accuracy on our own boards under a live camera (the eye-test card and one knife-cut board are the real-camera checks) |
| Per-defect conformal gate on held-out boards | The eye-test limit for etched copper rather than printed paper |
| Golden-board comparison and fusion on held-out DeepPCB boards | Copper segmentation on solder-masked boards under shop lighting |
| Eye-test reader on simulated photos of the card; in-silico detection limits under three camera conditions | Component presence/polarity and solder-joint judgement on assembled boards (anomaly flag only) |
| Anomaly memory on VisA pcb1-4 and Real-IAD `pcb` | Long-term drift on a real line |
| Llama advice grounding and choice vs chance; Llama rule reader (free, constrained, grounded) on 80 tuning and 40 held-out clauses | A taught board's gate and limits on real boards (built and tested on camera-like synthetic pictures; first real use is the Pi session) |
| Learning from operator feedback, simulated on held-out boards (DeepPCB: boards sent to a person 20% to 4%, no defective board passed) | |
| Raspberry Pi 5 timings, memory and a 30-minute soak (`docs/RESULTS.md` once run on the device) | |

The public datasets are scans or close-up photos, and some defects are synthetic: PKU-Market-PCB defects were drawn in with an image editor, and some of DeepPCB's were added by hand. Dataset results and real-camera results are therefore reported separately.

## Run it

**On the Raspberry Pi 5** (Raspberry Pi OS 64-bit, with the active cooler fitted):

```bash
tar xf pcb-station.tar && cd pcb-station
bash deploy/pi_setup.sh          # once: apt packages + Python env with the aarch64 ExecuTorch wheel
python deploy/arm_check.py       # loads every model and runs it once (fails loudly if Llama does not run)
bash deploy/run_on_pi.sh         # camera auto-detect (Pi Camera, else USB webcam)
bash deploy/run_on_pi.sh replay  # or cycle through held-out demo boards
bash deploy/hotspot.sh           # optional: a phone joins "PCB-Station" and opens http://10.42.0.1:8000
```

Press **Inspect**, the space bar or a USB foot pedal. Print `deploy/assets/scale_marker_sheet.pdf` at 100%, check its ruler, and place a marker beside the board.

**Measurements:** `bash deploy/measure_on_pi.sh` runs the functional check, every benchmark and a 30-minute soak, and packs the results.

**Eye test:** print `assets/eyetest/eyetest_card.pdf` on an inkjet at 100%, then follow the Eye test tab on the dashboard.

**Rebuilding the models** (on a PC):

- `training/prepare_bare.py` builds the dataset, and `train_bare.py` trains on a GPU.
- `eval_bare.py` evaluates and calibrates the detector.
- `build_visa_banks.py` builds the anomaly memories.
- `export_vision.py` and `llm/export_llama.sh` produce the ExecuTorch exports (executorch 1.6.0, the version the Pi runs). They run in `deploy/Dockerfile.export` or on Colab.
- `deploy/Dockerfile.pi` is an arm64 Debian image that installs exactly what the Pi installs. It is used to check the Arm build without hardware.

**Results:** `docs/RESULTS.md` is generated from the result JSON files by `docs/make_results.py`. Research and design decisions are in `docs/RESEARCH_AND_DECISIONS.md`. The full write-up is in `docs/TECHNICAL_REPORT.md`.

## Data and licences

Datasets, models and software licences, including Llama 3.2 and Ultralytics, are listed in `NOTICE.md`. No dataset is redistributed here. Real-IAD (non-commercial) is used for evaluation only and never for anything shipped. The IPC-style limits in `config/severity.yaml` come from a cited secondary source and are editable; they do not replace the licensed IPC standards or a customer's own specification.
