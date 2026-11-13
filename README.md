# PCB Inspection Station: offline defect detection with on-device Llama advice

Arm Bharat AI-SoC Challenge 2026-27, Problem Statement 5 (real-time industrial defect detection and quality-control assistant).

A Raspberry Pi 5 with a camera inspects printed circuit boards, bare or assembled, and does all of this offline:

- It finds each defect, names its type, measures it in millimetres and grades it against IPC-style acceptance limits.
- It gives every board a **PASS / FAIL / CHECK** verdict. The CHECK verdict (needs review) comes with a statistical guarantee on how many defective boards can slip through.
- It lights a green, amber or red stack light and logs every inspection on the device.
- It asks **Llama 3.2 running on ExecuTorch** what the operator should do. Every cause and action it gives is checked against a defect knowledge base before the operator sees it.
- It learns on the device:
  - A new board design is taught from 5 to 10 good boards.
  - Each false alarm the operator marks makes the next inspection better.
  - Every change can be undone and must pass a regression check.

No cloud is involved at any point. The operator's phone connects to the Pi's own Wi-Fi hotspot.

## How it meets the PS5 requirements

| PS5 requirement | What the station does | Where |
|---|---|---|
| Detect surface defects, cracks, anomalies | YOLO26n detector for 8 named bare-board defect types, plus a patch-memory anomaly model that flags anything unusual on assembled boards | `pcbinspect/detect_bare.py`, `pcbinspect/anomaly.py` |
| Classify type and severity | Defect type from the detector. Severity from size in mm against IPC-style width and spacing limits, with Class 1/2/3 accept or reject | `pcbinspect/severity.py`, `config/severity.yaml` |
| Alert operators with actionable recommendations | Verdict band, GPIO stack light and buzzer. Llama 3.2 advice grounded in `kb/defects.yaml` | `pcbinspect/app.py`, `pcbinspect/advisor.py`, `pcbinspect/gpio.py` |
| Log inspection results | SQLite (WAL) on the device, with images, operator decisions, advice and model versions. CSV export | `pcbinspect/store.py` |
| Real time at the edge, no cloud | Everything runs on the Pi 5 Arm CPU through ExecuTorch (XNNPACK) | `bench/bench_device.py` |
| Llama via ExecuTorch for reasoning | Llama 3.2 1B Instruct (SpinQuant INT4) `.pte`, run through ExecuTorch's `TextLLMRunner` | `pcbinspect/llm.py` |

## Architecture: two pipelines, documented separately

```
VISION PIPELINE (ExecuTorch, XNNPACK on the Arm CPU)             REASONING PIPELINE (ExecuTorch, Llama 3.2)
camera -> ArUco mm scale + QR board ID                            structured defect record (no pixels)
       -> YOLO26n tiles (.pte)  -> named defects                    + matching knowledge-base options
       -> ResNet18 patches (.pte) -> anomaly memory -> unknowns     + recent history on this line
       -> severity (mm vs IPC limits)                            -> Llama 3.2 1B (.pte) picks cause/action
       -> conformal gate -> PASS / FAIL / CHECK                  -> grounding checker (rejects anything not in
       -> SQLite log, stack light                                   the record or the knowledge base)
                                                                 -> operator sees knowledge-base text verbatim
```

The verdict never waits for Llama. Advice arrives a few seconds later on its own thread.

## Run it

**On the Raspberry Pi 5** (Raspberry Pi OS 64-bit, with the official active cooler fitted):

```bash
tar xf pcb-station.tar && cd pcb-station
bash deploy/pi_setup.sh          # once: apt packages + Python env with the aarch64 ExecuTorch wheel
bash deploy/run_on_pi.sh         # camera auto-detect: Pi Camera, else USB webcam
bash deploy/run_on_pi.sh replay  # or cycle through held-out demo boards
bash deploy/hotspot.sh           # optional: phone joins "PCB-Station", opens http://10.42.0.1:8000
```

Press **Inspect**, the space bar or a USB foot pedal.

**Measurements on the Pi:**

```bash
python -m bench.bench_device --out results/device_bench.json   # vision, Llama, end-to-end
python -m bench.soak --minutes 30                              # thermal soak
```

**Rebuilding the models** (on a PC):

- `training/prepare_bare.py` builds the dataset and `training/train_bare.py` trains the detector (on a GPU).
- `training/build_visa_banks.py` builds the anomaly memories.
- `training/eval_bare.py` evaluates and calibrates.
- `training/export_vision.py` and `llm/` produce the ExecuTorch exports. They run in `deploy/Dockerfile.export`.
- `deploy/Dockerfile.pi` is an arm64 Debian-trixie image that installs exactly what the Pi installs. It is used to check the Arm build before hardware is available.

## Results

All numbers are measured, never projected. The tables in `docs/RESULTS.md` are generated from `results/*.json` by `docs/make_results.py`.

## Data and licences

| Dataset | Use | Licence |
|---|---|---|
| [DeepPCB](https://github.com/tangsanli5201/DeepPCB) | Bare-board defects plus defect-free templates | Research use (per README) |
| [PKU-Market-PCB / HRIPCB](https://github.com/Ironbrotherstyle/PCB-DATASET) | Bare-board defects, split by physical board | Research use |
| [PCB-Defect (Mendeley, 2025)](https://data.mendeley.com/datasets/vdj74sngvn/1) | Real etched boards, flatbed scans | See the dataset page |
| [VisA pcb1-4](https://github.com/amazon-science/spot-diff) | Assembled boards, anomaly detection | CC BY 4.0 |

Llama 3.2 is used under the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/). IPC-style limits in `config/severity.yaml` come from secondary sources, which are cited in that file. They are editable and are not a substitute for the licensed IPC standards.
