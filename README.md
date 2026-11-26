# PCB Inspection Station

**An offline circuit-board inspector that runs entirely on a Raspberry Pi's Arm CPU.** It finds defects, measures them in millimetres, decides PASS, FAIL or CHECK at a miss rate you choose, and explains the likely cause with a fine-tuned Llama 3.2 on ExecuTorch. It also tells you, for your own camera and board, the smallest defect it can promise to catch.

Arm Bharat AI-SoC Challenge 2026-27, Problem Statement 5: real-time industrial defect detection and quality-control assistant.

## Highlights

- **83.3% mAP** on 1,006 boards it never trained on (7,093 defects, five public datasets); **96.3% on DeepPCB**.
- **4.0% of defects missed** at the 5% miss-rate setting on unseen boards, with **0.29 false calls per board**.
- **434 ms from picture to verdict** for a DeepPCB board **on a Raspberry Pi 4**. The INT8 detector takes **254 ms per tile** and keeps FP32's accuracy (83.2% vs 83.3% mAP).
- **Cool and stable:** a soak on the Pi 4 peaked at 63.3 °C with **no throttling**.
- **Our fine-tuned Llama 3.2 1B** (LoRA, ExecuTorch) picks the right cause first **70% of the time, against 40%** for the prompted Llama 3.2 3B it replaced.
- **It learns from the operator:** boards sent to a person dropped **from 20% to 4%**, with **no defective board passed** (simulated on held-out DeepPCB boards).
- **Assembled boards:**
  - the part finder scores **97.8% mAP** on boards it never saw;
  - the "anything unusual" model reaches **93–99.6% image AUROC** on VisA.
- **Teach a new board in minutes, on the Pi.** A real rain-sensor pad it had never seen was taught from **two photos in 10 minutes**. It then failed a damaged pad in **12.9 s**, boxing and measuring the solder bridges, stray solder and knife cuts.
- **Finds parts with no teaching:** on an Arduino Uno it had never seen, it found **47 parts in about a second**.
- **Nothing leaves the device:** no cloud, no GPU, no network needed.

## What it does for each board

1. **Takes the picture.** It uses a Pi camera or USB webcam with focus and exposure locked. A printed marker beside the board gives the scale in mm.
2. **Compares it with a known-good board.** Any change in the copper shows up. A connectivity check names it: copper joining two tracks is a short, a track split in two is an open. Each defect is **measured in millimetres**.
3. **Names the defects** with a YOLO26 detector (11 bare-board defect types) running through ExecuTorch.
4. **Checks assembled boards** with an "anything unusual" model and a part-by-part check (missing, extra or changed part).
5. **Grades severity** against IPC-style limits for the customer's class (1, 2 or 3).
6. **Decides PASS, FAIL or CHECK.** The operator picks how many defects may slip through (1%, 5% or 10%). Conformal calibration sets the thresholds to keep that promise. When the station isn't sure, it says CHECK and a person decides. A green, amber or red stack light shows the verdict.
7. **Explains each finding with Llama**, without ever holding up the verdict:
   - the likely process cause, with a percentage;
   - the clue behind it (same spot on earlier boards, a conveyor lane, the board edge, a trend, a laminator-roll repeat);
   - the check that confirms it.

   Causes and actions come word for word from a checked knowledge base, and every number in a sentence is filled in by the station.
8. **Logs everything** on the device in SQLite, exportable as CSV.

Two more abilities:

- **Eye test:** photograph a printed card of 252 defects of known size. The station then states, per defect type, the smallest size it finds 90% of the time with 95% confidence (the MIL-HDBK-1823A a90/95 method).
- **Customer rules:** a plain-English acceptance clause is turned into an executable rule by **Llama 3.2 3B on ExecuTorch**, then checked against what the station can actually see. On 40 held-out clauses, 62.5% came out exactly right. Most of the rest were sent to the operator to confirm.

## How it covers Problem Statement 5

| PS5 asks for | What the station does | Where |
|---|---|---|
| Detect surface defects, cracks, anomalies | YOLO26n for 11 defect types (s and m also shipped); known-good comparison with a connectivity check; patch-memory anomaly model and part-by-part check for assembled boards | `pcbinspect/detect_bare.py`, `golden_diff.py`, `fusion.py`, `anomaly.py`, `parts.py` |
| Classify type and severity | Type from the detector; size in mm from the known-good comparison; IPC-style class limits | `pcbinspect/golden.py`, `severity.py`, `config/severity.yaml` |
| Alert operators with actionable advice | PASS / FAIL / CHECK, GPIO stack light and buzzer, Llama advice grounded in `kb/defects.yaml` with the clues behind it | `pcbinspect/app.py`, `advisor.py`, `evidence.py`, `why.py`, `grounding.py` |
| Log inspection results | SQLite on the device: images, defects, decisions, advice, eye tests, rules, model versions; CSV export | `pcbinspect/store.py` |
| Real time at the edge, no cloud | Every model on the Pi's Arm CPU: 434 ms per DeepPCB board on a Raspberry Pi 4 | `bench/`, `deploy/measure_on_pi.sh` |
| Llama via ExecuTorch for reasoning | Llama 3.2 1B fine-tuned for this station (LoRA, 8da4w) for advice and shift questions; Llama 3.2 3B SpinQuant for customer rules; decisions read from next-token scores | `pcbinspect/llm_logits.py`, `llm_select.py`, `advisor.py`, `spec_llm.py`, `llm/` |

## Two pipelines

```
VISION (ExecuTorch, Arm CPU)                          REASONING (ExecuTorch, Llama 3.2, Arm CPU)
camera (focus + exposure locked) -> scale marker, QR  structured defect record (no pixels)
  -> known-good comparison + connectivity check         + knowledge-base options for each defect
  -> YOLO26n on tiles -> named defects                  + clues from the log (same spot, lane, edge, trend)
  -> fused evidence, size in mm                       -> fine-tuned 1B: probability per cause, why line
  -> ResNet-18 patch memory -> unusual areas          -> checker: knowledge-base text, station numbers
  -> IPC-style class limits -> disposition
  -> per-defect conformal gate -> PASS / FAIL / CHECK  customer clause in plain English
  -> customer rules -> SQLite log, stack light        -> 3B fills a fixed rule form, field by field
eye-test card -> hit/miss per size -> a90/95 limit ---> can the station keep this rule?
```

The verdict never waits for Llama: the light changes first, and the advice follows on a background thread.

## On the Raspberry Pi 4

Everything was measured on a Raspberry Pi 4 Model B (8 GB, four Arm Cortex-A72 cores) with a USB webcam:

| What | Measured |
|---|---|
| Defect detector, one tile, INT8, 4 threads | 254 ms |
| "Anything unusual" backbone, INT8, 4 threads | 186 ms |
| DeepPCB board, picture to verdict | 434 ms |
| VisA assembled board, picture to verdict | 0.88–1.11 s |
| Llama 3.2 1B generating text | 3.6 tokens/s |
| Station with Llama loaded | 1.7 GB of 8 GB |
| Soak of inspections with advice | peak 63.3 °C, no throttling |

**Getting Llama onto the Pi 4.** ExecuTorch's prebuilt Llama kernels use ARMv8.2 instructions that the Pi 4's Cortex-A72 doesn't have. We rebuilt that library for ARMv8.0 (`deploy/rebuild_llm_ops_pi4.sh`), and Llama now runs on the Pi 4.

**Real boards under the webcam:**

| Board | Result |
|---|---|
| Rain-sensor pad, new to the station | Taught on the Pi from two photos in 10 minutes. Promises shorts from 0.40 mm, mouse bites from 0.44 mm and spurious copper from 0.29 mm (each found at least 90% of the time with 95% confidence) |
| Damaged rain-sensor pad | **FAIL** in 12.9 s, 64 findings sized in mm, Llama advice written on the Pi |
| Good rain-sensor pad (a teaching photo, replayed) | **PASS** in 2.8 s at the 10% setting |
| Good pad placed differently | **REVIEW**: it doesn't guess when a board doesn't line up, a person decides |
| Arduino Uno, no teaching | 47 parts found and named in about a second |

All figures, with the files they come from: `docs/RESULTS.md`.

## Try it

- **Online:** a recording of the station's dashboard on held-out test boards, showing each verdict, the findings marked on the board, the dataset's known answer and Llama's advice: https://pcb-inspection-demo.vercel.app. It was recorded with `deploy/record_demo.py` on an x86 CPU.
- **On your computer:** one Docker command runs the real station in demo mode. Pick a test board or upload a photo, and the fine-tuned Llama writes the advice on your CPU. See `deploy/DEMO.md`.

## Run it on a Pi

Raspberry Pi 4 or 5, Raspberry Pi OS 64-bit, with a cooler fitted:

```bash
tar xf pcb-station.tar && cd pcb-station
bash deploy/pi_setup.sh          # once: apt packages + Python env with the aarch64 ExecuTorch wheel
python deploy/arm_check.py       # loads every model and runs it once
bash deploy/run_on_pi.sh         # camera auto-detect (Pi Camera, else USB webcam)
bash deploy/run_on_pi.sh replay  # or cycle through held-out demo boards
bash deploy/hotspot.sh           # optional: a phone joins "PCB-Station" and opens http://10.42.0.1:8000
```

On a Pi 4, run `bash deploy/rebuild_llm_ops_pi4.sh` after setup: it rebuilds ExecuTorch's Llama kernel library for ARMv8.0 (the build took 6 minutes on our Pi 4).

Press **Inspect**, the space bar or a USB foot pedal. Print `deploy/assets/scale_marker_sheet.pdf` at 100% and place a marker beside the board.

- **Teach a new board:** open the Teach tab and capture a good board two or more times. The station draws practice defects of known size onto those photos, sets the board's thresholds, and reports the sizes it can promise.
- **Measurements:** `bash deploy/measure_on_pi.sh` runs the functional check, every benchmark and a soak, and packs the results.
- **Eye test:** print `assets/eyetest/eyetest_card.pdf` at 100%, then follow the Eye test tab.

## Next steps (phase 2)

- **More real boards on the line:** a wider set of in-house boards, solder-masked boards, and a lighting hood for even light, so more designs pass at the 5% setting as well as 10%.
- **Raspberry Pi 5:** run the same benchmarks there, with a 30-minute soak and board-power readings from the Pi 5's power chip.
- **Deeper assembled-board checks:** part polarity, part values and solder joints, on top of today's missing, extra and changed parts.
- **Shop-floor tuning of the cause knowledge base:** a process engineer reviews it, and real-line feedback refines Llama's advice further.
- **Long-run monitoring:** drift tracking over weeks on a real line, building on the control charts already in the dashboard.

## Rebuilding the models (on a PC)

- `training/prepare_bare.py` builds the dataset, `train_bare.py` trains on a GPU, and `eval_bare.py` evaluates and calibrates.
- `build_visa_banks.py` builds the anomaly memories; `prepare_components.py` and `eval_parts*.py` build and score the parts check.
- `export_vision.py` and `llm/export_llama.sh` produce the ExecuTorch files (executorch 1.6.0). They run in `deploy/Dockerfile.export` or on Colab.
- The fine-tuned advice model is rebuilt end to end on Colab with `training/colab/job_llama_v31.sh`:
  - it generates training data from `kb/evidence.yaml`;
  - trains a LoRA with `llm/train_lora.py`;
  - merges it with `llm/merge_lora_meta.py`;
  - exports 8da4w and scores the `.pte`.

  Under the Llama licence, a distributed fine-tune's name must begin with "Llama" (see `NOTICE.md`).
- `deploy/Dockerfile.pi` is an arm64 Debian image matching the Pi, for checking the Arm build without hardware.
- `models/MODELS.md` lists every model file with its size, SHA-256, origin and rebuild command. `training/colab/` holds the exact jobs that produced the shipped models.

**Results:** `docs/RESULTS.md` is generated from `results/measured/` by `docs/make_results.py`. The full write-up is `docs/TECHNICAL_REPORT.md`, and the plain-words overview is `docs/PCB_Inspection_Station_Overview.pdf`.

## Where things are

| Folder | What is in it |
|---|---|
| `pcbinspect/` | The station: camera, vision, known-good comparison, parts check, severity, gate, Llama advice, log, dashboard |
| `config/`, `kb/` | Station settings, IPC-style limits, the defect knowledge base Llama chooses from |
| `models/` | Calibrated thresholds and anomaly memories; `MODELS.md` lists the model files and how to download them |
| `training/`, `llm/` | Dataset building, training, evaluation and export; `training/colab/` has the exact jobs |
| `results/measured/` | Every measured result, including `pi4/` from the Raspberry Pi 4 |
| `bench/`, `deploy/` | Pi benchmarks and soak, Pi setup, bundle builder, Arm checks, rig guide (`deploy/RIG.md`) |
| `docs/` | Technical report, generated results, research and design decisions, eye test and rules, demo video script, overview PDF |
| `assets/eyetest/` | The printable eye-test card |
| `tests/` | `pytest` suite |

## Data and licences

The project's own code is licensed under AGPL-3.0 (`LICENSE`). Datasets, models and software licences, including Llama 3.2 and Ultralytics, are listed in `NOTICE.md`. No dataset is redistributed here; Real-IAD is used for evaluation only. The IPC-style limits in `config/severity.yaml` come from a cited secondary source and can be edited for each customer.
