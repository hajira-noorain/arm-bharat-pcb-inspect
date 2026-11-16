# PCB Inspection Station: an offline inspector that knows its own limits

Arm Bharat AI-SoC Challenge 2026-27, Problem Statement 5 (real-time industrial defect detection and quality-control assistant).

**One station, one question: is this bare circuit board good?** Put a board under the camera of a Raspberry Pi 5. Its picture is compared with a known-good board; every difference is named, measured in millimetres and graded; the board gets PASS, FAIL or CHECK and a light; Llama 3.2 on ExecuTorch explains the likely cause and the next action; and the inspection is logged. Nothing leaves the Pi. On boards it had never seen, at a 5% miss-rate setting, it missed 4.0% of defects.

Every inspection system claims a high accuracy figure. This one also **measures the smallest defect it can find on your camera, says what it cannot see, and refuses customer rules it cannot guarantee.** Vision runs through ExecuTorch (or Arm's optimised YOLO26 runtime), Llama 3.2 through ExecuTorch, all on the Pi's Arm CPU.

In more detail, for each board it:

- **finds and names defects** on bare boards (11 types); on assembled boards it flags anything unusual and checks the parts one by one (missing, extra or changed part);
- **compares against a known-good board first**: this is the station's main evidence. It sees any change in the copper, checks connectivity (a short joins two conductors, an open splits one), **measures each defect in millimetres**, and tells the detector which parts of the picture to look at. If a board does not line up with its reference picture, it is never passed: a person looks;
- **grades severity** against IPC-style limits for the customer's class;
- **returns PASS, FAIL or CHECK** (a person decides). The operator picks a target for how many defects may slip through (1%, 5%, 10%); the threshold is set by conformal calibration, and what it achieved on held-out boards is published (4.0% missed at the 5% setting with the known-good board; 6.5% with the detector alone; each beside the level that calibration can promise with 95% confidence, see `docs/RESULTS.md`);
- **explains each finding with our fine-tuned Llama 3.2 1B** (Llama-3.2-1B-PCB-Advisor, LoRA, 8da4w on ExecuTorch):
  - It judges the likely process cause from clues the station measures in its own log: same spot on earlier boards, a lane along the conveyor, edge, middle of the board, shape, count, trend, and a laminator-roll repeat.
  - It gives a percentage, and shows the two most likely causes with the check that separates them when it is a close call.
  - It adds one line naming the clue behind the cause.
  - Causes and actions are knowledge-base text, word for word, and every number in a sentence is filled in by the station.
  - The operator can record what the cause turned out to be, and the station learns that line's causes from those answers.

  On held-out clue combinations it picks the right cause first 70% of the time, against 40% for the prompted Llama 3.2 3B it replaced. It matches the rule table it learned from (67.5%); the clues themselves allow no more. See `docs/TECHNICAL_REPORT.md` section 6.
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
| Detect surface defects, cracks, anomalies | YOLO26n (11 bare-board defect types; YOLO26s and YOLO26m also shipped), golden-board comparison with a connectivity check, and for assembled boards a patch-memory anomaly flag plus a part-by-part check | `pcbinspect/detect_bare.py`, `golden_diff.py`, `fusion.py`, `anomaly.py`, `parts.py` |
| Classify type and severity | Defect type from the detector; size in mm from the golden comparison against IPC-style class limits | `pcbinspect/golden.py`, `severity.py`, `config/severity.yaml` |
| Alert operators with actionable recommendations | PASS / FAIL / CHECK, GPIO stack light and buzzer, and Llama advice grounded in `kb/defects.yaml` with the clues behind it (`kb/evidence.yaml`) | `pcbinspect/app.py`, `advisor.py`, `evidence.py`, `why.py`, `grounding.py` |
| Log inspection results | SQLite on the device (images, defects, dispositions, operator decisions, advice, eye tests, rules, model versions), plus CSV export | `pcbinspect/store.py` |
| Real time at the edge, no cloud | Every model runs on the Pi 5 CPU; `deploy/measure_on_pi.sh` measures timings, memory, power and a 30-minute soak on the device (numbers pending the Pi session, 20-21 Nov) | `bench/`, `deploy/measure_on_pi.sh` |
| Llama via ExecuTorch for reasoning | Llama 3.2 Instruct exported with ExecuTorch `export_llm` and run through `executorch.runtime`, so decisions are read from next-token scores: a 1B fine-tuned with LoRA on the station's own rule table for advice and shift questions (merged, 8da4w), Meta's SpinQuant 3B for compiling customer rules, and the SpinQuant 1B as the fallback | `pcbinspect/llm_logits.py`, `llm_select.py`, `advisor.py`, `spec_llm.py`, `llm/` |

## Two pipelines

```
VISION (ExecuTorch / Arm runtime, Arm CPU)            REASONING (ExecuTorch, Llama 3.2, Arm CPU)
camera (focus + exposure locked) -> ArUco scale, QR   structured defect record (no pixels)
  -> YOLO26n on tiles -> named defects                  + knowledge-base options for each defect
  -> golden-board comparison + connectivity check     + clues from the log (same spot, lane, edge, trend)
  -> fused evidence, size in mm                       -> tuned 1B: probability per cause, why line
  -> ResNet-18 patch memory -> unusual areas          -> checker: knowledge-base options, station numbers
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
| Eye-test reader on simulated photos of the card; in-silico detection limits under three camera conditions | Polarity, part value and solder joints on assembled boards (not checked); the parts check on real boards under our camera |
| Anomaly memory on VisA pcb1-4 and Real-IAD `pcb` | Long-term drift on a real line |
| Parts check: part detector on 15 held-out FPIC boards; missing/extra/moved parts on VisA photos (development result) | |
| Every detector file loads and gives the same findings on aarch64 (arm64 emulator, `deploy/sim_check_detectors.py`) | |
| Fine-tuned Llama cause choice on held-out clue combinations, a second independently built generator and 8 published real cases, beside the rule table; shift answers and why lines through the station's checker; Llama rule reader on 80 tuning and 40 held-out clauses | A taught board's gate and limits on real boards (built and tested on camera-like synthetic pictures; first real use is the Pi session) |
| | The cause rule table: its directions come from published process sources for 22 of 45 rules, but its numbers are our estimates and have not been reviewed by anyone with shop-floor experience; 8 real cases are too few to rank models |
| Learning from operator feedback, simulated on held-out boards (DeepPCB: boards sent to a person 20% to 4%, no defective board passed) | |
| Raspberry Pi 5 timings, memory and a 30-minute soak (`docs/RESULTS.md` once run on the device) | |

The public datasets are scans or close-up photos, and some defects are synthetic: PKU-Market-PCB defects were drawn in with an image editor, and some of DeepPCB's were added by hand. Dataset results and real-camera results are therefore reported separately.

## Try it without a Pi

- **Online, recorded:** the station's own dashboard showing what it did on each held-out test board (verdict, findings marked on the board, the dataset's known answer, Llama's own advice): https://pcb-inspection-demo.vercel.app. It is a recording made with `deploy/record_demo.py` and packaged by `deploy/build_static_demo.py`, run on an x86 CPU, not the Pi; Llama's answers to example shift questions and customer requirements are recorded too.
- **Live, on your computer:** one Docker command runs the real station in demo mode. Pick any test board or upload a photo of a bare board, and the fine-tuned Llama writes the advice on your CPU. See `deploy/DEMO.md`.

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
- The fine-tuned advice model is rebuilt end to end on Colab with `training/colab/job_llama_v31.sh`. It generates the training data from `kb/evidence.yaml` (`llm/data/`), trains a LoRA with `llm/train_lora.py`, merges it into Meta's checkpoint (`llm/merge_lora_meta.py`), exports 8da4w and scores the `.pte`. Then fit its temperature with `llm/fit_temperature.py`, and set `advice_llm_pte` and `advice_temperature` in `config/app.yaml`. `llm/data/audit.py` checks the training data first. Under the Llama licence, a distributed fine-tune's name must begin with "Llama" (see `NOTICE.md`).
- `deploy/Dockerfile.pi` is an arm64 Debian image that installs exactly what the Pi installs. It is used to check the Arm build without hardware.

- `prepare_components.py` and `eval_parts*.py` build and score the parts check (FPIC, VisA).
- `training/colab/` holds the exact Colab jobs that produced the shipped models and measured results.
- `models/MODELS.md` lists every model file with its size, SHA-256, where it came from and the command that rebuilds it.

**Results:** `docs/RESULTS.md` is generated from the result JSON files in `results/measured/` by `docs/make_results.py`. Research and design decisions are in `docs/RESEARCH_AND_DECISIONS.md`. The full write-up is in `docs/TECHNICAL_REPORT.md`.

## Where things are

| Folder | What is in it |
|---|---|
| `pcbinspect/` | The station: camera, vision, golden comparison, parts check, severity, gate, Llama advice and checker, log, dashboard |
| `config/`, `kb/` | Station settings, IPC-style limits, the defect knowledge base Llama chooses from |
| `models/` | Calibrated thresholds and anomaly memories (in git); `MODELS.md` lists the model files that are not, and how to download them from the private Hugging Face repo |
| `training/`, `llm/` | Dataset building, training, evaluation and export scripts; `training/colab/` has the exact jobs that were run |
| `results/measured/` | Every measured result file; `results/README.md` explains the names |
| `bench/`, `deploy/` | Pi benchmarks and soak; Pi setup, bundle builder, Arm checks, the rig guide (`deploy/RIG.md`) |
| `docs/` | `TECHNICAL_REPORT.md` (the write-up), `RESULTS.md` (generated tables), `RESEARCH_AND_DECISIONS.md`, `EYE_TEST_AND_RULES.md`, `DEMO_VIDEO_SCRIPT.md` |
| `assets/eyetest/` | The printable eye-test card and its manifest |
| `tests/` | `pytest` suite |

## Data and licences

Datasets, models and software licences, including Llama 3.2 and Ultralytics, are listed in `NOTICE.md`. No dataset is redistributed here. Real-IAD (non-commercial) is used for evaluation only and never for anything shipped. The IPC-style limits in `config/severity.yaml` come from a cited secondary source and are editable; they do not replace the licensed IPC standards or a customer's own specification.
