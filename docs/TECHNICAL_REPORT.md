# PCB Inspection Station: Technical Report

Arm Bharat AI-SoC Challenge 2026-27, Problem Statement 5: real-time industrial defect detection and quality-control assistant.

## 1. Problem and scope

Printed circuit boards fail in a small set of ways:

- conductors are cut or nicked (open, mouse bite, pinhole)
- copper is left where it should not be (short, spur, spurious copper)
- holes or pads are missing or damaged
- on assembled boards, components and solder joints go wrong

A factory inspection station has to find these defects quickly, decide what each one means, tell the operator what to do, and keep a record. All of that has to work without a network connection.

The station covers **bare boards and assembled boards** with two complementary vision paths. It runs entirely on a **Raspberry Pi 5 (8 GB) CPU**. The operator dashboard is served by the Pi itself, to its own screen or to a phone on the Pi's Wi-Fi hotspot.

**What PS5 asks for, and where it is in this report:**

| PS5 requirement | Section |
|---|---|
| Detect defects, cracks, anomalies | 3.1, 3.2 |
| Classify type and severity | 3.1, 3.3 |
| Alert operators with actionable recommendations | 3.5, 4 |
| Log inspection results | 3.6 |
| Real time at the edge, no cloud | 5 |
| Llama via ExecuTorch for reasoning | 4 |

## 2. System overview: two pipelines

The PS asks for the vision and Llama pipelines to be documented separately, so they are.

**Vision pipeline (ExecuTorch, XNNPACK delegate, Arm CPU)**

1. **Capture.** Pi Camera through Picamera2, a USB webcam, or replay of held-out dataset boards.
2. **Board identity and scale.** A QR code gives the board ID. A printed 20 mm ArUco marker beside the board gives millimetres per pixel.
3. **Named defects.** A YOLO26n detector runs on 640-pixel tiles of the full-resolution frame and finds 8 bare-board defect types.
4. **Unknown defects.** A ResNet18 patch-feature model (PatchCore-style memory of good boards) flags anything that differs from known-good boards. This path covers assembled boards.
5. **Measurement.** Golden-board comparison measures the copper each defect actually removes or adds, in millimetres.
6. **Severity.** Measured reduction against IPC-style width and spacing limits for Class 1, 2 and 3.
7. **Decision.** A conformal gate gives PASS, FAIL or CHECK (needs a person) at an operator-chosen allowed miss rate.
8. **Outputs.** Stack light and buzzer on GPIO, and the SQLite log.

**Reasoning pipeline (ExecuTorch, Llama 3.2 Instruct, Arm CPU)**

1. **Input.** A structured defect record: type, severity, size, location, confidence and recent history on the line. No pixels.
2. **Context.** Knowledge-base options for each defect: causes and actions, each with an ID.
3. **Generation.** Step-wise structured generation through `TextLLMRunner`, with the KV cache kept across steps.
4. **Checking.** A grounding checker validates the answer. The operator sees knowledge-base text verbatim, plus Llama's choice and summary.
5. **Other uses.** Shift questions over the log, with whitelisted read-only queries, and wording for drift alerts.

The verdict never waits for Llama. Advice is produced on a background thread and appears seconds later.

## 3. Vision

### 3.1 Bare-board detector

**Data.** Three public sources were unified into 8 classes: missing hole, mouse bite, open, short, spur, spurious copper, pinhole and damaged pad.

| Source | What it is | How it is split |
|---|---|---|
| DeepPCB | 1,500 binarised image pairs, 48 px/mm | Official split |
| PKU-Market-PCB (HRIPCB) | 693 colour photos of 10 boards | **By physical board**, so test boards are designs the model never saw |
| Mendeley PCB-Defect (2025) | 230 real etched boards, flatbed scans | By board group |

The three sources look very different: binarised, green solder mask, and bare copper. That variety is deliberate, so the model does not overfit one look.

**Tiling.** Large frames are cut into 640-pixel tiles with 128 pixels of overlap, with the same code in training and inference. Detections are merged with NMS.

**Model.** YOLO26n. Ultralytics' own Pi 5 benchmark covers it under ExecuTorch, and it is small enough for CPU inference.

**Training.** On a laptop RTX 5060 before deployment. This is the "trained beforehand" half of the brief.

**Export.** To ExecuTorch with the XNNPACK delegate, in FP32 and INT8, through Ultralytics' ExecuTorch exporter.

### 3.2 Assembled boards: anomaly memory

The PS lists anomalies alongside named defects. On assembled boards the defect variety is open-ended, and labelled data is scarce. So the station learns what *good* looks like:

1. **Features.** A frozen ResNet18 (layer2 and layer3 features, 3x3 local averaging) gives a 384-dimensional descriptor for every 8x8-pixel patch of a 256x256 view.
2. **Memory.** Descriptors from good boards are kept in a memory, reduced by greedy coreset selection.
3. **Scoring.** A new board is scored by each patch's distance to its nearest good patch.
4. **Calibration.** Leave-one-out, or a held-out 20%, sets the per-design normalisation and region threshold. These numbers are therefore honest even with five good boards.
5. **Export.** The backbone is the only neural network. Its ExecuTorch FP32 export matches PyTorch to a relative error of about 1e-6, so a memory built on the laptop is valid on the Pi.

### 3.3 Measuring defects in millimetres

A detection box is drawn loosely around a defect, so its size is not the defect's size. An early version that measured boxes produced physically impossible values, such as "291% spacing reduction". It was replaced by **golden-board comparison**, which is how industrial AOI measures:

1. Align the board to a known-good image of the same design (ECC, affine).
2. Segment copper in both, choosing the most contrasting colour channel with Otsu thresholding.
3. **Missing copper** inside the detection: the nick depth is the short side of its minimum-area rectangle. The local conductor width comes from the golden image's distance transform. The result is a % width reduction.
4. **Extra copper**: the remaining gap is the shortest distance to a *different* golden conductor. The result is a % spacing reduction against the minimum spacing.
5. With no golden image, the station makes no millimetre claim and uses the defect-type default severity.

**Scale** comes from the printed ArUco marker. Rendering the marker sheet at a known 300 dpi and measuring it with the station's own code gives 0.12% error. For the Mendeley scans it comes from the dataset's 20 mil minimum-trace design rule, measured from the copper (implied 1,249 dpi).

### 3.4 IPC-style severity

The limits in `config/severity.yaml` are the commonly quoted IPC-A-600 width and spacing reduction limits, from cited secondary sources: 30% for Class 1, 20% for Class 2, 10% for Class 3. We do not have the licensed standard, so the file is editable, and the limits are stated as configurable rather than authoritative.

Severity is defined by the class outcome:

- **Critical:** rejected even at Class 1.
- **Major:** rejected at Class 2.
- **Minor:** rejected only at Class 3.

Shorts, opens and missing holes are functional failures, rejected at every class.

### 3.5 PASS / FAIL / CHECK with a guaranteed miss rate

Every detector has a threshold. The usual practice is to pick one by eye, which says nothing about how many defective boards will slip through. The station uses **split-conformal risk control** on held-out calibration boards instead. The threshold λ is the k-th smallest board score among n defective calibration boards, with k = ⌊α(n+1)⌋. A new defective board from the same process is then passed with probability at most α, whatever the score distribution. A second threshold, τ, bounds the share of good boards failed at β.

- Boards between λ and τ go to a person: the CHECK verdict, amber light.
- The operator picks α (1%, 5% or 10%) on the dashboard and sees the trade-off as the review queue grows.

The guarantee holds only while boards look like the calibration boards: same product, camera and lighting. The station therefore keeps one calibration per design and re-calibrates whenever its model changes (section 6).

A Monte-Carlo unit test (2,000 repetitions, skewed distributions) checks that the average miss rate stays at or below α.

### 3.6 Logging and traceability

SQLite in WAL mode holds:

- inspections
- defects with their measurements and IPC results
- operator decisions
- every piece of advice: the model used, whether it passed grounding, why it was rejected, and its speed
- feedback events, which can be undone
- model versions
- drift alerts

Images are stored by content hash. Each inspection and its defects are written in one transaction *before* the stack light changes, so a power cut cannot lose a verdict an operator has seen. A test checks that a failing defect row rolls back the whole inspection. CSV export is one click.

## 4. Llama on ExecuTorch: advice the operator can check

**Model.** Llama 3.2 1B Instruct, the official SpinQuant INT4 checkpoint from Meta. It is exported with ExecuTorch's `export_llm` (8da4w, XNNPACK, KV cache, custom SDPA) to a 1.07 GB `.pte` and run through the Python `TextLLMRunner`. The 3B model was evaluated the same way.

**Design principle.** The mentor guidance was to give Llama structured findings, not raw pixels. We go one step further: **give Llama choices, not a blank page.** For each defect the knowledge base (`kb/defects.yaml`) lists 2 to 4 possible causes and 3 to 4 actions, each with an ID and tied to a process step.

**How the advice is generated:**

1. The board context is prefilled once.
2. For each defect, most severe first, the advisor appends `{"defect": "Dk", "cause": ` and Llama fills in two option numbers. Generation is stopped from the token callback as soon as both are written.
3. Llama writes a one-to-two sentence summary.
4. Option order is shuffled per board, so a habit of picking the first option cannot look like a good choice.

**Grounding checker.** It rejects:

- choices that are out of range or don't parse
- a summary containing a number that is not in the record or the stated history
- a summary naming a defect type that is not on the board

A rejected answer is replaced by standard knowledge-base advice. The rejection reason is logged and shown to the operator. The operator always reads the knowledge-base text verbatim for every cause and action, so instructions are never invented.

**What we measured, and what we learned.** Section 7 has the numbers. Three things matter most:

1. **Prompt v1 failed.** Asking for knowledge-base IDs as strings passed grounding about 15% of the time, because the 1B model copied whole option lists.
2. **Numbered choices fixed the format.** Format and grounding pass rates rose above 80%. But every answer picked option 1 and wrote one step. Before shuffling, a "100% chose a process cause" figure was an artefact of option order, so it is **not** claimed.
3. **Step-wise generation with shuffled options** makes coverage complete by construction. It also makes the choice measurable against chance.

Step order and coverage are enforced by the advisor and are not claimed as model skill.

**Other Llama uses on the station:**

- **Shift questions over the log.** Llama only picks one of five read-only, parameterised queries; it never writes SQL. Its summary is rejected if it contains a number that is not in the returned rows.
- **Drift alerts.** Raised by deterministic SPC rules, worded for the operator.

## 5. Running on the Arm CPU

**Everything runs through ExecuTorch on the Pi 5's Cortex-A76 cores:**

- detector `.pte` (FP32 and INT8)
- backbone `.pte` (FP32 and INT8)
- Llama `.pte`, with KleidiAI micro-kernels inside XNNPACK

**Benchmarks** (`bench/bench_device.py`) record per-model latency at 1, 2 and 4 threads, set through ExecuTorch's threadpool API. Each is recorded with the board model, OS, temperature and throttle flags. They also cover Llama load time, prefill and decode tokens/s from the runtime's own statistics, resident memory, and a per-stage end-to-end breakdown. A 30-minute soak test (`bench/soak.py`) records whether performance holds on a shift.

**Before the Pi was available,** the same aarch64 wheels and `.pte` files were run in an arm64 Debian-trixie container under emulation (`deploy/Dockerfile.pi`). That proved the Arm build is functional. Speed numbers are taken only from the real board.

## 6. Learning during use, safely

The brief asked for a model that is trained beforehand and also learns during use. Nothing is retrained by backpropagation on the device. Learning is done by **editing memories and fitting a small head**:

- **Teach a new design.** Photograph 5 to 10 good boards. Their features become that design's memory in seconds.
- **False alarm on an anomaly.** The region's features are added to the good memory.
  - *Novelty gate:* a region far beyond normal variation needs a second confirmation, because one wrong "it's fine" could blind the model to a whole defect class.
  - Every change can be undone.
- **Confirm or reject a named defect.** A crop embedding is stored with the label.
- **"Update the model".** A logistic re-scoring head is fitted on frozen embeddings and combined with the detector confidence as an independent-evidence update. It is switched on only if all of these hold:
  1. 5-fold cross-validation shows it separates real defects from false alarms better than the detector alone.
  2. It keeps at least 95% of confirmed defects.
  3. The conformal gate is re-calibrated with it on the shipped calibration boards.

  Every attempt, promoted or refused, is logged as a model version.

## 7. Results

See `docs/RESULTS.md`, generated from the experiment outputs. Every number there was measured, and none is projected.

## 8. Limitations, stated plainly

- The public datasets are scans or close-up photos. Accuracy on a live camera is reported separately from dataset accuracy, never merged with it.
- IPC limits come from secondary sources and are configurable.
- The conformal guarantee is marginal, and holds only for boards like the calibration set.
- The 1B model's choice among causes is only modestly better than chance. Its value is in a grounded, readable summary. The 3B comparison is in the results.
- HRIPCB's physical scale is not published, so its severities use defect-type defaults rather than millimetres.
