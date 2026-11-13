# A station that knows, states and proves its own limits

Two features build on the inspection engine:

1. **The eye test.** The station measures the smallest defect it reliably finds on *this* camera, lighting and software, and says so in one sentence per defect type.
2. **Customer rules.** Llama on the station turns a customer's plain-language acceptance criteria into rules the station executes. Each rule is checked against the text, and against the eye-test result, before the operator confirms it.

The measured numbers are in `RESULTS.md`, generated from the result files; this document explains the method. Every number quoted here names its source file.

## 1. The eye test

### Why

Every AOI vendor page we found quotes a bare figure like "up to 98% accuracy". None says for what defect size, under what lighting, or measured how. A shop inspecting by eye has no number at all. When a board escapes, nobody can show what the inspection could have caught.

Non-destructive testing solved this decades ago with the probability of detection (POD) versus flaw size. The 90/95 figure means "found with 90% probability at 95% confidence". The sources are MIL-HDBK-1823A and ASTM E2862. We apply that practice to a low-cost PCB station.

### How

**The card.** Twelve inkjet-printed A4 pages (`deploy/make_eyetest_card.py`):

- **Patterns.** Held-out DeepPCB board patterns (MIT licence), never used for training.
- **Sheet 0** is defect-free: the golden reference.
- **Sheets 1–11** carry drawn defects of known size, 30 of each (type, size).
  - Types: opens, shorts, pin-holes, spurious copper, mouse bites, spurs.
  - Sizes: 0.2–1.0 mm.
  - Shorts are sized by the gap they bridge, 0.15–0.5 mm (see below).
- **Corner markers.** Four ArUco corner markers let the station rectify any photo to exact sheet coordinates. A fifth marker identifies the sheet.
- **Size of the frame.** The frame (125 × 64 mm) fits the Pi Camera Module 3's view at its closest focus. This matters because the test must run at the same zoom as real boards: the limit belongs to that setup.

**Every drawn defect is checked** (`pcbinspect/defect_render.py`):

- An open must actually split the conductor (the conductor count goes up).
- A short must actually join two different conductors.
- A bite must not sever the conductor.
- A spur must touch nothing else.

**Reading the card** (`pcbinspect/eyetest.py`):

1. Rectify the photo and identify the sheet.
2. For each pattern, run the same evidence the station uses on boards (detector, and the golden difference against sheet 0).
3. Count a planted defect as found only if a finding at or above the gate's operating threshold covers it.
4. Count false calls on the rest.

**Statistics** (`pcbinspect/pod.py`):

- **Curve fit.** A hit/miss logistic model on ln(size), with Firth's bias reduction so that perfect separation (all small misses, all large hits) doesn't break the fit.
- **The 90/95 bound.** Taken from the profile likelihood, the bound MIL-HDBK-1823A recommends for hit/miss data.
- **Model-free check.** A demonstrated window: the tested sizes over which the pooled Clopper-Pearson 95% lower bound is at least 90%, with no individual size significantly below 90%.
- **The stated limit** is the more conservative of the curve's bound and the window. It never extends beyond the largest tested size.
- **Tests** (`tests/test_pod.py`) include a coverage test: on simulated data from a known curve, the 95% bound covers the true size in at least 90% of 150 repetitions.

### What the eye test found in our own system (and what we changed)

The eye test was built to certify the station. Its first job was to find what was wrong with it.

**1. The miss-rate guarantee covered boards, not defects.**

- The first gate calibrated the board-level score, the strongest defect on the board.
- On DeepPCB at a 5% miss setting, 95% of defective boards were caught, but only **34% of individual defects** were shown. Boards carry several defects, and the maximum is high.
- The gate is now calibrated per defect: one randomly chosen ground-truth defect per calibration board, scored by the strongest covering detection. This gives the finite-sample conformal guarantee for "a defect on a new board is shown with probability ≥ 1−α". The board-level guarantee follows automatically.
- Result on held-out boards: about 94% of defects shown. The cost is more false calls per board. Source: `bare_eval_ep70.json` (`pcbinspect/gate.py`).

**2. A learned detector has a detection window, not a limit.**

- Drawn defects showed detection getting worse as defects got bigger. A large clean bridge between two pads looks like a designed trace, and a large round hole looks like a via.
- The detector learned the small, irregular defects of its training data.
- Consequences:
  - The statistics no longer assume POD rises with size. They state a window and say when larger defects are missed significantly more often.
  - The station compares against the golden board.

**3. Golden-board difference channel** (`pcbinspect/golden_diff.py`). Any copper change is found, and bigger changes are always easier. Four problems found on the way, each tested:

- **Colour noise.** JPEG colour blocks on webcam-grade images were taken as copper. The copper channel is now chosen by absolute contrast, not contrast relative to its own spread. Before the fix there were 29 false calls per board under webcam conditions; after it, about none on good boards.
- **Registration slivers.** Warp and scale differences between scans left thin strips along edges. Two fixes: local re-registration by phase correlation, and dropping thin, elongated differences that hug a golden edge along their length.
- **Thin shorts and opens.** These were deleted as slivers. A netlist check now catches them regardless of shape: two golden conductors joined in the test image means a short, and one golden conductor split means an open. That electrical evidence overrides shape and sets the score.
- **Narrow-gap shorts.** After print spread and lens blur, a narrow gap is a grey line that any global threshold calls copper. Differences are now taken in per-pixel copper fraction (0 = bare, 1 = copper) instead of black-and-white masks. On binary scans this is identical to the old method.

**4. For a short, the size that matters is the gap.** Any bridge is a functional short. Whether the station sees it depends on how narrow a gap the optics resolve. Shorts are therefore sized by the gap they bridge, and the stated limit reads as a spacing-resolution figure.

### What the eye test does not prove

- **Paper is not copper.** The card proves the limit on the card, for this rig. Carrying it over to real boards is a separate check: Phase 1 uses the real etched-board scans; Phase 2 adds real defective boards.
- **The simulated photos are a model.** Until the card is photographed on the Pi (20–21 Nov), card results come from a printing-and-camera model (`training/eyetest_sim.py`): inkjet dot gain, perspective, about 30 px/mm, blur, noise and JPEG. That validates the reader end to end; it does not validate the camera.
- **Not a certification.** It is a measured, re-runnable statement about one rig.

## 2. Customer rules, compiled by Llama and checked

### Why this job for Llama

We measured Llama choosing defect causes from a list: it was at chance level (`advice_eval_*.json`). A small model is better used where its output can be checked. Turning a sentence into a few fields is such a task: the number, defect, comparison and action can each be checked against the text.

### How

**The rule language** (`pcbinspect/spec.py`):

- Fields: defect type; measure (always, size, width-loss %, spacing-loss %, count per board); comparison ("more than" or "at least"); value; action (reject, rework, review, accept).
- Class-only rules set the IPC-style class.

**Llama fills the fields one at a time** on one KV cache (`pcbinspect/spec_llm.py`):

- Answers are words from listed options, not option numbers. Numbers showed a strong position-1 bias in the advice tests.
- Option lists are shuffled per clause during evaluation.

**Deterministic checks** (`spec.verify`):

- The schema is valid.
- The number appears in the clause after unit conversion (mm, mil, micron, percent). This catches a 0.1 mm requirement read as 0.01 mm, or a mil value read as mm.
- The defect word, comparison words and action words are consistent with the text. A disagreement sends the rule to the operator; nothing is silently fixed.

**Two independent readers:**

- Llama and a keyword parser each compile the clause.
- If they agree and the checks pass, the rule is offered as verified. Otherwise the operator sees both and chooses.
- This matters because the keyword checks cannot see the keyword parser's own mistakes: they share its vocabulary.

**Capability against the eye test** (`spec.capability`):

- If a rule needs defects smaller than the station's measured limit, it is marked NOT GUARANTEED, with the size gap stated. Example: "the rule needs defects from 0.10 mm found, but the station is only proven from 0.18 mm".
- Percentage rules are converted to millimetres through the design's minimum trace width and spacing.

**Execution.** Confirmed rules decide the disposition of the defect types they cover, and the IPC-style table covers the rest. A rule that needs a measurement the station could not make cannot clear a defect.

### Benchmark

- **Two clause sets:**
  - 80 clauses written while developing the checks: the tuning set.
  - 40 written independently and never used for tuning (`llm/spec_clauses_heldout.yaml`): the held-out set.
- **Two kinds of test:**
  - Rules are compared with gold rules, both field by field and by executing them on a probe grid of defects.
  - The checks are also tested on deliberately corrupted gold rules.
- **The key number is "silently wrong":** the checks passed, but the rule is wrong.
- Results: `spec_eval_keyword.json`, `spec_eval_1b.json`, `spec_eval_3b.json`.

### Bugs found by the benchmark

- **A yes/no "class only?" first field.** Llama 1B answered "yes" for every clause.
- **Few-shot examples were copied.** 1B answered "none", the first example's answer, for every clause, and 3B over-answered the example's defect.
- **Every answer was fed back into the model**, although the generated tokens were already in the KV cache. Each answer appeared twice in the context and the model began repeating itself.
