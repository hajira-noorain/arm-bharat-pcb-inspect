# Research, red-team findings and final decisions

13 Nov 2026. This record gathers three research tracks, three red teams and a code-level teardown of 8 rival repositories, plus the decisions taken from them. Every factual claim below was re-checked against its primary source by a fact-check pass. Claims that failed that check are listed under "Never claim".

## 1. Why PCB inspection is still not solved (verified evidence)

| Problem | Evidence (re-checked) | What we do about it |
|---|---|---|
| **False calls bury operators, and real defects then slip through** | "In some cases, AOI systems have captured three to four defects on a panel with hundreds of false alarms." "Operator acuity is impaired when verifying panels with high numbers of false calls as real defects inevitably escape." In one observed case operators "set the AOI parameters wider to reduce the number of alarms, rather than adjusting the production process." (SMT Magazine via iConnect007; undated online, likely 2000s.) Mycronic claims 50-100% fewer false calls on eligible components after retraining on the customer's images (vendor claim, 2024; its own engineer calls 50-90% realistic). | The station learns from false alarms on the floor. Every change is guarded and can be undone. False alarms per board are a reported KPI. Borderline boards go to a CHECK queue instead of a forced pass or fail. |
| **Programming each new board is slow** | Programming "for full solder inspection can, in some cases take a week". At one contract manufacturer, a new board took "four hours when a new component database needs to be defined" and under 30 minutes with an established one (iConnect007; dated). | **Teach a board** from good boards only, in seconds. The few-shot curve (accuracy against number of good boards) is measured, not claimed. |
| **Benchmarks are saturated, real conditions are not** | MVTec AD segmentation is near-saturated (best about 97.8% AU-PRO). On MVTec AD 2, which adds lighting variation, the evaluated methods score below 31% AU-PRO at a 0.05 false-positive limit (arXiv 2503.21622, 256x256 input). PKU-Market-PCB defects were "made by photoshop" on bare boards with no components. DeepPCB defects are partly added by hand and the images are binarised. | Results are reported per dataset, per source and split by physical board. Dataset accuracy and real-camera accuracy are reported separately and never merged. |
| **Thresholds give no guarantee on escapes** | Conformal risk control bounds the expected pixel-level miss rate on steel-surface data (Severstal, NEU; arXiv 2504.17721). We found no deployed commercial AOI that advertises a calibrated escape-rate bound. | Conformal PASS / FAIL / CHECK gate, worded honestly as a marginal bound, in expectation, at board level, and valid only for boards like the calibration set. Per-defect recall with exact confidence intervals is the headline figure. |
| **Black-box verdicts get overridden** | In a field experiment at a Siemens factory (n=48), heatmap explanations improved inspection accuracy by 7.7 percentage points (95% CI 3.3-12.0) over a black-box AI. Workers supported by the black-box AI were 3.6 times more likely to wrongly overrule a correct AI (Senoner et al., Scientific Reports 2024). | Every defect is drawn on the board with its measured size, the class decision and the knowledge-base reason. Llama never decides the verdict. |
| **Small data** | "Architectures built for hundreds of millions of images don't work with only 50 images." (Andrew Ng, IEEE Spectrum, 2022) | A frozen backbone, a memory of good boards and a small head trained on operator labels. No retraining of the network on the device. |
| **India context** | Electronics production was an estimated Rs 13.11 lakh crore in FY26 (MeitY). ELCINA estimates FY25 PCB demand at Rs 35,000 crore, with 88% imported. ECMS was originally notified at Rs 22,919 crore in April 2025, and has since been reported raised to Rs 40,000 crore. Estimates of the number of EMS companies range from 400+ to about 700, many of them small. IPC-A-610 is the widely used workmanship standard, and many Indian EMS firms hold IPC-certified staff. | **Target user:** a small EMS or PCB assembly shop that cannot justify a commercial AOI line and inspects by eye under a magnifier. **What they need:** offline operation, low cost, ACCEPT/REWORK/SCRAP against the customer's class, and a per-board record they can show their customer. |

**Never claim** (each failed the fact check, or the evidence is missing):

- "AOI programming takes about a week" as a general fact.
- Any iConnect007 number presented as current.
- The misquoted Andrew Ng line.
- "More than 3x more likely to override". The paper says 3.6x, against a black-box AI.
- The Mycronic figure without its "vendor claim" qualifier.
- "All SOTA under 31%" without naming the metric and input size.
- "~500 small/micro EMS".
- "IPC-A-610 is required in Indian OEM audits".
- ECMS Rs 22,919 crore presented as the current outlay.
- "PKU/DeepPCB defects are real".
- "Guaranteed escape rate on every board or lot".
- "First", "only" or "no one has".

## 2. Red-team findings and what changed

| Finding (red team) | Decision | Status |
|---|---|---|
| Measurements are graded below camera resolution (the Class 3 limit can be under 1 px) | Every measurement carries ±1 px. A class limit inside that band gives "uncertain" and a REVIEW disposition, never a guess. | Done, tested |
| Golden alignment quality is computed but never used, and registration slivers are read as nicks | No measurement when alignment cc < 0.85. Each defect crop is refined with phase correlation. Differences thinner than 2 px are rejected as slivers. | Done, tested |
| Local width taken from a nearby pad understates damage | The width is taken from the conductor's ridge (centre-line) widths next to the nick. | Done, tested |
| No sourced Class 1 limit. "Minor" read as "ignore". No dispositions. | Class 1 rejects functional failures only. Each design has a customer class. ACCEPT / REWORK / SCRAP / REVIEW come from a per-type repair policy, and severity is now only a sort key. Designs still on default design rules are flagged. | Done, tested |
| Copper segmentation is unvalidated on solder-masked boards | Claim withdrawn. The docstring and report state what has been validated. On masked boards the station shows type-level grading and no millimetres unless alignment and segmentation pass. | Done (wording). Validation needs real boards. |
| The conformal guarantee is board-level, calibration sets are tiny or degenerate (HRIPCB has 2 good boards, Mendeley none), and DeepPCB's good boards are templates | Per-defect recall with Clopper-Pearson intervals becomes the headline. Allowed-miss-rate options are hidden when the calibration set cannot support them (needs n ≥ 1/α − 1). HRIPCB and Mendeley are labelled "too few boards for a guarantee". The calibration source is stored. | In progress |
| The learning-loop check is not independent: folds share boards, the baseline is hard-coded, critical defects can be hidden | Cross-validation folds are grouped by board. The head must beat the detector's own score (AUROC). The head can only raise, never lower, confidence on critical types (short, open, missing hole). Per-defect recall on the shipped calibration defects must not fall. MIN_PER_CLASS is raised to 15. | In progress |
| Documents promise numbers the results folder does not hold | Every experiment is run (board-level detection, gate, anomaly, learning, 3B, Pi). Docs are rewritten from the generated tables. Any line without a number is cut. | In progress |
| 1B advice: position-1 bias at 78% against 42% chance, and near-chance process-cause choice | Measured both remedies. A worked example cut position bias to 61% but cause choice stayed at chance, and it adds 83 prompt tokens, so it is off. 3B passes grounding 98% (1B 88%) but its cause choice is also at chance. **Decision:** in production, causes are listed in the order a process rule ranks them (a repeating defect puts process causes first, a one-off puts handling first). Llama confirms, explains and summarises, and evaluation keeps shuffling to measure the model alone. | Done, tested |
| Default camera mode is 2028x1520 with automatic exposure | Full-sensor still capture. Exposure and white balance are locked after settling. | In progress |

## 3. Rival teardown (code read, nothing executed)

- **Strongest: QC-Assist (silentaf/BHARAT_SOC).**
  - **What it has:** real in-process ExecuTorch Llama; a deterministic policy plus a grounding checker; aarch64 numbers committed from a real Arm phone (12.5 tok/s decode, 1.3 GB RSS, 18/18 answers accepted); VisA results; Hindi output; a teach curve.
  - **What it lacks:** vision is not on ExecuTorch; no mm or IPC grading; no conformal gate; no learning from operator feedback; 2 tests.
- **edgeinspect-ai:** its Llama `.pte` is 1.7 KB, and its "benchmark" hard-codes the speeds with `time.sleep()`.
- **MadhanSaiKrishna:** Llama is mocked.
- **katyayanitiwari:** no Llama at all.
- **BhAi_SoC:** README only.

**Adopted from rivals:**

1. Commit device information and raw per-run JSON from real hardware, with tables generated from them.
2. Make the grounding-checker acceptance rate a headline metric.
3. A worked example in the prompt matched to the situation.
4. Optional Hindi: knowledge-base text translated once and reviewed, never free-form Hindi generation, because the Python binding has known Devanagari decoding issues.

**Our edge to show with evidence:**

- vision on ExecuTorch (FP32 and INT8)
- millimetre measurement with stated uncertainty
- class dispositions
- the conformal gate
- guarded learning that can be undone
- SPC drift alerts
- the GPIO stack light
- 80+ tests

## 4. Minimum physical setup for a credible real-board demo (from the QE red team)

- **Camera:** Raspberry Pi HQ camera (IMX477, 4056x3040) with a 16 mm C-mount lens at f/5.6-8, on a rigid vertical stand. A board field of about 100 mm gives about 40 px/mm. Exposure and white balance are locked.
- **Lighting:** diffuse ring light inside a hood. Cross-polarisers (a film on the light and a filter on the lens) to remove glare from solder mask and HASL.
- **Fixture:** a nest with edge stops so the board sits flat, and the ArUco marker at board-surface height.
- **Validation before any real-board claim:**
  - a known-good golden board and seeded defects checked under magnification
  - a repeatability run of the same board 30 times, reporting the mean and standard deviation of one measured nick
  - per-defect recall and false alarms per board with confidence intervals, and seconds per board on the Pi

## 5. Assembled boards: the parts check (research of 13 Nov 2026, and what it changed)

Three literature and practice reviews were run before and while building `pcbinspect/parts.py`. Sources that could not be opened are marked so; nothing below is quoted from a source we did not reach.

**What the evidence says**

- **"Unusual area" models are weak on absent or swapped parts.** On MVTec LOCO AD, PatchCore scores 74.0 image AUROC on logical anomalies (a required object missing, or one in the wrong place) against 89.3 on structural ones; a component-aware method reaches 98.1 on the logical ones (Kim et al., "Few Shot Part Segmentation Reveals Compositional Logic for Industrial Anomaly Detection", 2024, arXiv 2312.13783; the benchmark is Bergmann et al., IJCV 2022). Comparing a board part by part with the good board is the published remedy, and it is what the parts check does. Neither paper reports numbers on PCBs.
- **Classic optical inspection compares each component's window with a taught good board**, and early top-down 2D machines "mainly checked component presence, orientation, and solder presence" and "gave too many false calls" (P. Starkey, seminar report, I-Connect007, 2014). False calls, not misses, are the practical problem of this kind of check.
- **Appearance changes that are not faults are a known false-call source:** markings, package colours and finishes vary between lots and suppliers, and 2D comparison is sensitive to lighting and reflections (Koh Young America, "3D AOI vs 2D AOI"). A re-teach after a supplier change is normal practice.
- **A single top camera sees tall parts lean** with their distance from the optical axis; machines use telecentric lenses, side cameras or 3D for this (vendor documentation; the telecentric claim was read in search summaries only).
- **Resolution.** Inspection machines state 7 to 15 micrometres per pixel for the smallest parts (Yamaha YSi-V product page: 12 or 7 micrometre lens options). A 12-megapixel camera covering a 120 mm field gives about 30 micrometres per pixel: an 0402 part is about 34 by 17 pixels and an 0201 about 20 by 10. Our own arithmetic, not a published limit.
- **Component detection is hardest on the smallest and rarest parts** (Kuo et al., "Data-Efficient Graph Embedding Learning for PCB Component Detection", WACV 2019: 0.653 mAP on 48 images, with tiny parts the stated bottleneck). FPIC (Jessurun et al., 2023) labels parts by their printed designator and publishes no detector accuracy and no pixels-per-mm guidance.
- **Real test photos.** VisA pcb1 to pcb4 (Zou et al., 2022, CC BY 4.0) is the public set with good photos of the same design plus faults labelled missing, extra and wrong place. Its faults were made by hand on real boards, about 100 faulty photos per design. RobustAD's PCB subset (Amazon, 2025, CC BY 4.0) and Real-IAD's `pcb` object (Wang et al., 2024, non-commercial) also carry a missing-parts label. PCB-MC (2026, CC BY 4.0) has 9,435 labelled empty footprints but no good photo of each design.
- **No published accuracy to compare with.** We found no verified result for reference-based missing-part detection on real assembled boards, no per-fault-type PatchCore result on VisA pcb4, and no measured result for checking a part's identity from its marking.

**What it changed in the design**

1. Two independent cues instead of one: the part detector, and a direct comparison of each part's picture with the good board (the classic component-window comparison). A part is called missing only when both agree.
2. The map is taught from several good boards: parts the detector does not find every time are left off, and each part's normal board-to-board variation sets its own comparison threshold.
3. Every parts finding that calibrated evidence does not back sends the board to a person; it never fails a board by itself. A board that cannot be lined up with the taught one is not passed.
4. A third finding, "part looks different", covers a part that is present but clearly unlike the good one. Its advice includes the approved-alternative case and the re-teach.
5. The first real test is VisA, reported by fault type, with the result of each cue shown separately.

**What the documentation must not claim**

- That the station inspects solder joints, polarity, part value or lifted leads, or that it meets an IPC or NASA workmanship class.
- Any IPC-A-610 placement limit: we could not open the standard, and the secondary sources disagree.
- That 0201 parts are within reach of the camera, or that any pixels-per-mm figure is "validated".
- A share of real assembly defects covered: no citable breakdown was found.
- That the FPIC detector numbers are comparable with other papers' numbers (different data, classes and metrics), or that they hold on other boards and cameras before the VisA result says so.
- Knowledge-base causes for missing and tombstoned parts follow published troubleshooting notes (nozzle, vacuum, feeder, programme coordinates; paste, pad geometry and heating for tombstones: G. Smith, "SMT Stencils 101", I-Connect007, 2019). The causes listed for an unexpected part and for a part that looks different are our own engineering judgement: no source was found for them.
