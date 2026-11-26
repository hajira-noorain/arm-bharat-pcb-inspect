# Demo video script (target about 3 minutes)

One story: **an inspection station that runs entirely on a Raspberry Pi 4's Arm CPU, learns a board from a few photographs, says what size of defect it can promise to find on that board, and only passes a board it can vouch for.**

**Rule:** what is on screen always matches what is being said at that moment, and every number spoken is one that was measured (results files named at the end). A value that was not measured is cut, never estimated.

**Sources:**

- Webcam recording from the station itself (Logitech QuickCam Pro 9000, 1600x1200): `footage_raw/webcam_*_full.mkv`.
- Phone B-roll: `footage_raw/phone/`.
- Screen recording of the dashboard (http://localhost:8000 over the Ethernet cable).
- Narration recorded separately afterwards.

Where the dashboard shows a capture replayed from tonight's photographs rather than a live one, the caption says "replayed capture".

## 0:00 to 0:15: What it is

| Time | Shot | Narration |
|---|---|---|
| 0:00 | Title card: "PCB inspection station", PS5, team name, Arm Bharat AI-SoC 2026-27 | "An offline PCB inspection station, running entirely on a Raspberry Pi 4." |
| 0:06 | Phone B-roll: wide shot of the Pi, the webcam on its mount, the paper with the corner marks | "A Pi 4, a USB webcam and a lamp. No cloud, no GPU: the vision model and the language model both run on the Pi's four Arm Cortex-A72 cores." |

## 0:15 to 0:25: Offline

| Time | Shot | Narration |
|---|---|---|
| 0:15 | Phone B-roll: the Ethernet cable into the Pi; the laptop showing the dashboard header "Raspberry Pi 4 · offline" | "The laptop only shows the dashboard over a cable. Nothing leaves the Pi." |

## 0:25 to 1:00: Teaching a board it has never seen

| Time | Shot | Narration |
|---|---|---|
| 0:25 | Webcam footage: the good rain-sensor pad placed into the corner marks, lifted, placed again | "This rain-sensor pad is new to the station. I show it the good board a few times." |
| 0:33 | Screen: Teach a board, captures counted, "Finish teaching", progress bar | "It then draws practice defects of known size onto those photographs and checks how many it finds." |
| 0:42 | Screen: the teaching result card: the sizes it can promise (shorts, mouse bites, pin holes) and the types it does not promise, plus the note "compared inside the outline of the copper pattern only" | "And it tells me, per defect type, the smallest size it finds at least 90 percent of the time with 95 percent confidence on this camera, and which types it cannot promise. [Fill from the teaching result.]" |

## 1:00 to 1:30: Good board passes, damaged board fails

| Time | Shot | Narration |
|---|---|---|
| 1:00 | Webcam footage: the good pad placed. Screen: green PASS band, design rain-sensor-pad, miss rate 10%. Caption: "replayed capture: one of the photos it was taught from" | "The good pad passes, in 2.8 seconds, at the 10 percent miss rate." |
| 1:05 | Screen: a REVIEW result (B00047): the same good pad from a later video frame, lined up at 0.79, under the 0.85 minimum | "When it cannot line a board up with the taught one well enough, it does not guess a pass: it sends the board to a person." |
| 1:08 | Phone B-roll: close-ups of the damaged pad: solder bridges, solder dot, knife cuts | "This one has two solder bridges, a stray blob of solder and knife cuts across the fingers." |
| 1:15 | Webcam footage: the damaged pad placed. Screen: red FAIL band, defects boxed, the defect list with sizes in millimetres | "It fails, in 12.9 seconds. The solder bridges, the stray solder and the knife cuts are boxed and measured in millimetres against the good board." |

## 1:30 to 2:00: Advice from Llama on the Pi

| Time | Shot | Narration |
|---|---|---|
| 1:30 | Screen: the "What to do" panel filling in: likely cause, the clue line, the check to confirm it, footer naming Llama 3.2 1B on ExecuTorch | "Our fine-tuned Llama 3.2 1B runs on the Pi through ExecuTorch. It reads the station's findings, not the pixels, and gives the likely cause, the clue behind it and the check to confirm it." |
| 1:45 | Screen: "What was it, once checked?": pick the cause, Save, the toast | "Once the operator has checked, they record what it actually was, and the station learns this line's causes from those answers without retraining." |

## 2:00 to 2:20: The assembled board

| Time | Shot | Narration |
|---|---|---|
| 2:00 | Webcam footage: the Arduino Uno under the camera; screen: its result | "The same station handles assembled boards by comparing parts against a taught good board. [Cut this segment if the Uno run was not done.]" |

## 2:20 to 2:45: Measured on the Pi 4

| Time | Shot | Narration |
|---|---|---|
| 2:20 | Results slide (numbers from `results/measured/pi4/`) | "All measured on this Pi 4: the defect detector takes 254 milliseconds per tile on four threads in INT8, a full DeepPCB board gets its verdict in 434 milliseconds, and Llama 3.2 1B generates 3.6 tokens per second, with the whole station using 1.7 gigabytes. In a three-minute soak it peaked at 63.3 degrees with no throttling." |
| 2:35 | Slide: ExecuTorch on the Cortex-A72 | "The ExecuTorch wheel's LLM kernels used ARMv8.2 instructions the Pi 4 does not have, so we rebuilt them for ARMv8.0." |

## 2:45 to 3:00: Honest limits and close

| Time | Shot | Narration |
|---|---|---|
| 2:45 | README "What is and is not validated" table | "What it does not promise, it says: on this webcam it does not promise opens or spurs on this pad, it inspects only the copper pattern, and on this pad it can pass a board at a 10 percent miss rate, not 5." |
| 2:53 | Architecture slide, then the GitHub link | "A station that measures what it can see and only promises what it can keep. Everything is on GitHub. Thank you." |

The full shot list, captions, slides and overlays are in `footage_raw/edit/` (EDIT_PLAN.md).

## Where the numbers come from

- Pi 4 benchmarks, soak and the Arm check: `results/measured/pi4/results_pi4_20261126_0235.tgz`.
- Teaching result, verdicts and times for the rain-sensor pad: the station's log on the Pi (`var/inspections.db`) and the screenshots taken from it.
- Fill every bracketed value from those, or cut the line.
