# Demo video script (target about 3 minutes)

The structure follows a typical submission video: address the judges, show the setup, run it live, then measured numbers. It covers every PS5 requirement.

**Rule:** what is on screen always matches what is being said at that moment. If a line talks about the stack light, the stack light is on screen.

**Recording:**

- Two sources, both at 1080p: a phone on a tripod filming the physical station, and a screen capture of the dashboard.
- Narration is recorded separately and cleaned with the video-cleanup skill.
- Numbers on screen come from `docs/RESULTS.md` and the on-device `results/device_bench.json`.

## 0:00 to 0:12: Opening

| Time | Shot | Narration |
|---|---|---|
| 0:00 | Title card: "PCB inspection station", PS5, team name, Arm Bharat AI-SoC 2026-27 | "Hello judges. This is our entry for problem statement 5: a PCB inspection station that finds defects, explains them and learns, entirely on a Raspberry Pi." |

## 0:12 to 0:30: The device, offline

| Time | Shot | Narration |
|---|---|---|
| 0:12 | Wide shot: Pi 5 with active cooler, camera on a stand, ring light, three-LED stack light, phone beside it | "This is the whole station: a Raspberry Pi 5, a camera, a light and an LED stack light. There is no cloud." |
| 0:22 | Close-up: phone in airplane mode joining the Pi's "PCB-Station" Wi-Fi, opening the dashboard | "The phone joins the Pi's own Wi-Fi to show the dashboard. The internet is switched off for this whole video." |

## 0:30 to 1:20: Live inspection of a real board

| Time | Shot | Narration |
|---|---|---|
| 0:30 | A real bare board with a knife-cut trace and the 20 mm marker under the camera. Foot presses the pedal. | "I place a board and press the foot pedal." |
| 0:36 | Screen: red FAIL band, the board with D1 and D2 marked, the stack light turning red with a beep (picture-in-picture) | "It fails in under [verdict time from device_bench] seconds. The red light comes on and the defects are marked." |
| 0:46 | Screen: the defect list, zoomed. "Open circuit, critical, cuts the full conductor" and the C1/C2/C3 reject chips. | "Each defect is named and measured in millimetres against the board's own known-good image, then graded against IPC Class 1, 2 and 3 limits." |
| 1:00 | Screen: the "What to do" panel filling in. Footer: "Written by llama3_2_1b_spinquant on this device". | "Llama 3.2, running on the Pi's Arm CPU through ExecuTorch, picks the likely cause and next action for each defect. It reads only the structured findings, never the pixels." |
| 1:12 | Screen: highlight a knowledge-base ID beside an action. Cut to a rejected example: "Llama's answer was rejected: summary contains number 57…" | "Every cause and action comes from our defect knowledge base. If Llama states anything that is not in the findings, its answer is rejected and the operator gets standard advice instead." |

## 1:20 to 1:45: The guaranteed miss rate

| Time | Shot | Narration |
|---|---|---|
| 1:20 | Screen: an amber CHECK band on a borderline board, with Pass/Fail buttons | "When the evidence is borderline, the board goes to a person: amber light." |
| 1:28 | Screen: switch the allowed miss rate from 5% to 1%. Cut to the gate table in RESULTS.md, with the measured miss rate highlighted. | "The operator chooses how many defective boards may slip through. Conformal calibration guarantees that rate. On held-out boards the measured miss rate was [value] at a 5% setting." |

## 1:45 to 2:15: It learns on the floor

| Time | Shot | Narration |
|---|---|---|
| 1:45 | Screen: "Teach a board" tab. Capture eight good Arduino boards, then "Learned … in N seconds". | "To teach a new board design, we show it a few good boards. No training run is needed." |
| 1:58 | Screen: a false alarm marked. Toast: "Added to the good-board memory. Undo". | "When the operator marks a false alarm, the station learns from it, and the change can be undone." |
| 2:05 | Screen: "Update the model" with its result line, then the learning table from RESULTS.md (false alarms per board before and after) | "Model updates are switched on only if they hide no real defects. On held-out boards, false alarms dropped from [x] to [y] per board." |

## 2:15 to 2:40: Measured on the Pi

| Time | Shot | Narration |
|---|---|---|
| 2:15 | RESULTS table: per-stage latency, FP32 vs INT8, 1/2/4 threads, Llama tokens/s and memory, 30-minute soak with temperature | "Everything runs through ExecuTorch on the Arm cores. Here is what we measured on the Pi itself: [detector ms], [Llama decode tokens per second], and [peak temperature] after a 30-minute soak with no throttling." |
| 2:30 | Trends tab: Pareto chart and a drift alert "short on 7 of the last 20 boards: check etching" | "It also watches the line. When one defect starts repeating, it tells the operator which process step to check." |

## 2:40 to 2:55: Close

| Time | Shot | Narration |
|---|---|---|
| 2:40 | Architecture slide showing the two pipelines, then the GitHub link | "Two pipelines, both on the Arm CPU: vision and Llama. The code, the report and every measurement are on GitHub. Thank you to Arm, MeitY and IIT Delhi." |

## Fill-ins after the Pi session

Fill in every bracketed value from `results/device_bench.json`, `results/soak_*.json` and `docs/RESULTS.md`, never estimates. If a number is not measured yet, the line is cut rather than guessed.
