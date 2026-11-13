# Demo video script (target about 3 minutes 15 seconds)

One story: **every inspection system claims a high accuracy; ours measures the smallest defect it can find on your camera, says what it cannot see, and refuses rules it cannot keep.** The eye-chart analogy carries it. Every PS5 requirement still appears on screen.

**Rule:** what is on screen always matches what is being said at that moment. If a line talks about the stack light, the stack light is on screen.

**Recording:**

- Two sources, both at 1080p: a phone on a tripod filming the physical station, and a screen capture of the dashboard.
- Narration is recorded separately and cleaned with the video-cleanup skill.
- Numbers on screen come from `docs/RESULTS.md` and the on-device `results/device_bench.json` and `results/soak_*.json`. A bracketed value that was not measured is cut, never guessed.

## 0:00 to 0:15: The question nobody answers

| Time | Shot | Narration |
|---|---|---|
| 0:00 | Title card: "PCB inspection station", PS5, team name, Arm Bharat AI-SoC 2026-27 | "Hello judges. Every inspection system claims a high accuracy. But ask it the one thing a factory needs to know, the smallest defect it will catch on *your* camera, and it has no answer. Ours does." |

## 0:15 to 0:30: The device, offline

| Time | Shot | Narration |
|---|---|---|
| 0:15 | Wide shot: Pi 5 with active cooler, camera on a stand, diffused light, stack light, phone beside it | "This is the whole station: a Raspberry Pi 5, a camera, a light and a stack light. Everything runs on the Pi's Arm CPU." |
| 0:23 | Close-up: phone in airplane mode joining the Pi's "PCB-Station" Wi-Fi and opening the dashboard | "The phone joins the Pi's own Wi-Fi. There is no internet for this whole video." |

## 0:30 to 1:05: The eye test

| Time | Shot | Narration |
|---|---|---|
| 0:30 | Hands placing a printed card sheet under the camera; close-up of the coupons with tiny printed defects | "Like an eye chart, this printed card carries 252 copper patterns, each with one defect of known size, from a fifth of a millimetre up." |
| 0:40 | Screen: Eye test tab, sheets being captured one by one, hit/miss marks appearing on the coupons | "The station photographs each sheet and inspects every pattern, exactly as it inspects a board." |
| 0:50 | Screen: the statement panel, e.g. "Opens: certified from [x] mm. Mouse bites: not certified below [y] mm" | "Then it states, for each defect type, the smallest size it finds 90 percent of the time with 95 percent confidence, the standard used in aerospace testing. And it says plainly what it cannot certify." |

## 1:05 to 1:40: Live inspection of a real board

| Time | Shot | Narration |
|---|---|---|
| 1:05 | A real bare board with a knife-cut trace and the scale marker under the camera. Foot presses the pedal. | "Now a real board with a cut trace." |
| 1:10 | Screen: red FAIL band, the defect marked; picture-in-picture of the stack light turning red with a beep | "It fails in [verdict time] seconds. Red light, defect marked." |
| 1:17 | Screen: defect list zoomed: type, size in mm, Class 1/2/3 chips | "It compares the board with a known-good image of the same design, measures the cut in millimetres, and grades it against IPC class limits." |
| 1:27 | Screen: the "What to do" panel filling in; footer "Written by llama3_2_1b_spinquant on this device"; a knowledge-base ID highlighted | "Llama 3.2, running on the Arm CPU through ExecuTorch, reads the findings, never the pixels, and picks the likely cause and next step from our defect knowledge base. If it states anything not in the findings, its answer is rejected." |

## 1:40 to 2:00: It says when it is unsure

| Time | Shot | Narration |
|---|---|---|
| 1:40 | Screen: an amber CHECK band on a borderline board, with Pass/Fail buttons | "When the evidence is borderline, the board goes to a person: amber light." |
| 1:47 | Screen: switch the allowed miss rate from 5% to 1%; cut to the gate table in RESULTS.md with the measured value highlighted | "The operator chooses how many defects may slip through, and the station guarantees it. On boards it had never seen, it missed [value] of defects at a 5 percent setting." |

## 2:00 to 2:30: It refuses rules it cannot keep

| Time | Shot | Narration |
|---|---|---|
| 2:00 | Screen: Customer rules tab; paste "Reject any board with an open circuit wider than 0.1 mm" | "Customers write their acceptance rules in plain English." |
| 2:07 | Screen: the compiled rule rendered back as a sentence, green "checked" chips; footer "Llama 3.2 3B on this device" | "Llama 3.2 3B, on the Pi, turns each clause into a rule, and our checks verify every number and defect against the clause." |
| 2:17 | Screen: a clause asking for a smaller size than the card certified; red flag "This station cannot guarantee this: its measured limit for opens is [x] mm" | "And when a rule asks for a defect smaller than the eye test showed it can see, the station says so, instead of silently accepting a promise it cannot keep." |

## 2:30 to 2:50: Measured on the Pi

| Time | Shot | Narration |
|---|---|---|
| 2:30 | RESULTS table: detector on ExecuTorch vs Arm's optimised runtime, Llama tokens/s and memory, 30-minute soak with temperature | "Everything here was measured on the Pi itself: [detector ms] per tile on Arm's optimised runtime, Llama at [decode tokens per second], and [peak temperature] after 30 minutes with no throttling." |

## 2:50 to 3:05: It learns on the floor

| Time | Shot | Narration |
|---|---|---|
| 2:50 | Screen: "Teach a board" tab; eight good boards captured; "Learned … in N seconds" | "A new design is taught from a few good boards, in seconds. Operator corrections are learned too, and switched on only if they hide no real defects." |

## 3:05 to 3:15: Close

| Time | Shot | Narration |
|---|---|---|
| 3:05 | Architecture slide (two pipelines), then the GitHub link | "A station that measures what it can see, says what it cannot, and keeps only the promises it can. Code, report and every measurement are on GitHub. Thank you." |

## Fill-ins after the Pi session

Fill every bracketed value from `results/device_bench.json`, `results/soak_*.json`, the real eye-test result in the station's log, and `docs/RESULTS.md`, never estimates. If a number is not measured, the line is cut rather than guessed.
