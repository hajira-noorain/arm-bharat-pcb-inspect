# Demo video script (target about 3 minutes 15 seconds)

One story: **every inspection system claims a high accuracy; ours measures the smallest defect it can find on your camera and on your board, says what it cannot see, and refuses rules it cannot keep.** The eye-chart analogy carries it. Every PS5 requirement still appears on screen.

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

## 1:05 to 1:45: A real board, taught and inspected live

Everything in this segment is the station as built: the Teach a board screen (Setup), then Inspect. Rehearse it on the Pi before filming; if a step does not behave as written, film what it does and change the line.

| Time | Shot | Narration |
|---|---|---|
| 1:05 | Screen: Setup, Teach a board. A real bare board against the corner stops with the scale marker; "Capture a good board" pressed three times, the board lifted and put down between presses | "Now a real board the station has never seen. I show it the good board three times." |
| 1:13 | Screen: "Finish teaching", progress bar, then the result with the stated limits ("Finds opens from ... mm") and the line "Measured on test defects drawn onto your own board, not on real defects" | "It draws test defects of known size onto those pictures, sets its own pass and fail thresholds, and tells me what size of defect it can find on this board. That took [teach seconds] seconds on the Pi." |
| 1:22 | The same good board under the camera; foot pedal; green PASS band and green light | "The good board passes." |
| 1:27 | Hands: a knife cut across one trace. Board back under the camera; pedal. Red FAIL band, the cut boxed; picture-in-picture of the red light | "I cut one trace. It fails in [verdict time] seconds: an open circuit, at the cut." |
| 1:35 | Screen: defect list zoomed: "Open circuit", size in mm, Class 1/2/3 chips; then the "What to do" panel with a knowledge-base ID and the footer naming the Llama model | "It measures the cut in millimetres against the reference picture and grades it. Llama 3.2, on the Arm CPU through ExecuTorch, reads the findings, never the pixels, and writes the summary; every cause and action shown is knowledge-base text, and an answer that states anything else is rejected." |

## 1:45 to 2:00: It says when it is unsure

| Time | Shot | Narration |
|---|---|---|
| 1:45 | Screen: an amber CHECK band on a borderline board, with Pass/Fail buttons | "When the evidence is borderline, the board goes to a person: amber light." |
| 1:52 | Screen: switch the allowed miss rate from 5% to 1%; cut to the gate table in RESULTS.md with the measured value highlighted | "The operator chooses a target for how many defects may slip through. On boards it had never seen, at the 5 percent setting, it missed 4.0 percent." |

## 2:00 to 2:30: It refuses rules it cannot keep

| Time | Shot | Narration |
|---|---|---|
| 2:00 | Screen: Customer rules tab; paste "Reject any board with an open circuit wider than 0.1 mm" | "Customers write their acceptance rules in plain English." |
| 2:07 | Screen: the compiled rule rendered back as a sentence, green "checked" chips; footer "Llama 3.2 3B on this device" | "Llama 3.2 3B, on the Pi, turns each clause into a rule, and our checks verify every number and defect against the clause." |
| 2:17 | Screen: a clause asking for a smaller size than the card certified; the capability flag the station shows: "proven to find these from [x] mm; smaller ones may be missed" | "And when a rule asks for a defect smaller than the eye test showed it can see, the station says so, instead of silently accepting a promise it cannot keep." |

## 2:30 to 2:50: Measured on the Pi

| Time | Shot | Narration |
|---|---|---|
| 2:30 | RESULTS table: detector on ExecuTorch vs Arm's optimised runtime, Llama tokens/s and memory, 30-minute soak with temperature | "Everything here was measured on the Pi itself: [detector ms] per tile on Arm's optimised runtime, Llama at [decode tokens per second], and [peak temperature] after 30 minutes with no throttling." |

## 2:50 to 3:05: Honest about what is not proven

| Time | Shot | Narration |
|---|---|---|
| 2:50 | README table "What is and is not validated", the right-hand column highlighted | "What we have not proven: the limit for etched copper rather than drawn defects, solder-masked boards under shop lighting, and solder joints on assembled boards. The report lists every one." |

## 3:05 to 3:15: Close

| Time | Shot | Narration |
|---|---|---|
| 3:05 | Architecture slide (two pipelines), then the GitHub link | "A station that measures what it can see, says what it cannot, and keeps only the promises it can. Code, report and every measurement are on GitHub. Thank you." |

## Fill-ins after the Pi session

Fill every bracketed value from `results/device_bench.json`, `results/soak_*.json`, the real eye-test result in the station's log, and `docs/RESULTS.md`, never estimates. If a number is not measured, the line is cut rather than guessed.
