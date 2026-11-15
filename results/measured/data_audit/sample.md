# Training data sample (for reading by hand)

Each cause example: the clues the model sees for the defect, the cause the generator used, and the label (probability over causes). Mark anything a process engineer would disagree with.

## 1. missing_part
- Clues: same spot on 1 of the last 15 boards; near the edge; round; 1 of this type on the board; steady on recent boards.
- generator's cause: part knocked off after soldering (MPART-C3)
- label: part knocked off after soldering 0.75, part moved or stood up in reflow 0.19, part was never placed 0.06

## 2. pin_hole
- Clues: same spot unknown; near the edge; round; 7 of this type on the board; rising on recent boards.
- generator's cause: plating voids (PHOLE-C2)
- label: plating voids 0.69, dust or bubbles at lamination 0.31

## 3. missing_part
- Clues: same spot on 14 of the last 17 boards; not near the edge; irregular shape; 8 of this type on the board; trend unknown; also on the board: hole breakout.
- generator's cause: part was never placed (MPART-C1)
- label: part was never placed 0.82, part moved or stood up in reflow 0.17, part knocked off after soldering 0.00

## 4. missing_pad
- Clues: same spot on 4 of the last 11 boards; not near the edge; irregular shape; 3 of this type on the board; rising on recent boards; also on the board: changed part, foreign object.
- generator's cause: resist or phototool defect at pad (MPAD-C2)
- label: resist or phototool defect at pad 0.65, over-etching at pad 0.34, pad lifted or scraped 0.00

## 5. hole_breakout
- Clues: same spot unknown; not near the edge; irregular shape; 7 of this type on the board; rising on recent boards; also on the board: foreign object, missing pad.
- generator's cause: drill-to-pad misregistration (HBRK-C1)
- label: drill-to-pad misregistration 0.42, pattern shifted at exposure 0.37, drill wander 0.21

## 6. spur
- Clues: same spot on 16 of the last 16 boards; not near the edge; irregular shape; 6 of this type on the board; trend unknown.
- generator's cause: incomplete development (SPUR-C2)
- label: incomplete development 0.39, under-etching 0.32, phototool spot or scratch 0.29

## 7. scratch
- Clues: same spot on 1 of the last 19 boards; near the edge; long and thin; 2 of this type on the board; trend unknown.
- generator's cause: panel handling or stacking (SCR-C1)
- label: panel handling or stacking 0.76, scratched resist before etch 0.14, rollers or conveyor 0.10

## 8. spur
- Clues: same spot on 0 of the last 8 boards; not near the edge; irregular shape; 4 of this type on the board; rising on recent boards.
- generator's cause: incomplete development (SPUR-C2)
- label: under-etching 0.67, incomplete development 0.33, phototool spot or scratch 0.00

## 9. spur
- Clues: same spot on 0 of the last 11 boards; not near the edge; irregular shape; 1 of this type on the board; trend unknown; also on the board: unknown anomaly.
- generator's cause: incomplete development (SPUR-C2)
- label: incomplete development 0.47, under-etching 0.46, phototool spot or scratch 0.07

## 10. missing_hole
- Clues: same spot on 0 of the last 13 boards; not near the edge; irregular shape; 7 of this type on the board; trend unknown.
- generator's cause: broken drill bit (MHOLE-C2)
- label: broken drill bit 0.95, drill program error 0.05

## 11. spur
- Clues: same spot unknown; not near the edge; round; 3 of this type on the board; trend unknown; also on the board: extra part, mouse bite.
- generator's cause: under-etching (SPUR-C1)
- label: under-etching 0.58, incomplete development 0.36, phototool spot or scratch 0.06

## 12. scratch
- Clues: same spot on 0 of the last 9 boards; not near the edge; long and thin; 1 of this type on the board; rising on recent boards; also on the board: extra part, spur.
- generator's cause: rollers or conveyor (SCR-C2)
- label: panel handling or stacking 0.36, rollers or conveyor 0.32, scratched resist before etch 0.32

## 13. mouse_bite
- Clues: same spot on 0 of the last 16 boards; not near the edge; round; 3 of this type on the board; trend unknown.
- generator's cause: over-etching or undercut (MBITE-C1)
- label: over-etching or undercut 0.53, poor resist adhesion at edge 0.24, bubbles in pattern plating 0.23

## 14. missing_hole
- Clues: same spot on 1 of the last 11 boards; not near the edge; irregular shape; 2 of this type on the board; rising on recent boards; also on the board: hole breakout.
- generator's cause: broken drill bit (MHOLE-C2)
- label: broken drill bit 0.96, drill program error 0.04

## 15. missing_hole
- Clues: same spot on 5 of the last 10 boards; not near the edge; round; 2 of this type on the board; rising on recent boards.
- generator's cause: drill program error (MHOLE-C1)
- label: drill program error 0.64, broken drill bit 0.36

## 16. open
- Clues: same spot on 8 of the last 12 boards; edge unknown; irregular shape; 1 of this type on the board; trend unknown.
- generator's cause: resist scratch or poor adhesion (OPEN-C1)
- label: dust or hair on phototool 0.85, resist scratch or poor adhesion 0.10, over-etching 0.02

## 17. scratch
- Clues: same spot on 1 of the last 20 boards; not near the edge; irregular shape; 2 of this type on the board; trend unknown.
- generator's cause: panel handling or stacking (SCR-C1)
- label: scratched resist before etch 0.43, panel handling or stacking 0.43, rollers or conveyor 0.13

## 18. foreign_object
- Clues: same spot on 1 of the last 15 boards; not near the edge; long and thin; 7 of this type on the board; trend unknown.
- generator's cause: airborne dust or fibre (FOB-C1)
- label: airborne dust or fibre 0.78, process residue 0.16, metal particle from tooling 0.06

## 19. spurious_copper
- Clues: same spot unknown; not near the edge; long and thin; 3 of this type on the board; steady on recent boards; also on the board: changed part.
- generator's cause: incomplete etching (SPCU-C1)
- label: incomplete etching 0.58, resist residue or phototool spot 0.26, contamination before etching 0.16

## 20. missing_part
- Clues: same spot unknown; not near the edge; round; 1 of this type on the board; trend unknown; also on the board: spur.
- generator's cause: part moved or stood up in reflow (MPART-C2)
- label: part was never placed 0.52, part moved or stood up in reflow 0.31, part knocked off after soldering 0.17

## 21. short
- Clues: same spot on 0 of the last 17 boards; edge unknown; round; 1 of this type on the board; steady on recent boards.
- generator's cause: under-etching (SHORT-C1)
- label: conductive debris on panel 0.58, under-etching 0.22, phototool spot or resist residue 0.20

## 22. missing_pad
- Clues: same spot on 0 of the last 15 boards; edge unknown; round; 2 of this type on the board; steady on recent boards.
- generator's cause: resist or phototool defect at pad (MPAD-C2)
- label: pad lifted or scraped 0.41, over-etching at pad 0.37, resist or phototool defect at pad 0.21

## 23. short
- Clues: same spot on 0 of the last 6 boards; not near the edge; round; 2 of this type on the board; steady on recent boards; also on the board: hole breakout.
- generator's cause: conductive debris on panel (SHORT-C3)
- label: conductive debris on panel 0.51, under-etching 0.26, phototool spot or resist residue 0.23

## 24. open
- Clues: same spot on 6 of the last 18 boards; not near the edge; irregular shape; 1 of this type on the board; rising on recent boards; also on the board: missing pad, spurious copper.
- generator's cause: dust or hair on phototool (OPEN-C2)
- label: dust or hair on phototool 0.85, resist scratch or poor adhesion 0.11, over-etching 0.04

## 25. spur
- Clues: same spot on 0 of the last 14 boards; not near the edge; long and thin; 6 of this type on the board; trend unknown.
- generator's cause: under-etching (SPUR-C1)
- label: under-etching 0.64, incomplete development 0.35, phototool spot or scratch 0.01

## 26. missing_hole
- Clues: same spot on 10 of the last 15 boards; not near the edge; round; 1 of this type on the board; trend unknown; also on the board: hole breakout, spurious copper.
- generator's cause: drill program error (MHOLE-C1)
- label: drill program error 0.84, broken drill bit 0.16

## 27. mouse_bite
- Clues: same spot on 1 of the last 20 boards; not near the edge; long and thin; 2 of this type on the board; rising on recent boards.
- generator's cause: poor resist adhesion at edge (MBITE-C2)
- label: poor resist adhesion at edge 0.43, over-etching or undercut 0.41, bubbles in pattern plating 0.12

## 28. extra_part
- Clues: same spot on 0 of the last 11 boards; not near the edge; long and thin; 2 of this type on the board; steady on recent boards; also on the board: pin hole.
- generator's cause: loose part dropped on the board (XPART-C1)
- label: loose part dropped on the board 0.79, debris or a solder ball 0.18, wrong build variant 0.02

## 29. missing_pad
- Clues: same spot on 6 of the last 15 boards; near the edge; round; 2 of this type on the board; trend unknown.
- generator's cause: resist or phototool defect at pad (MPAD-C2)
- label: resist or phototool defect at pad 0.87, over-etching at pad 0.07, pad lifted or scraped 0.05

## 30. spurious_copper
- Clues: same spot on 5 of the last 8 boards; not near the edge; long and thin; 1 of this type on the board; steady on recent boards; also on the board: missing pad.
- generator's cause: resist residue or phototool spot (SPCU-C2)
- label: resist residue or phototool spot 0.92, incomplete etching 0.04, contamination before etching 0.04

## 31. scratch
- Clues: same spot on 0 of the last 14 boards; not near the edge; long and thin; 5 of this type on the board; steady on recent boards.
- generator's cause: scratched resist before etch (SCR-C3)
- label: rollers or conveyor 0.41, panel handling or stacking 0.31, scratched resist before etch 0.27

## 32. extra_part
- Clues: same spot on 9 of the last 9 boards; not near the edge; irregular shape; 4 of this type on the board; steady on recent boards.
- generator's cause: wrong build variant (XPART-C2)
- label: wrong build variant 0.97, debris or a solder ball 0.02, loose part dropped on the board 0.02

## 33. changed_part
- Clues: same spot unknown; not near the edge; long and thin; 5 of this type on the board; trend unknown; also on the board: scratch.
- generator's cause: same part from another supplier (CPART-C3)
- label: same part from another supplier 0.40, wrong part loaded 0.32, part turned or standing up 0.28

## 34. missing_hole
- Clues: same spot on 0 of the last 12 boards; not near the edge; irregular shape; 1 of this type on the board; rising on recent boards; also on the board: foreign object, hole breakout.
- generator's cause: broken drill bit (MHOLE-C2)
- label: broken drill bit 0.96, drill program error 0.04

## 35. short
- Clues: same spot on 8 of the last 8 boards; not near the edge; round; 2 of this type on the board; steady on recent boards; also on the board: spur.
- generator's cause: phototool spot or resist residue (SHORT-C2)
- label: phototool spot or resist residue 0.96, under-etching 0.02, conductive debris on panel 0.02

## 36. scratch
- Clues: same spot on 0 of the last 10 boards; near the edge; long and thin; 6 of this type on the board; steady on recent boards; also on the board: changed part, spur.
- generator's cause: panel handling or stacking (SCR-C1)
- label: panel handling or stacking 0.60, rollers or conveyor 0.24, scratched resist before etch 0.16

## 37. hole_breakout
- Clues: same spot on 0 of the last 10 boards; not near the edge; round; 5 of this type on the board; steady on recent boards; also on the board: spurious copper.
- generator's cause: pattern shifted at exposure (HBRK-C2)
- label: pattern shifted at exposure 0.38, drill-to-pad misregistration 0.32, drill wander 0.30

## 38. spurious_copper
- Clues: same spot on 0 of the last 19 boards; not near the edge; long and thin; 1 of this type on the board; steady on recent boards.
- generator's cause: contamination before etching (SPCU-C3)
- label: contamination before etching 0.59, incomplete etching 0.21, resist residue or phototool spot 0.20

## 39. changed_part
- Clues: same spot on 10 of the last 17 boards; not near the edge; round; 5 of this type on the board; steady on recent boards.
- generator's cause: wrong part loaded (CPART-C1)
- label: same part from another supplier 0.48, wrong part loaded 0.37, part turned or standing up 0.14

## 40. pin_hole
- Clues: same spot on 0 of the last 17 boards; not near the edge; irregular shape; 5 of this type on the board; trend unknown; also on the board: scratch, spurious copper.
- generator's cause: plating voids (PHOLE-C2)
- label: plating voids 0.63, dust or bubbles at lamination 0.37

## 41. spur
- Clues: same spot on 1 of the last 8 boards; not near the edge; irregular shape; 3 of this type on the board; trend unknown; also on the board: missing pad.
- generator's cause: incomplete development (SPUR-C2)
- label: under-etching 0.64, incomplete development 0.35, phototool spot or scratch 0.01

## 42. spurious_copper
- Clues: same spot on 8 of the last 20 boards; not near the edge; round; 2 of this type on the board; rising on recent boards.
- generator's cause: resist residue or phototool spot (SPCU-C2)
- label: resist residue or phototool spot 0.90, incomplete etching 0.08, contamination before etching 0.02

## 43. missing_part
- Clues: same spot on 0 of the last 13 boards; not near the edge; irregular shape; 2 of this type on the board; trend unknown; also on the board: unknown anomaly.
- generator's cause: part was never placed (MPART-C1)
- label: part moved or stood up in reflow 0.48, part knocked off after soldering 0.35, part was never placed 0.17

## 44. unknown_anomaly
- Clues: same spot on 5 of the last 16 boards; not near the edge; long and thin; 2 of this type on the board; steady on recent boards; also on the board: foreign object.
- generator's cause: physical damage (ANOM-C2)
- label: assembly fault 0.81, physical damage 0.13, contamination or foreign object 0.06

## 45. spur
- Clues: same spot on 12 of the last 12 boards; not near the edge; round; 2 of this type on the board; trend unknown.
- generator's cause: phototool spot or scratch (SPUR-C3)
- label: phototool spot or scratch 0.78, incomplete development 0.15, under-etching 0.07

## 46. missing_part
- Clues: same spot on 1 of the last 18 boards; edge unknown; round; 1 of this type on the board; rising on recent boards; also on the board: hole breakout.
- generator's cause: part knocked off after soldering (MPART-C3)
- label: part moved or stood up in reflow 0.57, part was never placed 0.23, part knocked off after soldering 0.19

## 47. hole_breakout
- Clues: same spot on 1 of the last 12 boards; not near the edge; round; 3 of this type on the board; rising on recent boards.
- generator's cause: drill wander (HBRK-C3)
- label: drill wander 0.40, pattern shifted at exposure 0.33, drill-to-pad misregistration 0.28

## 48. foreign_object
- Clues: same spot unknown; not near the edge; long and thin; 5 of this type on the board; steady on recent boards.
- generator's cause: airborne dust or fibre (FOB-C1)
- label: airborne dust or fibre 0.77, process residue 0.14, metal particle from tooling 0.09

## 49. short
- Clues: same spot on 1 of the last 7 boards; not near the edge; long and thin; 6 of this type on the board; rising on recent boards; also on the board: missing pad.
- generator's cause: under-etching (SHORT-C1)
- label: under-etching 0.92, conductive debris on panel 0.05, phototool spot or resist residue 0.02

## 50. pin_hole
- Clues: same spot on 0 of the last 19 boards; near the edge; round; 1 of this type on the board; steady on recent boards; also on the board: mouse bite.
- generator's cause: dust or bubbles at lamination (PHOLE-C1)
- label: dust or bubbles at lamination 0.67, plating voids 0.33


# Sentence sample

- **How serious were the defects this week?** (severity_counts)  
  facts: `{"hours": 168, "total_defects": 30, "critical": 6, "major": 6, "minor": 18}`  
  target: By severity in the past {hours|hour}: {critical} critical, {major} major, {minor} minor.

- **What happened with serial B00017?** (board_history)  
  facts: `{"board": "B00017", "times_inspected": 1, "last_verdict": "FAIL", "last_defects": 4, "last_time": "2026-11-12 14:42:00"}`  
  target: Board {board} was inspected {times_inspected|time}; the last verdict was {last_verdict} with {last_defects|defect}.

- **What was the main problem in the last 24 hours?** (defect_counts)  
  facts: `{"hours": 24, "total_defects": 21, "n_types": 1, "top_defect": "scratch", "top_count": 21, "count_scratch": 21}`  
  target: {top_defect} is the most frequent defect, with {top_count} found.

- **What kept going wrong on the boards this week?** (defect_counts)  
  facts: `{"hours": 168, "total_defects": 108, "n_types": 3, "top_defect": "missing part", "top_count": 40, "second_defect": "short", "second_count": 39, "count_missing_part": 40, "count_short": 39, "count_spurious_copper": 29}`  
  target: The main problem over the last {hours|hour} was {top_defect}, with {top_count} found. Next is {second_defect} with {second_count}.

- **How many boards did we inspect this shift?** (yield)  
  facts: `{"hours": 8, "boards": 3, "passed": 3, "failed": 0, "review": 0, "pass_rate_pct": 100}`  
  target: The station inspected {boards|board}: {passed} passed, {failed} failed and {review} went to a person.

- **Any major defects in the last 2 hours?** (severity_counts)  
  facts: `{"hours": 2, "total_defects": 53, "critical": 0, "major": 10, "minor": 43}`  
  target: There were {major} major defects in the past {hours|hour}, out of {total_defects}.

- **Which defect is the most common this shift?** (defect_counts)  
  facts: `{"hours": 8, "total_defects": 11, "n_types": 1, "top_defect": "foreign object", "top_count": 11, "count_foreign_object": 11}`  
  target: Most of the defects were {top_defect}: {top_count} of {total_defects}.

- **Split the defects from today into critical, major and minor** (severity_counts)  
  facts: `{"hours": 24, "total_defects": 103, "critical": 29, "major": 29, "minor": 45}`  
  target: Of {total_defects|defect} over the last {hours|hour}, {critical} were critical, {major} major and {minor} minor.

- **Show the history of board B87208** (board_history)  
  facts: `{"board": "B87208", "times_inspected": 3, "last_verdict": "FAIL", "last_defects": 8, "last_time": "2026-11-16 8:44:00"}`  
  target: Board {board} was inspected {times_inspected|time}; the last verdict was {last_verdict} with {last_defects|defect}.

- **What was the main problem in the last 8 hours?** (defect_counts)  
  facts: `{"hours": 8, "total_defects": 52, "n_types": 3, "top_defect": "open", "top_count": 23, "second_defect": "spurious copper", "second_count": 15, "count_open": 23, "count_spurious_copper": 15, "count_spur": 14}`  
  target: Most of the defects were {top_defect}: {top_count} of {total_defects}. Next is {second_defect} with {second_count}.

- **Show me the open alerts** (open_alerts)  
  facts: `{"n_alerts": 1, "latest_alert": "boards sent to a person above the control limit for 2 groups of 20", "latest_time": "2026-11-11 11:15:00"}`  
  target: The station has {n_alerts|open alert}. The newest: {latest_alert}

- **How many boards needed checking by an operator in the last 24 hours?** (yield)  
  facts: `{"hours": 24, "boards": 66, "passed": 29, "failed": 17, "review": 20, "pass_rate_pct": 44}`  
  target: Out of {boards|board} inspected, {review} went to a person.

- **How many boards did we inspect this week?** (yield)  
  facts: `{"hours": 168, "boards": 28, "passed": 22, "failed": 1, "review": 5, "pass_rate_pct": 79}`  
  target: The station inspected {boards|board} in the past {hours|hour}.

- **Look up board B13398** (board_history)  
  facts: `{"board": "B13398", "times_inspected": 1, "last_verdict": "CHECK", "last_defects": 3, "last_time": "2026-11-10 18:17:00"}`  
  target: {board} was inspected {times_inspected|time}. Its last result was {last_verdict}, with {last_defects|defect}.

- **Any drift alerts I should know about?** (open_alerts)  
  facts: `{"n_alerts": 2, "latest_alert": "short on 6 of the last 20 boards (3x the usual rate): check the etch line", "latest_time": "2026-11-10 16:15:00"}`  
  target: The station has {n_alerts|open alert}. The newest: {latest_alert}

- **Which defect is the most common this shift?** (defect_counts)  
  facts: `{"hours": 8, "total_defects": 20, "n_types": 2, "top_defect": "short", "top_count": 12, "second_defect": "missing hole", "second_count": 8, "count_short": 12, "count_missing_hole": 8}`  
  target: The main problem over the last {hours|hour} was {top_defect}, with {top_count} found.

- **Was board B00101 inspected more than once?** (board_history)  
  facts: `{"board": "B00101", "times_inspected": 3, "last_verdict": "PASS", "last_defects": 4, "last_time": "2026-11-15 9:35:00"}`  
  target: {board} has been through the station {times_inspected|time}, ending with {last_verdict} and {last_defects|defect}.

- **How many critical defects today?** (severity_counts)  
  facts: `{"hours": 24, "total_defects": 40, "critical": 5, "major": 16, "minor": 19}`  
  target: There were {critical} critical defects in the last {hours|hour}, out of {total_defects}.

- **Show me the open alerts** (open_alerts)  
  facts: `{"n_alerts": 2, "latest_alert": "open repeating at the same spot on 5 boards: inspect the phototool", "latest_time": "2026-11-15 10:15:00"}`  
  target: Open alerts: {n_alerts}. The latest says: {latest_alert}

- **Were most of the problems this shift serious or small?** (severity_counts)  
  facts: `{"hours": 8, "total_defects": 28, "critical": 1, "major": 22, "minor": 5}`  
  target: Of {total_defects|defect} over the last {hours|hour}, {critical} were critical, {major} major and {minor} minor.
