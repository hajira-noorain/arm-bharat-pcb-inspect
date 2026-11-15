# Training data sample (for reading by hand)

Each cause example: the clues the model sees for the defect, the cause the generator used, and the label (probability over causes). Mark anything a process engineer would disagree with.

## 1. mouse_bite
- Clues: same spot on 9 of the last 15 boards; near the edge; long and thin; 5 of this type on the board; trend unknown.
- generator's cause: over-etching or undercut (MBITE-C1)
- label: poor resist adhesion at edge 0.61, over-etching or undercut 0.39

## 2. scratch
- Clues: same spot on 1 of the last 6 boards; not near the edge; irregular shape; 2 of this type on the board; trend unknown; also on the board: foreign object, missing hole.
- generator's cause: panel handling or stacking (SCR-C1)
- label: scratched resist before etch 0.43, panel handling or stacking 0.43, rollers or conveyor 0.13

## 3. open
- Clues: same spot unknown; not near the edge; irregular shape; 2 of this type on the board; steady on recent boards.
- generator's cause: dust or hair on phototool (OPEN-C2)
- label: resist scratch or poor adhesion 0.37, dust or hair on phototool 0.33, handling scratch 0.23

## 4. pin_hole
- Clues: same spot on 2 of the last 6 boards; not near the edge; round; 1 of this type on the board; steady on recent boards; also on the board: changed part, unknown anomaly.
- generator's cause: dust or bubbles at lamination (PHOLE-C1)
- label: dust or bubbles at lamination 0.86, plating voids 0.14

## 5. mouse_bite
- Clues: same spot on 5 of the last 17 boards; not near the edge; irregular shape; 2 of this type on the board; trend unknown.
- generator's cause: poor resist adhesion at edge (MBITE-C2)
- label: poor resist adhesion at edge 0.82, over-etching or undercut 0.18

## 6. changed_part
- Clues: same spot on 0 of the last 8 boards; not near the edge; round; 7 of this type on the board; steady on recent boards; also on the board: hole breakout, pin hole.
- generator's cause: same part from another supplier (CPART-C3)
- label: part turned or standing up 0.57, same part from another supplier 0.32, wrong part loaded 0.11

## 7. missing_hole
- Clues: same spot on 1 of the last 18 boards; near the edge; round; 2 of this type on the board; steady on recent boards; also on the board: spur.
- generator's cause: broken drill bit (MHOLE-C2)
- label: broken drill bit 0.81, drill program error 0.19

## 8. unknown_anomaly
- Clues: same spot unknown; edge unknown; round; 2 of this type on the board; steady on recent boards; also on the board: hole breakout.
- generator's cause: assembly fault (ANOM-C1)
- label: assembly fault 0.46, contamination or foreign object 0.36, physical damage 0.18

## 9. missing_part
- Clues: same spot on 9 of the last 14 boards; not near the edge; round; 5 of this type on the board; rising on recent boards; also on the board: foreign object.
- generator's cause: part was never placed (MPART-C1)
- label: part was never placed 0.84, part moved or stood up in reflow 0.16, part knocked off after soldering 0.00

## 10. missing_pad
- Clues: same spot on 0 of the last 19 boards; near the edge; round; 4 of this type on the board; rising on recent boards; also on the board: foreign object, mouse bite.
- generator's cause: over-etching at pad (MPAD-C1)
- label: over-etching at pad 0.94, resist or phototool defect at pad 0.03, pad lifted or scraped 0.02

## 11. short
- Clues: same spot unknown; near the edge; long and thin; 7 of this type on the board; steady on recent boards.
- generator's cause: conductive debris on panel (SHORT-C3)
- label: under-etching 0.44, conductive debris on panel 0.41, phototool spot or resist residue 0.15

## 12. scratch
- Clues: same spot unknown; not near the edge; long and thin; 1 of this type on the board; steady on recent boards; also on the board: missing hole, missing part.
- generator's cause: panel handling or stacking (SCR-C1)
- label: panel handling or stacking 0.49, scratched resist before etch 0.27, rollers or conveyor 0.24

## 13. missing_pad
- Clues: same spot on 1 of the last 15 boards; near the edge; round; 2 of this type on the board; trend unknown; also on the board: foreign object, unknown anomaly.
- generator's cause: over-etching at pad (MPAD-C1)
- label: pad lifted or scraped 0.53, over-etching at pad 0.35, resist or phototool defect at pad 0.12

## 14. extra_part
- Clues: same spot on 8 of the last 8 boards; near the edge; irregular shape; 1 of this type on the board; trend unknown.
- generator's cause: loose part dropped on the board (XPART-C1)
- label: wrong build variant 0.65, loose part dropped on the board 0.29, debris or a solder ball 0.06

## 15. spur
- Clues: same spot on 6 of the last 6 boards; not near the edge; round; 1 of this type on the board; steady on recent boards.
- generator's cause: incomplete development (SPUR-C2)
- label: incomplete development 0.85, under-etching 0.15

## 16. missing_hole
- Clues: same spot on 1 of the last 11 boards; not near the edge; round; 5 of this type on the board; rising on recent boards; also on the board: extra part.
- generator's cause: broken drill bit (MHOLE-C2)
- label: broken drill bit 0.98, drill program error 0.02

## 17. scratch
- Clues: same spot on 0 of the last 16 boards; not near the edge; long and thin; 4 of this type on the board; rising on recent boards.
- generator's cause: panel handling or stacking (SCR-C1)
- label: rollers or conveyor 0.58, scratched resist before etch 0.25, panel handling or stacking 0.17

## 18. hole_breakout
- Clues: same spot on 0 of the last 13 boards; not near the edge; irregular shape; 8 of this type on the board; trend unknown; also on the board: missing pad, mouse bite.
- generator's cause: drill wander (HBRK-C3)
- label: pattern shifted at exposure 0.36, drill wander 0.33, drill-to-pad misregistration 0.31

## 19. missing_part
- Clues: same spot on 4 of the last 8 boards; not near the edge; irregular shape; 4 of this type on the board; trend unknown.
- generator's cause: part was never placed (MPART-C1)
- label: part was never placed 0.82, part moved or stood up in reflow 0.17, part knocked off after soldering 0.00

## 20. changed_part
- Clues: same spot on 0 of the last 13 boards; not near the edge; round; 4 of this type on the board; trend unknown; also on the board: extra part.
- generator's cause: part turned or standing up (CPART-C2)
- label: part turned or standing up 0.60, same part from another supplier 0.29, wrong part loaded 0.11

## 21. changed_part
- Clues: same spot on 16 of the last 20 boards; near the edge; long and thin; 5 of this type on the board; trend unknown.
- generator's cause: wrong part loaded (CPART-C1)
- label: same part from another supplier 0.44, wrong part loaded 0.40, part turned or standing up 0.16

## 22. missing_hole
- Clues: same spot on 0 of the last 8 boards; not near the edge; round; 3 of this type on the board; trend unknown; also on the board: scratch.
- generator's cause: broken drill bit (MHOLE-C2)
- label: broken drill bit 0.95, drill program error 0.05

## 23. open
- Clues: same spot unknown; not near the edge; irregular shape; 2 of this type on the board; rising on recent boards.
- generator's cause: dust or hair on phototool (OPEN-C2)
- label: resist scratch or poor adhesion 0.45, dust or hair on phototool 0.33, over-etching 0.16

## 24. unknown_anomaly
- Clues: same spot on 1 of the last 12 boards; not near the edge; irregular shape; 2 of this type on the board; trend unknown.
- generator's cause: assembly fault (ANOM-C1)
- label: assembly fault 0.44, physical damage 0.34, contamination or foreign object 0.22

## 25. foreign_object
- Clues: same spot on 1 of the last 16 boards; not near the edge; long and thin; 1 of this type on the board; trend unknown; also on the board: pin hole.
- generator's cause: airborne dust or fibre (FOB-C1)
- label: airborne dust or fibre 0.78, metal particle from tooling 0.15, process residue 0.07

## 26. foreign_object
- Clues: same spot unknown; not near the edge; round; 2 of this type on the board; rising on recent boards.
- generator's cause: process residue (FOB-C2)
- label: process residue 0.52, airborne dust or fibre 0.26, metal particle from tooling 0.21

## 27. pin_hole
- Clues: same spot on 1 of the last 20 boards; not near the edge; irregular shape; 2 of this type on the board; steady on recent boards.
- generator's cause: dust or bubbles at lamination (PHOLE-C1)
- label: dust or bubbles at lamination 0.71, plating voids 0.29

## 28. scratch
- Clues: same spot unknown; edge unknown; irregular shape; 4 of this type on the board; steady on recent boards; also on the board: changed part.
- generator's cause: scratched resist before etch (SCR-C3)
- label: scratched resist before etch 0.36, panel handling or stacking 0.33, rollers or conveyor 0.31

## 29. missing_hole
- Clues: same spot on 0 of the last 14 boards; not near the edge; irregular shape; 5 of this type on the board; rising on recent boards; also on the board: changed part.
- generator's cause: broken drill bit (MHOLE-C2)
- label: broken drill bit 0.98, drill program error 0.02

## 30. mouse_bite
- Clues: same spot on 5 of the last 6 boards; not near the edge; irregular shape; 5 of this type on the board; trend unknown; also on the board: scratch.
- generator's cause: over-etching or undercut (MBITE-C1)
- label: poor resist adhesion at edge 0.61, over-etching or undercut 0.39

## 31. spur
- Clues: same spot on 5 of the last 6 boards; near the edge; irregular shape; 3 of this type on the board; rising on recent boards.
- generator's cause: incomplete development (SPUR-C2)
- label: incomplete development 0.71, under-etching 0.29

## 32. open
- Clues: same spot on 0 of the last 14 boards; near the edge; long and thin; 2 of this type on the board; steady on recent boards; also on the board: extra part.
- generator's cause: handling scratch (OPEN-C4)
- label: handling scratch 0.86, resist scratch or poor adhesion 0.10, dust or hair on phototool 0.03

## 33. spurious_copper
- Clues: same spot on 8 of the last 10 boards; not near the edge; round; 2 of this type on the board; rising on recent boards; also on the board: missing pad.
- generator's cause: resist residue or phototool spot (SPCU-C2)
- label: resist residue or phototool spot 0.90, incomplete etching 0.08, contamination before etching 0.02

## 34. extra_part
- Clues: same spot unknown; near the edge; irregular shape; 1 of this type on the board; trend unknown.
- generator's cause: loose part dropped on the board (XPART-C1)
- label: loose part dropped on the board 0.71, debris or a solder ball 0.14, wrong build variant 0.14

## 35. missing_hole
- Clues: same spot on 1 of the last 9 boards; not near the edge; round; 6 of this type on the board; rising on recent boards.
- generator's cause: broken drill bit (MHOLE-C2)
- label: broken drill bit 0.98, drill program error 0.02

## 36. changed_part
- Clues: same spot on 2 of the last 8 boards; edge unknown; irregular shape; 2 of this type on the board; steady on recent boards; also on the board: hole breakout.
- generator's cause: part turned or standing up (CPART-C2)
- label: wrong part loaded 0.57, part turned or standing up 0.22, same part from another supplier 0.21

## 37. scratch
- Clues: same spot on 1 of the last 20 boards; near the edge; long and thin; 1 of this type on the board; rising on recent boards.
- generator's cause: panel handling or stacking (SCR-C1)
- label: panel handling or stacking 0.65, rollers or conveyor 0.18, scratched resist before etch 0.17

## 38. mouse_bite
- Clues: same spot on 1 of the last 14 boards; edge unknown; round; 6 of this type on the board; trend unknown; also on the board: changed part, extra part.
- generator's cause: over-etching or undercut (MBITE-C1)
- label: over-etching or undercut 0.71, poor resist adhesion at edge 0.29

## 39. mouse_bite
- Clues: same spot on 0 of the last 20 boards; not near the edge; irregular shape; 2 of this type on the board; rising on recent boards; also on the board: changed part, scratch.
- generator's cause: poor resist adhesion at edge (MBITE-C2)
- label: over-etching or undercut 0.52, poor resist adhesion at edge 0.48

## 40. open
- Clues: same spot on 8 of the last 13 boards; not near the edge; long and thin; 7 of this type on the board; steady on recent boards.
- generator's cause: dust or hair on phototool (OPEN-C2)
- label: dust or hair on phototool 0.90, resist scratch or poor adhesion 0.07, over-etching 0.02

## 41. mouse_bite
- Clues: same spot on 1 of the last 9 boards; not near the edge; irregular shape; 2 of this type on the board; trend unknown; also on the board: pin hole.
- generator's cause: poor resist adhesion at edge (MBITE-C2)
- label: poor resist adhesion at edge 0.54, over-etching or undercut 0.46

## 42. changed_part
- Clues: same spot on 5 of the last 6 boards; not near the edge; long and thin; 1 of this type on the board; trend unknown.
- generator's cause: wrong part loaded (CPART-C1)
- label: wrong part loaded 0.59, part turned or standing up 0.23, same part from another supplier 0.19

## 43. spur
- Clues: same spot on 1 of the last 14 boards; not near the edge; round; 7 of this type on the board; steady on recent boards; also on the board: changed part, mouse bite.
- generator's cause: under-etching (SPUR-C1)
- label: under-etching 0.62, incomplete development 0.38

## 44. short
- Clues: same spot on 9 of the last 9 boards; not near the edge; round; 3 of this type on the board; steady on recent boards.
- generator's cause: phototool spot or resist residue (SHORT-C2)
- label: phototool spot or resist residue 0.82, under-etching 0.17, conductive debris on panel 0.01

## 45. changed_part
- Clues: same spot on 9 of the last 18 boards; not near the edge; irregular shape; 7 of this type on the board; trend unknown; also on the board: pin hole, scratch.
- generator's cause: same part from another supplier (CPART-C3)
- label: same part from another supplier 0.44, wrong part loaded 0.40, part turned or standing up 0.16

## 46. missing_pad
- Clues: same spot on 12 of the last 14 boards; not near the edge; round; 2 of this type on the board; trend unknown; also on the board: pin hole, spur.
- generator's cause: resist or phototool defect at pad (MPAD-C2)
- label: resist or phototool defect at pad 0.93, over-etching at pad 0.05, pad lifted or scraped 0.01

## 47. changed_part
- Clues: same spot on 10 of the last 13 boards; not near the edge; irregular shape; 7 of this type on the board; trend unknown; also on the board: hole breakout, spurious copper.
- generator's cause: wrong part loaded (CPART-C1)
- label: same part from another supplier 0.44, wrong part loaded 0.40, part turned or standing up 0.16

## 48. foreign_object
- Clues: same spot on 0 of the last 18 boards; not near the edge; long and thin; 2 of this type on the board; trend unknown.
- generator's cause: airborne dust or fibre (FOB-C1)
- label: airborne dust or fibre 0.78, metal particle from tooling 0.15, process residue 0.07

## 49. missing_pad
- Clues: same spot on 8 of the last 14 boards; not near the edge; irregular shape; 2 of this type on the board; steady on recent boards; also on the board: missing hole.
- generator's cause: resist or phototool defect at pad (MPAD-C2)
- label: resist or phototool defect at pad 0.92, over-etching at pad 0.04, pad lifted or scraped 0.04

## 50. scratch
- Clues: same spot on 1 of the last 9 boards; not near the edge; long and thin; 5 of this type on the board; trend unknown.
- generator's cause: rollers or conveyor (SCR-C2)
- label: rollers or conveyor 0.47, scratched resist before etch 0.27, panel handling or stacking 0.27


# Sentence sample

- **Were there more major or minor defects in the last 4 hours?** (severity_counts)  
  facts: `{"hours": 4, "total_defects": 75, "critical": 46, "major": 0, "minor": 29}`  
  target: In the last {hours|hour}: {major} major and {minor} minor, out of {total_defects|defect}.

- **How severe were the defects over the past 6 hours?** (severity_counts)  
  facts: `{"hours": 6, "total_defects": 64, "critical": 41, "major": 0, "minor": 23}`  
  target: {critical} critical, {major} major and {minor} minor defects were found in the last {hours|hour}.

- **How severe were the defects over the past 6 hours?** (severity_counts)  
  facts: `{"hours": 6, "total_defects": 29, "critical": 16, "major": 13, "minor": 0}`  
  target: {critical} critical, {major} major and {minor} minor defects were found in the last {hours|hour}.

- **Any drift alerts I should know about?** (open_alerts)  
  facts: `{"n_alerts": 2, "latest_alert": "open repeating at the same spot on 5 boards: inspect the phototool", "latest_time": "2026-11-11 17:15:00"}`  
  target: There are {n_alerts|open alert}; the latest says: {latest_alert}

- **What is the pass rate this shift?** (yield)  
  facts: `{"hours": 8, "boards": 8, "passed": 8, "failed": 0, "review": 0}`  
  target: Out of {boards|board}, {passed} passed.

- **Failed versus passed boards today?** (yield)  
  facts: `{"hours": 24, "boards": 26, "passed": 20, "failed": 6, "review": 0}`  
  target: Of {boards|board}, {passed} passed, {failed} failed and {review} went to a person.

- **Split the defects today into critical, major and minor** (severity_counts)  
  facts: `{"hours": 24, "total_defects": 20, "critical": 0, "major": 20, "minor": 0}`  
  target: {critical} critical, {major} major and {minor} minor defects were found in the last {hours|hour}.

- **What happened with serial B00017?** (board_history)  
  facts: `{"board": "B00017", "times_inspected": 1, "last_verdict": "CHECK", "last_defects": 1, "last_time": "2026-11-13 15:22:00"}`  
  target: {board} has been through the station {times_inspected|time}, ending with {last_verdict} and {last_defects|defect}.

- **Any drift alerts I should know about?** (open_alerts)  
  facts: `{"n_alerts": 1, "latest_alert": "boards sent to a person above the control limit for 2 groups of 20", "latest_time": "2026-11-12 18:15:00"}`  
  target: There are {n_alerts|open alert}; the latest says: {latest_alert}

- **Show the drift alerts** (open_alerts)  
  facts: `{"n_alerts": 2, "latest_alert": "open repeating at the same spot on 5 boards: inspect the phototool", "latest_time": "2026-11-17 14:15:00"}`  
  target: There are {n_alerts|open alert}; the latest says: {latest_alert}

- **Were there more major or minor defects in the last 12 hours?** (severity_counts)  
  facts: `{"hours": 12, "total_defects": 26, "critical": 26, "major": 0, "minor": 0}`  
  target: In the last {hours|hour}: {major} major and {minor} minor, out of {total_defects|defect}.

- **How many times was board B73861 inspected?** (board_history)  
  facts: `{"board": "B73861", "times_inspected": 1, "last_verdict": "PASS", "last_defects": 7, "last_time": "2026-11-14 11:39:00"}`  
  target: Board {board} was inspected {times_inspected|time}; the last verdict was {last_verdict} with {last_defects|defect}.

- **List the defects found in the last 8 hours** (defect_counts)  
  facts: `{"hours": 8, "total_defects": 94, "n_types": 4, "top_defect": "changed part", "top_count": 38, "second_defect": "mouse bite", "second_count": 25, "count_changed_part": 38, "count_mouse_bite": 25, "count_spur": 19, "count`  
  target: {total_defects|defect} of {n_types|type} were found in the last {hours|hour}; the most common is {top_defect} ({top_count}).

- **Were most of the problems this shift serious or small?** (severity_counts)  
  facts: `{"hours": 8, "total_defects": 98, "critical": 48, "major": 25, "minor": 25}`  
  target: {critical} critical, {major} major and {minor} minor defects were found in the last {hours|hour}.

- **Any drift alerts I should know about?** (open_alerts)  
  facts: `{"n_alerts": 2, "latest_alert": "short on 6 of the last 20 boards (3x the usual rate): check the etch line", "latest_time": "2026-11-20 11:15:00"}`  
  target: Open alerts: {n_alerts}. The latest says: {latest_alert}

- **What happened with serial B00017?** (board_history)  
  facts: `{"board": "B00017", "times_inspected": 2, "last_verdict": "PASS", "last_defects": 3, "last_time": "2026-11-13 12:29:00"}`  
  target: {board} has been through the station {times_inspected|time}, ending with {last_verdict} and {last_defects|defect}.

- **Show me the open alerts** (open_alerts)  
  facts: `{"n_alerts": 1, "latest_alert": "short on 6 of the last 20 boards (3x the usual rate): check the etch line", "latest_time": "2026-11-16 9:15:00"}`  
  target: There are {n_alerts|open alert}; the latest says: {latest_alert}

- **Which defect is the most common today?** (defect_counts)  
  facts: `{"hours": 24, "total_defects": 57, "n_types": 3, "top_defect": "spur", "top_count": 24, "second_defect": "extra part", "second_count": 23, "count_spur": 24, "count_extra_part": 23, "count_foreign_object": 10}`  
  target: {top_defect} is the most frequent defect, with {top_count} found. Next is {second_defect} with {second_count}.

- **How many changed parts this week?** (defect_counts)  
  facts: `{"hours": 168, "total_defects": 89, "n_types": 5, "top_defect": "changed part", "top_count": 30, "second_defect": "short", "second_count": 27, "count_changed_part": 30, "count_short": 27, "count_spurious_copper": 17, "co`  
  target: Found in the last {hours|hour}: {count_changed_part|changed part}.

- **How severe were the defects over the past 6 hours?** (severity_counts)  
  facts: `{"hours": 6, "total_defects": 60, "critical": 9, "major": 20, "minor": 31}`  
  target: {critical} critical, {major} major and {minor} minor defects were found in the last {hours|hour}.
