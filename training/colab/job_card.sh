#!/usr/bin/env bash
# Detection-limit card, simulated photos, for the two retrained detectors (each with its own thresholds).
set -x
mkdir -p /content/card && mv /content/sheet_*.png /content/manifest.json /content/card/ 2>/dev/null; ls /content/card | wc -l
export PCB_UNSUPPORTED_SCALE=0.7
for M in s m; do
  until grep -q "EXPORT_${M}_DONE" /content/run_export.log 2>/dev/null; do sleep 30; done
  G=/content/ex5$M/models/gate
  [ -f $G/deeppcb.json ] && [ -f $G/deeppcb_golden.json ] || { echo "NO_GATES_$M"; continue; }
  cd /content/code
  python training/eyetest_sim.py --card /content/card --weights /content/w/pcb_y26${M}_v5.pt --gate $G/deeppcb.json \
     --golden-gate $G/deeppcb_golden.json --out /content/res/eyetest_sim_v5$M.json 2>&1 | grep -E "^==|Finds|NOT|rror|Traceback" | cut -c1-250
  echo CARD_${M}_DONE
done
echo CARD_ALL_DONE
