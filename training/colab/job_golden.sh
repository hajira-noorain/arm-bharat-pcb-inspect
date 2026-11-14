#!/usr/bin/env bash
# Golden-mode thresholds again, now with the good boards in, for all three detectors.
set -x
export PCB_UNSUPPORTED_SCALE=0.7
cp /content/eval_golden.py /content/code/training/eval_golden.py
mkdir -p /content/g3/results && mv /content/bare_dets_cal_v3.json /content/bare_dets_test_v3.json /content/g3/results/
cd /content/code
for pair in "/content/g3:v3" "/content/ex5s:v5s_et" "/content/ex5m:v5m_et"; do
  W=${pair%%:*}; T=${pair#*:}
  python training/eval_golden.py --work $W --tag $T --design deeppcb 2>&1 | grep -E "^cal|^test|fused|golden_diff|^detector|rror|Traceback" | cut -c1-220
done
rm -rf /content/gout && mkdir -p /content/gout/n /content/gout/s /content/gout/m
cp /content/g3/models/gate/deeppcb_golden.json /content/gout/n/; cp /content/g3/results/golden_eval_deeppcb_v3.json /content/gout/
cp /content/ex5s/models/gate/deeppcb_golden.json /content/gout/s/; cp /content/ex5s/results/golden_eval_deeppcb_v5s_et.json /content/gout/
cp /content/ex5m/models/gate/deeppcb_golden.json /content/gout/m/; cp /content/ex5m/results/golden_eval_deeppcb_v5m_et.json /content/gout/
cd /content && rm -f gout.zip && zip -qr gout.zip gout && ls -l gout.zip
echo GOLDEN_DONE
