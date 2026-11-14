#!/usr/bin/env bash
# Assembled-board anomaly sweep on VisA pcb1-4: resolution x (per-position normalisation, foreground mask, top-k).
set -x
rm -rf /content/pcb7 && mkdir -p /content/pcb7 && unzip -q -o /content/pcb_code7.zip -d /content/pcb7
cd /content/pcb7
O=/content/out4; mkdir -p $O/results
run() { python training/build_visa_banks.py --visa /content/VisA --out $O --device cuda --arch resnet18 --fewshot 10 --fewshot-reps 3 "$@"; }
run --input 320 --tag r18_320_pn --posnorm
run --input 320 --tag r18_320_pn_fg_top --posnorm --fg-mask --topk-frac 0.01
run --input 384 --tag r18_384 
run --input 384 --tag r18_384_pn_fg_top --posnorm --fg-mask --topk-frac 0.01
run --input 320 --tag r18_320_top --topk-frac 0.01
echo ANOM_DONE
