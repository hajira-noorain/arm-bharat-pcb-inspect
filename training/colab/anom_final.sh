#!/usr/bin/env bash
# Deployable VisA memories + gates with the chosen setting (ResNet-18, 320 px, top-1% image score).
set -x
rm -rf /content/pcb10 && mkdir -p /content/pcb10 && unzip -q -o /content/pcb_code10.zip -d /content/pcb10
cd /content/pcb10
python training/build_visa_banks.py --visa /content/VisA --out /content/out5 --device cuda --arch resnet18 --fewshot-reps 3 --save
cd /content && zip -qr /content/out5.zip out5 && ls -l /content/out5.zip
echo FINAL_DONE
