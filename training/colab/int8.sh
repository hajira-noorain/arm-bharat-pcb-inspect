#!/usr/bin/env bash
# INT8 detector export (the first attempt inherited the notebook's matplotlib backend). Separate work dir.
set -x
export MPLBACKEND=Agg
E=/content/out6b; mkdir -p $E/data && ln -sfn /content/data/bare_yolo_v3 $E/data/bare_yolo
cd /content/pcb13
/content/etv/bin/python training/export_vision.py --work $E --yolo /content/out2/runs/y26n_v3/weights/best.pt --only detector
cat $E/results/export_vision.json
ls -la $E/models/vision
echo INT8_DONE
