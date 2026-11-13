#!/usr/bin/env bash
# Colab job 2: rebuild the bare-board dataset from its original sources, verify the split matches the
# laptop's, train a camera-robust YOLO26n, and run the robustness sweep on base vs camera-robust.
# Expects: /content/pcb (code), /content/base_best.pt (laptop-trained detector).
set -x
CODE=$(cd "$(dirname "$0")/.." && pwd)
D=/content/data
mkdir -p /content/out/results
mkdir -p $D && cd $D
[ -d DeepPCB ] || git clone --depth 1 -q https://github.com/tangsanli5201/DeepPCB.git
[ -d HRIPCB ] || git clone --depth 1 -q https://github.com/Ironbrotherstyle/PCB-DATASET.git HRIPCB
if [ ! -d mendeley/PCB_Defect ]; then
  curl -sSL --retry 5 -o PCB_Defect_mendeley.zip "https://data.mendeley.com/public-files/datasets/vdj74sngvn/files/9a62af5d-f3ea-4b3d-a99b-e48115d51c2f/file_downloaded"
  mkdir -p mendeley && unzip -q -o PCB_Defect_mendeley.zip -d mendeley
fi
cd $CODE
python training/prepare_bare.py --data $D --out $D/bare_yolo
python training/split_fingerprint.py $D/bare_yolo | tee /content/out/split_fingerprint.txt
python training/camera_aug.py build --src $D/bare_yolo --dst $D/bare_yolo_cam --frac 0.6
pip -q install ultralytics
python training/train_bare.py --data $D/bare_yolo_cam/data.yaml --model yolo26n.pt --name y26n_cam \
  --epochs 120 --batch 64 --project /content/out/runs
python training/robustness.py --data $D/bare_yolo --weights /content/base_best.pt /content/out/runs/y26n_cam/weights/best.pt \
  --names base cam --out /content/out/results/robustness.json
echo BARE_DONE
