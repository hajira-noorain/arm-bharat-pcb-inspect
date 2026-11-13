#!/usr/bin/env bash
# Colab job: detector v2 = earlier sources + DsPCBSD+ (real production-line AOI defects, CC BY 4.0) with
# camera-condition augmentation, 11 classes. Data comes straight from the original hosts.
# Expects: this repo unpacked (any path); DeepPCB / HRIPCB / Mendeley already under /content/data (job 2)
# or fetched here.
set -x
CODE=$(cd "$(dirname "$0")/.." && pwd)
D=/content/data
mkdir -p /content/out2/results $D && cd $D
[ -d DeepPCB ] || git clone --depth 1 -q https://github.com/tangsanli5201/DeepPCB.git
[ -d HRIPCB ] || git clone --depth 1 -q https://github.com/Ironbrotherstyle/PCB-DATASET.git HRIPCB
if [ ! -d mendeley/PCB_Defect ]; then
  curl -sSL --retry 5 -o PCB_Defect_mendeley.zip "https://data.mendeley.com/public-files/datasets/vdj74sngvn/files/9a62af5d-f3ea-4b3d-a99b-e48115d51c2f/file_downloaded"
  mkdir -p mendeley && unzip -q -o PCB_Defect_mendeley.zip -d mendeley
fi
if [ ! -d dspcbsd/x/Data_YOLO ]; then
  mkdir -p dspcbsd && curl -sSL --retry 5 -o dspcbsd/DsPCBSD+.zip https://ndownloader.figshare.com/files/44069552
  (cd dspcbsd && unzip -q -o "DsPCBSD+.zip" "Data_YOLO/images/*/*" "Data_YOLO/labels/*/*" -d x)
fi
cd $CODE
rm -rf $D/dspcbsd_mosaics
python training/prepare_bare.py --data $D --out $D/bare_yolo_v2
python training/split_fingerprint.py $D/bare_yolo_v2 | tee /content/out2/split_fingerprint.txt
python training/camera_aug.py build --src $D/bare_yolo_v2 --dst $D/bare_yolo_v2_cam --frac 0.6
pip -q install ultralytics
python training/train_bare.py --data $D/bare_yolo_v2_cam/data.yaml --model yolo26n.pt --name y26n_v2 \
  --epochs 150 --batch 64 --project /content/out2/runs
cp $D/bare_yolo_v2/boards_*.json /content/out2/
echo V2_DONE
