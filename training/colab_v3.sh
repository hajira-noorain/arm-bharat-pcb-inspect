#!/usr/bin/env bash
# Colab job: detector v3 = v2 data + the TRAIN half of DefectDet (real manufactured boards, split by physical
# board) + solder-mask recolouring of binarised scans, fine-tuned from v2. Then every detector is scored on
# the same held-out boards, including the DefectDet boards NO model trained on, and a scale sweep on them.
set -x
CODE=$(cd "$(dirname "$0")/.." && pwd)
D=/content/data
OUT=/content/out3
mkdir -p $OUT/results
until grep -q V2_DONE /content/run_v2.log; do sleep 60; done
V2=/content/out2/runs/y26n_v2/weights/best.pt
cd $CODE
rm -rf $D/dspcbsd_mosaics
python training/prepare_bare.py --data $D --out $D/bare_yolo_v3 --defectdet /content/ext/defectdet
python training/split_fingerprint.py $D/bare_yolo_v3 | tee $OUT/split_fingerprint_v3.txt
python training/camera_aug.py build --src $D/bare_yolo_v3 --dst $D/bare_yolo_v3_aug --frac 0.6 --recolor-frac 0.5
python -c "
from ultralytics import YOLO
YOLO('$V2').train(data='$D/bare_yolo_v3_aug/data.yaml', epochs=70, imgsz=640, batch=64, optimizer='SGD',
    lr0=0.004, lrf=0.05, momentum=0.937, cos_lr=True, warmup_epochs=1, close_mosaic=15, fliplr=0.5, flipud=0.5,
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, mosaic=1.0, scale=0.5, patience=30, project='/content/out2/runs',
    name='y26n_v3', exist_ok=True, workers=8, seed=0)"
V3=/content/out2/runs/y26n_v3/weights/best.pt
cp $D/bare_yolo_v3/boards_*.json $OUT/
for pair in "base:/content/base_best.pt" "cam:/content/out/runs/y26n_cam/weights/best.pt" "v2:$V2" "v3:$V3"; do
  tag=${pair%%:*}; w=${pair#*:}
  python training/eval_bare.py --work $OUT --data $D/bare_yolo_v3 --weights $w --tag $tag --no-gate
done
python training/eval_external.py --coco-root /content/ext/defectdet --codes defectdet \
  --only-boards $D/bare_yolo_v3/boards_test.json --only-source defectdet \
  --weights /content/out/runs/y26n_cam/weights/best.pt $V2 $V3 --names cam v2 v3 --gate-lam 0.25 0.25 0.25 \
  --scales 1,1.5,2,3 --out $OUT/results/external_defectdet_heldout.json
echo V3_DONE
