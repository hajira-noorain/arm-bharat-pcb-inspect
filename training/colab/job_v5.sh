#!/usr/bin/env bash
# Detector v5: the same data and the same held-out boards as v3, larger models (YOLO26s, YOLO26m).
# 1. fetch the public sources again (this runtime is new), rebuild the v3 dataset, PROVE the cal/test split is
#    the one every earlier number was measured on (fingerprint), 2. re-score the deployed v3 weights as a check,
# 3. train s then m from the public pretrained weights, 4. score each on the same test boards.
set -x
D=/content/data; OUT=/content/out5; CODE=/content/code11
mkdir -p $OUT/results $D /content/ext
rm -rf $CODE && mkdir -p $CODE && unzip -q -o /content/code11.zip -d $CODE
cd $D
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
[ -d /content/ext/defectdet ] || python /content/fetch_ext.py defectdet
tail -n 3 /content/ext/fetch.log
echo FETCH5_DONE
# heavy CPU work waits for the Llama held-out advice run, so its timings stay clean
until grep -q HELD_ASK_DONE /content/run_held10.log; do sleep 20; done
cd $CODE
python training/prepare_bare.py --data $D --out $D/bare_yolo_v3 --defectdet /content/ext/defectdet 2>&1 | tail -n 15
python training/split_fingerprint.py $D/bare_yolo_v3 | tee $OUT/split_fingerprint_v5.txt
if ! grep -q "cal 744 356da8f161db0786" $OUT/split_fingerprint_v5.txt || ! grep -q "test 1006 3f6578122e0e601a" $OUT/split_fingerprint_v5.txt; then
  echo "SPLIT_MISMATCH: the held-out boards are not the ones v3 was measured on. Stopping."; echo V5_ABORTED; exit 1
fi
echo SPLIT_OK
python training/camera_aug.py build --src $D/bare_yolo_v3 --dst $D/bare_yolo_v3_aug --frac 0.6 --recolor-frac 0.5 2>&1 | tail -n 3
cp $D/bare_yolo_v3/boards_*.json $OUT/
python training/eval_bare.py --work $OUT --data $D/bare_yolo_v3 --weights /content/pcb_y26n.pt --tag v3chk --no-gate 2>&1 | tail -n 4
echo V3CHK_DONE
for M in s m; do
  python -c "
from ultralytics import YOLO
YOLO('yolo26$M.pt').train(data='$D/bare_yolo_v3_aug/data.yaml', epochs=150, imgsz=640, batch=64, patience=30,
    project='$OUT/runs', name='y26${M}_v5', exist_ok=True, seed=0, deterministic=False, workers=8, cos_lr=True,
    fliplr=0.5, flipud=0.5, hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, mosaic=1.0, close_mosaic=15, scale=0.4, plots=False)" \
    2>&1 | grep -E "^\s+(all|[0-9]+/150)\s|EarlyStopping|Error|error|Traceback|epochs completed" | awk 'NR%6==0 || /Early|rror|Trace|completed/'
  python training/eval_bare.py --work $OUT --data $D/bare_yolo_v3 --weights $OUT/runs/y26${M}_v5/weights/best.pt --tag v5$M --no-gate 2>&1 | tail -n 4
  echo V5_${M}_DONE
done
echo V5_ALL_DONE
