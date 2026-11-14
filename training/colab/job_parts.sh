#!/usr/bin/env bash
# Parts check for assembled boards: FPIC -> one-class "part" detector + part-kind classifier, scored on boards
# held out by physical board. Runs beside the detector retraining (smaller batch, fewer loader workers).
set -x
until grep -q "FPIC_DONE" /content/run_fpic.log; do sleep 20; done
grep -E "UNZIP_FAILED" /content/run_fpic.log && { echo PARTS_ABORTED; exit 1; }
CODE=/content/code12; OUT=/content/outp; D=/content/data/parts
rm -rf $CODE && mkdir -p $CODE $OUT/results && unzip -q -o /content/code12.zip -d $CODE
cd $CODE
python training/prepare_components.py --fpic /content/ext/fpic --out $D 2>&1 | tail -n 8
cp $D/manifest.json $OUT/results/parts_manifest.json
echo PARTS_PREP_DONE
python - <<PY 2>&1 | grep -E "^\s+all\s|EarlyStopping|rror|Traceback|completed|RESULT" | awk 'NR%8==0 || /Early|rror|Trace|completed|RESULT/'
import json
from ultralytics import YOLO
m = YOLO('yolo26n.pt')
m.train(data='$D/detect/data.yaml', epochs=100, imgsz=640, batch=32, patience=25, project='$OUT/runs', name='parts_det',
        exist_ok=True, seed=0, deterministic=False, workers=4, cos_lr=True, fliplr=0.5, flipud=0.5,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, mosaic=1.0, close_mosaic=10, scale=0.5, degrees=10, plots=False, single_cls=True)
best = YOLO('$OUT/runs/parts_det/weights/best.pt')
r = best.val(data='$D/detect/data.yaml', split='test', imgsz=640, batch=32, workers=4, plots=False, conf=0.001)
p, rc = float(r.box.mp), float(r.box.mr)
res = {"split": "test (boards never trained on)", "precision": p, "recall": rc, "f1": 2 * p * rc / max(p + rc, 1e-9),
       "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map)}
json.dump(res, open('$OUT/results/parts_detector_test.json', 'w'), indent=1)
print('RESULT det', res)
PY
echo PARTS_DET_DONE
python - <<PY 2>&1 | grep -E "EarlyStopping|rror|Traceback|completed|RESULT"
import json
from ultralytics import YOLO
m = YOLO('yolo26n-cls.pt')
m.train(data='$D/classify', epochs=40, imgsz=64, batch=256, patience=10, project='$OUT/runs', name='parts_cls',
        exist_ok=True, seed=0, workers=4, plots=False, fliplr=0.5, flipud=0.5)
best = YOLO('$OUT/runs/parts_cls/weights/best.pt')
r = best.val(data='$D/classify', split='test', imgsz=64, batch=256, workers=4, plots=False)
cm = r.confusion_matrix.matrix.tolist()
res = {"split": "test (boards never trained on)", "top1": float(r.top1), "names": best.names, "confusion_pred_by_true": cm}
json.dump(res, open('$OUT/results/parts_classifier_test.json', 'w'), indent=1)
print('RESULT cls', {"top1": res["top1"]})
PY
echo PARTS_ALL_DONE
