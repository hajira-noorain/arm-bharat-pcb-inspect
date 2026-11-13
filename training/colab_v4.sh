#!/usr/bin/env bash
# Colab job: detector v4 = v3 data + design-level synthetic tiles rendered from open-hardware KiCad boards
# (training/kicad/render.py; train split only, val unchanged so model selection is comparable), fine-tuned from v3.
# v4 replaces v3 only if it is better on the held-out DefectDet boards and not worse on the other test sets.
set -x
CODE=$(cd "$(dirname "$0")/.." && pwd)
D=/content/data
K=/content/kicad_out
OUT=/content/out3
until grep -q KICAD_DONE /content/run_kicad2.log; do sleep 60; done
until grep -q V3_DONE /content/run_v3.log; do sleep 60; done
V3=/content/out2/runs/y26n_v3/weights/best.pt
cd $CODE
python - <<PY
import json, yaml
from pathlib import Path
v3 = yaml.safe_load(open("$D/bare_yolo_v3_aug/data.yaml"))
root = Path(v3.get("path", "$D/bare_yolo_v3_aug"))
names = json.load(open("$K/manifest.json"))["classes"]
v3names = [v3["names"][i] for i in sorted(v3["names"])] if isinstance(v3["names"], dict) else list(v3["names"])
assert list(names) == v3names, ("class order differs", names, v3names)
n = len(list(Path("$K/images/train").glob("*.jpg")))
print("kicad train tiles", n)
v4 = {"path": "/", "train": [str(root / v3["train"]), "$K/images/train"], "val": str(root / v3["val"]),
      "names": v3["names"]}
Path("$D/data_v4.yaml").write_text(yaml.safe_dump(v4))
print(open("$D/data_v4.yaml").read())
PY
python -c "
from ultralytics import YOLO
YOLO('$V3').train(data='$D/data_v4.yaml', epochs=40, imgsz=640, batch=64, optimizer='SGD',
    lr0=0.003, lrf=0.05, momentum=0.937, cos_lr=True, warmup_epochs=1, close_mosaic=10, fliplr=0.5, flipud=0.5,
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, mosaic=1.0, scale=0.5, patience=20, project='/content/out2/runs',
    name='y26n_v4', exist_ok=True, workers=8, seed=0)"
V4=/content/out2/runs/y26n_v4/weights/best.pt
python training/eval_bare.py --work $OUT --data $D/bare_yolo_v3 --weights $V4 --tag v4 --no-gate
python training/eval_external.py --coco-root /content/ext/defectdet --codes defectdet \
  --only-boards $D/bare_yolo_v3/boards_test.json --only-source defectdet \
  --weights $V3 $V4 --names v3 v4 --gate-lam 0.25 0.25 --scales 1 --out $OUT/results/external_defectdet_v3_v4.json
echo V4_DONE
