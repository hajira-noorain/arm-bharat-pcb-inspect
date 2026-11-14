#!/usr/bin/env bash
# Parts check: end-to-end test on held-out boards (simulated removal), then export both models to ExecuTorch
# and check that the exported files give the same answers as the trained ones.
set -x
until grep -q "PARTS_ALL_DONE" /content/run_parts.log; do sleep 30; done
CODE=/content/code13; OUT=/content/outp; D=/content/data/parts
rm -rf $CODE && mkdir -p $CODE && unzip -q -o /content/code13.zip -d $CODE
cd $CODE
DET=$OUT/runs/parts_det/weights/best.pt; CLS=$OUT/runs/parts_cls/weights/best.pt
python training/eval_parts.py --fpic /content/ext/fpic --manifest $D/manifest.json --detector $DET --classifier $CLS \
   --out $OUT/results/parts_check_eval.json 2>&1 | grep -E "RESULT|rror|Traceback" | cut -c1-3000
echo PARTS_EVAL_DONE
mkdir -p $OUT/models/vision && cd $OUT/models/vision
python - <<PY 2>&1 | grep -E "RESULT|rror|Traceback|EXPORT"
import json, shutil, glob
from pathlib import Path
import numpy as np, cv2, torch
from ultralytics import YOLO
rep = {}
def run_pte(path, x):
    from executorch.runtime import Runtime
    m = Runtime.get().load_program(str(path)).load_method("forward")
    return np.asarray(m.execute([torch.from_numpy(x)])[0])
# detector
det = YOLO('$DET')
ex = Path(det.export(format='executorch', imgsz=640))
pte = next(ex.rglob('*.pte')) if ex.is_dir() else ex
shutil.copy(pte, 'parts_y26n_xnnpack.pte'); shutil.copy('$DET', 'parts_y26n.pt')
tiles = sorted(glob.glob('$D/detect/images/test/*.jpg'))[::40][:12]
same = []
for t in tiles:
    im = cv2.imread(t)
    x = np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    o = run_pte('parts_y26n_xnnpack.pte', x)
    n_pte = int((o[0][:, 4] >= 0.25).sum()) if o.ndim == 3 and o.shape[-1] == 6 else -1
    n_pt = len(det.predict(im, imgsz=640, conf=0.25, verbose=False)[0].boxes)
    same.append((n_pt, n_pte))
rep['detector'] = {'pte': 'parts_y26n_xnnpack.pte', 'bytes': Path('parts_y26n_xnnpack.pte').stat().st_size,
                   'parts_found_torch_vs_pte_on_12_test_tiles': same}
print('EXPORT det', rep['detector'])
# classifier
cls = YOLO('$CLS')
ex = Path(cls.export(format='executorch', imgsz=64))
pte = next(ex.rglob('*.pte')) if ex.is_dir() else ex
shutil.copy(pte, 'parts_kind_xnnpack.pte')
names = [cls.names[i] for i in sorted(cls.names)]
Path('parts_kind_xnnpack.json').write_text(json.dumps({'names': names}))
import sys; sys.path.insert(0, '$CODE')
from pcbinspect.parts import PartKindClassifier
k = PartKindClassifier('parts_kind_xnnpack.pte', names)
crops = sorted(glob.glob('$D/classify/test/*/*.jpg'))[::7][:400]
ims = [cv2.imread(c) for c in crops]
a = k(ims)
b = [(r.names[int(r.probs.top1)], float(r.probs.top1conf)) for r in cls.predict(ims, imgsz=64, verbose=False)]
truth = [Path(c).parent.name for c in crops]
rep['classifier'] = {'pte': 'parts_kind_xnnpack.pte', 'bytes': Path('parts_kind_xnnpack.pte').stat().st_size, 'names': names,
                     'crops': len(crops), 'same_answer_as_torch': float(np.mean([x[0] == y[0] for x, y in zip(a, b)])),
                     'max_confidence_difference': float(max(abs(x[1] - y[1]) for x, y in zip(a, b))),
                     'top1_pte': float(np.mean([x[0] == t for x, t in zip(a, truth)])),
                     'top1_torch': float(np.mean([y[0] == t for y, t in zip(b, truth)]))}
print('EXPORT cls', rep['classifier'])
Path('$OUT/results/parts_export.json').write_text(json.dumps(rep, indent=1))
PY
ls -la $OUT/models/vision
echo PARTS2_ALL_DONE
