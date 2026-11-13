"""Raw parts-check measurements on VisA pcb1-4, so decision rules can be chosen on GOOD TRAINING photos and then
read off, unchanged, on the test photos (training/eval_parts_visa.py reports one fixed rule; this keeps the
measurements each rule is made from).

For every photo: for each taught part, the strongest size-compatible detection at its place and its likeness to
the good board; for each confident detection that is not a taught part, its confidence and the likeness of its
place. Photos: 'dev' = good training photos not used for teaching; 'test' = the test split (good and faulty).

python training/dump_parts_visa.py --visa /content/VisA --detector best.pt --out results/parts_visa_raw.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.golden import align_warp  # noqa: E402
from pcbinspect.parts import (_is_this_part, _same_place, dedupe, find_parts, likeness, moved,  # noqa: E402
                              ref_in_test_frame, teach_parts)
from training.eval_parts_visa import on_mask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visa", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--designs", default="pcb1,pcb2,pcb3,pcb4")
    ap.add_argument("--goods", default="4,8", help="how many good photos teach the board (each is dumped)")
    ap.add_argument("--dev", type=int, default=80)
    ap.add_argument("--normals", type=int, default=80)
    a = ap.parse_args()
    from pcbinspect.detect_bare import BareDetector, UltralyticsTiles
    det = BareDetector(UltralyticsTiles(a.detector, conf_thr=0.1), names=("part",))
    visa = Path(a.visa)
    rows = list(csv.DictReader(open(visa / "split_csv" / "1cls.csv")))
    out: dict = {}
    for design in a.designs.split(","):
        types = {r["image"]: [t.strip() for t in r["label"].split(",")]
                 for r in csv.DictReader(open(visa / design / "image_anno.csv"))}
        mine = [r for r in rows if r["object"] == design]
        train = [r for r in mine if r["split"] == "train"]
        for g in [int(x) for x in a.goods.split(",")]:
            goods = [cv2.imread(str(visa / r["image"])) for r in train[:g]]
            pm = teach_parts(det, None, goods, max_goods=g)
            ref = goods[0]
            photos = [("dev", r) for r in train[16:16 + a.dev]] + \
                     [("test", r) for r in mine if r["split"] == "test" and r["label"] == "normal"][:a.normals] + \
                     [("test", r) for r in mine if r["split"] == "test" and r["label"] != "normal"]
            recs = []
            for split, r in photos:
                img = cv2.imread(str(visa / r["image"]))
                good = r["label"] == "normal"
                rec = {"split": split, "image": r["image"], "types": ["good"] if good else types.get(r["image"], ["anomaly"])}
                warp, cc = align_warp(img, ref)
                rec["cc"] = float(cc)
                if warp is None:
                    recs.append(rec)
                    continue
                h, w = img.shape[:2]
                mask = None
                if not good and r["mask"]:
                    mask = cv2.imread(str(visa / r["mask"]), cv2.IMREAD_GRAYSCALE)
                    if mask is not None and mask.shape[:2] != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                dets, _ = find_parts(det, img, scale=pm.scale)
                to_test = np.linalg.inv(np.vstack([np.asarray(warp, float), [0.0, 0.0, 1.0]]))
                there = moved(pm, to_test @ np.diag([w / pm.size[0], h / pm.size[1], 1.0]), (w, h))
                ref_there = ref_in_test_frame(np.asarray(warp, np.float32), ref, img)
                rec["parts"] = []
                for p in there.parts:
                    conf = max((d["conf"] for d in dets if _is_this_part(p.box, d["bbox"])), default=0.0)
                    lk = likeness(ref_there, img, p.box)
                    rec["parts"].append([round(conf, 3), None if lk is None else round(lk, 3),
                                         None if p.sim_min is None else round(p.sim_min, 3),
                                         bool(mask is not None and on_mask(p.box, mask))])
                rec["others"] = []
                for d in dedupe([d for d in dets if d["conf"] >= 0.3]):
                    if any(_same_place(d["bbox"], p.box) for p in there.parts):
                        continue
                    lk = likeness(ref_there, img, d["bbox"])
                    rec["others"].append([round(d["conf"], 3), None if lk is None else round(lk, 3),
                                          any(_same_place(d["bbox"], z) for z in there.ignore),
                                          bool(mask is not None and on_mask(d["bbox"], mask))])
                recs.append(rec)
            out[f"{design}:{g}"] = {"taught": pm.summary(), "scale": pm.scale, "ignore_zones": len(pm.ignore), "photos": recs}
            print(design, g, pm.summary(), len(recs), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    print("DUMP_DONE")


if __name__ == "__main__":
    main()
