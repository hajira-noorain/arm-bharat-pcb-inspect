"""The parts check on REAL photographs of assembled boards with real missing and extra parts: VisA pcb1-pcb4
(Zou et al. 2022). The part detector never saw these boards or this camera (it was trained on FPIC).

For each design the station is taught from a few good photos, exactly as it would be on the bench (no mm scale
is known for VisA, so teaching picks the resampling factor itself). Then:

  good boards (test split)       how often the parts check raises anything at all (false calls)
  boards VisA labels "missing"   is a missing part reported ON the labelled area?
  boards VisA labels "extra"     is an unexpected part reported ON the labelled area?
  other anomalies (melt, bent, scratch, dirt, ...)   reported for reference: these are not parts faults

python training/eval_parts_visa.py --visa /content/VisA --detector best.pt --classifier cls.pt --out results/parts_check_visa.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.golden import align_warp  # noqa: E402
from pcbinspect.parts import MIN_ALIGN_CC_PARTS as MIN_ALIGN_CC, check_board, find_parts, teach_parts  # noqa: E402


MODES = ("detector only", "detector and likeness")


def on_mask(box, mask: np.ndarray, grow: float = 0.15) -> bool:
    """Does the box (grown a little) touch the labelled anomaly area?"""
    h, w = mask.shape[:2]
    gx, gy = (box[2] - box[0]) * grow + 3, (box[3] - box[1]) * grow + 3
    x0, y0 = int(max(box[0] - gx, 0)), int(max(box[1] - gy, 0))
    x1, y1 = int(min(box[2] + gx, w)), int(min(box[3] + gy, h))
    return x1 > x0 and y1 > y0 and bool(mask[y0:y1, x0:x1].any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visa", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--classifier", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--designs", default="pcb1,pcb2,pcb3,pcb4")
    ap.add_argument("--goods", type=int, default=8, help="good photos used to teach each design")
    ap.add_argument("--normals", type=int, default=80, help="good test photos checked per design")
    a = ap.parse_args()
    from pcbinspect.detect_bare import BareDetector, UltralyticsTiles
    from training.eval_parts import UltraKinds
    det = BareDetector(UltralyticsTiles(a.detector, conf_thr=0.1), names=("part",))
    kinds = UltraKinds(a.classifier) if a.classifier else None
    visa = Path(a.visa)
    rows = list(csv.DictReader(open(visa / "split_csv" / "1cls.csv")))
    report: dict = {"note": "real photos (VisA); the part detector was trained on FPIC only", "designs": {}}
    for design in a.designs.split(","):
        types = {r["image"]: [t.strip() for t in r["label"].split(",")]
                 for r in csv.DictReader(open(visa / design / "image_anno.csv"))}
        mine = [r for r in rows if r["object"] == design]
        goods = [cv2.imread(str(visa / r["image"])) for r in mine if r["split"] == "train"][:a.goods]
        pm = teach_parts(det, kinds, goods)
        ref = goods[0]
        stats = {m: defaultdict(lambda: defaultdict(int)) for m in MODES}
        tests = [r for r in mine if r["split"] == "test" and r["label"] == "normal"][:a.normals] + \
                [r for r in mine if r["split"] == "test" and r["label"] != "normal"]
        for r in tests:
            img = cv2.imread(str(visa / r["image"]))
            good = r["label"] == "normal"
            kinds_here = ["good"] if good else types.get(r["image"], ["anomaly"])
            warp, cc = align_warp(img, ref)
            ok = warp is not None and cc >= MIN_ALIGN_CC
            found = {m: [] for m in MODES}
            if ok:
                dets, _ = find_parts(det, img, scale=pm.scale)
                found["detector only"] = check_board(pm, dets, warp, (img.shape[1], img.shape[0]))
                found["detector and likeness"] = check_board(pm, dets, warp, (img.shape[1], img.shape[0]),
                                                             ref=ref, test=img)
            mask = None
            if not good and r["mask"]:
                mask = cv2.imread(str(visa / r["mask"]), cv2.IMREAD_GRAYSCALE)
                if mask is not None and mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            for mode, fs in found.items():
                for k in kinds_here:
                    s = stats[mode][k]
                    s["images"] += 1
                    s["lined_up"] += ok
                    s["any_parts_finding"] += bool(fs)
                    s["missing_calls"] += sum(f.what == "missing_part" for f in fs)
                    s["changed_calls"] += sum(f.what == "changed_part" for f in fs)
                    s["unexpected_calls"] += sum(f.what == "extra_part" for f in fs)
                    if mask is not None:
                        s["finding_on_the_labelled_area"] += any(on_mask(f.box, mask) for f in fs)
                        s["missing_on_the_labelled_area"] += any(f.what == "missing_part" and on_mask(f.box, mask) for f in fs)
                        s["changed_on_the_labelled_area"] += any(f.what == "changed_part" and on_mask(f.box, mask) for f in fs)
                        s["unexpected_on_the_labelled_area"] += any(f.what == "extra_part" and on_mask(f.box, mask) for f in fs)
        report["designs"][design] = {
            "taught": pm.summary(), "scale_chosen": pm.scale, "ignore_zones": len(pm.ignore),
            "likeness_across_good_boards_min": min((p.sim_min for p in pm.parts if p.sim_min is not None), default=None),
            "by_type": {k: dict(v) for k, v in stats["detector and likeness"].items()},
            "by_type_detector_only": {k: dict(v) for k, v in stats["detector only"].items()}}
        print(design, json.dumps(report["designs"][design]), flush=True)
    for key in ("by_type", "by_type_detector_only"):
        tot = defaultdict(lambda: defaultdict(int))
        for d in report["designs"].values():
            for k, v in d[key].items():
                for kk, vv in v.items():
                    tot[k][kk] += vv
        report["all_designs" if key == "by_type" else "all_designs_detector_only"] = {k: dict(v) for k, v in tot.items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print("RESULT parts_visa", json.dumps(report["all_designs"]))


if __name__ == "__main__":
    main()
