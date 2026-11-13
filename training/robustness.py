"""Robustness sweep: how much detection quality is lost under camera-like conditions?

For each condition in camera_aug.CONDITIONS, every held-out TEST board (never trained on) is degraded
and run through the full tiled detector; board-level mAP@0.5 and per-defect recall at a fixed
confidence are reported per dataset. Run for several models to compare (e.g. base vs camera-robust).

python training/robustness.py --data <bare_yolo> --weights a.pt b.pt --names base cam --out results/robustness.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.detect_bare import BareDetector, UltralyticsTiles  # noqa: E402
from pcbinspect.record import BARE_CLASSES  # noqa: E402
from training.camera_aug import CONDITIONS, apply  # noqa: E402
from training.eval_bare import ap_voc, iou, match  # noqa: E402


def evaluate(det, boards, cond, seed=0, recall_conf=0.25):
    rng = np.random.default_rng(seed)
    per = defaultdict(lambda: {"tp": defaultdict(list), "conf": defaultdict(list), "ngt": defaultdict(int),
                               "found": 0, "total": 0})
    for b in boards:
        if b["good"]:
            continue
        im = cv2.imread(b["path"])
        im = apply(im, CONDITIONS[cond], rng)
        dets, _ = det.detect(im)
        dd = [{"cls": d.cls, "bbox": list(d.bbox), "conf": d.conf} for d in dets]
        for scope in ("all", b["source"]):
            s = per[scope]
            for g in b["boxes"]:
                s["ngt"][g["cls"]] += 1
                s["total"] += 1
                s["found"] += any(x["conf"] >= recall_conf and iou(x["bbox"], g["bbox"]) >= 0.3 for x in dd)
            flags, _ = match(dd, b["boxes"])
            for x, f in zip(dd, flags):
                s["tp"][x["cls"]].append(f)
                s["conf"][x["cls"]].append(x["conf"])
    out = {}
    for scope, s in per.items():
        aps = [ap_voc(s["tp"][c], s["conf"][c], s["ngt"][c]) for c in BARE_CLASSES if s["ngt"][c]]
        out[scope] = {"mAP50": float(np.mean(aps)) if aps else None,
                      "recall_at_0.25": s["found"] / max(s["total"], 1), "n_defects": s["total"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-boards", type=int, default=0, help="limit test boards per source (0 = all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    boards = json.loads((Path(a.data) / "boards_test.json").read_text())
    if a.max_boards:
        by = defaultdict(list)
        for b in boards:
            by[b["source"]].append(b)
        boards = [b for v in by.values() for b in v[:a.max_boards]]
    report = {"n_boards": len(boards), "models": {}}
    for w, name in zip(a.weights, a.names):
        det = BareDetector(UltralyticsTiles(w, device=a.device))
        report["models"][name] = {}
        for cond in a.conditions.split(","):
            r = evaluate(det, boards, cond)
            report["models"][name][cond] = r
            print(f"{name:6s} {cond:11s} mAP50 all={r['all']['mAP50']:.3f}  "
                  + "  ".join(f"{k}={v['mAP50']:.3f}" for k, v in r.items() if k != "all"), flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
