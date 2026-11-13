"""Golden-board mode: detector alone vs golden difference alone vs both fused, on held-out boards.

DeepPCB ships every test image with its defect-free template, i.e. a golden board. To mimic a real
re-capture, the golden is warped by a small random rotation/shift before use, so the station's ECC
alignment and local re-registration are exercised (no pre-aligned shortcut).

Per channel, the per-defect conformal gate (pcbinspect/gate.py) is calibrated on the CAL split and
evaluated on the TEST split: per-defect miss (all test defects), false calls per board, good boards
flagged. The fused gate is saved as models/gate/<design>_golden.json for the station.

python training/eval_golden.py --tag ep70
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.fusion import fuse  # noqa: E402
from pcbinspect.gate import ConformalGate, clopper_pearson, defect_level_scores  # noqa: E402
from pcbinspect.golden_diff import diff_candidates  # noqa: E402
from training.eval_bare import iou  # noqa: E402


def recapture(g, rng):
    M = cv2.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), rng.uniform(-0.3, 0.3), 1.0)
    M[:, 2] += rng.uniform(-2, 2, 2)
    return cv2.warpAffine(g, M, (g.shape[1], g.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def covers(bbox, gt):
    x0, y0, x1, y1 = gt
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return iou(bbox, gt) >= 0.1 or (x0 <= cx <= x1 and y0 <= cy <= y1)


def channels(b, rng):
    """Candidate lists [(bbox, score, cls)] per channel for one board."""
    det = [(tuple(d["bbox"]), d["conf"], d["cls"]) for d in b["dets"]]
    test = cv2.imread(b["path"])
    golden = recapture(cv2.imread(b["template"]), rng)
    dc, info = diff_candidates(test, golden)
    diff = [(c.bbox, c.score, c.cls) for c in dc]
    fused = [(c["bbox"], c["conf"], c["cls"]) for c in fuse(
        [{"bbox": d[0], "conf": d[1], "cls": d[2]} for d in det], dc)]
    return {"detector": det, "golden_diff": diff, "fused": fused}, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=r"C:\Users\sulai\pcb_work")
    ap.add_argument("--tag", default="ep70")
    ap.add_argument("--design", default="deeppcb")
    a = ap.parse_args()
    work = Path(a.work)
    res = work / "results"
    rng = np.random.default_rng(0)
    splits = {}
    for sp in ("cal", "test"):
        boards = [b for b in json.loads((res / f"bare_dets_{sp}_{a.tag}.json").read_text())
                  if b["source"] == a.design and b.get("template")]
        out = []
        for b in boards:
            ch, info = channels(b, rng)
            out.append({**b, "ch": ch, "align_cc": info["align_cc"]})
        splits[sp] = out
        print(sp, len(out), "boards", flush=True)

    report = {"design": a.design, "tag": a.tag, "channels": {}}
    for name in ("detector", "golden_diff", "fused"):
        def dscore(b, gt):
            return max((s for bb, s, _ in b["ch"][name] if covers(bb, gt["bbox"])), default=0.0)

        def bscore(b):
            return max((s for _, s, _ in b["ch"][name]), default=0.0)

        cal = splits["cal"]
        per = [[dscore(b, g) for g in b["boxes"]] for b in cal if not b["good"]]
        gate = ConformalGate(defect_level_scores(per), [bscore(b) for b in cal if b["good"]],
                             f"{a.design}_golden" if name == "fused" else a.design)
        if name == "fused":
            gdir = work / "models" / "gate"
            gdir.mkdir(parents=True, exist_ok=True)
            gate.save(gdir / f"{a.design}_golden.json")
        rows = {}
        for alpha in (0.01, 0.05, 0.10):
            th = gate.thresholds(alpha, 0.01)
            test = splits["test"]
            found = total = typed = 0
            fa = 0
            for b in test:
                for g in b["boxes"]:
                    total += 1
                    hit = [(s, c) for bb, s, c in b["ch"][name] if s >= th.lam and covers(bb, g["bbox"])]
                    found += bool(hit)
                    typed += any(c == g["cls"] for _, c in hit)
                fa += sum(1 for bb, s, _ in b["ch"][name]
                          if s >= th.lam and not any(covers(bb, g["bbox"]) for g in b["boxes"]))
            goods = [b for b in test if b["good"]]
            rows[f"alpha_{alpha}"] = {
                "lam": th.lam, "defect_miss": 1 - found / max(total, 1),
                "defect_miss_ci95": [1 - x for x in clopper_pearson(found, total)[::-1]],
                "typed_correct_share_of_found": typed / max(found, 1),
                "false_calls_per_board": fa / max(len(test), 1),
                "good_boards_flagged": (float(np.mean([bscore(b) >= th.lam for b in goods])) if goods else None),
                "n_defects": total, "n_boards": len(test)}
        report["channels"][name] = rows
        r = rows["alpha_0.05"]
        print(f"{name:12s} @5%: lam={r['lam']:.3f} defect miss={r['defect_miss']:.3f} "
              f"CI({r['defect_miss_ci95'][0]:.3f},{r['defect_miss_ci95'][1]:.3f}) false/board={r['false_calls_per_board']:.2f} "
              f"typed={r['typed_correct_share_of_found']:.2f}", flush=True)
    report["align_cc_min"] = float(min(b["align_cc"] for s in splits.values() for b in s))
    (res / f"golden_eval_{a.design}_{a.tag}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
