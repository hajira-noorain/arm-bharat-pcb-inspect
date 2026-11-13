"""Detection limit in silico: POD vs defect size on held-out clean boards with drawn defects.

Each held-out DeepPCB clean board (never trained on) gets one defect of each type at one size level
(defect_render checks every open really severs and every short really joins). The SAME drawn
boards are then judged under several camera conditions (paired design), and the untouched clean
boards give the false-call rate that goes with the detection limit.

A defect is a HIT at threshold t if a detection with conf >= t covers it (box contains the defect
centre, or IoU >= 0.3). Thresholds reported: the deployed gate's lambda at --alpha (what the
station actually shows) and the 0.25 display floor.

python training/pod_insilico.py --data <bare_yolo> --weights best.pt --gate models/gate/deeppcb.json \
    --out results/pod_insilico.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.defect_render import NOUN, TYPES, render_board  # noqa: E402
from pcbinspect.fusion import fuse  # noqa: E402
from pcbinspect.golden_diff import diff_candidates  # noqa: E402
from pcbinspect.detect_bare import BareDetector, UltralyticsTiles  # noqa: E402
from pcbinspect.gate import ConformalGate  # noqa: E402
from pcbinspect.pod import fit_pod  # noqa: E402
from training.camera_aug import CONDITIONS, apply  # noqa: E402
from training.eval_bare import iou  # noqa: E402
from training.eval_golden import recapture  # noqa: E402

PX_PER_MM = 48.0  # DeepPCB scan resolution
SIZES = (0.06, 0.10, 0.14, 0.18, 0.24, 0.30, 0.40, 0.55)
SHORT_GAPS = (0.15, 0.18, 0.21, 0.24, 0.30, 0.36, 0.45, 0.55)  # shorts: size = GAP bridged (few gaps < 0.15 mm exist)


def covers(det_box, placed) -> bool:
    x0, y0, x1, y1 = det_box
    cx, cy = placed.center
    tol = 4
    return (x0 - tol <= cx <= x1 + tol and y0 - tol <= cy <= y1 + tol) or iou(det_box, placed.bbox) >= 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--golden-gate", default="", help="fused gate (eval_golden.py); enables golden mode")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--per", type=int, default=40, help="trials per (type, size)")
    ap.add_argument("--conditions", default="clean,webcam,scale_0.5")
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-examples", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    boards = json.loads((Path(a.data) / "boards_test.json").read_text())
    temps = sorted({b["template"] for b in boards if b["source"] == "deeppcb" and b.get("template")})
    lam = float(ConformalGate.load(a.gate).thresholds(a.alpha, a.alpha).lam)
    thresholds = {"gate_lambda": lam, "floor_0.25": 0.25}
    lam_g = float(ConformalGate.load(a.golden_gate).thresholds(a.alpha, a.alpha).lam) if a.golden_gate else None
    print(f"{len(temps)} held-out clean boards; gate lambda@{a.alpha} = {lam:.3f}", flush=True)

    rng = np.random.default_rng(a.seed)
    jobs = []  # (template, size)
    for k in range(a.per):
        for s in SIZES:
            jobs.append((temps[rng.integers(len(temps))], s))
    t0 = time.time()
    drawn = []
    for i, (tp, s) in enumerate(jobs):
        specs = [(c, SHORT_GAPS[SIZES.index(s)] if c == "short" else s) for c in TYPES]
        img, placed = render_board(cv2.imread(tp), specs, PX_PER_MM, rng)
        drawn.append((tp, img, placed))
    n_placed = sum(len(p) for _, _, p in drawn)
    print(f"drew {n_placed}/{len(jobs) * len(TYPES)} defects in {time.time() - t0:.0f}s", flush=True)
    if a.save_examples:
        ex = Path(a.save_examples)
        ex.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(drawn), len(SIZES))[:3]:
            for j in range(len(SIZES)):
                tp, img, placed = drawn[i + j]
                cv2.imwrite(str(ex / f"ex{i}_{placed[0].size_mm if placed else 0:.2f}.png"), img)

    det = BareDetector(UltralyticsTiles(a.weights, device=a.device))
    report = {"px_per_mm": PX_PER_MM, "sizes_mm": SIZES, "short_gaps_mm": SHORT_GAPS, "alpha": a.alpha, "thresholds": thresholds,
              "n_boards_clean": len(temps), "n_drawn": n_placed, "conditions": {}}
    for cond in a.conditions.split(","):
        crng = np.random.default_rng(a.seed + 1)
        trials = defaultdict(list)  # type -> [(size, max_conf_covering, cls_of_best)]
        gtrials = defaultdict(list)  # golden mode: type -> [(size, max_fused_conf_covering, cls)]
        grng = np.random.default_rng(a.seed + 2)

        def golden_of(tp):
            # the golden is a separate capture of a good board: own misalignment, own sensor noise
            return apply(recapture(cv2.imread(tp), grng), CONDITIONS[cond], grng)

        for tp, img, placed in drawn:
            test = apply(img, CONDITIONS[cond], crng)
            dets, _ = det.detect(test)
            fz = None
            if lam_g is not None:
                dc, _ = diff_candidates(test, golden_of(tp))
                fz = fuse([{"bbox": d.bbox, "conf": d.conf, "cls": d.cls} for d in dets], dc)
            for p in placed:
                best = max((d for d in dets if covers(d.bbox, p)), key=lambda d: d.conf, default=None)
                trials[p.cls].append((p.size_mm, best.conf if best else 0.0, best.cls if best else None))
                if fz is not None:
                    fb = max((c for c in fz if covers(c["bbox"], p)), key=lambda c: c["conf"], default=None)
                    gtrials[p.cls].append((p.size_mm, fb["conf"] if fb else 0.0, fb["cls"] if fb else None))
        fc = {k: [] for k in thresholds}
        gfc = []
        for tp in temps:
            test = apply(cv2.imread(tp), CONDITIONS[cond], crng)
            dets, _ = det.detect(test)
            for k, t in thresholds.items():
                fc[k].append(sum(d.conf >= t for d in dets))
            if lam_g is not None:
                dc, _ = diff_candidates(test, golden_of(tp))
                fz = fuse([{"bbox": d.bbox, "conf": d.conf, "cls": d.cls} for d in dets], dc)
                gfc.append(sum(c["conf"] >= lam_g for c in fz))
        out = {"false_calls_per_board": {k: float(np.mean(v)) for k, v in fc.items()},
               "boards_with_false_call": {k: float(np.mean(np.array(v) > 0)) for k, v in fc.items()},
               "types": {}, "raw": {cls: [[s, round(c, 4), k] for s, c, k in trials[cls]] for cls in TYPES}}
        all_s, all_c = [], []
        for cls in TYPES:
            tr = trials[cls]
            s = np.array([x[0] for x in tr])
            c = np.array([x[1] for x in tr])
            all_s += list(s)
            all_c += list(c)
            right = [x[2] == cls for x in tr if x[2] is not None]
            out["types"][cls] = {k: fit_pod(s, (c >= t).astype(int)).to_dict() for k, t in thresholds.items()}
            out["types"][cls]["class_correct_when_found"] = float(np.mean(right)) if right else None
        out["all_types"] = {k: fit_pod(np.array(all_s), (np.array(all_c) >= t).astype(int)).to_dict()
                            for k, t in thresholds.items()}
        if lam_g is not None:
            out["golden"] = {"lambda": lam_g, "false_calls_per_board": float(np.mean(gfc)),
                             "boards_with_false_call": float(np.mean(np.array(gfc) > 0)), "types": {},
                             "raw": {cls: [[s_, round(c_, 4), k_] for s_, c_, k_ in gtrials[cls]] for cls in TYPES}}
            for cls in TYPES:
                tr = gtrials[cls]
                r = fit_pod(np.array([x[0] for x in tr]), (np.array([x[1] for x in tr]) >= lam_g).astype(int))
                out["golden"]["types"][cls] = r.to_dict()
                out["golden"]["types"][cls]["sentence"] = r.sentence(NOUN[cls])
            print(f"[{cond}] GOLDEN false calls/board {out['golden']['false_calls_per_board']:.2f}: " + " | ".join(
                out["golden"]["types"][c]["sentence"] for c in TYPES), flush=True)
        report["conditions"][cond] = out
        for cls in TYPES:
            out["types"][cls]["sentence"] = fit_pod(
                np.array([x[0] for x in trials[cls]]),
                (np.array([x[1] for x in trials[cls]]) >= lam).astype(int)).sentence(NOUN[cls])
        line = " | ".join(out["types"][cls]["sentence"] for cls in TYPES)
        print(f"[{cond}] DETECTOR false calls/board {out['false_calls_per_board']['gate_lambda']:.2f}: {line}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
