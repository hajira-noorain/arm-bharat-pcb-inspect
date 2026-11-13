"""Experiment: does operator feedback measurably reduce false alarms without losing real defects?

Protocol (bare boards, per design; uses cached detections from eval_bare.py):
  cal split  -> split by board into FEEDBACK half and CALIBRATION half (no leakage)
  FEEDBACK   -> every detection the operator would see (conf >= lam of the uncalibrated-head gate)
               is labelled by the simulated operator: matches a ground-truth defect (IoU>=0.3, same
               class) -> 'confirm', otherwise -> 'false alarm'. Labels feed the re-scoring head.
  CALIBRATION-> conformal gate re-calibrated WITH the head (as the station does on promotion)
  TEST split -> never seen. Measured before vs after:
                 false alarms shown per board, defect recall among shown detections,
                 board-level miss rate / false-reject rate / review rate at alpha=5%.
Also a learning curve: false alarms per board vs number of operator labels.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.anomaly import torch_feature_fn  # noqa: E402
from pcbinspect.gate import ConformalGate, evaluate  # noqa: E402
from pcbinspect.learning import FeedbackBuffer, combine, crop_embedding, fit_head  # noqa: E402
from training.eval_bare import iou  # noqa: E402

TOPK = 20


def label_dets(board, thr_iou=0.3):
    used, labels = set(), []
    for d in board["dets"]:
        best, bj = 0.0, None
        for j, g in enumerate(board["boxes"]):
            if j in used or g["cls"] != d["cls"]:
                continue
            o = iou(d["bbox"], g["bbox"])
            if o > best:
                best, bj = o, j
        if bj is not None and best >= thr_iou:
            used.add(bj)
            labels.append(1)
        else:
            labels.append(0)
    return labels


def embed_boards(boards, fn, cache: Path):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(z["emb"])
    out = []
    for k, b in enumerate(boards):
        im = cv2.imread(b["path"])
        top = sorted(range(len(b["dets"])), key=lambda i: -b["dets"][i]["conf"])[:TOPK]
        e = {i: crop_embedding(fn, im, b["dets"][i]["bbox"]) for i in top}
        out.append(e)
        if k % 100 == 0:
            print(f"  embedded {k}/{len(boards)}", flush=True)
    np.savez(cache, emb=np.array(out, dtype=object))
    return out


def rescored(board, emb, head):
    confs = []
    for i, d in enumerate(board["dets"]):
        c = d["conf"]
        if head is not None and i in emb:
            c = combine(c, head.p_real(emb[i]))
        confs.append(c)
    return confs


def shown_stats(boards, embs, head, lam):
    fa = tp = n_gt = 0
    for b, e in zip(boards, embs):
        labels = label_dets(b)
        confs = rescored(b, e, head)
        for c, l in zip(confs, labels):
            if c >= lam:
                fa += l == 0
                tp += l == 1
        n_gt += len(b["boxes"])
    return {"false_alarms_per_board": fa / max(len(boards), 1), "shown_recall": tp / max(n_gt, 1),
            "false_alarms": fa, "true_shown": tp, "n_gt": n_gt, "boards": len(boards)}


def board_scores(boards, embs, head):
    d, g = [], []
    for b, e in zip(boards, embs):
        s = max(rescored(b, e, head), default=0.0)
        (g if b["good"] else d).append(s)
    return d, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=r"C:\Users\sulai\pcb_work")
    ap.add_argument("--tag", default="ultralytics")
    ap.add_argument("--designs", default="deeppcb,mendeley,hripcb")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    work = Path(args.work)
    res = work / "results"
    fn = torch_feature_fn(args.device)
    cal_all = json.loads((res / f"bare_dets_cal_{args.tag}.json").read_text())
    test_all = json.loads((res / f"bare_dets_test_{args.tag}.json").read_text())
    report = {}
    for design in args.designs.split(","):
        cal = [b for b in cal_all if b["source"] == design]
        test = [b for b in test_all if b["source"] == design]
        rng = random.Random(0)
        idx = list(range(len(cal)))
        rng.shuffle(idx)
        fb_idx, cb_idx = set(idx[: len(idx) // 2]), idx[len(idx) // 2:]
        feedback = [cal[i] for i in sorted(fb_idx)]
        calib = [cal[i] for i in cb_idx]
        e_fb = embed_boards(feedback, fn, res / f"emb_{design}_fb_{args.tag}.npz")
        e_cb = embed_boards(calib, fn, res / f"emb_{design}_cb_{args.tag}.npz")
        e_te = embed_boards(test, fn, res / f"emb_{design}_te_{args.tag}.npz")

        # BEFORE: detector only, gate calibrated on the calibration half
        g0 = ConformalGate(*board_scores(calib, e_cb, None), design)
        th0 = g0.thresholds(0.05, 0.01)
        before = {"shown": shown_stats(test, e_te, None, th0.lam),
                  "gate": evaluate(th0, *board_scores(test, e_te, None))}

        # operator labels every detection shown on the feedback boards (in inspection order)
        stream = []
        for b, e in zip(feedback, e_fb):
            for i, (d, l) in enumerate(zip(b["dets"], label_dets(b))):
                if d["conf"] >= th0.lam and i in e:
                    stream.append((e[i], l))
        curve = []
        for n in (10, 25, 50, 100, 200, 400, len(stream)):
            if n > len(stream):
                continue
            buf = FeedbackBuffer()
            for emb, l in stream[:n]:
                buf.add(emb, l)
            head = fit_head(buf, version=1)
            if head is None:
                continue
            g = ConformalGate(*board_scores(calib, e_cb, head), design)
            th = g.thresholds(0.05, 0.01)
            curve.append({"labels": n, "confirmed": head.n_pos, "false_alarms_labelled": head.n_neg,
                          **shown_stats(test, e_te, head, th.lam),
                          "gate": evaluate(th, *board_scores(test, e_te, head))})
        report[design] = {"n_feedback_boards": len(feedback), "n_calibration_boards": len(calib),
                          "n_test_boards": len(test), "labels_available": len(stream),
                          "before": before, "curve": curve}
        if curve:
            a = curve[-1]
            print(f"{design}: false alarms/board {before['shown']['false_alarms_per_board']:.2f} -> "
                  f"{a['false_alarms_per_board']:.2f}; shown recall {before['shown']['shown_recall']:.3f} -> "
                  f"{a['shown_recall']:.3f}; review {before['gate']['review_rate']:.2f} -> "
                  f"{a['gate']['review_rate']:.2f}; miss {before['gate']['miss_rate']:.3f} -> "
                  f"{a['gate']['miss_rate']:.3f}  ({len(stream)} labels)", flush=True)
    (res / f"exp_learning_{args.tag}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
