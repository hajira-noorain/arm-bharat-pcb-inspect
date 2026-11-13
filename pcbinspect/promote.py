"""Promote an operator-trained re-scoring head only if it is provably not worse.

Steps (run on the device, in the background):
  1. fit the head on all operator labels
  2. regression check: 5-fold cross-validation on the labels. Versus the detector alone, the head must
       - separate real defects from false alarms better (balanced accuracy), and
       - keep >= MIN_RECALL of confirmed defects (it may never learn to hide real defects)
  3. re-calibrate the conformal gate WITH the head on the design's calibration boards, so the
     miss-rate guarantee still holds after the change
  4. save head + gate as a new version; otherwise keep the current version and log why
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .gate import ConformalGate
from .learning import FeedbackBuffer, fit_head

MIN_RECALL = 0.95


def cross_validate(buf: FeedbackBuffer, folds: int = 5, seed: int = 0) -> dict:
    X, y = np.stack(buf.X), np.array(buf.y)
    idx = np.random.default_rng(seed).permutation(len(y))
    pred = np.zeros(len(y))
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        sub = FeedbackBuffer()
        for i in train:
            sub.add(X[i], int(y[i]))
        head = fit_head(sub, version=0)
        if head is None:
            return {"ok": False, "reason": "not enough labels of each kind in a fold"}
        pred[test] = [head.p_real(X[i]) for i in test]
    real, fa = y == 1, y == 0
    recall = float((pred[real] >= 0.5).mean())
    fa_removed = float((pred[fa] < 0.5).mean())
    bal = (recall + fa_removed) / 2
    return {"ok": True, "recall_on_confirmed": recall, "false_alarms_removed": fa_removed,
            "balanced_accuracy": bal, "baseline_balanced_accuracy": 0.5, "n": int(len(y))}


def try_promote(inspector, design: str, buf: FeedbackBuffer, root: Path, store=None) -> dict:
    t0 = time.time()
    n_pos, n_neg = buf.counts()
    report = {"design": design, "confirmed": n_pos, "false_alarms": n_neg}
    cv = cross_validate(buf)
    report["cross_validation"] = cv
    if not cv.get("ok"):
        report.update(promoted=False, reason=cv.get("reason"))
        return report
    if cv["recall_on_confirmed"] < MIN_RECALL:
        report.update(promoted=False, reason=f"would hide real defects (recall {cv['recall_on_confirmed']:.0%})")
        return _log(report, store, design)
    if cv["balanced_accuracy"] <= cv["baseline_balanced_accuracy"] + 0.05:
        report.update(promoted=False, reason="no clear improvement over the detector alone")
        return _log(report, store, design)

    old = inspector.heads.get(design)
    head = fit_head(buf, version=(old.version + 1) if old else 1)
    # re-calibrate the gate with the new head on the shipped calibration boards
    man = root / "models" / "calib" / design / "manifest.json"
    if man.exists():
        items = json.loads(man.read_text())
        d_scores, g_scores = [], []
        for it in items:
            im = cv2.imread(str(man.parent / it["file"]))
            if im is None:
                continue
            a = inspector.analyse(im, design, head=head, read_id=False)
            (g_scores if it["good"] else d_scores).append(a.board_score)
        gate = ConformalGate(d_scores, g_scores, design)
        report["recalibrated_on"] = {"defective": len(d_scores), "good": len(g_scores)}
    else:
        gate = None
        report["recalibrated_on"] = None
        report["note"] = "no calibration boards on this station: gate kept, review rate may change"

    heads_dir = root / "models" / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)
    head.save(heads_dir / f"{design}.json")
    inspector.heads[design] = head
    if gate is not None:
        gpath = root / inspector.designs[design].gate_file
        gpath.parent.mkdir(parents=True, exist_ok=True)
        gate.save(gpath)
        inspector.gates[design] = gate
    report.update(promoted=True, version=head.version, seconds=round(time.time() - t0, 1))
    return _log(report, store, design)


def _log(report: dict, store, design: str) -> dict:
    if store is not None:
        store.record_model_version(f"rescore_head:{design}", str(report.get("version", "rejected")),
                                   report, bool(report.get("promoted")), report.get("reason", ""))
    return report
