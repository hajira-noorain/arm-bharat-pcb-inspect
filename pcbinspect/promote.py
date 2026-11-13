"""Switch on an operator-trained re-scoring head only if the checks say it is not worse.

These are checks against the operators' own labels and the shipped calibration boards, not an
independent proof. Steps (run on the device, in the background):

  1. fit the head on all operator labels
  2. board-grouped cross-validation (no board in both train and test folds):
       - the combined score must rank real defects above false alarms better than the detector's
         own confidence does (AUROC, held-out folds) by at least MIN_AUC_GAIN
       - the head must keep >= MIN_RECALL of confirmed defects
  3. re-calibrate the conformal gate WITH the head on the shipped calibration boards
  4. regression check on those boards' known defects: per-defect recall at the new threshold may
     not fall more than MAX_RECALL_DROP below the current version's
  5. save as a new version, or keep the current one and log why
Critical defect types (short, open, missing hole) can only be made more suspicious by the head.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .gate import ConformalGate
from .learning import FeedbackBuffer, fit_head, rescore

MIN_RECALL = 0.95
MIN_AUC_GAIN = 0.02
MAX_RECALL_DROP = 0.01


def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def cross_validate(buf: FeedbackBuffer, folds: int = 5, seed: int = 0) -> dict:
    X, y = np.stack(buf.X), np.array(buf.y)
    conf, groups = np.array(buf.conf, float), np.array(buf.group)
    uniq = np.unique(groups)
    if len(uniq) < folds:
        return {"ok": False, "reason": f"labels come from only {len(uniq)} boards; need at least {folds}"}
    perm = np.random.default_rng(seed).permutation(uniq)
    fold_of = {g: i % folds for i, g in enumerate(perm)}
    p = np.zeros(len(y))
    for f in range(folds):
        test = np.array([fold_of[g] == f for g in groups])
        sub = FeedbackBuffer()
        for i in np.flatnonzero(~test):
            sub.add(X[i], int(y[i]), conf=conf[i], group=groups[i])
        head = fit_head(sub, version=0)
        if head is None:
            return {"ok": False, "reason": "not enough labels of each kind in a fold"}
        p[test] = [head.p_real(X[i]) for i in np.flatnonzero(test)]
    combined = np.array([rescore("", c, pi) for c, pi in zip(conf, p)])
    real = y == 1
    return {"ok": True, "n": int(len(y)), "boards": int(len(uniq)),
            "recall_on_confirmed": float((p[real] >= 0.5).mean()),
            "false_alarms_removed": float((p[~real] < 0.5).mean()),
            "auc_detector": _auc(y, conf), "auc_with_head": _auc(y, combined)}


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _calibration_pass(inspector, design: str, items: list, base: Path, head) -> dict:
    """Board scores + per-defect hits on the calibration boards with a given head."""
    d_scores, g_scores, per_defect = [], [], []
    for it in items:
        im = cv2.imread(str(base / it["file"]))
        if im is None:
            continue
        a = inspector.analyse(im, design, head=head, read_id=False)
        (g_scores if it["good"] else d_scores).append(a.board_score)
        for gt in it.get("boxes", []):
            best = max((c.conf for c in a.candidates if c.cls == gt["cls"] and _iou(c.bbox, gt["bbox"]) >= 0.3),
                       default=0.0)
            per_defect.append(best)
    return {"d": d_scores, "g": g_scores, "defect_scores": per_defect}


def try_promote(inspector, design: str, buf: FeedbackBuffer, root: Path, store=None,
                alpha: float = 0.05) -> dict:
    t0 = time.time()
    n_pos, n_neg = buf.counts()
    report = {"design": design, "confirmed": n_pos, "false_alarms": n_neg}
    cv = cross_validate(buf)
    report["cross_validation"] = cv
    if not cv.get("ok"):
        report.update(promoted=False, reason=cv.get("reason"))
        return _log(report, store, design)
    if cv["recall_on_confirmed"] < MIN_RECALL:
        report.update(promoted=False, reason=f"would hide real defects (recall {cv['recall_on_confirmed']:.0%})")
        return _log(report, store, design)
    if not cv["auc_with_head"] >= cv["auc_detector"] + MIN_AUC_GAIN:
        report.update(promoted=False, reason=(f"no clear improvement over the detector alone "
                                              f"(AUROC {cv['auc_detector']:.3f} -> {cv['auc_with_head']:.3f})"))
        return _log(report, store, design)

    old = inspector.heads.get(design)
    head = fit_head(buf, version=(old.version + 1) if old else 1)
    man = root / "models" / "calib" / design / "manifest.json"
    gate = None
    if man.exists():
        items = json.loads(man.read_text())
        before = _calibration_pass(inspector, design, items, man.parent, old)
        after = _calibration_pass(inspector, design, items, man.parent, head)
        lam_before = ConformalGate(before["d"], before["g"]).thresholds(alpha).lam
        gate = ConformalGate(after["d"], after["g"], design)
        lam_after = gate.thresholds(alpha).lam
        rec = lambda s, lam: float(np.mean([x >= lam for x in s])) if s else float("nan")  # noqa: E731
        r0, r1 = rec(before["defect_scores"], lam_before), rec(after["defect_scores"], lam_after)
        report["calibration"] = {"boards_defective": len(after["d"]), "boards_good": len(after["g"]),
                                 "defect_recall_before": r0, "defect_recall_after": r1}
        if before["defect_scores"] and r1 < r0 - MAX_RECALL_DROP:
            report.update(promoted=False, reason=f"known defects on calibration boards would be missed "
                                                 f"(recall {r0:.1%} -> {r1:.1%})")
            return _log(report, store, design)
    else:
        report["calibration"] = None
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
