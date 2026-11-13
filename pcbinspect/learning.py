"""On-device learning for the named-defect detector: an operator-trained re-scoring head.

The detector itself (YOLO .pte) never changes on the device. Instead each detection's crop is
embedded with the frozen anomaly backbone (already loaded), and a small logistic-regression head
learns p(real defect | crop) from operator labels (confirm / false alarm). Fitting takes well under
a second on a Pi CPU.

Combined confidence (independent-evidence update with a 0.5 prior):
    c' = c*p / (c*p + (1-c)*(1-p))

Every refit creates a new version that is only activated if it passes the regression gate
(the caller re-runs the fixed regression set and compares; see pipeline.Inspector.try_promote_head).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .anomaly import FeatureFn, preprocess

MIN_PER_CLASS = 15  # don't fit a head until we have at least this many of each label
# Defect types the head may only make MORE suspicious, never less: a wrongly taught "false alarm"
# must not be able to hide a short, an open or a missing hole.
PROTECTED_TYPES = {"short", "open", "missing_hole"}


def rescore(cls: str, conf: float, p_real: float) -> float:
    c = combine(conf, p_real)
    return max(conf, c) if cls in PROTECTED_TYPES else c


def crop_embedding(feature_fn: FeatureFn, bgr: np.ndarray, bbox, pad: float = 0.5) -> np.ndarray:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh) * (1 + 2 * pad)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = bgr[b:d, a:c]
    if crop.size == 0:
        crop = bgr
    f = feature_fn(preprocess(crop))           # (1, C, H, W)
    v = np.concatenate([f.mean(axis=(2, 3))[0], f.max(axis=(2, 3))[0]])
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def combine(conf: float, p_real: float) -> float:
    num = conf * p_real
    den = num + (1 - conf) * (1 - p_real)
    return float(num / den) if den > 0 else conf


@dataclass
class RescoreHead:
    w: np.ndarray
    b: float
    version: int
    n_pos: int
    n_neg: int

    def p_real(self, emb: np.ndarray) -> float:
        z = float(emb @ self.w + self.b)
        return float(1 / (1 + np.exp(-np.clip(z, -30, 30))))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"w": self.w.tolist(), "b": self.b, "version": self.version,
                                          "n_pos": self.n_pos, "n_neg": self.n_neg}))

    @classmethod
    def load(cls, path: Path) -> "RescoreHead":
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["w"], np.float32), d["b"], d["version"], d["n_pos"], d["n_neg"])


class FeedbackBuffer:
    """Labelled crop embeddings: 1 = operator confirmed a real defect, 0 = false alarm.

    Each example also keeps the detector's own confidence (so a new head is compared against the
    detector, not a constant) and the board it came from (so validation folds never share a board).
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.X: list[np.ndarray] = []
        self.y: list[int] = []
        self.event_ids: list[int | None] = []
        self.conf: list[float] = []
        self.group: list[int] = []
        if self.path and self.path.exists():
            d = np.load(self.path, allow_pickle=False)
            self.X = list(d["X"])
            self.y = list(d["y"].astype(int))
            self.event_ids = [None if e < 0 else int(e) for e in d["ev"]]
            self.conf = list(d["conf"]) if "conf" in d else [0.5] * len(self.y)
            self.group = list(d["group"]) if "group" in d else list(range(len(self.y)))

    def add(self, emb: np.ndarray, label: int, event_id: int | None = None,
            conf: float = 0.5, group: int | None = None) -> None:
        self.X.append(emb.astype(np.float32))
        self.y.append(int(label))
        self.event_ids.append(event_id)
        self.conf.append(float(conf))
        self.group.append(int(group) if group is not None else len(self.y))
        self._flush()

    def remove_event(self, event_id: int) -> bool:
        for i, e in enumerate(self.event_ids):
            if e == event_id:
                del self.X[i], self.y[i], self.event_ids[i], self.conf[i], self.group[i]
                self._flush()
                return True
        return False

    def counts(self) -> tuple[int, int]:
        y = np.array(self.y, int)
        return int((y == 1).sum()), int((y == 0).sum())

    def _flush(self) -> None:
        if not self.path:
            return
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, X=np.array(self.X, np.float32).reshape(len(self.X), -1),
                 y=np.array(self.y, np.int8),
                 ev=np.array([-1 if e is None else e for e in self.event_ids], np.int64),
                 conf=np.array(self.conf, np.float32), group=np.array(self.group, np.int64))
        tmp.replace(self.path)


def fit_head(buf: FeedbackBuffer, version: int, l2: float = 1.0) -> RescoreHead | None:
    """L2-regularised logistic regression; class-balanced so a flood of false alarms can't dominate."""
    n_pos, n_neg = buf.counts()
    if n_pos < MIN_PER_CLASS or n_neg < MIN_PER_CLASS:
        return None
    from sklearn.linear_model import LogisticRegression
    t0 = time.perf_counter()
    clf = LogisticRegression(C=1.0 / l2, class_weight="balanced", max_iter=2000)
    clf.fit(np.stack(buf.X), np.array(buf.y))
    head = RescoreHead(clf.coef_[0].astype(np.float32), float(clf.intercept_[0]), version, n_pos, n_neg)
    head.fit_ms = (time.perf_counter() - t0) * 1e3  # type: ignore[attr-defined]
    return head
