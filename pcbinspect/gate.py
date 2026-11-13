"""Conformal PASS / FAIL / REVIEW gate with a finite-sample miss-rate guarantee PER DEFECT.

Board score s(x) = strongest defect evidence on the board (max detector confidence / normalised
anomaly score); a defect is SHOWN when its evidence is >= lam. From a held-out calibration set we
pick two thresholds:

  lam  (miss-rate threshold): a defect is shown iff its score >= lam; boards with s < lam PASS.
       The calibration "defective scores" are DEFECT scores: for each defective calibration board,
       ONE of its ground-truth defects chosen at random, scored by the strongest detection covering
       it (0 if none). One per board keeps the scores exchangeable (boards are the sampling unit).
       lam = k-th smallest of these n scores, k = floor(alpha*(n+1)). For a randomly chosen defect
       on a new board exchangeable with the calibration boards,
           P(defect not shown) = P(score < lam) <= k/(n+1) <= alpha     (split-conformal argument)
       and a defective board is then caught with probability >= 1 - alpha as well (it is caught
       whenever any of its defects is shown).
       History: the first version calibrated on BOARD max scores. That guaranteed the board verdict
       but, on DeepPCB at alpha 5%, showed only 34% of individual defects (boards carry several
       defects and the max is high). The detection-limit "eye test" exposed it; see docs.

  tau  (false-reject threshold): boards with s >= max(lam, tau) are FAILED automatically.
       Chosen from m calibration GOOD boards so that P(s_good >= tau) <= beta.

  Everything in [lam, max(lam, tau)) is sent to a human: REVIEW.

The guarantee is marginal and only holds while test boards look like the calibration boards
(same product family, camera, lighting). A new design or a new rig needs re-calibration — the
app enforces this by keeping one calibration per design.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist


def defect_level_scores(per_board: list[list[float]], seed: int = 0) -> list[float]:
    """One randomly chosen defect score per defective board (boards with no labelled defect skipped)."""
    rng = np.random.default_rng(seed)
    return [float(s[rng.integers(len(s))]) for s in per_board if len(s)]


def miss_threshold(defective_scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(defective_scores, dtype=float))
    n = len(s)
    if n == 0:
        raise ValueError("need at least one defective calibration board")
    k = math.floor(alpha * (n + 1))
    if k < 1:
        return -math.inf  # too few calibration boards for this alpha: never auto-pass
    return float(s[k - 1])


def false_reject_threshold(good_scores: np.ndarray, beta: float) -> float:
    s = np.sort(np.asarray(good_scores, dtype=float))
    m = len(s)
    if m == 0:
        return math.inf
    # smallest tau with P(s_good >= tau) <= beta:  tau just above the ceil((1-beta)(m+1))-th score
    k = math.ceil((1 - beta) * (m + 1))
    if k > m:
        return math.inf  # too few good boards: never auto-fail, send to review instead
    return float(np.nextafter(s[k - 1], math.inf))


@dataclass
class GateThresholds:
    alpha: float
    beta: float
    lam: float
    tau: float
    n_defective: int
    n_good: int

    def verdict(self, score: float) -> str:
        if score < self.lam:
            return "PASS"
        if score >= max(self.lam, self.tau):
            return "FAIL"
        return "REVIEW"


class ConformalGate:
    """Holds calibration scores for one design and produces thresholds for any alpha."""

    def __init__(self, defective_scores, good_scores, design: str = "default"):
        self.design = design
        self.defective = np.asarray(defective_scores, dtype=float)
        self.good = np.asarray(good_scores, dtype=float)

    def thresholds(self, alpha: float, beta: float = 0.01) -> GateThresholds:
        return GateThresholds(
            alpha=alpha,
            beta=beta,
            lam=miss_threshold(self.defective, alpha),
            tau=false_reject_threshold(self.good, beta),
            n_defective=len(self.defective),
            n_good=len(self.good),
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "design": self.design,
            "defective_scores": self.defective.tolist(),
            "good_scores": self.good.tolist(),
        }))

    @classmethod
    def load(cls, path: Path) -> "ConformalGate":
        d = json.loads(Path(path).read_text())
        return cls(d["defective_scores"], d["good_scores"], d.get("design", "default"))


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval for a rate k/n."""
    if n == 0:
        return (0.0, 1.0)
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a, k + 1, n - k))
    return lo, hi


def random_defect_miss(found_per_board: list[list[bool]]) -> float | None:
    """The quantity the per-defect gate guarantees: the chance that ONE randomly chosen defect on a new
    defective board is missed, i.e. the mean over boards of the share of that board's defects missed.
    (Pooling all defects instead weights boards by how many defects they carry.)"""
    rows = [f for f in found_per_board if f]
    return float(np.mean([1 - sum(f) / len(f) for f in rows])) if rows else None


def random_defect_miss_ci(found_per_board: list[list[bool]], n_boot: int = 4000, seed: int = 0,
                          level: float = 0.95) -> tuple[float, float] | None:
    """Percentile bootstrap over boards (the unit that was sampled) for random_defect_miss."""
    per = np.array([1 - sum(f) / len(f) for f in found_per_board if f], dtype=float)
    if per.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = per[rng.integers(0, per.size, size=(n_boot, per.size))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def evaluate(th: GateThresholds, defective_scores, good_scores) -> dict:
    """Empirical behaviour of a gate on a TEST set (never the calibration set)."""
    d = np.asarray(defective_scores, dtype=float)
    g = np.asarray(good_scores, dtype=float)
    vd = [th.verdict(s) for s in d]
    vg = [th.verdict(s) for s in g]
    misses = sum(v == "PASS" for v in vd)
    false_rejects = sum(v == "FAIL" for v in vg)
    review = sum(v == "REVIEW" for v in vd) + sum(v == "REVIEW" for v in vg)
    total = len(d) + len(g)
    return {
        "alpha": th.alpha,
        "beta": th.beta,
        "miss_rate": misses / max(len(d), 1),
        "miss_rate_ci95": clopper_pearson(misses, len(d)),
        "false_reject_rate": false_rejects / max(len(g), 1),
        "false_reject_rate_ci95": clopper_pearson(false_rejects, len(g)),
        "review_rate": review / max(total, 1),
        "auto_decided_rate": 1 - review / max(total, 1),
        "n_defective": len(d),
        "n_good": len(g),
    }
