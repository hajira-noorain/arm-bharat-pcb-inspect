"""Statistical process control over the inspection log: p-chart, Pareto, and drift alerts.

The RULES raise alerts (deterministic, testable); Llama only words them for the operator.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .kb import KnowledgeBase


@dataclass
class DriftAlert:
    design: str
    cls: str
    recent_count: int
    recent_n: int
    baseline_rate: float
    ratio: float
    process_step: str
    message: str


def p_chart(fail_flags: list[int], subgroup: int = 20) -> dict:
    """Proportion defective per subgroup of consecutive boards, with 3-sigma limits."""
    groups = [fail_flags[i:i + subgroup] for i in range(0, len(fail_flags), subgroup)]
    groups = [g for g in groups if len(g) == subgroup]
    if not groups:
        return {"p": [], "pbar": None, "ucl": None, "lcl": None, "out_of_control": []}
    p = [sum(g) / subgroup for g in groups]
    pbar = sum(p) / len(p)
    s = math.sqrt(max(pbar * (1 - pbar), 1e-12) / subgroup)
    ucl, lcl = min(1.0, pbar + 3 * s), max(0.0, pbar - 3 * s)
    return {"p": p, "pbar": pbar, "ucl": ucl, "lcl": lcl,
            "out_of_control": [i for i, x in enumerate(p) if x > ucl or x < lcl]}


def pareto(defect_classes: list[str]) -> list[tuple[str, int, float]]:
    c = Counter(defect_classes)
    total = sum(c.values()) or 1
    out, cum = [], 0
    for cls, n in c.most_common():
        cum += n
        out.append((cls, n, cum / total))
    return out


def drift_alerts(per_board_classes: list[set[str]], design: str, kb: KnowledgeBase,
                 window: int = 20, min_baseline_boards: int = 40, ratio_thr: float = 3.0,
                 min_count: int = 3) -> list[DriftAlert]:
    """per_board_classes: chronological list, one set of defect classes per inspected board.

    A class drifts when its rate in the last `window` boards is >= ratio_thr x its baseline rate
    (all earlier boards, Laplace-smoothed) and it occurred at least `min_count` times.
    """
    if len(per_board_classes) < window + min_baseline_boards:
        return []
    recent = per_board_classes[-window:]
    base = per_board_classes[:-window]
    alerts = []
    classes = set().union(*recent) if recent else set()
    for cls in sorted(classes):
        rc = sum(cls in b for b in recent)
        bc = sum(cls in b for b in base)
        base_rate = (bc + 1) / (len(base) + 2)
        rate = rc / window
        ratio = rate / base_rate
        if rc >= min_count and ratio >= ratio_thr:
            step_key = kb.entry(cls).get("process_step", "")
            step = kb.process_steps.get(step_key, step_key)
            name = kb.display_name(cls).lower()
            msg = (f"{name} on {rc} of the last {window} boards ({ratio:.1f}x the usual rate). "
                   f"Check {step.split(':')[0].lower()}.")
            alerts.append(DriftAlert(design, cls, rc, window, base_rate, ratio, step, msg))
    return alerts
