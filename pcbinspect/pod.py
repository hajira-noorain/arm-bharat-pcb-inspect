"""Probability of detection (POD) versus defect size: the station's measured detection limit.

This is the hit/miss POD method from non-destructive testing (MIL-HDBK-1823A, ASTM E2862), applied
to the whole inspection rig (camera + lighting + detector). Defects of known size are shown to the
station many times; each one is a hit or a miss. From that we report:

  a90      size detected with 90% probability (model estimate)
  a90/95   upper 95% confidence bound on a90: "detects >= a90/95 with 90% POD at 95% confidence"
  demonstrated
           model-free check: the smallest tested size from which every larger size, pooled, has a
           Clopper-Pearson 95% lower bound >= 90% (the classic "29 of 29" demonstration)

The reported limit is the LARGER (more conservative) of a90/95 and the demonstrated size.

Model: logit P(hit | a) = b0 + b1 * ln(a), fitted with Firth's bias-reduced likelihood (Firth 1993;
Heinze & Schemper 2002), which stays finite when small sizes all miss and large sizes all hit
(complete separation, common in clean POD data). The a90/95 bound is from the profile penalised
likelihood (one-sided 95% => chi-square(1) 0.90 quantile = 2.706), the method MIL-HDBK-1823A
recommends over the Wald bound for hit/miss data.

Scope: the limit holds for the rig, lighting and defect presentation it was measured on. A printed
card is not copper; transfer to real boards is a separate, reported check.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import optimize, stats

POD_TARGET = 0.90
CONF = 0.95
_L90 = float(np.log(POD_TARGET / (1 - POD_TARGET)))
_CHI = float(stats.chi2.ppf(2 * CONF - 1, 1))  # one-sided 95% -> 2.706


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def _pen_loglik(b, X, y):
    p = _sigmoid(X @ b)
    ll = np.sum(y * np.log(p + 1e-300) + (1 - y) * np.log(1 - p + 1e-300))
    W = p * (1 - p)
    info = X.T @ (X * W[:, None])
    sign, logdet = np.linalg.slogdet(info)
    return ll + 0.5 * (logdet if sign > 0 else -1e6)


def firth_logistic(X, y, iters: int = 100, tol: float = 1e-9):
    """Firth-penalised logistic regression (Newton with step-halving). Returns coefficients."""
    b = np.zeros(X.shape[1])
    cur = _pen_loglik(b, X, y)
    for _ in range(iters):
        p = _sigmoid(X @ b)
        W = p * (1 - p)
        info = X.T @ (X * W[:, None])
        inv = np.linalg.pinv(info)
        h = W * np.einsum("ij,jk,ik->i", X, inv, X)  # hat-matrix diagonal
        U = X.T @ (y - p + h * (0.5 - p))
        step = inv @ U
        t = 1.0
        while t > 1e-6:
            nb = b + t * step
            nv = _pen_loglik(nb, X, y)
            if nv >= cur - 1e-12:
                break
            t /= 2
        b, prev, cur = nb, cur, nv
        if np.max(np.abs(t * step)) < tol or abs(cur - prev) < tol:
            break
    return b


def clopper_pearson_lower(k: int, n: int, conf: float = CONF) -> float:
    """One-sided lower confidence bound on a binomial proportion."""
    if n == 0 or k == 0:
        return 0.0
    return float(stats.beta.ppf(1 - conf, k, n - k + 1))


@dataclass
class PODResult:
    n: int
    n_hits: int
    table: list = field(default_factory=list)   # [{size_mm, n, hits, pod, lcb95, significantly_low}]
    b0: float | None = None                     # logistic fit on the RISING branch of the curve
    b1: float | None = None
    a50: float | None = None
    a90: float | None = None
    a90_95: float | None = None                 # inf = not bounded inside the tested range
    window: tuple | None = None                 # model-free demonstrated window (lo, hi) of tested sizes
    demonstrated: float | None = None           # window lower end when the window reaches the largest size
    limit_mm: float | None = None               # stated lower limit = max(a90/95, window lo)
    upper_mm: float | None = None               # stated upper end (largest size the claim covers)
    max_tested_mm: float | None = None
    large_defect_drop: bool = False             # some size ABOVE the limit is significantly below 90%
    status: str = ""                            # ok | window | below_range | not_reached | no_trend
    note: str = ""

    def to_dict(self):
        d = asdict(self)
        for k in ("a90_95", "limit_mm", "a90", "a50"):
            if d[k] is not None and not np.isfinite(d[k]):
                d[k] = None
        return d

    def sentence(self, defect: str = "defects") -> str:
        if self.status in ("ok", "below_range"):
            lo = "every tested size from " if self.status == "below_range" else ""
            return (f"Finds {defect} from {lo}{self.limit_mm:.2f} mm up to {self.upper_mm:.2f} mm (largest tested) "
                    f"with at least 90% probability at 95% confidence (n={self.n} trials).")
        if self.status == "window":
            return (f"Finds {defect} only between {self.limit_mm:.2f} and {self.upper_mm:.2f} mm with at least 90% "
                    f"probability at 95% confidence; LARGER ones were missed significantly more often "
                    f"(n={self.n} trials).")
        if self.status == "not_reached":
            return (f"Does NOT reach 90% detection of {defect} at 95% confidence at any tested size "
                    f"(up to {self.max_tested_mm:.2f} mm, n={self.n}).")
        return f"No usable result for {defect} (n={self.n}). {self.note}"


def clopper_pearson_upper(k: int, n: int, conf: float = CONF) -> float:
    if n == 0 or k == n:
        return 1.0
    return float(stats.beta.ppf(conf, k + 1, n - k))


def _significantly_low(r) -> bool:
    """POD at this size is significantly BELOW the target (one-sided 95%)."""
    return clopper_pearson_upper(r["hits"], r["n"]) < POD_TARGET


def _window(table) -> tuple[float, float] | None:
    """Model-free demonstrated window [lo, hi] of tested sizes. Does NOT assume POD rises with size
    (a reference-free detector can miss LARGE defects that look like legitimate features).

    A window qualifies when (a) the hits pooled over its tested sizes have a Clopper-Pearson 95% lower
    bound >= 90%, (b) no tested size inside it is significantly below 90%, and (c) its smallest size
    is itself not significantly below. Among qualifying windows the one reaching the LARGEST size is
    preferred (a "from L upward" claim), then the one reaching the smallest lower end. Claims never
    extend beyond the largest tested size."""
    best = None
    m = len(table)
    for j in range(m - 1, -1, -1):
        for i in range(0, j + 1):
            win = table[i:j + 1]
            if any(_significantly_low(r) for r in win):
                continue
            n = sum(r["n"] for r in win)
            k = sum(r["hits"] for r in win)
            if clopper_pearson_lower(k, n) >= POD_TARGET:
                best = (table[i]["size_mm"], table[j]["size_mm"])
                break
        if best:
            return best
    return None


def _demonstrated(table) -> float | None:
    """Lower end of the demonstrated window, if that window reaches the largest tested size."""
    w = _window(table)
    if w and w[1] == table[-1]["size_mm"]:
        return w[0]
    return None


def _fit_branch(sizes, y):
    """Firth logistic on ln(size) + profile-likelihood a90/95. Returns dict or None (no rising trend)."""
    if len(np.unique(sizes)) < 3 or y.min() == y.max():
        return None
    X = np.column_stack([np.ones_like(sizes), np.log(sizes)])
    b = firth_logistic(X, y)
    if b[1] <= 0:
        return None
    th_hat = (_L90 - b[0]) / b[1]
    lmax = _pen_loglik(b, X, y)
    grid = np.log(np.logspace(-3, 3, 61))

    def prof(theta):
        """Profile penalised log-likelihood at ln(a90) = theta, maximised over the slope b1 > 0
        (grid over ln b1, then local refinement: robust when the optimum is narrow)."""
        f = lambda lb1: -_pen_loglik(np.array([_L90 - np.exp(lb1) * theta, np.exp(lb1)]), X, y)
        vals = [f(v) for v in grid]
        i = int(np.argmin(vals))
        r = optimize.minimize_scalar(f, bounds=(grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]),
                                     method="bounded", options={"xatol": 1e-8})
        return -min(r.fun, vals[i])

    hi = max(th_hat, float(np.log(sizes.max()))) + 1.0  # search up to e x the largest tested size
    if th_hat >= hi - 1e-9:
        a9095 = float("inf")
    else:
        lmax = max(lmax, prof(th_hat))
        g = lambda theta: 2 * (lmax - prof(theta)) - _CHI
        a9095 = float("inf") if g(hi) < 0 else float(np.exp(optimize.brentq(g, th_hat, hi, xtol=1e-6)))
    return {"b0": float(b[0]), "b1": float(b[1]), "a50": float(np.exp(-b[0] / b[1])),
            "a90": float(np.exp(th_hat)), "a90_95": a9095}


def fit_pod(sizes_mm, hits) -> PODResult:
    sizes = np.asarray(sizes_mm, float)
    y = np.asarray(hits, float)
    if sizes.size == 0 or np.any(sizes <= 0):
        raise ValueError("sizes must be positive and non-empty")
    table = []
    for s in np.unique(sizes):
        m = sizes == s
        n, k = int(m.sum()), int(y[m].sum())
        r = {"size_mm": float(s), "n": n, "hits": k, "pod": k / n, "lcb95": clopper_pearson_lower(k, n)}
        r["significantly_low"] = _significantly_low(r)
        table.append(r)
    res = PODResult(n=int(sizes.size), n_hits=int(y.sum()), table=table, max_tested_mm=float(sizes.max()))
    res.window = _window(table)
    res.demonstrated = _demonstrated(table)

    # the parametric curve is fitted on the RISING branch only (sizes up to the best observed POD):
    # POD-vs-size models assume detection improves with size, which need not hold above the peak
    peak = max(range(len(table)), key=lambda i: (table[i]["pod"], -i))
    keep = sizes <= table[peak]["size_mm"]
    fit = _fit_branch(sizes[keep], y[keep])
    if fit:
        res.b0, res.b1, res.a50, res.a90, res.a90_95 = fit["b0"], fit["b1"], fit["a50"], fit["a90"], fit["a90_95"]

    if res.window is None:
        res.status = "not_reached" if len(table) >= 2 else "no_trend"
        if len(table) < 2:
            res.note = "a single tested size cannot show a size limit"
        return res
    lo, hi = res.window
    if fit and np.isfinite(fit["a90_95"]):
        lo = max(lo, fit["a90_95"]) if fit["a90_95"] <= hi else None
    if lo is None:
        res.status, res.note = "not_reached", "the model bound lies above every demonstrated size"
        return res
    res.limit_mm, res.upper_mm = float(lo), float(hi)
    res.large_defect_drop = any(r["significantly_low"] for r in table if r["size_mm"] > hi)
    if hi < table[-1]["size_mm"]:
        res.status = "window"
    elif lo <= table[0]["size_mm"] + 1e-12:
        res.status = "below_range"
    else:
        res.status = "ok"
    if not fit:
        res.note = "model-free result only (no rising trend to fit)"
    return res


def required_trials(pod: float = POD_TARGET, conf: float = CONF) -> int:
    """Smallest n such that n/n hits demonstrates POD >= pod at confidence conf (29 for 90/95)."""
    n = 1
    while clopper_pearson_lower(n, n, conf) < pod:
        n += 1
    return n
