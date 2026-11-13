"""Teach a bare board on the station: golden image + a gate calibrated on the spot.

A new design has no defective boards to calibrate on. So the station makes its own, the same way the
eye-test card does: it draws defects of KNOWN type and size onto photographs of the good board, in photo
space (local copper / laminate colour, softened like the camera's own edges), and judges them exactly as
it will judge boards (golden comparison first, detector on the tiles that differ, fusion).

  golden        the first good capture
  defective     windows of the other good captures with planted defects; one random planted defect per
                window is a calibration score (per-defect split-conformal, see gate.py)
  good          the other good captures, plus re-captures of them (small shift/rotation, gain, noise)
  detection limit  hit/miss per planted size at the 5% threshold -> a90/95 per defect type (pod.py),
                stored like an eye-test result for this board

What this does and does not show: the miss rate holds for PLANTED defects of the stated size range on
this board, camera and lighting. Real defects can look different (etch residue, scratches); the stored
result says "calibrated on planted defects" wherever it is shown.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .defect_render import NOUN, TYPES, place
from .gate import ConformalGate, defect_level_scores
from .golden import MIN_ALIGN_CC, align, copper_mask
from .pod import fit_pod

WINDOW = 960            # planted-defect trials run on windows of the capture (scores are local)
PER_WINDOW = 4          # planted defects per window
MIN_COPPER, MAX_COPPER = 0.03, 0.85


def _local_colour(photo: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Per-pixel mean colour of the pixels in `mask` nearby (normalised blur): what copper, or bare
    laminate, looks like around here."""
    m = mask.astype(np.float32)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    num = cv2.GaussianBlur(photo.astype(np.float32) * m[..., None], (0, 0), sigma)
    glob = photo[mask].mean(0) if mask.any() else np.array([127.0, 127.0, 127.0])
    out = np.where(den[..., None] > 0.02, num / np.maximum(den[..., None], 1e-6), glob)
    return out.astype(np.float32)


def edge_sigma(photo: np.ndarray, cu: np.ndarray) -> float:
    """Blur (px) of the capture's own copper edges, from how wide the grey transition is."""
    g = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo, hi = float(np.median(g[cu])) if cu.any() else 0.0, float(np.median(g[~cu])) if (~cu).any() else 255.0
    if abs(hi - lo) < 10:
        return 1.0
    band = cv2.dilate(cu.astype(np.uint8), np.ones((9, 9), np.uint8)) - cv2.erode(cu.astype(np.uint8), np.ones((9, 9), np.uint8))
    f = np.abs((g - lo) / (hi - lo))
    mid = ((f > 0.15) & (f < 0.85) & (band > 0)).sum()
    edge_len = max(1, int((cv2.morphologyEx(cu.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0).sum()))
    width = mid / edge_len                       # px of transition per px of edge
    return float(np.clip(width / 2.0, 0.6, 3.0))


def plant_on_photo(photo: np.ndarray, specs, px_per_mm: float, rng: np.random.Generator):
    """Draw defects (cls, size_mm) onto a photograph of a good board. Returns (image, placed list);
    specs with no valid spot are skipped. Sizes are true sizes in the copper mask, before softening."""
    cu0 = copper_mask(photo) > 0
    frac = float(cu0.mean())
    if not MIN_COPPER <= frac <= MAX_COPPER:
        return photo.copy(), []
    cu = cu0.copy()
    occupied = np.zeros(cu.shape, bool)
    placed = []
    for cls, size in specs:
        r = place(cu, cls, size, px_per_mm, rng, occupied)
        if r is not None:
            placed.append(r)
    if not placed:
        return photo.copy(), []
    add, rem = cu & ~cu0, cu0 & ~cu
    sig_c = max(4.0, 0.25 * px_per_mm)
    out = photo.astype(np.float32)
    if add.any():
        out[add] = _local_colour(photo, cu0, sig_c)[add]
    if rem.any():
        out[rem] = _local_colour(photo, ~cu0, sig_c)[rem]
    # soften the new edges like the camera softens real ones, only around what changed
    s = edge_sigma(photo, cu0)
    near = cv2.dilate((add | rem).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    soft = cv2.GaussianBlur(out, (0, 0), s)
    w = cv2.GaussianBlur(near.astype(np.float32), (0, 0), 1.5)[..., None]
    out = out * (1 - w) + soft * w
    noise = float(np.std(photo.astype(np.float32) - cv2.GaussianBlur(photo, (0, 0), 1.2).astype(np.float32)))
    out[near] += rng.normal(0, min(noise, 6.0), out[near].shape)
    return np.clip(out, 0, 255).astype(np.uint8), placed


def recapture(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A plausible second photograph of the same board: put down slightly differently, slightly
    different exposure, fresh sensor noise."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-0.4, 0.4), 1.0)
    M[:, 2] += rng.uniform(-0.004, 0.004, 2) * w
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    out = out * rng.uniform(0.96, 1.04) + rng.normal(0, 2.0, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _covers(bbox, c) -> bool:
    return bbox[0] - 4 <= c[0] <= bbox[2] + 4 and bbox[1] - 4 <= c[1] <= bbox[3] + 4


@dataclass
class TeachResult:
    gate: ConformalGate
    trials: list[dict]           # {type, size_mm, conf, covered}
    good_scores: list[float]
    limits: dict                 # per type: PODResult dict + sentence, at the alpha=0.05 threshold
    lam05: float
    n_windows: int
    px_per_mm: float
    sizes_mm: tuple[float, float]
    notes: list[str]

    def summary(self) -> dict:
        return {"lambda": self.lam05, "types": self.limits, "alpha": 0.05,
                "evidence": "detector+golden", "calibrated_on": "planted defects on the taught board",
                "n_windows": self.n_windows, "n_good": len(self.good_scores),
                "sizes_mm": list(self.sizes_mm), "px_per_mm": self.px_per_mm, "notes": self.notes,
                "false_calls_per_coupon": None}


def calibrate_bare(score_fn, golden: np.ndarray, goods: list[np.ndarray], px_per_mm: float,
                   rng: np.random.Generator, n_windows: int = 120, n_good: int = 24,
                   sizes_mm: tuple[float, float] = (0.2, 1.0), progress=None) -> TeachResult:
    """score_fn(test_bgr, golden_bgr, aligned) -> list of (bbox, conf): the station's own evidence for a
    capture (golden-first). `goods` are good captures other than the golden one (>= 1)."""
    if not goods:
        raise ValueError("need at least one good capture besides the golden one")
    notes = []
    aligned = []                                   # (capture, golden warped onto it)
    for g in goods:
        ga, cc = align(g, golden)
        if cc >= MIN_ALIGN_CC:
            aligned.append((g, ga))
        else:
            notes.append(f"one good capture did not align with the golden image (cc {cc:.2f}) and was left out")
    if not aligned:
        raise ValueError("no good capture aligns with the golden image: same board, same position?")

    lo, hi = sizes_mm
    levels = np.round(np.geomspace(lo, hi, 5), 3)       # discrete sizes, so each has enough trials
    per_board, trials = [], []
    tries = 0
    while len(per_board) < n_windows and tries < 6 * n_windows:
        tries += 1
        cap, ga = aligned[int(rng.integers(len(aligned)))]
        h, w = cap.shape[:2]
        ws = min(WINDOW, h, w)
        x0, y0 = int(rng.integers(0, w - ws + 1)), int(rng.integers(0, h - ws + 1))
        win, gwin = cap[y0:y0 + ws, x0:x0 + ws], ga[y0:y0 + ws, x0:x0 + ws]
        specs = [(TYPES[int(rng.integers(len(TYPES)))], float(levels[int(rng.integers(len(levels)))]))
                 for _ in range(PER_WINDOW)]
        img, placed = plant_on_photo(win, specs, px_per_mm, rng)
        if not placed:
            continue
        cands = score_fn(img, gwin, True)
        scores = []
        for p in placed:
            best = max((c for b, c in cands if _covers(b, p.center)), default=None)
            scores.append(float(best) if best is not None else 0.0)
            trials.append({"type": p.cls, "size_mm": float(p.size_mm), "conf": scores[-1], "covered": best is not None})
        per_board.append(scores)
        if progress:
            progress(len(per_board) / n_windows * 0.7)
    if len(per_board) < 20:
        raise ValueError("could not plant enough defects on this board (too little visible copper pattern?)")

    good_scores = []
    for i in range(n_good):
        cap = goods[i % len(goods)]
        shot = cap if i < len(goods) else recapture(cap, rng)
        good_scores.append(max((c for _, c in score_fn(shot, golden, False)), default=0.0))
        if progress:
            progress(0.7 + 0.3 * (i + 1) / n_good)

    gate = ConformalGate(defect_level_scores(per_board, seed=int(rng.integers(1 << 30))), good_scores, "taught")
    lam = gate.thresholds(0.05, 0.05).lam
    limits = {}
    for cls in TYPES:
        tr = [t for t in trials if t["type"] == cls]
        if len(tr) < 12:
            continue
        r = fit_pod(np.array([t["size_mm"] for t in tr]), np.array([t["covered"] and t["conf"] >= lam for t in tr], int))
        d = r.to_dict()
        d["sentence"] = r.sentence(NOUN[cls])
        limits[cls] = d
    return TeachResult(gate, trials, good_scores, limits, float(lam), len(per_board), px_per_mm, sizes_mm, notes)
