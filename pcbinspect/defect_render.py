"""Draw bare-board defects of an exact physical size onto a clean (defect-free) board image.

Used to measure the station's detection limit: in silico on held-out clean boards, and to lay out
the printed "eye test" card. Every defect is checked after drawing so the label is true:
  open             a gap of length s (mm) cut across a trace; must split the conductor in two
  short            a copper bridge across a GAP OF WIDTH s (+-15%) joining two DIFFERENT conductors; must
                   merge them. For a short the gap is the size that matters: any bridge is a functional
                   failure, and how visible it is depends on how narrow a gap the optics can resolve.
                   Bridge width = clamp(s/2, SHORT_BRIDGE_MIN_MM, SHORT_BRIDGE_MAX_MM).
  pin_hole         a hole of diameter s inside copper (fully enclosed)
  spurious_copper  an isolated copper dot of diameter s on bare substrate (touches nothing)
  mouse_bite       a semicircular bite of diameter s out of a conductor edge (does not sever it)
  spur             a protrusion of width s, length 1.5 s, from a conductor edge (touches nothing)

Images are binary-looking scans with copper DARK (as in DeepPCB). The size s is the defect's
characteristic dimension above, the quantity a workmanship criterion would limit.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

TYPES = ("open", "short", "pin_hole", "spurious_copper", "mouse_bite", "spur")
# how each type reads in a sentence ("Finds <noun> from 0.2 mm ..."); shorts are sized by the gap they bridge
NOUN = {"open": "opens", "short": "shorts across gaps", "pin_hole": "pin holes",
        "spurious_copper": "spurious copper spots", "mouse_bite": "mouse bites", "spur": "spurs"}
BOX_PAD_PX = 8  # ground-truth box padding around the changed pixels (DeepPCB-like boxes)
SHORT_BRIDGE_MIN_MM, SHORT_BRIDGE_MAX_MM = 0.1, 0.3
SHORT_GAP_TOL = 0.15


@dataclass
class Placed:
    cls: str
    size_mm: float
    center: tuple[int, int]
    bbox: tuple[int, int, int, int]


def copper_mask(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return g < 128


def to_image(cu: np.ndarray) -> np.ndarray:
    g = np.where(cu, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _dist(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)


def _across_direction(mask: np.ndarray, p, r: int) -> np.ndarray:
    """Unit vector across the local structure at p (dominant edge-normal via the structure tensor)."""
    x, y = p
    h, w = mask.shape
    x0, x1, y0, y1 = max(0, x - r), min(w, x + r + 1), max(0, y - r), min(h, y + r + 1)
    win = cv2.GaussianBlur(mask[y0:y1, x0:x1].astype(np.float32), (0, 0), 1.5)
    gx, gy = cv2.Sobel(win, cv2.CV_32F, 1, 0), cv2.Sobel(win, cv2.CV_32F, 0, 1)
    T = np.array([[np.sum(gx * gx), np.sum(gx * gy)], [np.sum(gx * gy), np.sum(gy * gy)]])
    if np.trace(T) < 1e-6:
        return np.array([1.0, 0.0])
    vals, vecs = np.linalg.eigh(T)
    v = vecs[:, 1]
    return v / (np.linalg.norm(v) + 1e-12)


def _rot_rect(img, center, along, across_len, along_len, value):
    """Fill a rectangle centred at `center`, `across_len` long in direction `along`, `along_len` wide."""
    d = along / np.linalg.norm(along)
    n = np.array([-d[1], d[0]])
    c = np.asarray(center, float)
    pts = np.array([c + d * across_len / 2 + n * along_len / 2, c + d * across_len / 2 - n * along_len / 2,
                    c - d * across_len / 2 - n * along_len / 2, c - d * across_len / 2 + n * along_len / 2])
    cv2.fillPoly(img, [np.round(pts).astype(np.int32)], int(value))


def _n_components_local(mask, box):
    x0, y0, x1, y1 = box
    n, _ = cv2.connectedComponents(mask[y0:y1, x0:x1].astype(np.uint8), connectivity=8)
    return n - 1


def _box(changed: np.ndarray, shape):
    ys, xs = np.nonzero(changed)
    if not len(xs):
        return None
    h, w = shape
    return (max(0, int(xs.min()) - BOX_PAD_PX), max(0, int(ys.min()) - BOX_PAD_PX),
            min(w - 1, int(xs.max()) + BOX_PAD_PX), min(h - 1, int(ys.max()) + BOX_PAD_PX))


def _free(occupied: np.ndarray, p, r: int) -> bool:
    x, y = p
    h, w = occupied.shape
    if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
        return False
    return not occupied[y - r:y + r + 1, x - r:x + r + 1].any()


def place(cu: np.ndarray, cls: str, size_mm: float, px_per_mm: float, rng: np.random.Generator,
          occupied: np.ndarray, tries: int = 300) -> Placed | None:
    """Try to draw one defect in place (modifies `cu` and `occupied`). Returns None if no valid spot."""
    s = max(1.0, size_mm * px_per_mm)
    h, w = cu.shape
    sep = int(max(40, 3 * s))  # keep defects apart so each is judged on its own
    d_cu, d_sub = _dist(cu), _dist(~cu)
    if cls in ("pin_hole",):
        cand = np.argwhere(d_cu >= s / 2 + 4)
    elif cls == "spurious_copper":
        cand = np.argwhere(d_sub >= s / 2 + max(6, s))
    elif cls == "open":
        # trace ridge: copper whose distance-to-edge is a local maximum and not too wide
        ridge = (d_cu >= 2) & (d_cu <= 0.45 * px_per_mm) & (d_cu >= cv2.dilate(d_cu, np.ones((3, 3))) - 1e-3)
        cand = np.argwhere(ridge)
    elif cls == "short":
        # medial line of the gaps whose width (2 x distance to copper) is within +-15% of s
        half = s / 2
        ridge = (d_sub >= half * (1 - SHORT_GAP_TOL)) & (d_sub <= half * (1 + SHORT_GAP_TOL)) & \
                (d_sub >= cv2.dilate(d_sub, np.ones((3, 3))) - 1e-3)
        cand = np.argwhere(ridge)
    elif cls in ("mouse_bite", "spur"):
        edge = cu & (cv2.dilate((~cu).astype(np.uint8), np.ones((3, 3))) > 0)
        cand = np.argwhere(edge)
    else:
        raise ValueError(cls)
    if not len(cand):
        return None
    for _ in range(tries):
        y, x = cand[rng.integers(len(cand))]
        p = (int(x), int(y))
        if not _free(occupied, p, sep // 2):
            continue
        before = cu.copy()
        r_loc = int(max(12, 2 * s + 10))
        loc = (max(0, x - r_loc), max(0, y - r_loc), min(w, x + r_loc + 1), min(h, y + r_loc + 1))
        if cls == "pin_hole":
            cv2.circle(cu.view(np.uint8), p, int(round(s / 2)), 0, -1)
            ok = d_cu[y, x] >= s / 2 + 2
        elif cls == "spurious_copper":
            cv2.circle(cu.view(np.uint8), p, int(round(s / 2)), 1, -1)
            ok = True
        elif cls == "open":
            hw = float(d_cu[y, x])
            across = _across_direction(cu, p, int(max(6, 2 * hw)))
            n0, _ = cv2.connectedComponents(cu.astype(np.uint8), connectivity=8)
            _rot_rect(cu.view(np.uint8), p, across, 2 * hw + 6, s, 0)
            n1, _ = cv2.connectedComponents(cu.astype(np.uint8), connectivity=8)
            ok = n1 > n0  # the conductor is really severed (electrically), not just nicked
        elif cls == "short":
            g = float(d_sub[y, x])
            across = _across_direction(~cu, p, int(max(6, 2 * g)))
            lab_n, lab = cv2.connectedComponents(cu.astype(np.uint8), connectivity=8)
            a = np.round(np.asarray(p) + across * (g + 2)).astype(int)
            b = np.round(np.asarray(p) - across * (g + 2)).astype(int)
            inside = all(0 <= q[0] < w and 0 <= q[1] < h for q in (a, b))
            if not inside or not (cu[a[1], a[0]] and cu[b[1], b[0]]) or lab[a[1], a[0]] == lab[b[1], b[0]]:
                continue
            bridge = float(np.clip(s / 2, SHORT_BRIDGE_MIN_MM * px_per_mm, SHORT_BRIDGE_MAX_MM * px_per_mm))
            _rot_rect(cu.view(np.uint8), p, across, 2 * g + 4, max(2.0, bridge), 1)
            _, lab2 = cv2.connectedComponents(cu.astype(np.uint8), connectivity=8)
            ok = lab2[a[1], a[0]] == lab2[b[1], b[0]]  # the two conductors are now joined
        elif cls == "mouse_bite":
            # needs enough copper behind the edge that the bite does not sever the conductor
            n0 = _n_components_local(cu, loc)
            cv2.circle(cu.view(np.uint8), p, int(round(s / 2)), 0, -1)
            ok = _n_components_local(cu, loc) == n0 and (before & ~cu).sum() >= 0.25 * np.pi * (s / 2) ** 2
        else:  # spur
            out = _across_direction(cu, p, 6)
            # point outward (towards substrate)
            q = np.round(np.asarray(p) + out * 3).astype(int)
            if 0 <= q[0] < w and 0 <= q[1] < h and cu[q[1], q[0]]:
                out = -out
            L = 1.5 * s
            tip = np.asarray(p) + out * (L + s)
            tq = np.round(tip).astype(int)
            if not (0 <= tq[0] < w and 0 <= tq[1] < h) or d_sub[tq[1], tq[0]] < s:
                continue
            lab_n, lab = cv2.connectedComponents(cu.astype(np.uint8), connectivity=8)
            _rot_rect(cu.view(np.uint8), np.asarray(p) + out * L / 2, out, L, s, 1)
            n_after, _ = cv2.connectedComponents(cu.astype(np.uint8), connectivity=8)
            ok = n_after == lab_n and (cu & ~before).sum() >= 0.5 * L * s  # attached, nothing joined
        changed = before ^ cu
        if not ok or not changed.any():
            cu[...] = before
            continue
        box = _box(changed, cu.shape)
        x0, y0, x1, y1 = box
        occupied[max(0, y0 - sep // 2):y1 + sep // 2, max(0, x0 - sep // 2):x1 + sep // 2] = True
        return Placed(cls, float(size_mm), p, box)
    return None


def render_board(clean_bgr: np.ndarray, specs, px_per_mm: float, rng: np.random.Generator):
    """specs: list of (cls, size_mm). Returns (image, placed list). Specs that do not fit are skipped."""
    cu = copper_mask(clean_bgr)
    occupied = np.zeros(cu.shape, bool)
    placed = []
    for cls, size in specs:
        r = place(cu, cls, size, px_per_mm, rng, occupied)
        if r is not None:
            placed.append(r)
    return to_image(cu), placed
