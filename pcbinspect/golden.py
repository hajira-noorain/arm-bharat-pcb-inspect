"""Golden-board measurement: estimate what a defect does to the copper, in millimetres.

A detection box is drawn loosely around a defect, so its size is not the defect's size. Like
golden-board AOI, we compare with a known-good image of the same design:

  1. global alignment (ECC, affine); the correlation is checked, and below MIN_ALIGN_CC no
     measurement is made at all (the defect is graded by type and flagged)
  2. per-defect refinement: phase correlation of the crop at full resolution removes the residual
     shift the global affine leaves (lens distortion, warpage)
  3. copper masks; inside the box, golden-minus-board is missing copper, board-minus-golden is extra
     copper; differences thinner than MIN_THICK_PX are registration slivers, not defects
  4a. missing copper: depth = short side of the region's min-area rectangle; local conductor width
      = typical ridge (centre-line) width of the conductor next to the nick, not the widest copper
      nearby (a pad would understate the damage)
  4b. extra copper: remaining gap = shortest distance to a DIFFERENT golden conductor
  5. every value carries +/-1 pixel; a class limit inside that band is reported 'uncertain'

Validated so far on binarised scans and bare-copper scans. NOT yet validated on solder-masked
boards under in-line lighting: there, copper under the mask is a faint ridge and glare dominates.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MISSING_TYPES = {"mouse_bite", "pin_hole", "open", "missing_pad"}
EXTRA_TYPES = {"spur", "spurious_copper", "short"}
MIN_ALIGN_CC = 0.85
MIN_THICK_PX = 2.0
MAX_REFINE_SHIFT_PX = 6.0


@dataclass
class Measurement:
    kind: str            # 'width' | 'spacing'
    defect_mm: float     # nick depth (width) or remaining gap (spacing)
    ref_mm: float        # local conductor width (width) or minimum spacing (spacing)
    pct: float           # reduction in percent (100 = cut through / bridged)
    px_mm: float         # one pixel in mm: the measurement uncertainty
    method: str = "golden-diff"

    @property
    def unc_pct(self) -> float:
        return 100.0 * self.px_mm / max(self.ref_mm, 1e-9)

    def decide(self, limit_pct: float) -> str:
        """'accept' | 'reject' | 'uncertain' against a reduction limit (100 = functional only)."""
        if self.pct >= 100.0:
            return "reject"                     # cut through / bridged: no ambiguity
        if abs(self.pct - limit_pct) <= self.unc_pct:
            return "uncertain"
        return "reject" if self.pct > limit_pct else "accept"

    def text(self) -> str:
        u = f" (+/-{self.px_mm:.3f} mm)"
        if self.kind == "width":
            if self.pct >= 100:
                return f"cuts the full {self.ref_mm:.2f} mm conductor"
            return (f"{self.defect_mm:.3f} mm nick in a {self.ref_mm:.2f} mm conductor{u}: "
                    f"{self.pct:.0f}% of the width lost")
        if self.pct >= 100:
            return "bridges the gap to the next conductor"
        return (f"leaves a {self.defect_mm:.3f} mm gap against a {self.ref_mm:.2f} mm minimum{u}: "
                f"{self.pct:.0f}% of the spacing lost")


def copper_mask(bgr: np.ndarray) -> np.ndarray:
    """Conductor layer as a binary mask: Otsu on the LAB channel with the most contrast.
    Works on binarised scans and bare copper on laminate; not validated under solder mask."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    best, best_sep = None, -1.0
    for ch in (lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]):
        c8 = cv2.GaussianBlur(ch, (3, 3), 0)
        _, m = cv2.threshold(c8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        a, b = c8[m > 0], c8[m == 0]
        if len(a) < 10 or len(b) < 10:
            continue
        sep = abs(float(a.mean()) - float(b.mean()))  # absolute contrast (see golden_diff.consistent_masks)
        if sep > best_sep:
            best, best_sep = m, sep
    m = best if best is not None else np.zeros(bgr.shape[:2], np.uint8)
    if (m > 0).mean() > 0.5:                    # conductor is the minority phase on most boards
        m = 255 - m
    return m


NEAR_CC = 0.9        # fine alignment from the identity is accepted at this correlation


def _ecc(a, b, warp, motion, iters, gauss):
    try:
        return cv2.findTransformECC(a, b, warp, motion,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-5), None, gauss)
    except cv2.error:
        return None


def align_warp(test: np.ndarray, golden: np.ndarray, max_side: int = 1000) -> tuple[np.ndarray | None, float]:
    """Find where `golden` sits in `test`: the intensity-based search first (align_warp_ecc); if that would be
    refused, a second try that starts from matched image features, which copes with a board put down far off or
    turned well beyond the few degrees the first search covers. The better of the two is returned.
    On 80 good VisA photos (20 per design, training split) the second try lifted the worst cases of one design
    from a correlation of 0.38 to 0.73 and changed nothing where the first already worked."""
    warp, cc = align_warp_ecc(test, golden, max_side)
    if warp is not None and cc >= MIN_ALIGN_CC:
        return warp, cc
    w2, cc2 = align_warp_features(test, golden, max_side)
    return (w2, cc2) if w2 is not None and cc2 > cc else (warp, cc)


def align_warp_features(test: np.ndarray, golden: np.ndarray, max_side: int = 1000) -> tuple[np.ndarray | None, float]:
    """ORB feature matches -> a rigid-plus-scale transform by RANSAC -> affine ECC refinement from there.
    Same return convention as align_warp_ecc."""
    if test.shape[:2] != golden.shape[:2]:
        golden = cv2.resize(golden, (test.shape[1], test.shape[0]))
    g1, g2 = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY), cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
    s = min(1.0, max_side / max(g1.shape))
    a = cv2.resize(g1, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    b = cv2.resize(g2, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create(3000)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return None, 0.0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 12:
        return None, 0.0
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or inl is None or int(inl.sum()) < 10:
        return None, 0.0
    r = _ecc(a.astype(np.float32), b.astype(np.float32), M.astype(np.float32), cv2.MOTION_AFFINE, 40, 3)
    if r is None:
        return None, 0.0
    cc, warp = r
    warp[:, 2] /= s
    return warp, float(cc)


def align_warp_ecc(test: np.ndarray, golden: np.ndarray, max_side: int = 1000) -> tuple[np.ndarray | None, float]:
    """Find where `golden` sits in `test`. Returns (warp, cc): warp is the 2x3 matrix taking a point of the
    test picture to the same point of the golden picture (golden resized to the test's size if they differ);
    None with cc 0.0 = no alignment found.

    Coarse to fine, so a board put down a few millimetres off or a few degrees turned still locks on:
      0. `max_side` px from the identity: accepted when it already correlates well (board nearly in place)
      1. ~160 px, heavy blur: rigid (shift + rotation) from several starting rotations, best correlation wins
      2. ~400 px: rigid refinement
      3. `max_side` px: affine refinement (a few iterations: it starts almost aligned)
    ECC alone at step 3 from the identity only captured ~2% of the frame width and ~1 degree."""
    if test.shape[:2] != golden.shape[:2]:
        golden = cv2.resize(golden, (test.shape[1], test.shape[0]))
    g1 = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
    full = max(g1.shape)

    def level(side):
        s = min(1.0, side / full)
        f = (lambda im: cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32))
        return f(g1), f(g2), s

    # A board that is already nearly in place (a fixture, a rectified card coupon, a scan) is best served by
    # one fine alignment from where it is: the coarse search can settle a fraction of a trace pitch away.
    a, b, s1 = level(max_side)
    r = _ecc(a, b, np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE, 60, 3)
    if r is not None and r[0] >= NEAR_CC:
        cc, warp = r
        warp[:, 2] /= s1
        return warp, float(cc)

    a, b, s0 = level(160)
    best = None
    for deg in (0.0, -4.0, 4.0, -8.0, 8.0):
        th = np.deg2rad(deg)
        c, si = float(np.cos(th)), float(np.sin(th))
        cx, cy = a.shape[1] / 2, a.shape[0] / 2
        w0 = np.float32([[c, -si, cx - c * cx + si * cy], [si, c, cy - si * cx - c * cy]])
        r = _ecc(a, b, w0, cv2.MOTION_EUCLIDEAN, 80, 7)
        if r is not None and (best is None or r[0] > best[0]):
            best = r
        if best is not None and best[0] > 0.9:
            break
    if best is None:
        return None, 0.0
    cc, warp = best
    prev = s0
    for side, motion, iters, gauss in ((400, cv2.MOTION_EUCLIDEAN, 40, 5), (max_side, cv2.MOTION_AFFINE, 30, 3)):
        a, b, s = level(side)
        warp = warp.copy()
        warp[:, 2] *= s / prev
        prev = s
        r = _ecc(a, b, warp, motion, iters, gauss)
        if r is None:
            return None, 0.0
        cc, warp = r
    warp[:, 2] /= prev
    return warp, float(cc)


def align(test: np.ndarray, golden: np.ndarray, max_side: int = 1000) -> tuple[np.ndarray, float]:
    """Warp `golden` onto `test`. Returns (warped, cc); cc 0.0 = no alignment found (golden comes back as is)."""
    warp, cc = align_warp(test, golden, max_side)
    if test.shape[:2] != golden.shape[:2]:
        golden = cv2.resize(golden, (test.shape[1], test.shape[0]))
    if warp is None:
        return golden, 0.0
    out = cv2.warpAffine(golden, warp, (test.shape[1], test.shape[0]),
                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return out, cc


def golden_to_test(warp: np.ndarray) -> np.ndarray:
    """The 3x3 matrix taking a point of the golden picture to the same point of the test picture
    (align_warp's matrix goes the other way)."""
    return np.linalg.inv(np.vstack([np.asarray(warp, float), [0.0, 0.0, 1.0]]))




def align_single_scale(test: np.ndarray, golden: np.ndarray, max_side: int = 1000) -> tuple[np.ndarray, float]:
    """Warp `golden` onto `test` (ECC, affine, on a downscaled grey image). Returns (warped, cc)."""
    if test.shape[:2] != golden.shape[:2]:
        golden = cv2.resize(golden, (test.shape[1], test.shape[0]))
    g1 = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
    s = min(1.0, max_side / max(g1.shape))
    a = cv2.resize(g1, None, fx=s, fy=s).astype(np.float32)
    b = cv2.resize(g2, None, fx=s, fy=s).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5),
                                        None, 3)
    except cv2.error:
        return golden, 0.0
    warp[:, 2] /= s
    out = cv2.warpAffine(golden, warp, (test.shape[1], test.shape[0]),
                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return out, float(cc)


class GoldenMeasurer:
    def __init__(self, test_bgr: np.ndarray, golden_bgr: np.ndarray, mm_per_px: float,
                 min_spacing_mm: float, aligned: bool = False, min_align_cc: float = MIN_ALIGN_CC):
        self.mm = mm_per_px
        self.min_spacing_mm = min_spacing_mm
        if aligned:
            g, self.align_cc = golden_bgr, 1.0
        else:
            g, self.align_cc = align(test_bgr, golden_bgr)
        self.ok = self.align_cc >= min_align_cc
        self.reason = "" if self.ok else f"golden alignment too poor to measure (cc {self.align_cc:.2f})"
        self.t_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.g_gray = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.t = copper_mask(test_bgr) > 0
        self.g = copper_mask(g) > 0
        self.last_reason = ""

    def _refined_golden_crop(self, x1, y1, x2, y2):
        """Golden mask crop shifted by the residual translation found by phase correlation."""
        a, b = self.t_gray[y1:y2, x1:x2], self.g_gray[y1:y2, x1:x2]
        g = self.g[y1:y2, x1:x2]
        if min(a.shape) < 16:
            return g, (0.0, 0.0)
        win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(b, a, win)
        if resp < 0.1 or max(abs(dx), abs(dy)) > MAX_REFINE_SHIFT_PX or max(abs(dx), abs(dy)) < 0.5:
            return g, (0.0, 0.0)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        g2 = cv2.warpAffine(g.astype(np.uint8), M, (g.shape[1], g.shape[0]), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_REPLICATE) > 0
        return g2, (dx, dy)

    def measure(self, cls: str, bbox, pad: int = 12) -> Measurement | None:
        """Measurement, or None with self.last_reason explaining why no mm claim is made."""
        self.last_reason = ""
        if not self.ok:
            self.last_reason = self.reason
            return None
        if cls not in MISSING_TYPES and cls not in EXTRA_TYPES:
            return None
        h, w = self.t.shape
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        t = self.t[y1:y2, x1:x2]
        g, _ = self._refined_golden_crop(x1, y1, x2, y2)
        region = (g & ~t) if cls in MISSING_TYPES else (t & ~g)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), connectivity=8)
        if n <= 1:
            self.last_reason = "no copper difference found at the defect"
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        comp = lab == k
        ys, xs = np.nonzero(comp)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        thick = min(rw, rh) + 1.0                       # rect is fitted through pixel centres
        if stats[k, cv2.CC_STAT_AREA] < 6 or thick < MIN_THICK_PX:
            self.last_reason = "difference is a 1-pixel sliver (registration), not measurable"
            return None
        g_lab = cv2.connectedComponents(g.astype(np.uint8), connectivity=8)[1]
        ring = cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if cls in MISSING_TYPES:
            parents = set(np.unique(g_lab[ring])) - {0}
            if not parents:
                self.last_reason = "conductor next to the defect not found in the golden image"
                return None
            on_parent = np.isin(g_lab, list(parents))
            dt = cv2.distanceTransform(on_parent.astype(np.uint8), cv2.DIST_L2, 5)
            ridge = on_parent & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-6) & (dt >= 1)
            r = int(3 * thick + 6)
            near = cv2.dilate(comp.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
            widths = 2.0 * dt[ridge & near]
            if len(widths) == 0:
                widths = 2.0 * dt[on_parent]
            local_w_px = float(np.percentile(widths, 25))     # the conductor, not a nearby pad
            if local_w_px <= 0:
                return None
            depth_px = thick
            if cls == "open":
                depth_px = max(depth_px, local_w_px)          # an open is a full cut by definition
            pct = 100.0 * min(depth_px / local_w_px, 1.0) if cls == "open" else 100.0 * depth_px / local_w_px
            return Measurement("width", depth_px * self.mm, local_w_px * self.mm, round(pct, 1), self.mm)
        # extra copper: distance to the nearest DIFFERENT golden conductor
        touch = set(np.unique(g_lab[ring])) - {0}
        others = (g_lab > 0) & ~np.isin(g_lab, list(touch))
        min_sp = self.min_spacing_mm
        if cls == "short" or len(touch) >= 2:
            return Measurement("spacing", 0.0, min_sp, 100.0, self.mm)     # touches two conductors
        if not others.any():
            self.last_reason = "no neighbouring conductor inside the measured window"
            return None
        dist = cv2.distanceTransform((~others).astype(np.uint8), cv2.DIST_L2, 5)
        remaining_mm = float(dist[comp].min()) * self.mm
        pct = 100.0 * max(0.0, min_sp - remaining_mm) / min_sp
        return Measurement("spacing", remaining_mm, min_sp, round(pct, 1), self.mm)
