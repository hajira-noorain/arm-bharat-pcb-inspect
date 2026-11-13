"""Golden-board measurement: measure what a defect actually does to the copper, in millimetres.

A detector box is drawn loosely around a defect, so its size is not the defect's size. Industrial
AOI compares the board with a known-good ("golden") image of the same design instead:

  1. align the board to the golden image (ECC, affine)
  2. copper masks of both; their difference inside the detection box IS the defect
  3a. missing copper (mouse bite, pinhole, open): depth = short side of the missing region's
      min-area rectangle; local trace width = 2 x golden distance-transform there;
      width reduction % = depth / local width                                   (IPC width rule)
  3b. extra copper (spur, spurious copper, short): remaining spacing = shortest distance from the
      extra copper to a DIFFERENT golden conductor (not the one it grows from);
      spacing reduction % = (minimum spacing - remaining) / minimum spacing     (IPC spacing rule)

No golden image -> no millimetre claim (severity falls back to the defect-type default).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MISSING_TYPES = {"mouse_bite", "pin_hole", "open", "missing_pad"}
EXTRA_TYPES = {"spur", "spurious_copper", "short"}


@dataclass
class Measurement:
    kind: str            # 'width' | 'spacing'
    defect_mm: float     # nick depth (width) or remaining gap (spacing)
    ref_mm: float        # local trace width (width) or minimum spacing (spacing)
    pct: float           # reduction in percent (>=100 means fully cut / bridged)
    method: str = "golden-diff"

    def text(self) -> str:
        if self.kind == "width":
            if self.pct >= 100:
                return f"cuts the full {self.ref_mm:.2f} mm conductor"
            return f"{self.defect_mm:.3f} mm nick in a {self.ref_mm:.2f} mm conductor ({self.pct:.0f}% width lost)"
        if self.pct >= 100:
            return "bridges the gap to the next conductor"
        return f"leaves {self.defect_mm:.3f} mm gap against {self.ref_mm:.2f} mm minimum ({self.pct:.0f}% spacing lost)"


def copper_mask(bgr: np.ndarray) -> np.ndarray:
    """Copper (or the dark/bright conductor layer) as a binary mask, robust across binarised scans,
    bare copper on laminate and green solder mask: Otsu on the channel with most contrast."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    best, best_sep = None, -1.0
    for ch in (lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]):
        c8 = cv2.GaussianBlur(ch, (3, 3), 0)
        t, m = cv2.threshold(c8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        a, b = c8[m > 0], c8[m == 0]
        if len(a) < 10 or len(b) < 10:
            continue
        sep = abs(float(a.mean()) - float(b.mean())) / (float(c8.std()) + 1e-6)
        if sep > best_sep:
            best, best_sep = m, sep
    m = best if best is not None else np.zeros(bgr.shape[:2], np.uint8)
    # conductor = the minority phase on most boards; flip if the "copper" covers most of the image
    if (m > 0).mean() > 0.5:
        m = 255 - m
    return m


def align(test: np.ndarray, golden: np.ndarray, max_side: int = 1000) -> tuple[np.ndarray, float]:
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
                 min_spacing_mm: float, aligned: bool = False):
        self.mm = mm_per_px
        self.min_spacing_mm = min_spacing_mm
        if aligned:
            g, self.align_cc = golden_bgr, 1.0
        else:
            g, self.align_cc = align(test_bgr, golden_bgr)
        self.t = copper_mask(test_bgr) > 0
        self.g = copper_mask(g) > 0
        self.g_dt = cv2.distanceTransform(self.g.astype(np.uint8), cv2.DIST_L2, 5)
        self.g_lab = cv2.connectedComponents(self.g.astype(np.uint8), connectivity=8)[1]

    def measure(self, cls: str, bbox, pad: int = 6) -> Measurement | None:
        h, w = self.t.shape
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        t, g = self.t[y1:y2, x1:x2], self.g[y1:y2, x1:x2]
        if cls in MISSING_TYPES:
            region = g & ~t
        elif cls in EXTRA_TYPES:
            region = t & ~g
        else:
            return None
        # no morphological opening: an even kernel shifts the region by a pixel, which matters at
        # this scale; tiny noise components are removed by the area filter instead
        n, lab, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), connectivity=8)
        if n <= 1:
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[k, cv2.CC_STAT_AREA] < 4:
            return None
        comp = lab == k
        ys, xs = np.nonzero(comp)
        ring = cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # 2 px around it
        gl = self.g_lab[y1:y2, x1:x2]
        if cls in MISSING_TYPES:
            pts = np.stack([xs, ys], 1).astype(np.float32)
            (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
            depth_px = max(1.0, min(rw, rh) + 1.0)      # rect is fitted through pixel centres
            # local width of the conductor the nick is cut into: its golden distance-transform peak
            parents = set(np.unique(gl[ring])) - {0}
            if not parents:
                return None
            on_parent = np.isin(gl, list(parents))
            local_w_px = 2.0 * float(self.g_dt[y1:y2, x1:x2][on_parent].max())
            if local_w_px <= 0:
                return None
            if cls == "open":
                depth_px = max(depth_px, local_w_px)     # an open is a full cut by definition
            pct = 100.0 * min(depth_px / local_w_px, 1.0) if cls == "open" else 100.0 * depth_px / local_w_px
            return Measurement("width", depth_px * self.mm, local_w_px * self.mm, round(pct, 1))
        # extra copper: distance to the nearest DIFFERENT golden conductor
        touch = set(np.unique(gl[ring])) - {0}
        others = (self.g_lab > 0) & ~np.isin(self.g_lab, list(touch))
        dist = cv2.distanceTransform((~others).astype(np.uint8), cv2.DIST_L2, 5)
        remaining_px = float(dist[y1:y2, x1:x2][comp].min())
        min_sp = self.min_spacing_mm
        remaining_mm = remaining_px * self.mm
        if cls == "short" or len(touch) >= 2:
            remaining_mm = 0.0                             # touches two conductors: bridged
        pct = 100.0 * max(0.0, min_sp - remaining_mm) / min_sp
        return Measurement("spacing", remaining_mm, min_sp, round(pct, 1))
