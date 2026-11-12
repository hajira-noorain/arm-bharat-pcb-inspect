"""Millimetre scale from a printed ArUco marker, and board ID from a QR code.

Print deploy/assets/aruco_20mm.png at 100% scale and place it beside the board in the same plane.
mm_per_px = marker side (mm) / mean marker side (px). A strongly non-square marker means the
camera is tilted; we report that instead of silently producing wrong millimetres.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 7


@dataclass
class Scale:
    mm_per_px: float
    source: str           # 'aruco' | 'design-default'
    skew: float = 0.0     # relative difference between the marker's longest and shortest side
    corners: np.ndarray | None = None


def detect_scale(bgr: np.ndarray, marker_mm: float, max_skew: float = 0.06) -> Scale | None:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return None
    for c, i in zip(corners, ids.ravel()):
        if int(i) != MARKER_ID:
            continue
        p = c.reshape(4, 2)
        sides = np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1)
        skew = float((sides.max() - sides.min()) / sides.mean())
        if skew > max_skew:
            return Scale(marker_mm / float(sides.mean()), "aruco-skewed", skew, p)
        return Scale(marker_mm / float(sides.mean()), "aruco", skew, p)
    return None


def make_marker_png(path: str, marker_mm: float = 20.0, dpi: int = 300) -> None:
    """Printable marker with a quiet zone; prints at exactly marker_mm when printed at `dpi`."""
    px = int(round(marker_mm / 25.4 * dpi))
    img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), MARKER_ID, px)
    quiet = px // 4
    canvas = np.full((px + 2 * quiet, px + 2 * quiet), 255, np.uint8)
    canvas[quiet:quiet + px, quiet:quiet + px] = img
    cv2.putText(canvas, f"{marker_mm:g} mm  ID {MARKER_ID}", (quiet, px + quiet + quiet // 2),
                cv2.FONT_HERSHEY_SIMPLEX, px / 600, 0, max(1, px // 300))
    cv2.imwrite(path, canvas)


def read_board_id(bgr: np.ndarray) -> str | None:
    data, _, _ = cv2.QRCodeDetector().detectAndDecode(bgr)
    return data or None
