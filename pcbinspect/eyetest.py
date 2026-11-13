"""The detection-limit "eye test": a printed card with defects of known size, read by the station.

Like an optician's chart: rows of defects that get smaller, each size repeated ~30 times, so the
station can measure (not guess) the smallest defect it reliably finds on THIS camera, lighting and
software. One GOLDEN sheet carries the same board patterns with no defects (the known-good board);
the TEST sheets carry the planted defects listed in the card's manifest.

Sheet geometry (mm, origin = top-left outer corner of the frame):
  frame W x H with ArUco DICT_4X4_50 corner markers 10 (TL), 11 (TR), 12 (BR), 13 (BL), MARKER_MM wide,
  their OUTER corners exactly at the frame corners; sheet id marker 20 + k (k = 0 golden, 1.. test)
  at the top centre; a COLS x ROWS grid of board coupons, each COUPON_MM square.
A photo is rectified with the four outer corners to RECT_PX_PER_MM, so coupon and defect positions
are known exactly; test and golden captures are then compared coupon by coupon.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .defect_render import NOUN, TYPES
from .fusion import fuse
from .golden_diff import diff_candidates
from .pod import fit_pod

ARUCO = cv2.aruco.DICT_4X4_50
CORNER_IDS = (10, 11, 12, 13)
SHEET_ID_BASE = 20
MARKER_MM = 8.0
COUPON_MM = 640 / 48.0           # one DeepPCB pattern at its native 48 px/mm
GAP_MM = 2.0
# 7 x 3 coupons: the frame (125 x 64 mm) must fit the camera view AT THE SAME ZOOM AS REAL BOARDS
# (the limit belongs to that setup). Pi Camera Module 3 focuses down to ~10 cm, where it sees about
# 130 x 75 mm, i.e. ~35 px/mm on its 4608 x 2592 sensor.
COLS, ROWS = 7, 3
MARGIN_X, MARGIN_Y = 10.0, 10.0  # coupon grid offset inside the frame
FRAME_W = MARGIN_X * 2 + COLS * COUPON_MM + (COLS - 1) * GAP_MM
FRAME_H = MARGIN_Y * 2 + ROWS * COUPON_MM + (ROWS - 1) * GAP_MM
RECT_PX_PER_MM = 48.0
SIZES_MM = (0.2, 0.3, 0.4, 0.5, 0.7, 1.0)   # inkjet-friendly; see docs for why not smaller
# shorts are sized by the GAP they bridge (see defect_render): gaps on real boards are 0.1-0.5 mm
SIZES_BY_TYPE = {"short": (0.15, 0.2, 0.25, 0.3, 0.4, 0.5)}


def sizes_for(cls: str) -> tuple:
    return SIZES_BY_TYPE.get(cls, SIZES_MM)


def coupon_origin_mm(idx: int) -> tuple[float, float]:
    r, c = divmod(idx, COLS)
    return MARGIN_X + c * (COUPON_MM + GAP_MM), MARGIN_Y + r * (COUPON_MM + GAP_MM)


def marker_outer_corners_mm():
    """Outer corner of each corner marker (the point used for the homography), by id."""
    return {10: (0.0, 0.0), 11: (FRAME_W, 0.0), 12: (FRAME_W, FRAME_H), 13: (0.0, FRAME_H)}


@dataclass
class Rectified:
    img: np.ndarray
    sheet: int | None
    ok: bool
    reason: str = ""


def rectify(bgr: np.ndarray) -> Rectified:
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return Rectified(bgr, None, False, "no markers found")
    found = {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}
    missing = [i for i in CORNER_IDS if i not in found]
    if missing:
        return Rectified(bgr, None, False, f"corner markers not visible: {missing}")
    # ArUco corner order is TL, TR, BR, BL of the marker itself; the frame's outer corner of marker
    # 10 is its TL, of 11 its TR, of 12 its BR, of 13 its BL
    src = np.float32([found[10][0], found[11][1], found[12][2], found[13][3]])
    dst_mm = marker_outer_corners_mm()
    dst = np.float32([dst_mm[i] for i in CORNER_IDS]) * RECT_PX_PER_MM
    H = cv2.getPerspectiveTransform(src, dst)
    size = (int(round(FRAME_W * RECT_PX_PER_MM)), int(round(FRAME_H * RECT_PX_PER_MM)))
    out = cv2.warpPerspective(bgr, H, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    sheet_ids = [i - SHEET_ID_BASE for i in found if SHEET_ID_BASE <= i < SHEET_ID_BASE + 20]
    return Rectified(out, sheet_ids[0] if sheet_ids else None, True)


def sheet_raster(coupons: list[np.ndarray], sheet: int) -> np.ndarray:
    """The sheet exactly as the station sees it after rectification (white paper, black ink), at
    RECT_PX_PER_MM. Also the image printed into the PDF at physical size FRAME_W x FRAME_H mm."""
    k = RECT_PX_PER_MM
    W, H = int(round(FRAME_W * k)), int(round(FRAME_H * k))
    canvas = np.full((H, W), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(ARUCO)
    m = int(round(MARKER_MM * k))
    for mid, (x, y) in marker_outer_corners_mm().items():
        px, py = int(round(x * k)), int(round(y * k))
        x0 = px if mid in (10, 13) else px - m
        y0 = py if mid in (10, 11) else py - m
        canvas[y0:y0 + m, x0:x0 + m] = cv2.aruco.generateImageMarker(d, mid, m)
    sid = cv2.aruco.generateImageMarker(d, SHEET_ID_BASE + sheet, m)
    y0 = 0  # sheet id marker: top centre, between the two top corner markers
    x0 = (W - m) // 2
    canvas[y0:y0 + m, x0:x0 + m] = sid
    for idx, img in enumerate(coupons):
        x, y = coupon_origin_mm(idx)
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cx, cy = int(round(x * k)), int(round(y * k))
        canvas[cy:cy + g.shape[0], cx:cx + g.shape[1]] = g
    return canvas


def coupon_crop(rect_img: np.ndarray, idx: int, pad_mm: float = 1.0) -> tuple[np.ndarray, tuple[int, int]]:
    x, y = coupon_origin_mm(idx)
    k = RECT_PX_PER_MM
    x0, y0 = int(round((x - pad_mm) * k)), int(round((y - pad_mm) * k))
    x1, y1 = int(round((x + COUPON_MM + pad_mm) * k)), int(round((y + COUPON_MM + pad_mm) * k))
    return rect_img[y0:y1, x0:x1], (x0, y0)


def _covers(bbox, center_px) -> bool:
    return bbox[0] - 4 <= center_px[0] <= bbox[2] + 4 and bbox[1] - 4 <= center_px[1] <= bbox[3] + 4


def read_sheet(test_rect: np.ndarray, golden_rect: np.ndarray | None, manifest_sheet: dict, detector,
               use_golden: bool = True) -> list[dict]:
    """Score every planted defect on one rectified test sheet. Returns per-defect trials with the best
    covering evidence (fused when a golden sheet is given). False calls are counted per coupon."""
    trials = []
    k = RECT_PX_PER_MM
    for cp in manifest_sheet["coupons"]:
        crop, (ox, oy) = coupon_crop(test_rect, cp["idx"])
        dets, _ = detector.detect(crop)
        cands = [{"bbox": d.bbox, "conf": d.conf, "cls": d.cls} for d in dets]
        if use_golden and golden_rect is not None:
            gcrop, _ = coupon_crop(golden_rect, cp["idx"])
            dc, _ = diff_candidates(crop, gcrop)
            cands = fuse(cands, dc)
        used = set()
        for d in cp["defects"]:
            c = ((d["center_mm"][0]) * k - ox, (d["center_mm"][1]) * k - oy)
            best = max(((i, x) for i, x in enumerate(cands) if _covers(x["bbox"], c)),
                       key=lambda z: z[1]["conf"], default=None)
            if best:
                used.add(best[0])
            trials.append({"type": d["type"], "size_mm": d["size_mm"], "conf": best[1]["conf"] if best else 0.0,
                           "cls": best[1]["cls"] if best else None, "coupon": cp["idx"]})
        trials.append({"false_calls": [x["conf"] for i, x in enumerate(cands) if i not in used], "coupon": cp["idx"]})
    return trials


def summarise(trials: list[dict], lam: float) -> dict:
    """Detection limit per defect type at the operating threshold lam (the gate's per-defect lambda)."""
    out = {"lambda": lam, "types": {}}
    fc = [sum(c >= lam for c in t["false_calls"]) for t in trials if "false_calls" in t]
    out["false_calls_per_coupon"] = float(np.mean(fc)) if fc else None
    for cls in TYPES:
        tr = [t for t in trials if t.get("type") == cls]
        if not tr:
            continue
        r = fit_pod(np.array([t["size_mm"] for t in tr]), np.array([t["conf"] >= lam for t in tr], int))
        d = r.to_dict()
        d["sentence"] = r.sentence(NOUN[cls])
        out["types"][cls] = d
    return out


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
