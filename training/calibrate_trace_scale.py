"""Millimetre scale from a known design rule: the narrowest trace width.

When no ArUco marker was in the shot (e.g. a dataset), but the design's minimum track width is known
(Mendeley PCB-Defect: 20 mil = 0.508 mm), measure trace widths in pixels and solve for mm/px.

Method: copper mask (colour) -> distance transform -> ridge pixels (local maxima = centrelines)
-> width = 2 x distance. The minimum track width is taken as the 10th percentile of ridge widths
(the narrowest common conductor), robust to pads (wide) and noise (rare).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def copper_mask(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32)
    # copper is redder (higher a*) than the yellow-green laminate; Otsu picks the split per image
    a8 = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(cv2.GaussianBlur(a8, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def ridge_widths(mask: np.ndarray) -> np.ndarray:
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dil = cv2.dilate(dt, np.ones((3, 3), np.uint8))
    ridge = (dt >= dil - 1e-6) & (dt > 2)
    return 2 * dt[ridge]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=r"C:\Users\sulai\pcb_work\data\mendeley\PCB_Defect\images")
    ap.add_argument("--trace-mm", type=float, default=0.508)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=r"C:\Users\sulai\pcb_work\results\trace_scale_mendeley.json")
    args = ap.parse_args()
    paths = sorted(Path(args.images).glob("*.jpg"))
    random.Random(0).shuffle(paths)
    per_img = []
    for p in paths[:args.n]:
        w = ridge_widths(copper_mask(cv2.imread(str(p))))
        if len(w) > 500:
            per_img.append(float(np.percentile(w, 10)))
    px = float(np.median(per_img))
    res = {"min_trace_px_median": px, "min_trace_px_iqr": [float(np.percentile(per_img, 25)),
                                                            float(np.percentile(per_img, 75))],
           "images": len(per_img), "trace_mm": args.trace_mm, "mm_per_px": args.trace_mm / px,
           "implied_dpi": 25.4 / (args.trace_mm / px)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
