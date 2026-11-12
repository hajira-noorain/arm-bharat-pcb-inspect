"""Operator-facing annotated image: boxes coloured by severity, anomaly heatmap, verdict banner."""
from __future__ import annotations

import cv2
import numpy as np

SEV_BGR = {"critical": (40, 40, 230), "major": (0, 150, 255), "minor": (0, 210, 230), None: (200, 200, 200)}
VERDICT_BGR = {"PASS": (60, 170, 60), "REVIEW": (0, 170, 255), "FAIL": (40, 40, 220)}


def render(bgr: np.ndarray, insp, analysis=None, max_side: int = 1600) -> np.ndarray:
    img = bgr.copy()
    h, w = img.shape[:2]
    if analysis is not None and getattr(analysis, "anomaly", None) is not None and insp.verdict != "PASS":
        hm = analysis.anomaly.heatmap
        thr = 0.0
        norm = np.clip((hm - thr) / (hm.max() - thr + 1e-6), 0, 1)
        hm_img = cv2.applyColorMap((cv2.resize(norm, (w, h)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img = cv2.addWeighted(img, 0.7, hm_img, 0.3, 0)
    lw = max(2, int(round(max(h, w) / 600)))
    fs = max(0.5, max(h, w) / 1600)
    for d in insp.defects:
        x1, y1, x2, y2 = d.bbox
        col = SEV_BGR.get(d.severity, SEV_BGR[None])
        pad = lw * 3
        cv2.rectangle(img, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), col, lw)
        label = f"{d.id} {d.cls.replace('_', ' ')}"
        if d.size_mm is not None:
            label += f" {d.size_mm:.2f}mm"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, lw)
        ty = max(th + 4, y1 - pad - 4)
        cv2.rectangle(img, (x1 - pad, ty - th - 4), (x1 - pad + tw + 6, ty + 4), col, -1)
        cv2.putText(img, label, (x1 - pad + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), lw)
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img
