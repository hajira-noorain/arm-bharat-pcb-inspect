"""Shared tiling so training data and on-device inference cut images identically.

PCB defects are tiny (a 0.1 mm mouse bite is a few pixels after a naive resize), so large frames
are split into overlapping TILE x TILE crops and detections are merged back with NMS.
"""
from __future__ import annotations

import numpy as np

TILE = 640
OVERLAP = 128


def tile_origins(w: int, h: int, tile: int = TILE, overlap: int = OVERLAP) -> list[tuple[int, int]]:
    """Top-left corners of tiles covering a w x h image. A small image yields one tile at (0,0)."""
    def axis(n: int) -> list[int]:
        if n <= tile:
            return [0]
        step = tile - overlap
        xs = list(range(0, n - tile + 1, step))
        if xs[-1] != n - tile:
            xs.append(n - tile)
        return xs
    return [(x, y) for y in axis(h) for x in axis(w)]


def clip_box_to_tile(box, ox: int, oy: int, tile: int = TILE, min_keep: float = 0.6):
    """Box (x1,y1,x2,y2) in image coords -> tile coords, or None if <min_keep of it is inside."""
    x1, y1, x2, y2 = box
    cx1, cy1 = max(x1, ox), max(y1, oy)
    cx2, cy2 = min(x2, ox + tile), min(y2, oy + tile)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    full = max((x2 - x1) * (y2 - y1), 1e-9)
    if (cx2 - cx1) * (cy2 - cy1) / full < min_keep:
        return None
    return (cx1 - ox, cy1 - oy, cx2 - ox, cy2 - oy)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> list[int]:
    """Plain NumPy NMS. boxes: (N,4) x1y1x2y2."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep
