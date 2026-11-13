"""Bare-board defect detector runtime: tile -> model -> decode -> merge (NMS) in image coordinates.

Backends
  executorch   the deployed path on the Pi: YOLO .pte (XNNPACK) through executorch.runtime
  ultralytics  laptop development only (GPU), same weights before export
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .record import BARE_CLASSES
from .tiling import TILE, nms, tile_origins


@dataclass
class Det:
    cls: str
    bbox: tuple[int, int, int, int]
    conf: float


class TileModel(Protocol):
    name: str

    def infer(self, tile_bgr: np.ndarray) -> np.ndarray:
        """tile (<=TILE x <=TILE, BGR) -> (N, 6) [x1, y1, x2, y2, conf, cls] in tile pixels."""


def _to_input(tile_bgr: np.ndarray) -> np.ndarray:
    """Pad bottom/right to TILE (value 114, like Ultralytics letterbox), RGB, CHW, [0,1]."""
    h, w = tile_bgr.shape[:2]
    if (h, w) != (TILE, TILE):
        canvas = np.full((TILE, TILE, 3), 114, np.uint8)
        canvas[:h, :w] = tile_bgr
        tile_bgr = canvas
    x = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def decode(out: np.ndarray, conf_thr: float, iou_thr: float = 0.5) -> np.ndarray:
    """Handle both YOLO head layouts.

    end-to-end (YOLO26 default): (1, K, 6) rows [x1, y1, x2, y2, conf, cls]
    classic (YOLOv8/11):         (1, 4+nc, A) rows [cx, cy, w, h, p_cls0..p_clsN] -> needs NMS
    """
    out = np.asarray(out)
    if out.ndim == 3 and out.shape[-1] == 6:
        r = out[0]
        return r[r[:, 4] >= conf_thr].astype(np.float32)
    p = out[0].T  # (A, 4+nc)
    scores = p[:, 4:]
    cls = scores.argmax(1)
    conf = scores.max(1)
    keep = conf >= conf_thr
    p, cls, conf = p[keep], cls[keep], conf[keep]
    if not len(p):
        return np.zeros((0, 6), np.float32)
    boxes = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2,
                      p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1)
    res = []
    for c in np.unique(cls):
        m = cls == c
        for i in nms(boxes[m], conf[m], iou_thr):
            res.append([*boxes[m][i], conf[m][i], c])
    return np.array(res, np.float32).reshape(-1, 6)


class ExecuTorchTiles:
    name = "executorch"

    def __init__(self, pte_path: str | Path, conf_thr: float = 0.05):
        import torch
        from executorch.runtime import Runtime
        self._torch = torch
        self.method = Runtime.get().load_program(str(pte_path)).load_method("forward")
        self.conf_thr = conf_thr

    def infer(self, tile_bgr: np.ndarray) -> np.ndarray:
        x = self._torch.from_numpy(_to_input(tile_bgr))
        out = self.method.execute([x])[0]
        return decode(out.numpy(), self.conf_thr)


class ArmRuntimeTiles:
    """Arm Runtime for YOLO26 on Raspberry Pi 5 (github.com/arm/Optimized-Runtimes, Apache-2.0): a
    fixed-graph YOLO26n implementation with model-specific memory planning, custom operators and fused,
    patched XNNPACK kernels. Arm's published Pi 5 figure for YOLO26n at 640 px: 36.6 ms vs 136.9 ms for
    stock ExecuTorch (benchmarks/reporting/baseline.md in that repo). It accepts any YOLO26n class count,
    so our fine-tuned 11-class detector converts with `make model CHECKPOINT=best.pt`.
    Raw output is [1, 4 + classes, positions] (cx, cy, w, h + per-class scores): the classic layout that
    decode() already handles with NMS. Input is the same letterboxed FP32 NCHW tile as ExecuTorchTiles."""
    name = "arm-runtime"

    def __init__(self, runtime_lib: str | Path, model_dir: str | Path, conf_thr: float = 0.05):
        from arm_yolo26n import Yolo26nTensorSession
        self.session = Yolo26nTensorSession(runtime=str(runtime_lib), model=str(model_dir))
        self.session.__enter__()
        self.conf_thr = conf_thr

    def infer(self, tile_bgr: np.ndarray) -> np.ndarray:
        return decode(np.asarray(self.session.execute(_to_input(tile_bgr))), self.conf_thr)

    def close(self):
        self.session.__exit__(None, None, None)


class UltralyticsTiles:
    name = "ultralytics"

    def __init__(self, weights: str | Path, device: str = "0", conf_thr: float = 0.05):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.device, self.conf_thr = device, conf_thr

    def infer(self, tile_bgr: np.ndarray) -> np.ndarray:
        h, w = tile_bgr.shape[:2]
        if (h, w) != (TILE, TILE):
            canvas = np.full((TILE, TILE, 3), 114, np.uint8)
            canvas[:h, :w] = tile_bgr
            tile_bgr = canvas
        r = self.model.predict(tile_bgr, imgsz=TILE, conf=self.conf_thr, device=self.device,
                               verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 6), np.float32)
        b = r.boxes
        return np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                               b.cls.cpu().numpy()[:, None]], 1).astype(np.float32)


class BareDetector:
    def __init__(self, tiles: TileModel, merge_iou: float = 0.5):
        self.tiles = tiles
        self.merge_iou = merge_iou

    def detect(self, bgr: np.ndarray) -> tuple[list[Det], dict[str, float]]:
        h, w = bgr.shape[:2]
        t0 = time.perf_counter()
        allr = []
        origins = tile_origins(w, h)
        for ox, oy in origins:
            r = self.tiles.infer(bgr[oy:oy + TILE, ox:ox + TILE])
            if len(r):
                r = r.copy()
                r[:, [0, 2]] += ox
                r[:, [1, 3]] += oy
                allr.append(r)
        t1 = time.perf_counter()
        dets: list[Det] = []
        if allr:
            a = np.concatenate(allr)
            a[:, [0, 2]] = a[:, [0, 2]].clip(0, w)
            a[:, [1, 3]] = a[:, [1, 3]].clip(0, h)
            for c in np.unique(a[:, 5]):
                m = a[:, 5] == c
                sub = a[m]
                for i in nms(sub[:, :4], sub[:, 4], self.merge_iou):
                    x1, y1, x2, y2, conf, _ = sub[i]
                    dets.append(Det(BARE_CLASSES[int(c)], (int(x1), int(y1), int(x2), int(y2)),
                                    float(conf)))
        dets.sort(key=lambda d: -d.conf)
        return dets, {"detector_tiles": (t1 - t0) * 1e3, "detector_merge": (time.perf_counter() - t1) * 1e3,
                      "n_tiles": float(len(origins))}
