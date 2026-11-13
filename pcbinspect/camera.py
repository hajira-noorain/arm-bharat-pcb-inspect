"""Camera sources behind one interface: Pi Camera (Picamera2), USB webcam (OpenCV), or file replay."""
from __future__ import annotations

import itertools
import threading
from pathlib import Path

import cv2
import numpy as np


class Camera:
    name = "camera"

    def preview(self) -> np.ndarray | None:
        """Latest low-latency frame for the live view."""
        raise NotImplementedError

    def capture(self) -> np.ndarray:
        """Full-resolution still for inspection."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class PiCamera(Camera):
    name = "picamera2"

    def __init__(self, still_size=None, preview_size=(1014, 760), settle_s: float = 2.0):
        import time
        from picamera2 import Picamera2  # only on Raspberry Pi OS
        self.cam = Picamera2()
        # full sensor resolution by default: small defects need every pixel
        still_size = tuple(still_size or self.cam.sensor_resolution)
        cfg = self.cam.create_still_configuration(main={"size": still_size, "format": "RGB888"},
                                                  lores={"size": preview_size, "format": "YUV420"},
                                                  display="lores")
        self.cam.configure(cfg)
        self.cam.start()
        # let auto-exposure / white balance settle once, then LOCK them, so a change in room light
        # does not change what the model and the golden comparison see
        time.sleep(settle_s)
        md = self.cam.capture_metadata()
        self.locked = {"ExposureTime": md.get("ExposureTime"), "AnalogueGain": md.get("AnalogueGain"),
                       "ColourGains": md.get("ColourGains")}
        self.cam.set_controls({"AeEnable": False, "AwbEnable": False,
                               **{k: v for k, v in self.locked.items() if v is not None}})
        self.still_size = still_size

    def preview(self):
        yuv = self.cam.capture_array("lores")
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def capture(self):
        return self.cam.capture_array("main")  # RGB888 in Picamera2 is BGR byte order for OpenCV

    def close(self):
        self.cam.stop()


class USBCamera(Camera):
    name = "usb"

    def __init__(self, index: int = 0, size=(3840, 2160)):   # the driver picks the nearest it supports
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open USB camera {index}")
        # lock exposure where the driver allows it (V4L2: 1 = manual), after letting it settle
        for _ in range(15):
            self.cap.read()
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self._lock = threading.Lock()

    def preview(self):
        with self._lock:
            ok, f = self.cap.read()
        return f if ok else None

    def capture(self):
        with self._lock:
            for _ in range(3):  # drop stale buffered frames
                self.cap.grab()
            ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("USB camera read failed")
        return f

    def close(self):
        self.cap.release()


class ReplayCamera(Camera):
    """Cycles through image files — used for the dataset demo and for the simulator."""
    name = "replay"

    def __init__(self, paths: list[str | Path]):
        if not paths:
            raise ValueError("replay camera needs at least one image")
        self.paths = [Path(p) for p in paths]
        self._it = itertools.cycle(self.paths)
        self.current = self.paths[0]

    def next(self) -> Path:
        self.current = next(self._it)
        return self.current

    def preview(self):
        return cv2.imread(str(self.current))

    def capture(self):
        im = cv2.imread(str(self.current))
        if im is None:
            raise FileNotFoundError(self.current)
        return im

    def golden(self) -> np.ndarray | None:
        """Known-good image of the current board, if the demo set ships one (<name>.golden.jpg)."""
        p = self.current.with_name(self.current.stem + ".golden.jpg")
        return cv2.imread(str(p)) if p.exists() else None


def open_camera(kind: str, **kw) -> Camera:
    if kind == "picamera2":
        return PiCamera(**kw)
    if kind == "usb":
        return USBCamera(**kw)
    if kind == "replay":
        return ReplayCamera(**kw)
    if kind == "auto":
        try:
            return PiCamera()
        except Exception:
            return USBCamera()
    raise ValueError(kind)
