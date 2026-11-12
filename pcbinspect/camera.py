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

    def __init__(self, still_size=(2028, 1520), preview_size=(1014, 760)):
        from picamera2 import Picamera2  # only on Raspberry Pi OS
        self.cam = Picamera2()
        cfg = self.cam.create_still_configuration(main={"size": still_size, "format": "RGB888"},
                                                  lores={"size": preview_size, "format": "YUV420"},
                                                  display="lores")
        self.cam.configure(cfg)
        self.cam.start()
        # lock exposure/white balance after settling so lighting changes don't drift the model input
        self.cam.set_controls({"AeEnable": True, "AwbEnable": True})

    def preview(self):
        yuv = self.cam.capture_array("lores")
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def capture(self):
        return self.cam.capture_array("main")  # RGB888 in Picamera2 is BGR byte order for OpenCV

    def close(self):
        self.cam.stop()


class USBCamera(Camera):
    name = "usb"

    def __init__(self, index: int = 0, size=(1920, 1080)):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open USB camera {index}")
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
