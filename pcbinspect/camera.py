"""Camera sources behind one interface: Pi Camera (Picamera2), USB webcam (OpenCV), or file replay."""
from __future__ import annotations

import itertools
import threading
from pathlib import Path

import cv2
import numpy as np

from .imaging import average_frames, sharpness


class Camera:
    name = "camera"

    def preview(self) -> np.ndarray | None:
        """Latest low-latency frame for the live view."""
        raise NotImplementedError

    def _grab(self) -> np.ndarray:
        """One full-resolution frame. Subclasses implement this."""
        raise NotImplementedError

    def capture(self, n_average: int = 1) -> np.ndarray:
        """Full-resolution still for inspection; n_average > 1 averages frames (less sensor noise)."""
        n = max(1, int(n_average))
        if n == 1:
            return self._grab()
        return average_frames(self._grab() for _ in range(n))

    def settings(self) -> dict:
        """Current capture settings, to be stored with each inspection."""
        return {"camera": self.name}

    def close(self) -> None:
        pass


def focus_score(img: np.ndarray) -> tuple[float, str]:
    """Laplacian-variance sharpness inside the ArUco marker box if one is found, else whole frame."""
    roi = None
    try:
        from .calib import detect_scale
        sc = detect_scale(img, 1.0, max_skew=10.0)
        if sc is not None and sc.corners is not None:
            x0, y0 = np.floor(sc.corners.min(axis=0)).astype(int)
            x1, y1 = np.ceil(sc.corners.max(axis=0)).astype(int)
            if (x1 - x0) >= 16 and (y1 - y0) >= 16:
                roi = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
    except Exception:
        roi = None
    return sharpness(img, roi), ("marker" if roi else "frame")


def pick_focus(results: list[dict]) -> dict:
    """Best lens position from [{'lens_position', 'score', 'region'}, ...]."""
    if not results:
        raise ValueError("no focus positions given")
    best = max(results, key=lambda r: r["score"])
    scores = sorted(r["score"] for r in results)
    return {"best": best["lens_position"], "best_score": best["score"], "results": results,
            # a peak barely above the runner-up means the sweep saw no real focus peak
            "confident": len(scores) < 2 or best["score"] > 1.1 * scores[-2],
            "at_edge": best is results[0] or best is results[-1]}


class PiCamera(Camera):
    """Pi Camera Module 3 via Picamera2. Manual focus, locked exposure and white balance.

    Distance note: closest focus is about 10 cm (LensPosition 10 dioptres); 11-12 cm gives about
    30 px/mm. Depth of field is only a few mm, so run focus_sweep() once per rig height.
    """
    name = "picamera2"
    SETTLE_FRAMES = 3     # frames thrown away after any control change

    def __init__(self, still_size=None, preview_size=(1014, 760), settle_s: float = 2.0,
                 lens_position: float | None = None, raw: bool = False):
        import time
        from picamera2 import Picamera2  # only on Raspberry Pi OS
        self.cam = Picamera2()
        # full sensor resolution by default: small defects need every pixel
        still_size = tuple(still_size or self.cam.sensor_resolution)
        extra = {"raw": {}} if raw else {}   # raw stream is only needed for capture_raw_dng()
        cfg = self.cam.create_still_configuration(main={"size": still_size, "format": "RGB888"},
                                                  lores={"size": preview_size, "format": "YUV420"},
                                                  display="lores", **extra)
        self.cam.configure(cfg)
        self.cam.start()
        # let auto-exposure / white balance settle once, then LOCK them, so a change in room light
        # does not change what the model and the golden comparison see
        time.sleep(settle_s)
        md = self.cam.capture_metadata()
        self.locked = {"ExposureTime": md.get("ExposureTime"), "AnalogueGain": md.get("AnalogueGain"),
                       "ColourGains": md.get("ColourGains")}
        self.still_size = still_size
        self.lens_position = None
        self.raw_enabled = raw
        controls = {"AeEnable": False, "AwbEnable": False,
                    **{k: v for k, v in self.locked.items() if v is not None}}
        if lens_position is not None:
            controls.update({"AfMode": 0, "LensPosition": float(lens_position)})
            self.lens_position = float(lens_position)
        self._set_controls(controls)

    # -- control helpers (the Picamera2 calls are isolated here so the logic is testable)
    def _set_controls(self, controls: dict, discard: int | None = None) -> None:
        self.cam.set_controls(controls)
        for _ in range(self.SETTLE_FRAMES if discard is None else discard):
            self.cam.capture_array("main")      # frames in flight still carry the old controls

    def set_focus(self, lens_position: float) -> None:
        """Manual focus: AfMode 0 + LensPosition in dioptres (0 = infinity, 10 = 10 cm)."""
        self._set_controls({"AfMode": 0, "LensPosition": float(lens_position)})
        self.lens_position = float(lens_position)

    def focus_sweep(self, positions=None, frames_per_position: int = 1) -> dict:
        """Try each lens position, score sharpness, lock the best one; returns a report.

        Score = variance of the Laplacian inside the ArUco marker's box if a marker is seen,
        else the whole frame. Put the marker (or any detailed flat board) at working height first.
        """
        if positions is None:
            positions = [round(float(v), 2) for v in np.arange(7.0, 11.01, 0.5)]
        results = []
        for pos in positions:
            self.set_focus(pos)
            img = self.capture(frames_per_position)
            score, region = focus_score(img)
            results.append({"lens_position": float(pos), "score": score, "region": region})
        report = pick_focus(results)
        self.set_focus(report["best"])
        return report

    def preview(self):
        yuv = self.cam.capture_array("lores")
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def _grab(self):
        return self.cam.capture_array("main")  # RGB888 in Picamera2 is BGR byte order for OpenCV

    def capture_raw_dng(self, path) -> str:
        """Save the unprocessed sensor data as a DNG. Needs PiCamera(raw=True) and a Picamera2 whose
        requests have save_dng (Raspberry Pi OS Bookworm). Use for the golden board and eye-test card."""
        if not self.raw_enabled:
            raise RuntimeError("open the camera with raw=True to enable DNG capture")
        request = self.cam.capture_request()
        try:
            if not hasattr(request, "save_dng"):
                raise RuntimeError("this Picamera2 version has no save_dng")
            request.save_dng(str(path))
        finally:
            request.release()
        return str(path)

    def settings(self) -> dict:
        d = {"camera": self.name, "still_size": list(self.still_size), "af_mode": "manual",
             "lens_position": self.lens_position, "ae_locked": True, "awb_locked": True}
        for k, v in self.locked.items():
            d[k] = [float(x) for x in v] if isinstance(v, (tuple, list)) else v
        return d

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
        # let it settle, then lock exposure where the driver allows it (V4L2: 1 = manual)
        for _ in range(15):
            self.cap.read()
        self._lock = threading.Lock()
        # best effort: drivers silently ignore what they do not support, so report what was accepted
        self.requested = {}
        for label, prop, val in (("autofocus", cv2.CAP_PROP_AUTOFOCUS, 0),
                                 ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE, 1),
                                 ("auto_wb", cv2.CAP_PROP_AUTO_WB, 0)):
            try:
                self.requested[label] = bool(self.cap.set(prop, val))
            except Exception:
                self.requested[label] = False

    def preview(self):
        with self._lock:
            ok, f = self.cap.read()
        return f if ok else None

    def _grab(self):
        with self._lock:
            for _ in range(3):  # drop stale buffered frames
                self.cap.grab()
            ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("USB camera read failed")
        return f

    def settings(self) -> dict:
        def get(p):
            try:
                v = float(self.cap.get(p))
            except Exception:
                return None
            return v if v == v else None
        return {"camera": self.name, "width": get(cv2.CAP_PROP_FRAME_WIDTH),
                "height": get(cv2.CAP_PROP_FRAME_HEIGHT), "autofocus": get(cv2.CAP_PROP_AUTOFOCUS),
                "auto_exposure": get(cv2.CAP_PROP_AUTO_EXPOSURE), "exposure": get(cv2.CAP_PROP_EXPOSURE),
                "gain": get(cv2.CAP_PROP_GAIN), "auto_wb": get(cv2.CAP_PROP_AUTO_WB),
                "focus": get(cv2.CAP_PROP_FOCUS), "controls_accepted": dict(self.requested)}

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
        self.current = next(self._it)       # (not paths[0] without advancing: the first image came twice)

    def next(self) -> Path:
        self.current = next(self._it)
        return self.current

    def preview(self):
        return cv2.imread(str(self.current))

    def _grab(self):
        im = cv2.imread(str(self.current))
        if im is None:
            raise FileNotFoundError(self.current)
        return im

    def capture(self, n_average: int = 1):
        return self._grab()   # a file is already noise-free; averaging would change nothing

    def settings(self) -> dict:
        return {"camera": self.name, "file": str(self.current)}

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
