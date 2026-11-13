"""Image-quality helpers for the capture station: flat-field, white balance, sharpness,
a capture-quality gate ("shift sentinel") and lens calibration use.

Everything here is plain numpy/OpenCV so it runs (and is unit tested) without a camera.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ---- thresholds for capture_quality (8-bit images) ----
MIN_SHARPNESS = 15.0        # variance of the Laplacian; below this the frame is blurred
CLIP_HI_LEVEL = 250         # a pixel at or above this in any channel counts as clipped bright
CLIP_LO_LEVEL = 3           # a pixel at or below this in every channel counts as crushed dark
MAX_CLIP_FRACTION = 0.02    # more than 2 % clipped -> refuse
MIN_MEAN = 50.0             # mean grey below this -> under-exposed
MAX_MEAN = 205.0            # mean grey above this -> over-exposed


def average_frames(frames) -> np.ndarray:
    """Mean of several uint8 frames, computed in float and returned as uint8 (noise ~ 1/sqrt(N))."""
    acc = None
    n = 0
    for f in frames:
        f = np.asarray(f)
        acc = f.astype(np.float32) if acc is None else acc + f.astype(np.float32)
        n += 1
    if n == 0:
        raise ValueError("no frames to average")
    return np.clip(np.rint(acc / n), 0, 255).astype(np.uint8)


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def sharpness(img: np.ndarray, roi=None) -> float:
    """Variance of the Laplacian, optionally inside roi = (x, y, w, h)."""
    g = _gray(img)
    if roi is not None:
        x, y, w, h = (int(round(v)) for v in roi)
        x, y = max(x, 0), max(y, 0)
        g = g[y:y + h, x:x + w]
        if g.size == 0:
            raise ValueError("empty ROI")
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def grey_patch_white_balance(img: np.ndarray, roi) -> np.ndarray:
    """Scale B, G, R so the mean of the grey patch roi=(x, y, w, h) becomes neutral."""
    x, y, w, h = (int(round(v)) for v in roi)
    patch = img[max(y, 0):y + h, max(x, 0):x + w].reshape(-1, 3).astype(np.float64)
    if patch.size == 0:
        raise ValueError("empty ROI")
    m = patch.mean(axis=0)
    if np.any(m < 1):
        raise ValueError("grey patch is too dark to balance on")
    gains = m.mean() / m
    out = img.astype(np.float32) * gains.astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- flat field
class FlatField:
    """Per-channel gain map from frames of a blank matte card under the working light.

    Dividing a board image by the (normalised) map removes vignetting and uneven light.
    """

    def __init__(self, gain: np.ndarray):
        self.gain = gain.astype(np.float32)          # H x W x 3, mean ~ 1

    @classmethod
    def from_frames(cls, frames, sigma_frac: float = 1 / 30) -> "FlatField":
        avg = np.mean([np.asarray(f, np.float32) for f in frames], axis=0)
        if avg.ndim == 2:
            avg = avg[..., None]
        w = avg.shape[1]
        sigma = max(1.0, w * sigma_frac)
        sm = np.stack([cv2.GaussianBlur(avg[..., c], (0, 0), sigma) for c in range(avg.shape[2])], axis=-1)
        sm = np.maximum(sm, 1.0)
        gain = sm / sm.reshape(-1, sm.shape[2]).mean(axis=0)   # each channel normalised to mean 1
        return cls(gain)

    def apply(self, img: np.ndarray) -> np.ndarray:
        g = self.gain
        if img.shape[:2] != g.shape[:2]:
            g = cv2.resize(g, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            if g.ndim == 2:
                g = g[..., None]
        if img.ndim == 2:
            out = img.astype(np.float32) / g.mean(axis=2)
        else:
            out = img.astype(np.float32) / g
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)

    def save(self, path) -> None:
        np.savez_compressed(str(path), gain=self.gain)

    @classmethod
    def load(cls, path) -> "FlatField":
        with np.load(str(path)) as z:
            return cls(z["gain"])


# ---------------------------------------------------------------- quality gate
def _stats(img: np.ndarray) -> dict:
    g = _gray(img)
    if img.ndim == 3:
        hi = float((img >= CLIP_HI_LEVEL).any(axis=2).mean())
        lo = float((img <= CLIP_LO_LEVEL).all(axis=2).mean())
        m = img.reshape(-1, 3).astype(np.float64).mean(axis=0)          # B, G, R
        balance = (float(m[0] / max(m[1], 1e-6)), float(m[2] / max(m[1], 1e-6)))
    else:
        hi = float((img >= CLIP_HI_LEVEL).mean())
        lo = float((img <= CLIP_LO_LEVEL).mean())
        balance = (1.0, 1.0)
    return {"sharpness": sharpness(g), "clipped_high": hi, "clipped_low": lo,
            "mean": float(g.mean()), "balance_b_g": balance[0], "balance_r_g": balance[1]}


def capture_quality(img: np.ndarray) -> dict:
    """Is this capture inside the envelope the model was built for?  Returns stats, ok, reasons."""
    s = _stats(img)
    reasons = []
    if s["sharpness"] < MIN_SHARPNESS:
        reasons.append(f"blurred (sharpness {s['sharpness']:.1f} < {MIN_SHARPNESS:g})")
    if s["clipped_high"] > MAX_CLIP_FRACTION:
        reasons.append(f"clipped highlights ({s['clipped_high']:.1%} of pixels)")
    if s["clipped_low"] > MAX_CLIP_FRACTION:
        reasons.append(f"crushed shadows ({s['clipped_low']:.1%} of pixels)")
    if s["mean"] < MIN_MEAN:
        reasons.append(f"too dark (mean {s['mean']:.0f} < {MIN_MEAN:g})")
    if s["mean"] > MAX_MEAN:
        reasons.append(f"too bright (mean {s['mean']:.0f} > {MAX_MEAN:g})")
    return {**s, "ok": not reasons, "reasons": reasons,
            "exposure": "under" if s["mean"] < MIN_MEAN else "over" if s["mean"] > MAX_MEAN else "ok"}


@dataclass
class CaptureProfile:
    """Reference look of a good capture session; later frames are checked against it."""
    sharpness: float
    mean: float
    balance_b_g: float
    balance_r_g: float
    min_sharp_ratio: float = 0.6     # sharpness may not fall below 60 % of reference
    mean_tol: float = 0.20           # brightness within +/-20 % (relative)
    balance_tol: float = 0.08        # colour ratios within +/-0.08 (absolute)

    @classmethod
    def from_images(cls, images, **tol) -> "CaptureProfile":
        st = [_stats(i) for i in images]
        if not st:
            raise ValueError("need at least one reference image")
        med = lambda k: float(np.median([s[k] for s in st]))
        return cls(med("sharpness"), med("mean"), med("balance_b_g"), med("balance_r_g"), **tol)

    def check(self, img: np.ndarray) -> dict:
        s = _stats(img)
        reasons = []
        ratio = s["sharpness"] / max(self.sharpness, 1e-6)
        if ratio < self.min_sharp_ratio:
            reasons.append(f"blur: sharpness is {ratio:.0%} of reference")
        if abs(s["mean"] / max(self.mean, 1e-6) - 1) > self.mean_tol:
            reasons.append(f"brightness shifted: mean {s['mean']:.0f} vs reference {self.mean:.0f}")
        for key, label in (("balance_b_g", "blue"), ("balance_r_g", "red")):
            if abs(s[key] - getattr(self, key)) > self.balance_tol:
                reasons.append(f"colour cast ({label}/green {s[key]:.2f} vs {getattr(self, key):.2f})")
        if s["clipped_high"] > MAX_CLIP_FRACTION:
            reasons.append(f"clipped highlights ({s['clipped_high']:.1%} of pixels)")
        if s["clipped_low"] > MAX_CLIP_FRACTION:
            reasons.append(f"crushed shadows ({s['clipped_low']:.1%} of pixels)")
        return {**s, "sharpness_ratio": ratio, "ok": not reasons, "reasons": reasons}

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "CaptureProfile":
        return cls(**d)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "CaptureProfile":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------- lens calibration
def load_calibration(path) -> dict:
    """Read the JSON written by deploy/calibrate_camera.py (arrays become numpy)."""
    d = json.loads(Path(path).read_text())
    d["camera_matrix"] = np.array(d["camera_matrix"], np.float64)
    d["distortion"] = np.array(d["distortion"], np.float64).reshape(-1)
    return d


def undistort(img: np.ndarray, calib: dict) -> np.ndarray:
    """Remove lens distortion. calib is the dict from load_calibration (same image size as calibrated)."""
    K, D = calib["camera_matrix"], calib["distortion"]
    size = calib.get("image_size")
    if size and (img.shape[1], img.shape[0]) != tuple(size):
        sx, sy = img.shape[1] / size[0], img.shape[0] / size[1]
        K = K.copy()
        K[0] *= sx
        K[1] *= sy
    return cv2.undistort(img, K, D)
