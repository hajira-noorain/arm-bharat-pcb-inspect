"""Camera-condition degradations, used both to TRAIN a camera-robust detector and to TEST robustness.

All degradations are photometric or resolution-only, so bounding boxes stay valid:
  defocus  Gaussian blur (sigma in px)
  noise    additive Gaussian sensor noise (sigma in grey levels)
  jpeg     JPEG re-compression (quality)
  scale    down-sample then up-sample (fewer pixels per mm, same image size)
  light    brightness/contrast/gamma shift
  tint     white-balance colour cast
The public datasets are clean scans; a Pi camera or webcam over a bench is not. This quantifies
(and trains against) that gap without claiming it replaces real-board validation.

Build an augmented training set:
  python training/camera_aug.py build --src <bare_yolo> --dst <bare_yolo_cam> --frac 0.6
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def defocus(im, sigma):
    return im if sigma <= 0 else cv2.GaussianBlur(im, (0, 0), sigma)


def noise(im, sigma, rng):
    if sigma <= 0:
        return im
    n = rng.normal(0, sigma, im.shape)
    return np.clip(im.astype(np.float32) + n, 0, 255).astype(np.uint8)


def jpeg(im, q):
    if q >= 100:
        return im
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def rescale(im, f):
    if f >= 1:
        return im
    h, w = im.shape[:2]
    small = cv2.resize(im, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def light(im, brightness=0.0, contrast=1.0, gamma=1.0):
    x = im.astype(np.float32) / 255.0
    x = np.clip((x - 0.5) * contrast + 0.5 + brightness / 255.0, 0, 1) ** gamma
    return (x * 255).astype(np.uint8)


def tint(im, b=1.0, g=1.0, r=1.0):
    return np.clip(im.astype(np.float32) * np.array([b, g, r], np.float32), 0, 255).astype(np.uint8)


# fixed test conditions (robustness sweep); 'webcam' combines moderate everything
CONDITIONS = {
    "clean": {},
    "blur_1": {"defocus": 1.0}, "blur_2": {"defocus": 2.0}, "blur_3": {"defocus": 3.0},
    "noise_8": {"noise": 8}, "noise_16": {"noise": 16},
    "jpeg_60": {"jpeg": 60}, "jpeg_30": {"jpeg": 30},
    "scale_0.75": {"scale": 0.75}, "scale_0.5": {"scale": 0.5},
    "dark": {"light": (-40, 0.8, 1.2)}, "bright": {"light": (40, 0.9, 0.8)},
    "tint_warm": {"tint": (0.85, 1.0, 1.15)},
    "webcam": {"defocus": 1.2, "noise": 8, "jpeg": 60, "scale": 0.75, "light": (-15, 0.9, 1.1)},
}


def apply(im, cond: dict, rng: np.random.Generator):
    if "scale" in cond:
        im = rescale(im, cond["scale"])
    if "defocus" in cond:
        im = defocus(im, cond["defocus"])
    if "light" in cond:
        im = light(im, *cond["light"])
    if "tint" in cond:
        im = tint(im, *cond["tint"])
    if "noise" in cond:
        im = noise(im, cond["noise"], rng)
    if "jpeg" in cond:
        im = jpeg(im, cond["jpeg"])
    return im


def random_condition(rng: random.Random) -> dict:
    """Random mix of degradations for training (each applied with some probability)."""
    c = {}
    if rng.random() < 0.6:
        c["defocus"] = rng.uniform(0.3, 2.5)
    if rng.random() < 0.6:
        c["noise"] = rng.uniform(2, 15)
    if rng.random() < 0.5:
        c["jpeg"] = rng.randint(35, 90)
    if rng.random() < 0.5:
        c["scale"] = rng.uniform(0.5, 0.95)
    if rng.random() < 0.6:
        c["light"] = (rng.uniform(-45, 45), rng.uniform(0.7, 1.2), rng.uniform(0.75, 1.3))
    if rng.random() < 0.4:
        c["tint"] = (rng.uniform(0.85, 1.15), rng.uniform(0.9, 1.1), rng.uniform(0.85, 1.15))
    return c


def build(src: Path, dst: Path, frac: float, seed: int = 0):
    """Copy the dataset; add a degraded copy of `frac` of the TRAIN tiles (val stays clean)."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.cache"))
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    img_dir, lbl_dir = dst / "images" / "train", dst / "labels" / "train"
    n = 0
    for p in sorted(img_dir.glob("*.jpg")):
        if rng.random() > frac:
            continue
        im = apply(cv2.imread(str(p)), random_condition(rng), nrng)
        cv2.imwrite(str(img_dir / f"{p.stem}_cam.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        shutil.copy(lbl_dir / f"{p.stem}.txt", lbl_dir / f"{p.stem}_cam.txt")
        n += 1
    y = (dst / "data.yaml").read_text().splitlines()
    y[0] = f"path: {dst.as_posix()}"
    (dst / "data.yaml").write_text("\n".join(y) + "\n")
    print(f"added {n} degraded training tiles -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--frac", type=float, default=0.6)
    a = ap.parse_args()
    build(Path(a.src), Path(a.dst), a.frac)
