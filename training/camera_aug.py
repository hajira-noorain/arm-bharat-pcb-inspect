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


# Board finishes seen in real photos, BGR: (substrate under mask, copper under mask). Mask colours are
# typical of commercial boards; bare copper / HASL rows model unmasked boards (DeepPCB-like bare copper).
FINISHES = {
    "green":  ((40, 95, 30), (70, 170, 90)),
    "dark_green": ((25, 60, 20), (55, 120, 60)),
    "blue":   ((110, 50, 15), (170, 110, 60)),
    "red":    ((25, 25, 130), (70, 80, 200)),
    "black":  ((22, 22, 22), (60, 60, 60)),
    "white":  ((215, 220, 220), (150, 180, 195)),
    "yellow": ((30, 170, 200), (80, 210, 230)),
    "purple": ((90, 30, 70), (150, 90, 130)),
    "bare_copper": ((95, 135, 150), (70, 125, 205)),   # FR4 laminate, copper
    "hasl":   ((80, 120, 130), (185, 185, 180)),         # FR4, tinned (silver) copper
}


def recolor_binary(im: np.ndarray, rng: random.Random, nrng: np.random.Generator) -> np.ndarray:
    """Re-paint a binarised scan (copper dark) as a photographed board: a random finish, per-region colour
    jitter, a soft copper edge, uneven illumination, a faint specular band and sensor noise. The geometry
    (and therefore every label) is unchanged. TRAINING augmentation only, never used to make test images."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
    cu = (g < 128).astype(np.float32)
    cu = cv2.GaussianBlur(cu, (0, 0), rng.uniform(0.6, 1.6))            # soft, slightly rounded edges
    sub_c, cu_c = FINISHES[rng.choice(list(FINISHES))]
    jit = lambda c: np.clip(np.array(c, np.float32) * rng.uniform(0.8, 1.2) + nrng.normal(0, 8, 3), 0, 255)  # noqa: E731
    sub_c, cu_c = jit(sub_c), jit(cu_c)
    if np.abs(sub_c - cu_c).sum() < 90:   # keep copper visibly distinct from the substrate, as on real boards
        cu_c = np.clip(cu_c + np.sign(cu_c - sub_c + 1e-3) * 35, 0, 255)
    out = sub_c[None, None, :] * (1 - cu[..., None]) + cu_c[None, None, :] * cu[..., None]
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = 1 + rng.uniform(-0.12, 0.12) * ((xx / w - 0.5) * np.cos(ang) + (yy / h - 0.5) * np.sin(ang)) * 2
    out *= grad[..., None]
    if rng.random() < 0.5:   # a glossy band, as from a ring light on solder mask
        pos = rng.uniform(0, 1)
        d = np.abs((xx / w) * np.cos(ang + 1.3) + (yy / h) * np.sin(ang + 1.3) - pos)
        peak = rng.uniform(10, 45) * (1.0 - out.mean() / 400.0)   # weaker glare on already-bright finishes
        out += (peak * np.exp(-(d / rng.uniform(0.03, 0.12)) ** 2))[..., None]
    out += nrng.normal(0, rng.uniform(2, 7), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


RECOLOR_SOURCES = ("deeppcb_",)   # tiles from binarised scans (their file names start with the source)


def build(src: Path, dst: Path, frac: float, seed: int = 0, recolor_frac: float = 0.0):
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
    n_rc = 0
    if recolor_frac > 0:   # extra copies of binarised-scan tiles repainted as photographed boards
        for p in sorted(img_dir.glob("*.jpg")):
            if p.stem.endswith("_cam") or not p.name.startswith(RECOLOR_SOURCES) or rng.random() > recolor_frac:
                continue
            im = recolor_binary(cv2.imread(str(p)), rng, nrng)
            im = apply(im, random_condition(rng), nrng)
            cv2.imwrite(str(img_dir / f"{p.stem}_rc.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            shutil.copy(lbl_dir / f"{p.stem}.txt", lbl_dir / f"{p.stem}_rc.txt")
            n_rc += 1
    print(f"added {n_rc} recoloured tiles")
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
    ap.add_argument("--recolor-frac", type=float, default=0.0, help="repaint this share of binarised-scan tiles as photographed boards")
    a = ap.parse_args()
    build(Path(a.src), Path(a.dst), a.frac, recolor_frac=a.recolor_frac)
