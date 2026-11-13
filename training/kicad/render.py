"""Render defect/golden board pairs and cut labelled 640 px tiles (runs on Colab; pure parts are unit-tested locally).

Pipeline per board (see ``process_board``):
  1. inject defects into the .kicad_pcb text -> defective text; the untouched text is the golden twin.
  2. ``kicad-cli pcb export svg`` per layer (F.Cu, F.Mask, F.SilkS, Edge.Cuts), page = board outline, no sheet.
  3. ``render_2d``: rasterise the layer SVGs with cairosvg, then composite a top-down photo-like image from the
     layer masks with randomised appearance (mask colour, copper finish, silkscreen, lighting, camera).  The
     composite is a pure function of (layer masks, appearance, tile position), so clean and defective renders
     share identical appearance AND identical noise; they differ only where the design differs.
  4. cut 640 px tiles that contain defects (plus clean tiles), write YOLO labels with OUR class indices.

Board mm -> pixel mapping (exact, documented)
---------------------------------------------
Image pixel (col, row) covers board region x in [x0 + col/ppm, x0 + (col+1)/ppm), y in [y0 + row/ppm, ...), y down,
same orientation as KiCad (no flips).  ppm is ``W / svg_width_mm`` (the SVG is rasterised to an integer number of
pixels, so ppm differs from the requested value by <0.1 %).  x0,y0 are NOT assumed from KiCad internals; they are
MEASURED per board by cross-correlating the F.Cu SVG raster with an independent numpy rasterisation of the same
tracks/pads/vias computed from the .kicad_pcb text (``calibrate``); the residual shift is recorded in the manifest.
A defect's bbox_mm maps to pixels with ``Mapping.bbox_px``.  Every defect is additionally verified by differencing the
clean and defective layer rasters (``qa_defects``): the changed pixels must sit inside the labelled box, else the
defect (or the whole board, if the mass outside boxes is large) is dropped and logged.

``kicad-cli pcb render`` (3D ray-traced, KiCad 9) exists and ``render_3d`` wraps it, but it is NOT wired into ``main``:
the 2D path (``render_2d``) is the pipeline.  A 3D image would need its own pixel<->mm calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from training.kicad import inject as I          # noqa: E402
from training.kicad import sources as S         # noqa: E402


def bare_classes() -> tuple[str, ...]:
    """Our class order, imported from pcbinspect.record (never hard-coded)."""
    from pcbinspect.record import BARE_CLASSES
    return BARE_CLASSES


# ======================================================================================================
# mapping + numpy rasterisation (also used by the tests, no KiCad / cairosvg needed)
# ======================================================================================================
@dataclass
class Mapping:
    ppm_x: float
    ppm_y: float
    x0: float            # board mm at the left edge of pixel column 0
    y0: float            # board mm at the top edge of pixel row 0
    w: int
    h: int

    def px(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.x0) * self.ppm_x, (y - self.y0) * self.ppm_y

    def bbox_px(self, bb) -> tuple[float, float, float, float]:
        a, b = self.px(bb[0], bb[1])
        c, d = self.px(bb[2], bb[3])
        return a, b, c, d

    def mm(self, col: float, row: float) -> tuple[float, float]:
        return self.x0 + col / self.ppm_x, self.y0 + row / self.ppm_y

    @staticmethod
    def from_edge(board: "I.Board", ppm: float, margin_mm: float = 0.0) -> "Mapping":
        e = board.edge
        x0, y0 = e[0] - margin_mm, e[1] - margin_mm
        return Mapping(ppm, ppm, x0, y0, int(math.ceil((e[2] + margin_mm - x0) * ppm)), int(math.ceil((e[3] + margin_mm - y0) * ppm)))


def _paint_capsule(img: np.ndarray, m: Mapping, ax, ay, bx, by, r_mm) -> None:
    pax, pay = m.px(ax, ay)
    pbx, pby = m.px(bx, by)
    rp = r_mm * m.ppm_x
    x1, x2 = int(math.floor(min(pax, pbx) - rp - 1)), int(math.ceil(max(pax, pbx) + rp + 1))
    y1, y2 = int(math.floor(min(pay, pby) - rp - 1)), int(math.ceil(max(pay, pby) + rp + 1))
    x1c, x2c, y1c, y2c = max(x1, 0), min(x2, m.w), max(y1, 0), min(y2, m.h)
    if x1c >= x2c or y1c >= y2c:
        return
    xs = np.arange(x1c, x2c, dtype=np.float32)[None, :] + 0.5
    ys = np.arange(y1c, y2c, dtype=np.float32)[:, None] + 0.5
    dx, dy = pbx - pax, pby - pay
    L2 = dx * dx + dy * dy
    t = np.zeros((1, 1), np.float32) if L2 < 1e-12 else np.clip(((xs - pax) * dx + (ys - pay) * dy) / L2, 0, 1)
    d = np.hypot(xs - (pax + t * dx), ys - (pay + t * dy))
    cov = np.clip(rp - d + 0.5, 0, 1).astype(img.dtype)
    sl = img[y1c:y2c, x1c:x2c]
    np.maximum(sl, cov, out=sl)


def _paint_box(img: np.ndarray, m: Mapping, x1, y1, x2, y2) -> None:
    a, b = m.px(x1, y1)
    c, d = m.px(x2, y2)
    xi1, xi2, yi1, yi2 = max(int(math.floor(a)), 0), min(int(math.ceil(c)), m.w), max(int(math.floor(b)), 0), min(int(math.ceil(d)), m.h)
    if xi1 >= xi2 or yi1 >= yi2:
        return
    xs = np.arange(xi1, xi2, dtype=np.float32)
    ys = np.arange(yi1, yi2, dtype=np.float32)
    cx = np.clip(np.minimum(xs + 1, c) - np.maximum(xs, a), 0, 1)[None, :]
    cy = np.clip(np.minimum(ys + 1, d) - np.maximum(ys, b), 0, 1)[:, None]
    sl = img[yi1:yi2, xi1:xi2]
    np.maximum(sl, (cx * cy).astype(img.dtype), out=sl)


def _paint_poly(img: np.ndarray, m: Mapping, pts, ss: int = 4) -> None:
    from PIL import Image, ImageDraw
    P = [m.px(x, y) for x, y in pts]
    x1, x2 = int(math.floor(min(p[0] for p in P))) - 1, int(math.ceil(max(p[0] for p in P))) + 1
    y1, y2 = int(math.floor(min(p[1] for p in P))) - 1, int(math.ceil(max(p[1] for p in P))) + 1
    x1c, x2c, y1c, y2c = max(x1, 0), min(x2, m.w), max(y1, 0), min(y2, m.h)
    if x1c >= x2c or y1c >= y2c:
        return
    im = Image.new("L", ((x2c - x1c) * ss, (y2c - y1c) * ss), 0)
    ImageDraw.Draw(im).polygon([((px - x1c) * ss, (py - y1c) * ss) for px, py in P], fill=255)
    cov = np.asarray(im, np.float32).reshape(y2c - y1c, ss, x2c - x1c, ss).mean((1, 3)) / 255.0
    sl = img[y1c:y2c, x1c:x2c]
    np.maximum(sl, cov.astype(img.dtype), out=sl)


def raster_copper(board: "I.Board", m: Mapping) -> np.ndarray:
    """Float32 coverage (0..1) of F.Cu tracks, vias, top pads and gr_poly blobs.  Zone fills are not drawn."""
    img = np.zeros((m.h, m.w), np.float32)
    for s in board.segs:
        if s.layer == "F.Cu":
            _paint_capsule(img, m, *s.a, *s.b, s.w / 2)
    for v in board.vias:
        _paint_capsule(img, m, v.x, v.y, v.x, v.y, v.size / 2)
    for p in board.pads:
        if not (p.on_top and p.ptype in ("smd", "thru_hole", "connect")):
            continue
        if p.shape == "circle":
            _paint_capsule(img, m, p.x, p.y, p.x, p.y, p.hx)
        elif p.shape == "oval":
            if p.hx >= p.hy:
                _paint_capsule(img, m, p.x - (p.hx - p.hy), p.y, p.x + (p.hx - p.hy), p.y, p.hy)
            else:
                _paint_capsule(img, m, p.x, p.y - (p.hy - p.hx), p.x, p.y + (p.hy - p.hx), p.hx)
        else:
            _paint_box(img, m, *p.box)
    for g in board.root.findall("gr_poly"):
        if g.val("layer", "") == "F.Cu":
            pts = I._poly_pts(g)
            if len(pts) >= 3:
                _paint_poly(img, m, pts)
    return img


def raster_holes(board: "I.Board", m: Mapping) -> np.ndarray:
    """Coverage of drilled holes (vias + pads with a drill), from the board text; defect-aware (missing_hole)."""
    img = np.zeros((m.h, m.w), np.float32)
    for v in board.vias:
        if v.drill > 0:
            _paint_capsule(img, m, v.x, v.y, v.x, v.y, v.drill / 2)
    for p in board.pads:
        if p.drill and p.drill > 0 and p.ptype in ("thru_hole", "np_thru_hole"):
            _paint_capsule(img, m, p.x, p.y, p.x, p.y, p.drill / 2)
    return img


def changed_bbox_mm(a: np.ndarray, b: np.ndarray, m: Mapping, thr: float = 0.25) -> tuple | None:
    """Bounding box (board mm) of pixels where two coverage rasters differ by more than ``thr``."""
    ys, xs = np.nonzero(np.abs(a - b) > thr)
    if not len(xs):
        return None
    x1, y1 = m.mm(xs.min(), ys.min())
    x2, y2 = m.mm(xs.max() + 1, ys.max() + 1)
    return x1, y1, x2, y2


# ======================================================================================================
# appearance + 2D compositing
# ======================================================================================================
MASK_COLOURS = {                      # name -> (rgb 0..1 of mask over substrate, weight)
    "green": ((0.05, 0.36, 0.16), 5), "dark_green": ((0.03, 0.22, 0.11), 2), "blue": ((0.05, 0.20, 0.55), 2),
    "red": ((0.62, 0.07, 0.07), 2), "black": ((0.045, 0.045, 0.05), 2), "white": ((0.86, 0.86, 0.84), 1),
    "yellow": ((0.80, 0.66, 0.08), 1), "purple": ((0.30, 0.08, 0.42), 1),
}
FINISHES = {"hasl": (0.74, 0.74, 0.72), "enig": (0.80, 0.62, 0.22), "osp": (0.70, 0.42, 0.28), "bare_copper": (0.72, 0.44, 0.25)}


def sample_appearance(rng: random.Random) -> dict:
    names = list(MASK_COLOURS)
    mask = rng.choices(names, [MASK_COLOURS[n][1] for n in names])[0]
    fin = rng.choice(list(FINISHES))
    dark_mask = sum(MASK_COLOURS[mask][0]) < 0.6
    silk_on = rng.random() < 0.7
    silk = (0.93, 0.93, 0.9) if mask != "white" else (0.08, 0.08, 0.08)
    if mask == "yellow" and rng.random() < 0.5:
        silk = (0.08, 0.08, 0.08)
    return {
        "mask": mask, "mask_rgb": [round(v * rng.uniform(0.9, 1.1), 4) for v in MASK_COLOURS[mask][0]],
        "finish": fin, "finish_rgb": [round(v * rng.uniform(0.92, 1.08), 4) for v in FINISHES[fin]],
        "silk_on": silk_on, "silk_rgb": list(silk), "substrate_rgb": [0.72, 0.62, 0.36],
        "hole_rgb": [0.03, 0.03, 0.035] if rng.random() < 0.7 else [0.55, 0.52, 0.45],
        "background_rgb": rng.choice([[0.06, 0.06, 0.07], [0.45, 0.45, 0.46], [0.85, 0.85, 0.83], [0.20, 0.17, 0.12]]),
        "light_angle_deg": round(rng.uniform(0, 360), 1), "light_gradient": round(rng.uniform(0.0, 0.35), 3),
        "vignette": round(rng.uniform(0.0, 0.3), 3), "brightness": round(rng.uniform(0.8, 1.2), 3),
        "relief": round(rng.uniform(0.15, 0.5) * (0.6 if dark_mask else 1.0), 3),
        "texture": round(rng.uniform(0.01, 0.05), 4), "gamma": round(rng.uniform(0.8, 1.25), 3),
        "white_balance": [round(rng.uniform(0.92, 1.08), 3) for _ in range(3)],
        "blur_px": round(rng.uniform(0.0, 1.1), 2), "noise": round(rng.uniform(0.0, 0.025), 4),
        "jpeg_quality": rng.choice([75, 85, 92, 95]), "seed": rng.getrandbits(31),
    }


def compose_tile(layers: dict, app: dict, origin_px: tuple[int, int], full_px: tuple[int, int], ppm: float, margin: int = 0) -> np.ndarray:
    """Pure function of (layer arrays, appearance, absolute tile position).  Layer arrays are float32 (0..1) with
    keys cu, open (mask openings), silk, board, holes, all of the same shape (tile + ``margin`` on every side).
    Returns uint8 RGB with the margin cropped off."""
    from scipy.ndimage import gaussian_filter
    cu, mo, sk, bd, ho = (layers[k] for k in ("cu", "open", "silk", "board", "holes"))
    h, w = cu.shape
    gx0, gy0 = origin_px[0] - margin, origin_px[1] - margin
    rng = np.random.default_rng([app["seed"], max(gx0, 0) + 100000 * max(gy0, 0) + 7])
    mask_rgb = np.array(app["mask_rgb"], np.float32)
    fin = np.array(app["finish_rgb"], np.float32)
    sub = np.array(app["substrate_rgb"], np.float32)
    bgc = np.array(app["background_rgb"], np.float32)
    covered = (1 - mo)
    cu_under = cu * covered                                          # copper under the mask
    base = mask_rgb * 0.78                                           # mask over laminate
    lit = np.clip(mask_rgb * 1.28 + 0.045, 0, 1)                     # mask over copper looks brighter
    img = base[None, None, :] * (1 - cu_under[..., None]) + lit[None, None, :] * cu_under[..., None]
    # relief shading from the copper height field (light from angle)
    s = max(0.8, 0.05 * ppm)
    hf = gaussian_filter(cu, s)
    gy, gx = np.gradient(hf)
    a = math.radians(app["light_angle_deg"])
    shade = (gx * math.cos(a) + gy * math.sin(a)) * s * 2.2 * app["relief"] * 4
    img *= (1 + np.clip(shade, -0.5, 0.5))[..., None] * covered[..., None] + (1 - covered[..., None])
    # exposed laminate / exposed copper in mask openings
    tex = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), max(1.0, 0.03 * ppm))
    tex /= max(float(tex.std()), 1e-6)
    fin_img = fin[None, None, :] * (1 + (app["texture"] * 3) * tex[..., None])
    img = img * (1 - mo[..., None]) + mo[..., None] * ((1 - cu)[..., None] * sub + cu[..., None] * fin_img)
    if app["silk_on"]:
        sv = sk * covered
        img = img * (1 - sv[..., None]) + sv[..., None] * np.array(app["silk_rgb"], np.float32)
    img = img * (1 - ho[..., None]) + ho[..., None] * np.array(app["hole_rgb"], np.float32)
    img *= (1 + app["texture"] * tex)[..., None]
    # outside the board
    img = img * bd[..., None] + (1 - bd[..., None]) * bgc
    # lighting in absolute board pixel coordinates
    W, H = full_px
    yy, xx = np.mgrid[gy0:gy0 + h, gx0:gx0 + w].astype(np.float32)
    u, v = xx / max(W, 1) - 0.5, yy / max(H, 1) - 0.5
    light = 1 + app["light_gradient"] * (u * math.cos(a) + v * math.sin(a)) * 2 - app["vignette"] * (u * u + v * v) * 2
    img *= (light * app["brightness"])[..., None]
    # camera
    if app["blur_px"] > 0.05:
        img = np.stack([gaussian_filter(img[..., c], app["blur_px"]) for c in range(3)], -1)
    img *= np.array(app["white_balance"], np.float32)
    img = np.clip(img, 0, 1) ** app["gamma"]
    if app["noise"] > 0:
        img += rng.standard_normal(img.shape).astype(np.float32) * app["noise"]
    img = np.clip(img, 0, 1)
    if margin:
        img = img[margin:h - margin, margin:w - margin]
    return (img * 255 + 0.5).astype(np.uint8)


# ======================================================================================================
# KiCad CLI plumbing
# ======================================================================================================
LAYERS = ("F.Cu", "F.Mask", "F.SilkS", "Edge.Cuts")


def have_cli() -> bool:
    return shutil.which("kicad-cli") is not None


def export_svg_layers(board_path: Path, outdir: Path, timeout: int = 300) -> dict[str, Path]:
    """One SVG per layer: ``kicad-cli pcb export svg --mode-multi``, page sized to the board, no drawing sheet,
    black-and-white (each layer's ink is black on transparent), no drill marks (holes are drawn from the board text)."""
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = ["kicad-cli", "pcb", "export", "svg", "--mode-multi", "--layers", ",".join(LAYERS), "--page-size-mode", "2",
           "--exclude-drawing-sheet", "--black-and-white", "--drill-shape-opt", "0", "--output", str(outdir), str(board_path)]
    if shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a"] + cmd
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    files = {}
    for lay in LAYERS:
        suffix = lay.replace(".", "_") + ".svg"
        hit = [p for p in outdir.glob("*.svg") if p.name.endswith(suffix)]
        if hit:
            files[lay] = hit[0]
    if r.returncode != 0 or len(files) < len(LAYERS):
        raise RuntimeError(f"kicad-cli export failed rc={r.returncode} files={sorted(files)} err={r.stderr[-400:]}")
    return files


def render_3d(board_path: Path, out_png: Path, width: int = 2400, height: int = 1600, side: str = "top",
              quality: str = "basic", light: float | None = None, extra: list[str] | None = None, timeout: int = 900) -> None:
    """``kicad-cli pcb render`` (KiCad 9.0): ray-traced 3D render.  Options used: --width --height --side --quality
    --background --floor --light-top/--light-camera, optional --perspective --zoom --rotate.  Needs a GL-less software
    path; under xvfb-run on Colab it works on CPU.  Standalone helper, not used by ``main``."""
    cmd = ["kicad-cli", "pcb", "render", "--output", str(out_png), "--width", str(width), "--height", str(height),
           "--side", side, "--quality", quality, "--background", "opaque"]
    if light is not None:
        cmd += ["--light-top", str(light), "--light-camera", str(light)]
    cmd += (extra or []) + [str(board_path)]
    if shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a"] + cmd
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not out_png.exists():
        raise RuntimeError(f"kicad-cli pcb render failed rc={r.returncode} {r.stderr[-300:]}")


def _svg_size_mm(svg_text: str) -> tuple[float, float]:
    def num(attr):
        m = re.search(attr + r'="([0-9.eE+-]+)\s*(mm|cm|in|px)?"', svg_text[:2000])
        if not m:
            raise ValueError(f"svg has no {attr}")
        v = float(m.group(1))
        return v * {"mm": 1.0, "cm": 10.0, "in": 25.4, "px": 25.4 / 96, None: 25.4 / 96, "": 25.4 / 96}[m.group(2)]
    return num("width"), num("height")


def svg_to_cov(svg_path: Path, w: int, h: int) -> np.ndarray:
    """Rasterise an SVG layer to float32 ink coverage: alpha * darkness (black ink on transparent)."""
    import cairosvg
    from PIL import Image
    png = cairosvg.svg2png(url=str(svg_path), output_width=w, output_height=h)
    im = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"), np.float32) / 255.0
    lum = im[..., :3].mean(-1)
    return (im[..., 3] * (1.0 - lum)).astype(np.float32)


_SVG_TOK = re.compile(r"[MLA]|-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


def svg_edge_points(svg_text: str) -> list[tuple[float, float]]:
    """End points of every Edge.Cuts path in the SVG (mm, page coordinates): M/L points and arc end points."""
    pts = []
    for d in re.findall(r'\sd="([^"]+)"', svg_text):
        toks, i, cmd = _SVG_TOK.findall(d), 0, None
        while i < len(toks):
            t = toks[i]
            if t in ("M", "L", "A"):
                cmd, i = t, i + 1
                continue
            try:
                if cmd in ("M", "L"):
                    pts.append((float(toks[i]), float(toks[i + 1])))
                    i += 2
                elif cmd == "A":                      # rx ry rotation large-arc sweep x y
                    pts.append((float(toks[i + 5]), float(toks[i + 6])))
                    i += 7
                else:
                    i += 1
            except (IndexError, ValueError):
                break
    return pts


def board_edge_points(board_text: str) -> list[tuple[float, float]]:
    """The same primitives from the board file: Edge.Cuts line/arc start+end, rectangle corners, polygon points."""
    root = I.Board(board_text).root
    pts = []
    for k in root.nodes():
        if not k.name.startswith("gr_") or k.val("layer", "") != "Edge.Cuts":
            continue
        st, en = k.floats("start"), k.floats("end")
        if k.name in ("gr_line", "gr_arc") and len(st) >= 2 and len(en) >= 2 and not k.floats("angle"):
            pts += [(st[0], st[1]), (en[0], en[1])]
        elif k.name == "gr_rect" and len(st) >= 2 and len(en) >= 2:
            pts += [(st[0], st[1]), (en[0], st[1]), (en[0], en[1]), (st[0], en[1])]
        elif k.name == "gr_poly":
            pts += list(I._poly_pts(k))
    return pts


def calibrate_from_edges(svg_edge_text: str, board_text: str, tol: float = 0.02) -> tuple[float, float, float] | None:
    """Exact page offset from the board outline: the SVG page is the board area plus a margin that depends on
    every item on the board (so it cannot be predicted), but the outline primitives appear in both files in
    millimetres with no rotation or scale. Returns (x0, y0, 1.0) = board mm of the page's top-left corner, or None
    when the two outlines do not agree (e.g. circle outlines, v5 arcs) so the caller can fall back."""
    ps, pb = svg_edge_points(svg_edge_text), board_edge_points(board_text)
    if len(ps) < 2 or len(pb) < 2:
        return None
    sx, sy = zip(*ps)
    bx, by = zip(*pb)
    tx, ty = min(sx) - min(bx), min(sy) - min(by)
    if abs((max(sx) - max(bx)) - tx) > tol or abs((max(sy) - max(by)) - ty) > tol:
        return None
    return -tx, -ty, 1.0


def calibrate(svg_cu: np.ndarray, own: np.ndarray, ppm: float, gx: float, gy: float) -> tuple[float, float, float]:
    """Find x0,y0 (board mm of the SVG raster's top-left corner) by correlating it with the independent raster
    ``own`` (same shape, built with origin gx,gy).  Returns (x0, y0, correlation score 0..1)."""
    from scipy.signal import fftconvolve
    f = 8
    h, w = svg_cu.shape

    def ds(a):
        return a[:h // f * f, :w // f * f].reshape(h // f, f, w // f, f).mean((1, 3))
    A, B = ds(svg_cu), ds(own)
    A, B = A - A.mean(), B - B.mean()
    c = fftconvolve(A, B[::-1, ::-1], mode="full")
    ky, kx = np.unravel_index(int(np.argmax(c)), c.shape)
    lag_y, lag_x = (ky - (B.shape[0] - 1)) * f, (kx - (B.shape[1] - 1)) * f
    cs = 1200
    pad = f + 3
    # crop (in svg coordinates) inside the region where svg and shifted own overlap, whatever the coarse lag
    ox1, ox2 = max(lag_x, 0) + pad, min(w, w + lag_x) - pad
    oy1, oy2 = max(lag_y, 0) + pad, min(h, h + lag_y) - pad
    if ox2 - ox1 < 16 or oy2 - oy1 < 16:
        return gx, gy, 0.0
    cw, ch = min(cs, ox2 - ox1), min(cs, oy2 - oy1)
    x0c, y0c = (ox1 + ox2 - cw) // 2, (oy1 + oy2 - ch) // 2
    best, grid = -1.0, {}
    for dy in range(lag_y - pad, lag_y + pad + 1):
        for dx in range(lag_x - pad, lag_x + pad + 1):
            yo, xo = y0c - dy, x0c - dx
            if yo < 0 or xo < 0 or yo + ch > h or xo + cw > w:
                continue
            sc = float((svg_cu[y0c:y0c + ch, x0c:x0c + cw] * own[yo:yo + ch, xo:xo + cw]).sum())
            grid[(dx, dy)] = sc
            if sc > best:
                best, bd = sc, (dx, dy)
    if not grid:
        return gx, gy, 0.0
    dx, dy = bd

    def sub(axis):
        g = lambda ddx, ddy: grid.get((dx + ddx, dy + ddy))
        l, c0, r = (g(-1, 0), g(0, 0), g(1, 0)) if axis == 0 else (g(0, -1), g(0, 0), g(0, 1))
        if l is None or r is None or l + r - 2 * c0 >= 0:
            return 0.0
        return 0.5 * (l - r) / (l + r - 2 * c0)
    sx, sy = dx + sub(0), dy + sub(1)
    a = svg_cu[y0c:y0c + ch, x0c:x0c + cw]
    b = own[y0c - dy:y0c - dy + ch, x0c - dx:x0c - dx + cw]
    den = float(np.sqrt((a * a).sum() * (b * b).sum())) or 1.0
    return gx - sx / ppm, gy - sy / ppm, float((a * b).sum() / den)


@dataclass
class RenderedBoard:
    mapping: Mapping
    layers: dict            # uint8 arrays: cu, open, silk, board, holes
    calib_score: float

    def tile_layers(self, x: int, y: int, size: int, margin: int) -> dict:
        out = {}
        for k, a in self.layers.items():
            t = np.zeros((size + 2 * margin,) * 2, np.float32)
            xs, ys = x - margin, y - margin
            sx1, sy1 = max(xs, 0), max(ys, 0)
            sx2, sy2 = min(xs + size + 2 * margin, a.shape[1]), min(ys + size + 2 * margin, a.shape[0])
            if sx2 > sx1 and sy2 > sy1:
                t[sy1 - ys:sy2 - ys, sx1 - xs:sx2 - xs] = a[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0
            out[k] = t
        return out


def render_2d(board_text: str, work: Path, name: str, ppm: float, max_mpx: float = 60.0) -> RenderedBoard:
    """Export + rasterise the layer masks of one board version (clean or defective) and calibrate the mapping."""
    p = work / f"{name}.kicad_pcb"
    p.write_text(board_text, encoding="utf8", newline="")
    files = export_svg_layers(p, work / f"{name}_svg")
    wmm, hmm = _svg_size_mm(files["F.Cu"].read_text(encoding="utf8", errors="replace"))
    ppm = min(ppm, math.sqrt(max_mpx * 1e6 / (wmm * hmm)))
    W, H = int(round(wmm * ppm)), int(round(hmm * ppm))
    ppm_x, ppm_y = W / wmm, H / hmm
    cu = svg_to_cov(files["F.Cu"], W, H)
    b = I.Board(board_text)
    e = b.edge or (0, 0, wmm, hmm)
    m0 = Mapping(ppm_x, ppm_y, e[0], e[1], W, H)
    exact = calibrate_from_edges(files["Edge.Cuts"].read_text(encoding="utf8", errors="replace"), board_text)
    if exact is not None:    # primary: exact offset from the outline (copper pours make correlation unreliable)
        x0, y0, score = exact
    else:                    # fallback: correlate the copper layer with an independent rasterisation
        own = raster_copper(b, m0)
        x0, y0, score = calibrate(cu, own, ppm_x, e[0], e[1])
    m = Mapping(ppm_x, ppm_y, x0, y0, W, H)
    from scipy.ndimage import binary_fill_holes
    edge = svg_to_cov(files["Edge.Cuts"], W, H)
    board = binary_fill_holes(edge > 0.3)
    if board.mean() < 0.02 or board.mean() > 0.999:
        board = np.ones_like(board)
    layers = {
        "cu": cu, "open": svg_to_cov(files["F.Mask"], W, H), "silk": svg_to_cov(files["F.SilkS"], W, H),
        "board": board.astype(np.float32), "holes": raster_holes(b, m),
    }
    return RenderedBoard(m, {k: (np.clip(v, 0, 1) * 255 + 0.5).astype(np.uint8) for k, v in layers.items()}, score)


# ======================================================================================================
# QA, tiling, labels
# ======================================================================================================
def visual_change(a: RenderedBoard, b: RenderedBoard) -> np.ndarray:
    """Per-pixel change between clean ``a`` and defective ``b`` over the layers that determine the picture."""
    d = np.zeros(a.layers["cu"].shape, np.float32)
    for k in ("cu", "holes", "open"):
        d += np.abs(a.layers[k].astype(np.int16) - b.layers[k].astype(np.int16)).astype(np.float32) / 255.0
    return d


def qa_defects(clean: RenderedBoard, bad: RenderedBoard, defects: list[dict], pad_px: int = 4) -> tuple[list[dict], dict]:
    """Keep defects whose layer change lies inside their labelled box (+pad_px).  Returns (kept, stats)."""
    d = visual_change(clean, bad)
    m = clean.mapping
    total = float((d > 0.25).sum())
    inside_mask = np.zeros(d.shape, bool)
    kept, dropped = [], []
    for df in defects:
        x1, y1, x2, y2 = m.bbox_px(df["bbox_mm"])
        a, b, c, e = max(int(x1) - pad_px, 0), max(int(y1) - pad_px, 0), min(int(math.ceil(x2)) + pad_px, d.shape[1]), min(int(math.ceil(y2)) + pad_px, d.shape[0])
        win = d[b:e, a:c]
        n = int((win > 0.25).sum())
        inside_mask[b:e, a:c] = True
        if n >= 3:
            kept.append(df)
        else:
            dropped.append({"cls": df["cls"], "bbox_mm": df["bbox_mm"], "reason": "no visible change in box"})
    outside = float(((d > 0.25) & ~inside_mask).sum())
    stats = {"changed_px": total, "outside_box_px": outside, "dropped": dropped,
             "outside_frac": outside / total if total else 0.0}
    return kept, stats


def yolo_line(cls_idx: int, box_px: tuple, tile: int, min_px: float = 6.0) -> str:
    x1, y1, x2, y2 = box_px
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, min_px), max(y2 - y1, min_px)
    x1, x2, y1, y2 = max(cx - w / 2, 0), min(cx + w / 2, tile), max(cy - h / 2, 0), min(cy + h / 2, tile)
    return f"{cls_idx} {(x1 + x2) / 2 / tile:.6f} {(y1 + y2) / 2 / tile:.6f} {(x2 - x1) / tile:.6f} {(y2 - y1) / tile:.6f}"


def pick_tiles(defects: list[dict], m: Mapping, tile: int, rng: random.Random, max_tiles: int) -> list[tuple[int, int, list[dict]]]:
    """For each defect choose a tile origin (px) containing it fully with a random offset; other defects must be
    fully inside or fully outside (retry otherwise).  Greedy dedupe: a defect already covered by a chosen tile whose
    labels are complete is not given a second tile."""
    out, covered = [], set()
    order = list(range(len(defects)))
    rng.shuffle(order)
    for i in order:
        if i in covered or len(out) >= max_tiles:
            continue
        bx = m.bbox_px(defects[i]["bbox_mm"])
        w, h = bx[2] - bx[0], bx[3] - bx[1]
        if w > tile - 16 or h > tile - 16:
            continue
        for _ in range(12):
            tx = int(bx[0] - rng.uniform(8, tile - 8 - w))
            ty = int(bx[1] - rng.uniform(8, tile - 8 - h))
            if m.w >= tile:
                tx = min(max(tx, 0), m.w - tile)
            if m.h >= tile:
                ty = min(max(ty, 0), m.h - tile)
            inside, ok = [], True
            for j, d in enumerate(defects):
                b = m.bbox_px(d["bbox_mm"])
                ix = max(0.0, min(b[2], tx + tile) - max(b[0], tx))
                iy = max(0.0, min(b[3], ty + tile) - max(b[1], ty))
                frac = ix * iy / max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
                if frac > 0.98:
                    inside.append(j)
                elif frac > 0.0:
                    ok = False
                    break
            if ok and i in inside:
                out.append((tx, ty, [defects[j] for j in inside]))
                covered.update(inside)
                break
    return out


def pick_clean_tiles(rb: RenderedBoard, defects: list[dict], tile: int, rng: random.Random, n: int) -> list[tuple[int, int]]:
    """Clean tiles are cut from the golden render (image == golden, empty label), so they may sit anywhere on the board,
    including where a defective variant has defects."""
    m = rb.mapping
    out = []
    for _ in range(n * 40):
        if len(out) >= n:
            break
        if m.w < tile or m.h < tile:
            break
        tx, ty = rng.randrange(0, m.w - tile + 1), rng.randrange(0, m.h - tile + 1)
        sub_b = rb.layers["board"][ty:ty + tile:8, tx:tx + tile:8]
        sub_c = rb.layers["cu"][ty:ty + tile:4, tx:tx + tile:4]
        if sub_b.mean() / 255 > 0.9 and sub_c.mean() / 255 > 0.05:
            out.append((tx, ty))
    return out


def split_of(repo: str) -> str:
    h = int(hashlib.md5(repo.encode()).hexdigest(), 16) % 10
    return "val" if h == 0 else "test" if h == 1 else "train"


def save_jpg(arr: np.ndarray, path: Path, q: int) -> None:
    from PIL import Image
    Image.fromarray(arr).save(path, quality=q, subsampling=0)


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:70]


def process_board(src: dict, text: str, out: Path, work: Path, args, classes: tuple[str, ...]) -> dict:
    """Full pipeline for one board; returns its manifest entry."""
    bid = slug(src["repo"] + "-" + Path(src["path"]).stem)
    split = split_of(src["repo"])
    entry = {"id": bid, "repo": src["repo"], "licence": src["licence"], "commit": src["commit"], "path": src["path"],
             "source_url": S.url(src), "split": split, "tiles": [], "status": "ok"}
    rng = random.Random(f"{args.seed}/{bid}")
    try:
        clean = render_2d(text, work, bid + "_clean", args.ppm)
    except Exception as ex:                                   # noqa: BLE001
        entry.update(status="skip", reason=f"clean render failed: {ex}")
        return entry
    if clean.calib_score < 0.5:
        entry.update(status="skip", reason=f"mapping calibration weak (corr {clean.calib_score:.2f})")
        return entry
    entry["mapping"] = {"ppm": clean.mapping.ppm_x, "x0_mm": clean.mapping.x0, "y0_mm": clean.mapping.y0, "px": [clean.mapping.w, clean.mapping.h],
                        "calib_corr": clean.calib_score}
    apps = [sample_appearance(rng) for _ in range(args.appearances)]
    entry["appearances"] = apps
    counts = {c: args.per_class for c in I.DEFECT_CLASSES}
    tile_no = 0
    for ds in range(args.defect_sets):
        seed = rng.getrandbits(31)
        bad_text, defects = I.inject_many(text, counts, seed=seed, min_sep_mm=args.min_sep_mm)
        if not defects:
            continue
        try:
            bad = render_2d(bad_text, work, f"{bid}_set{ds}", args.ppm)
        except Exception as ex:                               # noqa: BLE001
            entry["tiles"].append({"set": ds, "skipped": f"defect render failed: {ex}"})
            continue
        if (bad.mapping.w, bad.mapping.h) != (clean.mapping.w, clean.mapping.h) or \
                abs(bad.mapping.x0 - clean.mapping.x0) * clean.mapping.ppm_x > 0.7 or abs(bad.mapping.y0 - clean.mapping.y0) * clean.mapping.ppm_x > 0.7:
            entry["tiles"].append({"set": ds, "skipped": "clean/defective mapping differ"})
            continue
        dd = [d.to_dict() for d in defects]
        kept, qa = qa_defects(clean, bad, dd)
        entry.setdefault("qa", []).append({"set": ds, **{k: v for k, v in qa.items()}})
        if qa["outside_frac"] > 0.2 or not kept:
            entry["tiles"].append({"set": ds, "skipped": f"QA failed (outside_frac {qa['outside_frac']:.2f}, kept {len(kept)})"})
            continue
        tiles = pick_tiles(kept, clean.mapping, args.tile, rng, args.max_tiles_per_set)
        clean_xy = pick_clean_tiles(clean, kept, args.tile, rng, max(1, int(len(tiles) * args.clean_frac)))
        jobs = [(x, y, ds_) for (x, y, ds_) in tiles] + [(x, y, []) for (x, y) in clean_xy]
        for (tx, ty, tdefs) in jobs:
            app = apps[tile_no % len(apps)]
            name = f"{bid}_s{ds}_{tile_no:03d}"
            tile_no += 1
            margin = 24
            full = (clean.mapping.w, clean.mapping.h)
            g_img = compose_tile(clean.tile_layers(tx, ty, args.tile, margin), app, (tx, ty), full, clean.mapping.ppm_x, margin)
            d_img = compose_tile(bad.tile_layers(tx, ty, args.tile, margin), app, (tx, ty), full, bad.mapping.ppm_x, margin) if tdefs else g_img
            lines = []
            for df in tdefs:
                bx = clean.mapping.bbox_px(df["bbox_mm"])
                lines.append(yolo_line(classes.index(df["cls"]), (bx[0] - tx, bx[1] - ty, bx[2] - tx, bx[3] - ty), args.tile))
            for sub in ("images", "labels", "golden"):
                (out / sub / split).mkdir(parents=True, exist_ok=True)
            save_jpg(d_img, out / "images" / split / f"{name}.jpg", app["jpeg_quality"])
            save_jpg(g_img, out / "golden" / split / f"{name}.jpg", app["jpeg_quality"])
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf8")
            entry["tiles"].append({"name": name, "set": ds, "origin_px": [tx, ty], "appearance": apps.index(app),
                                   "defects": tdefs})
        entry.setdefault("defect_sets", []).append({"set": ds, "seed": seed, "defects_requested": counts, "defects_injected": dd,
                                                    "defects_kept": len(kept)})
    return entry


def download(src: dict, dest: Path) -> str:
    if S.licence_ok(src) is False:
        raise ValueError("non-permissive licence")
    if not dest.exists():
        req = urllib.request.Request(S.url(src), headers={"User-Agent": "pcb-synth-dataset"})
        dest.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    return dest.read_text(encoding="utf8", errors="replace")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="/content/kicad_out")
    ap.add_argument("--work", default="/content/kicad_work")
    ap.add_argument("--ppm", type=float, default=40.0, help="pixels per mm (30-48 sensible)")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--boards", type=int, default=0, help="limit number of boards (0 = all)")
    ap.add_argument("--defect-sets", type=int, default=2, help="independent defective variants per board")
    ap.add_argument("--per-class", type=int, default=2, help="defects requested per class per set")
    ap.add_argument("--appearances", type=int, default=3)
    ap.add_argument("--max-tiles-per-set", type=int, default=14)
    ap.add_argument("--clean-frac", type=float, default=0.25, help="clean tiles as a fraction of defect tiles")
    ap.add_argument("--min-sep-mm", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="substring filter on repo/path")
    args = ap.parse_args(argv)
    if not have_cli():
        print("kicad-cli not found; run render_colab.sh first", file=sys.stderr)
        return 2
    I.check_class_names()
    classes = bare_classes()
    out, work = Path(args.out), Path(args.work)
    out.mkdir(parents=True, exist_ok=True)
    (work / "boards").mkdir(parents=True, exist_ok=True)
    srcs = [s for s in S.SOURCES if args.only in s["repo"] + s["path"]]
    if args.boards:
        srcs = srcs[:args.boards]
    manifest = {"classes": list(classes), "ppm": args.ppm, "tile": args.tile, "args": vars(args), "boards": [],
                "note": "licences are per source board; CC-BY needs attribution (repo + licence listed per board)."}
    t0 = time.time()
    for i, src in enumerate(srcs):
        try:
            text = download(src, work / "boards" / f"{slug(src['repo'] + '-' + Path(src['path']).stem)}.kicad_pcb")
            e = process_board(src, text, out, work, args, classes)
        except Exception as ex:                                # noqa: BLE001
            e = {"repo": src["repo"], "path": src["path"], "licence": src["licence"], "status": "error", "reason": repr(ex)}
        manifest["boards"].append(e)
        ntiles = sum(1 for t in e.get("tiles", []) if "name" in t)
        print(f"[{i + 1}/{len(srcs)}] {src['repo']} {e['status']} tiles={ntiles} {e.get('reason', '')} ({time.time() - t0:.0f}s)", flush=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")
        for d in work.glob(f"{slug(src['repo'] + '-' + Path(src['path']).stem)}*_svg"):
            shutil.rmtree(d, ignore_errors=True)
    n = sum(1 for b in manifest["boards"] for t in b.get("tiles", []) if "name" in t)
    print(f"done: {n} tiles from {sum(b['status'] == 'ok' for b in manifest['boards'])}/{len(srcs)} boards -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
