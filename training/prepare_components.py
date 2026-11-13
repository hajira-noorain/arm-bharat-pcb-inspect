"""FPIC (FICS PCB Image Collection, Jessurun et al. 2023) -> training data for the assembled-board parts check.

Two datasets come out, both split BY PHYSICAL BOARD (a board's front and back never straddle a split):

  <out>/detect     YOLO tiles, ONE class "part": every outlined component, named or not. About half of FPIC's
                   outlines carry no designator, so a multi-class detector would have to call those background
                   or guess; a one-class detector uses all of them and answers the question the station asks at
                   inspection time: is there a part here?
  <out>/classify   crops of the outlines that DO carry a designator, in folders by part kind (resistor,
                   capacitor, ic, ...). The small classifier trained on these names the parts once, when a good
                   board is taught; it is never needed while inspecting.

Every photo is resampled to one scale first (FPIC photos range from 4.5 to 44 px/mm; the scale comes from the
colour-checker record named in the file name), then cut into overlapping tiles.

python training/prepare_components.py --fpic /content/ext/fpic --out /content/data/parts
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

csv.field_size_limit(sys.maxsize)

# designator prefix -> part kind (IPC reference designators; the rare ones fold into "other")
KINDS = {
    "R": "resistor", "RA": "resistor", "RN": "resistor",
    "C": "capacitor", "CRA": "capacitor",
    "U": "ic", "IC": "ic",
    "L": "inductor", "FB": "inductor", "T": "inductor",
    "D": "diode", "CR": "diode", "LED": "diode", "LD": "diode",
    "Q": "transistor",
    "J": "connector", "P": "connector", "CN": "connector", "JP": "connector",
    "Y": "crystal",
    "SW": "switch", "S": "switch", "BTN": "switch",
}
KIND_NAMES = sorted(set(KINDS.values())) + ["other"]
NAME_RE = re.compile(r"pcb_(\d+)([fb])_cc_(\d+)", re.I)
VERT_RE = re.compile(r"\[(-?\d+), (-?\d+)\]")


def kind_of(designator: str) -> str | None:
    """Part kind from a reference designator ('C12' -> capacitor); None when the outline has no designator."""
    m = re.match(r"[A-Za-z]+", (designator or "").strip())
    if not m:
        return None
    return KINDS.get(m.group(0).upper(), "other")


def split_boards(board_ids: list[str], seed: int = 0, test: float = 0.2, val: float = 0.1) -> dict[str, str]:
    ids = sorted(set(board_ids), key=lambda s: int(s))
    random.Random(seed).shuffle(ids)
    n_test, n_val = max(1, round(len(ids) * test)), max(1, round(len(ids) * val))
    return {b: ("test" if i < n_test else "val" if i < n_test + n_val else "train") for i, b in enumerate(ids)}


def tiles(h: int, w: int, size: int = 640, overlap: int = 128):
    step = size - overlap
    ys = list(range(0, max(h - size, 0) + 1, step)) or [0]
    xs = list(range(0, max(w - size, 0) + 1, step)) or [0]
    if ys[-1] + size < h:
        ys.append(h - size)
    if xs[-1] + size < w:
        xs.append(w - size)
    return [(x, y) for y in ys for x in xs]


def boxes_in_tile(boxes: np.ndarray, x: int, y: int, size: int, keep: float = 0.5) -> np.ndarray:
    """Boxes (x0,y0,x1,y1) clipped to the tile; a box is kept when at least `keep` of its area is inside."""
    if not len(boxes):
        return boxes.reshape(0, 4)
    c = boxes.copy()
    c[:, [0, 2]] = np.clip(c[:, [0, 2]] - x, 0, size)
    c[:, [1, 3]] = np.clip(c[:, [1, 3]] - y, 0, size)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    inside = (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
    return c[(inside >= keep * np.maximum(area, 1e-6)) & (inside > 9)]


def read_outlines(csv_path: Path) -> list[dict]:
    out = []
    for r in csv.DictReader(open(csv_path, errors="replace")):
        v = VERT_RE.findall(r.get("Vertices") or "")
        if len(v) < 3:
            continue
        xs, ys = [int(a) for a, _ in v], [int(b) for _, b in v]
        out.append({"image": (r.get("Image File") or "").strip(), "box": [min(xs), min(ys), max(xs), max(ys)],
                    "kind": kind_of(r.get("Designator") or "")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpic", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--px-per-mm", type=float, default=16.0, help="every photo is resampled to this scale")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    fpic, out = Path(a.fpic), Path(a.out)
    ppmm = {r["id"]: float(r["ppmm"]) for r in csv.DictReader(open(next(fpic.rglob("color_checker.csv"))))}
    images = {p.name: p for p in (fpic / "pcb_image").rglob("*.png")}
    per_image: dict[str, list[dict]] = {}
    for f in sorted((fpic / "smd_annotation").rglob("*.csv")):
        for o in read_outlines(f):
            per_image.setdefault(o["image"], []).append(o)
    usable = [n for n in sorted(per_image) if n in images and NAME_RE.search(n) and NAME_RE.search(n).group(3) in ppmm]
    print(f"photos with outlines {len(per_image)}, with a file and a scale {len(usable)}")
    split = split_boards([NAME_RE.search(n).group(1) for n in usable], a.seed)
    stats, kinds = Counter(), Counter()
    manifest = {"px_per_mm": a.px_per_mm, "kinds": KIND_NAMES, "boards": split, "photos": {}}
    for s in ("train", "val", "test"):
        (out / "detect" / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "detect" / "labels" / s).mkdir(parents=True, exist_ok=True)
    for n in usable:
        m = NAME_RE.search(n)
        s, scale = split[m.group(1)], a.px_per_mm / ppmm[m.group(3)]
        img = cv2.imread(str(images[n]))
        if img is None:
            print("unreadable", n)
            continue
        if abs(scale - 1) > 0.02:
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        h, w = img.shape[:2]
        boxes = np.array([o["box"] for o in per_image[n]], np.float32) * scale
        manifest["photos"][n] = {"split": s, "board": m.group(1), "scale": scale, "size": [w, h],
                                 "parts": len(boxes)}
        if h < 640 or w < 640:                           # pad small photos so every tile is 640 x 640
            img = cv2.copyMakeBorder(img, 0, max(640 - h, 0), 0, max(640 - w, 0), cv2.BORDER_CONSTANT)
            h, w = img.shape[:2]
        for x, y in tiles(h, w):
            b = boxes_in_tile(boxes, x, y, 640)
            stem = f"{Path(n).stem}_{x}_{y}"
            cv2.imwrite(str(out / "detect" / "images" / s / f"{stem}.jpg"), img[y:y + 640, x:x + 640],
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(out / "detect" / "labels" / s / f"{stem}.txt", "w") as f:
                for x0, y0, x1, y1 in b:
                    f.write(f"0 {(x0 + x1) / 1280:.6f} {(y0 + y1) / 1280:.6f} {(x1 - x0) / 640:.6f} {(y1 - y0) / 640:.6f}\n")
            stats[f"{s}_tiles"] += 1
            stats[f"{s}_boxes"] += len(b)
        # named parts -> classifier crops (a fifth of the part's size as margin, squared, 64 px)
        for i, o in enumerate(per_image[n]):
            if o["kind"] is None:
                continue
            x0, y0, x1, y1 = [v * scale for v in o["box"]]
            side = max(x1 - x0, y1 - y0) * 1.4
            if side < 6:
                continue
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            xa, ya = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
            crop = img[ya:int(cy + side / 2), xa:int(cx + side / 2)]
            if crop.size == 0:
                continue
            d = out / "classify" / s / o["kind"]
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{Path(n).stem}_{i}.jpg"), cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA))
            kinds[f"{s}:{o['kind']}"] += 1
    (out / "detect" / "data.yaml").write_text(
        f"path: {out / 'detect'}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: part\n")
    manifest["stats"], manifest["kind_counts"] = dict(stats), dict(kinds)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("boards per split", dict(Counter(split.values())))
    print("tiles / boxes", dict(stats))
    print("named crops", dict(sorted(kinds.items())))


if __name__ == "__main__":
    main()
