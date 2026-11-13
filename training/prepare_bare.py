"""Build the unified bare-board defect dataset (YOLO format, 640 tiles) + board-level manifests.

Sources (unified to pcbinspect.record.BARE_CLASSES):
  DeepPCB   640x640 binarised pairs (test + defect-free template), official trainval/test split
  HRIPCB    PKU-Market-PCB, large colour photos of 10 boards, VOC xml; split BY BOARD
  Mendeley  PCB-Defect (2025), 230 real etched boards, COCO json; split by image

Splits:
  train / val  -> YOLO tiles (val is only for training early-stopping/model selection)
  cal  / test  -> full board images with ground truth, for conformal calibration and final metrics.
                  These boards are never seen in training.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.record import BARE_CLASSES  # noqa: E402
from pcbinspect.tiling import TILE, clip_box_to_tile, tile_origins  # noqa: E402

CI = {c: i for i, c in enumerate(BARE_CLASSES)}
DEEPPCB_MAP = {1: "open", 2: "short", 3: "mouse_bite", 4: "spur", 5: "spurious_copper", 6: "pin_hole"}
HRIPCB_MAP = {"missing_hole": "missing_hole", "mouse_bite": "mouse_bite", "open_circuit": "open",
              "short": "short", "spur": "spur", "spurious_copper": "spurious_copper"}
MENDELEY_MAP = {"missing_pad": "missing_pad", "mouse_bite": "mouse_bite", "open_circuit": "open",
                "short": "short", "spur": "spur", "spurious_copper": "spurious_copper"}


def board_item(path: Path, boxes: list, source: str, good: bool = False, template: Path | None = None):
    return {"path": str(path), "source": source, "good": good,
            "template": str(template) if template else None,
            "boxes": [{"cls": c, "bbox": [round(v, 1) for v in b]} for c, b in boxes]}


# ---------------------------------------------------------------- loaders
def load_deeppcb(root: Path, rng: random.Random):
    pc = root / "PCBData"
    def read(listfile):
        items = []
        for line in (pc / listfile).read_text().splitlines():
            if not line.strip():
                continue
            img_rel, ann_rel = line.split()
            stem = Path(img_rel).stem                         # e.g. 20085000
            img_dir = pc / Path(img_rel).parent
            test_img = img_dir / f"{stem}_test.jpg"
            temp_img = img_dir / f"{stem}_temp.jpg"
            boxes = []
            for row in (pc / ann_rel).read_text().split("\n"):
                parts = row.replace(",", " ").split()
                if len(parts) != 5:
                    continue
                x1, y1, x2, y2, t = map(int, parts)
                boxes.append((DEEPPCB_MAP[t], (x1, y1, x2, y2)))
            items.append((test_img, temp_img, boxes))
        return items
    trainval = read("trainval.txt")
    test = read("test.txt")
    rng.shuffle(trainval)
    rng.shuffle(test)
    n_val = len(trainval) // 5
    out = defaultdict(list)
    for t, tmp, b in trainval[n_val:]:
        out["train"].append(board_item(t, b, "deeppcb", template=tmp))
    for t, tmp, b in trainval[:n_val]:
        out["val"].append(board_item(t, b, "deeppcb", template=tmp))
    half = len(test) // 2
    for split, chunk in (("cal", test[:half]), ("test", test[half:])):
        for t, tmp, b in chunk:
            out[split].append(board_item(t, b, "deeppcb", template=tmp))
            out[split].append(board_item(tmp, [], "deeppcb", good=True))  # template = defect-free
    # templates of training images are also defect-free training tiles (true negatives)
    for t, tmp, b in trainval[n_val:]:
        out["train"].append(board_item(tmp, [], "deeppcb", good=True))
    return out


def load_hripcb(root: Path, rng: random.Random):
    """Split by physical board (filename prefix 01..12) so test boards are unseen designs."""
    items = []
    for xml in sorted((root / "Annotations").rglob("*.xml")):
        tree = ET.parse(xml).getroot()
        fname = tree.findtext("filename")
        cls_dir = xml.parent.name
        img = root / "images" / cls_dir / fname
        if not img.exists():
            cands = list((root / "images" / cls_dir).glob(Path(fname).stem + ".*"))
            if not cands:
                continue
            img = cands[0]
        boxes = []
        for o in tree.iter("object"):
            name = o.findtext("name").strip().lower()
            if name not in HRIPCB_MAP:
                raise ValueError(f"unknown HRIPCB class {name} in {xml}")
            bb = o.find("bndbox")
            boxes.append((HRIPCB_MAP[name], tuple(float(bb.findtext(k)) for k in
                                                  ("xmin", "ymin", "xmax", "ymax"))))
        board = Path(fname).stem.split("_")[0]
        items.append((board, img, boxes))
    boards = sorted({b for b, _, _ in items})
    rng.shuffle(boards)
    # 2 unseen boards for cal, 2 for test, 1 for val, rest train
    split_of = {}
    for i, b in enumerate(boards):
        split_of[b] = "cal" if i < 2 else "test" if i < 4 else "val" if i < 5 else "train"
    out = defaultdict(list)
    for b, img, boxes in items:
        out[split_of[b]].append(board_item(img, boxes, "hripcb"))
    # PCB_USED/<board>.JPG is the defect-free original of each board: a true "good board"
    for b in boards:
        tmpl = next(iter((root / "PCB_USED").glob(f"{b}.*")), None)
        if tmpl is not None:
            out[split_of[b]].append(board_item(tmpl, [], "hripcb", good=True))
    print(f"  HRIPCB boards: {boards}  ->  {split_of}")
    return out


def load_mendeley(root: Path, rng: random.Random):
    j = json.loads((root / "annotation" / "_annotations.coco.json").read_text())
    cats = {c["id"]: c["name"] for c in j["categories"]}
    by_img = defaultdict(list)
    for a in j["annotations"]:
        name = cats[a["category_id"]]
        mapped = MENDELEY_MAP.get(name)
        if mapped is None:
            continue
        x, y, w, h = a["bbox"]
        by_img[a["image_id"]].append((mapped, (x, y, x + w, y + h)))
    # group by physical board: extra.name like '61-1-4.png' -> board '61'
    imgs = j["images"]
    groups = defaultdict(list)
    for im in imgs:
        g = im.get("extra", {}).get("name", im["file_name"]).split("-")[0]
        groups[g].append(im)
    keys = sorted(groups)
    rng.shuffle(keys)
    n = len(keys)
    split_of = {}
    for i, k in enumerate(keys):
        f = i / n
        split_of[k] = "cal" if f < 0.15 else "test" if f < 0.30 else "val" if f < 0.40 else "train"
    out = defaultdict(list)
    for g, ims in groups.items():
        for im in ims:
            out[split_of[g]].append(board_item(root / "images" / im["file_name"],
                                               by_img.get(im["id"], []), "mendeley"))
    return out


DSP_CLASSES = {0: "short", 1: "spur", 2: "spurious_copper", 3: "open", 4: "mouse_bite",
               5: "hole_breakout", 6: "scratch", 7: "foreign_object", 8: "foreign_object"}  # SH SP SC OP MB HB CS CFO BMFO
DSP_GRID, DSP_CELL = 3, 213   # 3 x 3 crops of 226 px resized to 213 px -> one 640 px "board" (639 + 1 px pad)


def load_dspcbsd(root: Path, rng: random.Random, out_dir: Path):
    """DsPCBSD+ (Lv et al., Scientific Data 2024, CC BY 4.0): 10,259 crops of REAL production-line defects
    from an industrial AOI line-scan system, 226 x 226 px, no solder mask. The crops carry no board id, so
    the dataset is split by crop: its own val split is our TEST (never trained on); its train split is
    divided 80 / 10 / 10 into our train / val / cal. Crops are tiled 3 x 3 into 640 px mosaics so a defect
    keeps (almost) its native pixel size and the detector sees several patterns at once."""
    src = root / "x" / "Data_YOLO"
    if not src.exists():
        return {}
    mos_dir = out_dir / "dspcbsd_mosaics"
    out = defaultdict(list)
    for theirs in ("train", "val"):
        ims = sorted((src / "images" / theirs).glob("*.jpg"))
        rng.shuffle(ims)
        groups = [ims[i:i + DSP_GRID ** 2] for i in range(0, len(ims), DSP_GRID ** 2)]
        for gi, grp in enumerate(groups):
            if theirs == "val":
                split = "test"
            else:
                r = rng.random()
                split = "train" if r < 0.8 else ("val" if r < 0.9 else "cal")
            canvas = np.full((640, 640, 3), 255, np.uint8)
            boxes = []
            for k, p in enumerate(grp):
                cy, cx = divmod(k, DSP_GRID)
                im = cv2.resize(cv2.imread(str(p)), (DSP_CELL, DSP_CELL), interpolation=cv2.INTER_AREA)
                ox, oy = cx * DSP_CELL, cy * DSP_CELL
                canvas[oy:oy + DSP_CELL, ox:ox + DSP_CELL] = im
                lbl = src / "labels" / theirs / f"{p.stem}.txt"
                for line in (lbl.read_text().splitlines() if lbl.exists() else []):
                    f = line.split()
                    if len(f) < 5:
                        continue
                    c, x, y, w, h = int(f[0]), *map(float, f[1:5])
                    boxes.append((DSP_CLASSES[c], [ox + (x - w / 2) * DSP_CELL, oy + (y - h / 2) * DSP_CELL,
                                                   ox + (x + w / 2) * DSP_CELL, oy + (y + h / 2) * DSP_CELL]))
            d = mos_dir / split
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"dsp_{theirs}_{gi:05d}.jpg"
            cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            out[split].append(board_item(path, boxes, "dspcbsd"))
    return out


DD_CODES = {"MP": "missing_pad", "OC": "open", "SC": "short", "SP": "spur", "SPC": "spurious_copper"}


def load_defectdet(root: Path, rng: random.Random):
    """DefectDet (Mendeley t9d9zs3bmb, CC BY 4.0, 2026): 268 photos of REAL manufactured double-layer boards
    whose defects were introduced in the Gerber, so real copper, mask and optics. Split BY PHYSICAL BOARD
    (the leading number of the original file name, e.g. '22_2_BOTTOM_H.png' -> board 22): half of the boards
    train (with a 10% val slice), half are TEST only, so the test boards are designs the model never saw.
    Note: its 'SC' is short circuit (DsPCBSD+'s 'SC' is spurious copper)."""
    js = next(iter(sorted(root.rglob("defect_annotations.json"))), None)
    if js is None:
        return {}
    d = json.loads(js.read_text())
    img_dir = js.parent.parent / "images"
    cats = {c["id"]: DD_CODES.get(c["name"].upper()) for c in d["categories"]}
    by_img = defaultdict(list)
    for a in d["annotations"]:
        cls = cats.get(a["category_id"])
        if cls is None:
            continue
        x, y, w, h = a["bbox"]
        by_img[a["image_id"]].append((cls, [x, y, x + w, y + h]))
    board_of = {}
    for im in d["images"]:
        orig = (im.get("extra") or {}).get("name") or im["file_name"]
        board_of[im["id"]] = orig.split("_")[0]
    boards = sorted(set(board_of.values()))
    rng.shuffle(boards)
    test_boards = set(boards[: len(boards) // 2])
    out = defaultdict(list)
    for im in d["images"]:
        p = img_dir / im["file_name"]
        if not p.exists():
            continue
        if board_of[im["id"]] in test_boards:
            split = "test"
        else:
            split = "val" if rng.random() < 0.1 else "train"
        it = board_item(p, by_img.get(im["id"], []), "defectdet")
        it["board"] = board_of[im["id"]]
        out[split].append(it)
    return out


# ---------------------------------------------------------------- writers
def write_tiles(items, split: str, out_root: Path, rng: random.Random, neg_keep: float) -> Counter:
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for it in items:
        im = cv2.imread(it["path"], cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(it["path"])
        h, w = im.shape[:2]
        stem = f"{it['source']}_{Path(it['path']).stem}"
        for ox, oy in tile_origins(w, h):
            crop = im[oy:oy + TILE, ox:ox + TILE]
            ch, cw = crop.shape[:2]
            lines = []
            for b in it["boxes"]:
                cb = clip_box_to_tile(b["bbox"], ox, oy)
                if cb is None:
                    continue
                x1, y1, x2, y2 = cb
                lines.append(f"{CI[b['cls']]} {(x1 + x2) / 2 / cw:.6f} {(y1 + y2) / 2 / ch:.6f} "
                             f"{(x2 - x1) / cw:.6f} {(y2 - y1) / ch:.6f}")
                counts[b["cls"]] += 1
            if not lines and rng.random() > neg_keep:
                continue
            name = f"{stem}_{ox}_{oy}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines))
            counts["_tiles"] += 1
            counts["_neg_tiles"] += not lines
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\sulai\pcb_work\data")
    ap.add_argument("--out", default=r"C:\Users\sulai\pcb_work\data\bare_yolo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neg-keep", type=float, default=0.25,
                    help="fraction of defect-free tiles to keep (train/val)")
    ap.add_argument("--no-dspcbsd", action="store_true", help="leave out DsPCBSD+ (reproduces the earlier split)")
    ap.add_argument("--defectdet", default="", help="DefectDet root (split by physical board, half train / half test)")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    rng = random.Random(args.seed)
    if out.exists():
        shutil.rmtree(out)

    splits = defaultdict(list)
    loaders = [("deeppcb", lambda: load_deeppcb(data / "DeepPCB", rng)),
               ("hripcb", lambda: load_hripcb(data / "HRIPCB", rng)),
               ("mendeley", lambda: load_mendeley(data / "mendeley" / "PCB_Defect", rng))]
    if args.defectdet:
        loaders.append(("defectdet", lambda: load_defectdet(Path(args.defectdet), random.Random(args.seed + 202))))
    if not args.no_dspcbsd:
        # its own RNG: the earlier sources keep exactly the same split as before
        loaders.append(("dspcbsd", lambda: load_dspcbsd(data / "dspcbsd", random.Random(args.seed + 101), data)))
    for name, fn in loaders:
        print(f"loading {name}")
        for split, items in fn().items():
            splits[split].extend(items)

    stats = {}
    for split in ("train", "val"):
        stats[split] = dict(write_tiles(splits[split], split, out, rng, args.neg_keep))
        print(split, stats[split])
    for split in ("cal", "test"):
        (out / f"boards_{split}.json").write_text(json.dumps(splits[split], indent=1))
        c = Counter(it["source"] + (":good" if it["good"] else ":defective") for it in splits[split])
        stats[split] = dict(c)
        print(split, stats[split])

    yaml_text = (f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\n"
                 f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(BARE_CLASSES)))
    (out / "data.yaml").write_text(yaml_text)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
