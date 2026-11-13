"""The parts check, end to end, on FPIC boards the models never trained on.

There is no public set of photographed boards with parts pulled off, so the removal is SIMULATED and reported as
such: each held-out photo is the taught good board; a copy has a few labelled parts painted out (filled in from
the surrounding board), then is re-photographed in software (put down slightly off, exposure and noise changed).
The station's own code does the rest: teach the parts map, line the copy up, find parts, compare.

  removed parts reported missing          by part size; all removed parts, and those the taught map contained
  false "missing" and "unexpected" calls  per board, on the copy with parts removed and on a copy with none removed
  part names                              of mapped parts whose kind FPIC labels: named right / named wrong / left as "part"

python training/eval_parts.py --fpic /content/ext/fpic --manifest /content/data/parts/manifest.json \
    --detector best.pt --classifier cls_best.pt --out results/parts_check_eval.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.golden import MIN_ALIGN_CC, align_warp  # noqa: E402
from pcbinspect.parts import MIN_KIND_CONF, check_board, iou, teach_parts, find_parts  # noqa: E402
from training.prepare_components import NAME_RE, read_outlines  # noqa: E402

csv.field_size_limit(sys.maxsize)
SIZE_BINS = ((0.0, 1.0, "under 1 mm"), (1.0, 2.0, "1 to 2 mm"), (2.0, 5.0, "2 to 5 mm"), (5.0, 1e9, "over 5 mm"))


def second_photo(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """pcbinspect.teach.recapture, returning the matrix too (good picture -> second picture)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-0.4, 0.4), 1.0)
    M[:, 2] += rng.uniform(-0.004, 0.004, 2) * w
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    out = out * rng.uniform(0.96, 1.04) + rng.normal(0, 2.0, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8), M


def move_box(box, M) -> list[float]:
    x0, y0, x1, y1 = box
    p = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], float) @ np.asarray(M, float).T
    return [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]


def paint_out(img: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    """Remove parts: each box (grown by a fifth) is filled in from the board around it."""
    mask = np.zeros(img.shape[:2], np.uint8)
    for x0, y0, x1, y1 in boxes:
        gx, gy = (x1 - x0) * 0.1 + 2, (y1 - y0) * 0.1 + 2
        cv2.rectangle(mask, (int(x0 - gx), int(y0 - gy)), (int(x1 + gx), int(y1 + gy)), 255, -1)
    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


def overlaps(a, b) -> bool:
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return iou(a, b) >= 0.2 or (b[0] <= cx <= b[2] and b[1] <= cy <= b[3])


class UltraKinds:
    def __init__(self, weights):
        from ultralytics import YOLO
        self.m = YOLO(str(weights))

    def __call__(self, crops):
        out = []
        for i in range(0, len(crops), 256):
            for r in self.m.predict(crops[i:i + 256], imgsz=64, verbose=False):
                out.append((r.names[int(r.probs.top1)], float(r.probs.top1conf)))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpic", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--classifier", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--remove", type=int, default=6, help="parts painted out per photo")
    ap.add_argument("--min-mm", type=float, default=0.6, help="smallest part removed (longer side)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from pcbinspect.detect_bare import BareDetector, UltralyticsTiles
    det = BareDetector(UltralyticsTiles(a.detector, conf_thr=0.1), names=("part",))
    kinds = UltraKinds(a.classifier) if a.classifier else None
    man = json.loads(Path(a.manifest).read_text())
    ppmm = float(man["px_per_mm"])
    fpic = Path(a.fpic)
    images = {p.name: p for p in (fpic / "pcb_image").rglob("*.png")}
    outlines: dict[str, list[dict]] = {}
    for f in sorted((fpic / "smd_annotation").rglob("*.csv")):
        for o in read_outlines(f):
            outlines.setdefault(o["image"], []).append(o)
    rng = np.random.default_rng(a.seed)
    trials, boards, names = [], [], Counter()
    for n, info in sorted(man["photos"].items()):
        if info["split"] != "test" or n not in images or not NAME_RE.search(n):
            continue
        img = cv2.imread(str(images[n]))
        s = info["scale"]
        if abs(s - 1) > 0.02:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        gt = [{"box": [v * s for v in o["box"]], "kind": o["kind"]} for o in outlines.get(n, [])]
        pmap = teach_parts(det, kinds, img, 1.0 / ppmm)
        mapped = lambda box: next((p for p in pmap.parts if overlaps(p.box, box)), None)   # noqa: E731
        for g in gt:                                              # part names against FPIC's own labels
            p = mapped(g["box"])
            if p is not None and g["kind"] not in (None, "other"):
                names["right" if p.label == g["kind"] else "left as part" if p.label == "part" else "wrong"] += 1
        big = [g for g in gt if max(g["box"][2] - g["box"][0], g["box"][3] - g["box"][1]) / ppmm >= a.min_mm]
        pick = [big[i] for i in rng.choice(len(big), size=min(a.remove, len(big)), replace=False)] if big else []
        row = {"photo": n, "labelled_parts": len(gt), "mapped_parts": len(pmap.parts),
               "labelled_parts_on_map": sum(mapped(g["box"]) is not None for g in gt)}
        for tag, removed in (("removed", pick), ("clean", [])):
            shot, M = second_photo(paint_out(img, [g["box"] for g in removed]) if removed else img, rng)
            warp, cc = align_warp(shot, img)
            if warp is None or cc < MIN_ALIGN_CC:
                row[f"{tag}_aligned"] = False
                continue
            row[f"{tag}_aligned"] = True
            dets, _ = find_parts(det, shot, 1.0 / ppmm)
            fs = check_board(pmap, dets, warp, (shot.shape[1], shot.shape[0]))
            gone = [move_box(g["box"], M) for g in removed]
            miss = [f for f in fs if f.what == "missing_part"]
            row[f"{tag}_false_missing"] = sum(not any(overlaps(f.box, b) for b in gone) for f in miss)
            row[f"{tag}_unexpected"] = sum(f.what == "extra_part" for f in fs)
            for g, b in zip(removed, gone):
                mm = max(g["box"][2] - g["box"][0], g["box"][3] - g["box"][1]) / ppmm
                trials.append({"photo": n, "size_mm": round(mm, 2), "kind": g["kind"],
                               "on_map": mapped(g["box"]) is not None,
                               "reported": any(overlaps(f.box, b) for f in miss)})
        boards.append(row)
        print(n, {k: v for k, v in row.items() if k != "photo"}, flush=True)

    def rate(ts):
        return {"n": len(ts), "reported": sum(t["reported"] for t in ts) / max(len(ts), 1)}
    ok = [b for b in boards if b.get("removed_aligned")]
    clean = [b for b in boards if b.get("clean_aligned")]
    summary = {
        "note": "part removal is simulated (painted out and re-photographed in software) on boards never trained on",
        "photos": len(boards), "photos_lined_up": len(ok),
        "labelled_parts_found_when_teaching": sum(b["labelled_parts_on_map"] for b in boards) / max(
            sum(b["labelled_parts"] for b in boards), 1),
        "removed_parts": rate(trials), "removed_parts_that_were_on_the_map": rate([t for t in trials if t["on_map"]]),
        "by_size": {lab: rate([t for t in trials if lo <= t["size_mm"] < hi]) for lo, hi, lab in SIZE_BINS},
        "false_missing_per_board_with_removals": sum(b["removed_false_missing"] for b in ok) / max(len(ok), 1),
        "unexpected_per_board_with_removals": sum(b["removed_unexpected"] for b in ok) / max(len(ok), 1),
        "false_missing_per_board_clean": sum(b["clean_false_missing"] for b in clean) / max(len(clean), 1),
        "unexpected_per_board_clean": sum(b["clean_unexpected"] for b in clean) / max(len(clean), 1),
        "clean_boards_with_no_call_at_all": sum(b["clean_false_missing"] + b["clean_unexpected"] == 0 for b in clean) / max(len(clean), 1),
        "mapped_parts_per_board_mean": sum(b["mapped_parts"] for b in boards) / max(len(boards), 1),
        "part_names": {**{k: v / max(sum(names.values()), 1) for k, v in names.items()}, "n": sum(names.values()),
                       "shown_only_at_confidence": MIN_KIND_CONF},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "boards": boards, "trials": trials}, indent=1))
    print("RESULT parts_check", json.dumps(summary))


if __name__ == "__main__":
    main()
