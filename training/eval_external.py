"""Cross-dataset test: detectors on a dataset NONE of them was trained on (e.g. DefectDet, real manufactured
boards with design-introduced defects, CC BY 4.0). Reports mAP@0.5 and per-defect recall at the operating
threshold, per detector, on the classes the dataset shares with ours.

python training/eval_external.py --coco-root /content/ext/defectdet --weights a.pt b.pt --names v1 v2 \
    --gate-lam 0.67 0.67 --out results/external_defectdet.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.detect_bare import BareDetector, UltralyticsTiles  # noqa: E402
from training.eval_bare import ap_voc, iou, match  # noqa: E402

NAME_MAP = {  # dataset category name (lower-case, spaces/underscores/hyphens removed) -> our class
    "short": "short", "shortcircuit": "short", "open": "open", "opencircuit": "open",
    "spuriouscopper": "spurious_copper", "spur": "spur", "missingpad": "missing_pad",
    "mousebite": "mouse_bite", "pinhole": "pin_hole", "missinghole": "missing_hole",
}
# per-dataset abbreviation codes (the same code can mean different things in different datasets:
# DsPCBSD+ "SC" = spurious copper, DefectDet "SC" = short circuit)
DATASET_CODES = {
    "defectdet": {"mp": "missing_pad", "oc": "open", "sc": "short", "sp": "spur", "spc": "spurious_copper"},
}


def norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _find_image(js: Path, name: str) -> Path | None:
    for cand in (js.parent / name, js.parent.parent / "images" / name, js.parent / "images" / name):
        if cand.exists():
            return cand
    return None


def load_coco_boards(root: Path, codes: dict | None = None) -> tuple[list[dict], dict]:
    """All COCO json files under root; every image becomes a test board. `codes` maps dataset-specific
    abbreviations (checked first), then NAME_MAP maps spelled-out names."""
    codes = codes or {}
    boards, unmapped = [], defaultdict(int)
    for js in sorted(root.rglob("*.json")):
        d = json.loads(js.read_text())
        if "images" not in d or "annotations" not in d:
            continue
        cats = {c["id"]: codes.get(norm(c["name"])) or NAME_MAP.get(norm(c["name"])) for c in d["categories"]}
        names = {c["id"]: c["name"] for c in d["categories"]}
        by = defaultdict(list)
        for a in d["annotations"]:
            cls = cats.get(a["category_id"])
            if cls is None:
                unmapped[names.get(a["category_id"], str(a["category_id"]))] += 1
                continue
            x, y, w, h = a["bbox"]
            by[a["image_id"]].append({"cls": cls, "bbox": [x, y, x + w, y + h]})
        for im in d["images"]:
            p = _find_image(js, im["file_name"])
            if p is not None:
                boards.append({"path": str(p), "boxes": by.get(im["id"], []), "source": root.name, "good": False})
    return boards, dict(unmapped)


def evaluate(det, boards, lam: float, scale: float = 1.0):
    tp, conf, ngt = defaultdict(list), defaultdict(list), defaultdict(int)
    found = total = 0
    per_cls = defaultdict(lambda: [0, 0])
    fa = 0
    for b in boards:
        im = cv2.imread(b["path"])
        if scale != 1.0:   # resize the capture; boxes are mapped back to original pixels
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        dets, _ = det.detect(im)
        dd = [{"cls": d.cls, "bbox": [v / scale for v in d.bbox], "conf": d.conf} for d in dets]
        for g in b["boxes"]:
            ngt[g["cls"]] += 1
            hit = any(x["conf"] >= lam and iou(x["bbox"], g["bbox"]) >= 0.3 for x in dd)
            found += hit
            total += 1
            per_cls[g["cls"]][0] += hit
            per_cls[g["cls"]][1] += 1
        flags, _ = match(dd, b["boxes"])
        for x, f in zip(dd, flags):
            tp[x["cls"]].append(f)
            conf[x["cls"]].append(x["conf"])
        fa += sum(1 for x in dd if x["conf"] >= lam and not any(iou(x["bbox"], g["bbox"]) >= 0.3 for g in b["boxes"]))
    aps = {c: ap_voc(tp[c], conf[c], ngt[c]) for c in ngt}
    return {"mAP50_shared_classes": float(np.mean(list(aps.values()))) if aps else None,
            "AP50_per_class": aps, "defect_recall_at_lambda": found / max(total, 1),
            "recall_per_class": {c: v[0] / v[1] for c, v in per_cls.items()},
            "false_calls_per_board": fa / max(len(boards), 1), "n_boards": len(boards), "n_defects": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", required=True)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--gate-lam", nargs="+", type=float, required=True,
                    help="each detector's per-defect gate threshold at the 5%% setting (from its own calibration)")
    ap.add_argument("--codes", default="", help="dataset-specific category codes, e.g. defectdet")
    ap.add_argument("--only-boards", default="", help="boards_test.json: keep only these boards")
    ap.add_argument("--only-source", default="defectdet")
    ap.add_argument("--scales", default="1", help="comma list of resize factors to test, e.g. 1,1.5,2,3")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    boards, unmapped = load_coco_boards(Path(a.coco_root), DATASET_CODES.get(a.codes) if a.codes else None)
    if a.only_boards:   # e.g. the held-out half: never score boards that were used for training
        keep = {Path(b["path"]).name for b in json.loads(Path(a.only_boards).read_text())
                if b["source"] == a.only_source}
        boards = [b for b in boards if Path(b["path"]).name in keep]
    print(f"{len(boards)} boards, unmapped categories: {unmapped}", flush=True)
    rep = {"dataset": Path(a.coco_root).name, "n_boards": len(boards), "unmapped": unmapped, "models": {}}
    for w, n, lam in zip(a.weights, a.names, a.gate_lam):
        det = BareDetector(UltralyticsTiles(w, device=a.device))
        for sc in map(float, a.scales.split(",")):
            r = evaluate(det, boards, lam, sc)
            key = n if sc == 1.0 else f"{n}@x{sc:g}"
            rep["models"][key] = {"lambda": lam, "scale": sc, **r}
            print(key, f"mAP {r['mAP50_shared_classes']:.3f} recall {r['defect_recall_at_lambda']:.3f} "
                       f"FA/board {r['false_calls_per_board']:.2f}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
