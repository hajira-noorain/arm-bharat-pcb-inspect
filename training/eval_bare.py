"""Board-level evaluation of the bare-board detector + conformal gate calibration.

For every held-out board (cal and test splits from prepare_bare.py) the FULL deployed scoring path
runs: tiling -> detector -> merge. Outputs:
  results/bare_dets_<split>.json         all detections per board (cached for later experiments)
  models/gate/<design>.json              conformal calibration (cal split only)
  results/bare_eval.json                 mAP@0.5 / per-class AP & recall, gate behaviour on TEST split

--backend ultralytics  (laptop GPU, fast iteration)
--backend executorch   (the deployed .pte through executorch.runtime: the numbers we report)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.detect_bare import BareDetector  # noqa: E402
from pcbinspect.gate import ConformalGate, evaluate  # noqa: E402
from pcbinspect.record import BARE_CLASSES  # noqa: E402


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def ap_voc(tp, conf, n_gt):
    """All-point interpolated AP (VOC2010+/COCO-style at a single IoU)."""
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-np.asarray(conf))
    tp = np.asarray(tp, float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def match(dets, gts, thr=0.5):
    """Greedy matching by confidence, same class. Returns per-det TP flags and matched GT set."""
    used = set()
    flags = []
    for d in sorted(range(len(dets)), key=lambda i: -dets[i]["conf"]):
        best, bj = 0.0, None
        for j, g in enumerate(gts):
            if j in used or g["cls"] != dets[d]["cls"]:
                continue
            o = iou(dets[d]["bbox"], g["bbox"])
            if o > best:
                best, bj = o, j
        if bj is not None and best >= thr:
            used.add(bj)
            flags.append((d, 1))
        else:
            flags.append((d, 0))
    flags.sort()
    return [f for _, f in flags], used


def run_split(det: BareDetector, items, cache: Path):
    if cache.exists():
        return json.loads(cache.read_text())
    out = []
    t0 = time.time()
    for k, it in enumerate(items):
        im = cv2.imread(it["path"])
        dets, lat = det.detect(im)
        out.append({**it, "dets": [{"cls": d.cls, "bbox": list(d.bbox), "conf": d.conf} for d in dets],
                    "latency": lat, "shape": list(im.shape[:2])})
        if k % 50 == 0:
            print(f"  {k}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    cache.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=r"C:\Users\sulai\pcb_work")
    ap.add_argument("--backend", default="ultralytics", choices=["ultralytics", "executorch"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    work = Path(args.work)
    tag = args.tag or args.backend
    if args.backend == "ultralytics":
        from pcbinspect.detect_bare import UltralyticsTiles
        tiles = UltralyticsTiles(args.weights or work / "runs" / "y26n" / "weights" / "best.pt")
    else:
        from pcbinspect.detect_bare import ExecuTorchTiles
        tiles = ExecuTorchTiles(args.weights or work / "models" / "vision" / "pcb_y26n_xnnpack.pte")
    det = BareDetector(tiles)
    data = work / "data" / "bare_yolo"
    res_dir = work / "results"
    res_dir.mkdir(exist_ok=True)
    splits = {s: run_split(det, json.loads((data / f"boards_{s}.json").read_text()),
                           res_dir / f"bare_dets_{s}_{tag}.json") for s in ("cal", "test")}

    report = {"backend": args.backend, "tag": tag}
    # ---------------- detection metrics on TEST boards ----------------
    for scope in ("all", "deeppcb", "hripcb", "mendeley"):
        boards = [b for b in splits["test"] if scope == "all" or b["source"] == scope]
        per_cls_tp, per_cls_conf, per_cls_ngt = defaultdict(list), defaultdict(list), defaultdict(int)
        for b in boards:
            for g in b["boxes"]:
                per_cls_ngt[g["cls"]] += 1
            flags, _ = match(b["dets"], b["boxes"])
            for d, f in zip(b["dets"], flags):
                per_cls_tp[d["cls"]].append(f)
                per_cls_conf[d["cls"]].append(d["conf"])
        aps = {c: ap_voc(per_cls_tp[c], per_cls_conf[c], per_cls_ngt[c]) for c in BARE_CLASSES
               if per_cls_ngt[c] > 0}
        report[f"det_{scope}"] = {"mAP50": float(np.mean(list(aps.values()))) if aps else None,
                                  "AP50_per_class": aps, "n_boards": len(boards),
                                  "n_gt": dict(per_cls_ngt)}
        print(scope, f"mAP50={report[f'det_{scope}']['mAP50']}", flush=True)

    # ---------------- conformal gate per design ----------------
    gate_dir = work / "models" / "gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    report["gate"] = {}
    for design in ("deeppcb", "hripcb", "mendeley"):
        def scores(split):
            bs = [b for b in splits[split] if b["source"] == design]
            s = lambda b: max((d["conf"] for d in b["dets"]), default=0.0)  # noqa: E731
            return ([s(b) for b in bs if not b["good"]], [s(b) for b in bs if b["good"]])
        cal_d, cal_g = scores("cal")
        test_d, test_g = scores("test")
        g = ConformalGate(cal_d, cal_g, design)
        g.save(gate_dir / f"{design}.json")
        report["gate"][design] = {f"alpha_{a}": evaluate(g.thresholds(a, 0.01), test_d, test_g)
                                  for a in (0.01, 0.05, 0.10)}
        for a in (0.01, 0.05, 0.10):
            th = g.thresholds(a, 0.01)
            report["gate"][design][f"alpha_{a}"]["lam"] = th.lam
            report["gate"][design][f"alpha_{a}"]["tau"] = th.tau
        r5 = report["gate"][design]["alpha_0.05"]
        print(f"gate {design}: cal {len(cal_d)}d/{len(cal_g)}g  test miss@5%={r5['miss_rate']:.3f} "
              f"CI{tuple(round(x, 3) for x in r5['miss_rate_ci95'])} review={r5['review_rate']:.2f}",
              flush=True)

    lat = [b["latency"]["detector_tiles"] / b["latency"]["n_tiles"] for b in splits["test"]]
    report["latency_per_tile_ms_median_this_machine"] = float(np.median(lat))
    (res_dir / f"bare_eval_{tag}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
