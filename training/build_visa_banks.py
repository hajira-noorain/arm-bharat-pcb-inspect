"""Build + evaluate the assembled-board anomaly model on VisA pcb1-4.

Outputs (per pcbX):
  models/anomaly/<pcbX>/bank.npy + meta.json        deployable memory bank (full-shot enrolment)
  results/anomaly_<pcbX>.json                        image/pixel AUROC, few-shot curve, timings
  results/gate_scores_<pcbX>.json                    board scores for conformal calibration/test

The backbone used here is the same nn.Module that is exported to ExecuTorch (anomaly.build_torch_backbone),
run on the laptop GPU only to make evaluation fast. On-device numbers are measured separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.anomaly import enrol, torch_feature_fn  # noqa: E402
from pcbinspect.gate import ConformalGate, evaluate  # noqa: E402


def read_split(visa: Path, obj: str):
    rows = defaultdict(list)
    with open(visa / "split_csv" / "1cls.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["object"] == obj:
                rows[r["split"]].append(r)
    return rows["train"], rows["test"]


def load(visa: Path, rel: str):
    im = cv2.imread(str(visa / rel), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(visa / rel)
    return im


def evaluate(model, test_rows, visa: Path, with_pixels: bool):
    img_scores, labels, pix_s, pix_y = [], [], [], []
    t_backbone, t_knn = [], []
    for r in test_rows:
        im = load(visa, r["image"])
        res = model.score(im)
        img_scores.append(res.raw_score)
        labels.append(int(r["label"] == "anomaly"))
        t_backbone.append(res.latency_ms["anomaly_backbone"])
        t_knn.append(res.latency_ms["anomaly_knn"])
        if with_pixels:
            gh, gw = res.heatmap.shape
            if r["label"] == "anomaly" and r.get("mask"):
                m = cv2.imread(str(visa / r["mask"]), cv2.IMREAD_GRAYSCALE)
                m = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA) > 0
            else:
                m = np.zeros((gh, gw), bool)
            pix_s.append(res.heatmap.ravel())
            pix_y.append(m.ravel())
    out = {"image_auroc": float(roc_auc_score(labels, img_scores)),
           "n_test": len(labels), "n_anomalous": int(sum(labels)),
           "gpu_backbone_ms_median": float(np.median(t_backbone)),
           "knn_ms_median": float(np.median(t_knn))}
    if with_pixels:
        out["pixel_auroc_featuregrid"] = float(roc_auc_score(np.concatenate(pix_y),
                                                             np.concatenate(pix_s)))
    return out, np.array(img_scores), np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visa", default=r"C:\Users\sulai\pcb_work\data\VisA")
    ap.add_argument("--out", default=r"C:\Users\sulai\pcb_work")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--objects", default="pcb1,pcb2,pcb3,pcb4")
    ap.add_argument("--fewshot", default="5,10,20")
    ap.add_argument("--fewshot-reps", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=200,
                    help="good images used for the full-shot bank (coreset keeps it small)")
    args = ap.parse_args()
    visa, out = Path(args.visa), Path(args.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    fn = torch_feature_fn(args.device)

    for obj in args.objects.split(","):
        train, test = read_split(visa, obj)
        rng = random.Random(0)
        rng.shuffle(train)
        print(f"== {obj}: train {len(train)} good, test {len(test)}", flush=True)
        report = {"object": obj}

        # full-shot (capped) bank: what ships for a known design
        t0 = time.time()
        goods = [load(visa, r["image"]) for r in train[:args.max_train]]
        model = enrol(fn, goods, design=obj, coreset_ratio=0.02, max_bank=16384)
        report["enrol_seconds_gpu"] = time.time() - t0
        report["bank_size"] = int(len(model.bank))
        ev, scores, labels = evaluate(model, test, visa, with_pixels=True)
        report["full"] = {**ev, "n_train": len(goods)}
        print(f"   full ({len(goods)} good): image AUROC {ev['image_auroc']:.4f}  "
              f"pixel AUROC {ev['pixel_auroc_featuregrid']:.4f}", flush=True)
        model.save(out / "models" / "anomaly" / obj)

        # board scores for conformal calibration/test: split the VisA test set in half
        idx = list(range(len(test)))
        random.Random(1).shuffle(idx)
        half = len(idx) // 2
        norm = model.normalise(scores)
        gate = {"cal": {"defective": [], "good": []}, "test": {"defective": [], "good": []}}
        for j, i in enumerate(idx):
            split = "cal" if j < half else "test"
            gate[split]["defective" if labels[i] else "good"].append(float(norm[i]))
        (out / "results" / f"gate_scores_{obj}.json").write_text(json.dumps(gate))
        g = ConformalGate(gate["cal"]["defective"], gate["cal"]["good"], f"visa_{obj}")
        (out / "models" / "gate").mkdir(parents=True, exist_ok=True)
        g.save(out / "models" / "gate" / f"visa_{obj}.json")
        report["gate"] = {}
        for a in (0.01, 0.05, 0.10):
            th = g.thresholds(a, 0.01)
            report["gate"][f"alpha_{a}"] = {**evaluate(th, gate["test"]["defective"], gate["test"]["good"]),
                                            "lam": th.lam, "tau": th.tau}
        r5 = report["gate"]["alpha_0.05"]
        print(f"   gate @5%: test miss {r5['miss_rate']:.3f}, review {r5['review_rate']:.2f}, "
              f"good failed {r5['false_reject_rate']:.3f}", flush=True)

        # few-shot curve: "teach a new board" from k good images
        report["fewshot"] = {}
        for k in map(int, args.fewshot.split(",")):
            aurocs = []
            for rep in range(args.fewshot_reps):
                sample = random.Random(100 + rep).sample(train, k)
                m = enrol(fn, [load(visa, r["image"]) for r in sample], design=obj,
                          coreset_ratio=0.25, max_bank=16384, seed=rep)
                aurocs.append(evaluate(m, test, visa, with_pixels=False)[0]["image_auroc"])
            report["fewshot"][k] = {"image_auroc_mean": float(np.mean(aurocs)),
                                    "image_auroc_min": float(np.min(aurocs)), "reps": aurocs}
            print(f"   {k}-shot: image AUROC {np.mean(aurocs):.4f} (min {np.min(aurocs):.4f})",
                  flush=True)
        (out / "results" / f"anomaly_{obj}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
