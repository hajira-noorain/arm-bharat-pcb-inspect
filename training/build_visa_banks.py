"""Build + evaluate the assembled-board anomaly model on VisA pcb1-4 (official 1cls split).

Features of every image are computed ONCE (per backbone / input size) and reused for the full
memory, all few-shot draws and all scoring, so comparing backbones is cheap.

Outputs (per pcbX, with --tag):
  models/anomaly/<pcbX>/                       deployable memory (only for --save)
  models/gate/visa_<pcbX>.json                 conformal calibration (only for --save)
  results/anomaly_<pcbX>[_tag].json            image/pixel AUROC, few-shot curve, gate on held-out half

python training/build_visa_banks.py --arch resnet18 --input 256 --save
python training/build_visa_banks.py --arch wide_resnet50_2 --input 256 --tag wrn50
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
from pcbinspect.anomaly import (descriptors, enrol_from_desc, grey_thumb, preprocess,  # noqa: E402
                                torch_feature_fn)
from pcbinspect.gate import ConformalGate  # noqa: E402
from pcbinspect.gate import evaluate as gate_evaluate  # noqa: E402


def read_split(visa: Path, obj: str):
    rows = defaultdict(list)
    with open(visa / "split_csv" / "1cls.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["object"] == obj:
                rows[r["split"]].append(r)
    return rows["train"], rows["test"]


class FeatureCache:
    def __init__(self, fn, visa: Path, size: int):
        self.fn, self.visa, self.size, self.c, self.thumbs = fn, visa, size, {}, {}

    def get(self, rel: str):
        if rel not in self.c:
            im = cv2.imread(str(self.visa / rel), cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(self.visa / rel)
            f = self.fn(preprocess(im, self.size))
            self.c[rel] = (descriptors(f), f.shape[2], f.shape[3], im.shape[1], im.shape[0])
            self.thumbs[rel] = grey_thumb(im)    # for the optional foreground mask
        return self.c[rel]

    def enrol_opts(self, rels, fg_mask=False, posnorm=False, topk_frac=0.0) -> dict:
        """Keyword options for enrol_from_desc for the images `rels` (all off = the original model)."""
        for r in rels:
            self.get(r)
        gh, gw = self.c[rels[0]][1:3]
        return dict(fg_mask=fg_mask, posnorm=posnorm, topk_frac=topk_frac, grid=(gh, gw),
                    thumbs=[self.thumbs[r] for r in rels] if fg_mask else None)


def evaluate(model, test_rows, cache: FeatureCache, visa: Path, with_pixels: bool):
    img_scores, labels, pix_s, pix_y = [], [], [], []
    for r in test_rows:
        desc, gh, gw, w, h = cache.get(r["image"])
        res = model.score_desc(desc, gh, gw, w, h)
        img_scores.append(res.raw_score)
        labels.append(int(r["label"] == "anomaly"))
        if with_pixels:
            if r["label"] == "anomaly" and r.get("mask"):
                m = cv2.imread(str(visa / r["mask"]), cv2.IMREAD_GRAYSCALE)
                m = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA) > 0
            else:
                m = np.zeros((gh, gw), bool)
            pix_s.append(res.heatmap.ravel())
            pix_y.append(m.ravel())
    out = {"image_auroc": float(roc_auc_score(labels, img_scores)), "n_test": len(labels),
           "n_anomalous": int(sum(labels))}
    if with_pixels:
        out["pixel_auroc_featuregrid"] = float(roc_auc_score(np.concatenate(pix_y), np.concatenate(pix_s)))
    return out, np.array(img_scores), np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visa", default=r"C:\Users\sulai\pcb_work\data\VisA")
    ap.add_argument("--out", default=r"C:\Users\sulai\pcb_work")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--input", type=int, default=256)
    ap.add_argument("--objects", default="pcb1,pcb2,pcb3,pcb4")
    ap.add_argument("--fewshot", default="5,10,20")
    ap.add_argument("--fewshot-reps", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=200)
    ap.add_argument("--coreset", type=float, default=0.02)
    ap.add_argument("--tag", default="")
    ap.add_argument("--fg-mask", action="store_true", help="ignore patches outside the board foreground")
    ap.add_argument("--posnorm", action="store_true", help="per-grid-position normalisation of patch distances")
    ap.add_argument("--topk-frac", type=float, default=0.0, help="image score = mean of top this fraction of patches (0 = max)")
    ap.add_argument("--save", action="store_true", help="write the deployable memory and gate")
    args = ap.parse_args()
    visa, out = Path(args.visa), Path(args.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    fn = torch_feature_fn(args.device, args.arch)
    suffix = f"_{args.tag}" if args.tag else ""

    for obj in args.objects.split(","):
        t_obj = time.time()
        train, test = read_split(visa, obj)
        random.Random(0).shuffle(train)
        cache = FeatureCache(fn, visa, args.input)
        print(f"== {obj}: {args.arch}@{args.input}, train {len(train)} good, test {len(test)}", flush=True)
        report = {"object": obj, "arch": args.arch, "input": args.input, "tag": args.tag,
                  "fg_mask": args.fg_mask, "posnorm": args.posnorm, "topk_frac": args.topk_frac}

        goods = train[:args.max_train]
        t0 = time.time()
        feats = [cache.get(r["image"])[0] for r in goods]
        rels = [r["image"] for r in goods]
        model = enrol_from_desc(fn, feats, obj, args.coreset, 16384, input_size=args.input, arch=args.arch,
                                **cache.enrol_opts(rels, args.fg_mask, args.posnorm, args.topk_frac))
        report["fg_status"] = model.fg_status
        report["enrol_seconds"] = time.time() - t0
        report["bank_size"] = int(len(model.bank))
        ev, scores, labels = evaluate(model, test, cache, visa, with_pixels=True)
        report["full"] = {**ev, "n_train": len(goods)}
        print(f"   full ({len(goods)} good): image AUROC {ev['image_auroc']:.4f}  "
              f"pixel AUROC {ev['pixel_auroc_featuregrid']:.4f}", flush=True)

        # gate: calibrate on one half of the official test set, evaluate on the other half
        idx = list(range(len(test)))
        random.Random(1).shuffle(idx)
        half = len(idx) // 2
        norm = model.normalise(scores)
        gate = {"cal": {"defective": [], "good": []}, "test": {"defective": [], "good": []}}
        for j, i in enumerate(idx):
            gate["cal" if j < half else "test"]["defective" if labels[i] else "good"].append(float(norm[i]))
        g = ConformalGate(gate["cal"]["defective"], gate["cal"]["good"], f"visa_{obj}")
        report["gate"] = {}
        for a in (0.05, 0.10):
            th = g.thresholds(a, 0.01)
            report["gate"][f"alpha_{a}"] = {**gate_evaluate(th, gate["test"]["defective"], gate["test"]["good"]),
                                            "lam": th.lam, "tau": th.tau,
                                            "supported_alpha": len(gate["cal"]["defective"]) >= int(np.ceil(1 / a)) - 1}
        r5 = report["gate"]["alpha_0.05"]
        print(f"   gate @5%: test miss {r5['miss_rate']:.3f}, review {r5['review_rate']:.2f}, "
              f"good failed {r5['false_reject_rate']:.3f}", flush=True)

        report["fewshot"] = {}
        for k in map(int, args.fewshot.split(",")):
            aurocs = []
            for rep in range(args.fewshot_reps):
                sample = random.Random(100 + rep).sample(train, k)
                srel = [r["image"] for r in sample]
                m = enrol_from_desc(fn, [cache.get(x)[0] for x in srel], obj, 0.25, 16384,
                                    seed=rep, input_size=args.input, arch=args.arch,
                                    **cache.enrol_opts(srel, args.fg_mask, args.posnorm, args.topk_frac))
                aurocs.append(evaluate(m, test, cache, visa, with_pixels=False)[0]["image_auroc"])
            report["fewshot"][k] = {"image_auroc_mean": float(np.mean(aurocs)),
                                    "image_auroc_min": float(np.min(aurocs)), "reps": aurocs}
            print(f"   {k}-shot: image AUROC {np.mean(aurocs):.4f} (min {np.min(aurocs):.4f})", flush=True)
        report["seconds_total"] = time.time() - t_obj
        (out / "results" / f"anomaly_{obj}{suffix}.json").write_text(json.dumps(report, indent=1))
        if args.save:   # with --tag, each configuration keeps its own memory and gate
            sub = Path(args.tag) if args.tag else Path()
            model.save(out / "models" / "anomaly" / sub / obj)
            (out / "models" / "gate" / sub).mkdir(parents=True, exist_ok=True)
            g.save(out / "models" / "gate" / sub / f"visa_{obj}.json")


if __name__ == "__main__":
    main()
