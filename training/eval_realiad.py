"""Test-only: the assembled-board anomaly path on Real-IAD 'pcb' (CVPR 2024, CC BY-NC-SA 4.0, research only).

Real-IAD is NEVER copied into the repo and NEVER used for weights we publish; this script only measures.
Layout: pcb/OK/S####/pcb_####_OK_C{1..5}_*.jpg and pcb/NG/<type>/S####/..._C{v}_*.jpg (+ .png mask for the
views where the defect is visible). Five camera views per sample; each view gets its own memory, as a station
would learn one camera position.

Protocol (per view): good samples are split 80/20 BY SAMPLE (seeded); the memory learns from the 80%; the test
is the held-out good views plus every defective view that HAS a mask. Defective views without a mask (defect
not visible from that camera) are left out and counted, not called good. Few-shot curve as for VisA.

python training/eval_realiad.py --root /content/ext/realiad/pcb --arch resnet18 --input 256 --out results/realiad_pcb_r18_256.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.anomaly import enrol_from_desc, torch_feature_fn  # noqa: E402
from training.build_visa_banks import FeatureCache, evaluate  # noqa: E402

PAT = re.compile(r"_C(\d)_[^/]*\.jpg$")


def rows_by_view(root: Path):
    good = defaultdict(lambda: defaultdict(list))   # view -> sample -> rows
    bad = defaultdict(list)
    skipped = defaultdict(int)
    for p in sorted(root.rglob("*.jpg")):
        m = PAT.search(p.name)
        if not m:
            continue
        v = int(m.group(1))
        rel = str(p.relative_to(root))
        sample = p.parent.name
        if "/OK/" in p.as_posix() or "\\OK\\" in str(p):
            good[v][sample].append({"image": rel, "label": "normal", "mask": ""})
        else:
            mask = p.with_suffix(".png")
            dtype = p.parent.parent.name
            if mask.exists():
                bad[v].append({"image": rel, "label": "anomaly", "mask": str(mask.relative_to(root)), "type": dtype})
            else:
                skipped[f"C{v}"] += 1
    return good, bad, dict(skipped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--input", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--coreset", type=float, default=0.02)
    ap.add_argument("--max-train", type=int, default=200)
    ap.add_argument("--fewshot", default="5,10,20")
    ap.add_argument("--fg-mask", action="store_true", help="ignore patches outside the board foreground")
    ap.add_argument("--posnorm", action="store_true", help="per-grid-position normalisation of patch distances")
    ap.add_argument("--topk-frac", type=float, default=0.0, help="image score = mean of top this fraction of patches (0 = max)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    fn = torch_feature_fn(a.device, a.arch)
    good, bad, skipped = rows_by_view(root)
    rep = {"dataset": "Real-IAD pcb (test only, CC BY-NC-SA)", "arch": a.arch, "input": a.input,
           "fg_mask": a.fg_mask, "posnorm": a.posnorm, "topk_frac": a.topk_frac,
           "defective_views_without_mask_left_out": skipped, "views": {}}
    print("left out (no mask):", skipped, flush=True)
    for v in sorted(good):
        samples = sorted(good[v])
        random.Random(v).shuffle(samples)
        k = int(0.8 * len(samples))
        train = [r for s in samples[:k] for r in good[v][s]]
        test = [r for s in samples[k:] for r in good[v][s]] + bad[v]
        random.Random(0).shuffle(train)
        cache = FeatureCache(fn, root, a.input)
        t0 = time.time()
        feats = [cache.get(r["image"])[0] for r in train[:a.max_train]]
        opt = dict(fg_mask=a.fg_mask, posnorm=a.posnorm, topk_frac=a.topk_frac)
        rels = [r["image"] for r in train[:a.max_train]]
        model = enrol_from_desc(fn, feats, f"realiad_pcb_C{v}", a.coreset, 16384, input_size=a.input, arch=a.arch,
                                **cache.enrol_opts(rels, **opt))
        ev, scores, labels = evaluate(model, test, cache, root, with_pixels=True)
        per_type = {}
        normal = [s for s, y in zip(scores, labels) if not y]
        from sklearn.metrics import roc_auc_score
        for t in sorted({r["type"] for r in bad[v]}):
            ts = [s for s, r in zip(scores, test) if r.get("type") == t]
            if ts and normal:
                per_type[t] = float(roc_auc_score([0] * len(normal) + [1] * len(ts), normal + ts))
        few = {}
        for kk in map(int, a.fewshot.split(",")):
            aur = []
            for rep_i in range(3):
                smp = random.Random(100 + rep_i).sample(train, kk)
                srel = [r["image"] for r in smp]
                m = enrol_from_desc(fn, [cache.get(x)[0] for x in srel], "fs", 0.25, 16384, seed=rep_i,
                                    input_size=a.input, arch=a.arch, **cache.enrol_opts(srel, **opt))
                aur.append(evaluate(m, test, cache, root, with_pixels=False)[0]["image_auroc"])
            few[kk] = {"mean": float(np.mean(aur)), "min": float(np.min(aur))}
        rep["views"][f"C{v}"] = {**ev, "fg_status": model.fg_status, "per_defect_type_image_auroc": per_type, "n_train": len(feats),
                                 "fewshot": few, "seconds": time.time() - t0}
        print(f"C{v}: image AUROC {ev['image_auroc']:.4f} pixel {ev['pixel_auroc_featuregrid']:.4f} "
              f"({ev['n_anomalous']} defective / {ev['n_test']}) types {per_type} few {few}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1))
    au = [x["image_auroc"] for x in rep["views"].values()]
    rep["mean_image_auroc"] = float(np.mean(au))
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print("mean image AUROC over views", rep["mean_image_auroc"], flush=True)


if __name__ == "__main__":
    main()
