"""Thermal soak: inspect demo boards continuously (with Llama advice on every failing board) and log
latency, CPU temperature and throttle flags over time. Shows whether performance holds up on a
real shift, not just in a cold benchmark.

Usage on the Pi:  python -m bench.soak --minutes 30
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import time
from pathlib import Path

import cv2
import yaml

from bench.bench_device import set_threads, temp_c, throttled
from pcbinspect.advisor import Advisor
from pcbinspect.anomaly import executorch_feature_fn
from pcbinspect.detect_bare import BareDetector, ExecuTorchTiles
from pcbinspect.pipeline import Inspector, load_designs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--design", default="deeppcb")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    cfg = yaml.safe_load((root / "config" / "app.yaml").read_text())
    set_threads(cfg.get("llm_threads", 4))
    designs = load_designs(root / cfg["designs_dir"])
    det = BareDetector(ExecuTorchTiles(root / cfg["detector_pte"]))
    fn = executorch_feature_fn(root / cfg["backbone_pte"])
    insp = Inspector(root, designs, det, fn)
    llm = None
    if not args.no_llm:             # the same model and decision method the station itself uses
        from pcbinspect.llm_select import load_llama
        llm = load_llama(root, cfg)
    advisor = Advisor(llm, cause_mode=cfg.get("advice_cause_mode", "list"))
    imgs = sorted(glob.glob(str(root / "demo" / "boards" / f"{args.design}_*.jpg")))
    if not imgs:
        raise SystemExit(f"no demo boards for {args.design}")
    out_csv = root / "results" / f"soak_{args.design}.csv"
    out_csv.parent.mkdir(exist_ok=True)
    t_end = time.time() + args.minutes * 60
    rows, i = [], 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "board", "verdict", "verdict_ms", "advice_ms", "decode_tps", "temp_c", "throttled"])
        t0 = time.time()
        while time.time() < t_end:
            p = imgs[i % len(imgs)]
            i += 1
            bgr = cv2.imread(p)
            ins, a, _ = insp.inspect(bgr, args.design, log=False)
            adv_ms = tps = None
            if ins.defects:
                tt = time.perf_counter()
                adv = advisor.advise(ins.to_llm_record())
                adv_ms = (time.perf_counter() - tt) * 1e3
                tps = adv.gen.decode_tps if adv.gen else None
            row = [round(time.time() - t0, 1), Path(p).name, ins.verdict,
                   round(a.latency_ms["total_to_verdict"], 1), adv_ms and round(adv_ms, 1),
                   tps and round(tps, 2), temp_c(), throttled()]
            w.writerow(row)
            f.flush()
            rows.append(row)
            if i % 10 == 0:
                print(row, flush=True)
    summary = {"minutes": args.minutes, "inspections": len(rows),
               "temp_max_c": max((r[6] for r in rows if r[6] is not None), default=None),
               "throttled_any": sorted({r[7] for r in rows}),
               "verdict_ms_first10_median": sorted(r[3] for r in rows[:10])[len(rows[:10]) // 2],
               "verdict_ms_last10_median": sorted(r[3] for r in rows[-10:])[len(rows[-10:]) // 2]}
    (root / "results" / f"soak_{args.design}.json").write_text(json.dumps(summary, indent=1))
    print(summary)


if __name__ == "__main__":
    main()
