"""Assemble the self-contained station bundle that is copied to the Raspberry Pi (or the simulator).

  bundle/
    pcbinspect/ config/ kb/ bench/ deploy/ README.md pyproject.toml
    models/vision/*.pte          detector + backbone (ExecuTorch, XNNPACK)
    models/llm/*.pte, tokenizer  Llama 3.2 (ExecuTorch)
    models/anomaly/<design>/     memory banks
    models/gate/<design>.json    conformal calibrations
    demo/boards/<design>_NNN.jpg held-out TEST boards for replay and benchmarks (never trained on)

Usage: python deploy/make_bundle.py --work C:\\Users\\sulai\\pcb_work --out C:\\Users\\sulai\\pcb_work\\bundle
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CODE = ["pcbinspect", "config", "kb", "bench", "deploy", "README.md", "pyproject.toml"]


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=r"C:\Users\sulai\pcb_work")
    ap.add_argument("--out", default=r"C:\Users\sulai\pcb_work\bundle\pcb-station")
    ap.add_argument("--llm", default="llama3_2_1b_spinquant.pte")
    ap.add_argument("--extra-llm", default="", help="also ship another .pte (e.g. the 3B) for comparison")
    ap.add_argument("--demo-per-design", type=int, default=24)
    ap.add_argument("--tar", action="store_true")
    args = ap.parse_args()
    work, out = Path(args.work), Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for c in CODE:
        if (REPO / c).exists():
            copy(REPO / c, out / c)

    m = work / "models"
    for p in (m / "vision").glob("*.pte"):
        copy(p, out / "models" / "vision" / p.name)
    for name in [args.llm] + [x for x in args.extra_llm.split(",") if x]:
        copy(m / "llm" / name, out / "models" / "llm" / name)
    copy(m / "llm" / "tokenizer.model", out / "models" / "llm" / "tokenizer.model")
    if (m / "anomaly").exists():
        copy(m / "anomaly", out / "models" / "anomaly")
    if (m / "gate").exists():
        copy(m / "gate", out / "models" / "gate")
    if (m / "vision" / "pcb_y26n.pt").exists():
        copy(m / "vision" / "pcb_y26n.pt", out / "models" / "vision" / "pcb_y26n.pt")

    # demo boards: held-out TEST images only, a mix of defective and good
    rng = random.Random(0)
    demo = out / "demo" / "boards"
    demo.mkdir(parents=True, exist_ok=True)
    manifest = []
    test = json.loads((work / "data" / "bare_yolo" / "boards_test.json").read_text())
    for design in ("deeppcb", "hripcb", "mendeley"):
        items = [b for b in test if b["source"] == design]
        good = [b for b in items if b["good"]]
        bad = [b for b in items if not b["good"]]
        rng.shuffle(bad)
        pick = bad[: args.demo_per_design - min(len(good), args.demo_per_design // 4)]
        pick += good[: args.demo_per_design // 4]
        rng.shuffle(pick)
        for i, b in enumerate(pick):
            dst = demo / f"{design}_{i:03d}.jpg"
            shutil.copy2(b["path"], dst)
            # known-good image of the same board, when the dataset has one (used for mm measurement)
            golden = b.get("template")
            if golden is None and design == "hripcb":
                board = Path(b["path"]).stem.split("_")[0]
                golden = next(iter((work / "data" / "HRIPCB" / "PCB_USED").glob(f"{board}.*")), None)
            if golden and not b["good"]:
                shutil.copy2(golden, demo / f"{design}_{i:03d}.golden.jpg")
            manifest.append({"file": dst.name, "design": design, "good": b["good"], "boxes": b["boxes"],
                             "golden": bool(golden and not b["good"])})
    visa = work / "data" / "VisA"
    if visa.exists():
        with open(visa / "split_csv" / "1cls.csv", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == "test" and r["object"].startswith("pcb")]
        for obj in ("pcb1", "pcb2", "pcb3", "pcb4"):
            # only boards from the gate's TEST half would be cleanest; VisA test images are never in the bank
            rs = [r for r in rows if r["object"] == obj]
            rng.shuffle(rs)
            for i, r in enumerate(rs[: args.demo_per_design]):
                dst = demo / f"visa_{obj}_{i:03d}.jpg"
                shutil.copy2(visa / r["image"], dst)
                manifest.append({"file": dst.name, "design": f"visa_{obj}", "good": r["label"] == "normal"})
    (out / "demo" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "results").mkdir(exist_ok=True)
    for p in (work / "results").glob("*.json"):
        if not p.name.startswith(("bare_dets_", "emb_")):
            copy(p, out / "results" / "laptop" / p.name)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 2**20
    print(f"bundle at {out}  ({size:.0f} MB, {len(manifest)} demo boards)")
    if args.tar:
        t = out.with_suffix(".tar")
        with tarfile.open(t, "w") as tf:
            tf.add(out, arcname=out.name)
        print("tar:", t)


if __name__ == "__main__":
    main()
