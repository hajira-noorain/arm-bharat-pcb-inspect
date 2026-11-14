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


def teach_visa_parts(out: Path, work: Path, goods: int = 8) -> None:
    """Pre-teach the parts maps of the VisA assembled-board designs from their good TRAINING photos (the demo
    boards are test photos), with the shipped ExecuTorch models, so the parts check works in replay mode."""
    det_p, cls_p = out / "models" / "vision" / "parts_y26n_xnnpack.pte", out / "models" / "vision" / "parts_kind_xnnpack.pte"
    visa = work / "data" / "VisA"
    if not det_p.is_file() or not (visa / "split_csv" / "1cls.csv").is_file():
        print("parts maps: skipped (no parts detector or no VisA)")
        return
    import sys
    sys.path.insert(0, str(out))
    import cv2
    import yaml
    try:
        import executorch.runtime  # noqa: F401
    except ImportError:
        print("parts maps: ExecuTorch is not installed here; run them in the export container:\n"
              "  docker run --rm -v <work>:/work -v <repo>:/repo pcb-export python /repo/deploy/make_bundle.py "
              "--work /work --out /work/bundle/pcb-station --parts-maps-only")
        return
    from pcbinspect.detect_bare import BareDetector, ExecuTorchTiles
    from pcbinspect.parts import PartKindClassifier, teach_parts
    det = BareDetector(ExecuTorchTiles(det_p, conf_thr=0.1), names=("part",))
    cls = None
    if cls_p.is_file() and cls_p.with_suffix(".json").is_file():
        cls = PartKindClassifier(cls_p, json.loads(cls_p.with_suffix(".json").read_text())["names"])
    with open(visa / "split_csv" / "1cls.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train" and r["label"] == "normal"]
    for obj in ("pcb1", "pcb2", "pcb3", "pcb4"):
        dy = out / "config" / "designs" / f"visa_{obj}.yaml"
        if not dy.is_file():
            continue
        d = yaml.safe_load(dy.read_text())
        imgs = [cv2.imread(str(visa / r["image"])) for r in rows if r["object"] == obj][:goods]
        pm = teach_parts(det, cls, imgs)
        target = out / d["anomaly_dir"]
        target.mkdir(parents=True, exist_ok=True)
        pm.save(target / "parts.json")
        cv2.imwrite(str(target / "parts_ref.jpg"), imgs[0], [cv2.IMWRITE_JPEG_QUALITY, 95])
        d["parts_map"] = f"{d['anomaly_dir']}/parts.json"
        dy.write_text(yaml.safe_dump(d, sort_keys=False))
        print(f"parts map visa_{obj}: {pm.summary()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=r"C:\Users\sulai\pcb_work")
    ap.add_argument("--out", default=r"C:\Users\sulai\pcb_work\bundle\pcb-station")
    ap.add_argument("--llm", default="llama3_2_1b_spinquant.pte")
    ap.add_argument("--extra-llm", default="llama3_2_3b_spinquant.pte",
                    help="also ship these .pte files (the 3B decides advice, questions and rules; see config/app.yaml)")
    ap.add_argument("--demo-per-design", type=int, default=24)
    ap.add_argument("--tar", action="store_true")
    ap.add_argument("--parts-maps-only", action="store_true",
                    help="only (re)teach the VisA parts maps inside an existing bundle (needs ExecuTorch)")
    args = ap.parse_args()
    work, out = Path(args.work), Path(args.out)
    if args.parts_maps_only:
        teach_visa_parts(out, work)
        return
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
    for v in ("n", "s", "m"):
        if (m / "vision" / f"pcb_y26{v}.pt").exists():
            copy(m / "vision" / f"pcb_y26{v}.pt", out / "models" / "vision" / f"pcb_y26{v}.pt")
    for p in (m / "vision").glob("parts_kind*.json"):
        copy(p, out / "models" / "vision" / p.name)

    # calibration boards (cal split, never trained on): lets the station re-calibrate the conformal
    # gate itself after a learning update
    cal = json.loads((work / "data" / "bare_yolo" / "boards_cal.json").read_text())
    for design in ("deeppcb", "hripcb", "mendeley"):
        cdir = out / "models" / "calib" / design
        cdir.mkdir(parents=True, exist_ok=True)
        items = []
        for i, b in enumerate(x for x in cal if x["source"] == design):
            name = f"{i:04d}.jpg"
            shutil.copy2(b["path"], cdir / name)
            items.append({"file": name, "good": b["good"], "boxes": b["boxes"]})
        (cdir / "manifest.json").write_text(json.dumps(items))

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
            if golden is None and b["good"] and design == "deeppcb":
                golden = b["path"]  # a DeepPCB good board IS its template: the golden is the same pattern
            if golden:
                shutil.copy2(golden, demo / f"{design}_{i:03d}.golden.jpg")
            manifest.append({"file": dst.name, "design": design, "good": b["good"], "boxes": b["boxes"],
                             "golden": bool(golden)})
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

    # detection-limit eye test: the printable card + manifest, and SIMULATED photos of the printed card
    # for replay/demo mode (inkjet dot gain + camera model of training/eyetest_sim.py; not real photos)
    card = work / "eyetest"
    if (card / "manifest.json").exists():
        copy(card / "manifest.json", out / "assets" / "eyetest" / "manifest.json")
        copy(card / "eyetest_card.pdf", out / "assets" / "eyetest" / "eyetest_card.pdf")
        import cv2
        import numpy as np
        import sys
        sys.path.insert(0, str(REPO))
        from training.eyetest_sim import inkjet, photograph
        r = np.random.default_rng(1)
        ed = out / "demo" / "eyetest"
        ed.mkdir(parents=True, exist_ok=True)
        for sp in sorted(card.glob("sheet_*.png")):
            photo = photograph(inkjet(cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE), 40, r), 30, r)
            cv2.imwrite(str(ed / f"simulated_{sp.stem}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 92])
    teach_visa_parts(out, work)
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
