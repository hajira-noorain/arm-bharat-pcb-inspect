"""Which bare-board detector the station runs, and the PASS/FAIL/CHECK thresholds that belong to it.

Three detectors are shipped: YOLO26n (the first one), and the retrained YOLO26s and YOLO26m (more accurate on every
held-out test set, slower). Which one runs is decided by measured speed on the Pi, with one setting in
config/app.yaml:

    detector_variant: n      # n | s | m

A detector and its thresholds are never mixed: the thresholds are calibrated on that detector's own scores, so
the variant swaps both together. If a variant's model file or any of its threshold files is missing, the station
keeps the first detector and its thresholds and says so in the log.

    models/vision/pcb_y26{v}_xnnpack.pte    (n: the existing pcb_y26n_xnnpack.pte)
    models/gate/y26{v}/<design>.json        (n: models/gate/<design>.json)
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

log = logging.getLogger("pcbinspect.variants")
VARIANTS = ("n", "s", "m")


def apply_detector_variant(cfg: dict, designs: dict, root: Path) -> tuple[dict, dict, str]:
    """(cfg, designs, variant actually used). Returns new objects; the inputs are not changed."""
    v = str(cfg.get("detector_variant", "n")).strip().lower()
    if v not in VARIANTS:
        log.warning("detector_variant %r is not one of %s: using n", v, VARIANTS)
        return dict(cfg), dict(designs), "n"
    if v == "n":
        return dict(cfg), dict(designs), "n"
    root = Path(root)
    pte = f"models/vision/pcb_y26{v}_xnnpack.pte"
    gdir = f"models/gate/y26{v}"
    new_designs, missing = {}, []
    for name, d in designs.items():
        swap = {}
        for field in ("gate_file", "gate_golden"):
            cur = getattr(d, field)
            if not cur or not d.use_detector:
                continue
            alt = f"{gdir}/{Path(cur).name}"
            if (root / alt).is_file():
                swap[field] = alt
            else:
                missing.append(alt)
        new_designs[name] = replace(d, **swap) if swap else d
    if not (root / pte).is_file():
        missing.append(pte)
    if missing:
        log.warning("detector_variant %s incomplete (missing %s): keeping YOLO26n and its thresholds", v, missing)
        return dict(cfg), dict(designs), "n"
    out = dict(cfg)
    out["detector_pte"] = pte
    out["detector_pt"] = f"models/vision/pcb_y26{v}.pt"
    log.info("detector_variant %s: %s with thresholds from %s", v, pte, gdir)
    return out, new_designs, v
