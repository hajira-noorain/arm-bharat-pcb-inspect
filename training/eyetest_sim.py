"""End-to-end check of the eye-test reader on SIMULATED photos of the printed card (before printing).

Each sheet raster goes through: inkjet print (ink spread + blur, --ink-um), then a camera (perspective
view, rotation, --px-per-mm on the paper, lens blur, sensor noise, JPEG). The station side then has to
find the markers, rectify, read every coupon against the golden sheet and match every planted defect.
Reported per channel (detector only / fused with golden): the station's statement per defect type and
false calls per coupon. This validates the reader, NOT the physical print: that needs the real card.

python training/eyetest_sim.py --card <dir with manifest.json + sheet_XX.png> --weights best.pt \
    --gate models/gate/deeppcb.json --golden-gate models/gate/deeppcb_golden.json --out results/eyetest_sim.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.detect_bare import BareDetector, UltralyticsTiles  # noqa: E402
from pcbinspect.eyetest import FRAME_H, FRAME_W, RECT_PX_PER_MM, load_manifest, read_sheet, rectify, summarise  # noqa: E402
from pcbinspect.gate import ConformalGate  # noqa: E402


def inkjet(sheet: np.ndarray, ink_um: float, rng) -> np.ndarray:
    """Ink spreads into paper fibres: black grows by ~ink_um and edges soften."""
    px = ink_um / 1000 * RECT_PX_PER_MM
    ink = (sheet < 128).astype(np.uint8)
    r = max(1, int(round(px)))
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    img = np.where(ink > 0, 25, 240).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), max(0.5, px))
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


PAPER_MARGIN_MM = 8.0


def photograph(printed: np.ndarray, px_per_mm: float, rng, sensor=(4608, 2592)) -> np.ndarray:
    """Place the printed frame in a camera image with a mild perspective and rotation."""
    W, H = sensor
    # the frame sits on a white page: keep a white paper margin (ArUco needs a light quiet zone)
    m = int(PAPER_MARGIN_MM * RECT_PX_PER_MM)
    printed = cv2.copyMakeBorder(printed, m, m, m, m, cv2.BORDER_CONSTANT, value=240)
    fw = (FRAME_W + 2 * PAPER_MARGIN_MM) * px_per_mm
    fh = (FRAME_H + 2 * PAPER_MARGIN_MM) * px_per_mm
    cx, cy = W / 2 + rng.uniform(-80, 80), H / 2 + rng.uniform(-60, 60)
    ang = np.deg2rad(rng.uniform(-4, 4))
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    base = np.array([[-fw / 2, -fh / 2], [fw / 2, -fh / 2], [fw / 2, fh / 2], [-fw / 2, fh / 2]])
    keystone = rng.uniform(-0.02, 0.02) * fw
    base[0, 0] += keystone
    base[1, 0] -= keystone
    dst = (base @ R.T + [cx, cy]).astype(np.float32)
    h, w = printed.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    paper = cv2.warpPerspective(printed, M, (W, H), flags=cv2.INTER_AREA, borderValue=90)
    img = cv2.GaussianBlur(paper.astype(np.float32), (0, 0), 0.9)
    img = img * rng.uniform(0.85, 1.0) + rng.normal(0, 4, img.shape)
    img = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--golden-gate", required=True)
    ap.add_argument("--px-per-mm", type=float, default=30.0, help="camera resolution on the paper")
    ap.add_argument("--ink-um", type=float, default=40.0, help="inkjet dot gain (micrometres)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    card = Path(a.card)
    man = load_manifest(card / "manifest.json")
    rng = np.random.default_rng(0)
    det = BareDetector(UltralyticsTiles(a.weights, device=a.device))
    lam_d = ConformalGate.load(a.gate).thresholds(a.alpha, a.alpha).lam
    lam_g = ConformalGate.load(a.golden_gate).thresholds(a.alpha, a.alpha).lam

    shots = {}
    for sh in man["sheets"]:
        raster = cv2.imread(str(card / f"sheet_{sh['sheet']:02d}.png"), cv2.IMREAD_GRAYSCALE)
        photo = photograph(inkjet(raster, a.ink_um, rng), a.px_per_mm, rng)
        r = rectify(photo)
        if not r.ok or r.sheet != sh["sheet"]:
            raise SystemExit(f"sheet {sh['sheet']}: rectification failed ({r.reason}, read id {r.sheet})")
        shots[sh["sheet"]] = r.img
        print(f"sheet {sh['sheet']}: rectified, id read {r.sheet}", flush=True)
    golden = shots[0]
    report = {"px_per_mm": a.px_per_mm, "ink_um": a.ink_um, "alpha": a.alpha, "channels": {}}
    for name, use_golden, lam in (("detector", False, lam_d), ("fused_golden", True, lam_g)):
        trials = []
        for sh in man["sheets"]:
            if sh["golden"]:
                continue
            trials += read_sheet(shots[sh["sheet"]], golden, sh, det, use_golden=use_golden)
        s = summarise(trials, lam)
        report["channels"][name] = s
        print(f"== {name} (lambda {lam:.3f}), false calls per coupon {s['false_calls_per_coupon']:.2f}")
        for t, v in s["types"].items():
            print("   ", v["sentence"])
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
