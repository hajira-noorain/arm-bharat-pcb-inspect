"""Make the printable detection-limit "eye test" card: one golden sheet + test sheets, and a manifest.

Board patterns are held-out DeepPCB templates (MIT licence, (c) 2018 tangsanli5201), never used for
training. Defects are drawn with pcbinspect.defect_render (every open really severs, every short
really joins) at SIZES_MM, until each (type, size) has --per trials.

Print every page at 100% / "actual size" on white paper; check the 100 mm bar with a ruler.

python deploy/make_eyetest_card.py --data <bare_yolo> --out deploy/assets/eyetest
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.defect_render import TYPES, copper_mask, render_board, to_image  # noqa: E402
from pcbinspect.eyetest import (COLS, COUPON_MM, FRAME_H, FRAME_W, RECT_PX_PER_MM, ROWS, SIZES_MM,  # noqa: E402
                                coupon_origin_mm, sheet_raster, sizes_for)


def write_pdf(pages, path: Path, titles):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    from PIL import Image

    c = canvas.Canvas(str(path), pagesize=A4)
    W, H = A4
    for img, title in zip(pages, titles):
        x0 = (210 - FRAME_W) / 2
        y_top = 30.0
        c.setFont("Helvetica-Bold", 13)
        c.drawString(x0 * mm, H - 16 * mm, title)
        c.setFont("Helvetica", 8.5)
        c.drawString(x0 * mm, H - 22 * mm, "Print at 100% / actual size (no 'fit to page'). Keep flat; do not fold the pattern area.")
        c.drawImage(ImageReader(Image.fromarray(img)), x0 * mm, H - (y_top + FRAME_H) * mm,
                    width=FRAME_W * mm, height=FRAME_H * mm)
        # 100 mm scale check
        yb = H - (y_top + FRAME_H + 14) * mm
        c.setLineWidth(0.5)
        c.line(x0 * mm, yb, (x0 + 100) * mm, yb)
        for k in range(0, 101, 5):
            t = 3 if k % 10 == 0 else 1.5
            c.line((x0 + k) * mm, yb, (x0 + k) * mm, yb + t * mm)
            if k % 20 == 0:
                c.drawString((x0 + k - 1.5) * mm, yb - 4 * mm, str(k))
        c.drawString((x0 + 104) * mm, yb, "mm: must measure 100 mm with a ruler")
        c.setFont("Helvetica", 7)
        c.drawString(x0 * mm, 12 * mm, "Board patterns: DeepPCB templates, MIT licence (c) 2018 tangsanli5201. "
                                       "Defects drawn by pcbinspect.defect_render.")
        c.showPage()
    c.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", type=int, default=30)
    ap.add_argument("--max-sheets", type=int, default=16)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    boards = json.loads((Path(a.data) / "boards_test.json").read_text())
    temps = sorted({b["template"] for b in boards if b["source"] == "deeppcb" and b.get("template")})
    n_cp = COLS * ROWS
    chosen = [temps[i] for i in rng.choice(len(temps), n_cp, replace=False)]
    clean = [cv2.imread(t) for t in chosen]
    px = RECT_PX_PER_MM

    need = Counter({(t, s): a.per for t in TYPES for s in sizes_for(t)})
    sheets, manifest = [], {"sizes_mm": SIZES_MM, "sizes_by_type": {t: sizes_for(t) for t in TYPES}, "types": TYPES, "per": a.per, "frame_mm": [FRAME_W, FRAME_H],
                            "coupon_mm": COUPON_MM, "px_per_mm": px,
                            "templates": [Path(t).name for t in chosen], "sheets": []}
    # golden coupons go through the same binarisation as the drawn ones, so only the defects differ
    sheets.append(sheet_raster([cv2.cvtColor(to_image(copper_mask(c)), cv2.COLOR_BGR2GRAY) for c in clean], 0))
    manifest["sheets"].append({"sheet": 0, "golden": True, "coupons": [{"idx": i, "defects": []} for i in range(n_cp)]})
    k = 0
    while sum(need.values()) > 0 and k < a.max_sheets:
        k += 1
        coupons, rec = [], []
        for i in range(n_cp):
            # ask for up to 6 defects, favouring the (type, size) pairs still most needed
            pool = [p for p, v in need.items() if v > 0]
            rng.shuffle(pool)
            pool.sort(key=lambda p: -need[p])
            specs = pool[:6]
            img, placed = render_board(clean[i], specs, px, rng)
            for p in placed:
                need[(p.cls, p.size_mm)] -= 1
            ox, oy = coupon_origin_mm(i)
            rec.append({"idx": i, "defects": [{"type": p.cls, "size_mm": p.size_mm,
                                               "center_mm": [ox + p.center[0] / px, oy + p.center[1] / px],
                                               "bbox_mm": [ox + p.bbox[0] / px, oy + p.bbox[1] / px,
                                                           ox + p.bbox[2] / px, oy + p.bbox[3] / px]}
                                              for p in placed]})
            coupons.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        sheets.append(sheet_raster(coupons, k))
        manifest["sheets"].append({"sheet": k, "golden": False, "coupons": rec})
        print(f"sheet {k}: {sum(len(c['defects']) for c in rec)} defects; still needed {sum(max(v, 0) for v in need.values())}",
              flush=True)
    manifest["short_of_target"] = {f"{t}@{s}": v for (t, s), v in need.items() if v > 0}
    for i, s in enumerate(sheets):
        cv2.imwrite(str(out / f"sheet_{i:02d}.png"), s)
    titles = ["Eye test card - sheet 0: GOLDEN (no defects)"] + \
             [f"Eye test card - sheet {i}: test" for i in range(1, len(sheets))]
    write_pdf(sheets, out / "eyetest_card.pdf", titles)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(sheets)} pages -> {out / 'eyetest_card.pdf'}")


if __name__ == "__main__":
    main()
