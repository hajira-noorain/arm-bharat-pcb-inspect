"""Printable A4 sheet with the 20 mm ArUco scale marker and a 50 mm check ruler.

Print at 100% ("actual size", no "fit to page"), measure the ruler with a real ruler, then place a
marker in the same plane as the board, inside the camera view.
python deploy/make_marker_sheet.py --out deploy/assets/scale_marker_sheet.pdf
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.calib import MARKER_ID, make_marker_png  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "assets" / "scale_marker_sheet.pdf"))
    ap.add_argument("--marker-mm", type=float, default=20.0)
    a = ap.parse_args()
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    png = Path(tempfile.gettempdir()) / "aruco_marker.png"
    make_marker_png(str(png), a.marker_mm, dpi=600)
    # make_marker_png adds a quiet zone of marker/4 on each side and a caption strip
    import cv2
    im = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
    px_marker = int(round(a.marker_mm / 25.4 * 600))
    scale_mm_per_px = a.marker_mm / px_marker
    w_mm, h_mm = im.shape[1] * scale_mm_per_px, im.shape[0] * scale_mm_per_px

    c = canvas.Canvas(a.out, pagesize=A4)
    W, H = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, H - 20 * mm, "PCB inspection station: scale markers")
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, H - 27 * mm, "Print at 100% (actual size). Check the 50 mm ruler with a real ruler before use.")
    c.drawString(20 * mm, H - 32 * mm, f"Each black square is exactly {a.marker_mm:g} mm. Cut one out and lay it flat beside the board.")
    for i, (x, y) in enumerate([(20, 150), (110, 150), (20, 70), (110, 70)]):
        c.drawImage(str(png), x * mm, y * mm, width=w_mm * mm, height=h_mm * mm)
    # 50 mm check ruler
    y0 = 40 * mm
    c.setLineWidth(0.6)
    c.line(20 * mm, y0, 70 * mm, y0)
    for k in range(0, 51):
        tick = 4 if k % 10 == 0 else (2.5 if k % 5 == 0 else 1.5)
        c.line((20 + k) * mm, y0, (20 + k) * mm, y0 + tick * mm)
        if k % 10 == 0:
            c.drawString((20 + k - 1) * mm, y0 - 5 * mm, str(k))
    c.drawString(75 * mm, y0, "mm check ruler")
    c.setFont("Helvetica", 8)
    c.drawString(20 * mm, 15 * mm, f"ArUco DICT_4X4_50, marker id {MARKER_ID}.")
    c.save()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
