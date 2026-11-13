"""Lens calibration with a printed ChArUco board.

1. Print the board:   python deploy/calibrate_camera.py board --out deploy/assets/charuco_board.pdf
   (print at 100% / "actual size"; measure one square with a ruler: it must be 15 mm)
2. Photograph it 15-20 times at the working height with varied tilt, into one folder.
3. Calibrate:         python deploy/calibrate_camera.py calibrate PHOTOS_DIR --out calibration.json

The JSON holds the camera matrix, distortion, RMS reprojection error (pixels) and mm per pixel at
the board plane. Use pcbinspect.imaging.load_calibration / undistort to apply it.
Works with both the OpenCV 4.7+ CharucoDetector API and the older contrib API.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SQUARES = (7, 5)          # columns x rows
SQUARE_MM = 15.0
MARKER_MM = 11.0
DICT_ID = cv2.aruco.DICT_4X4_50


def make_board(squares=SQUARES, square_mm=SQUARE_MM, marker_mm=MARKER_MM):
    d = cv2.aruco.getPredefinedDictionary(DICT_ID)
    return cv2.aruco.CharucoBoard(tuple(squares), float(square_mm), float(marker_mm), d)


def board_image(board, px_per_mm: float, margin_px: int = 0) -> np.ndarray:
    w = int(round(board.getChessboardSize()[0] * board.getSquareLength() * px_per_mm))
    h = int(round(board.getChessboardSize()[1] * board.getSquareLength() * px_per_mm))
    return board.generateImage((w, h), marginSize=margin_px, borderBits=1)


def write_board_pdf(path, board=None, dpi: int = 600) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    import tempfile
    board = board or make_board()
    cols, rows = board.getChessboardSize()
    sq = board.getSquareLength()
    img = board_image(board, dpi / 25.4)
    png = Path(tempfile.gettempdir()) / "charuco_board.png"
    cv2.imwrite(str(png), img)
    w_mm, h_mm = cols * sq, rows * sq
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    W, H = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, H - 20 * mm, "ChArUco calibration board (DICT_4X4_50)")
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, H - 27 * mm, f"{cols}x{rows} squares, {sq:g} mm square, {board.getMarkerLength():g} mm marker."
                 " PRINT AT 100% (actual size, no fit-to-page).")
    x0, y0 = (W - w_mm * mm) / 2, H - 40 * mm - h_mm * mm
    c.drawImage(str(png), x0, y0, width=w_mm * mm, height=h_mm * mm)
    c.rect(x0, y0, w_mm * mm, h_mm * mm)
    # 50 mm check ruler
    ry = y0 - 15 * mm
    c.line(x0, ry, x0 + 50 * mm, ry)
    for i in range(0, 51, 10):
        c.line(x0 + i * mm, ry - 2 * mm, x0 + i * mm, ry + 2 * mm)
    c.drawString(x0, ry - 8 * mm, "This line must measure 50 mm with a real ruler. Stick the sheet flat on a stiff board.")
    c.save()


def detect_corners(gray: np.ndarray, board):
    """Return (charuco_corners Nx1x2, ids Nx1) or (None, None)."""
    if hasattr(cv2.aruco, "CharucoDetector"):
        det = cv2.aruco.CharucoDetector(board)
        cc, ci, _, _ = det.detectBoard(gray)
    else:  # legacy contrib API
        dic = board.getDictionary() if hasattr(board, "getDictionary") else board.dictionary
        mc, mi, _ = cv2.aruco.detectMarkers(gray, dic)
        if mi is None or len(mi) == 0:
            return None, None
        n, cc, ci = cv2.aruco.interpolateCornersCharuco(mc, mi, gray, board)
        if n == 0:
            return None, None
    if cc is None or ci is None or len(cc) < 6:
        return None, None
    return cc, ci


def _board_points(board, cc, ci):
    if hasattr(board, "matchImagePoints"):
        obj, img = board.matchImagePoints(cc, ci)
        return np.asarray(obj, np.float32).reshape(-1, 3), np.asarray(img, np.float32).reshape(-1, 2)
    corners = np.asarray(board.getChessboardCorners() if hasattr(board, "getChessboardCorners")
                         else board.chessboardCorners, np.float32).reshape(-1, 3)
    return corners[ci.ravel()], np.asarray(cc, np.float32).reshape(-1, 2)


def calibrate_gray(grays: list[np.ndarray], board=None, min_views: int = 4) -> dict:
    """Calibrate from grayscale views of the ChArUco board. Returns the calibration dict."""
    board = board or make_board()
    h, w = grays[0].shape[:2]
    obj_pts, img_pts, used = [], [], []
    for k, g in enumerate(grays):
        cc, ci = detect_corners(g, board)
        if cc is None:
            continue
        o, i = _board_points(board, cc, ci)
        obj_pts.append(o)
        img_pts.append(i)
        used.append(k)
    if len(used) < min_views:
        raise RuntimeError(f"board found in only {len(used)} of {len(grays)} photos (need {min_views}+)")
    # initial guess without tilt freedom problems: let OpenCV estimate f, with k3 fixed (mild lenses)
    flags = cv2.CALIB_FIX_K3
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None, flags=flags)
    # per-view reprojection error, computed ourselves so it is a true pixel RMS
    se, n = 0.0, 0
    for o, i, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, D)
        se += float(((proj.reshape(-1, 2) - i) ** 2).sum())
        n += len(o)
    rms_px = float(np.sqrt(se / max(n, 1)))
    # mm/px at the board plane: depth of the board centre divided by focal length (pixels)
    cols, rows = board.getChessboardSize()
    centre = np.array([cols * board.getSquareLength() / 2, rows * board.getSquareLength() / 2, 0.0])
    f = float((K[0, 0] + K[1, 1]) / 2)
    mm_px = []
    for rv, tv in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rv)
        z = float((R @ centre + tv.reshape(3))[2])
        mm_px.append(z / f)
    return {"image_size": [int(w), int(h)], "camera_matrix": K.tolist(),
            "distortion": D.reshape(-1).tolist(), "rms_reprojection_px": rms_px,
            "opencv_rms_px": float(rms), "views_used": len(used), "views_total": len(grays),
            "mm_per_px": float(np.median(mm_px)), "mm_per_px_spread": float(np.std(mm_px)),
            "board": {"squares": list(board.getChessboardSize()), "square_mm": board.getSquareLength(),
                      "marker_mm": board.getMarkerLength(), "dictionary": "DICT_4X4_50"},
            "opencv_version": cv2.__version__}


def calibrate_folder(folder, out=None) -> dict:
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    grays = []
    for p in files:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is not None:
            grays.append(im)
    if not grays:
        raise RuntimeError(f"no images in {folder}")
    if len({g.shape for g in grays}) != 1:
        raise RuntimeError("all photos must have the same size")
    result = calibrate_gray(grays)
    if out:
        Path(out).write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("board", help="write the printable board PDF")
    b.add_argument("--out", default=str(Path(__file__).resolve().parent / "assets" / "charuco_board.pdf"))
    c = sub.add_parser("calibrate", help="calibrate from a folder of photos")
    c.add_argument("folder")
    c.add_argument("--out", default="calibration.json")
    a = ap.parse_args()
    if a.cmd == "board":
        write_board_pdf(a.out)
        print("wrote", a.out)
    else:
        r = calibrate_folder(a.folder, a.out)
        print(f"views {r['views_used']}/{r['views_total']}  RMS {r['rms_reprojection_px']:.3f} px  "
              f"{r['mm_per_px']:.5f} mm/px ({1 / r['mm_per_px']:.1f} px/mm)  -> {a.out}")
        if r["rms_reprojection_px"] > 1.0:
            print("WARNING: error above 1 px; retake photos (sharper, flatter board, more tilt variety)")


if __name__ == "__main__":
    main()
