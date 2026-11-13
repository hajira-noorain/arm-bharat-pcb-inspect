"""Capture-quality, flat-field, white-balance, averaging and lens-calibration tests (synthetic data)."""
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcbinspect import imaging  # noqa: E402
from pcbinspect.camera import Camera, PiCamera, pick_focus, focus_score  # noqa: E402

_spec = importlib.util.spec_from_file_location("calibrate_camera", ROOT / "deploy" / "calibrate_camera.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


def textured(h=300, w=400, seed=0, base=120):
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 40, (h, w, 3)).astype(np.float32)
    return np.clip(base + n, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- flat field
def vignette(h, w, strength=0.6):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
    v = 1 - strength * r2
    return np.stack([v, v * 0.95, v * 0.9], axis=-1)        # slightly different per channel


def test_flat_field_removes_vignette():
    h, w = 240, 320
    vig = vignette(h, w)
    rng = np.random.default_rng(1)
    card = [np.clip(200 * vig + rng.normal(0, 1.5, vig.shape), 0, 255).astype(np.uint8) for _ in range(8)]
    ff = imaging.FlatField.from_frames(card)
    scene = np.clip(180 * vig, 0, 255).astype(np.uint8)       # uniform surface seen through the vignette
    resid = lambda a: float(np.mean([a[20:-20, 20:-20, c].std() for c in range(3)]))   # spatial, per channel
    before = resid(scene.astype(np.float32))
    after = ff.apply(scene).astype(np.float32)
    after_std = resid(after)
    assert before / after_std > 5
    assert abs(after.mean() - 180 * vig.mean()) < 60           # still a sensible brightness


def test_flat_field_save_load(tmp_path):
    frames = [np.full((60, 90, 3), 150, np.uint8)] * 2
    ff = imaging.FlatField.from_frames(frames)
    ff.save(tmp_path / "ff.npz")
    ff2 = imaging.FlatField.load(tmp_path / "ff.npz")
    assert np.allclose(ff.gain, ff2.gain)
    img = textured(60, 90)
    assert np.array_equal(ff.apply(img), ff2.apply(img))


# ---------------------------------------------------------------- white balance
def test_grey_patch_white_balance_neutralises_cast():
    img = np.full((100, 100, 3), 128, np.uint8)
    img[:, :, 0] = 100   # B low
    img[:, :, 2] = 160   # R high  (warm cast)
    img[40:60, 40:60] = (100, 128, 160)                      # grey patch under the cast
    out = imaging.grey_patch_white_balance(img, (40, 40, 20, 20))
    patch = out[40:60, 40:60].reshape(-1, 3).mean(axis=0)
    assert patch.max() - patch.min() <= 2
    before = img[40:60, 40:60].reshape(-1, 3).mean(axis=0)
    assert before.max() - before.min() > 50


# ---------------------------------------------------------------- quality gate
def test_capture_quality_good_passes():
    q = imaging.capture_quality(textured())
    assert q["ok"], q["reasons"]
    assert q["exposure"] == "ok"


def test_capture_quality_flags_blur():
    q = imaging.capture_quality(cv2.GaussianBlur(textured(), (0, 0), 4))
    assert not q["ok"] and any("blur" in r for r in q["reasons"])


def test_capture_quality_flags_clipping():
    img = textured()
    img[:, :100] = 255                                        # 25 % blown out
    q = imaging.capture_quality(img)
    assert not q["ok"] and any("clipped" in r for r in q["reasons"])
    assert q["clipped_high"] > 0.2


def test_capture_quality_flags_dark_and_bright():
    dark = (textured(base=120) * 0.2).astype(np.uint8)
    q = imaging.capture_quality(dark)
    assert not q["ok"] and q["exposure"] == "under" and any("dark" in r for r in q["reasons"])
    bright = np.clip(textured(base=235).astype(int), 0, 249).astype(np.uint8)
    q = imaging.capture_quality(bright)
    assert q["exposure"] == "over" and not q["ok"]


def test_capture_profile_tolerances():
    refs = [textured(seed=s) for s in range(3)]
    prof = imaging.CaptureProfile.from_images(refs)
    assert prof.check(textured(seed=9))["ok"]
    assert not prof.check(cv2.GaussianBlur(textured(seed=9), (0, 0), 3))["ok"]
    r = prof.check(np.clip(textured(seed=9).astype(int) * 1.3, 0, 249).astype(np.uint8))
    assert not r["ok"] and any("brightness" in x for x in r["reasons"])
    cast = textured(seed=9).astype(np.float32)
    cast[..., 2] *= 1.25                                       # red cast
    r = prof.check(np.clip(cast, 0, 249).astype(np.uint8))
    assert any("colour cast" in x for x in r["reasons"])
    # small drift inside tolerance still passes
    mild = np.clip(textured(seed=9).astype(np.float32) * 1.08, 0, 249).astype(np.uint8)
    assert prof.check(mild)["ok"]


def test_capture_profile_roundtrip(tmp_path):
    prof = imaging.CaptureProfile.from_images([textured()])
    prof.save(tmp_path / "p.json")
    assert imaging.CaptureProfile.load(tmp_path / "p.json") == prof


# ---------------------------------------------------------------- averaging, focus
class FakeNoisyCamera(Camera):
    name = "fake"

    def __init__(self, truth, sigma=10.0, seed=0):
        self.truth, self.sigma = truth, sigma
        self.rng = np.random.default_rng(seed)

    def _grab(self):
        n = self.rng.normal(0, self.sigma, self.truth.shape)
        return np.clip(np.rint(self.truth + n), 0, 255).astype(np.uint8)


def test_frame_averaging_reduces_noise_by_sqrt_n():
    truth = np.full((200, 200, 3), 128, np.uint8)
    cam = FakeNoisyCamera(truth, sigma=10.0)
    s1 = (cam.capture().astype(float) - 128).std()
    s4 = (cam.capture(n_average=4).astype(float) - 128).std()
    s16 = (cam.capture(n_average=16).astype(float) - 128).std()
    assert s1 / s4 == pytest.approx(2.0, rel=0.15)
    assert s1 / s16 == pytest.approx(4.0, rel=0.15)
    assert cam.capture(n_average=4).dtype == np.uint8


class FakeFocusPi(PiCamera):
    """PiCamera with the Picamera2 calls replaced: sharpness peaks at lens position 9."""

    def __init__(self):                                       # deliberately skips Picamera2
        self.lens_position = None
        self.sequence = []
        self.base = textured(200, 300, seed=3)

    def _set_controls(self, controls, discard=None):
        self.sequence.append(controls)

    def _grab(self):
        sigma = 0.3 + 1.5 * abs(self.lens_position - 9.0)
        return cv2.GaussianBlur(self.base, (0, 0), sigma)


def test_focus_sweep_picks_sharpest_and_locks():
    cam = FakeFocusPi()
    rep = cam.focus_sweep([7, 8, 9, 10, 11])
    assert rep["best"] == 9.0 and cam.lens_position == 9.0
    assert rep["confident"] and not rep["at_edge"]
    assert cam.sequence[-1] == {"AfMode": 0, "LensPosition": 9.0}   # manual focus locked at the end
    assert [r["region"] for r in rep["results"]] == ["frame"] * 5


def test_focus_score_uses_marker_region():
    from pcbinspect.calib import make_marker_png
    import tempfile
    p = Path(tempfile.gettempdir()) / "t_marker.png"
    make_marker_png(str(p), 20.0, dpi=300)
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    canvas = np.full((600, 800), 200, np.uint8)
    canvas[100:100 + m.shape[0], 100:100 + m.shape[1]] = m
    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    score, region = focus_score(bgr)
    assert region == "marker" and score > 0


def test_pick_focus_flags_flat_sweep():
    rep = pick_focus([{"lens_position": p, "score": 10.0, "region": "frame"} for p in (7, 8, 9)])
    assert not rep["confident"]


# ---------------------------------------------------------------- lens calibration (synthetic)
W, H = 1280, 960
F, PPM = 1500.0, 10.0          # focal length px; board texture px per mm
K_TRUE = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
D_TRUE = np.array([-0.08, 0.03, 0, 0, 0])


def render_view(board, rvec, tvec, tex, margin):
    """Camera image of the flat board: homography to the board plane plus radial distortion."""
    R, _ = cv2.Rodrigues(rvec)
    Hm = K_TRUE.copy() @ np.column_stack([R[:, 0], R[:, 1], tvec.reshape(3)])
    # distorted pixel -> undistorted pixel -> board mm -> texture px
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pts = np.stack([uu.ravel(), vv.ravel()], axis=-1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K_TRUE, D_TRUE, P=K_TRUE).reshape(-1, 2)
    hom = np.linalg.inv(Hm) @ np.vstack([und.T, np.ones(len(und))])
    xy = (hom[:2] / hom[2]).T
    mapx = (xy[:, 0] * PPM + margin).reshape(H, W).astype(np.float32)
    mapy = (xy[:, 1] * PPM + margin).reshape(H, W).astype(np.float32)
    return cv2.remap(tex, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def test_charuco_calibration_recovers_lens_and_scale():
    board = cc.make_board()
    margin = 60
    tex = board.generateImage((int(7 * 15 * PPM) + 2 * margin, int(5 * 15 * PPM) + 2 * margin),
                              marginSize=margin, borderBits=1)
    rng = np.random.default_rng(5)
    views, z_truth = [], []
    for k in range(14):
        rx, ry, rz = rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45), rng.uniform(-0.3, 0.3)
        rvec = np.array([rx, ry, rz])
        dist = rng.uniform(140, 165)
        # centre the board in view: translation = (0,0,dist) minus rotated board centre
        centre = np.array([52.5, 37.5, 0.0])
        R, _ = cv2.Rodrigues(rvec)
        tvec = np.array([rng.uniform(-8, 8), rng.uniform(-8, 8), dist]) - R @ centre
        img = render_view(board, rvec, tvec, tex, margin)
        views.append(img)
        z_truth.append(float((R @ centre + tvec)[2]))
    res = cc.calibrate_gray(views, board)
    assert res["views_used"] >= 10
    assert res["rms_reprojection_px"] < 1.0
    truth_mm_px = float(np.median(z_truth) / F)
    assert res["mm_per_px"] == pytest.approx(truth_mm_px, rel=0.03)
    K = np.array(res["camera_matrix"])
    assert K[0, 0] == pytest.approx(F, rel=0.03)


def test_undistort_roundtrip(tmp_path):
    import json
    calib = {"image_size": [W, H], "camera_matrix": K_TRUE.tolist(), "distortion": D_TRUE.tolist()}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(calib))
    c = imaging.load_calibration(p)
    img = textured(H, W)
    out = imaging.undistort(img, c)
    assert out.shape == img.shape
    assert np.array_equal(out[H // 2 - 5:H // 2 + 5, W // 2 - 5:W // 2 + 5].shape, (10, 10, 3))
    # centre pixel barely moves under mild radial distortion
    assert np.abs(out[H // 2, W // 2].astype(int) - img[H // 2, W // 2].astype(int)).max() < 100


def test_board_pdf_written(tmp_path):
    cc.write_board_pdf(tmp_path / "b.pdf", dpi=200)
    assert (tmp_path / "b.pdf").stat().st_size > 1000
