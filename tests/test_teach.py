"""Teaching a bare board: planted defects on a photo, and a gate calibrated from them."""
import cv2
import numpy as np
import pytest

from pcbinspect.golden import copper_mask
from pcbinspect.pipeline import Inspector
from pcbinspect.teach import calibrate_bare, plant_on_photo, recapture


def photo(seed=0, size=1400):
    """A camera-like picture of a bare board: copper-coloured traces and pads on green laminate, uneven
    lighting, blur and noise (not a binarised scan)."""
    rng = np.random.default_rng(seed)
    cu = np.zeros((size, size), np.uint8)
    for y in range(60, size - 60, 70):
        cv2.line(cu, (50, y), (size - 50, y + int(rng.integers(-15, 15))), 255, int(rng.integers(8, 16)))
    for _ in range(60):
        c = (int(rng.integers(80, size - 80)), int(rng.integers(80, size - 80)))
        cv2.circle(cu, c, int(rng.integers(10, 22)), 255, -1)
    img = np.empty((size, size, 3), np.float32)
    img[:] = (60, 110, 40)
    img[cu > 0] = (70, 140, 200)
    yy, xx = np.mgrid[0:size, 0:size]
    img *= (0.85 + 0.25 * xx / size)[..., None]
    img = cv2.GaussianBlur(img, (0, 0), 1.1) + np.random.default_rng(seed + 1).normal(0, 2.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_planted_defects_change_only_their_own_spot_and_look_like_the_photo():
    p = photo()
    rng = np.random.default_rng(3)
    out, placed = plant_on_photo(p, [("open", 0.4), ("short", 0.6), ("pin_hole", 0.3), ("spur", 0.4)], 40.0, rng)
    assert len(placed) >= 3
    changed = np.abs(out.astype(int) - p.astype(int)).max(2) > 25
    boxes = np.zeros(changed.shape, bool)
    for d in placed:
        x1, y1, x2, y2 = d.bbox
        boxes[max(0, y1 - 12):y2 + 12, max(0, x1 - 12):x2 + 12] = True
    assert changed.any() and not (changed & ~boxes).any()
    # the copper mask really changed at each planted defect, and new pixels take the local colours
    before, after = copper_mask(p) > 0, copper_mask(out) > 0
    for d in placed:
        x1, y1, x2, y2 = d.bbox
        assert (before[y1:y2, x1:x2] != after[y1:y2, x1:x2]).sum() >= 4
    lo, hi = p.reshape(-1, 3).min(0).astype(int) - 25, p.reshape(-1, 3).max(0).astype(int) + 25
    assert (out.reshape(-1, 3) >= lo).all() and (out.reshape(-1, 3) <= hi).all()


def test_a_photo_without_a_copper_pattern_is_refused():
    blank = np.full((800, 800, 3), 90, np.uint8)
    out, placed = plant_on_photo(blank, [("open", 0.4)], 40.0, np.random.default_rng(0))
    assert placed == [] and np.array_equal(out, blank)


def test_taught_gate_fails_a_cut_trace_and_passes_good_boards(tmp_path):
    """End to end with no detector (golden comparison only): teach from 3 good captures, then a good
    re-capture must PASS and a board with a cut trace must FAIL; the stated limits come from the trials."""
    ins = Inspector(tmp_path, {}, None, None)
    rng = np.random.default_rng(0)
    base = photo()
    golden, goods = base, [recapture(base, rng), recapture(base, rng)]
    res = calibrate_bare(lambda t, g, al: ins.golden_evidence(t, g, aligned=al), golden, goods, 40.0, rng,
                         n_windows=40, n_good=20)
    assert res.n_windows == 40 and len(res.good_scores) == 20
    th = res.gate.thresholds(0.05, 0.05)
    assert np.isfinite(th.tau) and 0 < th.lam <= 0.9 and th.tau <= 0.9   # good captures must not look like shorts/opens
    assert "open" in res.limits and res.limits["open"]["sentence"]
    # judged with the calibrated gate
    good = recapture(base, np.random.default_rng(9))
    assert th.verdict(max((c for _, c in ins.golden_evidence(good, golden)), default=0.0)) == "PASS"
    cut = base.copy()
    cv2.line(cut, (700, 40), (700, 100), (45, 100, 60), 9)       # across the first trace
    score = max((c for _, c in ins.golden_evidence(recapture(cut, np.random.default_rng(10)), golden)), default=0.0)
    assert th.verdict(score) == "FAIL"


def test_teaching_needs_a_second_capture():
    with pytest.raises(ValueError):
        calibrate_bare(lambda *a: [], photo(), [], 40.0, np.random.default_rng(0))
