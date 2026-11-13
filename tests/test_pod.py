import numpy as np
import pytest

from pcbinspect.pod import _L90, clopper_pearson_lower, firth_logistic, fit_pod, required_trials

SIZES = np.array([0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7])


def _simulate(rng, b0, b1, per_size=30):
    s = np.repeat(SIZES, per_size)
    p = 1 / (1 + np.exp(-(b0 + b1 * np.log(s))))
    return s, (rng.random(s.size) < p).astype(int)


def test_29_of_29_rule():
    assert required_trials() == 29
    assert clopper_pearson_lower(29, 29) >= 0.90 > clopper_pearson_lower(28, 28)


def test_firth_close_to_mle_when_data_are_plentiful():
    rng = np.random.default_rng(1)
    s = np.repeat(SIZES, 400)
    b_true = np.array([8.0, 6.0])
    X = np.column_stack([np.ones_like(s), np.log(s)])
    y = (rng.random(s.size) < 1 / (1 + np.exp(-(X @ b_true)))).astype(float)
    b = firth_logistic(X, y)
    assert np.allclose(b, b_true, rtol=0.15)


def test_a90_95_upper_bound_has_nominal_coverage():
    """The one-sided 95% bound on a90 must lie above the true a90 in ~95% of repeated experiments."""
    rng = np.random.default_rng(7)
    b0, b1 = 8.0, 6.0
    true_a90 = float(np.exp((_L90 - b0) / b1))  # ~0.36 mm, inside the tested range
    covered, n_rep = 0, 150
    for _ in range(n_rep):
        s, y = _simulate(rng, b0, b1)
        r = fit_pod(s, y)
        covered += (r.a90_95 is not None) and r.a90_95 >= true_a90
    assert covered / n_rep >= 0.90  # nominal 0.95, allow Monte-Carlo slack


def test_point_estimate_tracks_truth():
    rng = np.random.default_rng(3)
    b0, b1 = 8.0, 6.0
    true_a90 = float(np.exp((_L90 - b0) / b1))
    est = [fit_pod(*_simulate(rng, b0, b1, per_size=60)).a90 for _ in range(40)]
    assert abs(np.median(est) - true_a90) / true_a90 < 0.08


def test_complete_separation_stays_finite_and_conservative():
    s = np.repeat(SIZES, 30)
    y = (s >= 0.3).astype(int)  # every size below 0.3 missed, every size from 0.3 up hit
    r = fit_pod(s, y)
    assert r.status == "ok"
    assert np.isfinite(r.a90) and 0.2 < r.a90 < 0.4
    assert r.demonstrated == pytest.approx(0.3)
    assert r.limit_mm >= 0.3  # never claims smaller than the data demonstrate


def test_all_hits_is_below_range_and_all_misses_not_reached():
    s = np.repeat(SIZES, 30)
    hi = fit_pod(s, np.ones_like(s, dtype=int))
    assert hi.status == "below_range" and hi.limit_mm == pytest.approx(0.15)
    lo = fit_pod(s, np.zeros_like(s, dtype=int))
    assert lo.status in ("not_reached", "no_trend") and lo.limit_mm is None
    assert "NOT" in lo.sentence() or "No usable" in lo.sentence()


def test_too_few_trials_is_not_demonstrated():
    # 10 trials at one size can never show 90/95 (needs 29/29). Across 7 sizes, 70/70 pooled over the
    # window (no size significantly low) demonstrates the whole tested range.
    s = np.repeat(SIZES, 10)
    r = fit_pod(s, np.ones_like(s, dtype=int))
    assert r.window == (0.15, 0.7) and r.limit_mm == pytest.approx(0.15)
    r2 = fit_pod(np.repeat([0.5], 10), np.ones(10, int))
    assert r2.window is None and r2.limit_mm is None


def test_large_defects_missed_gives_a_window_not_a_lower_limit():
    # reference-free detectors can miss LARGE defects that look like legitimate features
    s = np.repeat(SIZES, 30)
    y = np.zeros_like(s, dtype=int)
    y[(s >= 0.2) & (s <= 0.3)] = 1
    y[s == 0.15] = np.r_[np.ones(15), np.zeros(15)]
    for big in (0.4, 0.5, 0.7):
        y[s == big] = np.r_[np.ones(5), np.zeros(25)]
    r = fit_pod(s, y)
    assert r.status == "window" and r.upper_mm == pytest.approx(0.3) and r.large_defect_drop
    assert 0.2 <= r.limit_mm <= 0.3
    assert "LARGER" in r.sentence("shorts")
    assert r.demonstrated is None  # never claims "from L upward"


def test_misses_at_a_size_block_pooling_through_it():
    # 30/30 at large sizes but 20/30 at 0.3: 0.3 itself is not demonstrated; 0.4 is (30/30 alone)
    s = np.repeat(SIZES, 30)
    y = np.ones_like(s, dtype=int)
    y[(s == 0.3)] = np.r_[np.ones(20), np.zeros(10)]
    y[s < 0.3] = 0
    r = fit_pod(s, y)
    assert r.demonstrated == pytest.approx(0.4)
    assert r.limit_mm >= 0.4


def test_uncovered_defects_are_misses_even_with_a_degenerate_threshold():
    """With too few calibration boards lam is -inf (or 0): a defect nothing covered has conf 0.0 and must
    still be a miss, or the stated detection limit would be a perfect score the station never earned."""
    from pcbinspect.eyetest import summarise
    trials = [{"type": "open", "size_mm": s, "conf": 0.0, "covered": False, "cls": None, "coupon": i}
              for i, s in enumerate([0.2, 0.3, 0.4, 0.5, 0.7, 1.0] * 10)]
    for lam in (float("-inf"), 0.0):
        r = summarise(trials, lam)["types"]["open"]
        assert all(row["hits"] == 0 for row in r["table"]) and r["limit_mm"] is None


def test_card_reader_shows_the_detector_one_pattern_as_one_tile():
    """A coupon is a 640 px pattern: the detector must get exactly that (no padded crop cut into tiles),
    and its boxes must come back in the padded-crop frame the planted positions are checked in."""
    from types import SimpleNamespace

    from pcbinspect.eyetest import (COUPON_MM, FRAME_H, FRAME_W, RECT_PX_PER_MM, coupon_origin_mm, read_sheet)
    k = RECT_PX_PER_MM
    rect = np.full((int(FRAME_H * k) + 2, int(FRAME_W * k) + 2, 3), 240, np.uint8)
    idx = 4
    x0, y0 = coupon_origin_mm(idx)
    centre_mm = (x0 + COUPON_MM / 2, y0 + COUPON_MM / 3)
    seen = []

    class Det:
        def detect(self, img, only_near=None):
            seen.append(img.shape[:2])
            cx, cy = COUPON_MM / 2 * k, COUPON_MM / 3 * k          # the planted spot, in pattern pixels
            return [SimpleNamespace(bbox=(cx - 10, cy - 10, cx + 10, cy + 10), conf=0.9, cls="open")], {}

    sheet = {"coupons": [{"idx": idx, "defects": [{"type": "open", "size_mm": 0.4, "center_mm": list(centre_mm)}]}]}
    trials = read_sheet(rect, None, sheet, Det(), use_golden=False)
    assert seen == [(640, 640)]
    hit = [t for t in trials if t.get("type") == "open"][0]
    assert hit["covered"] and hit["conf"] == 0.9
    assert [t for t in trials if "false_calls" in t][0]["false_calls"] == []
