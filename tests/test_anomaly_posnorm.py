"""Foreground mask, per-position normalisation and top-k scoring of the anomaly memory, on synthetic
64x64 'boards' with a deterministic fake feature map (8x8 average pool of the normalised pixels,
so one feature-grid cell = one 8x8 pixel block)."""
import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pcbinspect.anomaly import (AnomalyModel, NearestNeighbour, enrol, enrol_from_desc,  # noqa: E402
                                descriptors, foreground_mask, grey_thumb, preprocess)

S = 64
G = S // 8


def fake_fn(x):
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 8, 8, w // 8, 8).mean((3, 5)).astype(np.float32)


def _to_img(a):
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def noisy_board(seed, defect=None, b_offset=None):
    """Every cell has its own G level (position code). Cell (1,1) is naturally noisy: its B level
    differs a lot from board to board. defect=(row, col, delta_R) adds a red shift to one cell."""
    rng = np.random.default_rng(seed)
    img = np.zeros((S, S, 3), np.float32)                 # BGR
    for r in range(G):
        for c in range(G):
            img[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] = (100, 30 + 3 * (r * G + c), 100)
    img += rng.normal(0, 1.0, img.shape)
    b = rng.uniform(-40, 40) if b_offset is None else b_offset
    img[8:16, 8:16, 0] += b
    if defect:
        r, c, dr = defect
        img[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8, 2] += dr
    return _to_img(img)


def fg_board(seed, defect_cell=None, delta=90):
    """Striped board over cells 1..6 on a flat dark fixture; optional bright blob in one cell."""
    rng = np.random.default_rng(seed)
    img = np.full((S, S, 3), 20, np.float32)
    stripes = (np.arange(8, 56) // 3 % 2) * 40.0
    img[8:56, 8:56] = 110 + stripes[None, :, None]
    img += rng.normal(0, 1.0, img.shape)
    if defect_cell:
        r, c = defect_cell
        img[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] += delta
    return _to_img(img)


def score(m, img):
    return m.score(img).raw_score


# ------------------------------------------------------------------ resolution
@pytest.mark.parametrize("size", [384, 448])
def test_high_input_sizes_end_to_end(size):
    im = np.random.default_rng(0).integers(0, 255, (300, 500, 3), dtype=np.uint8)
    x = preprocess(im, size)
    assert x.shape == (1, 3, size, size)
    m = enrol(fake_fn, [im, im[::-1].copy()], "hi", coreset_ratio=0.5, input_size=size, posnorm=True)
    res = m.score(im)
    assert res.heatmap.shape == (size // 8, size // 8) and m.grid == (size // 8, size // 8)
    assert np.isfinite(res.raw_score)


# ------------------------------------------------------------------ posnorm
def test_posnorm_stops_noisy_position_dominating():
    train = [noisy_board(s) for s in range(10)]
    plain = enrol(fake_fn, train, "p", coreset_ratio=1.0, input_size=S)
    pn = enrol(fake_fn, train, "p", coreset_ratio=1.0, input_size=S, posnorm=True)
    noisy_good = noisy_board(100, b_offset=65)            # good, but extreme at the noisy cell
    defect = noisy_board(101, b_offset=0, defect=(5, 5, 12))   # real shift at a quiet cell
    # without normalisation the natural variation at the noisy cell wins ...
    assert score(plain, noisy_good) > score(plain, defect)
    # ... with it, the quiet-position anomaly ranks highest, and is located at its own cell
    assert score(pn, defect) > score(pn, noisy_good)
    hm = pn.score(defect).heatmap
    assert np.unravel_index(hm.argmax(), hm.shape) == (5, 5)


def test_posnorm_sd_floor_and_stats_shape():
    train = [noisy_board(s) for s in range(7)]
    m = enrol(fake_fn, train, "p", coreset_ratio=1.0, input_size=S, posnorm=True)
    assert m.pos_mu.shape == (G, G) == m.pos_sd.shape
    assert m.pos_sd.min() > 0 and np.isfinite(m.pos_mu).all()
    assert m.pos_sd[1, 1] > np.median(m.pos_sd)           # the noisy cell has the largest spread


def test_posnorm_refuses_other_grid():
    m = enrol(fake_fn, [noisy_board(s) for s in range(6)], "p", coreset_ratio=1.0, input_size=S, posnorm=True)
    with pytest.raises(ValueError, match="grid"):
        m.score_desc(np.zeros((16 * 16, 3), np.float32), 16, 16, 128, 128)


def test_posnorm_needs_two_images():
    f = descriptors(fake_fn(preprocess(noisy_board(0), S)))
    with pytest.raises(ValueError):
        enrol_from_desc(fake_fn, [f], "x", posnorm=True, grid=(G, G))


# ------------------------------------------------------------------ foreground mask
def test_foreground_mask_geometry_and_guard():
    thumbs = [grey_thumb(fg_board(s)) for s in range(6)]
    g = foreground_mask(thumbs, (G, G))
    assert g is not None and g[1:7, 1:7].all() and not g[0].any() and not g[:, 7].any()
    flat = [np.full((128, 128), 90, np.uint8)] * 3
    assert foreground_mask(flat, (G, G)) is None                  # nothing to segment
    assert foreground_mask(thumbs, (G, G), min_cov=0.9) is None   # coverage guard (too small)
    assert foreground_mask(thumbs, (G, G), max_cov=0.3) is None   # coverage guard (too large)


def test_foreground_mask_removes_background_only_anomaly():
    train = [fg_board(s) for s in range(6)]
    plain = enrol(fake_fn, train, "f", coreset_ratio=1.0, input_size=S)
    fg = enrol(fake_fn, train, "f", coreset_ratio=1.0, input_size=S, fg_mask=True)
    assert fg.fg_status == "on" and fg.fg_mask.shape == (G, G)
    bg_defect = fg_board(50, defect_cell=(0, 0))                  # blob on the bare fixture
    board_defect = fg_board(51, defect_cell=(3, 3))               # same blob on the board
    good = fg_board(52)
    assert score(plain, bg_defect) > 5 * score(plain, good)       # unmasked model is fooled
    assert score(fg, bg_defect) < 2 * score(fg, good)             # masked model ignores it
    assert score(fg, board_defect) > 3 * score(fg, bg_defect)
    assert fg.score(bg_defect).heatmap[0, 0] == 0.0


# ------------------------------------------------------------------ top-k
def test_topk_is_less_sensitive_to_one_spike():
    train = [fg_board(s) for s in range(6)]
    mx = enrol(fake_fn, train, "t", coreset_ratio=1.0, input_size=S)
    tk = enrol(fake_fn, train, "t", coreset_ratio=1.0, input_size=S, topk_frac=0.1)
    clean, spike = fg_board(60), fg_board(61, defect_cell=(3, 3), delta=150)
    r_max = score(mx, spike) / score(mx, clean)
    r_tk = score(tk, spike) / score(tk, clean)
    assert r_tk < 0.6 * r_max
    d = mx.score(spike).heatmap
    k = round(0.1 * d.size)
    assert score(tk, spike) == pytest.approx(np.sort(d.ravel())[-k:].mean(), rel=1e-5)


# ------------------------------------------------------------------ persistence / compatibility
def test_save_load_roundtrip_new_fields(tmp_path):
    train = [fg_board(s) for s in range(6)]
    m = enrol(fake_fn, train, "r", coreset_ratio=1.0, input_size=S, fg_mask=True, posnorm=True, topk_frac=0.05)
    m.save(tmp_path / "m")
    m2 = AnomalyModel.load(tmp_path / "m", fake_fn)
    assert np.array_equal(m.fg_mask, m2.fg_mask)
    assert np.allclose(m.pos_mu, m2.pos_mu) and np.allclose(m.pos_sd, m2.pos_sd)
    assert m2.topk_frac == 0.05 and m2.grid == (G, G) and m2.fg_status == "on"
    img = fg_board(70, defect_cell=(3, 3))
    assert score(m2, img) == pytest.approx(score(m, img), rel=1e-6)
    with pytest.raises(ValueError, match="grid"):
        m2.score_desc(np.zeros((16 * 16, 3), np.float32), 16, 16, 128, 128)


def test_old_format_model_loads_with_features_off(tmp_path):
    train = [fg_board(s) for s in range(6)]
    m = enrol(fake_fn, train, "o", coreset_ratio=1.0, input_size=S)
    m.save(tmp_path / "m")
    meta = json.loads((tmp_path / "m" / "meta.json").read_text())
    for k in ("grid", "topk_frac", "fg_status", "has_fg_mask", "has_posnorm"):
        meta.pop(k)                                   # what a pre-change model's meta looks like
    (tmp_path / "m" / "meta.json").write_text(json.dumps(meta))
    old = AnomalyModel.load(tmp_path / "m", fake_fn)
    assert old.fg_mask is None and old.pos_mu is None and old.topk_frac == 0.0 and old.grid is None
    img = fg_board(80, defect_cell=(3, 3))
    assert score(old, img) == pytest.approx(score(m, img), rel=1e-6)


def test_defaults_reproduce_original_scores():
    train = [fg_board(s) for s in range(6)]
    a = enrol(fake_fn, train, "d", coreset_ratio=1.0, input_size=S)
    b = enrol(fake_fn, train, "d", coreset_ratio=1.0, input_size=S, fg_mask=False, posnorm=False, topk_frac=0.0)
    img = fg_board(90, defect_cell=(2, 5))
    res = a.score(img)
    # the original definition: raw 1-NN distance, image score = max
    d = a.nn.dist(descriptors(fake_fn(preprocess(img, S)))).reshape(G, G)
    assert np.array_equal(res.heatmap, d) and res.raw_score == float(d.max())
    assert score(b, img) == res.raw_score and a.norm_scale == b.norm_scale and a.pixel_thr == b.pixel_thr
    # the original leave-one-out calibration, recomputed independently
    feats = [descriptors(fake_fn(preprocess(t, S))) for t in train]
    loo = [float(NearestNeighbour(np.concatenate([f for j, f in enumerate(feats) if j != i]))
                 .dist(feats[i]).max()) for i in range(len(feats))]
    assert a.norm_scale == pytest.approx(float(np.median(loo)), rel=1e-6)
