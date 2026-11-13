"""Functional tests of the anomaly memory and on-device learning, using the real ResNet18 backbone
on synthetic 'boards' (regular trace/pad pattern) with a defect drawn at a known location."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from pcbinspect.anomaly import AnomalyModel, enrol, torch_feature_fn  # noqa: E402
from pcbinspect.learning import FeedbackBuffer, RescoreHead, combine, fit_head  # noqa: E402


def board(seed: int, defect: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((512, 512, 3), (40, 110, 40), np.uint8)            # green solder mask
    for y in range(40, 512, 64):                                        # horizontal traces
        cv2.line(img, (0, y), (511, y), (60, 170, 190), 10)
    for x in range(48, 512, 96):                                        # pads
        for y in range(72, 512, 128):
            cv2.circle(img, (x, y), 12, (180, 180, 190), -1)
    img = cv2.add(img, rng.integers(0, 12, img.shape, dtype=np.uint8))  # sensor noise
    if defect:
        cv2.rectangle(img, (300, 290), (340, 330), (20, 20, 20), -1)   # burn mark / foreign object
    return img


@pytest.fixture(scope="module")
def fn():
    return torch_feature_fn("cpu")


@pytest.fixture(scope="module")
def model(fn):
    return enrol(fn, [board(s) for s in range(6)], design="synthetic", coreset_ratio=0.25)


def test_defect_scores_higher_and_is_localised(model):
    good = [model.score(board(100 + s)) for s in range(3)]
    bad = model.score(board(200, defect=True))
    assert bad.raw_score > max(g.raw_score for g in good) * 1.2
    assert bad.regions, "the defect should produce at least one region"
    x1, y1, x2, y2 = bad.regions[0].bbox
    assert x1 <= 340 and x2 >= 300 and y1 <= 330 and y2 >= 290, bad.regions[0].bbox
    assert 0 <= bad.norm_score <= 1 and bad.norm_score > max(g.norm_score for g in good)


def test_false_alarm_novelty_gate_learning_and_undo(fn, model, tmp_path):
    m = AnomalyModel(fn, model.bank.copy(), model.norm_scale, model.pixel_thr, "synthetic")
    res = m.score(board(300, defect=True))
    region = res.regions[0]
    n0 = len(m.bank)
    m.novelty_factor = 1.0          # force the gate: this region is far beyond normal variation
    first = m.false_alarm(res, region)
    assert not first["accepted"] and first["reason"] == "novelty_gate"
    assert len(m.bank) == n0
    second = m.false_alarm(res, region, confirmed_twice=True)
    assert second["accepted"] and len(m.bank) == n0 + second["count"]
    # after learning, the same pattern is no longer anomalous
    after = m.score(board(300, defect=True))
    assert after.raw_score < res.raw_score * 0.5
    # undo restores the exact previous memory
    m.undo_last()
    assert len(m.bank) == n0
    assert m.score(board(300, defect=True)).raw_score == pytest.approx(res.raw_score, rel=1e-4)
    # persistence round-trip
    m.save(tmp_path / "bank")
    m2 = AnomalyModel.load(tmp_path / "bank", fn)
    assert len(m2.bank) == n0 and m2.pixel_thr == pytest.approx(m.pixel_thr)


def test_combine_is_bayesian_update():
    assert combine(0.5, 0.9) == pytest.approx(0.9)
    assert combine(0.8, 0.5) == pytest.approx(0.8)          # uninformative head changes nothing
    assert combine(0.8, 0.1) < 0.5                           # strong 'false alarm' evidence wins


def test_head_learns_from_operator_labels(tmp_path):
    rng = np.random.default_rng(0)
    buf = FeedbackBuffer(tmp_path / "fb.npz")
    centre_real, centre_fa = rng.normal(size=64), rng.normal(size=64)
    assert fit_head(buf, 1) is None                          # not enough labels yet
    for i in range(20):
        buf.add((centre_real + 0.3 * rng.normal(size=64)).astype(np.float32), 1, event_id=i)
        buf.add((centre_fa + 0.3 * rng.normal(size=64)).astype(np.float32), 0, event_id=100 + i)
    head = fit_head(buf, 1)
    assert head is not None and head.n_pos == 20 and head.n_neg == 20
    assert head.p_real(centre_real.astype(np.float32)) > 0.9
    assert head.p_real(centre_fa.astype(np.float32)) < 0.1
    # buffer persists and undo removes exactly one example
    buf2 = FeedbackBuffer(tmp_path / "fb.npz")
    assert buf2.counts() == (20, 20)
    assert buf2.remove_event(105) and buf2.counts() == (20, 19)
    head.save(tmp_path / "h.json")
    assert RescoreHead.load(tmp_path / "h.json").p_real(centre_real.astype(np.float32)) > 0.9


def test_new_options_default_off_and_old_meta_loads(fn, model, tmp_path):
    assert model.fg_mask is None and model.pos_mu is None and model.topk_frac == 0.0
    res = model.score(board(400, defect=True))
    d = model.nn.dist(res.grid_desc).reshape(res.heatmap.shape)
    assert res.raw_score == float(d.max()) and np.array_equal(res.heatmap, d)
    model.save(tmp_path / "m")
    assert not (tmp_path / "m" / "scoring_extras.npz").exists()
    again = AnomalyModel.load(tmp_path / "m", fn)
    assert again.score(board(400, defect=True)).raw_score == pytest.approx(res.raw_score, rel=1e-5)


def test_real_backbone_grid_at_384(fn):
    from pcbinspect.anomaly import preprocess
    assert fn(preprocess(board(0), 384)).shape[2:] == (48, 48)
