import numpy as np
import pytest

from pcbinspect.learning import FeedbackBuffer
from pcbinspect.promote import cross_validate, try_promote
from pcbinspect.store import Store


class FakeInspector:
    def __init__(self):
        self.heads = {}
        self.gates = {}
        self.designs = {}


def separable(n=30, seed=0):
    rng = np.random.default_rng(seed)
    real, fa = rng.normal(size=32), rng.normal(size=32)
    buf = FeedbackBuffer()
    for _ in range(n):
        buf.add((real + 0.3 * rng.normal(size=32)).astype(np.float32), 1)
        buf.add((fa + 0.3 * rng.normal(size=32)).astype(np.float32), 0)
    return buf


def noise(n=30, seed=1):
    rng = np.random.default_rng(seed)
    buf = FeedbackBuffer()
    for i in range(2 * n):
        buf.add(rng.normal(size=32).astype(np.float32), i % 2)
    return buf


def test_cross_validation_on_learnable_feedback():
    cv = cross_validate(separable())
    assert cv["ok"] and cv["recall_on_confirmed"] >= 0.95 and cv["false_alarms_removed"] >= 0.9


def test_promotes_a_clearly_better_head(tmp_path):
    insp, store = FakeInspector(), Store(tmp_path / "var")
    rep = try_promote(insp, "d", separable(), tmp_path, store)
    assert rep["promoted"] and rep["version"] == 1
    assert (tmp_path / "models" / "heads" / "d.json").exists()
    assert insp.heads["d"].version == 1
    assert [v["active"] for v in store.model_versions("rescore_head:d")] == [1]


def test_refuses_a_head_that_cannot_beat_the_detector(tmp_path):
    insp, store = FakeInspector(), Store(tmp_path / "var")
    rep = try_promote(insp, "d", noise(), tmp_path, store)
    assert not rep["promoted"]
    assert "d" not in insp.heads
    assert not (tmp_path / "models" / "heads" / "d.json").exists()
    v = store.model_versions("rescore_head:d")
    assert len(v) == 1 and v[0]["active"] == 0 and v[0]["note"]


def test_too_few_labels_is_reported_not_crashed(tmp_path):
    buf = FeedbackBuffer()
    for i in range(6):
        buf.add(np.ones(8, np.float32) * i, i % 2)
    rep = try_promote(FakeInspector(), "d", buf, tmp_path, None)
    assert not rep["promoted"] and "labels" in rep["reason"]
