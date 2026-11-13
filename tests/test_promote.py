import numpy as np

from pcbinspect.learning import FeedbackBuffer, rescore
from pcbinspect.promote import cross_validate, try_promote
from pcbinspect.store import Store


class FakeInspector:
    def __init__(self):
        self.heads = {}
        self.gates = {}
        self.designs = {}


def buffer(separable: bool, n=30, boards=10, seed=0):
    """Operator labels spread over `boards` boards. The detector confidence is uninformative
    (random), so only a head that learns the embedding can beat it."""
    rng = np.random.default_rng(seed)
    real, fa = rng.normal(size=32), rng.normal(size=32)
    buf = FeedbackBuffer()
    for i in range(n):
        for label, centre in ((1, real), (0, fa)):
            emb = (centre + 0.3 * rng.normal(size=32)) if separable else rng.normal(size=32)
            buf.add(emb.astype(np.float32), label, conf=float(rng.uniform(0.3, 0.9)), group=i % boards)
    return buf


def test_cross_validation_is_grouped_and_compares_with_the_detector():
    cv = cross_validate(buffer(True))
    assert cv["ok"] and cv["boards"] == 10
    assert cv["auc_with_head"] > cv["auc_detector"] + 0.2
    assert cv["recall_on_confirmed"] >= 0.95


def test_promotes_a_clearly_better_head(tmp_path):
    insp, store = FakeInspector(), Store(tmp_path / "var")
    rep = try_promote(insp, "d", buffer(True), tmp_path, store)
    assert rep["promoted"], rep
    assert (tmp_path / "models" / "heads" / "d.json").exists()
    assert [v["active"] for v in store.model_versions("rescore_head:d")] == [1]


def test_refuses_a_head_that_cannot_beat_the_detector(tmp_path):
    insp, store = FakeInspector(), Store(tmp_path / "var")
    rep = try_promote(insp, "d", buffer(False), tmp_path, store)
    assert not rep["promoted"] and "d" not in insp.heads
    v = store.model_versions("rescore_head:d")
    assert len(v) == 1 and v[0]["active"] == 0 and v[0]["note"]


def test_labels_from_too_few_boards_are_refused(tmp_path):
    rep = try_promote(FakeInspector(), "d", buffer(True, boards=3), tmp_path, None)
    assert not rep["promoted"] and "boards" in rep["reason"]


def test_head_can_never_hide_a_critical_defect():
    assert rescore("short", 0.8, 0.01) == 0.8           # 'false alarm' head cannot lower a short
    assert rescore("short", 0.4, 0.99) > 0.9             # but can raise it
    assert rescore("mouse_bite", 0.8, 0.01) < 0.1        # non-critical types can be lowered
