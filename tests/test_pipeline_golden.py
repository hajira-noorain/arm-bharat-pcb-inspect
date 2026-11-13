import cv2
import numpy as np

from pcbinspect.gate import ConformalGate
from pcbinspect.pipeline import Design, Inspector
from pcbinspect.severity import DesignRules
from tests.test_golden_diff import board, recapture


class NoDetections:
    """Detector stand-in that finds nothing: whatever is reported must come from the golden channel."""

    class tiles:
        name = "none"

    def detect(self, bgr):
        return [], {"detector_tiles": 0.0, "n_tiles": 1}


def make(tmp_path):
    (tmp_path / "models" / "gate").mkdir(parents=True)
    rng = np.random.default_rng(0)
    ConformalGate(rng.uniform(0.8, 1.0, 200), rng.uniform(0.0, 0.3, 200), "t").save(
        tmp_path / "models" / "gate" / "t.json")
    ConformalGate(rng.uniform(0.85, 1.0, 200), rng.uniform(0.0, 0.3, 200), "t_golden").save(
        tmp_path / "models" / "gate" / "t_golden.json")
    d = Design("t", "bare", True, None, 0.02, 20.0, DesignRules(0.4, 0.12, 2, True), "models/gate/t.json",
               gate_golden="models/gate/t_golden.json")
    return Inspector(tmp_path, {"t": d}, NoDetections(), None)


def test_golden_channel_reports_a_short_the_detector_missed(tmp_path):
    ins = make(tmp_path)
    g = board()
    t = g.copy()
    cv2.line(t, (300, 90), (300, 96), (20, 20, 20), 3)  # hairline bridge between two traces
    insp, a, _ = ins.inspect(recapture(t, np.random.default_rng(1), noise=2), "t", log=False, golden=g)
    assert a.golden_used and insp.model_versions["evidence"] == "detector+golden"
    assert insp.verdict == "FAIL"
    shorts = [d for d in insp.defects if d.cls == "short" and d.source == "golden"]
    assert shorts and shorts[0].bbox[0] <= 302 and shorts[0].bbox[2] >= 298


def test_without_golden_the_detector_only_gate_is_used(tmp_path):
    ins = make(tmp_path)
    g = board()
    insp, a, _ = ins.inspect(recapture(g, np.random.default_rng(2), noise=2), "t", log=False)
    assert not a.golden_used and insp.model_versions["evidence"] == "detector"
    assert insp.verdict == "PASS" and not insp.defects


def test_good_board_with_golden_passes(tmp_path):
    ins = make(tmp_path)
    g = board()
    insp, a, _ = ins.inspect(recapture(g, np.random.default_rng(3), noise=4), "t", log=False, golden=g)
    assert a.golden_used and insp.verdict == "PASS"


def test_refused_alignment_falls_back_and_says_why(tmp_path):
    ins = make(tmp_path)
    g = board()
    other = np.full_like(g, 235)
    cv2.circle(other, (320, 320), 200, (20, 20, 20), -1)
    insp, a, _ = ins.inspect(other, "t", log=False, golden=g)
    assert not a.golden_used and "alignment" in insp.model_versions["evidence"]


def test_confirmed_customer_rule_sets_disposition_and_says_which_rule(tmp_path):
    from pcbinspect.spec import Rule
    ins = make(tmp_path)
    ins.customer_rules["t"] = [(7, Rule(defect="short", measure="always", action="review"))]
    g = board()
    t = g.copy()
    cv2.line(t, (300, 90), (300, 96), (20, 20, 20), 3)
    insp, _, _ = ins.inspect(recapture(t, np.random.default_rng(1), noise=2), "t", log=False, golden=g)
    short = [d for d in insp.defects if d.cls == "short"][0]
    assert short.disposition == "REVIEW" and "customer rule #7" in short.severity_reason


def test_store_keeps_eyetest_and_rules(tmp_path):
    from pcbinspect.store import Store
    st = Store(tmp_path / "var")
    st.log_eyetest("t", {"types": {"open": {"limit_mm": 0.2, "n": 180}, "short": {"limit_mm": None, "n": 180}}})
    st.log_eyetest("t", {"types": {"open": {"limit_mm": 0.25, "n": 180}}})
    assert st.detection_limits("t") == {"open": 0.25}
    rid = st.add_rule("t", "No shorts.", {"kind": "limit", "defect": "short"}, "llama", {"status": "verified"},
                      {"status": "limited"})
    assert st.rules("t")[0]["rule"]["defect"] == "short"
    st.deactivate_rule(rid)
    assert st.rules("t") == [] and len(st.rules("t", active_only=False)) == 1


def test_untrusted_capture_never_auto_passes(tmp_path):
    ins = make(tmp_path)
    g = board()
    insp, _, _ = ins.inspect(recapture(g, np.random.default_rng(3), noise=4), "t", log=False, golden=g,
                             force_review="capture outside the calibrated envelope: blurred",
                             capture_info={"sharpness": 12.0})
    assert insp.verdict == "REVIEW" and "blurred" in insp.model_versions["capture_check"]
    assert "sharpness" in insp.model_versions["capture"]
