import pytest

from pcbinspect.record import Defect
from pcbinspect.severity import DesignRules, SeverityEngine


def d(cls, width_mm=None, area_mm2=None, conf=0.9):
    return Defect(id="D1", cls=cls, source="detector", bbox=(0, 0, 10, 10), confidence=conf,
                  width_mm=width_mm, area_mm2=area_mm2)


@pytest.fixture(scope="module")
def eng():
    return SeverityEngine()


@pytest.mark.parametrize("cls", ["short", "open", "missing_hole"])
def test_functional_defects_reject_everywhere(eng, cls):
    x = eng.grade(d(cls))
    assert x.severity == "critical"
    assert x.ipc == {"class1": "reject", "class2": "reject", "class3": "reject"}


def test_mouse_bite_width_reduction_thresholds(eng):
    rules = DesignRules(min_trace_width_mm=0.20, min_spacing_mm=0.20)
    # 0.015 mm on a 0.20 mm trace = 7.5% -> accept at every class
    assert eng.grade(d("mouse_bite", 0.015), rules).ipc == {
        "class1": "accept", "class2": "accept", "class3": "accept"}
    # 0.03 mm = 15% -> fails Class 3 only -> minor
    x = eng.grade(d("mouse_bite", 0.03), rules)
    assert x.ipc["class3"] == "reject" and x.ipc["class2"] == "accept" and x.severity == "minor"
    # 0.05 mm = 25% -> fails Class 2 and 3 -> major
    x = eng.grade(d("mouse_bite", 0.05), rules)
    assert x.ipc == {"class1": "accept", "class2": "reject", "class3": "reject"}
    assert x.severity == "major"
    # 0.08 mm = 40% -> fails everything -> critical
    assert eng.grade(d("mouse_bite", 0.08), rules).severity == "critical"
    assert "25%" in eng.grade(d("mouse_bite", 0.05), rules).severity_reason


def test_spur_uses_spacing_rule(eng):
    rules = DesignRules(min_trace_width_mm=1.0, min_spacing_mm=0.10)
    # 0.025 mm against 0.10 mm spacing = 25% -> major (width rule would have said 2.5% -> accept)
    assert eng.grade(d("spur", 0.025), rules).severity == "major"


def test_no_scale_falls_back_to_class_default(eng):
    x = eng.grade(d("mouse_bite", None))
    assert x.severity == "major" and "no mm scale" in x.severity_reason
    assert x.ipc == {"class1": "accept", "class2": "reject", "class3": "reject"}


def test_anomaly_area_and_score(eng):
    assert eng.grade(d("unknown_anomaly", area_mm2=0.2)).severity == "minor"
    assert eng.grade(d("unknown_anomaly", area_mm2=2.0)).severity == "major"
    assert eng.grade(d("unknown_anomaly", area_mm2=9.0)).severity == "critical"
    assert eng.grade(d("unknown_anomaly", conf=0.95)).severity == "critical"
    assert eng.grade(d("unknown_anomaly", conf=0.3)).severity == "minor"


def test_unknown_relabel_is_conservative(eng):
    assert eng.grade(d("solder_bridge_custom")).severity == "critical"
