import pytest

from pcbinspect.golden import Measurement
from pcbinspect.record import Defect
from pcbinspect.severity import DesignRules, SeverityEngine


def d(cls, conf=0.9, area_mm2=None):
    return Defect(id="D1", cls=cls, source="detector", bbox=(0, 0, 10, 10), confidence=conf,
                  area_mm2=area_mm2)


def width(pct, ref=0.20, px=0.005):
    return Measurement("width", ref * pct / 100, ref, pct, px)


@pytest.fixture(scope="module")
def eng():
    return SeverityEngine()


@pytest.mark.parametrize("cls", ["short", "open", "missing_hole"])
def test_functional_defects_reject_everywhere(eng, cls):
    x = eng.grade(d(cls), DesignRules(0.2, 0.2, 2))
    assert x.ipc == {"class1": "reject", "class2": "reject", "class3": "reject"}
    assert x.severity == "critical"


def test_repair_policy_gives_dispositions(eng):
    assert eng.grade(d("short"), DesignRules(0.2, 0.2, 3)).disposition == "REWORK"
    assert eng.grade(d("open"), DesignRules(0.2, 0.2, 3)).disposition == "SCRAP"
    assert eng.grade(d("open"), DesignRules(0.2, 0.2, 2)).disposition == "REWORK"
    assert eng.grade(d("missing_hole"), DesignRules(0.2, 0.2, 1)).disposition == "SCRAP"


def test_measured_width_reduction_against_class_limits(eng):
    # 15% loss: accept at Class 2 (limit 20), reject at Class 3 (limit 10), Class 1 functional only
    x = eng.grade(d("mouse_bite"), DesignRules(0.2, 0.2, 2), measurement=width(15))
    assert x.ipc == {"class1": "accept", "class2": "accept", "class3": "reject"}
    assert x.disposition == "ACCEPT" and x.severity == "minor"
    x3 = eng.grade(d("mouse_bite"), DesignRules(0.2, 0.2, 3), measurement=width(15))
    assert x3.disposition == "SCRAP"          # conductor nick at Class 3: not repairable to class
    # 60% loss: rejected at Class 2 and 3 but not functional, so Class 1 accepts
    y = eng.grade(d("mouse_bite"), DesignRules(0.2, 0.2, 2), measurement=width(60))
    assert y.ipc["class1"] == "accept" and y.ipc["class2"] == "reject" and y.disposition == "REWORK"
    # cut through: rejected everywhere
    z = eng.grade(d("mouse_bite"), DesignRules(0.2, 0.2, 2), measurement=width(100))
    assert z.ipc["class1"] == "reject" and z.severity == "critical"


def test_limit_inside_one_pixel_is_uncertain_not_guessed(eng):
    # 1 px = 0.02 mm on a 0.2 mm trace = 10% uncertainty; 22% loss vs the 20% Class 2 limit
    m = Measurement("width", 0.044, 0.20, 22.0, 0.02)
    x = eng.grade(d("mouse_bite"), DesignRules(0.2, 0.2, 2), measurement=m)
    assert x.ipc["class2"] == "uncertain" and x.disposition == "REVIEW"
    assert x.uncertainty_mm == pytest.approx(0.02)
    assert "+/-0.020 mm" in x.severity_reason


def test_without_measurement_type_default_is_used_and_said(eng):
    x = eng.grade(d("mouse_bite"))
    assert "no golden-board measurement" in x.severity_reason
    assert x.ipc == {"class1": "accept", "class2": "reject", "class3": "reject"}


def test_anomaly_needs_a_person(eng):
    x = eng.grade(d("unknown_anomaly", area_mm2=9.0))
    assert x.severity == "critical" and x.disposition == "REVIEW"
    assert eng.grade(d("unknown_anomaly", conf=0.3)).severity == "minor"


def test_board_disposition_is_the_worst(eng):
    a = eng.grade(d("mouse_bite"), DesignRules(0.2, 0.2, 2), measurement=width(5))
    b = eng.grade(d("open"), DesignRules(0.2, 0.2, 3))
    assert SeverityEngine.board_disposition([a]) == "ACCEPT"
    assert SeverityEngine.board_disposition([a, b]) == "SCRAP"


def test_unknown_relabel_is_conservative(eng):
    x = eng.grade(d("solder_bridge_custom"))
    assert x.severity == "critical" and x.disposition == "REVIEW"
