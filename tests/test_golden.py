"""Golden-board measurement on synthetic copper with known geometry (0.02 mm/px)."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from pcbinspect.golden import GoldenMeasurer, copper_mask  # noqa: E402
from pcbinspect.record import Defect  # noqa: E402
from pcbinspect.severity import DesignRules, SeverityEngine  # noqa: E402

MM = 0.02  # 50 px/mm


def board():
    """Black copper on white (DeepPCB style): two horizontal traces 10 px wide, 12 px gap."""
    img = np.full((200, 300, 3), 255, np.uint8)
    img[90:100, 20:280] = 0     # trace A: y 90..99
    img[112:122, 20:280] = 0    # trace B: y 112..121  -> gap 100..111 = 12 px
    return img


def test_copper_mask_finds_the_traces():
    m = copper_mask(board()) > 0
    assert m[95, 150] and m[115, 150] and not m[105, 150] and not m[20, 20]


def test_mouse_bite_depth_and_width_reduction():
    golden, test = board(), board()
    test[90:93, 140:152] = 255                     # 3 px deep nick from the top of trace A
    m = GoldenMeasurer(test, golden, MM, 0.20, aligned=True).measure("mouse_bite", (138, 86, 154, 98))
    assert m.kind == "width"
    assert m.defect_mm == pytest.approx(3 * MM, abs=0.6 * MM)
    assert m.ref_mm == pytest.approx(10 * MM, abs=2 * MM)
    assert 20 <= m.pct <= 40                        # ~30%


def test_spur_remaining_spacing_against_other_conductor():
    golden, test = board(), board()
    test[100:104, 150:158] = 0                      # 4 px spur from trace A toward trace B
    m = GoldenMeasurer(test, golden, MM, 0.20, aligned=True).measure("spur", (146, 96, 162, 108))
    assert m.kind == "spacing"
    assert m.defect_mm == pytest.approx(8 * MM, abs=1.5 * MM)    # 12 px gap - 4 px spur
    assert m.pct == pytest.approx(100 * (0.20 - m.defect_mm) / 0.20, abs=0.5)


def test_bridge_and_full_cut_are_100_percent():
    golden, test = board(), board()
    test[100:112, 60:66] = 0                        # copper joining A and B
    m = GoldenMeasurer(test, golden, MM, 0.20, aligned=True).measure("spur", (56, 96, 70, 116))
    assert m.pct == 100.0 and "bridges" in m.text()
    test2 = board()
    test2[90:100, 200:206] = 255                    # trace A cut through
    m2 = GoldenMeasurer(test2, golden, MM, 0.20, aligned=True).measure("open", (196, 86, 210, 104))
    assert m2.pct == 100.0 and "cuts" in m2.text()


def test_alignment_recovers_a_shift():
    golden = board()
    test = np.roll(board(), (3, -5), axis=(0, 1))   # board placed 3 px down, 5 px left
    test[93:96, 135:147] = 255                       # nick, in shifted coordinates
    gm = GoldenMeasurer(test, golden, MM, 0.20)
    assert gm.align_cc > 0.9
    m = gm.measure("mouse_bite", (133, 89, 149, 101))
    assert m is not None and m.defect_mm == pytest.approx(3 * MM, abs=1.0 * MM)


def test_severity_uses_measurement():
    eng = SeverityEngine()
    golden, test = board(), board()
    test[90:93, 140:152] = 255
    meas = GoldenMeasurer(test, golden, MM, 0.20, aligned=True).measure("mouse_bite", (138, 86, 154, 98))
    d = eng.grade(Defect("D1", "mouse_bite", "detector", (138, 86, 154, 98), 0.9), DesignRules(0.2, 0.2),
                  measurement=meas)
    assert d.ipc["class3"] == "reject" and d.ipc["class1"] == "accept"
    assert "nick" in d.severity_reason
