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


def test_poor_alignment_refuses_to_measure():
    golden = board()
    test = np.full((200, 300, 3), 255, np.uint8)
    test[30:40, :] = 0                                   # a completely different layout
    gm = GoldenMeasurer(test, golden, MM, 0.20)
    assert not gm.ok
    assert gm.measure("mouse_bite", (138, 86, 154, 98)) is None
    assert "alignment" in gm.last_reason


def test_one_pixel_sliver_is_not_a_defect():
    golden, test = board(), board()
    test[90:91, 140:160] = 255                            # 1-px edge difference (registration)
    gm = GoldenMeasurer(test, golden, MM, 0.20, aligned=True)
    assert gm.measure("mouse_bite", (138, 86, 162, 98)) is None
    assert "sliver" in gm.last_reason


def test_nick_next_to_a_pad_uses_the_trace_width():
    golden = board()
    golden[80:110, 40:70] = 0                             # a 30 px pad on trace A at x 40..69
    test = golden.copy()
    test[90:93, 76:88] = 255                              # 3 px nick on the 10 px trace, beside the pad
    m = GoldenMeasurer(test, golden, MM, 0.20, aligned=True).measure("mouse_bite", (74, 86, 90, 98))
    assert m is not None
    assert m.ref_mm == pytest.approx(10 * MM, abs=2.5 * MM), m   # not the 30 px pad
    assert m.pct > 20


def test_severity_uses_measurement():
    eng = SeverityEngine()
    golden, test = board(), board()
    test[90:93, 140:152] = 255
    meas = GoldenMeasurer(test, golden, MM, 0.20, aligned=True).measure("mouse_bite", (138, 86, 154, 98))
    d = eng.grade(Defect("D1", "mouse_bite", "detector", (138, 86, 154, 98), 0.9), DesignRules(0.2, 0.2),
                  measurement=meas)
    assert d.ipc["class1"] == "accept" and d.ipc["class3"] in ("reject", "uncertain")
    assert "nick" in d.severity_reason


def _majority_copper_pad():
    """A bare pad like the rain sensor: black board on white paper, copper covering most of the board
    (wide orange fingers, narrow black gaps)."""
    img = np.full((400, 500, 3), 235, np.uint8)                 # white paper
    img[60:340, 80:420] = (25, 30, 25)                          # black board
    for x in range(100, 400, 30):                               # 24 px copper fingers, 6 px gaps
        img[80:320, x:x + 24] = (110, 150, 215)                 # BGR copper orange
    return img


def test_colour_copper_takes_the_orange_phase_when_copper_is_the_majority():
    from pcbinspect.golden import copper_mask, copper_mode
    img = _majority_copper_pad()
    finger, gap, paper = (200, 112), (200, 126), (20, 20)       # (y, x)
    auto = copper_mask(img) > 0
    assert not auto[finger]                                     # the minority rule picks the gaps here
    with copper_mode("colour"):
        cu = copper_mask(img) > 0
    assert cu[finger] and not cu[gap] and not cu[paper]
    assert (copper_mask(img) > 0)[gap] == auto[gap]             # mode is restored after the block


def test_colour_mode_ignores_the_background_in_the_golden_diff():
    from pcbinspect.golden import copper_mode
    from pcbinspect.golden_diff import diff_candidates
    golden = _majority_copper_pad()
    test = golden.copy()
    test[5:40, 5:60] = (200, 200, 150)                          # something on the paper, off the board
    test[:, 440:] = (180, 200, 240)                             # and a warmer patch of paper (light falloff)
    test[150:170, 130:140] = (25, 30, 25)                       # a cut across one finger: an open
    with copper_mode("colour"):
        cands, info = diff_candidates(test, golden, aligned=True)
    assert info["ok"]
    centres = [((c.bbox[0] + c.bbox[2]) / 2, (c.bbox[1] + c.bbox[3]) / 2) for c in cands]
    assert any(120 <= x <= 150 and 140 <= y <= 180 for x, y in centres)
    assert not any(y < 60 or x > 425 for x, y in centres)


def test_colour_mode_drops_detector_boxes_outside_the_copper_pattern():
    """Colour-mode designs inspect the copper pattern only; detector boxes on the paper, the board edge or
    lettering are dropped like the comparison drops them. Other designs keep every box."""
    from types import SimpleNamespace as Box
    from pcbinspect.golden import copper_mode
    from pcbinspect.pipeline import _in_compared_area
    img = _majority_copper_pad()
    on_copper = Box(bbox=(190, 150, 230, 190))       # centre on the fingers
    on_paper = Box(bbox=(5, 5, 40, 40))              # white paper around the board
    on_edge = Box(bbox=(82, 322, 98, 338))           # black board margin below the fingers
    dets = [on_copper, on_paper, on_edge]
    with copper_mode("colour"):
        assert _in_compared_area(dets, img) == [on_copper]
        assert _in_compared_area(dets, None) == dets  # no aligned golden: nothing to judge against
    assert _in_compared_area(dets, img) == dets      # auto mode keeps every box
