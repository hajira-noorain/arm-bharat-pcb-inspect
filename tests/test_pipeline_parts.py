"""The parts check through the real inspection path: an assembled board with a taught parts map, a stand-in part
detector that 'sees' bright rectangles, and the station's own alignment."""
from types import SimpleNamespace

import cv2
import numpy as np

from pcbinspect.parts import teach_parts
from pcbinspect.pipeline import Design, Inspector
from pcbinspect.severity import DesignRules

PARTS = [(120, 100, 60, 30), (320, 100, 60, 30), (120, 300, 110, 110), (480, 380, 40, 70), (560, 120, 50, 50)]


def board(skip=(), extra=(), shift=(0, 0)):
    """A textured board with bright 'parts' on it; skip = indexes left off, extra = more rectangles."""
    rng = np.random.default_rng(1)
    img = np.full((600, 800, 3), (40, 110, 50), np.uint8)
    for _ in range(60):                                       # tracks and marks, so there is something to line up
        x, y = int(rng.integers(20, 760)), int(rng.integers(20, 560))
        cv2.line(img, (x, y), (x + int(rng.integers(-80, 80)), y + int(rng.integers(-80, 80))), (60, 150, 70), 3)
    for i, (x, y, w, h) in enumerate(list(PARTS) + list(extra)):
        if i in skip:
            continue
        cv2.rectangle(img, (x, y), (x + w, y + h), (210, 210, 215), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), 2)
    if shift != (0, 0):
        img = cv2.warpAffine(img, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (800, 600),
                             borderMode=cv2.BORDER_REPLICATE)
    return cv2.GaussianBlur(img, (3, 3), 0)


class BrightBoxes:
    """Stand-in for the part detector: every bright rectangle is a part."""

    def detect(self, bgr, only_near=None):
        m = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > 180).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(m)
        return [SimpleNamespace(cls="part", bbox=(int(x), int(y), int(x + w), int(y + h)), conf=0.9)
                for x, y, w, h, a in st[1:] if a > 200], {}


def make(tmp_path, kinds=None):
    good = board()
    d = tmp_path / "models" / "anomaly" / "asm"
    d.mkdir(parents=True)
    pm = teach_parts(BrightBoxes(), kinds, good)
    assert len(pm.parts) == len(PARTS)
    pm.save(d / "parts.json")
    cv2.imwrite(str(d / "parts_ref.jpg"), good, [cv2.IMWRITE_JPEG_QUALITY, 98])
    design = Design("asm", "assembled", False, None, None, 20.0, DesignRules(0.2, 0.2, 2, False), None,
                    parts_map="models/anomaly/asm/parts.json")
    insp = Inspector(tmp_path, {"asm": design}, None, None)
    insp.parts_detector = BrightBoxes()
    return insp


def test_a_complete_board_put_down_a_little_off_passes_the_parts_check(tmp_path):
    insp = make(tmp_path)
    r, a, _ = insp.inspect(board(shift=(9, -6)), "asm", log=False)
    assert a.parts_note == "" and a.parts_only == []
    assert r.verdict == "PASS" and r.defects == []
    assert "parts check" in r.model_versions["evidence"]


def test_a_missing_part_is_named_shown_and_sent_to_a_person(tmp_path):
    names = lambda crops: [("capacitor", 0.95)] * len(crops)           # noqa: E731
    insp = make(tmp_path, kinds=names)
    r, a, _ = insp.inspect(board(skip=(1,), shift=(7, 5)), "asm", log=False)
    assert r.verdict == "REVIEW"                 # uncalibrated evidence never fails a board by itself
    assert "1 part(s) look missing" in r.model_versions["capture_check"]
    assert len(r.defects) == 1
    d = r.defects[0]
    assert d.cls == "missing_part" and d.part == "capacitor" and d.source == "parts"
    assert d.severity == "critical" and d.location == "top-centre"
    x0, y0, x1, y1 = d.bbox                       # where the part should be, in THIS picture (shifted by 7, 5)
    assert abs(x0 - 327) <= 4 and abs(y0 - 105) <= 4 and abs(x1 - 387) <= 4
    rec = r.to_llm_record()["defects"][0]
    assert rec["type"] == "missing_part" and rec["part"] == "capacitor"


def test_something_extra_on_the_board_is_reported_as_an_unexpected_part(tmp_path):
    insp = make(tmp_path)
    r, a, _ = insp.inspect(board(extra=[(650, 450, 45, 45)]), "asm", log=False)
    assert [d.cls for d in r.defects] == ["extra_part"] and r.verdict == "REVIEW"
    assert "1 unexpected part(s)" in r.model_versions["capture_check"]
    assert r.defects[0].part == "part" and r.defects[0].severity == "major"


def test_a_board_that_cannot_be_lined_up_is_not_judged_for_parts(tmp_path):
    insp = make(tmp_path)
    other = np.random.default_rng(5).integers(0, 255, (600, 800, 3), dtype=np.uint8)
    r, a, _ = insp.inspect(other, "asm", log=False)
    assert a.parts_only == [] and "could not be lined up" in a.parts_note
    assert "parts check not run" in r.model_versions["evidence"]
    assert not any(d.cls in ("missing_part", "extra_part") for d in r.defects)


def test_the_design_file_remembers_its_parts_map(tmp_path):
    from pcbinspect.pipeline import load_designs, save_design
    insp = make(tmp_path)
    save_design(tmp_path / "designs", insp.designs["asm"])
    assert load_designs(tmp_path / "designs")["asm"].parts_map == "models/anomaly/asm/parts.json"
