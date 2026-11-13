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
    assert r.verdict == "REVIEW" and "could not be lined up" in r.model_versions["capture_check"]   # never a pass
    assert not any(d.cls in ("missing_part", "extra_part") for d in r.defects)


def test_the_design_file_remembers_its_parts_map(tmp_path):
    from pcbinspect.pipeline import load_designs, save_design
    insp = make(tmp_path)
    save_design(tmp_path / "designs", insp.designs["asm"])
    assert load_designs(tmp_path / "designs")["asm"].parts_map == "models/anomaly/asm/parts.json"


class Flaky(BrightBoxes):
    """Sees every bright rectangle, except that on some pictures it misses the part near (560, 120) and on
    some it 'sees' a part at a board mark near (650, 500) that is not one."""

    def __init__(self):
        self.calls = 0

    def detect(self, bgr, only_near=None):
        self.calls += 1
        dets, lat = super().detect(bgr)
        if self.calls % 2 == 0:
            dets = [d for d in dets if not (540 <= d.bbox[0] <= 580 and 100 <= d.bbox[1] <= 140)]
            dets.append(SimpleNamespace(cls="part", bbox=(650, 500, 690, 530), conf=0.8))
        return dets, lat


def test_teaching_from_several_good_boards_leaves_off_what_is_not_seen_every_time(tmp_path):
    goods = [board(), board(shift=(5, 3)), board(shift=(-4, 6))]
    one = teach_parts(Flaky(), None, goods[0], mm_per_px=1 / 16)
    assert len(one.parts) == len(PARTS) and one.taught_from == 1 and one.ignore == []
    pm = teach_parts(Flaky(), None, goods, mm_per_px=1 / 16)
    assert pm.taught_from == 3 and pm.dropped == 1 and len(pm.parts) == len(PARTS) - 1
    assert not any(540 <= p.box[0] <= 580 and 100 <= p.box[1] <= 140 for p in pm.parts)   # the unreliable one
    assert len(pm.ignore) >= 1 and all(630 <= z[0] <= 670 for z in pm.ignore)            # the board mark
    s = pm.summary()
    assert s["good_pictures"] == 3 and s["left_off_as_unreliable"] == 1
    # the payoff: a good board on which the detector has one of its bad moments is still clean
    from pcbinspect.golden import align_warp
    from pcbinspect.parts import check_board, find_parts
    shot = board(shift=(6, -5))
    det = Flaky()
    det.calls = 1                                             # so this picture gets the bad moment
    dets, _ = find_parts(det, shot, scale=pm.scale)
    warp, cc = align_warp(shot, goods[0])
    assert cc > 0.6 and check_board(pm, dets, warp, (800, 600)) == []
    assert len(check_board(one, dets, warp, (800, 600))) == 2   # the single-picture map calls one missing, one extra


def test_the_scale_chosen_when_teaching_is_the_one_used_when_inspecting(tmp_path):
    insp = make(tmp_path)
    pm, ref = insp.parts["asm"]
    seen = []
    real = insp.parts_detector.detect
    insp.parts_detector.detect = lambda img, only_near=None: (seen.append(img.shape[:2]), real(img))[1]
    pm.scale = 0.5
    insp.inspect(board(), "asm", log=False)
    assert seen[-1] == (300, 400)


def test_a_part_the_detector_fails_to_see_is_not_called_missing_if_the_place_looks_like_the_good_board(tmp_path):
    from pcbinspect.golden import align_warp
    from pcbinspect.parts import check_board, find_parts
    good = board()
    pm = teach_parts(BrightBoxes(), None, good, mm_per_px=1 / 16)
    shot = board(shift=(6, -5))                               # a complete board
    dets, _ = find_parts(BrightBoxes(), shot, scale=pm.scale)
    blink = [d for d in dets if not (300 <= d["bbox"][0] <= 340)]          # the detector misses the part at x=320
    warp, _ = align_warp(shot, good)
    alone = check_board(pm, blink, warp, (800, 600))                       # detector only: a false "missing"
    assert [f.what for f in alone] == ["missing_part"]
    both = check_board(pm, blink, warp, (800, 600), ref=good, test=shot)   # ... but the place looks the same
    assert both == []
    gone = board(skip=(1,), shift=(6, -5))                                 # the part really is gone
    dets, _ = find_parts(BrightBoxes(), gone, scale=pm.scale)
    f = check_board(pm, dets, align_warp(gone, good)[0], (800, 600), ref=good, test=gone)
    assert [x.what for x in f] == ["missing_part"] and f[0].likeness is not None and f[0].likeness < 0.5


def swapped(shift=(0, 0)):
    """The good board with the part at (560, 120) replaced by something of the same size that looks different."""
    img = board()
    x, y, w, h = PARTS[4]
    cv2.rectangle(img, (x, y), (x + w, y + h), (25, 25, 30), -1)          # a dark body where the good one is light
    cv2.rectangle(img, (x + 15, y + 15), (x + w - 15, y + h - 15), (235, 235, 240), -1)   # with a bright label, so
    #                                                                     the stand-in detector still sees a part
    if shift != (0, 0):
        img = cv2.warpAffine(img, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (800, 600),
                             borderMode=cv2.BORDER_REPLICATE)
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_a_part_that_is_there_but_looks_unlike_the_good_one_is_reported_as_looking_different(tmp_path):
    insp = make(tmp_path)
    r, a, _ = insp.inspect(swapped(shift=(5, 4)), "asm", log=False)
    assert [d.cls for d in r.defects] == ["changed_part"] and r.verdict == "REVIEW"
    assert "1 part(s) look different" in r.model_versions["capture_check"]
    assert r.defects[0].severity == "major" and r.defects[0].location == "top-right"


def test_teaching_records_how_alike_each_part_looks_across_good_boards(tmp_path):
    goods = [board(), board(shift=(5, 3)), board(shift=(-4, 6))]
    pm = teach_parts(BrightBoxes(), None, goods, mm_per_px=1 / 16)
    assert len(pm.parts) == len(PARTS) and all(p.sim_min is not None and p.sim_min > 0.8 for p in pm.parts)
    from pcbinspect.parts import PartsMap, unlike_below
    assert all(0.3 <= unlike_below(p) <= 0.85 and unlike_below(p) <= p.sim_min for p in pm.parts)
    pm.save(tmp_path / "p.json")
    assert PartsMap.load(tmp_path / "p.json").parts[0].sim_min == pm.parts[0].sim_min
