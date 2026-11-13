"""Parts check: the map made from a good board, and what counts as missing or extra (no model needed)."""
import numpy as np

from pcbinspect.parts import Part, PartsMap, build_map, compare, dedupe, warp_boxes


def det(x, y, w=40, h=20, conf=0.9):
    return {"bbox": [x, y, x + w, y + h], "conf": conf}


GOOD = [det(100, 100), det(300, 100), det(100, 300, 80, 80), det(500, 400)]
KINDS = [("capacitor", 0.95), ("resistor", 0.9), ("ic", 0.99), ("diode", 0.4)]


def make_map():
    return build_map(GOOD, (800, 600), px_per_mm=10.0, kinds=KINDS)


def test_the_map_keeps_confident_parts_once_and_names_only_what_the_classifier_is_sure_of():
    doubtful, twice = det(600, 100, conf=0.3), det(102, 101, conf=0.7)
    m = build_map(GOOD + [doubtful, twice], (800, 600), 10.0, KINDS + [("ic", 0.9), ("ic", 0.9)])
    assert len(m.parts) == 4                                   # the doubtful one and the duplicate are left out
    assert sorted(p.label for p in m.parts) == ["capacitor", "ic", "part", "resistor"]   # diode at 0.4 -> "part"
    assert m.summary() == {"parts": 4, "by_kind": {"capacitor": 1, "resistor": 1, "ic": 1, "part": 1}}


def test_a_good_board_slightly_shifted_has_nothing_missing_and_nothing_extra():
    m = make_map()
    shifted = [{"bbox": [v + 6 for v in d["bbox"]], "conf": 0.8} for d in GOOD]
    assert compare(m, shifted) == []


def test_a_part_that_is_gone_is_reported_with_its_kind_place_and_size():
    m = make_map()
    f = compare(m, [GOOD[0], GOOD[2], GOOD[3]])               # the resistor is not found
    assert len(f) == 1 and f[0].what == "missing_part" and f[0].kind == "resistor"
    assert f[0].box == [300.0, 100.0, 340.0, 120.0] and f[0].size_mm == 4.0


def test_a_weak_detection_still_counts_as_the_part_being_there():
    m = make_map()
    weak = [GOOD[0], {**GOOD[1], "conf": 0.3}, GOOD[2], GOOD[3]]
    assert compare(m, weak) == []
    too_weak = [GOOD[0], {**GOOD[1], "conf": 0.1}, GOOD[2], GOOD[3]]
    assert [x.what for x in compare(m, too_weak)] == ["missing_part"]


def test_something_extra_needs_a_confident_detection_and_is_reported_once():
    m = make_map()
    blob = det(600, 250, conf=0.9)
    f = compare(m, GOOD + [blob, {**blob, "conf": 0.7}])       # found twice by overlapping tiles
    assert len(f) == 1 and f[0].what == "extra_part" and f[0].kind == "part"
    assert compare(m, GOOD + [det(600, 250, conf=0.4)]) == []  # not sure enough to call it extra


def test_parts_at_the_edge_of_what_the_picture_covers_are_not_judged():
    m = make_map()
    # the aligned picture only covers x >= 200: the capacitor and the chip at x = 100 cannot be judged
    f = compare(m, [GOOD[1], GOOD[3]], valid=(200, 0, 800, 600))
    assert f == []
    f = compare(m, [GOOD[3]], valid=(200, 0, 800, 600))
    assert [x.kind for x in f] == ["resistor"]


def test_the_map_survives_saving_and_loading(tmp_path):
    m = make_map()
    m.save(tmp_path / "parts.json")
    again = PartsMap.load(tmp_path / "parts.json")
    assert again.size == [800, 600] and again.px_per_mm == 10.0
    assert [p.box for p in again.parts] == [p.box for p in m.parts]
    assert isinstance(again.parts[0], Part) and compare(again, GOOD) == []


def test_boxes_follow_the_alignment_into_the_good_boards_picture():
    shift = np.array([[1, 0, 12], [0, 1, -5]], float)
    moved = warp_boxes([det(100, 100)], shift)
    assert moved[0]["bbox"] == [112.0, 95.0, 152.0, 115.0] and moved[0]["conf"] == 0.9
    quarter = np.array([[0, -1, 600], [1, 0, 0], [0, 0, 1]], float)        # a 90 degree turn
    b = warp_boxes([det(100, 100)], quarter)[0]["bbox"]
    assert b == [480.0, 100.0, 500.0, 140.0]                             # still upright, width and height swapped


def test_duplicates_from_overlapping_tiles_collapse_to_the_most_confident():
    d = dedupe([det(10, 10, conf=0.6), det(12, 11, conf=0.9), det(200, 200, conf=0.5)])
    assert [x["conf"] for x in d] == [0.9, 0.5]


def test_a_board_put_down_shifted_is_checked_where_its_parts_actually_are():
    from pcbinspect.parts import check_board
    m = make_map()
    # the board under inspection sits 15 px right and 8 px down of where the good board was photographed:
    # a point of the inspected picture maps to (x - 15, y - 8) in the good picture
    warp = np.array([[1, 0, -15], [0, 1, -8]], float)
    here = [{"bbox": [d["bbox"][0] + 15, d["bbox"][1] + 8, d["bbox"][2] + 15, d["bbox"][3] + 8], "conf": 0.9}
            for d in GOOD]
    assert check_board(m, here, warp, (800, 600)) == []
    f = check_board(m, here[:1] + here[2:], warp, (800, 600))
    assert len(f) == 1 and f[0].kind == "resistor" and f[0].box == [315.0, 108.0, 355.0, 128.0]   # in ITS pixels


def test_a_good_picture_of_another_size_is_scaled_before_comparing():
    from pcbinspect.parts import check_board
    m = make_map()                                             # taught at 800 x 600
    half = [{"bbox": [v / 2 for v in d["bbox"]], "conf": 0.9} for d in GOOD]      # inspected at 400 x 300
    assert check_board(m, half, np.array([[1, 0, 0], [0, 1, 0]], float), (400, 300)) == []


class FakeDetector:
    def __init__(self):
        self.shapes = []

    def detect(self, img):
        from types import SimpleNamespace
        self.shapes.append(img.shape[:2])
        s = img.shape[1] / 800
        return [SimpleNamespace(bbox=tuple(v * s for v in d["bbox"]), conf=d["conf"]) for d in GOOD], {}


def test_parts_are_found_at_the_detectors_training_scale_and_reported_in_the_pictures_own_pixels():
    from pcbinspect.parts import find_parts, teach_parts
    img = np.zeros((600, 800, 3), np.uint8)
    det = FakeDetector()
    found, ms = find_parts(det, img, mm_per_px=1 / 32)         # 32 px/mm here, the detector learned 16 px/mm
    assert det.shapes[-1] == (300, 400) and ms >= 0
    assert [round(v) for v in found[0]["bbox"]] in ([100, 100, 140, 120], [100, 300, 180, 380], [300, 100, 340, 120], [500, 400, 540, 420])
    find_parts(det, img)                                       # no scale known: the picture is used as it is
    assert det.shapes[-1] == (600, 800)
    m = teach_parts(det, lambda crops: [("capacitor", 0.9)] * len(crops), img)
    assert len(m.parts) == 4 and m.summary()["by_kind"] == {"capacitor": 4}
