"""The FPIC converter's pure parts: designator -> part kind, the by-board split, tiling and box clipping."""
import numpy as np

from training.prepare_components import KIND_NAMES, boxes_in_tile, kind_of, read_outlines, split_boards, tiles


def test_designators_map_to_part_kinds_and_blank_means_unnamed():
    assert kind_of("C12") == "capacitor" and kind_of(" r3 ") == "resistor" and kind_of("U1") == "ic"
    assert kind_of("RN4") == "resistor" and kind_of("LED2") == "diode" and kind_of("CN1") == "connector"
    assert kind_of("XYZ9") == "other"                       # a designator we do not know is still a named part
    assert kind_of("") is None and kind_of("  ") is None and kind_of("12") is None
    assert "other" in KIND_NAMES and len(set(KIND_NAMES)) == len(KIND_NAMES)


def test_boards_are_split_whole_and_the_split_is_repeatable():
    ids = [str(i) for i in range(50)] + ["3", "3", "7"]       # repeated ids: front and back of the same board
    s = split_boards(ids, seed=0)
    assert s == split_boards(list(reversed(ids)), seed=0)    # order of the listing does not matter
    assert set(s) == {str(i) for i in range(50)} and set(s.values()) == {"train", "val", "test"}
    n = {k: list(s.values()).count(k) for k in ("train", "val", "test")}
    assert n == {"test": 10, "val": 5, "train": 35}


def test_tiles_cover_the_whole_photo():
    for h, w in ((640, 640), (700, 1500), (2000, 1234)):
        seen = np.zeros((h, w), bool)
        for x, y in tiles(h, w):
            assert 0 <= x <= w - 640 and 0 <= y <= h - 640
            seen[y:y + 640, x:x + 640] = True
        assert seen.all()


def test_a_part_cut_by_the_tile_edge_is_kept_only_if_most_of_it_is_inside():
    boxes = np.array([[600, 100, 700, 140],      # 40% inside the first tile, 60% inside the tile starting at 512
                      [100, 100, 140, 120],      # fully inside
                      [900, 900, 950, 950]], np.float32)
    a = boxes_in_tile(boxes, 0, 0, 640)
    assert a.tolist() == [[100, 100, 140, 120]]
    b = boxes_in_tile(boxes, 512, 0, 640)
    assert b.tolist() == [[88, 100, 188, 140]]
    assert boxes_in_tile(np.zeros((0, 4), np.float32), 0, 0, 640).shape == (0, 4)


def test_outlines_are_read_as_boxes_with_their_kind(tmp_path):
    p = tmp_path / "pcb_1f_cc_3_smd.csv"
    p.write_text('Instance ID,Image File,Vertices,Validated,Designator,Notes\n'
                 '0,pcb_1f_cc_3.png,"[[[10, 20], [30, 20], [30, 45], [10, 45]]]",True,C7,\n'
                 '1,pcb_1f_cc_3.png,"[[[100, 100], [120, 100], [120, 110]]]",True, ,\n'
                 '2,pcb_1f_cc_3.png,"[[[5, 5], [6, 6]]]",True,R1,\n')
    o = read_outlines(p)
    assert len(o) == 2                                       # two points are not an outline
    assert o[0] == {"image": "pcb_1f_cc_3.png", "box": [10, 20, 30, 45], "kind": "capacitor"}
    assert o[1]["kind"] is None and o[1]["box"] == [100, 100, 120, 110]
