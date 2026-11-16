"""Evidence the station computes for each defect from its own log, for Llama's cause choice."""
from pcbinspect.evidence import board_evidence
from pcbinspect.record import Defect, Inspection
from pcbinspect.store import Store

MMPP = 0.05          # 20 px per mm


def _log(store, defects, design="d1", w=1000, h=800):
    ds = [Defect(id=f"D{i + 1}", cls=c, source="detector", bbox=b, confidence=0.9) for i, (c, b) in enumerate(defects)]
    insp = Inspection(board_id="B", design=design, image_path="", defects=ds, board_score=0.9, verdict="FAIL",
                      alpha=0.05, mm_per_px=MMPP, latency_ms={})
    return store.log_inspection(insp, None)


def test_a_defect_at_the_same_spot_on_earlier_boards_is_counted(tmp_path):
    s = Store(tmp_path)
    for _ in range(4):
        _log(s, [("open", (500, 400, 520, 410))])          # same place, same type
    _log(s, [("open", (100, 100, 120, 110))])              # same type, elsewhere
    _log(s, [("short", (500, 400, 520, 410))])             # same place, other type
    _log(s, [("open", (100, 700, 110, 705))], design="d2")  # other design
    iid = _log(s, [("open", (505, 402, 522, 411))])
    ev = board_evidence(s, iid, 1000, 800, MMPP)["D1"]
    assert ev["repeat_same_spot"] == {"boards": 4, "of": 6}


def test_no_history_is_unknown_never_zero(tmp_path):
    s = Store(tmp_path)
    iid = _log(s, [("open", (500, 400, 520, 410))])
    ev = board_evidence(s, iid, 1000, 800, MMPP)["D1"]
    assert ev["repeat_same_spot"] == "unknown" and ev["trend"] == "unknown"


def test_shape_edge_count_and_co_occurring(tmp_path):
    s = Store(tmp_path)
    iid = _log(s, [("open", (10, 300, 30, 400)),           # 20x100 px: long and thin, 0.5 mm from the left edge
                   ("open", (400, 400, 430, 425)),         # 30x25: round, centre
                   ("short", (600, 100, 660, 140))])       # 60x40: irregular
    ev = board_evidence(s, iid, 1000, 800, MMPP)
    assert ev["D1"]["shape"] == "long_thin" and ev["D1"]["near_edge"] is True
    assert ev["D2"]["shape"] == "round" and ev["D2"]["near_edge"] is False
    assert ev["D3"]["shape"] == "irregular"
    assert ev["D1"]["count_on_board"] == 2 and ev["D3"]["count_on_board"] == 1
    assert ev["D1"]["co_occurring"] == ["short"] and ev["D3"]["co_occurring"] == ["open"]


def test_near_edge_needs_a_scale_or_falls_back_to_a_share_of_the_frame(tmp_path):
    s = Store(tmp_path)
    iid = _log(s, [("open", (5, 300, 25, 320))])
    assert board_evidence(s, iid, 1000, 800, None)["D1"]["near_edge"] is True
    assert board_evidence(s, iid, None, None, None)["D1"]["near_edge"] == "unknown"


def test_trend_rising_when_the_type_appears_on_more_recent_boards(tmp_path):
    s = Store(tmp_path)
    for k in range(20):                                    # older 10: 1 board with a spur; newer 10: 6 boards
        has = (k == 3) or (k >= 10 and k % 10 < 6)
        _log(s, [("spur", (300, 300, 310, 310))] if has else [("open", (50, 50, 60, 60))])
    iid = _log(s, [("spur", (700, 200, 710, 210))])
    assert board_evidence(s, iid, 1000, 800, MMPP)["D1"]["trend"] == "rising"
    s2 = Store(tmp_path / "steady")
    for k in range(20):
        _log(s2, [("spur", (300, 300, 310, 310))] if k % 4 == 0 else [("open", (50, 50, 60, 60))])
    iid2 = _log(s2, [("spur", (700, 200, 710, 210))])
    assert board_evidence(s2, iid2, 1000, 800, MMPP)["D1"]["trend"] == "steady"


def test_the_llm_record_carries_each_defects_evidence(tmp_path):
    from pcbinspect.evidence import attach_evidence
    s = Store(tmp_path)
    for _ in range(3):
        _log(s, [("open", (500, 400, 520, 410))])
    iid = _log(s, [("open", (505, 402, 522, 411)), ("short", (10, 10, 30, 20))])
    rec = {"board_id": "B", "defects": [{"id": "D1", "type": "open"}, {"id": "D2", "type": "short"}]}
    out = attach_evidence(rec, s, iid, 1000, 800, MMPP)
    assert out["defects"][0]["evidence"]["repeat_same_spot"] == {"boards": 3, "of": 3}
    assert out["defects"][1]["evidence"]["co_occurring"] == ["open"]
    assert "evidence" not in rec["defects"][0]                        # the caller's record is not changed
    assert attach_evidence(rec, s, None, 1000, 800, MMPP) == rec      # not logged (demo upload): no history


def test_a_line_across_boards_is_a_lane_not_a_same_spot(tmp_path):
    s = Store(tmp_path)
    for k in range(4):                                   # same x, y moving along: a lane (a roller or a nozzle)
        _log(s, [("spur", (500, 100 + 150 * k, 510, 110 + 150 * k))])
    iid = _log(s, [("spur", (502, 700, 512, 710))])
    ev = board_evidence(s, iid, 1000, 800, MMPP)["D1"]
    assert ev["repeat_in_line"] == {"boards": 4, "of": 4}
    assert ev["repeat_same_spot"] == {"boards": 0, "of": 4}


def test_central_needs_the_frame_and_means_the_middle_of_the_board(tmp_path):
    s = Store(tmp_path)
    iid = _log(s, [("short", (490, 390, 510, 410)), ("short", (20, 20, 40, 40))])
    ev = board_evidence(s, iid, 1000, 800, MMPP)
    assert ev["D1"]["central"] is True and ev["D2"]["central"] is False
    assert board_evidence(s, iid, None, None, None)["D1"]["central"] == "unknown"


def test_a_repeat_at_the_laminator_roll_length_is_found_on_one_board(tmp_path):
    s = Store(tmp_path)
    mmpp = 0.1                                           # 10 px per mm: 220 mm apart = 2200 px, in line
    iid = _log(s, [("pin_hole", (100, 300, 110, 310)), ("pin_hole", (2300, 302, 2310, 312))])
    ev = board_evidence(s, iid, 2600, 800, mmpp)
    assert ev["D1"]["roll_pitch"] is True and ev["D2"]["roll_pitch"] is True
    near = _log(s, [("pin_hole", (100, 300, 110, 310)), ("pin_hole", (600, 300, 610, 310))])
    assert board_evidence(s, near, 2600, 800, mmpp)["D1"]["roll_pitch"] is False
    assert board_evidence(s, near, 2600, 800, None)["D1"]["roll_pitch"] == "unknown"   # needs a scale
