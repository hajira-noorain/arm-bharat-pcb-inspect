"""The evidence prompt: one wording shared by the station and the training data."""
from pcbinspect.cause_prompt import build_evidence_prompt, cause_question, clue_text
from pcbinspect.kb import default_kb

EV = {"repeat_same_spot": {"boards": 5, "of": 12}, "near_edge": False, "shape": "long_thin",
      "count_on_board": 1, "trend": "unknown", "co_occurring": ["short"]}


def test_clues_are_plain_and_unknown_is_said_not_hidden():
    t = clue_text(EV)
    assert "same spot on 5 of the last 12 boards" in t
    assert "not near the edge" in t and "long and thin" in t and "1 of this type on the board" in t
    assert "trend unknown" in t and "also on the board: short" in t
    assert "same spot unknown" in clue_text({"repeat_same_spot": "unknown"})


def test_variants_change_wording_and_order_but_not_facts():
    a, b = clue_text(EV, variant=0), clue_text(EV, variant=3)
    assert a != b
    for fact in ("5", "12", "short"):
        assert fact in a and fact in b


def test_prompt_and_question():
    kb = default_kb()
    rec = {"board_id": "B1", "verdict": "FAIL", "defects": [
        {"id": "D1", "type": "open", "severity": "critical", "size_mm": 0.4, "location": "centre", "confidence": 0.9,
         "evidence": EV}]}
    p = build_evidence_prompt(rec, kb)
    assert p.startswith("<|begin_of_text|>") and p.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert "D1: open, critical, 0.4 mm, centre" in p and "same spot on 5 of the last 12 boards" in p
    q = cause_question("D1", "OPEN-C2", kb)
    assert q == '\nQuestion: is "dust or hair on phototool" (imaging) the likely cause of D1? Answer:'


def test_new_clues_are_written_when_known_and_left_out_when_not():
    from pcbinspect.cause_prompt import clue_text
    t = clue_text({"repeat_same_spot": {"boards": 0, "of": 8}, "repeat_in_line": {"boards": 3, "of": 8},
                   "central": True, "roll_pitch": True})
    assert "in a line with the same type on 3 of the last 8 boards" in t
    assert "in the middle of the board" in t and "laminator roll" in t
    bare = clue_text({"repeat_same_spot": "unknown", "central": "unknown", "roll_pitch": False})
    assert "line" not in bare and "middle" not in bare and "roll" not in bare
