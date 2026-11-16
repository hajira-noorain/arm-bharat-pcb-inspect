"""The "why" line: one checkable sentence saying which clue points to the chosen cause."""
import pytest

from pcbinspect.causes import CauseModel
from pcbinspect.kb import default_kb
from pcbinspect.why import check_why, standard_why, why_facts

SAME = {"repeat_same_spot": {"boards": 6, "of": 10}, "repeat_in_line": {"boards": 0, "of": 10}, "near_edge": False,
        "shape": "long_thin", "count_on_board": 1, "trend": "unknown", "central": False, "roll_pitch": False}


@pytest.fixture(scope="module")
def kbcm():
    kb = default_kb()
    return kb, CauseModel(kb)


def test_the_key_clue_is_the_one_that_most_favours_the_cause(kbcm):
    kb, cm = kbcm
    f, key = why_facts("open", SAME, "OPEN-C2", cm, kb)        # phototool dust: the same spot is the tell
    assert key == "same_spot_boards"
    assert f["cause"] == "dust or hair on phototool" and f["same_spot_boards"] == 6 and f["same_spot_of"] == 10
    assert "trend" not in f                                    # unknown clues are not facts


def test_the_checker_wants_the_cause_and_the_key_clue_and_no_typed_numbers(kbcm):
    kb, cm = kbcm
    f, key = why_facts("open", SAME, "OPEN-C2", cm, kb)
    ok = "The defect sits at the same place on {same_spot_boards} of the last {same_spot_of} boards, which points to {cause}."
    assert check_why(ok, f, key) == []
    assert check_why("It points to {cause}.", f, key)                         # misses the clue
    assert check_why("Same spot on 6 of {same_spot_of} boards: {cause}.", f, key)   # typed a number
    assert check_why("{cause} because of {colour}.", f, key)                  # no such field
    assert "6 of the last 10" in standard_why(f, key) and "dust or hair on phototool" in standard_why(f, key).lower()


def test_no_clue_for_the_cause_means_no_why(kbcm):
    kb, cm = kbcm
    f, key = why_facts("open", {"repeat_same_spot": "unknown", "trend": "unknown"}, "OPEN-C2", cm, kb)
    assert key is None


def test_why_training_targets_pass_the_checker():
    from llm.data.gen_why import examples
    exs = examples(300, seed=3)
    assert len(exs) == 300 and len({e["key"] for e in exs}) >= 4
    for e in exs:
        assert check_why(e["target"], e["facts"], e["key"]) == [], e


def test_the_prompt_can_name_the_clue_to_use(kbcm):
    from pcbinspect.why import why_prompt
    kb, cm = kbcm
    f, key = why_facts("open", SAME, "OPEN-C2", cm, kb)
    assert "Name this clue: same_spot_boards" in why_prompt(f, key)
    assert "Name this clue" not in why_prompt(f)                 # the prompt v3 was trained on
