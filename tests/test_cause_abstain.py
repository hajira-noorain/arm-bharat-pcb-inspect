"""Evidence mode: Llama judges each candidate cause from the station's clues; a close call shows the top two."""
import re

from pcbinspect.advisor import LLAMA_BACKEND, Advisor
from pcbinspect.kb import default_kb

EV_SAME = {"repeat_same_spot": {"boards": 6, "of": 10}, "near_edge": False, "shape": "long_thin",
           "count_on_board": 1, "trend": "unknown", "co_occurring": []}


class Judge:
    """Stand-in for the logits backend: yes/no log-odds per cause, read from the question's quoted cause."""
    name, model = "llama-executorch-logits", "judge"

    def __init__(self, logodds: dict[str, float]):
        self.logodds, self.fed = logodds, []
        self.n_forward = self.pos = 0

    def reset(self):
        self.fed = []

    def feed(self, text):
        self.fed.append(text)

    def yes_no(self, text):
        short = re.search(r'"([^"]+)"', text).group(1)
        return self.logodds.get(short, -3.0)

    def score(self, options, end="\n"):
        return [(-1.0 * i, 2) for i, _ in enumerate(options)]       # first listed action

    def write(self, *a, **k):
        raise AssertionError("evidence mode does not write free text")


def rec(ev=EV_SAME):
    return {"board_id": "B1", "design": "d", "verdict": "FAIL", "defects": [
        {"id": "D1", "type": "open", "severity": "critical", "size_mm": 0.3, "location": "centre",
         "confidence": 0.9, "ipc": {}, "evidence": ev}]}


def test_a_clear_answer_names_one_cause_with_its_probability_clues_and_source():
    kb = default_kb()
    be = Judge({"dust or hair on phototool": 3.0, "handling scratch": -2.0})
    a = Advisor(be, kb, cause_mode="evidence").advise(rec())
    s = a.steps[0]
    assert a.backend == LLAMA_BACKEND and s["cause_id"] == "OPEN-C2" and not s["abstain"]
    assert s["p"]["OPEN-C2"] > 0.8 and abs(sum(s["p"].values()) - 1) < 1e-9
    assert s["evidence"] == EV_SAME and s["provenance"] == "sourced" and "phototool" in s["confirm"].lower()
    assert any("same spot on 6 of the last 10 boards" in t for t in be.fed)     # the clues were in the prompt
    assert a.summary_by == "template"                                          # no free text to contradict it


def test_a_close_call_shows_the_top_two_and_how_to_tell_them_apart():
    kb = default_kb()
    be = Judge({"dust or hair on phototool": 1.0, "resist scratch or poor adhesion": 0.95})
    a = Advisor(be, kb, cause_mode="evidence", abstain_gap=0.15).advise(rec())
    s = a.steps[0]
    assert s["abstain"] and [x["cause_id"] for x in s["alternatives"]] == ["OPEN-C2", "OPEN-C1"]
    assert all(x["confirm"] and x["p"] > 0.2 for x in s["alternatives"])
    assert a.to_json()["steps"][0]["abstain"] is True


def test_what_this_line_confirmed_shifts_the_choice_and_says_so():
    from pcbinspect.causes import CauseModel
    from pcbinspect.learned import LineCauses
    kb = default_kb()
    be = Judge({"dust or hair on phototool": 1.0, "resist scratch or poor adhesion": 0.6})
    line = LineCauses(CauseModel(kb), [("open", "OPEN-C1", EV_SAME)] * 40)
    s = Advisor(be, kb, cause_mode="evidence", line=line).advise(rec()).steps[0]
    assert s["cause_id"] == "OPEN-C1" and s["learned_from"] == 40
    plain = Advisor(Judge({"dust or hair on phototool": 1.0, "resist scratch or poor adhesion": 0.6}), kb,
                    cause_mode="evidence").advise(rec()).steps[0]
    assert plain["cause_id"] == "OPEN-C2" and plain["learned_from"] == 0


class Writer(Judge):
    """A judge that also writes the why sentence it is given."""
    def __init__(self, logodds, sentence):
        super().__init__(logodds)
        self.sentence = sentence

    def write(self, *a, **k):
        return self.sentence


def test_the_why_line_is_llamas_when_it_names_the_clue_and_standard_when_not():
    kb = default_kb()
    lo = {"dust or hair on phototool": 3.0}
    good = "The defect sits at the same place on {same_spot_boards} of the last {same_spot_of} boards, which points to {cause}."
    s = Advisor(Writer(lo, good), kb, cause_mode="evidence", why=True).advise(rec()).steps[0]
    assert s["why_by"] == "llama" and "6 of the last 10 boards" in s["why"]
    s = Advisor(Writer(lo, "It points to {cause}."), kb, cause_mode="evidence", why=True).advise(rec()).steps[0]
    assert s["why_by"] == "standard" and "6 of the last 10 boards" in s["why"]


class Recorder(Judge):
    """Records what was in the context each time a cause question was asked."""
    def __init__(self, logodds):
        super().__init__(logodds)
        self.contexts = []

    def yes_no(self, text):
        self.contexts.append(list(self.fed))
        return super().yes_no(text)


def test_every_defects_causes_are_judged_on_the_board_prompt_alone():
    """Training and evaluation ask each cause right after the evidence prompt; a board with several defects must
    not put an earlier defect's chosen cause or action lines in front of a later defect's questions."""
    kb = default_kb()
    r = rec()
    r["defects"].append({"id": "D2", "type": "short", "severity": "critical", "size_mm": 0.2, "location": "edge",
                         "confidence": 0.9, "ipc": {}, "evidence": {**EV_SAME, "count_on_board": 5}})
    be = Recorder({"dust or hair on phototool": 3.0, "under-etching": 2.0})
    Advisor(be, kb, cause_mode="evidence").advise(r)
    assert be.contexts and all(len(c) == 1 for c in be.contexts)       # only the evidence prompt was fed
