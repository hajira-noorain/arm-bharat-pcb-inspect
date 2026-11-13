"""The scored advisor path with a scripted backend: choices by option text, a summary that cannot
contain a number the findings do not, and the grounding checker still in charge."""
import numpy as np

from pcbinspect.advisor import LLAMA_BACKEND, Advisor, build_scored_prompt, option_orders
from pcbinspect.kb import default_kb

RECORD = {"board_id": "B-1", "design": "t", "verdict": "FAIL", "defects": [
    {"id": "D1", "type": "short", "severity": "critical", "confidence": 0.91, "size_mm": 0.18, "location": "centre"},
    {"id": "D2", "type": "open", "severity": "critical", "confidence": 0.8, "size_mm": 0.3, "location": "top-left"}]}


class Scripted:
    """Stands in for LogitLlama: prefers options containing a keyword; writes a scripted summary through
    the number filter exactly as write() would (token by token)."""
    name, model = "llama-executorch-logits", "scripted"

    def __init__(self, prefer, summary_pieces):
        self.prefer, self.pieces = prefer, summary_pieces
        self.pos = self.n_forward = 0
        self.fed, self.scored = [], []

    def reset(self):
        self.pos = 0

    def feed(self, text):
        self.fed.append(text)
        self.pos += max(1, len(text.split()))

    def score(self, options, end="\n"):
        self.scored.append([o.strip() for o in options])
        return [(0.0 if any(k in o for k in self.prefer) else -5.0 - i, 3) for i, o in enumerate(options)]

    def write(self, max_new_tokens, stop=("\n",), allow=None, top_k=40):
        text = ""
        for alternatives in self.pieces:                 # each step: candidates in the model's order
            piece = next((p for p in alternatives if allow is None or allow(text, p)), None)
            if piece is None:
                break
            text += piece
        return text


def test_prompt_tells_the_model_what_repeats_and_which_step_each_cause_belongs_to():
    kb = default_kb()
    orders = option_orders(RECORD, kb, mode="kb")
    p = build_scored_prompt(RECORD, {"short": 5}, kb, orders)
    assert "D1: short, critical, 0.18 mm, centre, confidence 0.91. REPEATING: on 5 of the last 20 boards." in p
    assert "ONE-OFF: not on other recent boards" in p            # the open has no history
    assert "under-etching (etching)" in p and "conductive debris on panel (handling)" in p
    assert "1 under-etching" not in p                            # no option numbers anywhere


def test_choices_are_made_on_option_text_and_the_result_is_grounded_advice():
    kb = default_kb()
    be = Scripted(prefer=("under-etching", "handling scratch", "continuity"),
                  summary_pieces=[["Two critical defects: a short of "], ["0.18"], [" mm and an open."]])
    a = Advisor(be, kb, option_order="kb").advise(RECORD, {"short": 5})
    assert a.backend == LLAMA_BACKEND and not a.rejected_reasons
    by = {s["defect"]: s for s in a.steps}
    assert by["D1"]["cause_id"] == "SHORT-C1" and by["D2"]["cause_id"] == "OPEN-C4"
    assert all("continuity" in s["action"].lower() for s in a.steps)
    assert be.scored[0] == ["under-etching", "phototool spot or resist residue", "conductive debris on panel"]
    assert " under-etching" in be.fed and "\nSummary:" in be.fed          # the chosen text enters the context
    assert a.choices[0]["cause_pos"] == 1 and a.choices[1]["cause_pos"] == 4 and a.summary.endswith("an open.")


def test_a_number_that_is_not_in_the_findings_cannot_be_written_into_the_summary():
    kb = default_kb()
    # the model's first wish is "0.25" (not a fact); the filter makes it fall through to its next candidate
    be = Scripted(prefer=("under-etching", "over-etching", "continuity"),
                  summary_pieces=[["The short measures "], ["0.25", "0.18"], [" mm."]])
    a = Advisor(be, kb, option_order="kb").advise(RECORD, None)
    assert a.backend == LLAMA_BACKEND and "0.18 mm" in a.summary and "0.25" not in a.summary


def test_the_checker_still_rejects_what_decoding_cannot_prevent():
    kb = default_kb()
    be = Scripted(prefer=("under-etching", "over-etching", "continuity"),
                  summary_pieces=[["A mouse bite was found."]])            # a defect type that is not on the board
    a = Advisor(be, kb, option_order="kb").advise(RECORD, None)
    assert a.backend == "template" and a.rejected_reasons and len(a.choices) == 2
    be2 = Scripted(prefer=("under-etching",), summary_pieces=[])
    assert Advisor(be2, kb, option_order="kb").advise(RECORD, None).rejected_reasons == ["empty summary"]


def test_pointwise_mode_judges_each_cause_alone_and_keeps_the_list_out_of_the_prompt():
    kb = default_kb()

    class YesNo(Scripted):
        def __init__(self, *a):
            super().__init__(*a)
            self.questions = []

        def yes_no(self, text):
            self.questions.append(text)
            repeating = "is a REPEATING defect" in text
            handling = "(handling)" in text
            return 3.0 if (repeating != handling) else -3.0      # the rule: repeating -> process, one-off -> handling

    be = YesNo(("continuity",), [["A short and an open were found."]])
    a = Advisor(be, kb, option_order="kb", cause_mode="pointwise").advise(RECORD, {"short": 5})
    by = {s["defect"]: s for s in a.steps}
    assert kb.by_id[by["D1"]["cause_id"]]["step"] != "handling"          # the repeating short: a process cause
    assert kb.by_id[by["D2"]["cause_id"]]["step"] == "handling"          # the one-off open: the handling cause
    assert len(be.questions) == 3 + 4 and all("Candidate cause:" in q for q in be.questions)
    assert "causes:" not in be.fed[0] and "actions:" in be.fed[0]        # no cause list to copy from
    assert a.backend == LLAMA_BACKEND
