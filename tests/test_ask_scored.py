"""Shift questions with a logits backend: Llama only chooses the query (by score); hours and board IDs
come from the question; the summary is decoded through the number filter."""
import yaml
from pathlib import Path

from pcbinspect import ask
from pcbinspect.record import Defect, Inspection
from pcbinspect.store import Store


class Scripted:
    name, model = "llama-executorch-logits", "scripted"

    def __init__(self, want, pieces):
        self.want, self.pieces, self.prompts = want, pieces, []

    def reset(self):
        pass

    def feed(self, text):
        self.prompts.append(text)

    def score(self, options, end="\n"):
        assert [o.strip() for o in options] == list(ask.QUERIES)
        return [(0.0 if o.strip() == self.want else -9.0, 2) for o in options]

    def write(self, max_new_tokens, stop=("\n",), allow=None, top_k=40):
        text = ""
        for alts in self.pieces:
            piece = next((p for p in alts if allow(text, p)), None)
            if piece is None:
                break
            text += piece
        return text


def log(tmp_path, n=3):
    s = Store(tmp_path)
    for i in range(n):
        d = [Defect(id="D1", cls="short", source="detector", bbox=(1, 2, 3, 4), confidence=0.9, severity="critical", ipc={})]
        s.log_inspection(Inspection(board_id=f"B{i:05d}", design="t", image_path="", defects=d, board_score=0.9,
                                    verdict="FAIL", alpha=0.05, mm_per_px=0.02, latency_ms={}), None)
    return s


def test_llama_chooses_the_query_and_the_question_supplies_the_parameters(tmp_path):
    s = log(tmp_path)
    be = Scripted("yield", [["The station inspected "], ["7", "3"], [" boards and all of them failed."]])
    r = ask.answer("How many boards failed in the last 3 hours?", s.conn, be)
    assert r["route"] == "llama" and r["query"] == "yield" and r["params"] == {"hours": 3}
    # the model's first wish was "7": not a number in the rows, so it could not be written
    assert r["summary_by"] == "llama" and r["answer"] == "The station inspected 3 boards and all of them failed."
    assert "yield:" in be.prompts[0] and '"params"' not in be.prompts[0]


def test_parameters_are_read_from_the_question_never_generated():
    assert ask.question_params("Pull up everything recorded for board B00088", "board_history") == {"board_id": "B00088"}
    assert ask.question_params("Has serial B00023 been through the station before?", "board_history") == {"board_id": "B00023"}
    assert ask.question_params("Passed versus failed boards this week?", "yield") == {"hours": 168}
    assert ask.question_params("Count of each defect over the past 4 hours please", "defect_counts") == {"hours": 4}
    assert ask.question_params("What defect types came up today?", "defect_counts") == {"hours": 24}
    assert ask.question_params("Faults in the last 2 days", "defect_counts") == {"hours": 48}
    assert ask.question_params("Any open alerts?", "open_alerts") == {}


def test_a_failing_model_falls_back_to_the_keyword_route(tmp_path):
    class Broken(Scripted):
        def score(self, options, end="\n"):
            raise RuntimeError("model unloaded")

    r = ask.answer("Are there any open alerts?", log(tmp_path).conn, Broken("yield", []))
    assert r["route"] == "keyword" and r["query"] == "open_alerts" and "route_error" in r


def test_question_sets_are_well_formed_and_do_not_overlap():
    sets = yaml.safe_load((Path(__file__).resolve().parent.parent / "llm" / "ask_questions.yaml").read_text("utf-8"))
    for items in sets.values():
        assert all(it["query"] in ask.QUERIES for it in items)
        assert {it["query"] for it in items} == set(ask.QUERIES)            # every query is asked for
    assert not {it["q"] for it in sets["tuning"]} & {it["q"] for it in sets["heldout"]}
