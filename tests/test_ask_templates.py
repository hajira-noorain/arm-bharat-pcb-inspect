"""Fill-in sentences for shift answers: Llama writes {field} placeholders, the station writes the numbers."""
import pytest

from pcbinspect.sentences import check_template, facts_for, render, required_fields, standard_sentence

YIELD = [{"boards": 28, "passed": 3, "failed": 0, "review": 25}]
COUNTS = [{"defect": "unknown_anomaly", "n": 28}, {"defect": "spurious_copper", "n": 25}, {"defect": "short", "n": 19}]
SEV = [{"severity": "major", "n": 66}, {"severity": "critical", "n": 44}]


def test_facts_flatten_rows_and_precompute_rankings():
    assert facts_for("yield", YIELD, {"hours": 24}) == {"boards": 28, "passed": 3, "failed": 0, "review": 25,
                                                        "pass_rate_pct": 11, "hours": 24}
    f = facts_for("defect_counts", COUNTS, {"hours": 8})
    assert (f["top_defect"], f["top_count"], f["second_defect"], f["second_count"]) == \
        ("unknown anomaly", 28, "spurious copper", 25)
    assert f["total_defects"] == 72 and f["n_types"] == 3 and f["count_short"] == 19
    s = facts_for("severity_counts", SEV, {"hours": 8})
    assert (s["critical"], s["major"], s["minor"], s["total_defects"]) == (44, 66, 0, 110)


def test_what_the_question_asks_for_must_be_in_the_answer():
    assert required_fields("How many boards went to a person for checking today?", "yield") == {"review"}
    assert required_fields("What is the pass rate today?", "yield") == {"passed", "pass_rate_pct"}
    assert required_fields("Which defect is the most common this shift?", "defect_counts") == {"top_defect"}
    assert required_fields("Split the defects from today into critical, major and minor", "severity_counts") == \
        {"critical", "major", "minor"}


def test_the_checker_rejects_the_mistakes_seen_on_the_demo():
    f = facts_for("yield", YIELD, {"hours": 24})
    q = "How many boards went to a person for checking today?"
    assert check_template("{boards} boards were sent for checking today.", q, "yield", f)      # wrong field
    assert check_template("Here is the summary:", q, "yield", f)                               # no answer
    assert check_template("", q, "yield", f)
    assert check_template("25 boards went to a person.", q, "yield", f)                        # typed number
    assert check_template("{reviewed} boards went to a person.", q, "yield", f)                # unknown field
    assert check_template("{review} of {boards} boards went to a person today.", q, "yield", f) == []
    c = facts_for("defect_counts", COUNTS, {"hours": 8})
    q2 = "Which defect is the most common this shift?"
    assert check_template("The most common defect is {second_defect} with {second_count}.", q2, "defect_counts", c)
    assert check_template("The most common defect is {top_defect}, found {top_count} times.", q2,
                          "defect_counts", c) == []
    assert check_template("One. Two. Three sentences is too long.", q2, "defect_counts", c)


def test_render_and_standard_sentences():
    f = facts_for("yield", YIELD, {"hours": 24})
    assert render("{review} of {boards} boards went to a person.", f) == "25 of 28 boards went to a person."
    assert standard_sentence("yield", f) == ("28 boards inspected in the last 24 hours: 3 passed (11%), 0 failed, "
                                             "25 sent to a person.")
    c = facts_for("defect_counts", COUNTS, {"hours": 8})
    assert standard_sentence("defect_counts", c).startswith("72 defects in the last 8 hours; most common: unknown anomaly (28)")
    assert standard_sentence("open_alerts", facts_for("open_alerts", [], {})) == "No open alerts."


class TemplateWriter:
    """A scored backend stand-in: routes to `want` and writes `tpl` as the sentence."""
    name, model = "llama-executorch-logits", "scripted"

    def __init__(self, want, tpl):
        self.want, self.tpl, self.prompts = want, tpl, []

    def reset(self):
        pass

    def feed(self, text):
        self.prompts.append(text)

    def score(self, options, end="\n"):
        return [(0.0 if o.strip() == self.want else -9.0, 2) for o in options]

    def write(self, max_new_tokens, stop=("\n",), allow=None, top_k=40, nonempty=False):
        assert allow is not None and not allow("", "2") and allow("", "{review}")   # digits cannot be typed
        return self.tpl


def _store(tmp_path):
    from pcbinspect.record import Inspection
    from pcbinspect.store import Store
    s = Store(tmp_path)
    for v in ["PASS", "REVIEW", "REVIEW", "FAIL"]:
        s.log_inspection(Inspection("B-1", "deeppcb", "", [], 0.0, v, 0.05, None), None)
    return s


def test_ask_renders_a_correct_fill_in_sentence(tmp_path):
    from pcbinspect import ask
    s = _store(tmp_path)
    be = TemplateWriter("yield", "{review} of {boards} boards went to a person today.")
    out = ask.answer("How many boards went to a person for checking today?", s.conn, backend=be)
    assert out["answer"] == "2 of 4 boards went to a person today." and out["summary_by"] == "llama"
    assert out["template"] == "{review} of {boards} boards went to a person today."
    assert any("review = 2" in p for p in be.prompts)          # Llama saw labelled facts, not raw rows


def test_ask_falls_back_to_the_standard_sentence_on_a_wrong_field(tmp_path):
    from pcbinspect import ask
    s = _store(tmp_path)
    out = ask.answer("How many boards went to a person for checking today?", s.conn,
                     backend=TemplateWriter("yield", "{boards} boards were sent for checking today."))
    assert out["summary_by"] == "template" and "review" in out["summary_rejected"]
    assert out["answer"] == "4 boards inspected in the last 24 hours: 1 passed (25%), 1 failed, 2 sent to a person."


def test_without_llama_the_answer_is_the_standard_sentence(tmp_path):
    from pcbinspect import ask
    out = ask.answer("What is the pass rate today?", _store(tmp_path).conn, backend=None)
    assert out["answer"].startswith("4 boards inspected in the last 24 hours: 1 passed")


def test_a_placeholder_can_carry_its_noun_so_the_station_gets_the_plural_right():
    f = {"n_alerts": 1, "times_inspected": 3, "count_scratch": 2, "n_types": 1}
    assert render("There is {n_alerts|alert} open; seen {times_inspected|time}.", f) == \
        "There is 1 alert open; seen 3 times."
    assert render("{count_scratch|scratch} of {n_types|type}.", f) == "2 scratches of 1 type."
    assert check_template("There are {n_alerts|alert} open.", "Any open alerts?", "open_alerts", f) == []
    assert check_template("There are {n_alrts|alert} open.", "Any open alerts?", "open_alerts", f)


def test_when_nothing_was_inspected_llama_is_not_asked(tmp_path):
    from pcbinspect import ask
    from pcbinspect.store import Store
    be = TemplateWriter("yield", "{failed} boards failed out of {boards}.")
    out = ask.answer("How many boards failed in the last 4 hours?", Store(tmp_path).conn, backend=be)
    assert out["answer"] == "No boards inspected in the last 4 hours." and out["summary_by"] == "template"
    assert not any("Facts:" in p for p in be.prompts)


def test_a_sentence_starting_with_a_filled_in_name_starts_with_a_capital():
    assert render("{top_defect} is the most frequent defect.", {"top_defect": "scratch"}) == "Scratch is the most frequent defect."


def test_a_rate_question_gets_the_rate_and_must_use_it():
    f = facts_for("yield", [{"boards": 8, "passed": 6, "failed": 1, "review": 1}], {"hours": 8})
    assert f["pass_rate_pct"] == 75
    need = required_fields("What is the pass rate this shift?", "yield")
    assert "pass_rate_pct" in need
    assert check_template("Out of {boards|board}, {passed} passed.", "What is the pass rate this shift?", "yield", f)       # no rate: refused
    assert not check_template("The pass rate is {pass_rate_pct}% ({passed} of {boards|board}).", "What is the pass rate this shift?", "yield", f)
    assert "75%" in standard_sentence("yield", f)


def test_a_comparison_question_gets_the_comparison_computed_by_the_station():
    q = "Were there more major or minor defects in the last 4 hours?"
    f = facts_for("severity_counts", [{"severity": "major", "n": 13}, {"severity": "minor", "n": 20}], {"hours": 4}, q)
    assert f["comparison"] == "more minor than major"
    need = required_fields(q, "severity_counts")
    assert "comparison" in need
    assert check_template("{major} major and {minor} minor.", q, "severity_counts", f)              # lists, does not answer
    assert not check_template("There were {comparison} defects: {major} major and {minor} minor.", q, "severity_counts", f)
    assert "more minor than major" in standard_sentence("severity_counts", f)
    eq = facts_for("severity_counts", [{"severity": "major", "n": 5}, {"severity": "minor", "n": 5}], {"hours": 4}, q)
    assert eq["comparison"] == "as many major as minor"


def test_comparing_two_defect_types_names_them_in_plural():
    q = "Are there more opens or shorts today?"
    f = facts_for("defect_counts", [{"defect": "short", "n": 2}, {"defect": "open", "n": 9}], {"hours": 24}, q)
    assert f["comparison"] == "more opens than shorts"
    assert "comparison" in required_fields(q, "defect_counts")
    g = facts_for("defect_counts", [{"defect": "short", "n": 2}], {"hours": 24}, q)
    assert g["comparison"] == "more shorts than opens"                                # a type nobody found counts 0
