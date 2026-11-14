import json

import pytest

from pcbinspect.advisor import Advisor, GenResult, assistant_prefix, build_prompt, template_advice
from pcbinspect.grounding import check, extract_json
from pcbinspect.kb import KnowledgeBase


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


RECORD = {
    "board_id": "B-0042", "design": "deeppcb", "verdict": "FAIL",
    "defects": [
        {"id": "D1", "type": "short", "severity": "critical", "confidence": 0.91, "size_mm": 0.31,
         "location": "top-left", "ipc": {}},
        {"id": "D2", "type": "mouse_bite", "severity": "minor", "confidence": 0.64, "size_mm": 0.12,
         "location": "centre", "ipc": {}},
    ],
}


def good_output():
    return json.dumps({"steps": [{"defect": "D1", "cause": "SHORT-C1", "action": "SHORT-A1"},
                                 {"defect": "D2", "cause": "MBITE-C1", "action": "MBITE-A1"}],
                       "summary": "A short circuit (0.31 mm) at top-left makes this board fail."})


def test_kb_ids_unique_and_complete(kb):
    for cls in ("short", "open", "mouse_bite", "spur", "spurious_copper", "missing_hole",
                "pin_hole", "missing_pad", "unknown_anomaly"):
        c, a = kb.allowed_ids(cls)
        assert c and a


def test_accepts_grounded_output(kb):
    r = check(good_output(), RECORD, kb)
    assert r.ok, r.reasons
    assert [s["defect"] for s in r.advice["steps"]] == ["D1", "D2"]


def test_rejects_cause_from_other_defect_type(kb):
    bad = good_output().replace("SHORT-C1", "OPEN-C1")
    r = check(bad, RECORD, kb)
    assert not r.ok and any("not allowed" in x for x in r.reasons)


def test_rejects_invented_id(kb):
    r = check(good_output().replace("SHORT-A1", "SHORT-A9"), RECORD, kb)
    assert not r.ok


def test_rejects_unknown_defect(kb):
    r = check(good_output().replace('"defect": "D2"', '"defect": "D7"'), RECORD, kb)
    assert not r.ok and any("D7" in x for x in r.reasons)


def test_rejects_made_up_number(kb):
    out = json.loads(good_output())
    out["summary"] = "A 2.5 mm short circuit at top-left."
    r = check(json.dumps(out), RECORD, kb)
    assert not r.ok and any("2.5" in x for x in r.reasons)


def test_accepts_confidence_as_percent(kb):
    out = json.loads(good_output())
    out["summary"] = "Short circuit found with 91% confidence."
    assert check(json.dumps(out), RECORD, kb).ok


def test_rejects_mentioning_absent_defect_type(kb):
    out = json.loads(good_output())
    out["summary"] = "Short circuit and a missing hole found."
    r = check(json.dumps(out), RECORD, kb)
    assert not r.ok and any("missing hole" in x for x in r.reasons)


def test_words_of_the_chosen_cause_are_not_a_claim_about_another_defect(kb):
    record = {"board_id": "B-1", "design": "deeppcb", "verdict": "FAIL", "defects": [
        {"id": "D1", "type": "open", "severity": "critical", "confidence": 0.9, "size_mm": 0.2,
         "location": "centre", "ipc": {}}]}
    said = "D1 is likely a handling scratch; check how panels are stacked."
    chosen = json.dumps({"steps": [{"defect": "D1", "cause": "OPEN-C4", "action": "OPEN-A1"}], "summary": said})
    assert check(chosen, record, kb).ok                        # OPEN-C4 is "handling scratch"
    other = json.dumps({"steps": [{"defect": "D1", "cause": "OPEN-C2", "action": "OPEN-A1"}], "summary": said})
    r = check(other, record, kb)                               # a cause it did not choose: still refused
    assert not r.ok and any("scratch" in x for x in r.reasons)


def test_ordinary_word_open_is_not_a_defect_mention(kb):
    out = json.loads(good_output())
    out["summary"] = "Keep the board open on the bench and check the short circuit."
    assert check(json.dumps(out), RECORD, kb).ok


def test_missing_critical_defect_is_filled_and_flagged(kb):
    out = json.loads(good_output())
    out["steps"] = [{"defect": "D2", "cause": "MBITE-C1", "action": "MBITE-A1"}]
    r = check(json.dumps(out), RECORD, kb)
    assert r.ok and r.filled_from_template == ["D1"]


def test_echoed_option_line_normalises_to_single_allowed_id(kb):
    out = json.loads(good_output())
    out["steps"][0]["cause"] = "causes: SHORT-C1: under-etching"
    r = check(json.dumps(out), RECORD, kb)
    assert r.ok and r.advice["steps"][0]["cause"] == "SHORT-C1"


def test_normalisation_never_rescues_ambiguous_or_foreign_ids(kb):
    out = json.loads(good_output())
    out["steps"][0]["cause"] = "SHORT-C1 or SHORT-C2"           # two IDs: ambiguous
    assert not check(json.dumps(out), RECORD, kb).ok
    out["steps"][0]["cause"] = "causes: OPEN-C1: over-etching"   # an ID from another defect type
    assert not check(json.dumps(out), RECORD, kb).ok


def test_numbered_choices_map_to_this_defects_own_options(kb):
    out = {"steps": [{"defect": "D1", "cause": 2, "action": "3"},
                     {"defect": "D2", "cause": 1, "action": 1}],
           "summary": "A short circuit makes this board fail."}
    r = check(json.dumps(out), RECORD, kb)
    assert r.ok
    assert r.advice["steps"][0] == {"defect": "D1", "cause": "SHORT-C2", "action": "SHORT-A3"}
    assert r.advice["steps"][1] == {"defect": "D2", "cause": "MBITE-C1", "action": "MBITE-A1"}
    out["steps"][1]["cause"] = 9            # mouse bite has only 2 causes
    r = check(json.dumps(out), RECORD, kb)
    assert not r.ok and any("out of range" in x for x in r.reasons)
    out["steps"][1]["cause"] = 0
    assert not check(json.dumps(out), RECORD, kb).ok


def test_history_numbers_are_grounded_only_when_given(kb):
    out = json.loads(good_output())
    out["summary"] = "Short circuit seen on 4 of the last 20 boards."
    assert not check(json.dumps(out), RECORD, kb).ok
    assert check(json.dumps(out), RECORD, kb, extra_numbers=[20, 4]).ok


def test_not_json_rejected(kb):
    assert not check("The board has a short.", RECORD, kb).ok


def test_extract_json_with_trailing_text():
    assert extract_json('noise {"a": {"b": "}"}} trailing {') == {"a": {"b": "}"}}


class FakeBackend:
    """Stands in for the ExecuTorch runtime: returns a fixed continuation of the prompt."""
    name, model = "llama-executorch", "fake"

    def __init__(self, continuation):
        self.continuation = continuation
        self.prompts = []

    def generate(self, prompt, max_new_tokens):
        self.prompts.append(prompt)
        return GenResult(self.continuation, latency_ms=1.0)


def test_advisor_uses_llm_when_grounded(kb):
    cont = good_output()[len(assistant_prefix(RECORD)):]
    adv = Advisor(FakeBackend(cont), kb).advise(RECORD)
    assert adv.backend == "llama-executorch" and adv.grounded
    assert adv.steps[0]["action"] == kb.text("SHORT-A1")  # KB text verbatim


def test_advisor_falls_back_to_template_on_hallucination(kb):
    cont = good_output().replace("SHORT-C1", "MADE-UP-1")[len(assistant_prefix(RECORD)):]
    adv = Advisor(FakeBackend(cont), kb).advise(RECORD)
    assert adv.backend == "template" and adv.rejected_reasons


class FakeStepwise:
    """Simulates the ExecuTorch runner's continuation API with scripted outputs."""
    name, model = "llama-executorch", "fake-stepwise"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.sent = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def cont(self, text, max_new_tokens, stop=None):
        self.sent.append(text)
        return GenResult(self.outputs.pop(0), latency_ms=1.0)


def test_stepwise_maps_shuffled_numbers_and_covers_all_defects(kb):
    from pcbinspect.advisor import option_orders
    be = FakeStepwise(['2, "action": 1}', '1, "action": 3},',
                       'Short circuit at top-left; the board fails."}'])
    adv = Advisor(be, kb, option_order="shuffle").advise(RECORD, {"short": 4})
    assert adv.backend == "llama-executorch", adv.rejected_reasons
    orders = option_orders(RECORD, kb)
    assert adv.steps[0]["cause_id"] == orders["D1"][0][1] and adv.steps[0]["action_id"] == orders["D1"][1][0]
    assert adv.steps[1]["cause_id"] == orders["D2"][0][0] and adv.steps[1]["action_id"] == orders["D2"][1][2]
    assert be.resets == 1 and len(be.sent) == 3          # context prefilled once, then continuations
    assert be.sent[0].endswith('{"steps": [{"defect": "D1", "cause": ')
    assert be.sent[1] == ', {"defect": "D2", "cause": '
    assert be.sent[2] == '], "summary": "'
    assert adv.choices[0]["cause_pos"] == 2


def test_join_drops_what_the_stop_token_already_wrote():
    from pcbinspect.advisor import _join
    assert _join(",", ', {"defect": "D2"') == ' {"defect": "D2"'
    assert _join("],", '], "summary": "') == ' "summary": "'
    assert _join("", ', {"defect"') == ', {"defect"'


def test_stepwise_tolerates_missing_brace(kb):
    from pcbinspect.advisor import _choice_done
    assert not _choice_done('2, "action": 1')               # might still be "1" of "13"
    assert _choice_done('2, "action": 1, {"defect": "D')     # second number finished
    be = FakeStepwise(['1, "action": 1, {"defect": "D', '2, "action": 2}',
                       'Two defects found."}'])
    adv = Advisor(be, kb).advise(RECORD, None)
    assert adv.backend == "llama-executorch", adv.rejected_reasons
    assert be.sent[1] == '2", "cause": '                     # model already wrote ', {"defect": "D'


def test_stepwise_bad_choice_falls_back_to_template(kb):
    be = FakeStepwise(['under-etching", "action": 1}'])
    adv = Advisor(be, kb).advise(RECORD, None)
    assert adv.backend == "template" and "expected form" in adv.rejected_reasons[0]


def test_evidence_order_puts_process_causes_first_for_repeats(kb):
    from pcbinspect.advisor import option_orders
    rep = option_orders(RECORD, kb, history={"short": 5}, mode="evidence")["D1"][0]
    one = option_orders(RECORD, kb, history=None, mode="evidence")["D1"][0]
    assert kb.by_id[rep[0]]["step"] != "handling"
    assert kb.by_id[one[0]]["step"] == "handling"


def test_option_order_is_shuffled_but_stable(kb):
    from pcbinspect.advisor import option_orders
    a, b = option_orders(RECORD, kb), option_orders(RECORD, kb)
    assert a == b                                          # deterministic per board
    assert sorted(a["D1"][0]) == sorted(kb.allowed_ids("short")[0])


def test_prompt_contains_only_present_defect_options(kb):
    p_ids = build_prompt(RECORD, {"short": 4}, kb, style="ids")
    assert "SHORT-C1" in p_ids and "MBITE-A1" in p_ids and "OPEN-C1" not in p_ids
    p = build_prompt(RECORD, {"short": 4}, kb)
    assert "1 under-etching" in p and "1 over-etching or undercut" in p
    assert "dust or hair on phototool" not in p     # open-circuit options must not appear
    assert "short on 4 of the last 20 boards" in p
    assert p.endswith(assistant_prefix(RECORD))


def test_template_orders_by_severity(kb):
    t = template_advice({**RECORD, "defects": RECORD["defects"][::-1]}, kb)
    assert t["steps"][0]["defect"] == "D1"


def test_the_summary_may_not_blame_handling_where_the_chosen_cause_is_a_process_step(kb):
    from pcbinspect.grounding import attribution_conflicts
    steps = [{"defect": "D1", "cause": "OPEN-C2", "action": "OPEN-A1"},        # imaging
             {"defect": "D2", "cause": "OPEN-C4", "action": "OPEN-A1"}]        # handling
    seen = "D1 is likely process-related, while D2 and D1 may be related to handling or local accidents."
    assert attribution_conflicts(seen, steps, kb)                              # D1 is not handling
    ok = "D1 is likely process-related, while D2 is likely a handling scratch."
    assert attribution_conflicts(ok, steps, kb) == []
    assert attribution_conflicts("D2 is likely process-related.", steps, kb)  # D2's cause is handling
    assert attribution_conflicts("Both look like etching problems; check the line.", steps, kb) == []
    only_process = [{"defect": "D1", "cause": "OPEN-C2", "action": "OPEN-A1"}]
    assert attribution_conflicts("Likely damage from handling the panel.", only_process, kb)
    assert attribution_conflicts("Likely damage from handling the panel.", steps, kb) == []   # one is handling


def test_a_contradicting_summary_falls_back_to_knowledge_base_text(kb):
    record = {"board_id": "B-1", "design": "deeppcb", "verdict": "FAIL", "defects": [
        {"id": "D1", "type": "open", "severity": "critical", "confidence": 0.9, "size_mm": 0.2, "location": "centre", "ipc": {}}]}
    bad = json.dumps({"steps": [{"defect": "D1", "cause": "OPEN-C2", "action": "OPEN-A1"}],
                      "summary": "D1 was probably damaged in handling."})
    r = check(bad, record, kb)
    assert not r.ok and any("handling" in x for x in r.reasons)
