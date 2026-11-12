import json

import pytest

from pcbinspect.advisor import ASSISTANT_PREFIX, Advisor, GenResult, build_prompt, template_advice
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
    return json.dumps({"summary": "A short circuit (0.31 mm) at top-left makes this board fail.",
                       "steps": [{"defect": "D1", "cause": "SHORT-C1", "action": "SHORT-A1"},
                                 {"defect": "D2", "cause": "MBITE-C1", "action": "MBITE-A1"}]})


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


def test_ordinary_word_open_is_not_a_defect_mention(kb):
    out = json.loads(good_output())
    out["summary"] = "Keep the board open on the bench and check the short circuit."
    assert check(json.dumps(out), RECORD, kb).ok


def test_missing_critical_defect_is_filled_and_flagged(kb):
    out = json.loads(good_output())
    out["steps"] = [{"defect": "D2", "cause": "MBITE-C1", "action": "MBITE-A1"}]
    r = check(json.dumps(out), RECORD, kb)
    assert r.ok and r.filled_from_template == ["D1"]


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
    cont = good_output()[len(ASSISTANT_PREFIX):]
    adv = Advisor(FakeBackend(cont), kb).advise(RECORD)
    assert adv.backend == "llama-executorch" and adv.grounded
    assert adv.steps[0]["action"] == kb.text("SHORT-A1")  # KB text verbatim


def test_advisor_falls_back_to_template_on_hallucination(kb):
    cont = good_output().replace("SHORT-C1", "MADE-UP-1")[len(ASSISTANT_PREFIX):]
    adv = Advisor(FakeBackend(cont), kb).advise(RECORD)
    assert adv.backend == "template" and adv.rejected_reasons


def test_prompt_contains_only_present_defect_options(kb):
    p = build_prompt(RECORD, {"short": 4}, kb)
    assert "SHORT-C1" in p and "MBITE-A1" in p and "OPEN-C1" not in p
    assert "short on 4 of the last 20 boards" in p
    assert p.endswith(ASSISTANT_PREFIX)


def test_template_orders_by_severity(kb):
    t = template_advice({**RECORD, "defects": RECORD["defects"][::-1]}, kb)
    assert t["steps"][0]["defect"] == "D1"
