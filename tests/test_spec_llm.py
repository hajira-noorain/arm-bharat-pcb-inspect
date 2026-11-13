import re

import pytest

from pcbinspect.advisor import GenResult
from pcbinspect.spec import Rule, same_rule
from pcbinspect.spec_llm import LlamaSpecCompiler, assemble, build_prompt, _lists


class Scripted:
    """Backend that answers each field from a dict; records what the compiler asked."""

    def __init__(self, answers):
        self.answers = answers
        self.asked = []

    def reset(self):
        self.asked.clear()

    def cont(self, text, n, stop=None):
        label = re.findall(r"([A-Z][A-Za-z ]+):$", text)[-1]
        self.asked.append(label)
        return GenResult(f" {self.answers[label]}\nNext", 5.0, None, 20.0, 3, 400 if len(self.asked) == 1 else 2)


@pytest.mark.parametrize("answers,expected", [
    ({"Defect": "pin hole", "Measure": "size", "Number as written": "4",
      "Unit": "mil", "Comparison": "more than", "Action": "scrap"},
     Rule(defect="pin_hole", measure="size_mm", op=">", value=0.1016, action="reject")),
    ({"Defect": "mouse bite", "Measure": "width loss", "Number as written": "15",
      "Unit": "percent", "Comparison": "at least", "Action": "rework"},
     Rule(defect="mouse_bite", measure="width_loss_pct", op=">=", value=15, action="rework")),
    ({"Defect": "short", "Measure": "always", "Action": "reject"},
     Rule(defect="short", measure="always", action="reject")),
    ({"Defect": "none", "Class number": "3"}, Rule(kind="class", customer_class=3)),
    ({"Defect": "spurious copper", "Measure": "size", "Number as written": "150",
      "Unit": "micron", "Comparison": "more than", "Action": "hold for review"},
     Rule(defect="spurious_copper", measure="size_mm", op=">", value=0.15, action="review")),
])
def test_compiler_turns_field_answers_into_rules(answers, expected):
    be = Scripted(answers)
    r = LlamaSpecCompiler(be).compile("any clause", order_seed=3)
    assert same_rule(r.rule, expected), (r.rule, r.error)
    # 'always' rules must not ask for a number, unit or comparison
    if answers.get("Measure") == "always":
        assert "Number as written" not in be.asked
    assert r.prompt_tokens == 400


def test_unusable_answers_produce_no_rule_not_a_guess():
    r, err = assemble({"Defect": "solder ball", "Measure": "size", "Action": "reject"})
    assert r is None and "defect" in err
    r, err = assemble({"Defect": "open", "Measure": "size", "Number as written": "small",
                       "Unit": "mm", "Action": "reject"})
    assert r is None and err == "no number"
    r, err = assemble({"Defect": "open", "Measure": "width loss", "Number as written": "3",
                       "Unit": "mil", "Comparison": "more than", "Action": "reject"})
    assert r is None and "unit" in err


def test_option_lists_are_shuffled_but_complete():
    a, b = _lists(__import__("random").Random(1)), _lists(__import__("random").Random(2))
    assert sorted(a["Defect"]) == sorted(b["Defect"]) and a["Defect"] != b["Defect"]
    p = build_prompt("No shorts are permitted.", a)
    assert p.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n") and "No shorts are permitted." in p
