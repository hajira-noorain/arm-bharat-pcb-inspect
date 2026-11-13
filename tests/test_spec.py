from pathlib import Path

import pytest
import yaml

from pcbinspect.spec import (Rule, apply_rules, capability, keyword_compile, numbers_in, render, rule_from_short,
                             same_rule, schema_errors, verify)

BENCH = yaml.safe_load((Path(__file__).resolve().parent.parent / "llm" / "spec_clauses.yaml").read_text("utf-8"))


def test_benchmark_gold_rules_are_valid_and_grounded():
    assert len(BENCH) >= 80
    for item in BENCH:
        r = rule_from_short(item)
        assert not schema_errors(r), item["t"]
        v = verify(r, item["t"])
        assert v["status"] in ("verified", "check"), (item["t"], v)  # never invalid / ungrounded


def test_verifier_false_flag_rate_on_gold_is_low():
    flagged = [it["t"] for it in BENCH if verify(rule_from_short(it), it["t"])["status"] != "verified"]
    assert len(flagged) / len(BENCH) <= 0.10, flagged


@pytest.mark.parametrize("factor", [10.0, 0.1, 1 / 0.0254])
def test_wrong_magnitude_or_unit_is_caught_as_ungrounded(factor):
    caught = n = 0
    for it in BENCH:
        r = rule_from_short(it)
        if r.kind != "limit" or r.measure in ("always", "count"):
            continue
        if r.measure.endswith("_pct") and r.value * factor > 100:
            continue
        r.value *= factor
        n += 1
        caught += verify(r, it["t"])["status"] in ("ungrounded", "invalid")
    assert n > 20 and caught == n


def test_units_parse():
    assert (4.0, "mil") in numbers_in("bigger than 4 mil")
    assert (150.0, "micron") in numbers_in("over 150 microns")
    assert (0.2, "mm") in numbers_in("bigger than .2 mm")
    assert (15.0, "%") in numbers_in("by 15 % or more")
    assert (2.0, None) in numbers_in("more than two mouse bites")


def test_wrong_defect_and_wrong_comparison_are_flagged():
    t = "Nicks that reduce the conductor width by more than 20% are rejectable."
    good = Rule(defect="mouse_bite", measure="width_loss_pct", op=">", value=20, action="reject")
    assert verify(good, t)["status"] == "verified"
    assert verify(Rule(**{**good.to_dict(), "defect": "short"}), t)["status"] == "check"
    assert verify(Rule(**{**good.to_dict(), "op": ">="}), t)["status"] == "check"
    assert verify(Rule(**{**good.to_dict(), "action": "accept"}), t)["status"] == "check"


def test_keyword_baseline_runs_on_every_clause():
    exact = sum(same_rule(keyword_compile(it["t"]), rule_from_short(it)) for it in BENCH)
    assert 0 < exact < len(BENCH)  # a real baseline: right on some, wrong on others


def test_render_is_plain_english():
    r = Rule(defect="pin_hole", measure="size_mm", op=">", value=0.1, action="reject")
    assert render(r) == "Reject the board for a pin hole when it is more than 0.1 mm in size."
    assert render(Rule(kind="class", customer_class=3)) == "Judge boards to IPC-style Class 3."


def test_apply_rules_worst_action_and_count():
    rules = [Rule(defect="short", measure="always", action="reject"),
             Rule(defect="mouse_bite", measure="width_loss_pct", op=">", value=20, action="rework"),
             Rule(defect="pin_hole", measure="count", op=">", value=2, action="review")]
    defects = [{"cls": "mouse_bite", "width_loss_pct": 10}, {"cls": "mouse_bite", "width_loss_pct": 35},
               {"cls": "pin_hole"}, {"cls": "pin_hole"}]
    acts, board = apply_rules(rules, defects)
    assert acts == [None, "rework", None, None] and board == "rework"
    acts, board = apply_rules(rules, defects + [{"cls": "pin_hole"}, {"cls": "short"}])
    assert acts[2] == "review" and board == "reject"
    # unmeasured width loss: a rework rule cannot be cleared
    acts, _ = apply_rules(rules, [{"cls": "mouse_bite", "width_loss_pct": None}])
    assert acts == ["rework"]


def test_capability_against_measured_limit():
    lim = {"open": 0.18, "short": 0.24, "pin_hole": None}
    r = Rule(defect="open", measure="size_mm", op=">", value=0.1, action="reject")
    c = capability(r, lim)
    assert c["status"] == "not_guaranteed" and "0.10" in c["reason"] and "0.18" in c["reason"]
    assert capability(Rule(defect="open", measure="size_mm", op=">", value=0.3, action="reject"), lim)["status"] == "guaranteed"
    assert capability(Rule(defect="short", measure="always", action="reject"), lim)["status"] == "limited"
    assert capability(Rule(defect="pin_hole", measure="always", action="reject"), lim)["status"] == "unmeasured"
    # 20% of a 0.5 mm trace = 0.10 mm needed, below the 0.18 mm limit
    w = Rule(defect="open", measure="width_loss_pct", op=">", value=20, action="reject")
    assert capability(w, lim, min_trace_width_mm=0.5)["status"] == "not_guaranteed"
    assert capability(w, lim, min_trace_width_mm=1.0)["status"] == "guaranteed"
