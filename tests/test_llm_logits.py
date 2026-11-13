"""Grammar-constrained decoding over a closed set of answers (no model needed: a scripted scorer)."""
import numpy as np
import pytest

from pcbinspect.llm_logits import constrained_pick
from pcbinspect.spec_llm import ConstrainedSpecCompiler, numbers_as_written

V = 50


def scripted(prefs):
    """step() whose logits favour, at each call, the tokens in prefs[call] (dict token -> logit)."""
    calls = []

    def step(tok):
        calls.append(tok)
        lg = np.full(V, -5.0, np.float32)
        for t, v in prefs[min(len(calls) - 1, len(prefs) - 1)].items():
            lg[t] = v
        return lg
    return step, calls


def test_only_listed_answers_can_come_out_even_if_the_model_prefers_something_else():
    step, _ = scripted([{40: 9.0, 3: 2.0, 7: 1.0}])           # token 40 is not the start of any option
    i, margin, used = constrained_pick(step, [[3, 4], [7], [8, 9]])
    assert i == 0 and used == [3, 4] and margin == pytest.approx(1.0, abs=1e-5)


def test_multi_token_options_follow_the_tree_and_feed_each_chosen_token_once():
    step, calls = scripted([{5: 1.0}, {6: 0.0, 7: 3.0}])
    i, _, used = constrained_pick(step, [[5, 6], [5, 7], [8]])
    assert i == 1 and used == [5, 7] and calls == [None, 5]     # the last token is not fed: nothing left to decide


def test_an_option_that_is_a_prefix_of_another_can_still_win():
    """'spur' vs 'spurious copper': after 'spur', if the model wants anything but the continuation
    (a newline), the short option is the answer."""
    step, _ = scripted([{1: 5.0}, {30: 9.0, 2: 0.0}])           # after token 1 the model wants 30 (not 2)
    assert constrained_pick(step, [[1], [1, 2]])[0] == 0
    step, _ = scripted([{1: 5.0}, {2: 9.0}])                    # ...or clearly continues
    assert constrained_pick(step, [[1], [1, 2]])[0] == 1


def test_numbers_offered_are_the_ones_written_in_the_clause():
    assert numbers_as_written("Reject opens wider than 0.15 mm or longer than 3 mil; Class 2.") == ["0.15", "3", "2"]
    assert numbers_as_written("More than two pin holes per board is a reject.") == ["two"]
    assert numbers_as_written("Any short is a reject.") == []


class FakeBackend:
    """Chooses the option that contains a keyword of the clause (stands in for the model)."""

    def __init__(self, picks):
        self.picks, self.pos, self.n_forward, self.asked = picks, 0, 0, []

    def reset(self):
        self.pos = 0

    def feed(self, text):
        self.pos += len(text.split())
        self.label = text.strip().rstrip(":").split("\n")[-1]

    def choose(self, options):
        self.asked.append((self.label, [o.strip() for o in options]))
        want = self.picks[self.label]
        assert " " + want in options, (self.label, want, options)
        return " " + want, 2.5


def test_constrained_compiler_builds_the_rule_and_never_offers_a_number_that_is_not_in_the_clause():
    be = FakeBackend({"Defect": "open", "Measure": "size", "Number as written": "0.15", "Unit": "mm",
                      "Comparison": "more than", "Action": "reject"})
    r = ConstrainedSpecCompiler(be).compile("Reject any open wider than 0.15 mm or longer than 3 mm on a Class 2 board.", order_seed=3)
    assert r.error is None and (r.rule.defect, r.rule.measure, r.rule.op, r.rule.value, r.rule.action) == \
        ("open", "size_mm", ">", 0.15, "reject")
    offered = dict(be.asked)["Number as written"]
    assert sorted(offered) == ["0.15", "3"]          # numbers written in the clause; the IPC class number is not one
    assert r.answers["_margins"]["Defect"] == 2.5


def test_a_clause_without_a_number_can_only_be_an_always_rule():
    be = FakeBackend({"Defect": "short", "Action": "reject"})
    r = ConstrainedSpecCompiler(be).compile("Any short is a reject.")
    assert r.rule.measure == "always" and r.rule.defect == "short" and "Measure" not in dict(be.asked)


def test_grounded_numbers_skip_the_class_number_and_keep_units():
    from pcbinspect.spec_llm import grounded_numbers
    assert grounded_numbers("On Class 3 boards reject opens wider than 0.15 mm or nicks over 20%.") == \
        [("0.15", "mm"), ("20", "%")]
    assert grounded_numbers("More than two pin holes on a board: hold for review.") == [("two", None)]
    assert grounded_numbers("Spurs of 4 mil or more must be reworked.") == [("4", "mil")]


def test_grounded_compiler_only_offers_what_the_clause_supports():
    be = FakeBackend({"Measure": "width loss", "Comparison": "more than", "Action": "reject"})
    be.fed = []
    orig = be.feed
    be.feed = lambda t: (be.fed.append(t), orig(t))[1]
    r = ConstrainedSpecCompiler(be, grounded=True).compile(
        "Mouse bites may not remove more than 20% of the trace width on a Class 2 board.", order_seed=1)
    asked = dict(be.asked)
    assert "Defect" not in asked and "Number as written" not in asked and "Unit" not in asked   # nothing to decide
    assert sorted(asked["Measure"]) == ["spacing loss", "width loss"]
    assert (r.rule.defect, r.rule.measure, r.rule.op, r.rule.value, r.rule.action) == \
        ("mouse_bite", "width_loss_pct", ">", 20.0, "reject")
    assert " mouse bite" in be.fed and " percent" in be.fed and " 20" in be.fed          # forced answers enter the context


def test_grounded_compiler_reads_a_class_only_clause():
    be = FakeBackend({"Defect": "none"})
    r = ConstrainedSpecCompiler(be, grounded=True).compile("Build and inspect to Class 3.")
    assert dict(be.asked)["Defect"] == ["none", "any defect"] or sorted(dict(be.asked)["Defect"]) == ["any defect", "none"]
    assert r.rule.kind == "class" and r.rule.customer_class == 3
