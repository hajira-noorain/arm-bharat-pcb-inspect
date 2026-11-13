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


def test_full_sequence_scores_use_every_token_and_rewind_between_options():
    """Option A = [1, 2], B = [1, 3], C = [4]. P(1)=0.5, P(4)=0.5 at the start; after 1: P(2)=0.9, P(3)=0.1.
    A first-token choice cannot tell A from B; the full score must (A 0.45, B 0.05, C 0.5)."""
    from pcbinspect.llm_logits import score_sequences
    calls = []

    def step(off, tok):
        calls.append((off, tok))
        lg = np.full(V, -1e9, np.float32)
        if tok is None:
            lg[1] = lg[4] = 0.0
        elif tok == 1:
            lg[2], lg[3] = np.log(0.9), np.log(0.1)
        return lg

    s = score_sequences(step, [[1, 2], [1, 3], [4]])
    assert [n for _, n in s] == [2, 2, 1]
    assert np.exp([lp for lp, _ in s]) == pytest.approx([0.45, 0.05, 0.5], rel=1e-4)
    assert calls == [(0, None), (0, 1)]                       # token 1 is evaluated once for both A and B


def test_scores_can_include_the_probability_of_stopping_after_the_option():
    from pcbinspect.llm_logits import score_sequences

    def step(off, tok):
        lg = np.full(V, -1e9, np.float32)
        if tok is None:
            lg[1] = 0.0
        elif tok == 1:                                        # after "spur": 80% stop, 20% go on to "ious copper"
            lg[9], lg[2] = np.log(0.8), np.log(0.2)
        elif tok == 2:
            lg[9] = 0.0
        return lg

    short, long_ = score_sequences(step, [[1], [1, 2]], end=[9])
    assert np.exp(short[0]) == pytest.approx(0.8, rel=1e-4) and np.exp(long_[0]) == pytest.approx(0.2, rel=1e-4)


def test_summary_decoding_refuses_numbers_that_are_not_in_the_findings():
    from pcbinspect.grounding import number_filter
    record = {"board_id": "B-7", "defects": [{"id": "D1", "type": "short", "size_mm": 0.18, "confidence": 0.87}]}
    allow = number_filter(record, extra_numbers=[5, 20])
    assert allow("The short is ", "0") and allow("The short is 0", ".") and allow("The short is 0.", "18")
    assert not allow("The short is 0.", "25")                 # 0.25 is not a size on this board
    assert allow("seen on ", "5") and allow("of the last ", "20") and allow("confidence ", "87")
    assert not allow("seen on ", "3") and not allow("about ", "12")
    assert not allow("seen on 5", "0")                        # 50 would not be a fact either
    assert allow("Defect D", "1") and allow("quarantine the board", " and")     # ids and words are free


def test_a_summary_cannot_end_before_it_says_anything():
    from pcbinspect.llm_logits import SPECIAL, LogitLlama
    words = {1: "\n", 2: " ", 3: "Board", 4: " fails", 5: "."}
    eot = SPECIAL["<|eot_id|>"]

    class Tok:
        def decode(self, ids):
            return "".join(words.get(i, "?") for i in ids)

    def logits(prefs):
        lg = np.full(128256, -9.0, np.float32)
        for t, v in prefs.items():
            lg[t] = v
        return lg
    # the model wants to stop at once (end token, then newline, then a space); afterwards it writes a sentence
    script = [logits({eot: 9, 1: 8, 2: 7, 3: 6}), logits({4: 9}), logits({5: 9}), logits({1: 9})]
    be = LogitLlama.__new__(LogitLlama)
    be.tok, be._last, fed = Tok(), script[0], []

    def forward(toks):
        fed.extend(toks)
        be._last = script[min(len(fed), len(script) - 1)]
    be._forward = forward
    assert be.write(10, stop=("\n",), nonempty=True) == "Board fails."
    be._last, fed[:] = script[0], []
    assert be.write(10, stop=("\n",)) == ""                   # without the option the old behaviour stands


def test_a_summary_cannot_end_on_a_number_that_is_not_a_fact():
    from pcbinspect.grounding import number_filter_values
    from pcbinspect.llm_logits import SPECIAL, LogitLlama
    words = {3: "Failed: ", 4: "1", 5: "0", 6: " boards."}
    eot = SPECIAL["<|eot_id|>"]

    class Tok:
        def decode(self, ids):
            return "".join(words.get(i, "?") for i in ids)

    def logits(prefs):
        lg = np.full(128256, -9.0, np.float32)
        for t, v in prefs.items():
            lg[t] = v
        return lg
    # after "1" the model wants to stop, but 1 is not in the rows (10 is): it has to finish the number
    script = [logits({3: 9}), logits({4: 9}), logits({eot: 9, 5: 5}), logits({6: 9}), logits({eot: 9})]
    be = LogitLlama.__new__(LogitLlama)
    be.tok, be._last, fed = Tok(), script[0], []

    def forward(toks):
        fed.extend(toks)
        be._last = script[min(len(fed), len(script) - 1)]
    be._forward = forward
    assert be.write(10, stop=(), allow=number_filter_values({10.0})) == "Failed: 10 boards."
