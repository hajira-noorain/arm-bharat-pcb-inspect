"""Training data for Llama's cause choice: sampled from the cause model, labelled with its exact posterior."""
import pytest

from llm.data.gen_cause import Generator, alt_examples, examples, training_rows
from pcbinspect.causes import CauseModel, features_from
from pcbinspect.kb import default_kb


@pytest.fixture(scope="module")
def kb():
    return default_kb()


def test_labels_are_the_exact_posterior_of_the_clues_the_example_shows(kb):
    cm = CauseModel(kb)
    for ex in examples(300, seed=1, split="train"):
        d = next(x for x in ex["record"]["defects"] if x["id"] == ex["defect_id"])
        assert ex["posterior"] == pytest.approx(cm.posterior(d["type"], features_from(d["evidence"])))
        assert abs(sum(ex["posterior"].values()) - 1) < 1e-9 and ex["true_cause"] in ex["posterior"]


def test_dev_has_its_own_unseen_combinations():
    train = {e["combo"] for e in examples(3000, seed=2, split="train")}
    dev = {e["combo"] for e in examples(300, seed=7, split="dev")}
    test = {e["combo"] for e in examples(300, seed=3, split="test")}
    assert dev and not (dev & train) and not (dev & test)


def test_held_out_clue_combinations_never_appear_in_training():
    train = {e["combo"] for e in examples(3000, seed=2, split="train")}
    test = {e["combo"] for e in examples(3000, seed=3, split="test")}
    assert test and not (train & test)


def test_same_seed_same_data_and_every_defect_type_is_covered(kb):
    a, b = examples(200, seed=4, split="train"), examples(200, seed=4, split="train")
    assert a == b
    types = {next(x for x in e["record"]["defects"] if x["id"] == e["defect_id"])["type"]
             for e in examples(3000, seed=5, split="train")}
    assert types == set(kb.defects)


def test_training_rows_one_yes_no_question_per_candidate_cause(kb):
    ex = examples(5, seed=6, split="train")[0]
    rows = training_rows(ex, kb)
    assert len(rows) == len(ex["posterior"])
    assert all(r["prompt"].endswith("Answer:") and 0 <= r["p_yes"] <= 1 for r in rows)
    assert sorted(r["p_yes"] for r in rows) == pytest.approx(sorted(ex["posterior"].values()))


def test_the_second_generator_is_built_differently_and_only_for_testing(kb):
    alt = alt_examples(400, seed=7)
    assert len(alt) == 400 and all(e["true_cause"] in e["posterior"] for e in alt)
    # its clues are correlated through a hidden fault strength, so naive Bayes is not exact on it
    assert Generator.__name__ and alt[0]["generator"] == "alt"


def test_generated_clues_include_the_new_ones_with_matching_unknowns():
    exs = examples(400, seed=8, split="train")
    evs = [d["evidence"] for e in exs for d in e["record"]["defects"]]
    assert any(isinstance(ev.get("repeat_in_line"), dict) for ev in evs)
    assert any(ev.get("central") is True for ev in evs) and any(ev.get("roll_pitch") is True for ev in evs)
    for ev in evs:      # no history means neither repeat can be known
        assert (ev["repeat_same_spot"] == "unknown") == (ev["repeat_in_line"] == "unknown")
