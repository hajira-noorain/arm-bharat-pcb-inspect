"""Training data for fill-in sentences: every target must pass the station's own checker."""
from llm.data.gen_sentences import QIDS, examples, heldout_questions
from pcbinspect.sentences import check_template, required_fields


def test_naming_a_defect_type_requires_its_count():
    assert required_fields("How many shorts did we get in the last 8 hours?", "defect_counts") == {"count_short"}
    assert required_fields("Are we seeing more opens or more mouse bites?", "defect_counts") == \
        {"count_open", "count_mouse_bite"}


def test_every_target_passes_the_checker_and_every_query_is_covered():
    exs = examples(1500, seed=1)
    assert {e["qid"] for e in exs} == set(QIDS)
    for e in exs:
        assert check_template(e["target"], e["question"], e["qid"], e["facts"]) == [], e
    assert any(e["facts"].get("review") == 0 for e in exs if e["qid"] == "yield")            # zeros
    assert any(e["facts"].get("top_count") == e["facts"].get("second_count") for e in exs
               if e["qid"] == "defect_counts")                                              # ties


def test_held_out_questions_are_never_in_training():
    train_q = {e["question"] for e in examples(3000, seed=2)}
    assert heldout_questions() and not (train_q & set(heldout_questions()))
