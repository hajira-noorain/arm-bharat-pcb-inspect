"""Which Llama file the station picks for advice (no model is loaded here)."""
from pcbinspect.llm_select import advice_pte, load_llama


def test_the_advice_model_is_used_when_present_and_the_small_model_otherwise(tmp_path):
    cfg = {"llm_pte": "small.pte", "advice_llm_pte": "big.pte", "llm_tokenizer": "tok.model"}
    assert advice_pte(tmp_path, cfg) is None and load_llama(tmp_path, cfg) is None     # nothing there: templates
    (tmp_path / "small.pte").write_bytes(b"x")
    assert advice_pte(tmp_path, cfg).name == "small.pte"
    (tmp_path / "big.pte").write_bytes(b"x")
    assert advice_pte(tmp_path, cfg).name == "big.pte"
    assert advice_pte(tmp_path, {"llm_pte": "small.pte"}).name == "small.pte"         # no advice model configured
