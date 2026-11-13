"""Switching detector size swaps the model and its own thresholds together, or neither."""
from pcbinspect.pipeline import Design
from pcbinspect.severity import DesignRules
from pcbinspect.variants import apply_detector_variant


def designs():
    r = DesignRules(0.2, 0.2, 2, True)
    return {"deeppcb": Design("deeppcb", "bare", True, None, 0.02, 20.0, r, "models/gate/deeppcb.json",
                              gate_golden="models/gate/deeppcb_golden.json"),
            "asm": Design("asm", "assembled", False, "models/anomaly/asm", None, 20.0, r, None)}


def touch(root, *rels):
    for rel in rels:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")


def test_the_first_detector_is_the_default_and_nothing_changes(tmp_path):
    cfg = {"detector_pte": "models/vision/pcb_y26n_xnnpack.pte"}
    c, d, v = apply_detector_variant(cfg, designs(), tmp_path)
    assert v == "n" and c == cfg and d["deeppcb"].gate_file == "models/gate/deeppcb.json"


def test_a_complete_variant_swaps_model_and_thresholds_together(tmp_path):
    touch(tmp_path, "models/vision/pcb_y26s_xnnpack.pte", "models/gate/y26s/deeppcb.json",
          "models/gate/y26s/deeppcb_golden.json")
    ds = designs()
    c, d, v = apply_detector_variant({"detector_variant": "s"}, ds, tmp_path)
    assert v == "s" and c["detector_pte"] == "models/vision/pcb_y26s_xnnpack.pte"
    assert d["deeppcb"].gate_file == "models/gate/y26s/deeppcb.json"
    assert d["deeppcb"].gate_golden == "models/gate/y26s/deeppcb_golden.json"
    assert d["asm"] is ds["asm"]                                   # assembled boards do not use the detector
    assert ds["deeppcb"].gate_file == "models/gate/deeppcb.json"   # the input is left as it was


def test_a_variant_with_a_missing_threshold_file_is_not_used_at_all(tmp_path):
    touch(tmp_path, "models/vision/pcb_y26m_xnnpack.pte", "models/gate/y26m/deeppcb.json")   # no golden gate
    c, d, v = apply_detector_variant({"detector_variant": "m", "detector_pte": "n.pte"}, designs(), tmp_path)
    assert v == "n" and c["detector_pte"] == "n.pte" and d["deeppcb"].gate_file == "models/gate/deeppcb.json"


def test_a_variant_without_its_model_file_is_not_used(tmp_path):
    touch(tmp_path, "models/gate/y26s/deeppcb.json", "models/gate/y26s/deeppcb_golden.json")
    _, d, v = apply_detector_variant({"detector_variant": "s"}, designs(), tmp_path)
    assert v == "n" and d["deeppcb"].gate_file == "models/gate/deeppcb.json"


def test_an_unknown_variant_falls_back_to_the_first_detector(tmp_path):
    assert apply_detector_variant({"detector_variant": "xl"}, designs(), tmp_path)[2] == "n"
