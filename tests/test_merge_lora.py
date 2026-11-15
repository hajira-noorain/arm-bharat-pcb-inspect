"""The q/k row reordering between Hugging Face and Meta Llama checkpoints, and the merge itself."""
import json

import torch

from llm.merge_lora_meta import hf_to_meta_rows, merge, meta_to_hf_rows


def test_row_reordering_round_trips_and_is_not_the_identity():
    w = torch.arange(64 * 8, dtype=torch.float32).reshape(64, 8)
    hf = meta_to_hf_rows(w, n_heads=4)
    assert not torch.equal(hf, w)
    assert torch.equal(hf_to_meta_rows(hf, n_heads=4), w)


def test_merge_adds_the_delta_in_meta_order(tmp_path):
    from safetensors.torch import save_file
    torch.manual_seed(0)
    dim, heads, r = 16, 2, 4
    base = tmp_path / "base"
    base.mkdir()
    sd = {"layers.0.attention.wq.weight": torch.randn(dim, dim), "layers.0.feed_forward.w2.weight": torch.randn(dim, 32)}
    torch.save(sd, base / "consolidated.00.pth")
    (base / "params.json").write_text(json.dumps({"n_heads": heads, "n_kv_heads": heads, "dim": dim}))
    ad = tmp_path / "ad"
    ad.mkdir()
    A, B = torch.randn(r, dim), torch.randn(dim, r)
    A2, B2 = torch.randn(r, 32), torch.randn(dim, r)
    save_file({"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": A,
               "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": B,
               "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": A2,
               "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": B2}, str(ad / "adapter_model.safetensors"))
    (ad / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": 2 * r}))
    info = merge(base, ad, tmp_path / "out")
    got = torch.load(tmp_path / "out" / "consolidated.00.pth", weights_only=True)
    assert info["merged_matrices"] == 2
    # HF q = permute(meta q); after merging, HF view of the merged weight = HF base + LoRA delta
    assert torch.allclose(meta_to_hf_rows(got["layers.0.attention.wq.weight"], heads),
                          meta_to_hf_rows(sd["layers.0.attention.wq.weight"], heads) + 2 * (B @ A), atol=1e-5)
    assert torch.allclose(got["layers.0.feed_forward.w2.weight"], sd["layers.0.feed_forward.w2.weight"] + 2 * (B2 @ A2),
                          atol=1e-5)
