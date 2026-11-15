"""Merge a PEFT LoRA adapter (trained on the Hugging Face Llama 3.2 weights) into Meta's original checkpoint
(consolidated.00.pth), so ExecuTorch's export_llm can quantise and export it like any Llama checkpoint. A merged
model runs at exactly the speed of a plain export: no adapter branch at run time.

The Hugging Face weights store the query and key projections with their rows reordered for its rotary-embedding
layout (transformers' convert_llama_weights_to_hf.py permutes them). A LoRA delta learned on those layouts must be
put back into Meta's order before it is added; every other projection maps one to one.

    python llm/merge_lora_meta.py --base <dir with consolidated.00.pth, params.json, tokenizer.model> \
        --adapter <PEFT adapter dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

META = {"self_attn.q_proj": "attention.wq", "self_attn.k_proj": "attention.wk", "self_attn.v_proj": "attention.wv",
        "self_attn.o_proj": "attention.wo", "mlp.gate_proj": "feed_forward.w1", "mlp.up_proj": "feed_forward.w3",
        "mlp.down_proj": "feed_forward.w2"}


def hf_to_meta_rows(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Inverse of transformers' permute(w, n_heads, dim1, dim2) for q/k: HF row order back to Meta's."""
    dim1, dim2 = w.shape
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def meta_to_hf_rows(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    dim1, dim2 = w.shape
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def merge(base: Path, adapter: Path, out: Path) -> dict:
    from safetensors.torch import load_file
    params = json.loads((base / "params.json").read_text())
    n_heads, n_kv = params["n_heads"], params.get("n_kv_heads", params["n_heads"])
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = torch.load(base / "consolidated.00.pth", map_location="cpu", weights_only=True)
    ad = load_file(str(adapter / "adapter_model.safetensors"))
    n = 0
    for key, a in ad.items():
        if ".lora_A." not in key:
            continue
        b = ad[key.replace(".lora_A.", ".lora_B.")]
        layer = int(key.split(".layers.")[1].split(".")[0])
        mod = next(m for m in META if f".{m}." in key)
        delta = (b.float() @ a.float()) * scale
        if mod == "self_attn.q_proj":
            delta = hf_to_meta_rows(delta, n_heads)
        elif mod == "self_attn.k_proj":
            delta = hf_to_meta_rows(delta, n_kv)
        mk = f"layers.{layer}.{META[mod]}.weight"
        assert sd[mk].shape == delta.shape, (mk, sd[mk].shape, delta.shape)
        sd[mk] = (sd[mk].float() + delta).to(sd[mk].dtype)
        n += 1
    out.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out / "consolidated.00.pth")
    for f in ("params.json", "tokenizer.model"):
        if (base / f).exists():
            shutil.copy(base / f, out / f)
    return {"merged_matrices": n, "scale": scale, "rank": cfg["r"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(merge(Path(a.base), Path(a.adapter), Path(a.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
