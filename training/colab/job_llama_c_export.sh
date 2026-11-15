#!/usr/bin/env bash
# Llama tuning, stage C (CPU): merge the LoRA adapter into Meta's checkpoint, export to ExecuTorch with 8da4w
# (8-bit dynamic activations, 4-bit weights, group size 32, XNNPACK, KV cache), and measure the exported file with
# the same evaluations as the prompted baselines. Also exports the untuned BF16 1B the same way, to separate the
# effect of tuning from the effect of our own quantisation (Meta's SpinQuant file is not used here).
set -x
P=/content/etv/bin/python
E=$($P -c "import executorch;print(list(executorch.__path__)[0])")
cd /content/code
$P -m pip -q install safetensors 2>/dev/null || uv pip install -q --python $P safetensors
$P llm/merge_lora_meta.py --base /content/llama1b_bf16/original --adapter /content/lora1b --out /content/merged1b
mkdir -p /content/pte && cd /content/pte
export_one() {   # $1 checkpoint dir, $2 output name
  $P -m executorch.extension.llm.export.export_llm --config $E/examples/models/llama/config/llama_q8da4w.yaml \
    +base.model_class=llama3_2 +base.checkpoint=$1/consolidated.00.pth +base.params=$1/params.json \
    +model.use_kv_cache=True +model.use_sdpa_with_kv_cache=True +backend.xnnpack.enabled=True \
    +backend.xnnpack.extended_ops=True quantization.group_size=32 +export.max_seq_length=2048 \
    +export.max_context_length=2048 +export.output_name=$2 2>&1 | grep -vE "Warning|warn" | tail -n 5
  ls -la /content/pte/$2
}
export_one /content/merged1b llama3_2_1b_pcb_lora.pte
export_one /content/llama1b_bf16/original llama3_2_1b_base_8da4w.pte
echo EXPORT_DONE
cd /content/code
TOK=/content/hf/models/llm/tokenizer.model
for M in pcb_lora base_8da4w; do
  $P llm/eval_cause.py --scorer llama --pte /content/pte/llama3_2_1b_$M.pte --tok $TOK --n 300 \
     --out /content/res/cause_1b_$M.json 2>&1 | grep -vE "Warning|stats.h|PyTorchObserver|^\s*$" | cut -c1-400
  $P llm/eval_sentences.py --pte /content/pte/llama3_2_1b_$M.pte --tok $TOK \
     --out /content/res/sentences_1b_$M.json 2>&1 | grep -vE "Warning|stats.h|PyTorchObserver|^\s*$" | cut -c1-400
  echo EVAL_${M}_DONE
done
echo STAGE_C_DONE
