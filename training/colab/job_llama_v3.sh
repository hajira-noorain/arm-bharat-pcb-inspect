#!/usr/bin/env bash
# Llama tuning v3, end to end on Colab: data with all nine clues and the "why" line, LoRA training (GPU), scoring in
# BF16, merge into Meta's checkpoint, 8da4w export, and scoring of the exported .pte exactly as the station runs it.
# Size from the learning curve (job_llama_b2_curve.sh): 6,000 cause examples reached the ceiling and 12,000 added
# nothing; 9,000 here because the extra clues make more combinations. Needs /content/llama1b_bf16 and /content/etv.
#   nohup bash /content/code/training/colab/job_llama_v3.sh > /content/run_v3.log 2>&1 &
set -x
cd /content/code
pip -q install peft
pip -q uninstall -y torchao
D=/content/ltdata_v3
python -m llm.data.gen_cause --n 9000 --out $D
python -m llm.data.gen_sentences --n 4500 --out $D
python llm/train_lora.py --data $D --base /content/llama1b_bf16 --out /content/lora1b_v3 --epochs 2 --eval
echo TRAIN_V3_DONE
P=/content/etv/bin/python
E=$($P -c "import executorch;print(list(executorch.__path__)[0])")
$P llm/merge_lora_meta.py --base /content/llama1b_bf16/original --adapter /content/lora1b_v3 --out /content/merged1b_v3
mkdir -p /content/pte && cd /content/pte
$P -m executorch.extension.llm.export.export_llm --config $E/examples/models/llama/config/llama_q8da4w.yaml \
  +base.model_class=llama3_2 +base.checkpoint=/content/merged1b_v3/consolidated.00.pth \
  +base.params=/content/merged1b_v3/params.json +model.use_kv_cache=True +model.use_sdpa_with_kv_cache=True \
  +backend.xnnpack.enabled=True +backend.xnnpack.extended_ops=True quantization.group_size=32 \
  +export.max_seq_length=2048 +export.max_context_length=2048 +export.output_name=llama3_2_1b_pcb_lora_v3.pte \
  2>&1 | grep -vE "Warning|warn" | tail -n 5
ls -la /content/pte/llama3_2_1b_pcb_lora_v3.pte
echo EXPORT_V3_DONE
cd /content/code
TOK=/content/hf/models/llm/tokenizer.model
mkdir -p /content/res_v3
$P llm/eval_cause.py --scorer llama --pte /content/pte/llama3_2_1b_pcb_lora_v3.pte --tok $TOK --n 300 \
   --out /content/res_v3/cause_1b_pcb_lora_v3.json 2>&1 | grep -vE "Warning|stats.h|PyTorchObserver|^\s*$" | cut -c1-400
$P llm/eval_sentences.py --pte /content/pte/llama3_2_1b_pcb_lora_v3.pte --tok $TOK \
   --out /content/res_v3/sentences_1b_pcb_lora_v3.json 2>&1 | grep -vE "Warning|stats.h|PyTorchObserver|^\s*$" | cut -c1-400
python llm/eval_cause.py --scorer bayes --n 1000 --out /content/res_v3/cause_bayes.json | tail -n 3
echo STAGE_V3_DONE
