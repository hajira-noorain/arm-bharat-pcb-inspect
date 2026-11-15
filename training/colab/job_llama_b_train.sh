#!/usr/bin/env bash
# Llama tuning, stage B (GPU): generate the data, LoRA-train Llama 3.2 1B Instruct (BF16), score it in BF16.
set -x
cd /content/code
python -m llm.data.gen_cause --n 6000 --out /content/ltdata
python -m llm.data.gen_sentences --n 3000 --out /content/ltdata
pip -q install peft
pip -q uninstall -y torchao   # Colab ships torchao 0.10; PEFT refuses to load next to it and training does not need it
python - <<'PY'
from huggingface_hub import snapshot_download as s
print(s("meta-llama/Llama-3.2-1B-Instruct", local_dir="/content/llama1b_bf16"))
PY
ls -la /content/llama1b_bf16 /content/llama1b_bf16/original
python llm/train_lora.py --data /content/ltdata --base /content/llama1b_bf16 --out /content/lora1b --epochs 2 --eval
echo STAGE_B_DONE
