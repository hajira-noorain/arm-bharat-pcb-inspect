#!/usr/bin/env bash
# Export Meta's Llama 3.2 SpinQuant INT4 checkpoints to ExecuTorch .pte files (XNNPACK, KV cache, custom SDPA).
# Needs executorch==1.6.0 (the same version the Pi installs) and access to the gated meta-llama repos
# (accept the licence on Hugging Face; HF_TOKEN in the environment). The weights are never committed here.
#
#   HF_TOKEN=... bash llm/export_llama.sh 1B models/llm      # advice model  -> llama3_2_1b_spinquant.pte
#   HF_TOKEN=... bash llm/export_llama.sh 3B models/llm      # rule reader   -> llama3_2_3b_spinquant.pte
set -euo pipefail
SIZE=${1:-1B}; OUT=$(realpath -m "${2:-models/llm}"); PY=${PYTHON:-python}
size=$(echo "$SIZE" | tr 'A-Z' 'a-z')
REPO="meta-llama/Llama-3.2-${SIZE}-Instruct-SpinQuant_INT4_EO8"
SRC=$(mktemp -d)/ckpt
mkdir -p "$OUT"
$PY -c "from huggingface_hub import snapshot_download as s; s('$REPO', local_dir='$SRC')"
CFG=$($PY -c "import executorch, os; print(os.path.join(list(executorch.__path__)[0], 'examples/models/llama/config/llama_xnnpack_spinquant.yaml'))")   # namespace package: no __file__
cd "$OUT"
$PY -m executorch.extension.llm.export.export_llm --config "$CFG" +base.model_class=llama3_2 \
    +base.checkpoint="$SRC/consolidated.00.pth" +base.params="$SRC/params.json" \
    +export.output_name="llama3_2_${size}_spinquant.pte"
cp "$SRC/tokenizer.model" "$OUT/tokenizer.model"
rm -rf "$(dirname "$SRC")"
ls -la "$OUT"
