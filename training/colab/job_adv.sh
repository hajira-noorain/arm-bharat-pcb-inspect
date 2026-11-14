#!/usr/bin/env bash
# Llama 3.2 1B advice on ExecuTorch: text runner with numbered options vs scored option text.
# $1 = record-set seed (1 = development, 2 = held-out), $2 = variants to run (space separated).
set -x
SEED=${1:-1}; VARIANTS=${2:-"runner logits_sum logits_mean"}
P=/content/etv/bin/python
if [ ! -x $P ]; then
  pip -q install uv && uv venv -q /content/etv --python 3.12
  uv pip install -q --python $P executorch==1.6.0 pyyaml numpy huggingface_hub tiktoken
fi
if [ ! -f /content/llm/llama3_2_1b_spinquant.pte ]; then
  $P -c "from huggingface_hub import snapshot_download as s; s('meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8', local_dir='/content/llama1b')"
  CFG=$($P -c "import executorch, os; print(os.path.join(list(executorch.__path__)[0], 'examples/models/llama/config/llama_xnnpack_spinquant.yaml'))")
  mkdir -p /content/llm && cd /content/llm
  $P -m executorch.extension.llm.export.export_llm --config "$CFG" +base.model_class=llama3_2 \
    +base.checkpoint=/content/llama1b/consolidated.00.pth +base.params=/content/llama1b/params.json \
    +export.output_name=llama3_2_1b_spinquant.pte 2>&1 | tail -n 2
fi
ls -la /content/llm | grep pte
rm -rf /content/code7 && mkdir -p /content/code7 && unzip -q -o /content/code7.zip -d /content/code7
cd /content/code7 && mkdir -p /content/wadv
for V in $VARIANTS; do
  case $V in
    runner)      ARGS="--backend runner" ;;
    logits_sum)  ARGS="--backend logits --score-norm sum" ;;
    logits_mean) ARGS="--backend logits --score-norm mean" ;;
  esac
  echo "VARIANT $V seed $SEED"
  $P llm/eval_advice.py --pte /content/llm/llama3_2_1b_spinquant.pte --tok /content/llama1b/tokenizer.model \
     $ARGS --seed $SEED --n 150 --out /content/wadv/advice_${V}_seed${SEED}.json 2>&1 | grep -vE "^[0-9]+/150|Warning|stats.h|PyTorchObserver|^\s*$"
done
echo ADV_DONE_$SEED
