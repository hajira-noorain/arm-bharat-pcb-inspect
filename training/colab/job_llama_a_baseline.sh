#!/usr/bin/env bash
# Llama tuning, stage A (CPU): ExecuTorch 1.6 venv, the station's current Llama files from the private HF model repo,
# and the "before" numbers: prompted 1B and 3B SpinQuant on cause choice and fill-in sentences.
# HF_TOKEN comes from the Colab secret (set in the notebook's Python, inherited here); it is never printed.
set -x
P=/content/etv/bin/python
if [ ! -x $P ]; then
  pip -q install uv && uv venv -q /content/etv --python 3.12
  uv pip install -q --python $P executorch==1.6.0 pyyaml numpy tiktoken huggingface_hub
fi
$P - <<'PY'
from huggingface_hub import hf_hub_download as d
for f in ("models/llm/llama3_2_1b_spinquant.pte", "models/llm/llama3_2_3b_spinquant.pte", "models/llm/tokenizer.model"):
    print(d("Sulaiman-shariff/arm-bharat-pcb-inspect", f, local_dir="/content/hf"))
PY
E=$($P -c "import executorch;print(list(executorch.__path__)[0])")
cat $E/examples/models/llama/config/llama_xnnpack_spinquant.yaml
ls $E/examples/models/llama/config/
echo SETUP_DONE
cd /content/code
L=/content/hf/models/llm; mkdir -p /content/res
for M in 1b 3b; do
  $P llm/eval_cause.py --scorer llama --pte $L/llama3_2_${M}_spinquant.pte --tok $L/tokenizer.model --n 300 \
     --out /content/res/cause_prompted_${M}.json 2>&1 | grep -vE "Warning|stats.h|PyTorchObserver|^\s*$" | cut -c1-400
  $P llm/eval_sentences.py --pte $L/llama3_2_${M}_spinquant.pte --tok $L/tokenizer.model \
     --out /content/res/sentences_prompted_${M}.json 2>&1 | grep -vE "Warning|stats.h|PyTorchObserver|^\s*$" | cut -c1-400
  echo BASELINE_${M}_DONE
done
echo STAGE_A_DONE
