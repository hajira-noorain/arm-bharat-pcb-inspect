#!/usr/bin/env bash
# Llama tuning, stage B2 (GPU): a learning curve. Same LoRA recipe at three data sizes, each scored in BF16 on the
# held-out cause sets, so the data size is chosen from measurements. Needs /content/llama1b_bf16 (stage B).
#   nohup bash /content/code/training/colab/job_llama_b2_curve.sh > /content/run_b2c.log 2>&1 &
set -x
cd /content/code
pip -q install peft
pip -q uninstall -y torchao
for N in 3000 6000 12000; do
  D=/content/ltdata_$N
  python -m llm.data.gen_cause --n $N --out $D
  python -m llm.data.gen_sentences --n $((N / 2)) --out $D
  python llm/train_lora.py --data $D --base /content/llama1b_bf16 --out /content/lora1b_$N --epochs 2 --eval
  grep -h '"top1"' /content/lora1b_$N/cause_eval_bf16.json > /dev/null && echo "CURVE_$N done"
done
python - <<'PY'
import json
for n in (3000, 6000, 12000):
    try:
        d = json.load(open(f"/content/lora1b_{n}/cause_eval_bf16.json"))["sets"]
        print(n, {k: (round(v["top1"], 3), round(v["top2"], 3)) for k, v in d.items()})
    except Exception as e:
        print(n, "missing", e)
PY
echo STAGE_B2_DONE
