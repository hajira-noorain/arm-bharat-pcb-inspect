#!/usr/bin/env bash
# Held-out sets, run once with the method fixed on the development sets.
#   advice seed 2: the station's setting (3B, one-at-a-time causes) and the 1B text-runner baseline
#   shift questions: the 20 held-out questions, 3B scored choice
set -x
cd /content/code10 && mkdir -p /content/wadv
TOK=/content/llama1b/tokenizer.model; L=/content/llm
/content/etv/bin/python llm/eval_advice.py --pte $L/llama3_2_3b_spinquant.pte --tok $TOK --backend logits --cause-mode pointwise \
   --seed 2 --n 150 --out /content/wadv/advice_point_3b_seed2.json 2>&1 | grep -vE "^[0-9]+/150|Warning|stats.h|PyTorchObserver|^\s*$"
echo HELD_ADV3B_DONE
/content/etv/bin/python llm/eval_ask.py --pte $L/llama3_2_3b_spinquant.pte --tok $TOK --backend logits --sets heldout \
   --out /content/wadv/ask_logits3b_heldout.json 2>&1 | grep -vE "Warning|stats.h|PyTorchObserver|^\s*$"
echo HELD_ASK_DONE
/content/etv/bin/python llm/eval_advice.py --pte $L/llama3_2_1b_spinquant.pte --tok $TOK --backend runner \
   --seed 2 --n 150 --out /content/wadv/advice_runner_seed2.json 2>&1 | grep -vE "^[0-9]+/150|Warning|stats.h|PyTorchObserver|^\s*$"
echo HELD_RUNNER_DONE
