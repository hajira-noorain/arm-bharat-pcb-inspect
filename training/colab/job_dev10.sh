#!/usr/bin/env bash
# Development-set rerun after the summary fixes: advice (3B, one-at-a-time causes, seed 1) then shift questions (tuning).
set -x
rm -rf /content/code10 && mkdir -p /content/code10 && unzip -q -o /content/code10.zip -d /content/code10
cd /content/code10 && mkdir -p /content/wadv
TOK=/content/llama1b/tokenizer.model; PTE=/content/llm/llama3_2_3b_spinquant.pte
/content/etv/bin/python llm/eval_advice.py --pte $PTE --tok $TOK --backend logits --cause-mode pointwise --seed 1 --n 150 \
   --out /content/wadv/advice_point_3b_seed1_v2.json 2>&1 | grep -vE "^[0-9]+/150|Warning|stats.h|PyTorchObserver|^\s*$"
echo ADVP2_DONE_dev
/content/etv/bin/python llm/eval_ask.py --pte $PTE --tok $TOK --backend logits --sets tuning \
   --out /content/wadv/ask_logits3b_tuning_v2.json 2>&1 | grep -vE "Warning|stats.h|PyTorchObserver|^\s*$"
echo ASK2_DONE_dev
