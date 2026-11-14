#!/usr/bin/env bash
# Shift questions with the 3B model choosing the query (scored choice). $1 = sets, $2 = log line to wait for.
SETS=${1:-tuning}; WAIT=${2:-ADVP_DONE_3b_1}
until grep -q "$WAIT" /content/run_a*.log 2>/dev/null; do sleep 20; done
set -x
cd /content/code9
/content/etv/bin/python llm/eval_ask.py --pte /content/llm/llama3_2_3b_spinquant.pte --tok /content/llama1b/tokenizer.model \
  --backend logits --sets $SETS --out /content/wadv/ask_logits3b_$(echo $SETS | tr ',' '_').json 2>&1 | grep -vE "Warning|stats.h|PyTorchObserver|^\s*$"
echo ASK3B_DONE_$SETS
