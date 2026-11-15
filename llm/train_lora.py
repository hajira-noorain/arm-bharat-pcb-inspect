"""LoRA fine-tuning of Llama 3.2 1B Instruct (BF16) for the station's two Llama jobs, on a GPU (Colab).

Task 1, cause choice: the evidence prompt plus one yes/no question per candidate cause (pcbinspect/cause_prompt.py).
  The target is the probability of "yes": the exact posterior of that cause given the clues
  (llm/data/gen_cause.py). Loss: cross-entropy between that target and the model's two-way yes/no distribution,
  where "yes" and "no" each pool the four spellings the station reads (pcbinspect/llm_logits.py).
Task 2, fill-in sentences: the facts prompt (pcbinspect/sentences.py) and a correct fill-in sentence; ordinary
  next-token loss on the sentence only (llm/data/gen_sentences.py).

The adapter is saved in PEFT format (adapter_model.safetensors + adapter_config.json), which ExecuTorch's
export_llm reads (base.adapter_checkpoint / base.adapter_config). Measures on the dev split after each epoch and
keeps the best; --eval also scores the held-out cause sets in BF16 with the same metrics as llm/eval_cause.py.

    python llm/train_lora.py --data <dir with train.jsonl, dev.jsonl, sentences_train.jsonl, sentences_dev.jsonl> \
        --out <adapter dir> [--epochs 2 --rank 16 --lr 2e-4] [--eval]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm.data.gen_cause import training_rows  # noqa: E402
from pcbinspect.kb import default_kb  # noqa: E402

BASE = "meta-llama/Llama-3.2-1B-Instruct"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EOT = "<|eot_id|>"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def yes_no_ids(tok):
    ids = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})
    return ids((" yes", " Yes", "yes", "Yes")), ids((" no", " No", "no", "No"))


def cause_items(exs, kb):
    return [{"kind": "cause", "text": r["prompt"], "p": r["p_yes"]} for e in exs for r in training_rows(e, kb)]


def sentence_items(rows):
    return [{"kind": "sent", "prompt": r["prompt"], "completion": r["completion"] + EOT} for r in rows]


def batches(items, size, rng):
    by = {"cause": [i for i in items if i["kind"] == "cause"], "sent": [i for i in items if i["kind"] == "sent"]}
    out = []
    for kind, xs in by.items():
        rng.shuffle(xs)
        out += [xs[k:k + size] for k in range(0, len(xs), size)]
    rng.shuffle(out)
    return out


def encode(batch, tok, device, max_len=1024):
    pad = tok.pad_token_id
    if batch[0]["kind"] == "cause":
        seqs = [tok.encode(b["text"], add_special_tokens=False)[-max_len:] for b in batch]
        labels = None
    else:
        seqs, labels = [], []
        for b in batch:
            p = tok.encode(b["prompt"], add_special_tokens=False)
            c = tok.encode(b["completion"], add_special_tokens=False)
            s = (p + c)[-max_len:]
            seqs.append(s)
            labels.append(([-100] * len(p) + c)[-max_len:])
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    lab = torch.full((len(seqs), n), -100, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
        if labels is not None:
            lab[i, :len(s)] = torch.tensor(labels[i])
    last = att.sum(1) - 1
    return ids.to(device), att.to(device), lab.to(device), last.to(device)


def loss_of(model, batch, tok, yes, no, device):
    ids, att, lab, last = encode(batch, tok, device)
    logits = model(input_ids=ids, attention_mask=att).logits.float()
    if batch[0]["kind"] == "cause":
        lg = logits[torch.arange(len(batch), device=device), last]
        two = torch.stack([torch.logsumexp(lg[:, yes], -1), torch.logsumexp(lg[:, no], -1)], -1)
        lp = torch.log_softmax(two, -1)
        p = torch.tensor([b["p"] for b in batch], device=device)
        return -(p * lp[:, 0] + (1 - p) * lp[:, 1]).mean()
    return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1),
                                             ignore_index=-100)


@torch.no_grad()
def dev_loss(model, items, tok, yes, no, device, size=16):
    model.eval()
    tot = {"cause": [0.0, 0], "sent": [0.0, 0]}
    for b in batches(items, size, random.Random(0)):
        tot[b[0]["kind"]][0] += float(loss_of(model, b, tok, yes, no, device)) * len(b)
        tot[b[0]["kind"]][1] += len(b)
    model.train()
    return {k: v[0] / max(v[1], 1) for k, v in tot.items()}


def make_hf_scorer(model, tok, yes, no, device):
    """Probabilities over a defect's causes from the fine-tuned model in BF16, as the station computes them."""
    from pcbinspect.cause_prompt import build_evidence_prompt, cause_question

    @torch.no_grad()
    def probs(ex, cm, kb):
        d = next(d for d in ex["record"]["defects"] if d["id"] == ex["defect_id"])
        causes = cm.causes(d["type"])
        prompt = build_evidence_prompt(ex["record"], kb, variant=0)
        texts = [{"kind": "cause", "text": prompt + cause_question(d["id"], c, kb), "p": 0.0} for c in causes]
        ids, att, _, last = encode(texts, tok, device)
        lg = model(input_ids=ids, attention_mask=att).logits.float()[torch.arange(len(texts), device=device), last]
        lo = (lg[:, yes].max(-1).values - lg[:, no].max(-1).values).tolist()   # the station's log-odds
        m = max(lo)
        e = [math.exp(v - m) for v in lo]
        return {c: v / sum(e) for c, v in zip(causes, e)}
    return probs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval", action="store_true", help="score the held-out cause sets with the best adapter")
    a = ap.parse_args()
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    d, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kb = default_kb()
    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.convert_tokens_to_ids("<|finetune_right_pad_id|>")
    yes, no = yes_no_ids(tok)
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16).to(device)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    train = cause_items(load_jsonl(d / "train.jsonl"), kb) + sentence_items(load_jsonl(d / "sentences_train.jsonl"))
    dev = cause_items(load_jsonl(d / "dev.jsonl"), kb) + sentence_items(load_jsonl(d / "sentences_dev.jsonl"))
    print(f"train items {len(train)}, dev items {len(dev)}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    rng = random.Random(0)
    steps_total = a.epochs * math.ceil(len(train) / a.batch) + 2
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps_total, pct_start=0.05)
    log = {"args": vars(a), "epochs": []}
    best = None
    t0 = time.perf_counter()
    base_dev = dev_loss(model, dev, tok, yes, no, device)
    print("before training, dev loss", base_dev, flush=True)
    log["dev_before"] = base_dev
    for ep in range(a.epochs):
        for i, b in enumerate(batches(train, a.batch, rng)):
            loss = loss_of(model, b, tok, yes, no, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if i % 100 == 0:
                print(f"epoch {ep} step {i} {b[0]['kind']} loss {float(loss):.4f} {time.perf_counter() - t0:.0f}s",
                      flush=True)
        dl = dev_loss(model, dev, tok, yes, no, device)
        score = dl["cause"] + dl["sent"]
        log["epochs"].append({"epoch": ep, "dev": dl, "seconds": round(time.perf_counter() - t0)})
        print("epoch", ep, "dev loss", dl, flush=True)
        if best is None or score < best:
            best = score
            model.save_pretrained(out)
            print("saved best adapter", flush=True)
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    if a.eval:
        from llm.eval_cause import real_examples, score
        from llm.data.gen_cause import alt_examples, examples
        from pcbinspect.causes import CauseModel
        base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16).to(device)
        model = PeftModel.from_pretrained(base, out).to(device).eval()
        fn, cm = make_hf_scorer(model, tok, yes, no, device), CauseModel(kb)
        res = {"scorer": "tuned-1b-bf16", "sets": {}}
        for name, exs in (("test", examples(600, 13, "test", kb)), ("alt", alt_examples(600, 14, kb)),
                          ("real", real_examples(kb))):
            res["sets"][name] = score(exs, fn, cm, kb)
            print(name, json.dumps(res["sets"][name]), flush=True)
        (out / "cause_eval_bf16.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
