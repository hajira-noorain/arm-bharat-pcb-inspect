"""Advice-quality evaluation on a fixed, seeded set of realistic defect records.

For each record the full advisor path runs (prompt -> Llama on ExecuTorch -> grounding checker).
Reported:
  accepted_rate         share of Llama answers that pass every grounding rule (shown to operators)
  rejection_reasons     why the rest were replaced by template advice (the hallucination profile)
  cause_position1_share how often option 1 is chosen vs chance (options are shuffled per board)
  repeating_defect_process_cause / one_off_defect_handling_cause
                        reasoning check against chance: a defect that keeps repeating should get a
                        process cause, a one-off should lean to handling/local causes
  (step order and coverage are by construction and are NOT reported as model skill)
  prompt_tokens, prefill/decode tok/s, latency
All raw outputs are saved for human review.

python llm/eval_advice.py --pte ... --tok ... --n 60 --out results/advice_eval_1b.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.advisor import Advisor  # noqa: E402
from pcbinspect.kb import default_kb  # noqa: E402
from pcbinspect.record import BARE_CLASSES, Defect  # noqa: E402
from pcbinspect.severity import SeverityEngine  # noqa: E402

LOCS = ["top-left", "top-centre", "top-right", "middle-left", "centre", "middle-right",
        "bottom-left", "bottom-centre", "bottom-right"]


def make_records(n: int, seed: int = 0) -> list[tuple[dict, dict]]:
    rng = random.Random(seed)
    sev = SeverityEngine()
    classes = list(BARE_CLASSES) + ["unknown_anomaly"]
    out = []
    for i in range(n):
        k = rng.choice([1, 1, 2, 2, 3])
        defects = []
        for j in range(k):
            cls = rng.choice(classes)
            w = round(rng.uniform(0.01, 0.12), 3)
            d = Defect(f"D{j + 1}", cls, "anomaly" if cls == "unknown_anomaly" else "detector",
                       (0, 0, 10, 10), round(rng.uniform(0.35, 0.98), 2), size_mm=round(w * rng.uniform(1.5, 4), 2),
                       width_mm=w, area_mm2=round(w * w * 3, 3), location=rng.choice(LOCS))
            defects.append(sev.grade(d))
        record = {"board_id": f"B-{1000 + i}", "design": "eval", "verdict": "FAIL",
                  "defects": [{"id": d.id, "type": d.cls, "severity": d.severity,
                               "confidence": d.confidence, "size_mm": d.size_mm,
                               "location": d.location, "ipc": d.ipc} for d in defects]}
        history = {}
        if rng.random() < 0.5:   # half the boards come from a line where one defect keeps repeating
            history[rng.choice(defects).cls] = rng.randint(3, 8)
        out.append((record, history))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pte", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fewshot", action="store_true")
    args = ap.parse_args()
    from pcbinspect.llm import ExecuTorchLlama
    kb = default_kb()
    llm = ExecuTorchLlama(args.pte, args.tok)
    adv = Advisor(llm, kb, fewshot=args.fewshot)
    rows = []
    t0 = time.time()
    for i, (rec, hist) in enumerate(make_records(args.n)):
        a = adv.advise(rec, hist)
        accepted = a.backend == "llama-executorch"
        first = a.steps[0]["defect"] if a.steps else None

        rows.append({"record": rec, "history": hist, "accepted": accepted, "raw": a.raw_output,
                     "reasons": a.rejected_reasons, "filled": a.filled_from_template,
                     "summary": a.summary, "first_step": first,
                     "choices": a.choices,
                     "prompt_tokens": a.gen.prompt_tokens if a.gen else None,
                     "latency_ms": a.gen.latency_ms if a.gen else None,
                     "prefill_tps": a.gen.prefill_tps if a.gen else None,
                     "decode_tps": a.gen.decode_tps if a.gen else None})
        print(f"{i + 1}/{args.n} accepted={accepted} {a.rejected_reasons[:1]} ({time.time() - t0:.0f}s)", flush=True)
    acc = [r for r in rows if r["accepted"]]
    reasons = Counter(r["reasons"][0].split(":")[0] if ":" in r["reasons"][0] else r["reasons"][0]
                      for r in rows if not r["accepted"] and r["reasons"])
    med = lambda xs: statistics.median([x for x in xs if x]) if any(xs) else None  # noqa: E731
    # ---- choice analysis (all parsed choices, accepted or not) ----
    choices = [c for r in rows for c in r["choices"]]
    pos1 = sum(c["cause_pos"] == 1 for c in choices)
    pos1_chance = sum(1 / c["n_causes"] for c in choices)

    def category_stats(want_process: bool, repeat: bool):
        hit = chance = n = 0
        for r in rows:
            rep_types = set(r["history"])
            for c in r["choices"]:
                causes = kb.entry(c["type"])["causes"]
                proc = [x for x in causes if x.get("step") != "handling"]
                if len(proc) in (0, len(causes)):
                    continue          # all causes in one category: uninformative
                if (c["type"] in rep_types) != repeat:
                    continue
                is_proc = kb.by_id[c["cause"]].get("step") != "handling"
                hit += is_proc if want_process else (not is_proc)
                chance += (len(proc) if want_process else len(causes) - len(proc)) / len(causes)
                n += 1
        return {"n": n, "rate": hit / n if n else None, "chance": chance / n if n else None}

    summary = {
        "model": Path(args.pte).name, "fewshot": args.fewshot, "n": len(rows), "load_ms": llm.load_ms,
        "accepted_rate": len(acc) / len(rows),
        "rejection_reasons": dict(reasons),
        "choices_parsed": len(choices),
        "cause_position1_share": pos1 / max(len(choices), 1),
        "cause_position1_chance": pos1_chance / max(len(choices), 1),
        "repeating_defect_process_cause": category_stats(True, True),
        "one_off_defect_handling_cause": category_stats(False, False),
        "prompt_tokens_median": med([r["prompt_tokens"] for r in rows]),
        "latency_ms_median": med([r["latency_ms"] for r in rows]),
        "prefill_tps_median": med([r["prefill_tps"] for r in rows]),
        "decode_tps_median": med([r["decode_tps"] for r in rows]),
        "note": "step order (most severe first) and coverage (every defect gets a step) are by construction",
    }
    print(json.dumps(summary, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
