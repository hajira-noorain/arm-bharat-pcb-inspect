"""Benchmark: plain-language acceptance clause -> executable rule. Llama (ExecuTorch) vs keyword baseline.

Two clause sets: spec_clauses.yaml (80, also used while writing the verifier) and
spec_clauses_heldout.yaml (40, written independently, never used for tuning). Report both.

Per compiler:
  exact            rule identical to gold (defect, measure, op, value, action / class)
  exec_equivalent  same decision as gold on a probe grid of defects (sizes, % losses, counts)
  fields           per-field accuracy
Through the verifier (spec.verify), the SYSTEM outcome:
  auto_correct     verified and right
  silent_wrong     verified but WRONG   <- the dangerous number
  to_operator      not verified -> the operator confirms or edits (shown with the rendered sentence)
  failed           no rule produced
Verifier on deliberately corrupted gold rules (value x10 / x0.1 / mil-as-mm, wrong defect, flipped
comparison, wrong action): share caught.

docker run ... pcb-export python llm/eval_spec.py --pte /work/models/llm/llama3_2_1b_spinquant.pte \
    --tok /work/models/llm/tokenizer.model --out /work/results/spec_eval_1b.json
python llm/eval_spec.py --keyword-only --out results/spec_eval_keyword.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pcbinspect.record import BARE_CLASSES  # noqa: E402
from pcbinspect.spec import (MM_PER_MIL, Rule, fires, keyword_compile, rule_from_short, same_rule,  # noqa: E402
                             verify)

SETS = {"tuning": ROOT / "llm" / "spec_clauses.yaml", "heldout": ROOT / "llm" / "spec_clauses_heldout.yaml"}
PROBE_SIZES = [0.01, 0.03, 0.05, 0.0508, 0.06, 0.0762, 0.08, 0.1, 0.1016, 0.12, 0.15, 0.1524, 0.2, 0.25, 0.254,
               0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
PROBE_PCT = list(range(0, 101, 5)) + [12, 18, 22, 28, 32]
PROBE_COUNT = [1, 2, 3, 4, 5, 6, 7]


def decisions(r: Rule | None):
    if r is None:
        return None
    if r.kind == "class":
        return ("class", r.customer_class)
    out = []
    for cls in BARE_CLASSES:
        for s in PROBE_SIZES:
            for p in PROBE_PCT:
                d = {"cls": cls, "size_mm": s, "width_loss_pct": p, "spacing_loss_pct": p}
                for n in PROBE_COUNT:
                    out.append(r.action if fires(r, d, n) else None)
    return tuple(out)


def corruptions(r: Rule):
    if r.kind != "limit":
        return
    d = r.to_dict()
    if r.measure not in ("always", "count"):
        yield "value_x10", Rule(**{**d, "value": r.value * 10})
        yield "value_x0.1", Rule(**{**d, "value": r.value / 10})
        if r.measure == "size_mm":
            yield "mil_read_as_mm", Rule(**{**d, "value": r.value / MM_PER_MIL})
        yield "flip_comparison", Rule(**{**d, "op": ">=" if r.op == ">" else ">"})
    if r.defect != "any":
        other = "short" if r.defect != "short" else "open"
        yield "wrong_defect", Rule(**{**d, "defect": other})
    yield "wrong_action", Rule(**{**d, "action": "accept" if r.action != "accept" else "reject"})


def score(results):
    n = len(results)
    c = Counter()
    fields = Counter()
    nf = Counter()
    for x in results:
        gold, got, v = x["gold"], x["rule"], x["verify"]
        ok = same_rule(got, gold)
        eq = got is not None and decisions(got) == decisions(gold)
        c["exact"] += ok
        c["exec_equivalent"] += eq
        if got is None:
            c["failed"] += 1
        elif v["status"] == "verified":
            c["auto_correct" if eq else "silent_wrong"] += 1
        else:
            c["to_operator"] += 1
            c["to_operator_but_right"] += eq
        if gold.kind == "limit" and got is not None and got.kind == "limit":
            for f in ("defect", "measure", "action"):
                nf[f] += 1
                fields[f] += getattr(got, f) == getattr(gold, f)
            if gold.measure not in ("always",) and got.measure == gold.measure:
                nf["op"] += 1
                fields["op"] += got.op == gold.op
                nf["value"] += 1
                fields["value"] += got.value is not None and abs(got.value - gold.value) <= 1e-3 * max(1, gold.value)
        c["kind_right"] += got is not None and got.kind == gold.kind
    out = {k: c[k] / n for k in ("exact", "exec_equivalent", "auto_correct", "silent_wrong", "to_operator",
                                 "failed", "kind_right")}
    out["to_operator_but_right"] = c["to_operator_but_right"] / n
    out["fields"] = {f: fields[f] / nf[f] for f in nf if nf[f]}
    out["n"] = n
    return out


def run(compile_fn, items):
    res = []
    for i, it in enumerate(items):
        gold = rule_from_short(it)
        t0 = time.perf_counter()
        got, extra = compile_fn(it["t"], i)
        ms = (time.perf_counter() - t0) * 1e3
        v = verify(got, it["t"]) if got is not None else {"status": "failed", "problems": []}
        res.append({"t": it["t"], "gold": gold, "rule": got, "verify": v, "ms": ms, **extra})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pte")
    ap.add_argument("--tok")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--keyword-only", action="store_true")
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sets = {k: yaml.safe_load(p.read_text("utf-8")) for k, p in SETS.items() if p.exists()}
    report = {"sets": {k: len(v) for k, v in sets.items()}, "compilers": {}}

    # verifier on corrupted gold rules (no model involved)
    caught, total = Counter(), Counter()
    for items in sets.values():
        for it in items:
            for kind, r in corruptions(rule_from_short(it)):
                total[kind] += 1
                caught[kind] += verify(r, it["t"])["status"] != "verified"
    report["verifier_corruption_catch"] = {k: caught[k] / total[k] for k in total}
    report["verifier_corruption_n"] = dict(total)
    report["verifier_flags_on_gold"] = {
        k: sum(verify(rule_from_short(it), it["t"])["status"] != "verified" for it in v) / len(v)
        for k, v in sets.items()}
    print("verifier catch on corrupted gold:", {k: round(v, 3) for k, v in report["verifier_corruption_catch"].items()})
    print("verifier flags on correct gold:", report["verifier_flags_on_gold"])

    compilers = {"keyword": lambda t, i: (keyword_compile(t), {})}
    if not a.keyword_only:
        from pcbinspect.llm import ExecuTorchLlama
        from pcbinspect.spec_llm import LlamaSpecCompiler
        be = ExecuTorchLlama(a.pte, a.tok, a.threads)
        comp = LlamaSpecCompiler(be)
        name = Path(a.pte).stem

        def llm_fn(t, i):
            r = comp.compile(t, order_seed=None if a.no_shuffle else 1000 + i)
            return r.rule, {"answers": r.answers, "error": r.error, "prompt_tokens": r.prompt_tokens,
                            "generated_tokens": r.generated_tokens,
                            "decode_tps": statistics.median(r.decode_tps) if r.decode_tps else None}
        compilers[name] = llm_fn
        report["llm_load_ms"] = be.load_ms

    for name, fn in compilers.items():
        report["compilers"][name] = {}
        for sname, items in sets.items():
            res = run(fn, items)
            sc = score(res)
            sc["latency_ms_median"] = statistics.median(x["ms"] for x in res)
            sc["latency_ms_p90"] = sorted(x["ms"] for x in res)[int(0.9 * (len(res) - 1))]
            if "prompt_tokens" in res[0]:
                sc["prompt_tokens_median"] = statistics.median(x["prompt_tokens"] or 0 for x in res)
                sc["generated_tokens_median"] = statistics.median(x["generated_tokens"] for x in res)
                tps = [x["decode_tps"] for x in res if x.get("decode_tps")]
                sc["decode_tps_median"] = statistics.median(tps) if tps else None
            sc["outputs"] = [{"t": x["t"], "gold": x["gold"].to_dict(),
                              "rule": x["rule"].to_dict() if x["rule"] else None, "verify": x["verify"],
                              "answers": x.get("answers"), "error": x.get("error"), "ms": round(x["ms"])}
                             for x in res]
            report["compilers"][name][sname] = sc
            print(f"{name:28s} {sname:8s} exact={sc['exact']:.3f} exec_eq={sc['exec_equivalent']:.3f} "
                  f"auto_correct={sc['auto_correct']:.3f} SILENT_WRONG={sc['silent_wrong']:.3f} "
                  f"to_operator={sc['to_operator']:.3f} failed={sc['failed']:.3f} "
                  f"lat={sc['latency_ms_median']:.0f}ms fields={ {k: round(v, 2) for k, v in sc['fields'].items()} }",
                  flush=True)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
