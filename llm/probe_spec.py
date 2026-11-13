"""Probe prompt formats for the rule compiler on the TUNING clauses only (held-out stays untouched).

Variants for the first field (Defect), which decides everything after it:
  stepwise_fewshot   the production format (fields + 4 worked examples, one KV cache)
  stepwise_zeroshot  same without examples (examples invite copying in small models)
  qa_zeroshot        one short independent question per field, natural wording
python llm/probe_spec.py --pte ... --tok ... --n 20
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pcbinspect.advisor import HDR  # noqa: E402
from pcbinspect.spec import DISPLAY, rule_from_short  # noqa: E402
from pcbinspect.spec_llm import DEFECT_NAMES, SYSTEM, _line, _lists, build_prompt, map_defect  # noqa: E402


def qa_prompt(clause, names):
    user = (f"Requirement: \"{clause}\"\n\nWhich PCB defect is this requirement about? Answer with one of: "
            f"{', '.join(names)}. If it names no defect and only sets the IPC class, answer none.")
    return (HDR + "You answer with a few words only." + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            + user + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pte", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    from pcbinspect.llm import ExecuTorchLlama
    be = ExecuTorchLlama(a.pte, a.tok, 4)
    items = yaml.safe_load((ROOT / "llm" / "spec_clauses.yaml").read_text("utf-8"))[: a.n]
    stop = lambda s: "\n" in s  # noqa: E731
    import pcbinspect.spec_llm as SL
    for variant in ("stepwise_fewshot", "stepwise_zeroshot", "qa_zeroshot"):
        ok = 0
        outs = []
        for i, it in enumerate(items):
            gold = rule_from_short(it)
            gd = "none" if gold.kind == "class" else gold.defect
            lists = _lists(random.Random(1000 + i))
            be.reset()
            if variant == "qa_zeroshot":
                g = be.cont(qa_prompt(it["t"], lists["Defect"]), 8, stop=stop)
            else:
                ex = SL.EXAMPLES
                if variant == "stepwise_zeroshot":
                    SL.EXAMPLES = []
                p = build_prompt(it["t"], lists)
                SL.EXAMPLES = ex
                g = be.cont(p + "Defect:", 8, stop=stop)
            ans = _line(g.text)
            got = "none" if ans.startswith("none") else map_defect(ans)
            ok += got == gd
            outs.append((ans, gd))
        print(f"{variant:18s} defect right {ok}/{len(items)}  sample answers: {outs[:8]}", flush=True)


if __name__ == "__main__":
    main()
