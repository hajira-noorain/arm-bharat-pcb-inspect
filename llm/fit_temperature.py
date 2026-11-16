"""Fit the temperature for the station's cause probabilities, on the development split.

The fine-tuned model ranks causes well but its percentages run high (ECE 0.10 against the lookup table's 0.036 on
v3). One number fixes that without retraining: divide the yes/no log-odds by a temperature T before turning them
into probabilities (standard temperature scaling). T is chosen to minimise the negative log-likelihood of the
generator's true causes on the development split, whose clue combinations are in neither training nor test.
The station then uses it (config/app.yaml: advice_temperature), and so does llm/eval_cause.py --temperature.

    python llm/fit_temperature.py --pte <model.pte> --tok <tokenizer.model> --n 300 --out results/measured/temperature_<name>.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _logp(lo: dict[str, float], truth: str, t: float) -> float:
    m = max(lo.values())
    z = sum(math.exp((v - m) / t) for v in lo.values())
    return (lo[truth] - m) / t - math.log(z)


def nll(rows: list[tuple[dict[str, float], str]], t: float) -> float:
    return -sum(_logp(lo, truth, t) for lo, truth in rows) / max(len(rows), 1)


def fit(rows: list[tuple[dict[str, float], str]]) -> float:
    """Grid search in log space, then a finer pass around the best point."""
    grid = [math.exp(x / 20) for x in range(-40, 80)]          # about 0.14 to 54
    best = min(grid, key=lambda t: nll(rows, t))
    fine = [best * math.exp(x / 200) for x in range(-10, 11)]
    return min(fine, key=lambda t: nll(rows, t))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pte", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from llm.data.gen_cause import examples
    from pcbinspect.cause_prompt import build_evidence_prompt, cause_question
    from pcbinspect.causes import CauseModel
    from pcbinspect.kb import default_kb
    from pcbinspect.llm_logits import LogitLlama
    kb = default_kb()
    cm = CauseModel(kb)
    be = LogitLlama(a.pte, a.tok)
    rows = []
    for ex in examples(a.n, 12, "dev", kb):
        d = next(d for d in ex["record"]["defects"] if d["id"] == ex["defect_id"])
        be.reset()
        be.feed(build_evidence_prompt(ex["record"], kb, variant=0))
        rows.append(({c: be.yes_no(cause_question(d["id"], c, kb)) for c in cm.causes(d["type"])}, ex["true_cause"]))
    t = fit(rows)
    out = {"model": be.model, "n_dev": len(rows), "temperature": round(t, 3),
           "nll_at_1": round(nll(rows, 1.0), 4), "nll_fitted": round(nll(rows, t), 4)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
