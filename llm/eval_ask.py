"""Shift questions over the inspection log: does Llama (ExecuTorch) pick the right query, and is its
summary of the rows free of numbers that are not in the rows?

A fixed log is built (seeded), every question in llm/ask_questions.yaml is answered through
pcbinspect.ask.answer(), and the query chosen is compared with the one that answers the question.
  route_accuracy       the query used = the right query          (keyword router shown as the baseline)
  summary_by_llama     share of answers where the summary shown was written by Llama (else: the table)
  summary_rejected     why the others were not
Only the query CHOICE is by Llama: hours and board IDs are read from the question by rule.

python llm/eval_ask.py --pte ... --tok ... --backend logits --sets tuning --out results/ask_eval.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pcbinspect import ask  # noqa: E402
from pcbinspect.record import Defect, Inspection  # noqa: E402
from pcbinspect.store import Store  # noqa: E402


def build_log(root: Path, n: int = 120, seed: int = 0) -> Store:
    rng = random.Random(seed)
    s = Store(root)
    now = time.time()
    for i in range(n):
        k = rng.choice([0, 0, 1, 1, 2, 3])
        defects = [Defect(id=f"D{j + 1}", cls=rng.choice(["short", "open", "mouse_bite", "spur", "pin_hole"]),
                          source="detector", bbox=(1, 2, 3, 4), confidence=round(rng.uniform(0.5, 0.99), 2),
                          severity=rng.choice(["critical", "major", "minor"]), ipc={}) for j in range(k)]
        verdict = "PASS" if not defects else rng.choice(["FAIL", "FAIL", "REVIEW"])
        iid = s.log_inspection(Inspection(board_id=f"B{i % 110:05d}", design="eval", image_path="", defects=defects,
                                          board_score=0.9 if defects else 0.1, verdict=verdict, alpha=0.05,
                                          mm_per_px=0.02, latency_ms={}), None)
        s.conn.execute("UPDATE inspections SET ts=? WHERE id=?", (now - rng.uniform(0, 30 * 3600), iid))
    s.log_alert("eval", "drift:short", "short on 7 of the last 20 boards: check the etch line", {})
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pte")
    ap.add_argument("--tok")
    ap.add_argument("--backend", default="logits", choices=["logits", "runner", "none"])
    ap.add_argument("--sets", default="tuning")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sets = yaml.safe_load((ROOT / "llm" / "ask_questions.yaml").read_text("utf-8"))
    be = None
    if a.backend == "logits":
        from pcbinspect.llm_logits import LogitLlama
        be = LogitLlama(a.pte, a.tok)
    elif a.backend == "runner":
        from pcbinspect.llm import ExecuTorchLlama
        be = ExecuTorchLlama(a.pte, a.tok)
    store = build_log(Path(tempfile.mkdtemp()))
    report = {"backend": a.backend, "model": Path(a.pte).name if a.pte else None, "sets": {}}
    for name in a.sets.split(","):
        items = sets[name][:a.limit] if a.limit else sets[name]
        rows, t_all = [], []
        for it in items:
            t0 = time.perf_counter()
            r = ask.answer(it["q"], store.conn, be)
            t_all.append((time.perf_counter() - t0) * 1e3)
            kq, _ = ask.keyword_route(it["q"])
            rows.append({"q": it["q"], "gold": it["query"], "query": r["query"], "route": r["route"],
                         "keyword_query": kq, "params": r["params"], "summary_by": r["summary_by"],
                         "rejected": r.get("summary_rejected") or r.get("summary_error"), "answer": r["answer"],
                         "n_rows": len(r["rows"])})
        n = len(rows)
        with_rows = [x for x in rows if x["n_rows"]]
        sc = {"n": n,
              "route_accuracy": sum(x["query"] == x["gold"] for x in rows) / n,
              "routed_by_llama": sum(x["route"] == "llama" for x in rows) / n,
              "keyword_route_accuracy": sum(x["keyword_query"] == x["gold"] for x in rows) / n,
              "wrong_by_query": dict(Counter(f"{x['gold']} -> {x['query']}" for x in rows if x["query"] != x["gold"])),
              "summary_by_llama": sum(x["summary_by"] == "llama" for x in with_rows) / max(len(with_rows), 1),
              "summary_rejected": dict(Counter(str(x["rejected"])[:40] for x in with_rows if x["rejected"])),
              "latency_ms_median": statistics.median(t_all)}
        report["sets"][name] = {"score": sc, "rows": rows}
        print(name, json.dumps(sc), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
