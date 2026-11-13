"""Choose the parts-check decision rules on GOOD TRAINING photos, then read the test photos off unchanged.

Input: the raw measurements written by training/dump_parts_visa.py. A rule is a small set of numbers:
  missing     no size-compatible detection at conf >= present_conf, and likeness < min(cap, sim_min - margin)
  changed     a detection is there, and likeness < (the missing threshold) - changed_extra   [only with sim_min]
  unexpected  a detection that is not a taught part, conf >= extra_conf, not in an ignore zone, likeness < extra_like

Among a small grid, the rule chosen is the most sensitive one (highest thresholds) whose share of good DEV
photos raising anything stays within a target (5% by default); dev photos are good training photos that were
not used to teach. Nothing about the test photos, good or faulty, enters the choice. What this cannot do is
tell us how sensitive the chosen rule is: only the test photos say that, and they are read once.

python training/tune_parts_rules.py --raw results/parts_visa_raw_pcb1.json ... --out results/parts_rules_visa.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path


def decide(rec: dict, r: dict) -> dict:
    """What the rule reports on one photo: counts, and whether each kind lands on the labelled area."""
    out = {"missing": 0, "changed": 0, "unexpected": 0, "missing_on": False, "changed_on": False, "unexpected_on": False}
    for conf, like, sim_min, on in rec.get("parts", []):
        thr = r["cap"] if sim_min is None else min(r["cap"], max(0.3, sim_min - r["margin"]))
        if conf < r["present_conf"]:
            if like is None or like < thr:
                out["missing"] += 1
                out["missing_on"] |= on
        elif like is not None and sim_min is not None and like < thr - r["changed_extra"]:
            out["changed"] += 1
            out["changed_on"] |= on
    for conf, like, ignored, on in rec.get("others", []):
        if conf >= r["extra_conf"] and not ignored and (like is None or like < r["extra_like"]):
            out["unexpected"] += 1
            out["unexpected_on"] |= on
    return out


def score(photos: list[dict], r: dict, min_cc: float) -> dict:
    by = defaultdict(lambda: defaultdict(int))
    for rec in photos:
        ok = rec.get("cc", 0) >= min_cc and "parts" in rec
        d = decide(rec, r) if ok else None
        for t in rec["types"]:
            s = by[t]
            s["photos"] += 1
            s["lined_up"] += ok
            if d:
                s["anything"] += (d["missing"] + d["changed"] + d["unexpected"]) > 0
                s["missing_calls"] += d["missing"]
                s["changed_calls"] += d["changed"]
                s["unexpected_calls"] += d["unexpected"]
                s["on_area"] += d["missing_on"] or d["changed_on"] or d["unexpected_on"]
                s["missing_on_area"] += d["missing_on"]
                s["unexpected_on_area"] += d["unexpected_on"]
                s["changed_on_area"] += d["changed_on"]
    return {k: dict(v) for k, v in by.items()}


GRID = {"present_conf": [0.25], "cap": [0.6, 0.7, 0.8], "margin": [0.15, 0.25, 0.35],
        "changed_extra": [0.2, 0.3, 0.45], "extra_conf": [0.6, 0.7, 0.8], "extra_like": [0.4, 0.5, 0.6, 0.7]}
CURRENT = {"present_conf": 0.25, "cap": 0.85, "margin": 0.15, "changed_extra": 0.2, "extra_conf": 0.6, "extra_like": 0.7}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-cc", type=float, default=0.6)
    ap.add_argument("--max-false", type=float, default=0.05, help="target share of good dev photos raising anything")
    a = ap.parse_args()
    raw = {}
    for f in a.raw:
        raw.update(json.loads(Path(f).read_text()))
    report = {"note": "rule chosen on good training photos only (dev); test photos read off unchanged", "by_goods": {}}
    for goods in sorted({k.split(":")[1] for k in raw}, key=int):
        keys = [k for k in raw if k.endswith(":" + goods)]
        dev = [p for k in keys for p in raw[k]["photos"] if p["split"] == "dev"]
        test = [p for k in keys for p in raw[k]["photos"] if p["split"] == "test"]

        def false_rate(r, photos=dev):
            s = score(photos, r, a.min_cc).get("good", {})
            return s.get("anything", 0) / max(s.get("lined_up", 0), 1)
        rules = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
        # the most sensitive rule (highest thresholds) that keeps dev false calls within the target; if none
        # does, the one with the fewest dev false calls
        rules.sort(key=lambda r: (-(r["cap"] - r["margin"] - r["changed_extra"] + r["extra_like"] - r["extra_conf"])))
        ok = [r for r in rules if false_rate(r) <= a.max_false]
        chosen = ok[0] if ok else min(rules, key=false_rate)
        entry = {"designs": {k.split(":")[0]: {"taught": raw[k]["taught"], "scale": raw[k]["scale"]} for k in keys},
                 "chosen_rule": chosen, "dev_false_call_rate": false_rate(chosen),
                 "dev_false_call_rate_before": false_rate(CURRENT),
                 "dev_by_design": {k.split(":")[0]: false_rate(chosen, [p for p in raw[k]["photos"] if p["split"] == "dev"])
                                   for k in keys},
                 "test": score(test, chosen, a.min_cc), "test_before": score(test, CURRENT, a.min_cc),
                 "test_by_design": {k.split(":")[0]: score([p for p in raw[k]["photos"] if p["split"] == "test"], chosen, a.min_cc)
                                    for k in keys}}
        report["by_goods"][goods] = entry
        print(f"== taught from {goods} good photos: rule {chosen}")
        print(f"   dev good photos raising anything: {entry['dev_false_call_rate']:.3f} (before {entry['dev_false_call_rate_before']:.3f}) "
              f"by design {({k: round(v, 3) for k, v in entry['dev_by_design'].items()})}")
        for lab, sc in (("before", entry["test_before"]), ("chosen", entry["test"])):
            for t in ("good", "missing", "extra", "wrong place"):
                s = sc.get(t)
                if s:
                    print(f"   test {lab:6s} {t:11s} n={s['photos']:3d} lined_up={s['lined_up']:3d} anything={s.get('anything', 0):3d} "
                          f"on_area={s.get('on_area', 0):3d} missing_on={s.get('missing_on_area', 0):3d} "
                          f"unexpected_on={s.get('unexpected_on_area', 0):3d} changed_on={s.get('changed_on_area', 0):3d}")
        for d, sc in entry["test_by_design"].items():
            g, m = sc.get("good", {}), sc.get("missing", {})
            print(f"   {d}: good raising anything {g.get('anything', 0)}/{g.get('photos', 0)}; "
                  f"missing on area {m.get('on_area', 0)}/{m.get('photos', 0)} (as missing {m.get('missing_on_area', 0)})")
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
