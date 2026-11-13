"""Generate docs/RESULTS.md from the measured results/*.json files. Never type a number by hand.

python docs/make_results.py --results C:\\Users\\sulai\\pcb_work\\results [--device results_from_pi_dir]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def pct(x, nd=1):
    return "n/a" if x is None else f"{100 * x:.{nd}f}%"


def num(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def bare_section(r: Path) -> list[str]:
    out = []
    for tag in ("executorch", "ultralytics"):
        ev = load(r / f"bare_eval_{tag}.json")
        if not ev:
            continue
        out.append(f"### Bare-board detection (held-out test boards, `{tag}` runtime)\n")
        rows = []
        for scope in ("all", "deeppcb", "hripcb", "mendeley"):
            d = ev.get(f"det_{scope}")
            if d:
                rows.append([scope, d["n_boards"], sum(d["n_gt"].values()), pct(d["mAP50"])])
        out.append(table(["Test set", "Boards", "Defects", "mAP@0.5"], rows))
        d = ev.get("det_all", {})
        if d.get("AP50_per_class"):
            out.append("\nPer class (all test boards):\n")
            out.append(table(["Class", "AP@0.5", "Defects"],
                             [[c.replace("_", " "), pct(a), d["n_gt"].get(c, 0)]
                              for c, a in d["AP50_per_class"].items()]))
        out.append("\n### Conformal PASS / FAIL / CHECK gate (test boards never used for calibration)\n")
        rows = []
        for design, by_a in ev.get("gate", {}).items():
            for a, g in by_a.items():
                lo, hi = g["miss_rate_ci95"]
                rows.append([design, f"{100 * g['alpha']:.0f}%", g["n_defective"], g["n_good"],
                             f"{pct(g['miss_rate'])} ({pct(lo)} to {pct(hi)})",
                             pct(g["false_reject_rate"]), pct(g["review_rate"])])
        out.append(table(["Design", "Allowed miss rate", "Defective", "Good",
                          "Measured miss rate (95% CI)", "Good boards failed", "Sent to a person"], rows))
        break
    return out


def anomaly_section(r: Path) -> list[str]:
    rows, few = [], []
    for k in (1, 2, 3, 4):
        a = load(r / f"anomaly_pcb{k}.json")
        if not a:
            continue
        f = a["full"]
        rows.append([f"pcb{k}", f["n_train"], f["n_test"], pct(f["image_auroc"]),
                     pct(f.get("pixel_auroc_featuregrid")), a["bank_size"]])
        few.append([f"pcb{k}"] + [pct(a["fewshot"][s]["image_auroc_mean"]) for s in sorted(a["fewshot"], key=int)])
    if not rows:
        return []
    shots = sorted(load(r / "anomaly_pcb1.json")["fewshot"], key=int) if load(r / "anomaly_pcb1.json") else []
    return ["### Assembled boards: anomaly memory on VisA (good boards only, no defect labels)\n",
            table(["Board", "Good boards used", "Test boards", "Image AUROC", "Pixel AUROC", "Memory size"], rows),
            "\nTeach a new design from a few good boards (image AUROC, mean of 3 random draws):\n",
            table(["Board"] + [f"{s} good boards" for s in shots], few)]


def anomaly_compare_section(r: Path) -> list[str]:
    """Backbone / input-size comparison (results/anomaly_<pcb>_<tag>.json)."""
    tags = [("r18_256", "ResNet18, 256 px"), ("r18_320", "ResNet18, 320 px"),
            ("wrn50_256", "WideResNet50, 256 px"), ("wrn50_320", "WideResNet50, 320 px")]
    rows = []
    for tag, label in tags:
        vals = [load(r / f"anomaly_pcb{k}_{tag}.json") for k in (1, 2, 3, 4)]
        if not all(vals):
            continue
        img = [v["full"]["image_auroc"] for v in vals]
        pix = [v["full"]["pixel_auroc_featuregrid"] for v in vals]
        rev = [v["gate"]["alpha_0.05"]["review_rate"] for v in vals]
        miss = [v["gate"]["alpha_0.05"]["miss_rate"] for v in vals]
        few = [v["fewshot"].get("10", v["fewshot"].get(10, {})).get("image_auroc_mean") for v in vals]
        rows.append([label] + [pct(x) for x in img] + [pct(sum(img) / 4), pct(sum(pix) / 4),
                     pct(sum(f for f in few if f is not None) / max(1, sum(f is not None for f in few))),
                     pct(sum(miss) / 4), pct(sum(rev) / 4)])
    if not rows:
        return []
    return ["### Assembled boards: anomaly backbone comparison (VisA official split)\n",
            table(["Backbone", "pcb1", "pcb2", "pcb3", "pcb4", "Mean image AUROC", "Mean pixel AUROC",
                   "Mean 10-shot AUROC", "Mean miss rate at 5%", "Mean share sent to a person"], rows),
            "\nImage AUROC per board with the full memory (200 good boards). The miss rate and the "
            "share sent to a person come from the conformal gate on the held-out half of the test set."]


def advice_section(r: Path) -> list[str]:
    rows = []
    for name, label in (("advice_eval_1b_v2.json", "1B, single-shot, numbered"),
                        ("advice_eval_1b_v3.json", "1B, step-wise, shuffled"),
                        ("advice_eval_1b_v4_fewshot.json", "1B, step-wise, shuffled, worked example"),
                        ("advice_eval_3b_v3.json", "3B, step-wise, shuffled")):
        s = (load(r / name) or {}).get("summary")
        if not s:
            continue
        rep, one = s.get("repeating_defect_process_cause") or {}, s.get("one_off_defect_handling_cause") or {}
        rows.append([label, s["n"], pct(s["accepted_rate"]),
                     f"{pct(s.get('cause_position1_share'))} vs {pct(s.get('cause_position1_chance'))}",
                     f"{pct(rep.get('rate'))} vs {pct(rep.get('chance'))} (n={rep.get('n')})",
                     num(s.get("prompt_tokens_median"), 0), num((s.get("latency_ms_median") or 0) / 1000, 1)])
    if not rows:
        return []
    return ["### Llama advice quality (60 seeded defect records, x86 host)\n",
            table(["Variant", "Records", "Passed grounding", "Option 1 chosen vs chance",
                   "Repeating defect -> process cause vs chance", "Prompt tokens", "Latency (s)"], rows),
            "\nStep order (most severe first) and coverage (every defect gets a step) are enforced by "
            "the advisor, not claimed as model skill."]


def learning_section(r: Path) -> list[str]:
    e = load(r / "exp_learning_executorch.json") or load(r / "exp_learning_ultralytics.json")
    if not e:
        return []
    rows = []
    for design, d in e.items():
        if not d.get("curve"):
            continue
        b, a = d["before"], d["curve"][-1]
        rows.append([design, d["labels_available"],
                     f"{num(b['shown']['false_alarms_per_board'], 2)} -> {num(a['false_alarms_per_board'], 2)}",
                     f"{pct(b['shown']['shown_recall'])} -> {pct(a['shown_recall'])}",
                     f"{pct(b['gate']['review_rate'])} -> {pct(a['gate']['review_rate'])}",
                     f"{pct(b['gate']['miss_rate'])} -> {pct(a['gate']['miss_rate'])}"])
    return ["### Learning from the operator (feedback boards, separate calibration boards, unseen test boards)\n",
            table(["Design", "Operator labels", "False alarms per board", "Real defects shown",
                   "Boards sent to a person", "Miss rate at 5%"], rows)]


def device_section(dev: Path | None) -> list[str]:
    if dev is None:
        return []
    b = load(dev / "device_bench.json")
    if not b:
        return []
    s = b["system"]
    out = [f"### On the device: {s['board']} ({s['os']}, Python {s['python']})\n",
           f"Temperature {s.get('temp_c_start')} C at start, {s.get('temp_c_end')} C at end; "
           f"throttle flags {s.get('throttled_end')}.\n"]
    rows = []
    for name, by_t in (b.get("vision") or {}).items():
        if "skipped" in by_t:
            continue
        for t, m in by_t.items():
            rows.append([name.replace("_", " "), t.replace("threads_", ""), m["median_ms"], m["p90_ms"]])
    if rows:
        out.append(table(["Model", "Threads", "Median ms", "p90 ms"], rows))
    llm = b.get("llm") or {}
    if "runs" in llm:
        out.append("\n" + table(["Llama", "File MB", "Load ms", "RSS after load MB", "Prefill tok/s",
                                 "Decode tok/s"],
                                [[llm["model"], llm["file_mb"], llm["load_ms"], llm["rss_after_load_mb"],
                                  num(llm["prefill_tps_median"]), num(llm["decode_tps_median"])]]))
    e2e = b.get("end_to_end") or {}
    if e2e:
        out.append("\nEnd to end, median per inspection (ms):\n")
        keys = ["total_to_verdict", "detector_tiles", "anomaly_backbone", "anomaly_knn", "golden_align",
                "severity", "n_tiles"]
        out.append(table(["Design"] + [k.replace("_", " ") for k in keys],
                         [[d] + [v.get(k, "") for k in keys] for d, v in e2e.items()]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=r"C:\Users\sulai\pcb_work\results")
    ap.add_argument("--device", default=None, help="directory with device_bench.json / soak json from the Pi")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "RESULTS.md"))
    a = ap.parse_args()
    r = Path(a.results)
    parts = ["# Measured results\n",
             "Generated by `docs/make_results.py` from the JSON files the experiments write. "
             "Nothing in this file is typed by hand.\n"]
    for sec in (bare_section(r), anomaly_compare_section(r), anomaly_section(r), advice_section(r),
                learning_section(r),
                device_section(Path(a.device) if a.device else None)):
        if sec:
            parts += sec + [""]
    ts = load(r / "trace_scale_mendeley.json")
    if ts:
        parts.append(f"Mendeley scale from the 20 mil trace rule: {ts['mm_per_px']:.4f} mm/px "
                     f"({ts['images']} boards, implied {ts['implied_dpi']:.0f} dpi).")
    Path(a.out).write_text("\n".join(parts), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
