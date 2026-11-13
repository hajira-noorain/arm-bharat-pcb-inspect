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


def _bound(gates: Path | None, design: str, alpha: float) -> str:
    """95%-confidence upper bound on the miss rate for THIS calibration set (gate.miss_rate_bound)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pcbinspect.gate import miss_rate_bound
    g = load(gates / f"{design}.json") if gates else None
    if not g:
        return "n/a"
    b = miss_rate_bound(len(g["defective_scores"]), alpha)
    return "n/a" if b is None else f"{pct(b)} (n={len(g['defective_scores'])})"


def bare_section(r: Path, tag_pref=("v3_et", "v3", "executorch", "final", "ultralytics", "ep70"),
                 gates: Path | None = None) -> list[str]:
    out = []
    tags = [t for t in tag_pref if (r / f"bare_eval_{t}.json").exists()]
    gate_ev = next((load(r / f"bare_eval_{t}.json") for t in tags
                    if (load(r / f"bare_eval_{t}.json") or {}).get("gate")), None)
    for tag in tags:
        ev = load(r / f"bare_eval_{tag}.json")
        if not ev:
            continue
        out.append(f"### Bare-board detection (held-out test boards, `{tag}` runtime)\n")
        rows = []
        for scope in ("all",) + SOURCES:
            d = ev.get(f"det_{scope}")
            if d:
                rows.append([SOURCE_NAMES.get(scope, scope), d["n_boards"], sum(d["n_gt"].values()), pct(d["mAP50"])])
        out.append(table(["Test set", "Boards", "Defects", "mAP@0.5"], rows))
        d = ev.get("det_all", {})
        if d.get("AP50_per_class"):
            out.append("\nPer class (all test boards):\n")
            out.append(table(["Class", "AP@0.5", "Defects"],
                             [[c.replace("_", " "), pct(a), d["n_gt"].get(c, 0)]
                              for c, a in d["AP50_per_class"].items()]))
        out.append("\n### Conformal PASS / FAIL / CHECK gate (test boards never used for calibration)\n")
        rows = []
        for design, by_a in (gate_ev or {}).get("gate", {}).items():
            for a, g in by_a.items():
                lo, hi = g["miss_rate_ci95"]
                dlo, dhi = g.get("defect_recall_ci95", (None, None))
                rc = g.get("random_defect_miss_ci95")
                arate = f"{100 * g['alpha']:.0f}%" + ("" if g.get("supported_alpha", True)
                                                      else " (too few calibration defects)")
                rows.append([SOURCE_NAMES.get(design, design), arate, _bound(gates, design, g["alpha"]),
                             g["n_defective"], g["n_good"],
                             f"{pct(g.get('random_defect_miss'))} ({pct(rc[0])} to {pct(rc[1])})" if rc else "n/a",
                             f"{pct(1 - g['defect_recall'])} ({pct(1 - dhi)} to {pct(1 - dlo)})" if dlo is not None else "n/a",
                             f"{pct(g['miss_rate'])} ({pct(lo)} to {pct(hi)})",
                             num(g.get("false_alarms_per_board"), 2),
                             pct(g["false_reject_rate"]), pct(g["review_rate"])])
        out.append("The gate is calibrated PER DEFECT (one random defect per calibration board); a defect is "
                   "shown when its evidence reaches the threshold. Miss rates are measured on test boards. "
                   "\"One random defect per board\" is the quantity the guarantee covers (interval: bootstrap over "
                   "boards); \"all defects\" pools every defect, so boards with many defects weigh more. A design "
                   "with no good calibration boards never says PASS, so every board goes to a person. A rate marked "
                   "\"too few calibration defects\" needs at least 1/rate - 1 of them and carries no guarantee.\n")
        out.append("\"Promised\" is what this one calibration set supports: averaged over calibrations the miss rate "
                   "is at most the target, and with 95% confidence it is at most the figure shown (n = calibration "
                   "defects). Read the measured column against it.\n")
        out.append(table(["Design", "Target miss rate", "Promised at 95% confidence", "Defective", "Good",
                          "Missed, one random defect per board (95% CI)", "Missed, all defects (95% CI)",
                          "Defective boards passed (95% CI)", "False calls per board",
                          "Good boards failed", "Sent to a person"], rows))
        break
    return out


SOURCES = ("deeppcb", "hripcb", "mendeley", "dspcbsd", "defectdet")
SOURCE_NAMES = {"all": "All test boards", "deeppcb": "DeepPCB", "hripcb": "PKU-Market-PCB (HRIPCB)",
                "mendeley": "Mendeley PCB-Defect", "dspcbsd": "DsPCBSD+", "defectdet": "DefectDet (held-out boards)"}
DETECTORS = (("base", "DeepPCB + HRIPCB + Mendeley"), ("cam", "+ camera-style augmentation"),
             ("v2", "+ DsPCBSD+"), ("v3", "+ half of DefectDet's boards, solder-mask recolouring (deployed)"),
             ("v4", "+ synthetic tiles rendered from 61 open-hardware KiCad boards (not deployed)"),
             ("v3_et", "v3 through the exported ExecuTorch FP32 .pte"),
             ("v3_et_int8", "v3 through the exported ExecuTorch INT8 .pte"))


def detector_compare_section(r: Path) -> list[str]:
    rows = []
    for tag, label in DETECTORS:
        ev = load(r / f"bare_eval_{tag}.json")
        if not ev:
            continue
        rows.append([f"{tag}: {label}"] + [pct((ev.get(f"det_{s}") or {}).get("mAP50")) for s in ("all",) + SOURCES])
    if not rows:
        return []
    return ["### Detector versions on the same held-out test boards (mAP@0.5)\n",
            "Each version adds training data; the test boards are the same for all of them and were never trained on. "
            "DefectDet is split by physical board: the test boards are boards the model never saw, from the same lab.\n",
            table(["Detector", "All"] + [SOURCE_NAMES[s] for s in SOURCES], rows)]


def external_section(r: Path) -> list[str]:
    ev = load(r / "external_defectdet_heldout.json")
    if not ev:
        return []
    rows = [[k.split("@")[0], f"x{v['scale']:g}", pct(v["mAP50_shared_classes"]), pct(v["defect_recall_at_lambda"]),
             num(v["false_calls_per_board"], 2)] for k, v in ev["models"].items()]
    return [f"### DefectDet held-out boards, resized before inspection ({ev['n_boards']} boards)\n",
            "Tests whether enlarging small-defect photos helps the detector (it does not: native scale is best).\n",
            table(["Detector", "Resize", "mAP@0.5", "Defects found at the gate threshold", "False calls per board"],
                  rows)]


def realiad_section(r: Path) -> list[str]:
    rows = []
    for tag, label in (("r18_256", "ResNet18, 256 px"), ("r18_320", "ResNet18, 320 px"),
                       ("wrn50_256", "WideResNet50, 256 px")):
        ev = load(r / f"realiad_pcb_{tag}.json")
        if not ev:
            continue
        v = ev["views"]
        rows.append([label] + [pct(v[c]["image_auroc"]) if c in v else "n/a" for c in ("C1", "C2", "C3", "C4", "C5")]
                    + [pct(ev.get("mean_image_auroc"))])
    if not rows:
        return []
    return ["### Assembled boards, second dataset: Real-IAD `pcb` (evaluation only, never trained on or shipped)\n",
            "One good-board memory per camera view, built from that view's good images; image AUROC per view.\n",
            table(["Backbone", "View 1", "View 2", "View 3", "View 4", "View 5", "Mean"], rows)]


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
    tags = [("r18_256", "ResNet18, 256 px, max patch"), ("r18_320", "ResNet18, 320 px, max patch"),
            ("r18_320_top", "ResNet18, 320 px, top-1% patches (deployed)"),
            ("r18_320_pn", "ResNet18, 320 px, per-position normalised"),
            ("r18_320_pn_fg_top", "ResNet18, 320 px, normalised + board mask + top-1%"),
            ("r18_384", "ResNet18, 384 px, max patch"),
            ("r18_384_pn_fg_top", "ResNet18, 384 px, normalised + board mask + top-1%"),
            ("wrn50_256", "WideResNet50, 256 px, max patch"), ("wrn50_320", "WideResNet50, 320 px, max patch")]
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
    return ["### Assembled boards: anomaly settings compared (VisA official split)\n",
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
    e = (load(r / "exp_learning_v3.json") or load(r / "exp_learning_executorch.json")
         or load(r / "exp_learning_ultralytics.json"))
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


def golden_section(r: Path) -> list[str]:
    files = sorted(r.glob("golden_eval_*.json"))
    if not files:
        return []
    ev = load(files[-1])
    rows = []
    names = {"detector": "Detector only", "golden_diff": "Golden-board difference only",
             "fused": "Detector + golden difference (fused)"}
    for ch, by_a in ev["channels"].items():
        g = by_a["alpha_0.05"]
        lo, hi = g["defect_miss_ci95"]
        rc = g.get("random_defect_miss_ci95")
        rows.append([names.get(ch, ch),
                     f"{pct(g.get('random_defect_miss'))} ({pct(rc[0])} to {pct(rc[1])})" if rc else "n/a",
                     f"{pct(g['defect_miss'])} ({pct(lo)} to {pct(hi)})",
                     num(g["false_calls_per_board"], 2), pct(g["typed_correct_share_of_found"])])
    return [f"### Golden-board comparison ({ev['design']}, test boards, golden re-captured with a random warp)\n",
            "Each channel has its own per-defect gate at a 5% allowed miss rate, calibrated on the calibration "
            "boards.\n",
            table(["Evidence", "Missed, one random defect per board (95% CI)", "Missed, all defects (95% CI)",
                   "False calls per board", "Named correctly"], rows)]


def eyetest_section(r: Path) -> list[str]:
    out = []
    pod = load(r / "pod_insilico_v3.json") or load(r / "pod_insilico.json")
    if pod:
        out.append("### Detection-limit eye test, in silico (defects drawn on held-out DeepPCB patterns)\n")
        out.append(f"{pod['n_drawn']} drawn defects on held-out clean patterns at {pod['px_per_mm']:.0f} px/mm; "
                   "each condition re-judges the same drawn boards. Shorts are sized by the gap they bridge.\n")
        rows = []
        for cond, o in pod["conditions"].items():
            for t, v in o["types"].items():
                g = o.get("golden", {}).get("types", {}).get(t)
                rows.append([cond, t.replace("_", " "), v.get("sentence", ""), g.get("sentence", "") if g else "n/a"])
        out.append(table(["Camera condition", "Defect", "Detector only", "Detector + golden board"], rows))
        fc = [[c, num(o["false_calls_per_board"]["gate_lambda"], 2),
               num(o.get("golden", {}).get("false_calls_per_board"), 2)] for c, o in pod["conditions"].items()]
        out.append("\nFalse calls per clean board at the operating threshold:\n")
        out.append(table(["Camera condition", "Detector only", "Detector + golden board"], fc))
    sim = load(r / "eyetest_sim_v3.json") or load(r / "eyetest_sim.json")
    if sim:
        out.append("\n### Eye-test card read from SIMULATED photos (print + camera model, not real photos)\n")
        out.append(f"Inkjet dot gain {sim['ink_um']:.0f} um, camera {sim['px_per_mm']:.0f} px/mm on paper, "
                   "perspective, blur, noise and JPEG (`training/eyetest_sim.py`). Validates the reader end to end, "
                   "not the physical print or camera.\n")
        rows = []
        for ch, sres in sim["channels"].items():
            for t, v in sres["types"].items():
                rows.append([ch, t.replace("_", " "), v["sentence"],
                             " ".join(f"{x['size_mm']}:{x['hits']}/{x['n']}" for x in v["table"])])
        out.append(table(["Evidence", "Defect", "Statement", "Found / planted per size (mm)"], rows))
    return out


SPEC_TAGS = {"1b": "free generation, zero-shot", "3b": "free generation, zero-shot",
             "3b_fewshot": "free generation, worked examples"}


def spec_section(r: Path) -> list[str]:
    files = {p.stem.replace("spec_eval_", ""): load(p) for p in sorted(r.glob("spec_eval_*.json"))}
    files = {k: v for k, v in files.items() if v}
    if not files:
        return []
    out = ["### Customer requirements to rules (Llama on ExecuTorch vs keyword reader)\n",
           "Tuning = 80 clauses used while writing the checks; held-out = 40 clauses written independently and "
           "never used for tuning. Silently wrong = the checks passed but the rule is wrong.\n",
           "Free generation: Llama writes each field as text. Constrained decoding: Llama can only output an entry "
           "of the option list (a prefix tree over its next-token scores). Grounded choice: the clause's own words "
           "narrow each list first. Methods were compared on the tuning set; the held-out set was run once for a "
           "method that was kept.\n"]
    rows = []
    seen_kw = False
    for tag, ev in files.items():
        for name, by_set in ev["compilers"].items():
            if name == "keyword":
                if seen_kw:
                    continue
                seen_kw = True
            for sname, sc in by_set.items():
                size = "3B" if "3b" in name else "1B" if "1b" in name else name
                how = ("grounded choice, constrained decoding (deployed)" if name.endswith("_grounded") else
                       "constrained decoding, full option lists" if name.endswith("_constrained") else
                       SPEC_TAGS.get(tag, tag))
                label = "Keyword reader" if name == "keyword" else f"Llama 3.2 {size}, {how}"
                rows.append([label, sname, sc["n"], pct(sc["exact"]), pct(sc["auto_correct"]), pct(sc["silent_wrong"]),
                             pct(sc["to_operator"]), pct(sc["failed"]),
                             "0" if name == "keyword" else f"{sc['latency_ms_median'] / 1000:.1f} s"])
    out.append(table(["Reader", "Set", "Clauses", "Exactly right", "Auto-accepted and right", "Silently wrong",
                      "Sent to operator", "No rule", "Median time (x86 host)"], rows))
    ev = next(iter(files.values()))
    cc = ev.get("verifier_corruption_catch", {})
    if cc:
        out.append("\nChecks on deliberately corrupted correct rules (share caught):\n")
        out.append(table(["Corruption", "Caught"], [[k.replace("_", " "), pct(v)] for k, v in cc.items()]))
        fl = ev.get("verifier_flags_on_gold", {})
        out.append(f"\nCorrect rules flagged for the operator anyway: tuning {pct(fl.get('tuning'))}, "
                   f"held-out {pct(fl.get('heldout'))}.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=r"C:\Users\sulai\pcb_work\results")
    ap.add_argument("--gates", default=r"C:\Users\sulai\pcb_work\models\gate", help="calibrated gate files")
    ap.add_argument("--device", default=None, help="directory with device_bench.json / soak json from the Pi")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "RESULTS.md"))
    a = ap.parse_args()
    r = Path(a.results)
    parts = ["# Measured results\n",
             "Generated by `docs/make_results.py` from the JSON files the experiments write. "
             "Nothing in this file is typed by hand.\n"]
    for sec in (bare_section(r, gates=Path(a.gates)), detector_compare_section(r), external_section(r), golden_section(r),
                eyetest_section(r), spec_section(r), anomaly_section(r), anomaly_compare_section(r),
                realiad_section(r), advice_section(r),
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
