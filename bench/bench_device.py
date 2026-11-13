"""On-device benchmark suite (run ON the Raspberry Pi 5; also runs in the arm64 simulator).

Measures, with the conditions recorded alongside every number:
  - system: board model, CPU, RAM, OS, Python, ExecuTorch version, temperature, throttle flags
  - vision: detector tile latency and anomaly backbone latency at 1/2/4 threads, FP32 vs INT8
  - Llama: load time, prefill & decode tokens/s from the runtime's own stats, RSS after load
  - end-to-end: per-stage latency of real inspections on the demo boards (capture->verdict, ->advice)
  - power: the Pi 5 power chip's own rail readings, sampled twice a second while each part runs

Usage:  python -m bench.bench_device --root . --out results/device_bench.json [--quick]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def temp_c() -> float | None:
    out = sh("vcgencmd measure_temp")
    if out.startswith("temp="):
        return float(out.split("=")[1].split("'")[0])
    p = Path("/sys/class/thermal/thermal_zone0/temp")
    return int(p.read_text()) / 1000 if p.exists() else None


def parse_pmic(text: str) -> float | None:
    """Watts from `vcgencmd pmic_read_adc` on a Pi 5: the sum over the regulated rails of voltage x current
    (lines like ' VDD_CORE_A current(7)=0.698A' and ' VDD_CORE_V volt(15)=0.72V'). This is what the board's
    own power chip reports for the rails it feeds; it leaves out the regulator losses and anything powered
    straight from 5 V (USB devices, fan), so it is an estimate that reads lower than a wall meter."""
    import re
    amps, volts = {}, {}
    for name, kind, val in re.findall(r"(\w+)_([AV])\s+\w+\(\d+\)=([0-9.]+)", text or ""):
        (amps if kind == "A" else volts)[name] = float(val)
    rails = [amps[k] * volts[k] for k in amps if k in volts]
    return round(sum(rails), 3) if rails else None


class PowerLog:
    """Samples the board power twice a second in the background while a benchmark section runs."""

    def __init__(self, period: float = 0.5):
        self.period, self.samples, self._stop = period, [], None

    def __enter__(self):
        import threading
        self._stop = threading.Event()

        def run():
            while not self._stop.is_set():
                w = parse_pmic(sh("vcgencmd pmic_read_adc"))
                if w is None:
                    return                      # not a Pi 5: nothing to record
                self.samples.append(w)
                self._stop.wait(self.period)
        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=5)
        return False

    def summary(self) -> dict | None:
        if not self.samples:
            return None
        return {"mean_w": round(statistics.mean(self.samples), 2), "max_w": round(max(self.samples), 2),
                "samples": len(self.samples)}


def throttled() -> str:
    return sh("vcgencmd get_throttled") or "n/a"


def rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def system_info() -> dict:
    import executorch
    model = Path("/proc/device-tree/model")
    return {
        "board": model.read_text().strip("\x00") if model.exists() else platform.node(),
        "machine": platform.machine(), "cpus": os.cpu_count(),
        "cpu_model": sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2").strip(),
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 2)
        if hasattr(os, "sysconf") else None,
        "os": sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2").strip('"'),
        "python": platform.python_version(),
        "executorch": getattr(executorch, "__version__", "unknown"),
        "temp_c_start": temp_c(), "throttled_start": throttled(),
        "simulated": not model.exists(),
    }


def set_threads(n: int) -> int:
    from executorch.extension.pybindings import portable_lib as pl
    pl._unsafe_reset_threadpool(n)
    return pl._threadpool_get_thread_count()


def time_method(pte: Path, x: np.ndarray, warmup: int, reps: int) -> dict:
    import torch
    from executorch.runtime import Runtime
    t0 = time.perf_counter()
    m = Runtime.get().load_program(str(pte)).load_method("forward")
    load_ms = (time.perf_counter() - t0) * 1e3
    xt = torch.from_numpy(x)
    for _ in range(warmup):
        m.execute([xt])
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        m.execute([xt])
        ts.append((time.perf_counter() - t) * 1e3)
    return {"load_ms": round(load_ms, 1), "median_ms": round(statistics.median(ts), 2),
            "p90_ms": round(float(np.percentile(ts, 90)), 2), "min_ms": round(min(ts), 2), "reps": reps}


try:
    from pcbinspect.anomaly import INPUT as BACKBONE_INPUT
except Exception:   # noqa: BLE001
    BACKBONE_INPUT = 320


def bench_vision(root: Path, threads: list[int], quick: bool) -> dict:
    out = {}
    rng = np.random.default_rng(0)
    cases = [("detector_fp32", "models/vision/pcb_y26n_xnnpack.pte", (1, 3, 640, 640)),
             ("detector_int8", "models/vision/pcb_y26n_xnnpack_int8.pte", (1, 3, 640, 640)),
             # the retrained larger detectors: which one ships is decided by these timings
             ("detector_s_fp32", "models/vision/pcb_y26s_xnnpack.pte", (1, 3, 640, 640)),
             ("detector_s_int8", "models/vision/pcb_y26s_xnnpack_int8.pte", (1, 3, 640, 640)),
             ("detector_m_fp32", "models/vision/pcb_y26m_xnnpack.pte", (1, 3, 640, 640)),
             ("detector_m_int8", "models/vision/pcb_y26m_xnnpack_int8.pte", (1, 3, 640, 640)),
             # the backbone is exported at the anomaly model's input size (320 px), not a fixed 256
             ("backbone_fp32", "models/vision/resnet18_patch_xnnpack.pte", (1, 3, BACKBONE_INPUT, BACKBONE_INPUT)),
             ("backbone_int8", "models/vision/resnet18_patch_xnnpack_int8.pte", (1, 3, BACKBONE_INPUT, BACKBONE_INPUT))]
    for name, rel, shape in cases:
        p = root / rel
        if not p.exists():
            out[name] = {"skipped": "file not found"}
            continue
        x = rng.random(shape, dtype=np.float32)
        out[name] = {}
        for n in threads:
            got = set_threads(n)
            try:
                out[name][f"threads_{got}"] = time_method(p, x, 2 if quick else 5, 5 if quick else 30)
            except Exception as exc:    # noqa: BLE001  (one model failing must not lose the others' numbers)
                out[name][f"threads_{got}"] = {"error": repr(exc)[:300]}
                print(name, "FAILED", repr(exc)[:300], flush=True)
                break
            out[name][f"threads_{got}"]["temp_c"] = temp_c()
            print(name, got, out[name][f"threads_{got}"], flush=True)
    return out


def bench_detector_runtimes(root: Path, cfg: dict, quick: bool) -> dict:
    """The SAME fine-tuned detector on the same real 640 px tiles: ExecuTorch (XNNPACK) vs Arm Runtime for
    YOLO26 (Arm Optimized-Runtimes). Reports per-tile latency and whether both give the same detections."""
    import glob
    import cv2
    from pcbinspect.detect_bare import ArmRuntimeTiles, ExecuTorchTiles, resolve_arm_runtime
    tiles = []
    for p in sorted(glob.glob(str(root / "demo" / "boards" / "deeppcb_*.jpg")))[:8]:
        im = cv2.imread(p)
        if im is not None and "golden" not in p:
            tiles.append(im[:640, :640])
    out = {}
    backends = {}
    if (root / cfg.get("detector_pte", "")).is_file():
        backends["executorch"] = lambda: ExecuTorchTiles(root / cfg["detector_pte"])
    found = resolve_arm_runtime(root, cfg.get("arm_runtime_lib", ""), cfg.get("arm_runtime_model", ""))
    if found is not None:
        print("Arm runtime:", *found, flush=True)
        backends["arm_runtime"] = lambda: ArmRuntimeTiles(*found)
    else:
        out["arm_runtime_missing"] = "libarm-yolo26n.so or a converted model (model.bin) not found"
    dets = {}
    for name, make in backends.items():
        try:
            m = make()
        except Exception as exc:   # noqa: BLE001
            out[name] = {"error": str(exc)}
            continue
        for t in tiles[:2]:
            m.infer(t)                       # warm-up
        reps = 3 if quick else 10
        ts = []
        for _ in range(reps):
            for t in tiles:
                t0 = time.perf_counter()
                m.infer(t)
                ts.append((time.perf_counter() - t0) * 1e3)
        dets[name] = [m.infer(t) for t in tiles]
        out[name] = {"tile_ms_median": float(np.median(ts)), "tile_ms_p90": float(np.percentile(ts, 90)),
                     "n": len(ts), "temp_c": temp_c()}
        print("detector runtime", name, out[name], flush=True)
    if len(dets) == 2:   # agreement: same boxes above 0.25 confidence
        a, b = dets["executorch"], dets["arm_runtime"]
        agree = []
        for x, y in zip(a, b):
            x, y = x[x[:, 4] >= 0.25], y[y[:, 4] >= 0.25]
            agree.append(abs(len(x) - len(y)))
        out["count_difference_per_tile"] = float(np.mean(agree))
    return out


def bench_llm(root: Path, cfg: dict, quick: bool) -> dict:
    from pcbinspect.advisor import build_prompt
    from pcbinspect.kb import default_kb
    from pcbinspect.llm import ExecuTorchLlama
    pte = root / cfg["llm_pte"]
    if not pte.exists():
        return {"skipped": f"{pte} not found"}
    set_threads(cfg.get("llm_threads", 4))
    rss0 = rss_mb()
    llm = ExecuTorchLlama(pte, root / cfg["llm_tokenizer"])
    rss1 = rss_mb()
    record = {"board_id": "B-0042", "design": "deeppcb", "verdict": "FAIL", "defects": [
        {"id": "D1", "type": "short", "severity": "critical", "confidence": 0.91, "size_mm": 0.31,
         "location": "top-left", "ipc": {"class1": "reject", "class2": "reject", "class3": "reject"}},
        {"id": "D2", "type": "mouse_bite", "severity": "minor", "confidence": 0.64, "size_mm": 0.12,
         "location": "centre", "ipc": {"class1": "accept", "class2": "accept", "class3": "reject"}}]}
    prompt = build_prompt(record, {"short": 4}, default_kb())
    runs = []
    for _ in range(1 if quick else 3):
        g = llm.generate(prompt, 160)
        st = llm.last_stats
        ttft = st["first_token"] - st["inference_start"] if "first_token" in st else None
        runs.append({"wall_ms": round(g.latency_ms, 1), "first_token_ms": ttft, "prefill_tps": g.prefill_tps,
                     "decode_tps": g.decode_tps, "generated": g.n_generated,
                     "prompt_tokens": llm.last_stats.get("prompt_tokens"), "temp_c": temp_c()})
        print("llm", runs[-1], flush=True)
    return {"model": pte.name, "file_mb": round(pte.stat().st_size / 2**20, 1),
            "load_ms": round(llm.load_ms, 1), "rss_before_mb": round(rss0), "rss_after_load_mb": round(rss1),
            "rss_after_runs_mb": round(rss_mb()), "runs": runs,
            "wall_ms_median": statistics.median([r["wall_ms"] for r in runs]),
            "first_token_ms_median": statistics.median([r["first_token_ms"] for r in runs
                                                        if r["first_token_ms"] is not None] or [0]) or None,
            "decode_tps_median": statistics.median([r["decode_tps"] for r in runs if r["decode_tps"]]),
            "prefill_tps_median": statistics.median([r["prefill_tps"] for r in runs if r["prefill_tps"]])}


def bench_llm_scores(root: Path, cfg: dict, quick: bool) -> dict:
    """The decisions Llama makes through next-token scores (pcbinspect/llm_logits.py), timed per model:
    full advice for a two-defect board (causes judged one at a time, actions scored, number-safe summary),
    one customer rule, one shift-question routing. These are the latencies an operator actually waits for."""
    import sqlite3

    from pcbinspect import ask
    from pcbinspect.advisor import LLAMA_BACKEND, Advisor
    from pcbinspect.kb import default_kb
    from pcbinspect.llm_logits import LogitLlama
    from pcbinspect.spec_llm import ConstrainedSpecCompiler
    record = {"board_id": "B-0042", "design": "deeppcb", "verdict": "FAIL", "defects": [
        {"id": "D1", "type": "short", "severity": "critical", "confidence": 0.91, "size_mm": 0.31, "location": "top-left"},
        {"id": "D2", "type": "open", "severity": "critical", "confidence": 0.84, "size_mm": 0.22, "location": "centre"}]}
    out = {}
    for key in ("advice_llm_pte", "spec_llm_pte", "llm_pte"):
        pte = root / cfg.get(key, "")
        if not cfg.get(key) or not pte.is_file() or pte.name in out:
            continue
        rss0 = rss_mb()
        be = LogitLlama(pte, root / cfg["llm_tokenizer"], threads=cfg.get("llm_threads", 4))
        r = {"file_mb": round(pte.stat().st_size / 2**20, 1), "load_ms": round(be.load_ms, 1),
             "rss_added_mb": round(rss_mb() - rss0), "advice": [], "rule": [], "route": []}
        adv = Advisor(be, default_kb(), cause_mode="pointwise")
        comp = ConstrainedSpecCompiler(be, grounded=True)
        conn = sqlite3.connect(":memory:")
        for _ in range(1 if quick else 3):
            a = adv.advise(record, {"short": 5})
            r["advice"].append({"wall_ms": round(a.gen.latency_ms, 1) if a.gen else None, "forwards": adv.last_forwards,
                                "prompt_tokens": a.gen.prompt_tokens if a.gen else None,
                                "prefill_tps": round(a.gen.prefill_tps, 1) if a.gen else None,
                                "shown": a.backend == LLAMA_BACKEND, "temp_c": temp_c()})
            c = comp.compile("Reject mouse bites that remove more than 20% of the trace width.")
            r["rule"].append({"wall_ms": round(c.latency_ms, 1), "ok": c.rule is not None})
            t = time.perf_counter()
            qid, _ = ask.route_scored("How many boards failed in the last 3 hours?", be)
            r["route"].append({"wall_ms": round((time.perf_counter() - t) * 1e3, 1), "query": qid})
            print("llm scores", pte.name, r["advice"][-1], r["rule"][-1], r["route"][-1], flush=True)
        conn.close()
        for k in ("advice", "rule", "route"):
            r[f"{k}_ms_median"] = statistics.median([x["wall_ms"] for x in r[k] if x["wall_ms"]])
        out[pte.name] = r
        del be, adv, comp
    return out or {"skipped": "no Llama .pte found"}


def bench_e2e(root: Path, cfg: dict, n: int) -> dict:
    from pcbinspect.advisor import Advisor
    from pcbinspect.anomaly import executorch_feature_fn
    from pcbinspect.detect_bare import BareDetector, ExecuTorchTiles
    from pcbinspect.pipeline import Inspector, load_designs
    from pcbinspect.variants import apply_detector_variant
    cfg, designs, _ = apply_detector_variant(cfg, load_designs(root / cfg["designs_dir"]), root)
    det = BareDetector(ExecuTorchTiles(root / cfg["detector_pte"])) if (root / cfg["detector_pte"]).exists() else None
    fn = executorch_feature_fn(root / cfg["backbone_pte"]) if (root / cfg["backbone_pte"]).exists() else None
    insp = Inspector(root, designs, det, fn)
    out = {}
    for design in designs:
        imgs = sorted(glob.glob(str(root / "demo" / "boards" / f"{design}_*.jpg")))[:n]
        if not imgs:
            continue
        import cv2
        stages: dict[str, list] = {}
        for p in imgs:
            bgr = cv2.imread(p)
            _, a, _ = insp.inspect(bgr, design, log=False)
            for k, v in a.latency_ms.items():
                stages.setdefault(k, []).append(v)
        out[design] = {k: round(statistics.median(v), 2) for k, v in stages.items()}
        out[design]["boards"] = len(imgs)
        out[design]["image_size"] = list(cv2.imread(imgs[0]).shape[:2])
        print("e2e", design, out[design], flush=True)
    return out


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--config", default="config/app.yaml")
    ap.add_argument("--out", default="results/device_bench.json")
    ap.add_argument("--quick", action="store_true", help="few repetitions (simulator)")
    ap.add_argument("--skip", default="", help="comma list: vision,llm,e2e")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    cfg = yaml.safe_load((root / args.config).read_text())
    res = {"system": system_info(), "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(res["system"], flush=True)
    skip = set(args.skip.split(","))
    power = res["power"] = {"method": "sum of rail voltage x current from vcgencmd pmic_read_adc, sampled every "
                                      "0.5 s; an estimate that reads lower than a wall meter"}

    def measured(name, fn):
        with PowerLog() as pw:
            out = fn()
        power[name] = pw.summary()
        return out
    measured("idle", lambda: time.sleep(5))
    if "vision" not in skip:
        res["vision"] = measured("vision", lambda: bench_vision(root, [1, 2, 4], args.quick))
    if "runtimes" not in skip:
        set_threads(4)
        res["detector_runtimes"] = measured("detector_runtimes",
                                            lambda: bench_detector_runtimes(root, cfg, args.quick))
    if "e2e" not in skip:
        set_threads(4)
        res["end_to_end"] = measured("end_to_end", lambda: bench_e2e(root, cfg, 5 if args.quick else 20))
    if "llm" not in skip:
        res["llm"] = measured("llm_text", lambda: bench_llm(root, cfg, args.quick))
        try:
            res["llm_scores"] = measured("llm_decisions", lambda: bench_llm_scores(root, cfg, args.quick))
        except Exception as exc:    # noqa: BLE001  (record it: a missing number must be visible, not silent)
            res["llm_scores"] = {"error": repr(exc)}
            print("llm scores FAILED:", repr(exc), flush=True)
    res["system"]["temp_c_end"] = temp_c()
    res["system"]["throttled_end"] = throttled()
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
