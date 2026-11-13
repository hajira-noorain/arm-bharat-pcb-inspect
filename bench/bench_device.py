"""On-device benchmark suite (run ON the Raspberry Pi 5; also runs in the arm64 simulator).

Measures, with the conditions recorded alongside every number:
  - system: board model, CPU, RAM, OS, Python, ExecuTorch version, temperature, throttle flags
  - vision: detector tile latency and anomaly backbone latency at 1/2/4 threads, FP32 vs INT8
  - Llama: load time, prefill & decode tokens/s from the runtime's own stats, RSS after load
  - end-to-end: per-stage latency of real inspections on the demo boards (capture->verdict, ->advice)

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


def bench_vision(root: Path, threads: list[int], quick: bool) -> dict:
    out = {}
    rng = np.random.default_rng(0)
    cases = [("detector_fp32", "models/vision/pcb_y26n_xnnpack.pte", (1, 3, 640, 640)),
             ("detector_int8", "models/vision/pcb_y26n_xnnpack_int8.pte", (1, 3, 640, 640)),
             ("backbone_fp32", "models/vision/resnet18_patch_xnnpack.pte", (1, 3, 256, 256)),
             ("backbone_int8", "models/vision/resnet18_patch_xnnpack_int8.pte", (1, 3, 256, 256))]
    for name, rel, shape in cases:
        p = root / rel
        if not p.exists():
            out[name] = {"skipped": "file not found"}
            continue
        x = rng.random(shape, dtype=np.float32)
        out[name] = {}
        for n in threads:
            got = set_threads(n)
            out[name][f"threads_{got}"] = time_method(p, x, 2 if quick else 5, 5 if quick else 30)
            out[name][f"threads_{got}"]["temp_c"] = temp_c()
            print(name, got, out[name][f"threads_{got}"], flush=True)
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
        runs.append({"wall_ms": round(g.latency_ms, 1), "prefill_tps": g.prefill_tps,
                     "decode_tps": g.decode_tps, "generated": g.n_generated,
                     "prompt_tokens": llm.last_stats.get("prompt_tokens"), "temp_c": temp_c()})
        print("llm", runs[-1], flush=True)
    return {"model": pte.name, "file_mb": round(pte.stat().st_size / 2**20, 1),
            "load_ms": round(llm.load_ms, 1), "rss_before_mb": round(rss0), "rss_after_load_mb": round(rss1),
            "rss_after_runs_mb": round(rss_mb()), "runs": runs,
            "decode_tps_median": statistics.median([r["decode_tps"] for r in runs if r["decode_tps"]]),
            "prefill_tps_median": statistics.median([r["prefill_tps"] for r in runs if r["prefill_tps"]])}


def bench_e2e(root: Path, cfg: dict, n: int) -> dict:
    from pcbinspect.advisor import Advisor
    from pcbinspect.anomaly import executorch_feature_fn
    from pcbinspect.detect_bare import BareDetector, ExecuTorchTiles
    from pcbinspect.pipeline import Inspector, load_designs
    designs = load_designs(root / cfg["designs_dir"])
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
    if "vision" not in skip:
        res["vision"] = bench_vision(root, [1, 2, 4], args.quick)
    if "e2e" not in skip:
        set_threads(4)
        res["end_to_end"] = bench_e2e(root, cfg, 5 if args.quick else 20)
    if "llm" not in skip:
        res["llm"] = bench_llm(root, cfg, args.quick)
    res["system"]["temp_c_end"] = temp_c()
    res["system"]["throttled_end"] = throttled()
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
