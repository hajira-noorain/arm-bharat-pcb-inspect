"""Functional check of the deployed artefacts on an Arm (aarch64) Python: run in the Pi simulator
or as the first thing on the real Pi.  python deploy/arm_check.py --root <bundle-or-work-dir>"""
from __future__ import annotations

import argparse
import glob
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--image", default=None)
    ap.add_argument("--llm-tokens", type=int, default=16)
    a = ap.parse_args()
    root = Path(a.root)
    import executorch
    print("arch:", platform.machine(), "python:", platform.python_version(),
          "executorch:", getattr(executorch, "__version__", "?"), flush=True)

    from pcbinspect.detect_bare import BareDetector, ExecuTorchTiles
    img = a.image or sorted(glob.glob(str(root / "demo" / "boards" / "deeppcb_*.jpg")))[0]
    bgr = cv2.imread(img)
    det = BareDetector(ExecuTorchTiles(root / "models" / "vision" / "pcb_y26n_xnnpack.pte"))
    t = time.perf_counter()
    dets, lat = det.detect(bgr)
    print(f"detector: {len(dets)} detections in {(time.perf_counter() - t) * 1e3:.0f} ms on {Path(img).name}; "
          f"top: {[(d.cls, round(d.conf, 2), d.bbox) for d in dets[:3]]}", flush=True)

    from pcbinspect.anomaly import executorch_feature_fn, preprocess
    fn = executorch_feature_fn(root / "models" / "vision" / "resnet18_patch_xnnpack.pte")
    f = fn(preprocess(bgr))
    print("backbone features:", f.shape, "finite:", bool(np.isfinite(f).all()), flush=True)

    # Llama on ExecuTorch is MANDATORY for PS5: a missing model, a failed load or empty output is a hard
    # failure here (the station itself would silently fall back to template advice).
    if a.llm_tokens > 0:
        from pcbinspect.llm import ExecuTorchLlama
        p = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nIn one sentence, what is a "
             "short circuit on a PCB?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        for label, name, required in (("small model, the fallback", "llama3_2_1b_spinquant.pte", True),
                                      ("advice, questions and rule reader", "llama3_2_3b_spinquant.pte", False)):
            pte = root / "models" / "llm" / name
            if not pte.exists():
                if required:
                    sys.exit(f"ARM CHECK FAILED: Llama {label} model missing at {pte}")
                print(f"llama {label}: {name} not shipped (optional)", flush=True)
                continue
            try:
                llm = ExecuTorchLlama(pte, root / "models" / "llm" / "tokenizer.model")
                g = llm.generate(p, a.llm_tokens)
            except Exception as exc:   # noqa: BLE001
                sys.exit(f"ARM CHECK FAILED: Llama {label} did not run on ExecuTorch: {exc}")
            if not g.text.strip():
                sys.exit(f"ARM CHECK FAILED: Llama {label} produced no text")
            print(f"llama {label} ({llm.load_ms:.0f} ms load): {g.text!r}  decode {g.decode_tps} tok/s", flush=True)
            del llm
    # The advice model the config names (the fine-tuned 1B): one board through the station's evidence-mode advisor,
    # read from next-token scores like on the station. It must choose knowledge-base causes with probabilities.
    if a.llm_tokens > 0:
        import yaml
        cfg = yaml.safe_load((root / "config" / "app.yaml").read_text())
        adv_pte = root / str(cfg.get("advice_llm_pte", ""))
        if cfg.get("advice_cause_mode") == "evidence" and adv_pte.is_file():
            from pcbinspect.advisor import LLAMA_BACKEND, Advisor
            from pcbinspect.kb import default_kb
            from pcbinspect.llm_logits import LogitLlama
            ev = {"repeat_same_spot": {"boards": 6, "of": 10}, "repeat_in_line": {"boards": 0, "of": 10},
                  "near_edge": False, "shape": "long_thin", "count_on_board": 1, "trend": "steady", "central": True,
                  "roll_pitch": False, "co_occurring": []}
            rec = {"board_id": "ARM-CHECK", "design": "check", "verdict": "FAIL", "defects": [
                {"id": "D1", "type": "open", "severity": "critical", "size_mm": 0.4, "location": "centre",
                 "confidence": 0.9, "ipc": {}, "evidence": ev}]}
            try:
                ll = LogitLlama(adv_pte, root / "models" / "llm" / "tokenizer.model")
                t = time.perf_counter()
                adv = Advisor(ll, default_kb(), cause_mode="evidence", why=bool(cfg.get("advice_why", True)),
                              why_key=bool(cfg.get("advice_why_key", False)),
                              temperature=float(cfg.get("advice_temperature", 1.0))).advise(rec)
                ms = (time.perf_counter() - t) * 1e3
            except Exception as exc:   # noqa: BLE001
                sys.exit(f"ARM CHECK FAILED: advice model {adv_pte.name} did not run: {exc}")
            s = adv.steps[0] if adv.steps else {}
            if adv.backend != LLAMA_BACKEND or not s.get("p"):
                sys.exit(f"ARM CHECK FAILED: advice model {adv_pte.name} fell back to standard advice "
                         f"({adv.rejected_reasons})")
            print(f"advice model {adv_pte.name}: {s['cause_id']} at {s['p'][s['cause_id']]:.2f}, "
                  f"why ({s.get('why_by')}): {s.get('why')!r}, {ms:.0f} ms", flush=True)
            del ll
    # Constrained decoding for the rule reader: the same .pte through executorch.runtime must return scores
    # for every token, and a closed choice must come out right.
    if a.llm_tokens > 0:
        from pcbinspect.llm_logits import LogitLlama
        big = root / "models" / "llm" / "llama3_2_3b_spinquant.pte"
        pte = big if big.exists() else root / "models" / "llm" / "llama3_2_1b_spinquant.pte"
        try:
            ll = LogitLlama(pte, root / "models" / "llm" / "tokenizer.model")
            t = time.perf_counter()
            ll.feed("<|begin_of_text|>The capital of France is")
            pick, margin = ll.choose([" Berlin", " Paris", " Madrid"])
            ms = (time.perf_counter() - t) * 1e3
        except Exception as exc:   # noqa: BLE001
            sys.exit(f"ARM CHECK FAILED: constrained decoding did not run on {pte.name}: {exc}")
        if pick != " Paris":
            sys.exit(f"ARM CHECK FAILED: constrained decoding chose {pick!r}")
        print(f"llama constrained choice on {pte.name}: {pick.strip()} (margin {margin:.1f}) in {ms:.0f} ms", flush=True)
        # The station keeps Llama loaded and runs each inspection in a web-server worker thread. That exact
        # situation once crashed the process (segmentation fault), so it is checked here, not assumed.
        import threading
        box = {}

        def worker():
            try:
                box["n"] = len(det.detect(bgr)[0])
            except Exception as exc:   # noqa: BLE001
                box["err"] = repr(exc)
        th = threading.Thread(target=worker)
        th.start()
        th.join(120)
        if th.is_alive() or "n" not in box:
            sys.exit(f"ARM CHECK FAILED: detector in a worker thread with Llama loaded: {box.get('err', 'hung')}")
        print(f"detector in a worker thread with Llama loaded: {box['n']} detections", flush=True)
        del ll
    print("ARM CHECK OK")


if __name__ == "__main__":
    main()
