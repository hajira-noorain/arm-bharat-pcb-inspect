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
        for label, name, required in (("advice", "llama3_2_1b_spinquant.pte", True),
                                      ("rule reader", "llama3_2_3b_spinquant.pte", False)):
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
    print("ARM CHECK OK")


if __name__ == "__main__":
    main()
