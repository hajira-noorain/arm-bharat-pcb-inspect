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

    pte = root / "models" / "llm" / "llama3_2_1b_spinquant.pte"
    if pte.exists() and a.llm_tokens > 0:
        from pcbinspect.llm import ExecuTorchLlama
        llm = ExecuTorchLlama(pte, root / "models" / "llm" / "tokenizer.model")
        p = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nIn one sentence, what is a "
             "short circuit on a PCB?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        g = llm.generate(p, a.llm_tokens)
        print(f"llama ({llm.load_ms:.0f} ms load): {g.text!r}  decode {g.decode_tps} tok/s", flush=True)
    print("ARM CHECK OK")


if __name__ == "__main__":
    main()
