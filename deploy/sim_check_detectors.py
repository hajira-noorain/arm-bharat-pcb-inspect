"""Do all bare-board detector files load and run on Arm (aarch64)? Run inside the arm64 simulator image:

    docker run --rm --platform linux/arm64 -v <bundle>:/b pcb-pi-sim python /b/deploy/sim_check_detectors.py

This proves the exported files work on the Pi's instruction set and give the same findings. Its times come from
software emulation and are NOT Pi times; they only hint at relative cost. Real times: deploy/measure_on_pi.sh.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import cv2  # noqa: E402

from pcbinspect.detect_bare import BareDetector, ExecuTorchTiles  # noqa: E402


def main() -> int:
    img_path = root / "demo" / "boards" / "deeppcb_010.jpg"
    img = cv2.imread(str(img_path))
    out = {"arch": platform.machine(), "board": img_path.name,
           "note": "emulated: times are not Pi times; findings and load success are what this checks", "runs": {}}
    failed = False
    for v in ("n", "s", "m"):
        for q in ("", "_int8"):
            name = f"pcb_y26{v}_xnnpack{q}.pte"
            p = root / "models" / "vision" / name
            if not p.is_file():
                out["runs"][name] = {"skipped": "not in bundle"}
                continue
            try:
                t = time.perf_counter()
                det = BareDetector(ExecuTorchTiles(p))
                load_s = time.perf_counter() - t
                t = time.perf_counter()
                d, _ = det.detect(img)
                run_s = time.perf_counter() - t
                out["runs"][name] = {"loaded": True, "findings": [(x.cls, round(x.conf, 2)) for x in d],
                                     "load_s_emulated": round(load_s, 1), "run_s_emulated": round(run_s, 1)}
            except Exception as exc:   # noqa: BLE001
                out["runs"][name] = {"loaded": False, "error": repr(exc)[:300]}
                failed = True
            print(name, out["runs"][name], flush=True)
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "sim_check_detectors.json").write_text(json.dumps(out, indent=1))
    print("SIM CHECK", "FAILED" if failed else "OK", "on", out["arch"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
