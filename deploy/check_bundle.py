"""End-to-end check of a built bundle with the models it ships: inspect every held-out demo board (with its
golden image when one ships), compare the verdict with ground truth, and check that shown defects match real
ones. Hard-fails on a load error or a configuration mismatch (detector class count, anomaly input size versus
the backbone export, a design that does not load); otherwise prints the measured rates.

    python deploy/check_bundle.py --root <bundle>        # laptop (x86 ExecuTorch) after building, and on the Pi
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--config", default="config/app.yaml")
    ap.add_argument("--per-design", type=int, default=0, help="limit boards per design (0 = all)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    cfg = yaml.safe_load((root / a.config).read_text())

    import torch
    from executorch.runtime import Runtime
    from pcbinspect.anomaly import executorch_feature_fn
    from pcbinspect.detect_bare import BareDetector, ExecuTorchTiles
    from pcbinspect.pipeline import Inspector, load_designs
    from pcbinspect.record import BARE_CLASSES

    problems = []
    det_pte, bb_pte = root / cfg["detector_pte"], root / cfg["backbone_pte"]
    out = Runtime.get().load_program(str(det_pte)).load_method("forward").execute(
        [torch.zeros(1, 3, 640, 640)])[0]
    if out.shape[1] != 4 + len(BARE_CLASSES):
        problems.append(f"detector outputs {out.shape[1]} channels, expected {4 + len(BARE_CLASSES)}")
    det = BareDetector(ExecuTorchTiles(det_pte))
    fn = executorch_feature_fn(bb_pte)
    designs = load_designs(root / cfg["designs_dir"])
    insp = Inspector(root, designs, det, fn)
    for name, m in insp.anomaly.items():
        try:
            m.score(np.full((m.input_size, m.input_size, 3), 128, np.uint8))
        except Exception as exc:                      # noqa: BLE001
            problems.append(f"anomaly memory {name} does not run on the shipped backbone: {exc}")
    if problems:
        print("BUNDLE CHECK FAILED:\n  " + "\n  ".join(problems))
        return 1

    manifest = json.loads((root / "demo" / "manifest.json").read_text())
    by_design = defaultdict(list)
    for item in manifest:
        by_design[item["design"]].append(item)
    report = {}
    for design, items in sorted(by_design.items()):
        if design not in designs:
            problems.append(f"demo design {design} has no config")
            continue
        items = items[:a.per_design] if a.per_design else items
        verdicts, evidence = Counter(), Counter()
        found = total = shown_n = shown_real = 0
        lat = []
        for it in items:
            img = root / "demo" / "boards" / it["file"]
            bgr = cv2.imread(str(img))
            gpath = img.with_name(img.stem + ".golden.jpg")
            golden = cv2.imread(str(gpath)) if gpath.exists() else None
            ins, an, _ = insp.inspect(bgr, design, log=False, golden=golden)
            truth = "good" if it["good"] else "defective"
            verdicts[(truth, ins.verdict)] += 1
            evidence[ins.model_versions.get("evidence", "?")] += 1
            lat.append(an.latency_ms.get("total_to_verdict", 0.0))
            boxes = [b["bbox"] for b in it.get("boxes", [])]
            for b in boxes:
                total += 1
                found += any(overlaps(d.bbox, b) for d in ins.defects)
            for d in ins.defects:
                shown_n += 1
                shown_real += any(overlaps(d.bbox, b) for b in boxes)
        nd = sum(v for (t, _), v in verdicts.items() if t == "defective")
        ng = sum(v for (t, _), v in verdicts.items() if t == "good")
        r = {"boards": len(items), "defective": nd, "good": ng,
             "defective_passed": verdicts[("defective", "PASS")], "good_failed": verdicts[("good", "FAIL")],
             "sent_to_person": sum(v for (_, vv), v in verdicts.items() if vv == "REVIEW") / max(len(items), 1),
             "defects_shown": found / total if total else None, "n_defects": total,
             "shown_that_are_real": shown_real / shown_n if shown_n else None,
             "evidence": dict(evidence), "median_ms_this_machine": float(np.median(lat)) if lat else None}
        report[design] = r
        shown = "n/a" if r["defects_shown"] is None else f"{r['defects_shown']:.0%}"
        print(f"{design:10s} boards {r['boards']:3d} ({nd} defective, {ng} good): defective PASSED "
              f"{r['defective_passed']}, good FAILED {r['good_failed']}, to a person {r['sent_to_person']:.0%}, "
              f"defects shown {shown} of {total}, evidence {dict(evidence)}, "
              f"median {r['median_ms_this_machine']:.0f} ms", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
    if problems:
        print("BUNDLE CHECK FAILED:\n  " + "\n  ".join(problems))
        return 1
    print("BUNDLE CHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
