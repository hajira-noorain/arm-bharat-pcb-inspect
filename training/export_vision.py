"""Export the vision models to ExecuTorch (.pte, XNNPACK delegate) — run inside the pcb-export container.

  backbone  ResNet18 patch-feature extractor (anomaly + learning embeddings), FP32 and INT8 (PT2E)
  detector  YOLO26n bare-board defect detector via Ultralytics' ExecuTorch exporter, FP32 (+INT8 if supported)

Also writes a parity report: max |PyTorch - ExecuTorch| on real images, so the deployed numerics
are checked, not assumed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcbinspect.anomaly import INPUT, build_torch_backbone, preprocess  # noqa: E402


def lower_xnnpack(module: torch.nn.Module, example: tuple, out: Path) -> None:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    ep = torch.export.export(module, example)
    prog = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    out.write_bytes(prog.buffer)


def quantize_int8(module: torch.nn.Module, example: tuple, calib: list[np.ndarray]) -> torch.nn.Module:
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config)
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    q = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
    m = torch.export.export(module, example).module()
    m = prepare_pt2e(m, q)
    with torch.no_grad():
        for x in calib:
            m(torch.from_numpy(x))
    return convert_pt2e(m)


def run_pte(path: Path, x: np.ndarray) -> np.ndarray:
    from executorch.runtime import Runtime
    prog = Runtime.get().load_program(str(path))
    m = prog.load_method("forward")
    return m.execute([torch.from_numpy(x)])[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/work")
    ap.add_argument("--yolo", default="/work/runs/y26n/weights/best.pt")
    ap.add_argument("--calib-glob", default="/work/data/VisA/pcb*/Data/Images/Normal/*.JPG")
    ap.add_argument("--only", default="backbone,detector")
    ap.add_argument("--name", default="pcb_y26n", help="detector file names: <name>.pt, <name>_xnnpack[_int8].pte")
    args = ap.parse_args()
    work = Path(args.work)
    out_dir = work / "models" / "vision"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    calib_paths = sorted(Path("/").glob(args.calib_glob.lstrip("/")))[::40][:32]
    calib = [preprocess(cv2.imread(str(p))) for p in calib_paths]

    if "backbone" in args.only:
        net = build_torch_backbone()
        ex = (torch.zeros(1, 3, INPUT, INPUT),)
        fp32 = out_dir / "resnet18_patch_xnnpack.pte"
        lower_xnnpack(net, ex, fp32)
        int8 = out_dir / "resnet18_patch_xnnpack_int8.pte"
        lower_xnnpack(quantize_int8(net, ex, calib), ex, int8)
        diffs, diffs8 = [], []
        with torch.no_grad():
            for x in calib[:8]:
                ref = net(torch.from_numpy(x)).numpy()
                diffs.append(float(np.abs(run_pte(fp32, x) - ref).max() / (np.abs(ref).max() + 1e-9)))
                diffs8.append(float(np.abs(run_pte(int8, x) - ref).mean() / (np.abs(ref).mean() + 1e-9)))
        report["backbone"] = {"fp32_pte": fp32.name, "fp32_bytes": fp32.stat().st_size,
                              "fp32_max_rel_err_vs_torch": max(diffs),
                              "int8_pte": int8.name, "int8_bytes": int8.stat().st_size,
                              "int8_mean_rel_err_vs_torch": float(np.mean(diffs8))}
        print("backbone", report["backbone"], flush=True)

    if "detector" in args.only:
        from ultralytics import YOLO
        yolo = Path(args.yolo)
        local = out_dir / f"{args.name}.pt"
        shutil.copy(yolo, local)
        # INT8 calibration reads the val tiles: point a container-local data yaml at them
        import yaml
        dy = yaml.safe_load((work / "data" / "bare_yolo" / "data.yaml").read_text())
        dy["path"] = str(work / "data" / "bare_yolo")
        cyaml = work / "data" / "bare_yolo" / "data_container.yaml"
        cyaml.write_text(yaml.safe_dump(dy))
        for int8 in (False, True):
            try:
                kw = {"format": "executorch", "imgsz": 640}
                if int8:
                    kw["int8"] = True
                    kw["data"] = str(cyaml)
                exported = Path(YOLO(str(local)).export(**kw))
                pte = next(exported.rglob("*.pte")) if exported.is_dir() else exported
                name = f"{args.name}_xnnpack_int8.pte" if int8 else f"{args.name}_xnnpack.pte"
                shutil.copy(pte, out_dir / name)
                x = np.random.default_rng(0).random((1, 3, 640, 640), dtype=np.float32)
                o = run_pte(out_dir / name, x)
                report[f"detector_{'int8' if int8 else 'fp32'}"] = {
                    "pte": name, "bytes": (out_dir / name).stat().st_size, "output_shape": list(o.shape)}
                print("detector", int8, report[f"detector_{'int8' if int8 else 'fp32'}"], flush=True)
            except Exception as exc:  # INT8 support depends on the exporter version: record, don't hide
                report[f"detector_{'int8' if int8 else 'fp32'}"] = {"error": repr(exc)[:500]}
                print("detector export failed", int8, repr(exc)[:500], flush=True)

    (work / "results").mkdir(exist_ok=True)
    (work / "results" / f"export_vision_{args.name}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
