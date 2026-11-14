# Model files

The station looks for its models under `models/`. Only the small files are in git:

- `models/gate/`: the calibrated PASS/FAIL/CHECK thresholds, one JSON per design. The detector's own ones are at the top level; those for the other sizes are in `y26s/` and `y26m/`. `detector_variant` in `config/app.yaml` picks them together with the detector.
- `models/anomaly/`: the good-board patch memories for VisA pcb1-4 (`bank.npy` + `meta.json`).

The weights below are too large for git (the 3B Llama file alone is 2.4 GB). `deploy/make_bundle.py` copies them into the Pi bundle. To run the station from this repo, put each file at the path shown. Check it against its SHA-256 (`sha256sum -c`), or rebuild it with the command given.

## Files the station loads

| Path | Size | What it is | Built by |
|---|---|---|---|
| `models/vision/pcb_y26n_xnnpack.pte` | 9.8 MB | Bare-board detector, YOLO26n v3, 11 defect types, FP32 (default) | `training/colab_v3.sh`, then `training/export_vision.py --only detector` |
| `models/vision/pcb_y26n_xnnpack_int8.pte` | 3.2 MB | the same, INT8 | `training/colab/int8.sh` |
| `models/vision/pcb_y26s_xnnpack.pte` | 38.1 MB | YOLO26s v5, FP32 | `training/colab/job_v5.sh`, then `job_rebuild.sh` (export) |
| `models/vision/pcb_y26s_xnnpack_int8.pte` | 11.3 MB | the same, INT8 | `job_rebuild.sh` |
| `models/vision/pcb_y26m_xnnpack.pte` | 81.7 MB | YOLO26m v5, FP32 | `job_v5.sh`, then `job_rebuild.sh` |
| `models/vision/pcb_y26m_xnnpack_int8.pte` | 25.1 MB | the same, INT8 | `job_rebuild.sh` |
| `models/vision/resnet18_patch_xnnpack.pte` | 11.1 MB | ResNet-18 patch-feature backbone for the anomaly memory, FP32 | `training/export_vision.py --only backbone` |
| `models/vision/resnet18_patch_xnnpack_int8.pte` | 2.8 MB | the same, INT8 (benchmarked for speed only, section 7 of the report) | `training/export_vision.py --only backbone` |
| `models/vision/parts_y26n_xnnpack.pte` | 9.8 MB | One-class part detector for assembled boards (FPIC) | `training/colab/job_parts.sh`, then `job_parts2.sh` (export) |
| `models/vision/parts_kind_xnnpack.pte` | 6.2 MB | Part-kind classifier, used only while teaching a board | `job_parts.sh`, then `job_parts2.sh` |
| `models/vision/parts_kind_xnnpack.json` | 120 B | the classifier's class names | `job_parts2.sh` |
| `models/llm/llama3_2_3b_spinquant.pte` | 2.38 GB | Llama 3.2 3B Instruct, Meta's SpinQuant INT4 checkpoint, ExecuTorch XNNPACK (default advice model) | `HF_TOKEN=... bash llm/export_llama.sh 3B models/llm` |
| `models/llm/llama3_2_1b_spinquant.pte` | 1.08 GB | Llama 3.2 1B, the same way (fallback) | `HF_TOKEN=... bash llm/export_llama.sh 1B models/llm` |
| `models/llm/tokenizer.model` | 2.2 MB | Llama 3.2 tokenizer | written by `llm/export_llama.sh` |

The `.pt` training weights are kept as well, so that any export can be redone. The station never loads them:
`models/vision/pcb_y26n.pt` (5.5 MB), `pcb_y26s.pt` (20.3 MB), `pcb_y26m.pt` (44.1 MB), `parts_y26n.pt` (5.5 MB)
and `parts_kind.pt` (3.2 MB).

All exports use `executorch==1.6.0`, the version `deploy/pi_setup.sh` installs on the Pi, either on Colab
(`training/colab/README.md`) or in `deploy/Dockerfile.export`. Each export was checked against its trained
model: `results/measured/export_vision*.json` and `parts_export.json`. On aarch64 (arm64 emulator), all six
detector files load and give the same findings: `results/measured/sim_check_detectors.json`.

## SHA-256

```
2db757eeb74fe60721c4551d8a716c0a1731753631ad778c1f20091fc4d314e0  vision/pcb_y26n_xnnpack.pte
79b43fe2ef87f7fe498b1d48f00e1d1ab55af48e27656b75e758f900a716b528  vision/pcb_y26n_xnnpack_int8.pte
055a0db586b032df1e93dbf78d450e46b3d949f0dd163c66081642a42e1fc0b8  vision/pcb_y26s_xnnpack.pte
0f7ff72d35e8a0e3750f4ca06e2a73cd5fa842d4a429aab32a6396db99dc867c  vision/pcb_y26s_xnnpack_int8.pte
9af00e6b35998298782c178fdcc3fc88eead5ae24312bfc7cf80eafade9b8be0  vision/pcb_y26m_xnnpack.pte
83b39e144aa13a1522f99156cf0de5bbbe00ce8cf99a22fb7aeb3930d848901b  vision/pcb_y26m_xnnpack_int8.pte
e4e0386583383a0ee01b3439fd7c722799683dbfc4c38e34047ade8c7c4a3ab2  vision/resnet18_patch_xnnpack.pte
57eca1a90e127dca1726e7b50784fbed4603050b6bbdcc115635abd46a7d35c4  vision/resnet18_patch_xnnpack_int8.pte
55423a30ea4444f9effa60a7773545d7665bb78adef7cd65006dde9817e2b03d  vision/parts_y26n_xnnpack.pte
413d49b316cda80594eaeb58f9c1525c551e7493e7d4d696cc7532cb123dcbbc  vision/parts_kind_xnnpack.pte
dec633787b24e73c6b0401c299271bb08a2658bd321fab0b40986d98211ae0a1  vision/parts_kind_xnnpack.json
840fa099d30cb0c49f92bc81d9ecec8c35c492fbbae56540a5bea4c236e8a98e  llm/llama3_2_3b_spinquant.pte
f6065b057aa1a155fa934fbbf64e2df62ed6d840bdd1e6541cb60aebb94dff4e  llm/llama3_2_1b_spinquant.pte
82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55  llm/tokenizer.model
45cb29813bab5b52a6f617d30e433530033bc5eea9f0863609412a8f0142e14f  vision/pcb_y26n.pt
2df2322a58878d641cc552bec72a46b1fb3ea2cb891366291d15d3bb8abd784b  vision/pcb_y26s.pt
38149dfa360495917699f4ea8fa237b0b3bcdf1fbc3333023fd1c378f81672f4  vision/pcb_y26m.pt
a4ef64c2d878632e06627009ee7d203cc2a3530451e288a8a4eef118b776d50d  vision/parts_y26n.pt
5bbae4366733ab4d53f4cad6adb2512eeb83d6f009ed409d632bb5b524a971ff  vision/parts_kind.pt
```

Run `cd models && sha256sum -c` with the block above saved as a file.

## Licences

YOLO26 weights are fine-tuned from Ultralytics' pretrained weights (AGPL-3.0). Llama 3.2 files are covered by the
Llama 3.2 Community License. The training data is listed in `NOTICE.md`; Real-IAD was used for testing only and
went into none of these files.
