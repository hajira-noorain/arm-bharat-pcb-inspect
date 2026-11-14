# Results

`measured/` holds every result file the experiments wrote: the JSON (and a few CSV) files behind each number
in `docs/RESULTS.md` and `docs/TECHNICAL_REPORT.md`, and the run logs (`log_run_*.txt`). Nothing in it is
typed by hand. `python docs/make_results.py` reads this folder and `models/gate/` and regenerates
`docs/RESULTS.md`.

How the files are named:

| Prefix | What it measured | Made by |
|---|---|---|
| `bare_eval_<tag>` | Bare-board detector on held-out boards (mAP, per class, the PASS/FAIL/CHECK gate); `_et` means the exported ExecuTorch file was scored, `_int8` its INT8 version | `training/eval_bare.py` |
| `bare_dets_<split>_<tag>` | the raw detections behind those scores | `training/eval_bare.py` |
| `golden_eval_deeppcb_<tag>` | Golden-board comparison and fusion, good boards included | `training/eval_golden.py` |
| `external_*` | Cross-lab boards (DefectDet, EPID) | `training/eval_external.py` |
| `pod_insilico*` | Detection limits under three simulated camera conditions | `training/pod_insilico.py` |
| `exp_learning_*` | Learning from operator feedback, simulated on held-out boards | `training/exp_learning.py` |
| `trace_scale_*`, `kicad_manifest` | Trace-width scale calibration; the open-hardware boards rendered for synthetic tiles (with licences) | `training/calibrate_trace_scale.py`, `training/kicad/` |
| `anomaly_pcb*_<tag>`, `realiad_pcb_*` | Assembled-board anomaly memory on VisA and Real-IAD (Real-IAD: scores only, test use only) | `training/build_visa_banks.py`, `eval_realiad.py` |
| `parts_*` | Parts check: part detector and classifier on held-out FPIC boards, simulated removal, VisA, export check, chosen rule | `training/colab/job_parts*.sh`, `eval_parts*.py`, `tune_parts_rules.py` |
| `eyetest_sim*` | Detection limit (a90/95) read from simulated photos of the eye-test card | `training/eyetest_sim.py` |
| `advice_*`, `ask_*`, `spec_*` | Llama: advice choices and grounding, shift questions, customer-rule compiling | `llm/eval_advice.py`, `eval_ask.py`, `eval_spec.py` |
| `export_vision*` | Each ExecuTorch export compared with its trained model | `training/export_vision.py` |
| `sim_check_detectors` | All six detector files on aarch64 (arm64 emulator): load and findings | `deploy/sim_check_detectors.py` |
| `split_fingerprint*` | Proof that rebuilt datasets reuse the same held-out boards | `training/split_fingerprint.py` |
| `y26*_results.csv`, `parts_det_results.csv` | Per-epoch training curves (Ultralytics) | the training jobs |
| `gate_scores_*` | Anomaly scores the VisA gate was calibrated on | `training/build_visa_banks.py` |

Raspberry Pi timings are not here yet: `deploy/measure_on_pi.sh` writes them on the device (20-21 Nov), and
`docs/make_results.py --device <that folder>` adds them to `docs/RESULTS.md`.
