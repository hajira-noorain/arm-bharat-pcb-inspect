# Colab jobs

These are the exact shell jobs that produced the shipped models and the files in `results/measured/`, run on a
Colab A100 runtime. They are kept as a record of what was run, in what order and with which settings. They are
not a polished pipeline: paths are Colab's (`/content/...`) and each job expects the repo zipped and uploaded
as `/content/codeNN.zip` (the number only kept runs apart).

The earlier detector jobs (v1 to v4) are one level up: `training/colab_bare.sh`, `colab_v2.sh`, `colab_v3.sh`,
`colab_v4.sh`. The detector that ships first, YOLO26n v3, comes from `colab_v3.sh`.

## Two Python environments on the runtime

- Colab's own Python, with `ultralytics`, for training and for scoring `.pt` files.
- `/content/etv`, a `uv` venv with `executorch==1.6.0` (the version the Pi installs), for every ExecuTorch
  export and every score of an exported `.pte`. `job_adv.sh` shows how it is created.

## Data

Every dataset is fetched from its original host onto the runtime; none is stored in this repo
(`NOTICE.md` lists them with their licences).

| Job | What it does |
|---|---|
| `fetch_ext.py` | DefectDet and EPID (checksums checked against the published values). Real-IAD `pcb` only if the `HF_TOKEN` Colab secret is set; the token is read from the environment and never printed. Real-IAD is for testing only. |
| `fetch_fpic.sh` | FPIC (University of Florida) from the file links on its own page, 7.2 GB of images. |

## Bare-board detectors

| Job | What it does | Writes |
|---|---|---|
| `job_v5.sh` | Rebuilds the v3 dataset, proves the cal/test split is the one every earlier number used (fingerprint), re-scores the v3 weights as a check, then trains YOLO26s and YOLO26m and scores them on the same held-out boards. | `pcb_y26s_v5.pt`, `pcb_y26m_v5.pt`, `bare_eval_v5s.json`, `bare_eval_v5m.json` |
| `job_export_v5.sh` | ExecuTorch FP32 + INT8 export of s and m; re-scores the exported FP32 file and calibrates PASS/FAIL/CHECK on its scores; INT8 accuracy; the golden-board gate. | `pcb_y26{s,m}_xnnpack[_int8].pte`, `models/gate/y26{s,m}/` |
| `job_rebuild.sh` | Same steps as `job_export_v5.sh` after a runtime reset, with every input re-fetched or re-uploaded with a checksum; then the parts check on FPIC and VisA. This is the run the shipped s/m files and gates come from. | as above, plus `parts_check_*.json` |
| `int8.sh` | INT8 export of the v3 YOLO26n. | `pcb_y26n_xnnpack_int8.pte` |
| `job_golden.sh` | Golden-mode thresholds for n, s and m, with good boards included. | `golden_eval_deeppcb_*.json`, `models/gate/**/deeppcb_golden.json` |
| `job_card.sh` | Detection-limit (eye-test) card on simulated photos for s and m. | `eyetest_sim_v5{s,m}.json` |

## Assembled boards

| Job | What it does | Writes |
|---|---|---|
| `colab_anom.sh` | Anomaly-memory sweep on VisA pcb1-4 (input size, per-position normalisation, foreground mask, top-k score). | `anomaly_pcb*_<tag>.json` |
| `anom_final.sh` | The chosen setting (ResNet-18, 320 px, top 1% score): deployable memories and gates. | `models/anomaly/pcb*/`, `models/gate/visa_pcb*.json` |
| `job_parts.sh` | FPIC to a one-class part detector and a part-kind classifier, both scored on boards held out by physical board. | `parts_y26n.pt`, `parts_kind.pt`, `parts_detector_test.json`, `parts_classifier_test.json` |
| `job_parts2.sh` | Simulated part removal on held-out boards; ExecuTorch export of both models and a check that the `.pte` files answer like the trained ones. | `parts_y26n_xnnpack.pte`, `parts_kind_xnnpack.pte/.json`, `parts_export.json`, `parts_check_eval.json` |
| `job_dump2.sh` | Raw parts-check measurements on VisA, one process per design, used to choose the rule on good training photos only (`training/tune_parts_rules.py`). | `parts_visa_raw_pcb*.json` (raw, not kept); the chosen rule is in `parts_rules_visa.json` |
| `job_parts8.sh` | The station's own parts-check code, end to end, once, with the chosen rule and eight teaching photos. | `parts_check_eval.json`, `parts_check_visa.json` |

## Llama

The `.pte` files come from `llm/export_llama.sh` (Meta's SpinQuant INT4 checkpoints, gated on Hugging Face).

| Job | What it does | Writes |
|---|---|---|
| `job_adv.sh` | Builds the ExecuTorch venv and the 1B `.pte` if missing, then 1B advice: text runner against scored choice. `$1` is the record seed (1 development, 2 held-out). | `advice_runner_seed*.json`, `advice_logits_*_seed*.json` |
| `job_ask3b.sh` | Shift questions, 3B choosing the query. | `ask_logits3b_*.json` |
| `job_dev10.sh` | Development run after the summary fixes: 3B advice (one cause at a time) and shift questions. | `advice_point_3b_seed1_v2.json`, `ask_logits3b_tuning_v2.json` |
| `job_held10.sh` | The held-out run, once, with the method fixed on the development sets. | `advice_point_3b_seed2.json`, `ask_logits3b_heldout.json` |

## After a run

The JSON files go into `results/measured/`, the model files to the locations listed in `models/MODELS.md`, and
`python docs/make_results.py` regenerates `docs/RESULTS.md`. No secret is written into any job: the Hugging
Face token lives only in Colab Secrets and in the gitignored `.env`.
