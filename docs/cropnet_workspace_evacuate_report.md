# CropNet Workspace Evacuation Report

## 1. Project identity

- Project name: COS40007 CropNet final project / CropNet blank-fill forecasting workflow
- Main objective: monthly 35-feature AG/NDVI/weather forecasting, recursive blank-fill forecasting, model comparison, seasonal-residual learning, SARIMA and ensemble evaluation, and feature-group ablations
- Known active folders under `/home/student/YouZheng/`:
  - No primary CropNet repo found under `/home/student/YouZheng/`
  - `/home/student/YouZheng/` is dominated by unrelated Taotern and gamma projects
  - CropNet-relevant work was not discovered there beyond general user-home overlap
- Relevant folders outside `/home/student/YouZheng/`:
  - `/home/student/CROPNET/` - active CropNet workspace and outputs
  - `/dev/shm/feature_ablation_v1/` - temporary feature ablation results
  - `/home/student/CROPNET/cleanup_archives/` - preserved copy of temporary ablation report artifacts

## 2. Git repositories

### CropNet-related git state

| Path | Remote | Branch | HEAD | Status | Untracked | Ignored | Ahead/Behind | Recommended sync action |
|---|---|---|---|---|---:|---:|---|---|
| `/home/student` | none | `master` | no commits | dirty | 50 | 292 | n/a | Treat as accidental repo init. Do not use as project repo. Back up code first, then remove or ignore during reset only after user review. |
| `/home/student/CROPNET/Crop-Net` | none | `master` | no commits | dirty | 50 | 292 | n/a | Not an independent clone. `git rev-parse --show-toplevel` resolves to `/home/student`. Must not be treated as synced GitHub state. |

Notes:
- `git -C /home/student/CROPNET/Crop-Net rev-parse --show-toplevel` returned `/home/student`
- `git -C /home/student/CROPNET/Crop-Net rev-parse --git-dir` returned `/home/student/.git`
- No current remote matching `perrywsle/Crop-Net` was found under `/home/student/YouZheng/`, `/home/student/CROPNET/`, or `/home/student/`
- Current CropNet code under `/home/student/CROPNET/` is therefore not safely represented by a proper synced project repository

### Other repos found under `/home/student/YouZheng/`

These appear unrelated to the CropNet/COS40007 project and should be reviewed separately before any whole-workspace reset.

| Path | Remote | Branch | HEAD | Status | Ahead/Behind | Recommended sync action |
|---|---|---|---|---|---|---|
| `/home/student/YouZheng/Taotern_VM_int8_10e_benchmark` | `git@github-taotern-vm:StarMists/Taotern_VM.git` | `main` | `ddb1830...` | clean | `0/0` | No CropNet action. User review only. |
| `/home/student/YouZheng/Taotern_VM_int8_train_check` | `git@github-taotern-vm:StarMists/Taotern_VM.git` | detached/blank | `f84a629...` | clean | no upstream | User review. |
| `/home/student/YouZheng/Taotern_VM_int8_train_hook_verify` | `git@github-taotern-vm:StarMists/Taotern_VM.git` | detached/blank | `f84a629...` | dirty | no upstream | User review before reset. |
| `/home/student/YouZheng/Taotern_VM_int8_verify` | `git@github-taotern-vm:StarMists/Taotern_VM.git` | detached/blank | `358f6b2...` | dirty | no upstream | User review before reset. |
| `/home/student/YouZheng/Taotern_VM_repro_check` | `git@github-taotern-vm:StarMists/Taotern_VM.git` | detached/blank | `f3bc102...` | dirty | no upstream | User review before reset. |
| `/home/student/YouZheng/Taotern_VM_stage0` | local bundle `/home/student/YouZheng/taotern_vm_stage0.bundle` | detached/blank | `340ca2d...` | dirty | no upstream | Local-only sync path. User review. |
| `/home/student/YouZheng/Taotern_VM_throughput` | `https://github.com/StarMists/Taotern_VM.git` | detached/blank | `f3bc102...` | dirty | no upstream | User review before reset. |
| `/home/student/YouZheng/gamma_SSM_S4_enhanced` | `https://github.com/StarMists/gamma_SSM_S4_enhanced.git` | `codex/strict-ternary-runtime-snapshot` | `0330997...` | dirty | no upstream | User review before reset. |
| `/home/student/YouZheng/gamma_SSM_S4_enhanced_clean` | `git@github-gamma-ssm:StarMists/gamma_SSM_S4_enhanced.git` | `codex/strict-ternary-runtime-snapshot` | `0330997...` | clean | `0/0` | Clean unrelated repo. No CropNet action. |
| `/home/student/YouZheng/gamma_ssm_repo` | local bundle `/home/student/YouZheng/.repobridge-transfers/repobridge-1777378379.bundle` | `main` | `5844c3f...` | dirty | ahead 48 | Strongly user review. Not CropNet. |
| `/home/student/YouZheng/repo` | local bundle `/home/student/YouZheng/.repobridge-transfers/repobridge-1777378192.bundle` | `codex/taonet-ssm-core` | `c52eb8d...` | dirty | ahead 26 | Strongly user review. Not CropNet. |

## 3. Important non-git files

These are the main CropNet files that appear to contain active project state but are not safely represented by a proper synced GitHub repo today.

### Source and workflow scripts

| Path | Size | Purpose | Recommendation |
|---|---:|---|---|
| `/home/student/CROPNET/cropnet_feature_forecasting_v12_server.py` | 164 KB | Main forecasting, evaluation, blank-fill, SARIMA, ensemble, and ablation pipeline | GitHub after review |
| `/home/student/CROPNET/summarize_experiments.py` | 8 KB | Experiment summary builder | GitHub after review |
| `/home/student/CROPNET/make_report_artifacts.py` | 36 KB | Report artifact generator | GitHub after review |
| `/home/student/CROPNET/generate_loss_plots.py` | 2.5 KB | Loss curve plotting | GitHub after review |
| `/home/student/CROPNET/analyze_blank_fill_diagnostics.py` | 36 KB | Scaling audit and diagnostics | GitHub after review |
| `/home/student/CROPNET/run_controlled_model_comparison.py` | 12 KB | Controlled comparison runner | GitHub after review |
| `/home/student/CROPNET/build_broader_blank_fill_ensemble.py` | 32 KB | Broader ensemble builder | GitHub after review |
| `/home/student/CROPNET/scripts/run_controlled_model_comparison.py` | 12 KB | Script copy/helper | User review for deduplication, then GitHub if canonical |
| `/home/student/CROPNET/scripts/build_broader_blank_fill_ensemble.py` | 33 KB | Script copy/helper | User review for deduplication, then GitHub if canonical |
| `/home/student/CROPNET/scripts/run_feature_ablation_matrix.sh` | 3.1 KB | Feature ablation runner | GitHub after review |
| `/home/student/CROPNET/scripts/summarize_feature_ablation.py` | 15 KB | Feature ablation summarizer | GitHub after review |

### Key summaries and reports

| Path | Size | Purpose | Recommendation |
|---|---:|---|---|
| `/home/student/CROPNET/outputs/experiments/experiment_summary.csv` | 31 KB | Top-level experiment summary | Local backup and optional GitHub if small-summary policy allows |
| `/home/student/CROPNET/outputs/experiments/blank_fill_experiment_summary.csv` | 43 KB | Top-level blank-fill summary | Local backup and optional GitHub if small-summary policy allows |
| `/home/student/CROPNET/outputs/experiments/model_comparison_v1/model_comparison_summary.csv` | 27 KB | Controlled model comparison summary | Local backup and optional GitHub |
| `/home/student/CROPNET/outputs/experiments/model_comparison_v1/blank_fill_comparison_summary.csv` | 73 KB | Controlled blank-fill comparison summary | Local backup and optional GitHub |
| `/home/student/CROPNET/outputs/experiments/model_comparison_v1/model_specs_summary.csv` | 12 KB | Model specs summary | Local backup and optional GitHub |
| `/home/student/CROPNET/outputs/experiments/model_comparison_v1/README_model_comparison.md` | 7.2 KB | Comparison README | GitHub after review |
| `/home/student/CROPNET/outputs/experiments/ia30_seq6_seasonal_residual_blankfill/artifacts/model_specs.csv` | 1.2 KB | Key run model specs | Local backup |
| `/home/student/CROPNET/outputs/experiments/ia30_seq6_seasonal_residual_blankfill/artifacts/prediction_visualizations.zip` | 1.8 MB | Main visualization bundle | Local backup or cloud archive |
| `/home/student/CROPNET/outputs/experiments/ia30_seq6_seasonal_residual_blankfill/artifacts/strict_blank_fill_sarima_ensemble_full30_diagnostics.zip` | 6.1 MB | Full30 SARIMA+ensemble diagnostics bundle | Local backup or cloud archive |
| `/home/student/CROPNET/outputs/experiments/ia30_seq6_seasonal_residual_blankfill/artifacts/broader_ensemble_summary.csv` | 6.4 KB | Broader ensemble summary | Local backup and optional GitHub |
| `/home/student/CROPNET/cleanup_report.md` | 5.3 KB | Prior server cleanup audit | Local backup |

### Temporary but preserved feature ablation artifacts

| Path | Size | Purpose | Recommendation |
|---|---:|---|---|
| `/dev/shm/feature_ablation_v1/feature_ablation_v1_report.zip` | 676 KB | Feature ablation report bundle | Already preserved in `/home/student/CROPNET/cleanup_archives/`; fetch soon |
| `/home/student/CROPNET/cleanup_archives/feature_ablation_v1_report.zip` | 676 KB | Preserved copy of feature ablation report bundle | Local backup or cloud archive |
| `/home/student/CROPNET/cleanup_archives/feature_ablation_summary.csv` | 62 KB | Preserved ablation summary | Local backup and optional GitHub |
| `/home/student/CROPNET/cleanup_archives/feature_ablation_known1_summary.csv` | 13 KB | Preserved known-months=1 ablation summary | Local backup and optional GitHub |
| `/home/student/CROPNET/cleanup_archives/README_feature_ablation_v1.md` | 776 B | Preserved ablation README | GitHub after review |

## 4. Large artifacts

| Path | Size | Can regenerate? | Back up? | Delete after reset? | Risk level |
|---|---:|---|---|---|---|
| `/home/student/CROPNET/outputs/cropnet_v12_full` | 44 GB | Mostly yes | Preserve summaries/checkpoints first | Yes, after backup plan | medium |
| `/home/student/CROPNET/outputs/cropnet_v12_full/raw_chunks` | 44 GB | Yes | No if reproducible from dataset | Yes after reset | low risk to regenerate |
| `/home/student/CROPNET/outputs/experiments/ia_more_counties_seq12_bestsmall` | 44 GB | Mostly yes | Preserve small artifacts/checkpoints only | Yes after reset | medium |
| `/home/student/CROPNET/outputs/experiments/ia30_seq6_lstm_seasonal_residual_weather_only` | 44 GB | Mostly yes | Preserve small artifacts/checkpoints only | Yes after reset | medium |
| `/home/student/CROPNET/outputs/experiments/ia_more_counties_seq6_bestsmall_blankfill` | 17 GB | Mostly yes | Preserve small artifacts/checkpoints only | Yes after reset | medium |
| `/home/student/CROPNET/outputs/experiments/ia30_seq6_seasonal_residual_blankfill` | 337 MB | Partly | Yes, this is a key successful run | Keep or archive | must preserve |
| `/home/student/CROPNET/outputs/experiments/model_comparison_v1` | 592 MB | Partly | Yes, contains comparison outputs and checkpoints | Keep or archive | must preserve |
| `/home/student/CROPNET/debug_raw_ndvi_h5_issue_package.zip` | 2.2 GB | No simple rebuild guarantee | Yes if debugging package still needed | User review after backup | medium |
| `/home/student/.cache/huggingface` | 392 GB | Yes | No | Yes after reset | low risk to regenerate |
| `/home/student/.cache/uv` | 27 GB | Yes | No | Yes after reset | low risk to regenerate |
| `/home/student/.cache/pip` | 2.7 GB | Yes | No | Yes after reset | low risk to regenerate |
| `/home/student/YouZheng/jobs` | 90 GB | Unclear and unrelated | User review | Not a CropNet delete recommendation | needs user review |
| `/home/student/YouZheng/Taotern_VM/.venv` | 6.7 GB | Yes | No | Yes after user review | low risk to regenerate |

Notes:
- `/home/student/CROPNET` total size is about 153 GB
- `/home/student/YouZheng` total size is about 102 GB and appears largely unrelated to CropNet
- `/home/student/.cache` total size is about 421 GB and is the main space pressure outside project outputs

## 5. Sensitive files check

- No obvious `.env`, private SSH key, `.ppk`, or credential file was found under `/home/student/YouZheng/` or `/home/student/CROPNET/`
- Filename-only scan found one package CA bundle path:
  - `/home/student/YouZheng/Taotern_VM/.venv/lib/python3.12/site-packages/pip/_vendor/certifi/cacert.pem`
- The earlier broader `*token*` filename scan produced many false positives such as tokenizer files and benchmark CSV/JSON names. Those should not be treated as secrets by filename alone.
- Recommendation: do not push any `.ssh/`, cache, environment, or credential-like paths from `/home/student/` if the workspace reset later includes the full home directory.

## 6. Recovery checklist

### A. Must push to GitHub

These are the main code files that should be reviewed and committed into the intended clean repository, likely `perrywsle/Crop-Net`, before reset:

- `/home/student/CROPNET/cropnet_feature_forecasting_v12_server.py`
- `/home/student/CROPNET/summarize_experiments.py`
- `/home/student/CROPNET/make_report_artifacts.py`
- `/home/student/CROPNET/generate_loss_plots.py`
- `/home/student/CROPNET/analyze_blank_fill_diagnostics.py`
- `/home/student/CROPNET/run_controlled_model_comparison.py`
- `/home/student/CROPNET/build_broader_blank_fill_ensemble.py`
- `/home/student/CROPNET/scripts/run_feature_ablation_matrix.sh`
- `/home/student/CROPNET/scripts/summarize_feature_ablation.py`
- whichever single canonical copy should be kept for duplicated helpers in both `/home/student/CROPNET/` and `/home/student/CROPNET/scripts/`

Why:
- These are the active research workflow scripts and are currently not protected by a proper synced CropNet repository.

### B. Must back up locally or to cloud/archive

- `/home/student/CROPNET/outputs/experiments/ia30_seq6_seasonal_residual_blankfill/artifacts/`
- `/home/student/CROPNET/outputs/experiments/model_comparison_v1/`
- `/home/student/CROPNET/outputs/experiments/experiment_summary.csv`
- `/home/student/CROPNET/outputs/experiments/blank_fill_experiment_summary.csv`
- `/home/student/CROPNET/cleanup_archives/feature_ablation_v1_report.zip`
- `/home/student/CROPNET/cleanup_archives/feature_ablation_summary.csv`
- `/home/student/CROPNET/cleanup_archives/feature_ablation_known1_summary.csv`
- `/home/student/CROPNET/cleanup_archives/README_feature_ablation_v1.md`
- `/home/student/CROPNET/debug_raw_ndvi_h5_issue_package.zip` if the NDVI debugging evidence is still needed

Why:
- These contain successful experiment outputs, final summaries, visualizations, preserved temporary ablations, and debugging evidence that would be painful to reconstruct exactly.

Approximate sizes:
- key residual run: `337 MB`
- model comparison bundle: `592 MB`
- feature ablation preserved zip: `676 KB`
- NDVI debug package: `2.2 GB`

### C. Can ignore or delete after reset

After verified backup and/or GitHub sync:
- `/home/student/CROPNET/outputs/**/raw_chunks/`
- `/home/student/CROPNET/outputs/**/feature_cache/`
- `/home/student/.cache/huggingface/`
- `/home/student/.cache/uv/`
- `/home/student/.cache/pip/`
- Python caches such as `__pycache__/`, `.pytest_cache/`, and `.ruff_cache/`
- virtual environments such as `/home/student/YouZheng/Taotern_VM/.venv`
- temporary `/dev/shm/feature_ablation_v1/` after verifying the preserved copy under `/home/student/CROPNET/cleanup_archives/`

Why safe:
- These are caches, temporary artifacts, or reproducible intermediate outputs rather than the irreplaceable final research state.

### D. Needs user review

- `/home/student/YouZheng/` large unrelated repos and job outputs, especially:
  - `/home/student/YouZheng/jobs`
  - `/home/student/YouZheng/gamma_ssm_repo`
  - `/home/student/YouZheng/repo`
  - `/home/student/YouZheng/gamma_SSM_S4_enhanced`
  - `/home/student/YouZheng/gamma_SSM_S4_enhanced_clean`
  - `/home/student/YouZheng/Taotern_VM*`
- `/home/student/CROPNET/debug_raw_ndvi_h5_issue_package.zip`
- duplicate helper copies under both `/home/student/CROPNET/` and `/home/student/CROPNET/scripts/`

Question for user:
- Should the reset preserve only the CropNet/COS40007 project, or also the unrelated Taotern/gamma work under `/home/student/YouZheng/`?

### E. Rebuild commands for a clean workspace

Do not run these yet. These are later-use commands for rebuilding after reset.

```bash
git clone git@github.com:perrywsle/Crop-Net.git /home/student/CROPNET-clean/Crop-Net
cd /home/student/CROPNET-clean/Crop-Net
python3 -m venv /home/student/.venvs/cropnet
source /home/student/.venvs/cropnet/bin/activate
pip install -r requirements.txt || pip install -e .
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available())"
```

Restore backed-up artifacts after clone if needed:

```bash
mkdir -p /home/student/CROPNET-clean/outputs/experiments
# copy back selected summaries, zips, and key run artifacts from local or cloud backup
```

Smoke checks:

```bash
python /home/student/CROPNET-clean/Crop-Net/server_env_probe.py
python /home/student/CROPNET-clean/Crop-Net/cropnet_feature_forecasting_v12_server.py --help
```

Blank-fill evaluation smoke test after artifact restore, if the restored run artifacts remain compatible:

```bash
python /home/student/CROPNET-clean/Crop-Net/cropnet_feature_forecasting_v12_server.py \
  --eval-only \
  --from-output-dir outputs/experiments/ia30_seq6_seasonal_residual_blankfill \
  --run-blank-fill-eval \
  --blank-fill-residual-seasonal \
  --strict-blank-fill-no-future-fill \
  --blank-fill-year 2021 \
  --blank-fill-known-months 1 \
  --blank-fill-output-prefix smoke_recheck
```

## 7. Final project status

**NOT READY - HAS UNSYNCED CODE**

Why:
- The active CropNet code is not currently backed by a proper synced project repository.
- `/home/student/CROPNET/Crop-Net` resolves to the accidental home-directory repo at `/home/student/.git`.
- That accidental repo has no commits and no remote.
- Important successful experiment artifacts also exist locally and should be backed up before any destructive reset.
