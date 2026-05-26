# CropNet Artifact Backup Manifest

## Must-backup artifacts

| Path | Approx size | Reason | Recommended destination | Regenerable |
|---|---:|---|---|---|
| `/home/student/CROPNET/outputs/experiments/ia30_seq6_seasonal_residual_blankfill/artifacts/` | key run artifacts, hundreds of MB | main residual blank-fill run, strict evaluation, SARIMA/ensemble diagnostics, plots, specs | local backup or cloud archive | partly |
| `/home/student/CROPNET/outputs/experiments/model_comparison_v1/` | about 592 MB | controlled comparison outputs, model specs, histories, summaries | local backup or cloud archive | partly |
| `/home/student/CROPNET/cleanup_archives/feature_ablation_v1_report.zip` | about 676 KB | preserved feature ablation report bundle from `/dev/shm` | local backup or cloud archive | not worth regenerating |
| `/home/student/CROPNET/cleanup_archives/feature_ablation_summary.csv` | about 62 KB | compact top-level feature ablation summary | GitHub-safe backup plus local copy | yes, but preserve |
| `/home/student/CROPNET/cleanup_archives/feature_ablation_known1_summary.csv` | about 13 KB | key known_months=1 ablation summary | GitHub-safe backup plus local copy | yes, but preserve |
| `/home/student/CROPNET/cleanup_archives/README_feature_ablation_v1.md` | under 1 KB | ablation summary README | GitHub-safe backup plus local copy | yes |

## Additional recommended backups

| Path | Approx size | Reason | Recommended destination | Regenerable |
|---|---:|---|---|---|
| `/home/student/CROPNET/outputs/experiments/experiment_summary.csv` | about 31 KB | top-level experiment index | GitHub-safe or local backup | yes |
| `/home/student/CROPNET/outputs/experiments/blank_fill_experiment_summary.csv` | about 43 KB | top-level blank-fill index | GitHub-safe or local backup | yes |
| `/home/student/CROPNET/debug_raw_ndvi_h5_issue_package.zip` | about 2.2 GB | NDVI debugging evidence package | local or cloud archive if still needed | uncertain |

## Exclude from GitHub and usually regenerate

- `raw_chunks/`
- `feature_cache/`
- Hugging Face cache
- pip/uv caches
- `.venv/` and other virtual environments
- `.pt`, `.pth`, `.ckpt`
- `.h5` and `.hdf5`
- huge prediction-long CSVs and transient zip bundles
