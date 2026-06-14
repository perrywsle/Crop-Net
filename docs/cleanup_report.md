# Cleanup Report

## 1. Disk usage before cleanup

`	ext
Filesystem      Size  Used Avail Use% Mounted on
tmpfs           6.3G  2.6M  6.3G   1% /run
/dev/nvme0n1p5  922G  875G     0 100% /
tmpfs            32G  254M   31G   1% /dev/shm
`

## 2. Largest directories/files found

### Largest project directories
- /home/student/CROPNET/outputs/cropnet_v12_full/raw_chunks — about 44G
- /home/student/CROPNET/outputs/experiments/ia30_seq6_lstm_seasonal_residual_weather_only/raw_chunks — about 44G
- /home/student/CROPNET/outputs/experiments/ia_more_counties_seq12_bestsmall/raw_chunks — about 44G
- /home/student/CROPNET/outputs/experiments/ia_more_counties_seq6_bestsmall_blankfill/raw_chunks — about 17G
- /home/student/CROPNET/outputs/experiments/ia30_seq6_seasonal_residual_blankfill — about 8.3G before cleanup, about 337M after removing the bad archive
- /home/student/CROPNET/outputs/experiments/model_comparison_v1 — about 592M

### Largest project files
- /home/student/CROPNET/outputs/experiments/ia30_seq6_seasonal_residual_blankfill/artifacts/broader_ensemble_full30_report.zip — 8.49G
  - known failed partial self-including archive
- /home/student/CROPNET/debug_raw_ndvi_h5_issue_package.zip — about 2.30G
- Many raw Sentinel HDF5 chunks under aw_chunks/ between about 0.94G and 1.19G each

## 3. Safe-delete candidates identified

### A. Safe to delete immediately
- /home/student/CROPNET/outputs/experiments/ia30_seq6_seasonal_residual_blankfill/artifacts/broader_ensemble_full30_report.zip
  - known bad partial archive
- Python cache directories inside /home/student/CROPNET
  - __pycache__
  - .pytest_cache

### B. Archive/fetch before delete
- /dev/shm/feature_ablation_v1/feature_ablation_v1_report.zip
- /dev/shm/feature_ablation_v1/feature_ablation_summary.csv
- /dev/shm/feature_ablation_v1/feature_ablation_known1_summary.csv
- /dev/shm/feature_ablation_v1/README_feature_ablation_v1.md

### C. Do not delete
- source code in /home/student/CROPNET
- successful key run artifacts under:
  - /home/student/CROPNET/outputs/cropnet_v12_full/artifacts
  - /home/student/CROPNET/outputs/experiments/ia30_seq6_seasonal_residual_blankfill/artifacts
  - /home/student/CROPNET/outputs/experiments/model_comparison_v1
- checkpoints, metrics, histories, scaler.csv, eature_contract_diagnostic.json, config.json, un_status.json

## 4. Files deleted

Deleted exactly:
- /home/student/CROPNET/outputs/experiments/ia30_seq6_seasonal_residual_blankfill/artifacts/broader_ensemble_full30_report.zip
- cache directories removed from /home/student/CROPNET:
  - /home/student/CROPNET/__pycache__
  - /home/student/CROPNET/Crop-Net/scripts/__pycache__
  - /home/student/CROPNET/Crop-Net/tests/__pycache__
  - /home/student/CROPNET/Crop-Net/.pytest_cache

## 5. Files preserved

Archived to /home/student/CROPNET/cleanup_archives:
- eature_ablation_v1_report.zip
- eature_ablation_summary.csv
- eature_ablation_known1_summary.csv
- README_feature_ablation_v1.md

Original temporary copies remain in /dev/shm/feature_ablation_v1 for now.

## 6. Space freed

- Freed about 8.0G immediately from deleting the failed archive
- Additional cache cleanup was small but safe

## 7. Disk usage after cleanup

`	ext
Filesystem      Size  Used Avail Use% Mounted on
tmpfs           6.3G  2.6M  6.3G   1% /run
/dev/nvme0n1p5  922G  867G  7.9G 100% /
tmpfs            32G  254M   31G   1% /dev/shm
`

## 8. Files recommended for user fetch

Primary fetch candidate:
- /home/student/CROPNET/cleanup_archives/feature_ablation_v1_report.zip

Suggested command:
`ash
scp -i ~/.ssh/cropnet_codex_temp_key student@10.147.20.110:/home/student/CROPNET/cleanup_archives/feature_ablation_v1_report.zip .
`

## 9. Remaining risks

- Root filesystem still has only about 7.9G free, so it is writable again but still tight for new large experiment outputs.
- The main remaining project space consumers are duplicated aw_chunks trees from reproducible runs.
- Non-project storage under /home/student/.cache is extremely large, especially:
  - /home/student/.cache/huggingface — about 392G
  - /home/student/.cache/uv — about 20G
  - /home/student/.cache/pip — about 2.7G
  These were not touched because they are broader environment caches and not strictly project-local.

## 10. Recommended next cleanup action if more space is needed

Lowest-risk project cleanup candidates to review next, but **not deleted in this pass**:
- /home/student/CROPNET/outputs/experiments/ia30_seq6_lstm_seasonal_residual_weather_only/raw_chunks — about 44G
  - reproducible; final artifacts remain
  - risk: low to medium
- /home/student/CROPNET/outputs/experiments/ia_more_counties_seq12_bestsmall/raw_chunks — about 44G
  - reproducible; final artifacts remain
  - risk: low to medium
- /home/student/CROPNET/outputs/experiments/ia_more_counties_seq6_bestsmall_blankfill/raw_chunks — about 17G
  - reproducible; final artifacts remain
  - risk: low to medium
- /home/student/CROPNET/outputs/cropnet_v12_full/raw_chunks — about 44G
  - reproducible, but belongs to the main validated baseline
  - risk: medium

If space pressure continues, the next best action is to review and remove one or more **duplicate experiment raw_chunks directories** while keeping final artifacts, metrics, and checkpoints.
