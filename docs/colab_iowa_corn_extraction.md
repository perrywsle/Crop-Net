# Colab Iowa Corn Monthly Extraction

This guide runs the full Iowa Corn monthly CropNet extraction on Google Colab using Google Drive storage. It is independent from any local Windows extraction process.

## Drive Layout To Upload

Create this folder in Google Drive:

```text
MyDrive/cropnet_iowa_corn/
  Crop-Net-repo/
    src/
    scripts/
    configs/
    data/usda_labels/USDA Crop Dataset/Corn/2017/USDA_Corn_County_2017.csv
    data/usda_labels/USDA Crop Dataset/Corn/2018/USDA_Corn_County_2018.csv
    data/usda_labels/USDA Crop Dataset/Corn/2019/USDA_Corn_County_2019.csv
    data/usda_labels/USDA Crop Dataset/Corn/2020/USDA_Corn_County_2020.csv
    data/usda_labels/USDA Crop Dataset/Corn/2021/USDA_Corn_County_2021.csv
    data/usda_labels/USDA Crop Dataset/Corn/2022/USDA_Corn_County_2022.csv
    outputs/experiments/corn_ia_monthly_2017_2022/batch_manifest.csv
    outputs/experiments/corn_ia_monthly_2017_2022/batch_commands.txt
    requirements.txt
    pyproject.toml
    README.md
    Crop-Net/
      src/
      requirements.txt
      pyproject.toml
```

Required uploads:

- `src/`
- `scripts/`
- `configs/`
- `requirements.txt`
- `pyproject.toml`
- `README.md`
- `Crop-Net/src/`
- `Crop-Net/requirements.txt`
- `Crop-Net/pyproject.toml`
- USDA Corn county yield CSVs for 2017, 2018, 2019, 2020, 2021, and 2022
- `outputs/experiments/corn_ia_monthly_2017_2022/batch_manifest.csv`
- `outputs/experiments/corn_ia_monthly_2017_2022/batch_commands.txt`

Optional uploads that can save time:

- Completed batch folders such as `outputs/experiments/corn_ia_monthly_2017_2022_batch_001/`
- Existing `feature_cache/` folders inside completed or partial batch folders
- Existing Hugging Face cache content under `hf_cache/`
- Any already valid `official_monthly_feature_table.parquet` batch artifacts

Not required:

- Local Windows PIDs
- PowerShell scripts
- Local raw HDF5 chunks

Credentials:

- No AG, NDVI, weather, or USDA API keys are expected.
- The extractor downloads public AG, NDVI, and weather files from Hugging Face dataset `CropNet/CropNet`.
- If Hugging Face requires authentication or rate-limit relief, set `HF_TOKEN` in Colab before running extraction.

## Colab Notebook

Open:

```text
notebooks/colab_iowa_corn_monthly_10batch_extraction.ipynb
```

Run the notebook top to bottom:

1. Mount Google Drive.
2. Set `DRIVE_ROOT` to `/content/drive/MyDrive/cropnet_iowa_corn`.
3. Install dependencies.
4. Verify imports and required files.
5. Show current batch status.
6. Run one missing batch, or run all missing batches sequentially.
7. Resume by rerunning the status cell and then the run-missing cell.
8. Merge only after all 10 batch artifacts validate.
9. Retrain the monthly yield model.
10. Print the final readiness report.

## Resume Rules

The Colab runner is designed for disconnections:

- If a batch artifact exists and validates, it is skipped.
- If a batch folder exists but the final parquet is missing, that batch is rerun.
- If a batch artifact is invalid, that batch is rerun.
- Status is written after every batch to:
  `outputs/experiments/corn_ia_monthly_2017_2022/colab_batch_status.json`
- Runner output is appended to:
  `outputs/experiments/corn_ia_monthly_2017_2022/colab_batch_runner.log`
- Per-batch pipeline logs are written to:
  `outputs/experiments/corn_ia_monthly_2017_2022_batch_###/logs/corn_ia_monthly_2017_2022_batch_###.log`

Run one batch at a time by default. The notebook exposes `parallelism=2` as an experimental option for high-RAM Colab runtimes, but sequential execution is safer for disk, cache, and network stability.

## Expected Outputs

After all extraction batches:

```text
outputs/experiments/corn_ia_monthly_2017_2022_batch_001/artifacts/official_monthly_feature_table.parquet
...
outputs/experiments/corn_ia_monthly_2017_2022_batch_010/artifacts/official_monthly_feature_table.parquet
```

After merge:

```text
outputs/experiments/corn_ia_monthly_2017_2022/artifacts/official_monthly_feature_table.parquet
outputs/experiments/corn_ia_monthly_2017_2022/artifacts/monthly_merge_diagnostic.json
```

After retraining:

```text
outputs/yield_baseline/corn_ia_2017_2022_monthly_full/best_yield_model.joblib
outputs/yield_baseline/corn_ia_2017_2022_monthly_full/yield_model_benchmark.csv
outputs/yield_baseline/corn_ia_2017_2022_monthly_full/month_benchmark.csv
outputs/yield_baseline/corn_ia_2017_2022_monthly_full/window_benchmark.csv
outputs/yield_baseline/corn_ia_2017_2022_monthly_full/prediction_residuals.csv
outputs/yield_baseline/corn_ia_2017_2022_monthly_full/yield_model_metadata.json
```

The merged table should usually contain about `7128` rows for 99 Iowa counties, 6 years, and 12 months. If every batch had 10 counties, the upper reference count would be `7200` rows.

## Baseline Comparison

The notebook reports model metrics and compares them against the previous smoke baseline:

```text
BaselinePreviousYearSameCounty
RMSE: 24.02
MAE: 23.23
R2: 0.231
MAPE: 13.56%
```

The final report marks the workflow ready for yield prediction only when all 10 batches are valid, the merged table is valid, retraining metrics exist, and `best_yield_model.joblib` exists.
