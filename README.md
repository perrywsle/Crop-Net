# CropNet Blank-Fill Forecasting Handover

This branch packages the active COS40007 CropNet forecasting workflow in a more teammate-friendly form without removing the original research scripts.

## Project Objective
Forecast missing future months in a target year from partial-year observations using monthly AG, NDVI, and weather features.

## Current Best Findings
- Best raw RMSE for the main `known_months=1` industrial case: `LSTM seasonal_residual`
- Best normalized RMSE: `ensemble_mean`
- Best classical model: `SARIMA`
- Best AG raw behavior: `seasonal_last_year`
- Best NDVI raw behavior: `SARIMA`
- Best weather raw behavior: `LSTM seasonal_residual`

## New to the Project?
If you are new to this project, read `docs/PROJECT_QUICK_BRIEF.md` first, then `docs/PROJECT_UNDERSTANDING_GUIDE.md`.

## Developer Handover Quick Start
### Branch purpose
This branch keeps the validated research workflow intact while adding a cleaner Python package, sample configs, example scripts, small checkpoints, and a small set of report figures.

### Folder structure
- `src/cropnet_forecasting/`: cleaner package modules
- `examples/`: training and inference examples
- `configs/`: sample YAML configs
- `weights/`: small checkpoints and scaler/config artifacts
- `reports/`: small markdown summaries, CSV tables, and a few figures
- `scripts/research/`: legacy research workflow scripts preserved as reference

### Environment setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Inference example
```bash
python examples/inference_example.py --checkpoint weights/lstm_best.pt --scaler weights/scaler.csv --config configs/residual_lstm_all.yaml
```

### Blank-fill example
```bash
python examples/blank_fill_example.py --monthly-table path/to/monthly_features.parquet --checkpoint weights/lstm_best.pt --scaler weights/scaler.csv --config configs/residual_lstm_all.yaml --year 2021 --known-months 1 --output outputs/blank_fill_predictions.csv
```

### Desktop GUI
```bash
python -m crop_fusion_ai.gui.app
```

The GUI expects a folder that contains modality subfolders such as:
```text
sample_data/
  ag/
    2017_12_21.png
  ndvi/
    2017_12_21.png
  weather/
    2017_12.csv
```

The app scans the directory recursively, extracts monthly features, and shows a 12-month autoregressive forecast in tabs.

### Download sample data
```bash
python fetch_data.py --county-id 01003 --crop corn --years 2017 2018 2019 2020 2021 2022
```

### Direct Corn IA yield baseline
Fetch USDA labels without downloading imagery or weather chunks:
```bash
python fetch_data.py --labels-only --crop corn --years 2017 2018 2019 2020 2021 2022 --output-dir data/usda_labels
```

Create a label-driven county batch manifest and extraction commands for the
growing-season run:
```bash
python src/cropnet_forecasting/yield_batching.py manifest \
  --usda-path "data/usda_labels/USDA Crop Dataset/Corn/2017/USDA_Corn_County_2017.csv" \
    "data/usda_labels/USDA Crop Dataset/Corn/2018/USDA_Corn_County_2018.csv" \
    "data/usda_labels/USDA Crop Dataset/Corn/2019/USDA_Corn_County_2019.csv" \
    "data/usda_labels/USDA Crop Dataset/Corn/2020/USDA_Corn_County_2020.csv" \
    "data/usda_labels/USDA Crop Dataset/Corn/2021/USDA_Corn_County_2021.csv" \
    "data/usda_labels/USDA Crop Dataset/Corn/2022/USDA_Corn_County_2022.csv" \
  --batch-size 10 \
  --years 2017 2018 2019 2020 2021 2022 \
  --quarters Q2 Q3 \
  --output-path outputs/experiments/corn_ia_gs_2017_2022/batch_manifest.csv \
  --write-commands outputs/experiments/corn_ia_gs_2017_2022/batch_commands.txt
```

Run each generated command to extract real monthly AG, NDVI, and weather
features without training forecasting models. Each command follows this shape:
```bash
python scripts/research/cropnet_feature_forecasting_v12_server.py \
  --full-run \
  --extract-only \
  --state-codes IA \
  --crop Corn \
  --years 2017 2018 2019 2020 2021 2022 \
  --quarters Q2 Q3 \
  --fips-codes 19001 19003 19005 19007 19009 \
  --run-name corn_ia_gs_2017_2022_batch_001 \
  --resume \
  --delete-raw-after-extract
```

Merge completed batch monthly tables into one canonical full-IA table:
```bash
python src/cropnet_forecasting/yield_batching.py merge \
  --monthly-path outputs/experiments/corn_ia_gs_2017_2022_batch_*/artifacts/official_monthly_feature_table.parquet \
  --output-path outputs/experiments/corn_ia_gs_2017_2022/artifacts/official_monthly_feature_table.parquet \
  --diagnostic-path outputs/experiments/corn_ia_gs_2017_2022/artifacts/monthly_merge_diagnostic.json
```

Train/test the direct yield baseline from ground-truth monthly features and USDA labels:
```bash
python src/cropnet_forecasting/yield_regression.py \
  --monthly-path outputs/experiments/corn_ia_gs_2017_2022/artifacts/official_monthly_feature_table.parquet \
  --usda-path "data/usda_labels/USDA Crop Dataset/Corn/2017/USDA_Corn_County_2017.csv" \
    "data/usda_labels/USDA Crop Dataset/Corn/2018/USDA_Corn_County_2018.csv" \
    "data/usda_labels/USDA Crop Dataset/Corn/2019/USDA_Corn_County_2019.csv" \
    "data/usda_labels/USDA Crop Dataset/Corn/2020/USDA_Corn_County_2020.csv" \
    "data/usda_labels/USDA Crop Dataset/Corn/2021/USDA_Corn_County_2021.csv" \
    "data/usda_labels/USDA Crop Dataset/Corn/2022/USDA_Corn_County_2022.csv" \
  --crop-type corn \
  --feature-group all \
  --target-grain monthly \
  --output-dir outputs/yield_baseline/corn_ia_2017_2022_monthly
```

The monthly direct yield baseline copies each county-year USDA annual yield onto the matching monthly rows, so a model can learn to predict annual yield from any available month. It rejects blank-fill or forecast prediction tables and saves a merged monthly training frame, model benchmark, per-month benchmark, month-window benchmark, pruning report, residuals, best model artifact, and metadata JSON.

For true January-only prediction, extract all quarters (`Q1 Q2 Q3 Q4`) before training; the Q2/Q3 workflow only supports April-September monthly benchmarks.

### Convert GUI sample data
```bash
python convert_data.py --source data/sample_data --output test_data
```

### List available counties
```bash
python list_county.py --years 2017 2018 2019 2020 2021 2022
```

### Training wrapper example
```bash
python examples/train_example.py --config configs/residual_lstm_all.yaml --mode fit
```

### Where weights and figures live
- `weights/`
- `reports/figures/`
- `reports/README_RESULTS_SUMMARY.md`

### Legacy research code
The original validated workflow remains in `scripts/research/cropnet_feature_forecasting_v12_server.py`.

### What not to commit
Do not commit raw datasets, `raw_chunks/`, `feature_cache/`, HDF5 files, virtual environments, large output directories, or secrets.
