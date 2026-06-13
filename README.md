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

### Local web app
```bash
python main.py
```

The browser app starts a local server on `http://127.0.0.1:8000` by default and opens a farmer-friendly dashboard. It accepts a folder that contains modality subfolders such as:
```text
sample_data/
  ag/
    2017_12_21.png
  ndvi/
    2017_12_21.png
  weather/
    2017_12.csv
```

The app scans the directory recursively, extracts monthly features, and shows yield estimates, grouped feature cards, and clean charts in tabs.

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

### New training workflow
Prepare a canonical split once:
```bash
python prepare_dataset.py --output-dir data/training --raw-root data/raw/cropnet --crop-type corn --state-codes IA --years 2017 2018 2019 2020 2021 2022 --train-years 2017 2018 2019 2020 --val-years 2021 --test-years 2022
```

Then train a model from the prepared split:
```bash
python training/train.py --dataset-dir data/training --run-name lstm_clean_run --models lstm
```

### Where weights and figures live
- `weights/`
- `reports/figures/`
- `reports/README_RESULTS_SUMMARY.md`

### Legacy research code
The original validated workflow remains in `scripts/research/cropnet_feature_forecasting_v12_server.py`.

### What not to commit
Do not commit raw datasets, `raw_chunks/`, `feature_cache/`, HDF5 files, virtual environments, large output directories, or secrets.

## Iowa Corn Monthly Yield Prediction Results

This project trains an Iowa Corn yield prediction model using monthly CropNet AG, NDVI, and weather features joined with USDA county-level annual corn yield labels. The current full experiment uses Iowa Corn data from 2017-2022, trains on 2017-2021, and tests on 2022.

The yield task is built at monthly grain: each `county_id`, `crop_type`, `year`, and `month` row receives the USDA annual yield label for that county-year. This lets the model estimate final annual yield from one month or a month window while still using ground-truth monthly CropNet features only.

### Prediction Logic

The model does not first aggregate all 12 months into one annual feature row. Instead, it keeps the data at monthly grain and copies the USDA annual yield label onto every monthly row for the same county-year.

Example:

```text
County 19001, Corn, 2022 annual USDA yield = 164.2 bu/acre

January 2022 features  -> label 164.2
February 2022 features -> label 164.2
March 2022 features    -> label 164.2
...
December 2022 features -> label 164.2
```

The model therefore learns this mapping:

```text
monthly AG + NDVI + weather features + month timing -> final annual corn yield
```

This design allows early-season or partial-window prediction. For example, the same trained model can be evaluated on January-only rows, Jan-Mar rows, Apr-Jun rows, Apr-Sep rows, or the full year. The split remains year-based, with 2017-2021 used for training and 2022 held out for testing, so duplicated monthly labels from the same county-year do not leak across train and test.

The pipeline also compares ML models against simple baselines. The most important baseline is `BaselinePreviousYearSameCounty`, which predicts a county's 2022 yield from that same county's 2021 yield. This baseline is strong because county yield history already captures soil, management, and local productivity patterns.

Main Colab/result artifacts:

- `outputs/experiments/corn_ia_monthly_2017_2022/artifacts/official_monthly_feature_table.parquet`
- `outputs/yield_baseline/corn_ia_2017_2022_monthly_full/`
- `outputs/yield_baseline/corn_ia_2017_2022_monthly_improved/`
- `metrics.csv`
- `predictions_2022.csv`
- `plots/*.png`

For the 2022 holdout set:

- Average actual yield: `195.52`
- Average predicted yield: `191.74`
- Average absolute error: `16.33`

The model under-predicts by about `3.78` yield units on average, and the typical prediction error is around `16.33` yield units. The best ML model is useful, but the `BaselinePreviousYearSameCounty` baseline remains slightly stronger on this 2022 test set.
