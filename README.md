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
