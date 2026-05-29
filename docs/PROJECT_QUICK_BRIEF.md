# Project Quick Brief

## What this project is

This project forecasts **monthly AG, NDVI, and weather features** for partially observed years. It does **not** directly predict crop yield yet. Instead, it fills the missing months so the finished yearly feature table can later be used by a separate downstream yield model.

## Core task

The main task is **blank-fill forecasting**.

Example:
- known: 2017-2026 full history plus January 2027
- predict: February-December 2027
- result: full 2027 monthly feature table for downstream yield prediction

## Main pipeline

1. preprocess CropNet data into monthly AG / NDVI / weather features
2. build a 35-feature monthly table
3. scale using train-year mean/std
4. build rolling sequences with `seq_len=6`
5. train/evaluate forecasting models
6. run strict recursive blank-fill evaluation for `known_months = 0, 1, 3, 6, 9`

## Models used

- `naive_lag1`: previous month baseline
- `seasonal_last_year`: same county, same month, previous year baseline
- `LSTM`, `GRU`, `tiny_mamba_ssm`, `transformer_encoder`: learned sequence models
- `SARIMA`: classical seasonal time-series model
- `ensemble_mean`, `ensemble_weighted`: deployable ensembles

## Best current results

- best raw RMSE for the main `known_months=1` case: **LSTM seasonal_residual**
- best normalized RMSE for the main `known_months=1` case: **ensemble_mean**
- best classical model: **SARIMA**
- best AG raw behavior: **seasonal_last_year**
- best NDVI raw behavior: **SARIMA**
- best weather raw behavior: **LSTM seasonal_residual**

## Important interpretation

- raw RMSE is strongly influenced by large-scale weather features
- normalized RMSE is better for cross-feature fairness
- no single model wins every metric
- the broader ensemble did not beat the narrower ensemble
- feature ablations showed the task is highly modality-dependent

## Most important findings from ablation

- AG/NDVI context helps weather forecasting
- all features are best for the full raw industrial metric
- AG+NDVI can look strongest under normalized comparison
- SARIMA is especially strong for NDVI-oriented settings

## Where to look first

- `README.md`
- `docs/PROJECT_UNDERSTANDING_GUIDE.md`
- `reports/README_RESULTS_SUMMARY.md`
- `reports/figures/`
- `examples/`
- `src/cropnet_forecasting/`
- `scripts/research/cropnet_feature_forecasting_v12_server.py` for the validated legacy path

## What teammates should inspect

- `examples/inference_example.py`
- `examples/blank_fill_example.py`
- `examples/train_example.py`
- `configs/residual_lstm_all.yaml`
- `weights/`
- `reports/tables/*.csv`

## Limitations

- not a final production deployment
- not direct yield prediction yet
- most comparison runs are IA / 30-county focused
- storage and artifact management were active constraints
- legacy research script still exists and remains important

## Next recommended work

- continue ablations
- consider data augmentation
- scale to all IA counties after stable comparison conclusions
- keep improving the deployment-friendly package layer
- prepare the final report and presentation
