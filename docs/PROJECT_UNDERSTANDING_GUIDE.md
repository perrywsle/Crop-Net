# Project Understanding Guide

## 1. Executive Summary

This project studies **monthly feature forecasting** for CropNet-style agricultural data. The goal is not to predict yearly crop yield directly in this workflow. Instead, we forecast the missing monthly **AG, NDVI, and weather** feature values for a partially observed year, then use those completed year-level features later in a separate downstream yield model.

The main problem we solve is **future blank filling**. For example, if only the first few months of the current year are available, we want to estimate the remaining months in a way that is accurate enough to support downstream analysis or yield prediction.

What the current system can do:
- load or reuse processed monthly feature tables
- train and evaluate several forecasting models
- run **one-step forecasting** and **recursive blank-fill forecasting**
- compare raw and normalized metrics
- perform strict no-future-fill evaluation
- generate summaries, plots, model specs, and small handover artifacts

What it cannot fully do yet:
- serve as a polished production deployment
- guarantee robust superiority of one learned model across all metrics
- directly replace a final downstream yearly yield pipeline without more validation

Current best findings:
- best raw RMSE for the main `known_months=1` case: **LSTM seasonal_residual**
- best normalized RMSE for the same case: **ensemble_mean**
- best classical model: **SARIMA**
- best AG raw behavior: **seasonal_last_year**
- best NDVI raw behavior: **SARIMA**
- best weather raw behavior: **LSTM seasonal_residual**

## 2. Project Objective

A key point is that this workflow is **not** directly a yield predictor.

The current objective is:
1. build a reliable monthly feature table for each county and crop
2. forecast the future missing monthly feature values in a partially observed year
3. assemble a completed yearly feature table
4. later hand that completed year into a separate yearly yield model

So the target of the forecasting model is a **35-feature monthly vector**, not a single final yield number.

This matters because forecasting monthly features is a different problem from predicting yield. The feature-forecasting stage needs to preserve seasonal shape, modality behavior, and scale, not just optimize a single yearly label.

## 3. Industrial Use Case

The motivating use case is practical blank filling for a current or future year.

Example:
- historical data available: 2017 to January 2027
- missing months: February to December 2027
- task: forecast the missing monthly AG, NDVI, and weather feature values
- result: a full 2027 feature table
- next step: feed the completed 2027 feature table into a separate downstream yearly yield prediction model

This is why `known_months=1` is especially important. It corresponds to the realistic scenario where only **January is observed** and the rest of the year must be estimated recursively.

## 4. Dataset and Feature Design

The project is based on the CropNet data context and works with three modalities:
- **AG**: image-derived agricultural surface / field appearance features
- **NDVI**: vegetation index summary statistics
- **Weather**: monthly aggregated meteorological features

The main monthly feature table uses **35 features** total:
- 8 AG features
- 12 NDVI features
- 15 weather features

Feature groups used in experiments:
- `ag`
- `ndvi`
- `weather`
- `ag_ndvi`
- `ag_weather`
- `ndvi_weather`
- `all`

These groups are important because the later ablation experiments showed that performance depends strongly on which modalities are included.

## 5. Data Preprocessing Pipeline

The preprocessing pipeline was a major part of the work.

Important preprocessing topics:
- **raw chunks / HDF5 issue**: the original server workflow had to deal with large raw chunk files and several extraction edge cases
- **NDVI direct-FIPS fallback**: some NDVI files required a more direct county lookup path to extract the right data reliably
- **weather aggregation fix**: weather extraction had to be standardized into consistent monthly features
- **feature contract 35/35**: the pipeline was validated to ensure the expected 35 features were present and aligned
- **monthly table construction**: AG, NDVI, and weather features were merged into a monthly table by county, crop, year, and month
- **sequence construction**: supervised sequences were built for forecasting from rolling monthly windows
- **scaler**: training used per-feature mean/std scaling based on train years
- **train/val/test split**: the main split was 2017?2019 train, 2020 validation, 2021 test

A practical note: the legacy preprocessing and forecasting path lives in the research script and is still the validated reference implementation.

## 6. What `seq_len` Means

`seq_len` is the number of previous monthly time steps used as model input.

Example:
- `seq_len=6` means the model sees the previous **6 months** and predicts the next month

For one-step forecasting, this means:
- input: months `t-5 ... t`
- output: month `t+1`

For recursive blank-fill forecasting, it matters even more:
- first predicted month uses the last observed `seq_len` months
- second predicted month may use one predicted month in its window
- later months use more and more previously predicted values

So recursive blank filling is harder than one-step forecasting because errors can accumulate over the horizon.

## 7. What Blank-Fill Forecasting Means

There are two related tasks in this project:

1. **One-step forecasting**
- predict the next month from the previous window
- useful for measuring general short-horizon prediction quality

2. **Recursive blank-fill forecasting**
- start from a partially known target year
- repeatedly forecast the next missing month
- append that prediction back into history
- continue until the remaining year is filled

Key settings:
- `known_months=0`: forecast January to December
- `known_months=1`: January known, forecast February to December
- `known_months=3`: January to March known, forecast April to December
- `known_months=6`: January to June known, forecast July to December
- `known_months=9`: January to September known, forecast October to December

The most important industrial test is `known_months=1`, because it is a hard and realistic early-year forecasting problem.

## 8. Evaluation Setup

Main evaluation setup:
- train years: **2017?2019**
- validation year: **2020**
- test / blank-fill year: **2021**
- state focus: **IA**
- main scale: **30 counties**
- main lookback: **seq_len=6**

Why strict no-future-fill evaluation matters:
- early in the project, a possible evaluation leakage risk was identified
- the preprocessing path could interpolate / forward-fill / back-fill a dense monthly frame before forecasting
- if future target-year months influence earlier missing values, blank-fill evaluation becomes optimistic

Strict mode was added to prevent that.

The strict evaluation confirmed:
- no meaningful advantage was coming from future target-year fill leakage in the main residual LSTM result
- the LSTM raw-RMSE advantage survived strict no-future-fill evaluation

This strengthens the credibility of the research comparison.

## 9. Models Compared

### `naive_lag1`
A deterministic baseline.
- predicts the next month using the most recent available month
- very simple, useful as a low bar

### `seasonal_last_year`
A strong deterministic baseline.
- predicts a month using the **same county + same month + previous year**
- works well when seasonality is stable
- this turned out to be a very strong benchmark, especially for AG and normalized comparisons

### `LSTM`
A recurrent neural network model.
- learned sequence model
- strongest single learned model for the main raw industrial metric

### `GRU`
Another recurrent sequence model.
- simpler than LSTM in gating structure
- sometimes competitive, but not the overall winner

### `tiny_mamba_ssm`
A compact state-space style neural model.
- added for broader architectural comparison
- competitive in some cases, but not decisively superior overall

### `transformer_encoder`
A transformer-style sequence model.
- larger parameter count than several alternatives
- did not outperform the simpler models here

### `SARIMA`
Classical seasonal time-series model.
- strong lecturer-motivated baseline
- especially effective on NDVI-oriented behavior

### `ensemble_mean`
Deployable ensemble.
- simple mean of selected component predictions
- important because it was the first deployable method to beat `seasonal_last_year` under normalized overall error in the main comparison

### `ensemble_weighted`
Deployable weighted ensemble.
- uses validation-based weighting rather than test labels
- useful, but did not consistently beat the simpler narrower ensemble mean

### `ensemble_oracle_report_only`
Diagnostic upper-bound ensemble.
- chooses the best component using test outcomes
- useful for analysis only
- **not** a fair deployable result

## 10. Raw Target vs `seasonal_residual` Target

Two important target formulations were used for learned models.

### Raw target
The model predicts the monthly feature vector directly.

### Seasonal residual target
The model predicts:
- `y - seasonal_last_year`

Final prediction becomes:
- `seasonal_last_year + predicted_residual`

Why this helps:
- `seasonal_last_year` already captures strong recurring seasonal structure
- the learned model can focus on modeling the **correction** rather than the entire signal

Why `seasonal_last_year` is such a strong baseline:
- agriculture and vegetation patterns are highly seasonal
- same-month previous-year values are often already close to the truth
- this makes any learned improvement hard-earned

In the main results, `seasonal_residual` was the most important learned target mode.

## 11. Loss Modes

Two loss modes were compared for learned models.

### `raw_mse`
Standard MSE on the scaled target outputs.

### `feature_normalized_mse`
Loss reweighting so that large-scale features do not dominate training as much.

Why this matters:
- raw weather features often have much larger numeric scale than AG or NDVI features
- a model can look strong under raw aggregate RMSE mainly by improving a few large-scale weather dimensions
- normalized views are needed to judge cross-feature robustness more fairly

This is why raw RMSE and normalized RMSE can point to different ?best? models.

## 12. Main Results

The main picture after the controlled comparison, SARIMA addition, ensemble evaluation, strict blank-fill checking, and feature ablations is:

- Best raw RMSE for `known_months=1`: **LSTM seasonal_residual**
- Best normalized RMSE for `known_months=1`: **ensemble_mean**
- Best classical model: **SARIMA**
- Best AG raw behavior: **seasonal_last_year**
- Best NDVI raw behavior: **SARIMA**
- Best Weather raw behavior: **LSTM seasonal_residual**
- Broader ensemble with extra GRU / Transformer components did **not** improve over the narrower ensemble
- Feature ablation showed performance is strongly **modality-dependent**

Interpretation:
- if the headline claim is based on raw industrial blank-fill RMSE, the residual LSTM is strongest
- if the claim is about more balanced cross-feature behavior, `ensemble_mean` is stronger
- no single neural model robustly dominates every metric

## 13. Feature Ablation Findings

The feature-group ablation matrix was one of the most informative parts of the project.

Main findings:
- **AG/NDVI context helps weather forecasting**
  - weather-only LSTM was worse than all-features LSTM for the main case
- **all features** remain best for the full raw industrial metric
- **AG+NDVI** can be strongest under normalized comparison in some settings
- all features are **not automatically best** under normalized RMSE
- **SARIMA dominates NDVI-oriented settings** more often than the neural models

This means model performance depends not just on architecture, but also on **what information is present in the feature set**.

## 14. How to Interpret Metrics

### RMSE
Root Mean Squared Error.
- penalizes large errors strongly
- useful for the headline industrial metric
- can be dominated by high-scale weather features

### MAE
Mean Absolute Error.
- more interpretable in some cases
- less sensitive to extreme errors than RMSE

### Normalized RMSE
Typically RMSE divided by train-set feature std or range.
- better for comparing across features with very different scales
- important when AG, NDVI, and weather have different magnitudes

### Feature-level wins
How many features one model beats a baseline on.
- useful for breadth of improvement
- helps reveal whether a gain is narrow or broad

### Modality-level metrics
Aggregate results for AG, NDVI, or weather groups.
- useful for diagnosing where a model is strong or weak

Why raw RMSE and normalized RMSE can disagree:
- raw RMSE can be heavily influenced by weather-scale variables
- normalized RMSE gives a more balanced cross-feature comparison

Which metric to use for which claim:
- use **raw RMSE** for the industrial blank-fill headline
- use **normalized RMSE** for fairness across feature scales
- use **feature wins** to describe breadth
- use **modality metrics** to explain where the gain comes from

## 15. How to Interpret Plots

### Prediction vs ground truth overlays
Useful for seeing:
- whether the seasonal shape is captured
- whether the model tracks peaks and valleys
- whether errors are smooth drift or structural mismatch

### Loss curves
Useful for seeing:
- convergence speed
- overfitting behavior
- whether validation loss diverges from train loss

### RMSE comparison plots
Useful for quick model ranking.
- good for report summary slides
- should always be interpreted together with normalized metrics

### Horizon degradation plots
Useful for recursive forecasting diagnosis.
- show how performance worsens as the model rolls farther into the unknown future

What teammates should look for visually:
- is the model simply copying seasonality?
- does it correct errors meaningfully?
- do improvements happen only in weather-heavy features?
- are later forecast months visibly unstable?

## 16. Current Limitations

Important limitations to state honestly:
- this is **not** a direct yield-prediction system yet
- experiments so far are focused on **IA / 30 counties / 2017?2021** rather than full multi-state scaling
- no single neural model has a robust normalized-metric win over all alternatives
- server storage constraints affected some workflow choices
- there is still a large legacy research script in the codebase
- the new deployment-style package is a handover improvement, but still needs further real-world testing

## 17. Codebase Map

### `src/cropnet_forecasting/`
Cleaner package layer for teammates.
- `data.py`: monthly table loading and input coercion
- `features.py`: feature lists and feature-group utilities
- `scaling.py`: scaler loading and transforms
- `models.py`: model factory and checkpoint loading
- `trainer.py`: clean wrapper around validated training flow
- `predictor.py`: main inference / blank-fill interface
- `blank_fill.py`: recursive blank-fill rollout logic
- `metrics.py`: reusable metric helpers
- `plotting.py`: simple plot helpers
- `config.py`: config loading and dataclass definitions

### `examples/`
Simple usage examples for teammates.

### `configs/`
Starter YAML configs for residual LSTM and ensemble reference use.

### `weights/`
Small included checkpoints and scaler/config artifacts for smoke tests and teammate inference.

### `scripts/research/`
Legacy research workflow preserved as the validated reference implementation.

### `reports/`
Small summaries, CSV tables, and a few representative figures.

### `docs/`
Project notes, artifact manifests, and now this understanding guide.

## 18. How to Use the Developer Handover

Recommended reading order:
1. `docs/PROJECT_QUICK_BRIEF.md`
2. this guide
3. `README.md`
4. `reports/README_RESULTS_SUMMARY.md`

Recommended first actions:
- run the import and example smoke tests
- inspect `weights/` and `configs/`
- read the legacy research script only after understanding the cleaner package layer

How to load weights:
- use `BlankFillPredictor.from_artifacts(checkpoint, scaler, config)`
- the relationship is:
  - checkpoint = learned weights
  - scaler = train-time mean/std normalization reference
  - config = model name, feature group, seq len, target mode, and run settings

How to run blank-fill prediction:
- prepare a monthly feature table with the expected metadata columns and selected feature columns
- run `examples/blank_fill_example.py`

How to train / evaluate:
- use `examples/train_example.py`
- this wraps the validated legacy script rather than replacing it

What not to touch first:
- do not refactor the big research script immediately unless you first understand which parts are already validated
- do not assume the current package layer fully replaces all research functionality yet

## 19. Reproduction Guide

### Environment setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Import smoke test
```bash
python -m compileall src/cropnet_forecasting examples
```

### Inference example
```bash
python examples/inference_example.py \
  --checkpoint weights/lstm_best.pt \
  --scaler weights/scaler.csv \
  --config configs/residual_lstm_all.yaml
```

### Blank-fill example
```bash
python examples/blank_fill_example.py \
  --monthly-table path/to/monthly_features.parquet \
  --checkpoint weights/lstm_best.pt \
  --scaler weights/scaler.csv \
  --config configs/residual_lstm_all.yaml \
  --year 2021 \
  --known-months 1 \
  --output outputs/blank_fill_predictions.csv
```

### Training wrapper example
```bash
python examples/train_example.py --config configs/residual_lstm_all.yaml --mode fit
```

### Where to find artifacts
- small review assets: `reports/`
- small included checkpoints/configs: `weights/`
- deeper research utilities: `scripts/research/`
- large experiment artifacts remain outside Git and must be restored separately from backups

## 20. Report / Presentation Talking Points

- **Problem**: future monthly feature blank filling for partially observed agricultural years
- **Method**: monthly feature forecasting with one-step and recursive blank-fill evaluation
- **Models**: lag baseline, seasonal baseline, neural sequence models, SARIMA, ensembles
- **Evaluation**: raw RMSE, normalized RMSE, modality metrics, feature wins, strict no-future-fill validation
- **Results**: residual LSTM best on raw `known_months=1`; ensemble mean best on normalized RMSE; SARIMA strong on NDVI
- **Limitations**: modality-dependent results, limited temporal scope, no final deployment pipeline yet
- **Future work**: more ablations, scaling, deployment cleanup, downstream yield integration

## 21. Common Questions and Answers

### Why not predict yield directly?
Because this stage is designed to fill missing monthly feature values that a downstream yearly yield model can use later.

### Why is `seasonal_last_year` so strong?
Because the data are seasonal and same-month previous-year values are often already a strong approximation.

### Why use `seasonal_residual`?
Because it lets the learned model correct a strong seasonal baseline instead of learning everything from scratch.

### Why is RMSE so large?
Mostly because weather features have much larger numeric scale than AG and NDVI features.

### Why does normalized RMSE matter?
Because it reduces scale bias and shows whether gains are broad across different feature types.

### Why did Transformer not win despite more parameters?
More capacity does not automatically help when the dataset is limited and seasonality is already strong.

### Why use ensembles?
Ensembles can reduce variance and provide more robust cross-feature behavior when single-model wins are narrow.

### What is SARIMA doing?
It is a classical seasonal time-series model that explicitly models autoregressive seasonal behavior, which works especially well for NDVI-like seasonal patterns.

### What does `known_months=1` mean?
January of the target year is observed, and February to December must be forecast.

### What does `seq_len` mean?
The number of previous monthly steps used as input for the next prediction.

### What is strict no-future-fill?
An evaluation mode that prevents future target-year values from leaking into the model input through interpolation/backfill.

### What should we do next?
Continue the ablation and scaling strategy before attempting final deployment cleanup.

## 22. Recommended Next Work

Recommended next directions:
- continue ablations
- try data augmentation ideas carefully
- scale to all IA counties after the comparison picture is stable
- keep improving the cleaner deployment / handover layer
- start final report writing using the current stable findings
- explore later integration with a downstream yearly yield model
