# Model Comparison V1

## Scope

- State: IA
- Max counties: 30
- Years: 2017-2021
- Train/val/test: 2017-2019 / 2020 / 2021
- Sequence length: 6
- Strict blank-fill enabled
- Models compared: lstm, gru, tiny_mamba_ssm, transformer_encoder
- Target modes: raw, seasonal_residual
- Loss modes: raw_mse, feature_normalized_mse

## Best Results

### One-Step Mean RMSE

| run_name                                | model   |   mean_rmse |
|:----------------------------------------|:--------|------------:|
| ia30_seq6_lstm_seasonal_residual_rawmse | lstm    |     157.082 |

### Best known_months=1 by Raw RMSE

| model   |   known_months |   count |    rmse |     mae |   nrmse_std |   nrmse_range |   smape |   pearson_corr |       r2 | modality   |   lag1_rmse |   seasonal_last_year_rmse | beats_lag1   | beats_seasonal_last_year   |   avg_horizon_rmse |   worst_horizon_rmse |   lag1_rmse_raw |   seasonal_last_year_rmse_raw |   beats_lag1_raw |   beats_seasonal_last_year_raw | run_name                                | target_mode       | loss_mode   | feature_group   |   seq_len |   max_counties |   runtime_seconds |
|:--------|---------------:|--------:|--------:|--------:|------------:|--------------:|--------:|---------------:|---------:|:-----------|------------:|--------------------------:|:-------------|:---------------------------|-------------------:|---------------------:|----------------:|------------------------------:|-----------------:|-------------------------------:|:----------------------------------------|:------------------|:------------|:----------------|----------:|---------------:|------------------:|
| lstm    |              1 |   11550 | 311.802 | 35.3254 |    0.743805 |      0.151854 | 65.6714 |       0.956358 | 0.902214 | all        |     806.103 |                   333.227 | True         | True                       |                nan |                  nan |             nan |                           nan |              nan |                            nan | ia30_seq6_lstm_seasonal_residual_rawmse | seasonal_residual | raw_mse     | all             |         6 |             30 |            6.9757 |

### Best known_months=1 by Normalized RMSE

| model              |   known_months |   count |    rmse |     mae |   nrmse_std |   nrmse_range |   smape |   pearson_corr |       r2 | modality   |   lag1_rmse |   seasonal_last_year_rmse | beats_lag1   | beats_seasonal_last_year   |   avg_horizon_rmse |   worst_horizon_rmse |   lag1_rmse_raw |   seasonal_last_year_rmse_raw |   beats_lag1_raw |   beats_seasonal_last_year_raw | run_name                   | target_mode   | loss_mode              | feature_group   |   seq_len |   max_counties |   runtime_seconds |
|:-------------------|---------------:|--------:|--------:|--------:|------------:|--------------:|--------:|---------------:|---------:|:-----------|------------:|--------------------------:|:-------------|:---------------------------|-------------------:|---------------------:|----------------:|------------------------------:|-----------------:|-------------------------------:|:---------------------------|:--------------|:-----------------------|:----------------|----------:|---------------:|------------------:|
| seasonal_last_year |              1 |   11550 | 333.227 | 35.6949 |    0.693587 |      0.140959 | 41.8924 |       0.948912 | 0.888314 | all        |     806.103 |                   333.227 | True         | False                      |                nan |                  nan |             nan |                           nan |              nan |                            nan | ia30_seq6_gru_raw_normloss | raw           | feature_normalized_mse | all             |         6 |             30 |           7.13436 |

### Best known_months=6 by Raw RMSE

| model   |   known_months |   count |    rmse |    mae |   nrmse_std |   nrmse_range |   smape |   pearson_corr |       r2 | modality   |   lag1_rmse |   seasonal_last_year_rmse | beats_lag1   | beats_seasonal_last_year   |   avg_horizon_rmse |   worst_horizon_rmse |   lag1_rmse_raw |   seasonal_last_year_rmse_raw |   beats_lag1_raw |   beats_seasonal_last_year_raw | run_name                                | target_mode       | loss_mode   | feature_group   |   seq_len |   max_counties |   runtime_seconds |
|:--------|---------------:|--------:|--------:|-------:|------------:|--------------:|--------:|---------------:|---------:|:-----------|------------:|--------------------------:|:-------------|:---------------------------|-------------------:|---------------------:|----------------:|------------------------------:|-----------------:|-------------------------------:|:----------------------------------------|:------------------|:------------|:----------------|----------:|---------------:|------------------:|
| lstm    |              6 |    6300 | 79.6397 | 15.076 |    0.700839 |      0.147164 | 65.4039 |       0.995001 | 0.988751 | all        |     607.564 |                   80.2317 | True         | True                       |                nan |                  nan |             nan |                           nan |              nan |                            nan | ia30_seq6_lstm_seasonal_residual_rawmse | seasonal_residual | raw_mse     | all             |         6 |             30 |            6.9757 |

### Best known_months=6 by Normalized RMSE

| model              |   known_months |   count |    rmse |     mae |   nrmse_std |   nrmse_range |   smape |   pearson_corr |       r2 | modality   |   lag1_rmse |   seasonal_last_year_rmse | beats_lag1   | beats_seasonal_last_year   |   avg_horizon_rmse |   worst_horizon_rmse |   lag1_rmse_raw |   seasonal_last_year_rmse_raw |   beats_lag1_raw |   beats_seasonal_last_year_raw | run_name                   | target_mode   | loss_mode              | feature_group   |   seq_len |   max_counties |   runtime_seconds |
|:-------------------|---------------:|--------:|--------:|--------:|------------:|--------------:|--------:|---------------:|---------:|:-----------|------------:|--------------------------:|:-------------|:---------------------------|-------------------:|---------------------:|----------------:|------------------------------:|-----------------:|-------------------------------:|:---------------------------|:--------------|:-----------------------|:----------------|----------:|---------------:|------------------:|
| seasonal_last_year |              6 |    6300 | 80.2317 | 14.0951 |    0.621558 |      0.131057 | 41.8913 |       0.997337 | 0.988583 | all        |     607.564 |                   80.2317 | True         | False                      |                nan |                  nan |             nan |                           nan |              nan |                            nan | ia30_seq6_gru_raw_normloss | raw           | feature_normalized_mse | all             |         6 |             30 |           7.13436 |

## Caveats

- Raw RMSE remains strongly weather-scale sensitive.
- Seasonal_last_year is still the key comparator under normalized metrics.
- These runs stay within IA 30-county scope and are not yet the final deployment recipe.