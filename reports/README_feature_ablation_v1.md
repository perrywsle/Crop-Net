# Feature Ablation v1

## Scope

- State: IA
- Counties: 30
- Years: 2017-2021
- Split: train 2017-2019 / val 2020 / test 2021
- Seq len: 6
- Target mode: seasonal_residual
- Loss mode: raw_mse
- Models: naive_lag1, seasonal_last_year, sarima, lstm, ensemble_mean
- Strict blank-fill: enabled

## Key Results

- Best known_months=1 raw RMSE: sarima on ndvi (RMSE 3.208)
- Best known_months=1 normalized RMSE: ensemble_mean on ag (nRMSE_std 0.5847)
- Best known_months=6 raw RMSE: lstm on ag (RMSE 4.171)

## Caveats

- Ablations were staged under /dev/shm on the server because /home/student was full.
- Ensemble rows are only included where the per-run pipeline emitted ensemble_mean directly.
- SARIMA fit/fallback counts come from each run's strict blank-fill fit summary.
