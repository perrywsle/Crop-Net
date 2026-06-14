# 5. Conclusions

## 5.1 Summary and Reflection

The final system is a complete CropNet workflow that turns raw AG imagery, NDVI scenes, and weather records into a monthly 35-feature table, forecasts missing future months, trains a separate yield regression model from USDA county labels, and exposes the result through the TaoCrop demonstrator. In other words, the project does not stop at model comparison: it connects preprocessing, forecasting, yield estimation, and user-facing presentation into one end-to-end technical pipeline.

The strongest result from the validated forecasting comparison is the **LSTM seasonal_residual** model for the main early-season blank-fill setting, especially when only January is known and the model must recursively generate the rest of the year. That result matters because it reflects the real operational case the project is designed for: filling in a partially observed year well enough that downstream annual analysis can still proceed. At the same time, **SARIMA** and **seasonal_last_year** remain central comparators, because crop signals are strongly seasonal and a good baseline already captures a large fraction of the structure in the data.

## 5.2 Final Technical Outcome

The final forecasting pipeline is built around a few practical design decisions that held up across the experiments:

- The monthly feature table is built from aligned AG, NDVI, and weather data rather than from one modality in isolation.
- The model input is a fixed rolling window with `seq_len=6`, which gives the learner enough recent context to infer short-term trajectory while still keeping the sequence length manageable.
- Forecasting is evaluated in both direct one-step form and recursive blank-fill form, with strict checks to avoid leaking future target-year information into the prediction window.
- Learned models are compared against deterministic persistence-style baselines, a seasonal baseline, a classical seasonal time-series model, and ensemble combinations.

The final conclusion from this pipeline is not that one model dominates every setting. Instead, the most defensible outcome is that the learned residual forecaster is the best choice for the main raw blank-fill metric, while the ensemble and seasonal baselines remain very strong depending on how the error is measured. That is the right outcome for an agricultural forecasting system, because the task is not just to minimize a single loss value; it is to preserve seasonal shape, cross-modal balance, and stable behavior over a recursive horizon.

## 5.3 Physics-Informed Training

The **physics-informed training** path was an important part of the final system, even though it is not a replacement for the data-driven forecast objective. In the implementation, the chosen learned backbone is wrapped by `PINNForecaster`, and the auxiliary `CropPhysicsModule` adds soft penalties for crop-state consistency, plausible growth dynamics, NDVI seasonality, and weather relationships. The latent variables are interpreted as biomass, phenology, and water condition, and the physics term is introduced after a warmup period rather than immediately.

That warmup is important. If the physics term is applied too early, it can constrain the model before it has learned the basic empirical structure of the monthly data. The final training design therefore treats the physics objective as a regularizer: it guides the forecast toward realistic crop behavior, but it does not override the supervised target loss. This gave the project a better inductive bias and a clearer story for interpretability, while also showing a key limitation of physics-informed learning in this setting: the physics prior can improve structure, but it cannot compensate for weak labels, noisy extraction, or an underpowered seasonal baseline.

## 5.4 Yield Model

The **yield model** is intentionally separated from the monthly feature forecaster. That separation is the right architecture for this project because the forecasting stage and the yield stage solve different problems. The forecasting stage completes the monthly AG, NDVI, and weather table; the yield stage maps that completed table to annual county yield.

In the official yield benchmark, the dataset is prepared by copying the USDA annual yield label onto the monthly rows for each county-year, then training several candidate regressors on the resulting monthly-grain table. The codebase shows that the yield pipeline is benchmarked against both simple baselines and trainable models, with **Ridge** emerging as the best trainable model in the current official run. That result is useful because it keeps the yield stage simple, explainable, and reproducible, rather than forcing a deep model where a linear method is already competitive.

The main lesson from the yield stage is that forecast quality and yield quality are related but not identical. A monthly forecaster can produce a visually good feature trajectory and still not guarantee the best annual yield regression, because the yield target compresses information across time and modalities differently. That is why the project keeps the two stages separate and treats the yield model as a downstream consumer of the monthly feature table rather than as the primary forecasting target.

## 5.5 Demonstrator

The **demonstrator** turns the technical pipeline into a usable application. The TaoCrop browser dashboard runs on a local FastAPI backend, accepts uploaded crop data, rebuilds the monthly feature table, applies the saved models, and presents the output through charts, summary cards, feature drivers, and a local assistant layer. This matters because it shifts the project from a notebook-style experiment to an end-to-end system that can actually be shown to a user.

The demonstrator also clarified the design boundary of the project. The dashboard is not just a polished wrapper around a single model. It demonstrates the whole stack:

- preprocessing converts raw files into monthly features,
- forecasting fills in the missing months,
- the yield model converts the monthly table into a yield estimate,
- and the chat/summary layer explains the result in plain language.

That architecture makes the project easier to reason about and easier to present. It also exposes what still needs work, because each stage can be inspected independently instead of hiding failure modes inside one opaque prediction.

## 5.6 Challenges Encountered

Several challenges shaped the final outcome:

- Leakage control was critical. Recursive blank-fill evaluation can look artificially strong if future months leak into the known window, so strict no-future-fill validation was necessary.
- Seasonality is a hard prior to beat. Same-county, same-month previous-year values are already strong for agriculture, so learned models have to earn their improvement over a very competitive baseline.
- Raw and normalized metrics tell different stories. Weather features have larger numeric scales than AG or NDVI features, so raw RMSE can overstate weather-heavy improvements unless normalized views are also checked.
- Physics-informed training needed tuning. The auxiliary loss had to be weighted carefully and delayed with warmup so it improved structure without destabilizing the forecast objective.
- The yield stage did not automatically improve just because the forecasting stage improved. The downstream regression task remained sensitive to feature-table quality, feature grouping, and the choice of regressor.

These issues are not incidental. They are the main reason the project needed a validated comparison pipeline rather than a single training script.

## 5.7 What Was Learned

The project showed that monthly crop forecasting is fundamentally a seasonal sequence problem with multiple valid notions of success. A model that wins on raw RMSE is not always the best balanced model across modalities, and a model that looks strong on one feature family can still be weak on another. This is why the report keeps raw RMSE, normalized metrics, modality-level analysis, and baseline comparisons side by side.

The project also showed that **seasonal baselines remain important**. The `seasonal_last_year` rule is not a trivial placeholder. It captures the fact that agricultural growth patterns recur annually, and it often provides a very strong reference point for AG and NDVI behavior. Even when a learned model wins overall, it should still be judged against the seasonal baseline because that baseline reflects a real prior about crop development, not just a weak strawman.

The most useful modeling insight is that residual learning is more effective than trying to learn everything from scratch. Predicting the correction to a seasonal anchor gives the network a simpler job and better aligns the model with the way crop signals actually evolve over time. The most useful systems insight is that the project works best when the forecasting pipeline, physics-informed training, yield model, and demonstrator are treated as separate but connected layers of one workflow.

## 5.8 Closing Statement

Overall, the final technical outcome is a working end-to-end CropNet system that can reconstruct missing monthly crop features, apply physics-informed learned forecasting when appropriate, train a separate annual yield regressor, and present the result in a usable dashboard. The project did not eliminate the value of seasonal heuristics; instead, it showed that strong seasonal baselines are still essential reference points for any learned model in this domain. The main achievement is therefore not absolute model dominance, but a credible and well-instrumented pipeline that combines forecasting, interpretability, and deployment-oriented presentation in a single research-to-demonstration flow.
