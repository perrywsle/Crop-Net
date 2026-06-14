# CropNet Monthly Blank-Fill Forecasting and Yield Modeling Draft

## Table of Contents
1. Introduction
2. Dataset
3. Methods, Training, and Evaluation
4. AI Demonstrator
5. Conclusions
6. References
7. Appendix A
# 1. Introduction

This report studies monthly blank-fill forecasting for CropNet-style agricultural data. The task is to reconstruct missing monthly AG, NDVI, and weather feature vectors for a partially observed year using the observed portion of the year together with prior seasonal history. The system is therefore a feature-completion pipeline rather than a direct yield predictor: it fills the missing monthly table first, and only then hands the completed year to downstream yield analysis or regression.

That separation matters. Agricultural forecasting is often discussed as a single end-to-end yield problem, but the codebase in this repository makes a different design choice. It preserves the monthly structure of the data, maintains modality-specific feature behavior, and evaluates whether a model can realistically infer the missing months of a target year without looking ahead. In practical terms, the project asks whether January, or the first few months of a year, contain enough signal to support a credible reconstruction of the rest of the year. The answer is useful for early-season decision support, for operational blank filling, and for any downstream yield model that depends on a completed monthly feature table.

The main benchmark is centered on Iowa corn. The repository’s validated comparison runs focus on Iowa counties, with the primary blank-fill study scoped to a 30-county IA subset and a sequence length of six months. The most important scenario is `known_months=1`, which corresponds to a realistic early-season setting in which only January is observed and February through December must be inferred recursively. This is the hardest and most operationally meaningful case in the project because the model must roll forward month by month, append its own predictions back into the history, and continue forecasting under compounding error.

The problem is also inherently multi-user. Operational users need a filled monthly table early enough to support decisions and downstream estimation. Researchers and engineers need a reproducible comparison framework that can distinguish between raw-feature accuracy, scale-normalized performance, and modality-specific behavior. The browser application in the repository is explicitly framed as a farmer-friendly dashboard, while the research scripts and report artifacts are organized for teammate handoff and method comparison. The introduction below therefore frames the problem both as a practical agricultural blank-fill task and as a controlled forecasting study.

## 1.1 Background and Motivation

CropNet-style monthly data combine several agricultural modalities into one longitudinal table. In this repository, the canonical monthly representation contains 35 features spanning three groups: AG, NDVI, and weather. Those modalities are not interchangeable. AG features capture agricultural surface or field appearance signals, NDVI features summarize vegetation condition, and weather features encode monthly meteorological context. Because each modality evolves seasonally and at different scales, forecasting the missing months of a year is not equivalent to predicting a single annual target. A useful blank-fill model must preserve seasonal structure, respect modality-specific dynamics, and remain numerically stable when the predictions are fed back into later months.

This is why recursive blank fill is the central task rather than one-step forecasting alone. In a recursive setting, the model predicts the next missing month, appends that prediction to the history, and then uses the augmented sequence to forecast the next month. That procedure reflects the actual use case for a partially observed year: the system may begin with only a short observed prefix, then continue filling the rest of the calendar as the year unfolds. The `known_months` setting makes this explicit. `known_months=0` means the system must forecast the entire year, while `known_months=1` leaves only January observed and forces the model to generate the remaining eleven months. The repository treats `known_months=1` as the most important industrial test because it captures the earliest practical forecasting window.

The motivating application is downstream yield relevance, not yield prediction itself. The blank-fill stage exists because many yield workflows consume monthly features rather than raw imagery or raw weather records. If the monthly AG, NDVI, and weather table can be completed credibly for the current year, then a later yield model can use that completed table to estimate annual corn yield. This separation is reflected in the repository structure: the forecasting code focuses on monthly feature completion, while separate yield-training utilities and dashboard components consume monthly inputs later. The blank-fill model therefore acts as the bridge between partially observed seasonal data and a downstream yield pipeline.

The Iowa corn focus is also motivated by practical agricultural seasonality. Corn features are strongly seasonal, county-specific history matters, and same-month previous-year behavior is often a powerful predictor. The repo makes this explicit by comparing learned models against deterministic baselines that exploit temporal persistence. The strongest of those baselines, `seasonal_last_year`, predicts a month from the same county and same month in the previous year. That baseline is not merely a trivial comparator: in agricultural time series it often carries substantial signal because weather patterns, crop phenology, soil conditions, and management practices all create recurring annual structure. Any learned model that claims improvement must therefore beat a strong seasonal prior, not just a naive lag.

## 1.2 Project Objectives

The project has three concrete objectives.

1. Build a validated monthly forecasting pipeline for CropNet-style AG, NDVI, and weather features, with Iowa corn as the primary benchmark and the IA 30-county subset as the core evaluation scope.
2. Compare learned sequence models, classical time-series baselines, and ensemble methods under strict recursive blank-fill evaluation, especially for `known_months=1`.
3. Produce completed monthly feature tables that can be handed to a downstream yearly yield workflow, while keeping the feature-filling stage distinct from the final yield model.

The implementation reflects those objectives in several ways. First, the model family is broad enough to test whether the problem is best handled by a learned neural sequence model, a classical seasonal model, or a deployable ensemble. The repository includes LSTM, GRU, transformer encoder, and tiny Mamba-style learned models; deterministic baselines such as `naive_lag1` and `seasonal_last_year`; classical SARIMA; and ensemble methods such as `ensemble_mean` and `ensemble_weighted`. This is a deliberate design choice. The task is not so specialized that one architecture should be assumed in advance, and the evaluation needs to answer whether the modeling gains justify the additional complexity.

Second, the project uses a seasonal-residual target formulation in addition to the raw target formulation. In residual mode, the model learns the correction relative to the same-month previous-year baseline instead of predicting the full feature vector from scratch. That formulation is central to the project because it matches the structure of the data: the seasonal baseline already carries much of the recurring annual pattern, and the learned model can focus on the deviations that matter for the current year. In the codebase, this residual setup is treated as the main learned target mode and is one of the reasons the strongest learned model remains competitive under strict blank-fill evaluation.

Third, the evaluation protocol is designed to be realistic rather than optimistic. The main blank-fill comparison uses fixed monthly sequence windows, no-future-fill logic, and year-based splits that keep evaluation honest. The repository’s summaries consistently emphasize raw RMSE and normalized RMSE because the two views answer different questions. Raw RMSE shows which model is closest in the original feature space, while normalized RMSE tests whether a model remains balanced across features with very different numeric scales. This distinction is important in a multi-modality setting where weather variables can dominate raw error simply because they have larger magnitudes than AG or NDVI features.

The intended user base follows from those objectives. Farmers and farm-facing users need a practical early-year feature completion tool. Analysts and model developers need reproducible comparisons between residual neural models, classical seasonal methods, and ensembles. Downstream yield workflows need a completed monthly table that preserves seasonality and monthly granularity. The report is therefore written to support all three audiences without collapsing the blank-fill task into a generic forecasting benchmark.

## 1.3 Summary of Outcomes

The validated results point to a clear but nuanced ranking. For the main `known_months=1` case, the best raw RMSE is achieved by `LSTM seasonal_residual`. This is the most important result for the early-season blank-fill setting because it shows that a learned residual sequence model can outperform the deterministic seasonal baseline when the evaluation is strict and recursive. In other words, the learned model is not merely memorizing last year’s pattern; it is learning useful corrections on top of a strong seasonal prior.

When the same comparison is viewed through normalized error, the best result comes from `ensemble_mean`. This matters because it indicates that the most balanced cross-feature behavior is not necessarily the same as the best raw aggregate score. The ensemble is therefore the strongest choice when the goal is stable performance across modalities and feature scales, even though the residual LSTM remains the best raw predictor in the primary industrial case.

The classical benchmark `SARIMA` remains important as well. It is the best classical model in the repository’s validated comparisons and performs especially well for NDVI-oriented settings. That outcome is consistent with the broader seasonal structure of the problem: NDVI series often exhibit strong periodic behavior that classical seasonal time-series models can capture effectively. The result is a useful reminder that modern neural models do not automatically dominate on agricultural monthly data.

The deterministic `seasonal_last_year` baseline also earns a central role in the final interpretation. It is the best raw AG behavior in the validated summaries and remains the key seasonal comparator throughout the experiments. Its strength is exactly why the problem is challenging: if a same-month previous-year lookup is already close to the truth, then a learned model must do more than approximate the obvious seasonal trend. It must also correct for year-specific deviations, modality-specific dynamics, and recursive forecasting error.

The best raw weather behavior is again `LSTM seasonal_residual`, which suggests that the residual formulation is particularly helpful for the more dynamic and noisy weather modality. Taken together, the results support the following project-level conclusion: the blank-fill problem is best approached as a seasonal, multi-modal forecasting task with strong baseline structure, not as a generic sequence regression problem. The residual LSTM is the strongest raw model for the key early-season case, `ensemble_mean` is the strongest normalized choice, `SARIMA` is the best classical comparator, and `seasonal_last_year` remains a formidable baseline that any deployment-oriented system must respect.

These outcomes set up the rest of the report. The next sections can focus on how the monthly tables are constructed, how the models are trained and rolled out recursively, and how the evaluation protocol distinguishes between raw predictive accuracy, normalized robustness, and downstream usefulness for yield-oriented workflows. The introduction establishes the main claim: monthly blank-fill forecasting is a distinct and practically useful agricultural problem, and the repository contains validated evidence that the best solution depends on whether one values raw error, normalized stability, or classical interpretability.

# 2. Dataset

## 2.1 Data Source

The dataset used in this project is a CropNet monthly feature table built from the Hugging Face repository `CropNet/CropNet`. The repository workflow narrows the download to the exact crop, state, and year range needed for the experiment instead of pulling the full archive. In this report, the main focus is Iowa corn, with `crop_type = corn` and `state_codes = IA`, across the 2017 to 2022 window. This choice is not accidental. The Iowa corn setting gives a strong agricultural seasonal signal, enough county coverage for the comparison runs, and a realistic environment for testing recursive blank fill under a temporal split.

The raw inputs come from three CropNet modalities plus USDA labels. The AG stream is stored as Sentinel 2 agriculture imagery in HDF5 form. The NDVI stream is stored as Sentinel 2 vegetation imagery in HDF5 form. The weather stream is stored as WRF HRRR CSV files with daily meteorological observations. The downstream yield task uses USDA county level corn yield CSV files as the label source. The forecasting stage does not consume the USDA yield label directly. Instead, the yield label becomes relevant later when the completed monthly feature table is handed to a separate regression pipeline.

The canonical monthly extraction artifact recorded in the repository is `data/raw/cropnet/corn-IA-2017_2022/official_monthly_feature_table.parquet`. The prepared split used for modeling is written to `data/training/all.parquet`, `data/training/train.parquet`, `data/training/val.parquet`, `data/training/test.parquet`, `data/training/scaler.csv`, and `data/training/metadata.json`. These artifacts are the source of truth for the monthly forecasting experiments because they preserve the exact monthly table, split assignment, and training statistics used by the model code.

The dataset is organized at monthly grain, with one row per county, crop, year, and month. This is the correct grain for the project because the models are asked to forecast missing months, not a single annual summary. The prepared metadata in the repository reports 7128 total rows, split into 4752 training rows, 1188 validation rows, and 1188 test rows. The train years are 2017 to 2020, the validation year is 2021, and the test year is 2022.

The feature contract is fixed at 35 monthly predictors. Those predictors are split into 8 AG features, 12 NDVI features, and 15 weather features. This fixed contract matters because the forecasting models, the physics module, and the downstream yield pipeline all assume the same ordered feature vector. The repository does not treat the 35 features as interchangeable columns. Each feature family serves a different agronomic purpose, and the model design reflects that separation.

| Modality | Count | Features |
| --- | ---: | --- |
| AG | 8 | `ag_green_pixel_ratio`, `ag_vegetation_area_percent`, `ag_brown_yellow_pixel_ratio`, `ag_soil_exposure_ratio`, `ag_shadow_cloud_ratio`, `ag_mean_brightness`, `ag_texture_entropy`, `ag_field_uniformity_score` |
| NDVI | 12 | `ndvi_mean`, `ndvi_median`, `ndvi_max`, `ndvi_std`, `ndvi_cv`, `ndvi_p25`, `ndvi_p75`, `ndvi_above_0_3_ratio`, `ndvi_above_0_5_ratio`, `ndvi_above_0_7_ratio`, `ndvi_low_ratio`, `ndvi_valid_coverage_ratio` |
| Weather | 15 | `weather_temp_mean`, `weather_temp_max`, `weather_temp_min`, `weather_gdd`, `weather_heat_stress_days`, `weather_cold_stress_days`, `weather_total_precipitation`, `weather_precipitation_days`, `weather_heavy_rain_days`, `weather_drought_index`, `weather_humidity_mean`, `weather_wind_mean`, `weather_solar_radiation_mean`, `weather_vpd_mean`, `weather_temp_range_mean` |

The monthly table also relies on a small metadata contract. `county_id` is the county identifier and is used as a FIPS normalized join key. `crop_type` identifies the crop and keeps the downstream joins stable. `year` is used for split assignment and seasonal alignment. `month` is the monthly time index and is the key that allows the model to preserve seasonal ordering. The combination of these four metadata fields is what turns the crop data into a panel structure instead of a disconnected set of monthly records.

## 2.2 Data Processing

The preprocessing pipeline turns the raw CropNet files into a ground truth monthly table that can be used directly by the forecasting models. The most important design choice is to keep the data at monthly grain, rather than collapsing the year into a single annual record. That design is necessary because the forecasting problem in this project is about filling missing months in a partially observed year. If the data were aggregated too early, the model would lose the temporal structure that makes blank fill meaningful.

Selective acquisition is the first step. The download logic builds allow patterns from the requested crop, years, and state codes. Only the relevant Iowa corn AG, NDVI, weather, and USDA files are fetched. This keeps the local snapshot focused on the experiment rather than mirroring the full CropNet repository. The benefit is practical as well as computational. The models only need the years and counties used in the benchmark, so pulling unrelated states or crops would add noise without improving the comparison.

County and crop normalization comes next. County identifiers are normalized to five digit FIPS strings with leading zeros preserved, and crop names are normalized to a consistent lower case form. This is a small but important step because the merge keys must be exact for the monthly table to remain stable across files and years. State codes are normalized before download so the raw directory layout stays deterministic and the downstream file discovery logic remains reproducible.

Monthly feature extraction and alignment then convert the raw files into the canonical panel. AG and NDVI HDF5 files are read county by county, the month is inferred from the source date key, and each image grid is converted into monthly features before aggregation. Weather CSV files are read as daily observations, filtered to the selected counties, and aggregated into monthly weather features. The modality tables are then merged on `county_id`, `crop_type`, `year`, and `month`. The merged table is sorted by the same keys so the output is stable and the sequence builder can later assume that rows are in chronological order.

Schema enforcement is deliberately strict. The canonical 35 feature schema is preserved even if some columns are not present in a given raw slice. Any missing canonical feature column is added explicitly as `NaN` so the table shape stays consistent. The validation script also checks that the monthly table has the required metadata columns and that the county, crop, year, and month key does not duplicate rows. This protects the forecasting code from silent schema drift, which would otherwise be very easy to miss in a large monthly table.

The table is then prepared for sequence modeling with a train only scaling step. The training split is used to compute the feature means and standard deviations, and the resulting scaler is written to `scaler.csv`. Validation and test rows reuse the same statistics so no future information leaks into the normalization stage. In the scaler, zero or missing standard deviations are replaced with 1.0 so the transform remains numerically safe. That detail matters because the monthly panel includes both weather variables with large magnitude and ratio style features with small magnitude, and the scale difference is large enough to distort training if it is not handled explicitly.

The repository also separates the ground truth dataset from any generated forecast artifacts. The monthly table loader rejects files that contain forecast specific columns such as `forecast_step`, `known_months`, `source_note`, or `y_pred`. It also rejects source paths that look like blank fill or forecast outputs. That safeguard matters because the blank fill evaluation itself produces recursive prediction tables later in the workflow, and those outputs must never be confused with the original ground truth monthly features.

The final preprocessing step is the construction of validated rolling windows for forecasting. The models operate on a fixed sequence length of 6. Each sample therefore uses the previous 6 months to predict the next month. The sequence builder only accepts contiguous calendar windows, discards windows with non finite values, and requires seasonal residual mode to have a valid same county same month previous year reference. The result is not just a row split. It is a set of temporal windows that satisfy the monthly chronology required by the forecasting task.

| Split | Years | Rows |
| --- | --- | ---: |
| Train | `2017-2020` | `4,752` |
| Validation | `2021` | `1,188` |
| Test | `2022` | `1,188` |

The downstream yield dataset is built from the same monthly table and USDA yield labels, but it is a separate artifact. In the committed yield preparation output, the dataset contains 6624 rows with 4452 training rows, 1008 validation rows, and 1164 test rows. That yield dataset is not used for the monthly forecasters. It exists to support the later yield regression stage, which consumes monthly features that have already been prepared and joined to county year yield labels.

# 3. Methods, Training, and Evaluation

This section describes the full modeling stack used in CropNet. The pipeline begins with monthly feature extraction from multimodal remote sensing and weather data, continues through sequence forecasting with both learned and physics informed backbones, and ends with a separate yield regression pipeline that consumes the completed monthly table. The codebase treats these as connected but distinct problems. The monthly forecaster is responsible for reconstructing missing months, while the yield model is responsible for mapping the completed monthly history to annual county yield.

## 3.1 Feature Engineering, Feature Extraction, and Image Processing

The canonical monthly feature table contains 35 model inputs. Eight of those inputs come from AG imagery, twelve come from NDVI imagery, and fifteen come from monthly weather summaries. The 35 feature contract is the central interface of the project because every learned forecaster, every baseline model, every physics loss term, and the downstream yield model assumes the same feature order. The design is therefore not just a convenient preprocessing choice. It is the contract that makes the entire system coherent.

| Modality | Core features | Count | Explanation |
|---|---:|---:|---|
| AG imagery | vegetation color, canopy coverage, patch structure, texture, and field quality | 8 | These features help the model see visible crop cover, bare soil, senescence, and image quality, which are all important when the monthly canopy signal is noisy or partially obscured. |
| NDVI imagery | pixel statistics, threshold coverage, and spatial health proxies | 12 | These features summarize vegetation vigor and within scene variability, which lets the model capture crop growth stage, stress heterogeneity, and valid coverage quality. |
| Weather | monthly temperature, stress, precipitation, drought, humidity, wind, and solar summaries | 15 | These features describe the physical forcing on crop growth, which is essential for predicting seasonal deviations and for conditioning the physics loss. |
| Total |  | 35 | The fixed contract gives every model the same monthly state vector, which keeps the forecasting comparison controlled. |

The feature engineering path is intentionally conservative. The repository does not learn the monthly features from scratch inside the forecast model. Instead, the upstream preprocessing code converts raw AG imagery, NDVI scenes, and weather observations into a stable monthly table first, and the forecasting models operate on that completed table. This design reduces the burden on the sequence model and makes the comparison more interpretable because the model is learning temporal structure, not raw image semantics.

### Image Processing Summary

The AG and NDVI streams are derived from county indexed HDF5 grids. The preprocessing logic reads the grids month by month, infers the timestamp from the source key, computes monthly summaries at the county level, and merges the outputs into one panel table. AG imagery is transformed into canopy and field appearance statistics that reflect vegetation color, exposed soil, shadow contamination, texture, and uniformity. NDVI imagery is reduced to statistics that summarize the distribution of valid vegetation pixels, the shape of the vegetation response, and the proportion of the scene that passes key thresholds. Weather data is taken from daily CSV records and aggregated into monthly agronomic summaries so it aligns with the imagery time scale. In effect, the image processing step converts heterogeneous raw sources into a common monthly state representation that can be compared across years and counties.

The AG feature family is useful because visible canopy structure is an early indicator of crop condition even when the numerical vegetation index is noisy. Green pixel ratios and vegetation area percentages provide a proxy for canopy closure. Brown and yellow ratios, soil exposure ratios, and shadow or cloud ratios capture late season decline, thin canopy, bare ground, and image contamination. Brightness, texture entropy, and field uniformity score provide a second layer of information about image quality and spatial regularity. These last three are particularly useful when the extracted image is visually ambiguous, because they help the model distinguish a weak crop signal from a noisy acquisition artifact.

The NDVI family is useful because it gives a more direct view of vegetative vigor than AG appearance alone. Mean, median, and maximum NDVI values summarize the overall greenness of the county scene. Standard deviation and coefficient of variation quantify how uneven that greenness is within the county. Percentiles and threshold ratios reveal whether the county is dominated by weak vegetation, moderate vegetation, or dense vegetation. Valid coverage ratio is important because a scene with many invalid pixels should not be treated like a clean scene even if its visible pixels look healthy. In short, NDVI features provide the model with both vegetation level and measurement quality.

The weather family is useful because it encodes the physical environment that drives crop growth. Temperature means and ranges describe general thermal conditions. Growing degree days summarize heat accumulation and crop development potential. Heat stress and cold stress days capture harmful extremes. Precipitation, precipitation days, and heavy rain days describe water supply and rainfall intensity. Drought index, humidity, wind, solar radiation, and vapor pressure deficit describe atmospheric demand and water loss pressure. These variables are especially important for the physics module because they allow the latent crop states to be regularized against a plausible growth and stress trajectory.

The monthly table is standardized after feature extraction using training only statistics. The model consumes scaled monthly features during training, and predictions are inverse transformed back to the raw feature scale for evaluation. This is important because the 35 features are not commensurate. Some are ratios in the unit interval, some are counts, and some are meteorological quantities with much larger numerical ranges. Without scaling, the model would be biased toward the largest magnitude variables rather than the most informative ones.

### 3.1.1 Gamma SSM

In this report, Gamma SSM refers to the repository’s custom state space model variant, which is implemented as `tiny_mamba_ssm` in the codebase. The alias is not cosmetic. The model factory normalizes `gamma_ssm` to `tiny_mamba_ssm` when creating models, inferring architectures, and loading checkpoints, so the report should treat the two names as the same family. The important point is that this is not a stock transformer or a standard recurrent unit. It is the repository’s own compact state space forecaster built to capture long range monthly dependencies with a small number of learned parameters.

The model is organized as a projection layer, one or more state space blocks, and a prediction head. A compact Mermaid sketch of the main data flow is below.

```mermaid
flowchart LR
    X[Input monthly window, 35 features] --> P[Linear projection, 35 to d_model]
    P --> B1[TinyMambaBlock, about 58k params in shared run]
    B1 --> B2[Optional additional block, if num_layers greater than 1]
    B2 --> H[LayerNorm, Dropout, Linear head]
    H --> Y[Next month 35 feature forecast]
```

The first design choice is the input projection. The raw monthly feature vector has 35 dimensions, but the state space computation is much more effective in a latent width that is wide enough to represent shared crop dynamics across modalities. The linear projection therefore maps the input into a hidden space before any sequence mixing happens. That is a standard but important step because it lets the model separate the representation space from the output space. The output still has to recover the 35 feature vector, but the internal state is allowed to be richer than the observed features.

The second design choice is the block structure. The custom block uses a gated content stream, a depthwise temporal convolution, and a small recurrent style state update. The depthwise convolution is important because it injects short range locality before the state update. Monthly crop trajectories are smooth enough that adjacent months often matter more than distant months, but they are also seasonal enough that the model still needs a longer range memory. The convolution handles the short range interaction, while the state update carries the longer range memory. The gate stream then controls how much of the content stream is allowed to influence the hidden state at each time step.

The third design choice is the state representation. The block uses a learned decay matrix and a small state tensor per latent channel. This allows the model to evolve a latent crop state over the sequence without explicitly using attention over all past months. The point is not to imitate a transformer with a different name. The point is to build a small state space system that can maintain seasonal memory with less overhead than a large self attention stack. That is why Gamma SSM is a useful comparison point in this project. It tests whether a compact state space model can match or outperform the more familiar recurrent baselines on a monthly agricultural sequence.

The fourth design choice is residual placement. The state space block adds its output back to the residual input and then normalizes the result. This is important because the model is being trained on a small and strongly seasonal dataset. Residual connections make the optimization less brittle and help the hidden state preserve the direct monthly signal from the input window rather than replacing it entirely with a transformed latent sequence. The final LayerNorm and linear head then map that stabilized representation back to the 35 output features.

The custom Gamma SSM family is therefore best understood as a compact sequence learner that balances short range smoothing, latent state retention, and residual preservation. In the shared comparison run, the model has 58432 trainable parameters. In the dedicated sweep preset used for the best Mamba style configuration, the width is increased and the parameter count rises accordingly, which reflects the usual tradeoff between capacity and overfitting risk on a small agricultural panel.

### 3.1.2 Attention

The attention based forecaster is the repository’s `TransformerEncoderForecaster`. It is a standard encoder only monthly sequence model, but it is carefully sized so that the comparison is controlled rather than dominated by an oversized transformer. The model begins with a linear projection from the 35 dimensional monthly feature vector into a learned hidden space. It then adds a learned positional tensor so the month ordering is explicit, and passes the resulting sequence through one or more transformer encoder layers. The final hidden state at the last month is fed to a normalized regression head.

```mermaid
flowchart LR
    X[Input monthly window, 35 features] --> P[Linear projection, 35 to hidden size]
    P --> Q[Learned positional embedding]
    Q --> T[Transformer encoder layer, multi head self attention]
    T --> H[LayerNorm, Dropout, Linear head]
    H --> Y[Next month 35 feature forecast]
```

The role of this model is not just to provide another neural baseline. It tests whether explicit month to month pairwise interaction helps the forecasting task more than a recurrent or state space memory. Self attention is attractive because it can compare any month in the window against any other month without passing information through a single recurrent state. That flexibility is useful when the model needs to recognize a seasonal inflection point or a delayed weather effect. At the same time, the data are short and the feature set is already strongly structured, so the attention mechanism has to earn its extra capacity. That is why the report should interpret the transformer as an architectural comparison, not as an automatically superior choice.

The projection and positional design are also deliberate. A crop month is not just a vector. It is a vector placed in a calendar sequence. The learned positional tensor makes that ordering explicit, which is necessary because self attention alone is permutation agnostic. The output head is intentionally simple because the main challenge lies in extracting the temporal context, not in mapping the context to the final feature vector. In the shared comparison run, the transformer uses 172224 parameters, which makes it the largest learned backbone in the main benchmark. That size is relevant because the project is working with a relatively small county month panel, and larger capacity can improve expressiveness while also increasing the risk of fitting spurious seasonal noise.

### 3.1.3 LSTM

The LSTM backbone is the strongest classical recurrent comparator in the repository. It is implemented as a one layer LSTM followed by a normalized regression head. The design is conventional, but that is exactly why it matters. LSTM remains a strong reference model whenever the task is short sequence forecasting with smooth temporal dynamics and limited training data. The model receives the six month window, processes it recurrently, and uses the last hidden state to predict the next monthly feature vector.

```mermaid
flowchart LR
    X[Input monthly window, 35 features] --> R[LSTM recurrent core, hidden state width 96]
    R --> H[LayerNorm, Dropout, Linear head]
    H --> Y[Next month 35 feature forecast]
```

The block size is chosen for a reason. A hidden width around the mid double digits is large enough to encode shared seasonal behavior across AG, NDVI, and weather, but still small enough to train stably on a county level panel. The single layer choice matters as well. Stacking more recurrent layers would increase representational depth, but it would also make the model harder to optimize and more likely to overfit the limited number of monthly windows. Because the input sequence is only six months long, the model does not need a deep temporal stack to represent very long histories. It needs a stable mapping from a short seasonal prefix to the next month.

The LSTM gate structure is especially suitable for this problem because monthly crop data have both persistence and forgetting. A strong month should influence the immediate future, but it should not dominate every later month in the same way. The input, forget, and output gates give the model a controlled way to retain useful context while allowing old context to decay. That makes the LSTM a natural fit for short horizon monthly forecasting, particularly when the evaluation is recursive and forecast errors can accumulate. In the shared comparison run, the LSTM uses 58848 trainable parameters. That parameter count is a good compromise for the problem scale because it is large enough to model a multi modality monthly sequence, but still small enough to remain competitive against the seasonal baseline.

### 3.1.4 GRU

The GRU model is the lighter recurrent alternative to LSTM. It uses a gated recurrent unit instead of the full LSTM gating stack, but it keeps the same overall structure, namely a recurrent core followed by a normalized regression head. The GRU is useful because it tests whether a simpler hidden state update is sufficient for the monthly crop forecasting task. If it performs comparably to LSTM, then the extra gate complexity of LSTM may not be justified. If it performs worse, the result suggests that the monthly crop sequence really does benefit from the additional control that LSTM provides.

```mermaid
flowchart LR
    X[Input monthly window, 35 features] --> G[GRU recurrent core, hidden state width 96]
    G --> H[LayerNorm, Dropout, Linear head]
    H --> Y[Next month 35 feature forecast]
```

The GRU is particularly attractive in this project because it shares the same input and output contract as the LSTM while reducing the internal gating complexity. That makes the comparison more informative. If the GRU loses only slightly, then the project could prefer the simpler model for deployment. If it wins, then the data do not justify the heavier gate structure. If it loses substantially, then the crop data likely require the stronger memory control that LSTM provides. In the shared comparison run, the GRU has 46080 trainable parameters, which makes it the smallest of the main learned backbones. That lower count is useful for deployment oriented discussion because it shows that the task can be attacked with a relatively compact recurrent system, even if it is not the top performer.

### 3.1.5 Deterministic Baselines and Ensembling

The deterministic baselines are not included to fill space. They are there because crop forecasting is strongly seasonal and the learned models must be judged against rules that are already surprisingly strong. The first baseline, `naive_lag1`, simply copies the most recent observed month forward. The second baseline, `seasonal_last_year`, returns the same county and same month from the previous year, falling back to the latest available month when the seasonal lookup is missing. The third family is SARIMA, which fits a per feature seasonal time series model when enough history exists. These baselines represent three different priors, recent persistence, annual seasonality, and classical autoregression, and they give the report a robust floor for comparison.

Ensembling is treated as a post training selection layer rather than a learned model. `ensemble_mean` averages the learned model predictions, while `ensemble_weighted` uses inverse validation RMSE weights so that stronger validation models influence the final prediction more strongly. This is useful because monthly crop forecasting is noisy at the feature level, and different architectures may excel on different modalities. An ensemble can therefore stabilize the final output even when no single backbone dominates every metric. The weighted ensemble is still a deterministic combination of existing predictions, not a separately trained neural network.

The baseline and ensemble design also clarifies the project logic. The point is not simply to outperform a naive month copy rule. The point is to show that a learned model or ensemble can provide a more credible reconstruction of the missing months while preserving agricultural seasonality. The baseline family and the ensemble family make that argument measurable.

## 3.2 Train Test Split

The prepared monthly dataset used by the main forecasting run comes from `data/training`. Its source metadata records a six year crop history and a fixed year based split. Training uses 2017 to 2020, validation uses 2021, and testing uses 2022. The split is not random because random splitting would let future seasons influence the training set and would make the recursive blank fill evaluation optimistic. The task is temporal by definition, so the evaluation must also be temporal.

The sequence construction is stricter than a simple year split. Rows are sorted by `county_id`, `crop_type`, and date. A sliding window of length six is built for each county crop pair. A window is kept only when the six months are contiguous calendar months. Windows with missing or non finite feature values are discarded. In seasonal residual mode, the same county and month from the previous year must exist so the seasonal base can be computed. This means the training examples are not just row counts. They are validated temporal windows with no gaps and no hidden leakage from future months.

The year based split is also important for interpretation. Training on 2017 to 2020 gives the model multiple full growing seasons. Validation on 2021 allows the code to select the best configuration without using the final test year. Testing on 2022 gives a clean future year holdout. That is exactly the split structure one would want if the model were going to be deployed on a later season that is not yet observed. The same logic applies to the yield regression stage, which uses its own year aware split but remains separated from the monthly forecaster.

The prepared yield dataset in `data/yield_training` is a separate artifact. In the committed output, it contains 6624 rows with 4452 training rows, 1008 validation rows, and 1164 test rows. Those rows are derived from the monthly table and USDA yield labels, but they are not the same training examples used by the monthly forecasters. The yield stage uses its own split so that downstream regression can be evaluated without contaminating the monthly forecasting benchmark.

## 3.3 Training Model

The learned monthly forecasters are not plain sequence to vector regressors. They are wrapped in `PINNForecaster`, which combines a backbone, a latent head, a forecast head, and a physics module. The backbone can be LSTM, GRU, Transformer Encoder, or Gamma SSM. The latent head maps hidden states into a three state latent crop vector. The forecast head combines the final hidden state and the final latent state to produce the 35 monthly feature outputs. The physics module then adds a soft auxiliary penalty that makes the latent trajectory more consistent with crop growth, vegetation shape, and weather forcing.

The target mode can be raw or seasonal residual. In raw mode, the model predicts the 35 dimensional monthly feature vector directly. In seasonal residual mode, the model predicts a correction over the same county and same month from the prior year. That second formulation is particularly useful in agricultural data because seasonal structure is already strong and the model only needs to learn the year specific deviation. In the codebase, the shared comparison run uses raw targets, while the residual mode is used for blank fill and ablation studies.

The main training loop is deliberately conservative. The optimizer is AdamW. The scheduler is ReduceLROnPlateau. Gradients are clipped to 1.0. Early stopping is driven by validation total loss. Inputs are standardized before entering the model. Small Gaussian input noise and feature masking are applied during training so the backbone is not overly sensitive to a single feature or a single month. The model logs forecast loss and physics loss separately, then restores the best validation checkpoint after training. After prediction, outputs are inverse transformed back to the raw feature scale for evaluation.

The objective used for a learned forecast batch is the sum of a supervised forecast term and an optional physics term. In the code, the base form is

`L_total = L_forecast + lambda_phys * L_phys`.

The forecast loss is mean squared error on the scaled target values, optionally weighted by per feature importance weights when the configuration requests it. The physics term is computed from the raw input sequence and the latent crop states. This distinction matters because the model is trained to fit the observed monthly data, but it is also gently constrained to respect domain logic. The physics term is therefore a regularizer, not a replacement for the supervised objective.

### 3.3.1 Monthly Feature Forecasting Model

The monthly forecasting model predicts a full 35 feature vector for each county month window. In the main comparison run, the model family is trained with the same sequence length, the same feature set, and the same physics configuration so that the architecture comparison is controlled. This controlled setup is essential because otherwise the comparison would mix architectural gains with changes in target formulation or feature availability.

The grid search process is JSON driven. The sweep runner expands a Cartesian product of hyperparameters, launches one job per model family, and selects the winning run by validation metric. The sweep configuration in the repository explores hidden size, learning rate, weight decay, physics weight, and physics warmup epochs. The best preset files stored in `training/presets` capture the retained configurations for LSTM, GRU, Transformer Encoder, and Mamba style runs. The key point is that the model selection process is not ad hoc. It is encoded in the training workflow and tied to validation performance rather than test performance.

The forecasting target mode changes the training problem in a meaningful way. Raw mode asks the network to reproduce the target month directly. Seasonal residual mode gives the model a seasonal anchor and asks it to predict the correction relative to that anchor. In a crop setting, residual learning is attractive because the previous year same month baseline is often already close to the answer. The residual formulation can therefore make the learning problem easier, especially when the recursive blank fill horizon becomes long.

The main learned model settings are summarized in Table 3.1. The table is not merely descriptive. It is part of the justification for why the comparison is fair. Every backbone sees the same 35 feature contract, the same six month sequence length, and the same high level training recipe. The differences in performance are therefore attributable to architecture and target choice rather than to accidental changes in the data interface.

| Model | Hidden size | Layers | Trainable parameters | Explanation |
|---|---:|---:|---:|---|
| LSTM | 96 | 1 | 58848 | The recurrent gate structure is a strong baseline for seasonal monthly forecasting and often preserves long context better than simpler cells. |
| GRU | 96 | 1 | 46080 | The lighter recurrent structure tests whether the task needs the full LSTM gating stack or whether a smaller hidden update is enough. |
| Transformer Encoder | 96 | 1 | 172224 | Self attention tests whether explicit month to month interaction improves the reconstruction of seasonal structure. |
| Gamma SSM, TinyMamba | 96 shared run, 128 in best sweep preset | 1 | 58432 in the shared run | The custom state space model tests whether compact state evolution can capture monthly crop dynamics with fewer recurrent style assumptions. |
| naive lag1 | n.a. | n.a. | 0 | A persistence baseline that simply copies the latest observed month forward. |
| seasonal last year | n.a. | n.a. | 0 | A strong agricultural prior that reuses the same county and same month from the prior year. |

The table shows why the comparison is informative. The transformer has the largest parameter count in the shared run, but more parameters do not automatically guarantee better agricultural blank fill. The Gamma SSM variant is much more compact, yet it still needs to be evaluated against the recurrent models because state space memory and recurrent memory behave differently on strongly seasonal panels. The best model is therefore not the one with the largest capacity. It is the one whose inductive bias matches the crop data.

### 3.3.1.1 Hyperparameter Tuning and Grid Search

The grid search logic is designed to support controlled model comparison rather than exhaustive architecture search. The sweep runner reads a JSON configuration, expands a hyperparameter grid, and launches a training job for each combination. The search space includes hidden size, learning rate, weight decay, physics weight, and physics warmup epochs. Validation R2 is used as the main objective in the committed sweep presets, which keeps the search focused on a metric that is sensitive to explained variance rather than only raw scale error.

The retained presets show the practical tuning outcome. The shared LSTM, GRU, and Transformer runs use a hidden size of 96, dropout of 0.05, physics weight of 0.03, and warmup of 10 epochs. The best Mamba style preset uses a larger hidden size of 128 and a smaller physics weight of 0.01, which reflects the model specific balance between expressive state width and auxiliary regularization. These settings are not arbitrary. They were selected because the monthly dataset is not large enough to support aggressively deep or aggressively wide models without overfitting.

The sweep is also constrained by the fact that the output must still be comparable across models. The data split, the monthly feature contract, the sequence length, and the physics configuration are held fixed while the architecture and a small number of optimizer settings vary. This is the correct experimental design for a report like this. If too many knobs change at once, then the final comparison becomes impossible to interpret.

### 3.3.1.2 Physics Loss Implementation

The physics loss is implemented in `CropPhysicsModule`, and it is best understood as a soft crop process prior rather than a hard physical simulator. The module does not solve a full crop growth differential equation. Instead, it constrains the latent states and the raw monthly features to move in ways that are biologically and meteorologically plausible. That design choice is appropriate because the data are monthly, the labels are noisy, and the goal is regularization rather than exact mechanistic simulation. The physics term helps the model avoid pathological fits on a relatively small training set, which is especially important because the learned forecasters can overfit easily if they are left to memorize the seasonal table.

The module uses three latent channels. Channel 0 represents biomass, channel 1 represents phenology, and channel 2 represents water condition. If the latent state dimension is larger than 3, the extra dimensions are penalized with a quadratic term. The model then compares those latent channels with modality proxies extracted from the raw input sequence. For AG, the latent biomass proxy is compared with green pixel ratio and vegetation area percent. For NDVI, the latent phenology proxy is compared with mean, median, and threshold style vegetation summaries. For weather, the latent water proxy is compared with precipitation, drought, and vapor pressure deficit related summaries.

The code implements several loss components. The latent consistency term encourages the latent crop states to match the modality proxies. The latent dynamics term penalizes month to month differences that violate a simple growth, senescence, and water balance rule. The AG term encourages canopy closure and suppresses impossible canopy and bare ground combinations. The NDVI term encourages a double logistic seasonal profile, preserves percentile ordering, and keeps threshold ratios within valid bounds. The weather term checks whether derived quantities such as growing degree days, vapor pressure deficit, heat stress, cold stress, and precipitation counts are internally consistent.

The dynamics are written in the code as simple difference equations over the latent state sequence. If `b_t`, `p_t`, and `w_t` denote biomass, phenology, and water at month `t`, then the latent transitions are regularized toward forms such as

`b_{t+1} - b_t ≈ dt * (growth_rate * forcing_t * b_t * (1 - b_t) + coupling * w_t * (1 - b_t) - senescence_rate * drought_t * b_t)`,

`p_{t+1} - p_t ≈ dt * (phenology_gain * b_t * (1 - p_t) * forcing_t - phenology_senescence * drought_t * p_t)`,

and

`w_{t+1} - w_t ≈ dt * (water_gain * relu(gdd_t + solar_t) - water_loss * drought_t - coupling * b_t * w_t)`.

These expressions are not meant to be exact agronomic laws. They are controlled monotonicity and smoothness priors that encode what a physically reasonable monthly crop trajectory should look like. The AG loss similarly encourages a canopy growth curve with limited complementarity violations between canopy and bare ground. The NDVI loss encourages a double logistic shaped seasonal curve because vegetation in crop systems typically rises during the growing season and declines later in the year. The weather loss encourages derived metrics such as GDD, VPD, and stress day counts to remain coherent with the observed monthly temperatures, humidity, precipitation, and solar radiation.

The weights in `training/physics_weights.json` are chosen to keep the physics term useful without overwhelming the data term. The top level group weights are 0.1 for the latent term, 0.3 for AG, 0.4 for NDVI, and 0.3 for weather. The latent warmup is 5 epochs, while the training loop level warmup is 10 epochs for the combined physics weight. This two stage delay is useful because the network should first learn the broad empirical mapping from monthly inputs to monthly outputs before the regularizer begins enforcing crop logic. If the physics term comes too early, it can prevent the model from fitting the basic seasonal pattern. If it comes too late, it cannot meaningfully steer the representation. The chosen warmup and weight values are therefore a compromise between data fit and inductive bias.

The reason to use PINN style training in this project is not that crop growth can be fully expressed as a closed form equation. The reason is that the monthly data are limited, the feature space is structured, and the learned models can overfit the seasonal idiosyncrasies of a few counties very quickly. A physics informed term acts as a structural prior. It encourages the model to respect crop growth logic even when the supervised signal is noisy or sparse. In that sense, the physics loss is an anti overfitting device as much as it is a scientific prior.

### 3.3.1.3 Model Comparison and Selection

The shared comparison run shows a clear hierarchy. The ensemble mean is the best overall forecasting model on the test set, while the Transformer Encoder is the strongest single learned backbone in the shared raw target run. The GRU has the best validation R2 among the learned models in that particular configuration, which shows that validation and test rankings are not identical. The Gamma SSM family remains competitive, but it is slightly behind the transformer and the recurrent models in the shared 96 width run. These results matter because they show that the seasonal crop sequence is not solved by a single architecture family. The best choice depends on whether the goal is raw accuracy, validation stability, or deployment simplicity.

| Model | Params | Train loss | Val R2 | Test RMSE | Test MAE | Test R2 |
|---|---:|---:|---:|---:|---:|---:|
| `ensemble_mean` | 0 | NaN | 0.9174 | 2.4998 | 0.3912 | 0.9191 |
| `ensemble_weighted` | 0 | NaN | 0.9174 | 2.5002 | 0.3913 | 0.9191 |
| `transformer_encoder` | 172224 | 0.5405 | 0.9074 | 2.5555 | 0.4072 | 0.9155 |
| `lstm` | 58848 | 0.5402 | 0.9138 | 2.6014 | 0.4366 | 0.9124 |
| `gru` | 46080 | 0.5397 | 0.9143 | 2.6130 | 0.4374 | 0.9116 |
| `tiny_mamba_ssm`, `gamma_ssm` | 58432 | 0.5351 | 0.9138 | 2.6471 | 0.4456 | 0.9093 |
| `naive_lag1` | 0 | NaN | 0.7278 | 4.4642 | 0.6240 | 0.7421 |
| `seasonal_last_year` | 0 | NaN | -0.0285 | 8.9200 | 1.5209 | -0.0298 |

The table shows that the learned models are strong, but the seasonal baseline is still a serious reference point. The ensemble mean is the best overall because it reduces the variance of individual model errors. The transformer is the best single backbone because its self attention mechanism captures cross month interactions well. The LSTM and GRU are close behind, which suggests that the monthly sequence is regular enough that a conventional recurrent memory is already very effective. The Gamma SSM model is compact and competitive, which is encouraging from a deployment perspective, but it does not dominate the other learned models in the shared run.

### 3.3.2 Yield Prediction Model

The yield prediction stack is separate from the monthly forecasting stack. It does not use the PINN forecasters. Instead, it trains conventional regressors on monthly CropNet features joined to USDA yield labels. This distinction is important because the yield model solves a different problem. The monthly forecaster reconstructs missing feature months, while the yield regressor maps the completed monthly table to an annual county yield value.

The committed yield run uses the monthly table with timing features. The monthly inputs are the canonical 35 CropNet features, and the yield model also uses month sine and month cosine encodings to make seasonality explicit. The target is `yield_bu_acre`. The label join is performed by county and year, so each monthly row inherits the county year USDA yield label. The yield pipeline can also construct annualized growing season summaries from April through September, but the main committed `yield_all` artifact uses the monthly table path.

Before model fitting, the yield code prunes low information columns. Features with a missing fraction above 0.20 are dropped. Features with no variance are dropped. Highly correlated pairs are reduced when the absolute correlation reaches 0.995 or above. This pruning matters because monthly yield regression is sensitive to redundant features, and the downstream model should not be forced to learn from nearly duplicate signals. After pruning, the remaining features are passed through median imputation and standard scaling before model fitting.

The candidate regressors are Ridge, RandomForest, and ExtraTrees. Ridge is the simplest linear baseline and remains highly interpretable. RandomForest and ExtraTrees provide nonlinear comparisons and test whether the yield target benefits from tree based interactions. The hyperparameter grids search over ridge alpha, tree depth, and minimum leaf size, and the best candidate is selected by validation RMSE before the final refit on train and validation data.

The yield results show a different pattern from the monthly forecasting results. In the committed yield artifact, Ridge is the best trainable model, but the strongest test set model overall is the previous year same county baseline. This is not a contradiction. It reflects the fact that yield prediction is a harder and more compressed target than monthly feature forecasting. The seasonal persistence prior is very strong at the yield level, and a simple regression model can still struggle to beat it unless the feature representation is especially informative.

| Model | Type | Val RMSE | Val R2 | Test RMSE | Test R2 | Params |
|---|---|---|---|---|---|---:|
| `BaselineTrainMean` | baseline | 20.1297 | -0.4351 | 21.6732 | -0.0230 | 0 |
| `Ridge` | trainable | 20.2002 | -0.4452 | 20.4359 | 0.0905 | 19 |
| `ExtraTrees` | trainable | 20.8653 | -0.5419 | 19.6136 | 0.1622 | 0 |
| `RandomForest` | trainable | 21.4633 | -0.6315 | 19.5342 | 0.1690 | 0 |
| `BaselinePreviousYearSameCounty` | baseline | 32.9192 | -2.8380 | 18.3540 | 0.2663 | 0 |

The interpretation is straightforward. Ridge is the best trainable yield model in the saved yield report, but the previous year same county baseline still wins on the test set. The practical lesson is that yield estimation and monthly blank fill should not be conflated. A model can be strong at reconstructing monthly features and still face a difficult annual yield regression problem. The repository therefore keeps the two stages separate and treats the yield model as a downstream consumer of the monthly table rather than as a replacement for the forecasting task.

## 3.4 Evaluation of the AI Model

The evaluation layer is intentionally broader than a single scalar score. The repository saves evaluation artifacts at both the monthly forecasting stage and the yield regression stage. That separation is important because the two tasks are related but not identical. The monthly models are judged on raw feature reconstruction. The yield models are judged on annual yield prediction. Both tasks need their own metrics, their own baselines, and their own interpretation.

The monthly forecasting models are evaluated on raw scale feature predictions after inverse scaling. The core metrics are RMSE, MAE, MSE, and R2. RMSE is the most important single score because it penalizes large reconstruction errors, which is especially relevant when the model is used recursively. MAE provides a more stable average error view. MSE is retained for completeness. R2 checks how much variance in the true monthly features is explained by the prediction. The shared comparison run also records train loss, validation loss, physics loss, and the loss curves so the report can show whether the physics term is behaving as intended.

The physics loss curves matter because they show that the auxiliary term remains bounded and acts as a regularizer rather than overwhelming the forecast objective. In the learned runs, the physics penalty settles around 15.55 while the forecast metrics continue to improve. That is exactly the behavior one wants from a soft physics prior. It should shape the representation without replacing the data fit objective.

The yield regression report records RMSE, MAE, MSE, R2, and MAPE. It also stores feature importance, residuals, month benchmark tables, window benchmark tables, feature group benchmark tables, and year wise cross validation summaries. These diagnostics are important because the yield problem is sensitive to time aggregation, feature group composition, and county specific differences. The model may appear strong on one year window and weak on another, so the additional artifacts are needed for a complete interpretation.

The final evaluation summary is shown below.

| Stage | Best evaluated model | Test metric takeaway |
|---|---|---|
| Monthly feature forecasting | `ensemble_mean` | RMSE 2.4998, R2 0.9191 |
| Monthly single model leader | `transformer_encoder` | RMSE 2.5555, R2 0.9155 |
| Yield regression | `BaselinePreviousYearSameCounty` | RMSE 18.3540, R2 0.2663 |
| Best trainable yield model | `Ridge` | RMSE 20.4359, R2 0.0905 |

This final table makes the project structure clear. The monthly forecasting stage is the primary modeling contribution and reaches strong reconstruction quality. The yield stage is deliberately conservative and remains harder than the monthly task. The two stages are connected, but they should not be judged by the same metric or the same ranking logic. A strong monthly forecaster is a necessary upstream component, but it does not automatically produce a dominant yield regressor.

# 4. AI Demonstrator

The AI demonstrator is the integration layer that turns the CropNet forecasting workflow into a browser-based TaoCrop dashboard. It does not train models live in the browser. Instead, it loads saved artifacts, preprocesses a user-supplied farm folder into monthly features, runs the forecasting model to extend those features into future months, and then applies the saved yield model to produce a farmer-facing yield estimate. The result is presented as cards, charts, feature panels, and an optional chat assistant rather than as raw CSV outputs.

The demonstrator is built as a local web application. `main.py` adds `src/` to the import path and launches `crop_fusion_ai.web.app:main`, which in turn creates a FastAPI app in `src/crop_fusion_ai/web/app.py`. The frontend is a static HTML/CSS/JavaScript shell in `src/crop_fusion_ai/web/static/`, while the reusable data and model logic lives in `src/crop_fusion_ai/gui/controller.py`, `src/crop_fusion_ai/gui/forecasting.py`, `src/crop_fusion_ai/web/service.py`, and `src/cropnet_forecasting/`. This separation keeps the browser UI thin and makes the preprocessing and forecasting code testable outside the dashboard.

## 4.1 Purpose and system design

The demonstrator exists to show that the CropNet workflow can be used end to end outside the training scripts. The codebase is organized around a simple chain:

| Layer | Code path | Role |
| --- | --- | --- |
| Launcher | `main.py` | Starts the local browser app on `http://127.0.0.1:8000` by default. |
| Web server | `src/crop_fusion_ai/web/app.py` | Exposes the API, stages uploads, manages jobs, and serves the static dashboard. |
| Modality preprocessing | `src/crop_fusion_ai/gui/controller.py` | Caches AG, NDVI, and weather extraction per file and metadata tuple. |
| Directory forecasting | `src/crop_fusion_ai/gui/forecasting.py` | Scans folder trees, builds monthly features, and rolls the learned feature forecasters forward. |
| Yield inference | `src/crop_fusion_ai/web/service.py` | Loads the saved monthly yield model, scores monthly rows, and builds the payload consumed by the UI. |
| Browser UI | `src/crop_fusion_ai/web/static/index.html`, `app.js`, `app.css` | Renders the upload form, summary cards, tabs, charts, tables, and chat panel. |

This architecture mirrors the report’s evaluation logic. The monthly feature table is constructed first, the forecasting model produces future monthly feature vectors second, and the yield model converts those features into a yield estimate last. The UI then exposes the intermediate artifacts so the user can inspect not just the final number, but also the signals that drove it.

The dashboard is intentionally local. The upload processing runs on the same machine, and the chat assistant connects to Ollama on `localhost:11434` rather than to a hosted API. That design keeps the demonstrator simple to run in a lab or classroom setting and avoids introducing an external dependency into the user-facing flow.

## 4.2 Required input

The browser UI asks for two things: a crop type and a folder of farm files. The crop selector in `src/crop_fusion_ai/web/static/index.html` includes corn, cotton, soybeans, and winter wheat. The folder picker uses the browser’s directory upload support so the user can submit a whole farm directory in one step.

The current browser form does not expose `county_id`, but the backend still accepts it and defaults to `19001` in `src/crop_fusion_ai/web/app.py`. That means the visible user input is just the crop and the folder, while county metadata is handled by the service layer.

The folder structure is expected to separate the three modalities into subdirectories such as `ag/`, `ndvi/`, and `weather/`. The README shows the intended shape:

```text
sample_data/
  ag/
    2017_12_21.png
  ndvi/
    2017_12_21.png
  weather/
    2017_12.csv
```

The directory scanner in `src/crop_fusion_ai/gui/forecasting.py` walks the tree recursively, so nested folders are accepted as long as the modality names appear in the path. The supported file types are:

| Modality | Accepted files | Notes |
| --- | --- | --- |
| AG | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff` | The file name or path should contain year and month tokens so the loader can infer the timestamp. |
| NDVI | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff` | The timestamp is inferred in the same way as AG. |
| Weather | `.csv`, `.tsv`, `.parquet`, `.feather` | The weather extractor expects date information or year/month/day columns. |

The browser upload path preserves the relative file paths by sending `files` plus `relative_paths` in a multipart request to `/api/predict/upload`. The backend validates that the file count and relative-path count match, rejects invalid paths, and stages the upload in a content-addressed cache directory under `data/cache/web_uploads/`.

## 4.3 What the system does with the input

Once the folder arrives, the backend creates a background job and moves the work off the request thread. The job state is tracked by `JobStore` in `src/crop_fusion_ai/web/app.py`, which exposes `queued`, `running`, `completed`, and `failed` states through `/api/jobs/{job_id}`. The frontend polls that endpoint every few hundred milliseconds until the result is ready.

The monthly feature construction happens in `src/crop_fusion_ai/gui/forecasting.py`:

1. `scan_directory()` finds supported AG, NDVI, and weather files.
2. The year and month are inferred from the file name or path tokens when possible.
3. `PreprocessingController` routes each file to `extract_ag_features()`, `extract_ndvi_features()`, or `extract_weather_features()`.
4. `FeatureCache` stores per-file preprocessing outputs in `data/cache/gui_features/` so repeated runs can reuse them.
5. The modality frames are aggregated with `aggregate_monthly_feature_frame()` and merged with `combine_modality_feature_frames()`.
6. The monthly table is normalized, sorted by `county_id`, `crop_type`, `year`, and `month`, and converted into a form that the forecasting model can consume.

The feature extractors themselves are modality-specific. AG imagery is converted into HSV-based masks and field-structure statistics, NDVI scenes are reduced to vegetation statistics and threshold ratios, and weather records are aggregated from daily observations into monthly summaries. The code is designed to tolerate slightly different source table schemas, especially in the weather path, where multiple column aliases are accepted.

After the observed monthly table is built, the forecasting layer in `src/crop_fusion_ai/gui/forecasting.py` loads four saved feature forecasters with `BlankFillPredictor.from_artifacts()`:

| Feature model | Checkpoint source | Role in the demonstrator |
| --- | --- | --- |
| LSTM | `training/runs/lstm_best/lstm/checkpoint.pt` by default | Baseline learned forecaster; also the default checkpoint path. |
| GRU | `training/runs/gru_best/gru/checkpoint.pt` | Learned comparison model. |
| Transformer Encoder | `training/runs/transformer_best/transformer_encoder/checkpoint.pt` | Learned comparison model. |
| Tiny Mamba SSM | `training/runs/mamba_best/tiny_mamba_ssm/checkpoint.pt` | Learned comparison model that can expose latent-state outputs when the checkpoint supports them. |

Each forecaster consumes the same prepared monthly feature frame, predicts future monthly feature vectors for a fixed horizon, and may also emit latent-state channels. The predictor is sequence-based: it uses a fixed `seq_len` window, scales the window with the saved scaler, runs the loaded network, inverts the scaling, and clips values where the feature semantics require it. In other words, the demonstrator does not invent a separate GUI-specific model. It reuses the same forecasting stack that was trained for monthly blank filling.

The yield layer is separate from the forecasting layer. `src/crop_fusion_ai/web/service.py` loads the saved monthly yield regression model from `outputs/yield_baseline/corn_ia_2017_2022_monthly/best_yield_model.joblib`, together with its metadata, feature importance CSV, and benchmark CSV. The service prepares the monthly feature frame for prediction, adds cyclical month encodings when needed, aligns the feature columns to the model’s expected input order, and then calls `predict()` on the saved pipeline. Additional yield-model variants are loaded from `outputs/predicted_yield_experiments/yield_models/` when present so the dashboard can compare the current model against alternative downstream yield regressors.

This division is important. The current yield estimate comes from the saved monthly yield model, but the future-yield trajectory comes from the forecasted monthly features produced by the feature forecasters. The browser dashboard can therefore show both the yield implied by the observed monthly table and the yield implied by the rolled-forward feature table.

## 4.4 Outputs produced by the demonstrator

The backend response from `/api/predict/upload` is a structured payload that combines the current yield estimate, the forecasted feature tables, the downstream yield trajectories, and the supporting diagnostics. The most important outputs are:

| Output group | Payload fields | What the user sees |
| --- | --- | --- |
| Top-line estimate | `headline`, `forecast_headline` | The large yield card at the top of the page. When the rolled-forward forecast is available, the dashboard prefers `forecast_headline`. |
| Yield history | `yield_series`, `yield_series_by_model` | The yield-over-time chart in the Overview tab. |
| Forecast trajectories | `yield_trajectory_by_model`, `yield_models` | The Monthly yield trajectory chart and the comparison lines for alternative yield models. |
| Model quality | `summary`, `benchmark_summary`, `feature_model_runs`, `feature_model_best` | The model-quality card and the Benchmark summary table. |
| Drivers | `drivers`, `feature_importance`, `feature_groups` | The Top driver card, the Drivers tab, and the feature cards grouped by canopy, vegetation, and weather. |
| Forecasted features | `feature_forecasts_by_model` | The feature sparkline panels that compare the learned forecasters. |
| Latent states | `derived_drivers_by_model`, `derived_driver_models` | The derived-driver cards for biomass, phenology, and water stress. |
| Support data | `monthly_features`, `source_files` | The prepared monthly feature preview and the list of files that were ingested. |

The top cards on the dashboard are not hard-coded summaries. They are assembled from the model outputs. The `YieldModelService` computes the current estimate on the prepared monthly features first, then `predict_from_directory()` computes a future feature frame, applies the yield model to that forecasted feature frame, and stores the last future row as `forecast_headline`. The frontend uses that forecast headline when it is available, so the user sees the most forward-looking estimate rather than only the last observed-month estimate.

The dashboard also shows a model-quality card. In the current implementation, the card lists the strongest feature-forecasting runs available under `training/runs/` and displays metrics such as test R2 and RMSE. If the run summary files are missing, the UI falls back to a message explaining that no trained feature runs were found. This keeps the display truthful to the artifacts actually present on disk.

The `Details` tab exposes two additional views. The benchmark summary table lists the current yield model name, feature count, target units, best holdout model, and holdout metrics. The prepared monthly feature table shows a short preview of the latest monthly rows, including predicted yield, green canopy, vegetation vigor, rainfall, and growing degree days. That table is intentionally compact because its purpose is inspection, not raw export.

## 4.5 What the user sees in TaoCrop

The browser UI in `src/crop_fusion_ai/web/static/index.html` is designed as a left-hand sidebar plus a main content area. Before any prediction run, the page shows the TaoCrop brand, a short “How it works” card, the upload form, and a status pill that reads `Ready`. The main hero text explains that the user should upload farm data, review the estimated yield, and inspect the crop and weather signals behind the result.

After a run completes, the page transitions into a result state:

1. A headline yield card appears with the latest yield estimate and unit.
2. A small card shows the top driver.
3. Another card reports data coverage as the number of monthly rows prepared for the model.
4. A model-quality card summarizes the strongest training runs found on disk.
5. The tab strip becomes active with `Overview`, `Yield result`, `Drivers`, and `Details`.

The `Overview` tab combines a yield-over-time chart with a short natural-language summary. The summary tells the user which month was analyzed, how many monthly rows were used, what the top driver was, and how the estimate should be interpreted. The `Yield result` tab focuses on the monthly trajectory chart and legend, which compare the current model against alternative yield-model trajectories where available.

The `Drivers` tab is the most explanatory part of the interface. It begins with feature-group tabs, then shows the primary driver panel with a bar chart of feature importance. When latent outputs are available, a derived-driver panel appears with three unitless channels: biomass state, phenology state, and water stress state. For each visible feature group, the UI draws sparklines that compare the observed monthly signals against the feature forecasters from LSTM, GRU, Transformer Encoder, and Tiny Mamba SSM. Season features are still part of the model contract, but they are not exposed as a separate tab in the current browser layout.

The `Details` tab is the audit view. It shows the benchmark summary table and the prepared monthly features preview. This is the place to check whether the monthly table was assembled correctly and whether the loaded yield model matches the expected artifact set.

The TaoCrop assistant is launched from a floating chat button. When opened, the panel lists locally available Ollama models, displays the current dashboard snapshot, and sends that snapshot plus the conversation turns to `ChatOllama`. The chat system prompt instructs the assistant to stay grounded in the current dashboard state and to say when a requested detail is missing. This means the assistant is an explanation layer, not a separate predictor.

## 4.6 End-to-end interaction flow

The full user flow is:

1. Open the local TaoCrop server.
2. Choose a crop type.
3. Upload a farm folder containing AG, NDVI, and weather inputs.
4. Wait while the backend stages the files, preprocesses them, and runs the background job.
5. Review the headline yield estimate, feature drivers, model quality, and monthly tables.
6. Use the Drivers, Yield result, and Details tabs to inspect the supporting evidence.
7. Open the TaoCrop assistant if a plain-language explanation is needed.

This flow is important because it keeps the demonstrator honest about where the answer comes from. The dashboard does not hide the model chain behind a single opaque number. It shows the monthly inputs, the forecasted monthly features, the yield trajectory, and the stored benchmarks in one place, which makes the system easier to validate and explain.

## 4.7 Suggested figures

If screenshots are unavailable, the following captions fit the implemented UI:

1. `Figure 4.1`: TaoCrop landing page with the crop selector, folder upload control, and status pill before a prediction run.
2. `Figure 4.2`: Completed TaoCrop prediction showing the headline yield card, top driver, data coverage, and model-quality summary.
3. `Figure 4.3`: Drivers tab with feature-group tabs, feature-importance bar chart, and derived-driver latent state panels.
4. `Figure 4.4`: Details tab showing the benchmark summary and the prepared monthly feature preview table.
5. `Figure 4.5`: TaoCrop assistant chat panel with the Ollama model selector and dashboard snapshot context.

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

# 6. References

- `docs/PROJECT_QUICK_BRIEF.md`
- `docs/PROJECT_UNDERSTANDING_GUIDE.md`
- `reports/README_model_comparison.md`
- `reports/README_feature_ablation_v1.md`
- `reports/README_RESULTS_SUMMARY.md`
- `src/cropnet_forecasting/models.py`
- `src/cropnet_forecasting/pinn.py`
- `src/cropnet_forecasting/training_engine.py`
- `src/cropnet_forecasting/yield_training.py`
- `src/cropnet_forecasting/yield_regression.py`
- `src/crop_fusion_ai/web/app.py`
- `src/crop_fusion_ai/web/service.py`
- `src/crop_fusion_ai/gui/controller.py`
- `src/crop_fusion_ai/gui/forecasting.py`
- `training/sweep.py`

# 7. Appendix A

- Source code: local repository root at `/mnt/c/users/lobakkang/github/crop-net`
- Intermediate data: `data/training/`, `data/yield_training/`, and `outputs/`
- Final model artifacts: `training/runs/`, `training/yield_runs/`, and `weights/`
- Demonstrator run guide: `README.md`
- Sample data and manifest: `data/sample_data/manifest.json`
