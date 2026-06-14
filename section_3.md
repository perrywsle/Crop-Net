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
