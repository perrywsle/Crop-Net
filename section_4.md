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
