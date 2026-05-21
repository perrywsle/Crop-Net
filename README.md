# Multi-Modal Crop Health and Yield Prediction System

Python demo project for a university AI engineering workflow that combines:

- Plant health image inference.
- CropNet-style tabular yield prediction.
- Late fusion where image-derived `health_score` and `image_confidence` become
  extra yield-model features.
- A multistage demo path where MobileNet image embeddings and previous weather
  summaries feed a supervised yield regressor trained against CropNet USDA yield
  labels.
- A simple Tkinter desktop UI.

The project is intentionally runnable without the real 2TB+ CropNet dataset.
Real CropNet access is isolated behind an optional API wrapper.

## Project Layout

```text
src/crop_fusion_ai/config/       Pydantic schemas and app settings
src/crop_fusion_ai/data/         Fusion feature helpers
src/crop_fusion_ai/data_sources/ Optional CropNet API adapter
src/crop_fusion_ai/models/       Image and yield model wrappers
src/crop_fusion_ai/training/     Training and evaluation scripts
src/crop_fusion_ai/inference/    CLI inference and fusion pipeline
src/crop_fusion_ai/ui/           Tkinter desktop demo
tests/                           Automated tests
data/                            Local datasets and processed features
models/                          Saved model artifacts
reports/                         Metrics, figures, and saved UI results
```

## Setup

Use Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run quality checks:

```bash
ruff check .
ruff format .
mypy src
pytest
```

## Demo Mode

Demo mode does not require CropNet or any real downloaded dataset.

Train a small synthetic yield model:

```bash
python -m crop_fusion_ai.training.train_yield_model
```

This creates:

```text
models/yield_model/yield_regressor.joblib
reports/metrics/yield_metrics.json
```

Run yield prediction from command line:

```bash
python -m crop_fusion_ai.inference.predict_yield \
  --crop-type corn \
  --region 01003 \
  --year 2022 \
  --temperature-mean 25 \
  --rainfall-total 140 \
  --health-score 0.85 \
  --image-confidence 0.70
```

Run plant health placeholder inference:

```bash
python -m crop_fusion_ai.inference.predict_health --image path/to/image.jpg
```

Train the lightweight segmentation PoC on local crop images and run one
segmentation inference:

```bash
python -m crop_fusion_ai.training.train_segmentation_model \
  --image-root data/processed/cropnet_images \
  --epochs 2 \
  --batch-size 2 \
  --max-samples 16 \
  --device cuda

python -m crop_fusion_ai.inference.predict_segmentation \
  --image path/to/image.png \
  --model models/image_model/segmentation_model.pt \
  --device cuda
```

The Tkinter UI automatically loads `models/image_model/segmentation_model.pt`
when present and displays the segmentation overlay after prediction.

Train a synthetic multistage model shaped like MobileNet + weather + USDA yield:

```bash
python -m crop_fusion_ai.training.train_multistage_demo
```

Build a bounded real-data-style multistage dataset for the feasibility path
using prior-year corn NDVI/weather features and next-year USDA yield labels:

```bash
python -m crop_fusion_ai.training.build_multistage_cropnet_dataset \
  --crop-type corn \
  --image-type NDVI \
  --input-year 2021 \
  --target-year 2022 \
  --fips 01003 01005 \
  --weather-csv data/processed/weather_manifest.csv \
  --yield-csv data/processed/usda_yield.csv \
  --image-manifest data/processed/ndvi_images.csv \
  --download \
  --pretrained
```

The builder keeps downloads bounded by requesting only input-year HRRR/NDVI and
target-year USDA data. It writes compact features to
`data/processed/multistage_cropnet_features.csv` and deletes the raw CropNet
cache by default after feature extraction. Pass `--keep-raw-cache` if you want
to inspect the downloaded files.

Train the real-data multistage model from the compact CSV:

```bash
python -m crop_fusion_ai.training.train_multistage_real \
  --csv data/processed/multistage_cropnet_features.csv
```

Launch the desktop UI:

```bash
python -m crop_fusion_ai.ui.tkinter_app
```

The UI uses the saved yield model at:

```text
models/yield_model/yield_regressor.joblib
```

Generate evaluation plots from a CSV if available, or synthetic data if the CSV
is missing:

```bash
python -m crop_fusion_ai.training.evaluate_models \
  --csv data/processed/cropnet_features.csv \
  --target yield
```

This writes:

```text
reports/metrics/yield_metrics.json
reports/figures/predicted_vs_actual.png
reports/figures/error_distribution.png
```

## Folder-Based Image Dataset

The image training skeleton expects this layout:

```text
data/processed/images/
  healthy/
  mild_disease/
  severe_disease/
```

Run the skeleton:

```bash
python -m crop_fusion_ai.training.train_image_model --dataset data/processed/images
```

Current image inference is a deterministic placeholder that validates the image
and returns a safe `ImagePrediction`. It is designed to be replaced later by a
MobileNet/EfficientNet-style model without changing the UI or fusion pipeline.

## Real Dataset Mode

The real CropNet dataset is hosted at:

```text
https://huggingface.co/datasets/CropNet/CropNet
```

It is too large to download fully for this project. The code therefore uses a
bounded API-adapter design:

- `CropNetQuery` selects crop type, FIPS counties, years, and modalities.
- `CropNetClient` lazily imports the optional official `cropnet` package.
- Normal tests and demo mode do not require the real CropNet package or network.
- USDA Crop data provides the ground-truth crop yield or production target.
- Sentinel-2 AG/NDVI imagery can be passed through MobileNet to create crop
  condition embeddings.
- WRF-HRRR daily/monthly records provide previous weather time-series inputs.

Install optional CropNet dependencies only when needed:

```bash
python -m pip install ".[cropnet]"
```

Install optional MobileNet dependencies only when needed:

```bash
python -m pip install ".[vision]"
```

Smoke-test MobileNet feature extraction on one downloaded Sentinel-2 AG/NDVI
image:

```bash
python -m crop_fusion_ai.inference.extract_mobilenet_features \
  --image path/to/sentinel2_image.png
```

Then use `CropNetClient` for small, selected requests rather than full-dataset
downloads. Start with one crop, one year, and one or two FIPS counties. The
official CropNet API controls exact downloaded file sizes, so the project uses
bounded queries instead of assuming a fixed megabyte cap.

## Known Limitations

- The plant health image model is a placeholder, not a trained disease model.
- The yield model is a RandomForest baseline trained on CSV or synthetic demo
  data.
- The multistage MobileNet/weather path is implemented as a demo-ready skeleton;
  real CropNet parsing and GPU training are still future work.
- CropNet API integration is an adapter boundary; full real-data parsing and
  aggregation are not implemented yet.
- The Tkinter UI requires a saved yield model before running predictions.
- Evaluation plots are for tabular yield predictions only; image model
  evaluation is currently a skeleton.
