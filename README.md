# Multi-Modal Crop Health and Yield Prediction System

This repository contains the scaffold for a university AI engineering project
that will combine:

- A plant health image classifier.
- A CropNet-style crop yield prediction model.
- A late-fusion simulation pipeline using image-derived `health_score` and
  `image_confidence` features.
- A Tkinter desktop demo UI.

Training and inference implementations are intentionally not included yet. This
initial version focuses on clean project structure, configuration, schemas, and
tooling.

## Project Layout

```text
src/crop_fusion_ai/     Python package
tests/                  Automated tests
data/raw/               Original datasets
data/processed/         Prepared datasets and derived features
models/                 Saved model artifacts
reports/                Metrics, figures, and experiment notes
```

## Development

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run checks:

```bash
ruff check .
mypy src
pytest
```
