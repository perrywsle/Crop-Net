# COS40007 CropNet Blank-Fill Forecasting

This repository stages the active COS40007 CropNet research workflow for monthly
AG, NDVI, and weather forecasting, with a focus on recursive blank-fill
forecasting for missing future months within a target year.

## Objective

The project models a 35-feature monthly CropNet representation and studies how
to forecast the remaining months of a year from partial-year observations.
This supports an industrial-style use case where the completed year-level table
is later passed into a downstream yield prediction model.

## Industrial Use Case

Given historical monthly feature data up to the latest observed month, predict
all remaining missing months of the current or future year.

Example:
- observed: full monthly data from previous years plus January of the target year
- forecast: February through December of the target year

## Dataset and Feature Scope

The forecasting workflow uses a monthly tabular representation built from three
modalities:
- AG features
- NDVI features
- weather features

The main research setup uses a 35-feature monthly table and evaluates one-step
forecasting as well as recursive blank-fill forecasting for held-out target
months.

## Models Compared

- `naive_lag1`
- `seasonal_last_year`
- `LSTM`
- `GRU`
- `tiny_mamba_ssm`
- `transformer_encoder`
- `SARIMA`
- deployable ensembles such as `ensemble_mean` and `ensemble_weighted`

## Current Findings

- `LSTM seasonal_residual` is the best model under raw RMSE for the main
  `known_months=1` blank-fill case.
- `ensemble_mean` is the strongest model under normalized RMSE.
- `SARIMA` is especially strong for NDVI-oriented forecasting.
- AG and NDVI context help weather forecasting; weather-only did not outperform
  the full-feature residual setup.

## Repository Contents

- `scripts/research/` contains the active remote forecasting, diagnostics,
  reporting, comparison, ensemble, and ablation scripts.
- `reports/` contains small markdown summaries and staged CSV tables that are
  useful for review.
- `docs/` contains cleanup, evacuation, and backup-planning notes.
- `src/crop_fusion_ai/` and the original scaffold files remain from the base
  repository and have not yet been refactored around the research workflow.

## How To Run Research Scripts

Typical entry points currently staged in this repository:

```bash
python scripts/research/cropnet_feature_forecasting_v12_server.py --help
python scripts/research/run_controlled_model_comparison.py --help
python scripts/research/analyze_blank_fill_diagnostics.py --help
bash scripts/research/run_feature_ablation_matrix.sh
```

Most of these scripts were copied from the active remote research workspace and
should be treated as research workflow utilities rather than polished deployment
entry points.

## Artifact Policy

Large datasets, caches, raw HDF5 files, virtual environments, checkpoints, and
heavy experiment outputs are intentionally excluded from GitHub.

Before any workspace reset or migration, back up key experiment artifacts such
as:
- residual blank-fill run artifacts
- controlled model comparison outputs
- feature ablation report bundles and summary tables

See `docs/artifact_backup_manifest.md` for the backup list.

## Restoring Backed-Up Artifacts

After cloning this repository into a clean workspace, restore preserved summary
files and important experiment artifacts into a local outputs directory outside
Git history. The staged markdown and CSV summaries in `reports/` are intended as
small review-friendly references, not replacements for the full experiment
artifact backups.

## Current Limitations and Caveats

- The active research scripts were copied from `/home/student/CROPNET/` and may
  still contain hardcoded remote paths that should be cleaned up later.
- The repository still contains the earlier scaffold structure from the base
  `perrywsle/Crop-Net` repository.
- Large experiment outputs, checkpoints, raw chunks, and caches are not staged
  here and must be backed up separately.
