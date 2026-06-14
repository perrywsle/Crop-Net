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
