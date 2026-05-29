import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

USDA_PATH = ROOT / "data/sample_data/USDA Crop Dataset/Corn/2021/USDA_Corn_County_2021.csv"
OUT_PATH  = ROOT / "outputs/sample_monthly_features.csv"

NDVI_COLS = [
    "ndvi_mean","ndvi_median","ndvi_max","ndvi_std","ndvi_cv",
    "ndvi_p25","ndvi_p75","ndvi_above_0_3_ratio","ndvi_above_0_5_ratio",
    "ndvi_above_0_7_ratio","ndvi_low_ratio","ndvi_valid_coverage_ratio",
]
AG_COLS = [
    "ag_green_pixel_ratio","ag_vegetation_area_percent","ag_brown_yellow_pixel_ratio",
    "ag_soil_exposure_ratio","ag_shadow_cloud_ratio","ag_mean_brightness",
    "ag_texture_entropy","ag_field_uniformity_score",
]
WEATHER_COLS = [
    "weather_temp_mean","weather_temp_max","weather_temp_min","weather_gdd",
    "weather_heat_stress_days","weather_cold_stress_days","weather_total_precipitation",
    "weather_precipitation_days","weather_heavy_rain_days","weather_drought_index",
    "weather_humidity_mean","weather_wind_mean","weather_solar_radiation_mean",
    "weather_vpd_mean","weather_temp_range_mean",
]

# Seasonal NDVI curve template
NDVI_SEASONAL = [0.10, 0.12, 0.18, 0.30, 0.50, 0.65, 0.75, 0.72, 0.55, 0.35, 0.18, 0.11]

np.random.seed(42)

def make_monthly_rows(county_id, crop_type, year):
    rows = []
    base_ndvi = np.random.uniform(0.7, 1.3)   # county productivity scalar
    for mo in range(1, 13):
        row = {"county_id": county_id, "crop_type": crop_type, "year": year, "month": mo}
        gs_ndvi = NDVI_SEASONAL[mo - 1] * base_ndvi
        for col in NDVI_COLS:
            row[col] = float(np.clip(gs_ndvi + np.random.normal(0, 0.03), 0, 1))
        for col in AG_COLS:
            row[col] = float(np.clip(np.random.normal(0.4, 0.15), 0, 1))
        for col in WEATHER_COLS:
            row[col] = float(np.random.normal(50, 20))
        rows.append(row)
    return rows

# Load real FIPS from USDA — use first 30 counties
usda = pd.read_csv(USDA_PATH)
usda["county_id"] = usda["state_ansi"].astype(str).str.zfill(2) + usda["county_ansi"].astype(str).str.zfill(3)
counties = usda["county_id"].unique()[:30].tolist()

records = []
for fips in counties:
    for year in [2017, 2018, 2019, 2020, 2021]:
        records.extend(make_monthly_rows(fips, "Corn", year))

OUT_PATH.parent.mkdir(exist_ok=True)
df = pd.DataFrame(records)
df.to_csv(OUT_PATH, index=False)
print(f"Saved {len(df)} rows → {OUT_PATH}")
print(f"Counties: {len(counties)}  |  Years: 2017-2021  |  Rows per county-year: 12")