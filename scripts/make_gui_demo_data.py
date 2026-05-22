"""Create small CropNet-shaped demo inputs for the preprocessing GUI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"
TABULAR_DIR = PROJECT_ROOT / "data" / "raw" / "tabular"


def _ag_demo_image() -> np.ndarray:
    image = np.full((128, 128, 3), (160, 132, 88), dtype=np.uint8)
    image[16:72, 14:64] = (46, 170, 58)
    image[58:104, 68:114] = (58, 182, 70)
    image[20:44, 78:112] = (186, 164, 78)
    image[90:122, 0:24] = (72, 60, 52)
    image[0:12, :] = (238, 238, 238)
    image[:, 0:8] = (92, 92, 92)
    return image


def _ndvi_demo_array() -> np.ndarray:
    array = np.full((64, 64), -0.05, dtype=np.float64)
    array[0:32, 0:32] = 0.48
    array[0:32, 32:64] = 0.68
    array[32:64, 0:32] = 0.18
    array[32:64, 32:64] = -0.02
    return array


def _weather_demo_frame() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for year, temp_offset, rain_offset in ((2022, 0.0, 0.0), (2023, 1.5, 3.0)):
        for month, days in ((1, 31), (2, 28), (3, 31)):
            for day in range(1, days + 1):
                temp_mean = 19.0 + month + temp_offset + day * 0.05
                temp_max = temp_mean + 5.5
                temp_min = temp_mean - 5.5
                precipitation = 0.0 if day % 6 else 5.0 + rain_offset
                rows.append(
                    {
                        "date": f"{year}-{month:02d}-{day:02d}",
                        "temp_mean": temp_mean,
                        "temp_max": temp_max,
                        "temp_min": temp_min,
                        "precipitation": precipitation,
                        "humidity": 67.0 + month,
                        "solar_radiation": 125.0 + month * 4.0 + day * 0.2,
                        "wind_speed": 2.2 + month * 0.1,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TABULAR_DIR.mkdir(parents=True, exist_ok=True)

    ag_path = IMAGES_DIR / "demo_ag.png"
    ndvi_path = IMAGES_DIR / "demo_ndvi.png"
    weather_path = TABULAR_DIR / "demo_weather.csv"

    Image.fromarray(_ag_demo_image(), mode="RGB").save(ag_path)

    ndvi_scaled = np.clip((_ndvi_demo_array() + 1.0) / 2.0, 0.0, 1.0)
    Image.fromarray((ndvi_scaled * 255.0).astype(np.uint8), mode="L").save(ndvi_path)

    _weather_demo_frame().to_csv(weather_path, index=False)

    print(f"Wrote {ag_path}")
    print(f"Wrote {ndvi_path}")
    print(f"Wrote {weather_path}")


if __name__ == "__main__":
    main()
