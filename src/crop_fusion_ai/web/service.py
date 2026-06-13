"""Yield-model loading and inference helpers for the web app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from crop_fusion_ai.gui.forecasting import build_monthly_features_from_directory
from crop_fusion_ai.preprocessing.common import aggregate_monthly_feature_frame
from cropnet_forecasting.data import prepare_monthly_features

from .feature_labels import GROUP_LABELS, feature_label, group_for_feature, visible_feature_names

ProgressCallback = Callable[[str, int, int, str], None]

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "outputs"
    / "yield_baseline"
    / "corn_ia_2017_2022_monthly"
    / "best_yield_model.joblib"
)
DEFAULT_METADATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "outputs"
    / "yield_baseline"
    / "corn_ia_2017_2022_monthly"
    / "yield_model_metadata.json"
)
DEFAULT_FEATURE_IMPORTANCE_PATH = (
    Path(__file__).resolve().parents[3]
    / "outputs"
    / "yield_baseline"
    / "corn_ia_2017_2022_monthly"
    / "yield_feature_importance.csv"
)
DEFAULT_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[3]
    / "outputs"
    / "yield_baseline"
    / "corn_ia_2017_2022_monthly"
    / "yield_model_benchmark.csv"
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value):
        return None
    return value


def _format_number(value: float | int | None, *, display: str = "decimal") -> str:
    if value is None or pd.isna(value):
        return "Not available"
    numeric = float(value)
    if display == "percent":
        return f"{numeric * 100.0:.1f}%"
    if display == "integer":
        return f"{int(round(numeric)):,}"
    return f"{numeric:.2f}"


def _normalize_county_id(values: pd.Series) -> pd.Series:
    coerced = values.astype(str).str.extract(r"(\d+)", expand=False).fillna("")
    return coerced.str.zfill(5)


def _normalize_crop_type(value: str | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return normalized or None


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "county_id" in out.columns:
        out["county_id"] = _normalize_county_id(out["county_id"])
    if "crop_type" in out.columns:
        out["crop_type"] = out["crop_type"].astype(str).map(_normalize_crop_type)
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    if "month" in out.columns:
        out["month"] = pd.to_numeric(out["month"], errors="coerce").astype("Int64")
    return out


def _month_label(row: pd.Series) -> str:
    year = row.get("year")
    month = row.get("month")
    if pd.isna(year) or pd.isna(month):
        return ""
    return f"{int(year)}-{int(month):02d}"


@dataclass(slots=True)
class YieldModelSummary:
    model_name: str
    feature_count: int
    holdout_rmse: float | None
    holdout_mae: float | None
    holdout_r2: float | None
    target_units: str
    feature_importance: pd.DataFrame
    benchmark: pd.DataFrame

    def to_payload(self) -> dict[str, Any]:
        top_features = []
        for row in self.feature_importance.head(10).itertuples(index=False):
            spec = feature_label(str(row.feature))
            top_features.append(
                {
                    "name": spec.name,
                    "label": spec.label,
                    "group": spec.group,
                    "group_label": GROUP_LABELS.get(spec.group, spec.group.title()),
                    "description": spec.description,
                    "display": spec.display,
                    "unit": spec.unit,
                    "importance": float(row.importance),
                }
            )

        best_row = None
        if not self.benchmark.empty:
            candidates = self.benchmark[self.benchmark["model_type"] == "ml"].copy()
            if candidates.empty:
                candidates = self.benchmark.copy()
            best = candidates.sort_values(["rmse", "mae", "model"]).iloc[0]
            best_row = {
                "model": str(best["model"]),
                "model_type": str(best["model_type"]),
                "rmse": float(best["rmse"]),
                "mae": float(best["mae"]),
                "r2": float(best["r2"]) if pd.notna(best.get("r2")) else None,
                "mape": float(best["mape"]) if pd.notna(best.get("mape")) else None,
            }

        return {
            "model_name": self.model_name,
            "feature_count": self.feature_count,
            "target_units": self.target_units,
            "holdout": {
                "rmse": self.holdout_rmse,
                "mae": self.holdout_mae,
                "r2": self.holdout_r2,
            },
            "best_model": best_row,
            "top_features": top_features,
        }


class YieldModelService:
    """Load the saved monthly yield model and produce farmer-friendly results."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
        feature_importance_path: str | Path = DEFAULT_FEATURE_IMPORTANCE_PATH,
        benchmark_path: str | Path = DEFAULT_BENCHMARK_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.feature_importance_path = Path(feature_importance_path)
        self.benchmark_path = Path(benchmark_path)
        self._model = joblib.load(self.model_path)
        self._metadata = self._load_json(self.metadata_path)
        benchmark = self._load_benchmark(self.benchmark_path)
        self._summary = YieldModelSummary(
            model_name=str(self._metadata.get("best_model_name") or self._model.named_steps["model"].__class__.__name__),
            feature_count=len(getattr(self._model, "feature_names_in_", [])),
            holdout_rmse=self._extract_metric(benchmark, "rmse"),
            holdout_mae=self._extract_metric(benchmark, "mae"),
            holdout_r2=self._extract_metric(benchmark, "r2"),
            target_units=", ".join(self._metadata.get("target_units", []) or ["BU / ACRE"]),
            feature_importance=self._load_feature_importance(self.feature_importance_path),
            benchmark=benchmark,
        )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        import json

        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_feature_importance(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=["feature", "importance"])
        frame = pd.read_csv(path)
        if "feature" not in frame.columns or "importance" not in frame.columns:
            return pd.DataFrame(columns=["feature", "importance"])
        return frame.sort_values("importance", ascending=False).reset_index(drop=True)

    @staticmethod
    def _load_benchmark(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_csv(path)
        expected = {"model", "model_type", "rmse", "mae", "r2", "mape"}
        if not expected.issubset(frame.columns):
            return pd.DataFrame()
        return frame

    @staticmethod
    def _extract_metric(benchmark: pd.DataFrame, metric: str) -> float | None:
        if benchmark.empty:
            return None
        candidates = benchmark[benchmark["model_type"] == "ml"].copy()
        if candidates.empty:
            candidates = benchmark.copy()
        if candidates.empty or metric not in candidates.columns:
            return None
        best = candidates.sort_values(["rmse", "mae", "model"]).iloc[0]
        value = best.get(metric)
        return None if pd.isna(value) else float(value)

    @property
    def summary(self) -> YieldModelSummary:
        return self._summary

    @property
    def model_name(self) -> str:
        return self._summary.model_name

    @property
    def feature_names(self) -> list[str]:
        names = getattr(self._model, "feature_names_in_", None)
        if names is None:
            return []
        return [str(name) for name in names]

    def build_monthly_frame(
        self,
        root_dir: str | Path,
        *,
        county_id: str,
        crop_type: str,
        progress: ProgressCallback | None = None,
    ) -> tuple[pd.DataFrame, list[Any]]:
        monthly_features, source_files = build_monthly_features_from_directory(
            root_dir,
            county_id=county_id,
            crop_type=crop_type,
            progress=progress,
        )
        monthly_features = _normalize_frame(monthly_features)
        return monthly_features, source_files

    def predict_from_monthly_frame(self, monthly_frame: pd.DataFrame) -> dict[str, Any]:
        if monthly_frame.empty:
            raise ValueError("The monthly feature table is empty.")

        frame = _normalize_frame(monthly_frame)
        if "month" not in frame.columns:
            raise ValueError("Monthly feature table must include a month column.")

        feature_names = self.feature_names or [str(name) for name in getattr(self._model, "feature_names_in_", [])]
        prepared = prepare_monthly_features(frame, feature_names)
        prepared = aggregate_monthly_feature_frame(prepared)
        prepared = _normalize_frame(prepared)
        if prepared.empty:
            raise ValueError("The monthly feature table could not be prepared for prediction.")

        if "month_sin" not in prepared.columns:
            radians = 2.0 * np.pi * pd.to_numeric(prepared["month"], errors="coerce").to_numpy(dtype=float) / 12.0
            prepared["month_sin"] = np.sin(radians)
        if "month_cos" not in prepared.columns:
            radians = 2.0 * np.pi * pd.to_numeric(prepared["month"], errors="coerce").to_numpy(dtype=float) / 12.0
            prepared["month_cos"] = np.cos(radians)

        for column in feature_names:
            if column not in prepared.columns:
                prepared[column] = np.nan

        prepared = prepared.sort_values(["county_id", "crop_type", "year", "month"]).reset_index(drop=True)
        prediction_frame = prepared.copy()
        prediction_frame["predicted_yield"] = self._model.predict(prediction_frame[feature_names])

        latest_row = prediction_frame.iloc[-1].to_dict()
        headline = {
            "predicted_yield": float(latest_row["predicted_yield"]),
            "unit": self._summary.target_units or "BU / ACRE",
            "model_name": self.model_name,
            "year": int(latest_row["year"]) if pd.notna(latest_row.get("year")) else None,
            "month": int(latest_row["month"]) if pd.notna(latest_row.get("month")) else None,
        }

        prediction_rows = []
        for row in prediction_frame.itertuples(index=False):
            month_value = int(getattr(row, "month")) if pd.notna(getattr(row, "month")) else None
            prediction_rows.append(
                {
                    "county_id": str(getattr(row, "county_id")),
                    "crop_type": str(getattr(row, "crop_type")),
                    "year": int(getattr(row, "year")) if pd.notna(getattr(row, "year")) else None,
                    "month": month_value,
                    "month_label": _month_label(pd.Series(row._asdict())),
                    "predicted_yield": float(getattr(row, "predicted_yield")),
                }
            )

        feature_group_payload = []
        latest = prediction_frame.iloc[-1]
        for group in ("canopy", "vegetation", "weather", "season"):
            group_features = [
                name
                for name in feature_names
                if group_for_feature(name) == group
                and name in prepared.columns
                and visible_feature_names([name])
            ]
            if not group_features:
                continue

            cards = []
            for feature_name in group_features:
                spec = feature_label(feature_name)
                series_columns = list(dict.fromkeys(["year", "month", feature_name]))
                cards.append(
                    {
                        "name": feature_name,
                        "label": spec.label,
                        "description": spec.description,
                        "display": spec.display,
                        "unit": spec.unit,
                        "latest_value": _format_number(latest.get(feature_name), display=spec.display),
                        "series": [
                            {
                                "month_label": row["month_label"],
                                "value": None
                                if pd.isna(row[feature_name])
                                else float(row[feature_name]),
                            }
                            for _, row in prepared[series_columns].assign(
                                month_label=prepared[["year", "month"]].apply(_month_label, axis=1)
                            ).iterrows()
                        ],
                    }
                )
            feature_group_payload.append(
                {
                    "group": group,
                    "label": GROUP_LABELS.get(group, group.title()),
                    "features": cards,
                }
            )

        importance = self._summary.feature_importance.head(8).copy()
        importance_payload = []
        for row in importance.itertuples(index=False):
            spec = feature_label(str(row.feature))
            importance_payload.append(
                {
                    "name": spec.name,
                    "label": spec.label,
                    "group": spec.group,
                    "group_label": GROUP_LABELS.get(spec.group, spec.group.title()),
                    "description": spec.description,
                    "importance": float(row.importance),
                }
            )

        drivers = [item for item in importance_payload[:5]]
        yield_series = prediction_rows
        return {
            "summary": self._summary.to_payload(),
            "headline": headline,
            "prediction_rows": prediction_rows,
            "yield_series": prediction_rows,
            "feature_groups": feature_group_payload,
            "monthly_features": prediction_frame.assign(
                month_label=prediction_frame[["year", "month"]].apply(_month_label, axis=1)
            ).replace({np.nan: None}).to_dict(orient="records"),
            "feature_importance": importance_payload,
            "drivers": drivers,
        }

    def predict_from_directory(
        self,
        root_dir: str | Path,
        *,
        county_id: str,
        crop_type: str,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        monthly_features, source_files = self.build_monthly_frame(
            root_dir,
            county_id=county_id,
            crop_type=crop_type,
            progress=progress,
        )
        payload = self.predict_from_monthly_frame(monthly_features)
        payload["source_files"] = [
            {
                "path": str(item.path),
                "modality": item.modality,
                "year": item.year,
                "month": item.month,
                "day": item.day,
            }
            for item in source_files
        ]
        return payload

    def predict_from_sample_directory(
        self,
        *,
        county_id: str = "19001",
        crop_type: str = "corn",
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        sample_root = Path(__file__).resolve().parents[3] / "data" / "sample_data"
        return self.predict_from_directory(
            sample_root,
            county_id=county_id,
            crop_type=crop_type,
            progress=progress,
        )
