"""Yield-model loading and inference helpers for the web app."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from crop_fusion_ai.gui.forecasting import build_monthly_features_from_directory
from crop_fusion_ai.gui.forecasting import build_forecast_from_monthly_features
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
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "outputs"
    / "yield_baseline"
    / "corn_ia_2017_2022_monthly"
    / "yield_model_metadata.json"
)
DEFAULT_FEATURE_FORECAST_CONFIG = Path(__file__).resolve().parents[3] / "training" / "runs" / "lstm_best" / "config.json"
DEFAULT_FEATURE_FORECAST_SCALER = Path(__file__).resolve().parents[3] / "training" / "runs" / "lstm_best" / "scaler.csv"
DEFAULT_FEATURE_FORECAST_CHECKPOINT = Path(__file__).resolve().parents[3] / "training" / "runs" / "lstm_best" / "lstm" / "checkpoint.pt"
YIELD_MODEL_DIR = Path(__file__).resolve().parents[3] / "outputs" / "predicted_yield_experiments" / "yield_models"
FEATURE_MODEL_RUNS: dict[str, dict[str, Any]] = {
    "lstm": {
        "label": "LSTM",
        "run_dir": Path(__file__).resolve().parents[3] / "training" / "runs" / "lstm_best",
    },
    "gru": {
        "label": "GRU",
        "run_dir": Path(__file__).resolve().parents[3] / "training" / "runs" / "gru_best",
    },
    "transformer_encoder": {
        "label": "Transformer Encoder",
        "run_dir": Path(__file__).resolve().parents[3] / "training" / "runs" / "transformer_best",
    },
    "tiny_mamba_ssm": {
        "label": "Tiny Mamba SSM",
        "run_dir": Path(__file__).resolve().parents[3] / "training" / "runs" / "mamba_best",
    },
}
YIELD_MODEL_VARIANTS: dict[str, dict[str, Any]] = {
    "naive_lag1": {
        "label": "Naive lag-1",
        "path": YIELD_MODEL_DIR / "naive_lag1" / "best_yield_model.joblib",
    },
    "seasonal_last_year": {
        "label": "Seasonal last year",
        "path": YIELD_MODEL_DIR / "seasonal_last_year" / "best_yield_model.joblib",
    },
    "lstm": {
        "label": "LSTM",
        "path": YIELD_MODEL_DIR / "lstm" / "best_yield_model.joblib",
    },
    "transformer_encoder": {
        "label": "Transformer Encoder",
        "path": YIELD_MODEL_DIR / "transformer_encoder" / "best_yield_model.joblib",
    },
    "gru": {
        "label": "GRU",
        "path": YIELD_MODEL_DIR / "gru" / "best_yield_model.joblib",
    },
    "ensemble_mean": {
        "label": "Ensemble mean",
        "path": YIELD_MODEL_DIR / "ensemble_mean" / "best_yield_model.joblib",
    },
    "ensemble_weighted": {
        "label": "Ensemble weighted",
        "path": YIELD_MODEL_DIR / "ensemble_weighted" / "best_yield_model.joblib",
    },
}


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


def _predict_yield_rows(model: Any, frame: pd.DataFrame, feature_names: list[str]) -> list[dict[str, Any]]:
    prediction_frame = frame.copy()
    model_feature_names_attr = getattr(model, "feature_names_in_", None)
    model_feature_names = [str(name) for name in model_feature_names_attr] if model_feature_names_attr is not None else feature_names
    aligned_features = prediction_frame.reindex(columns=model_feature_names)
    prediction_frame["predicted_yield"] = model.predict(aligned_features)
    rows: list[dict[str, Any]] = []
    for row in prediction_frame.itertuples(index=False):
        month_value = int(getattr(row, "month")) if pd.notna(getattr(row, "month")) else None
        rows.append(
            {
                "county_id": str(getattr(row, "county_id")),
                "crop_type": str(getattr(row, "crop_type")),
                "year": int(getattr(row, "year")) if pd.notna(getattr(row, "year")) else None,
                "month": month_value,
                "month_label": _month_label(pd.Series(row._asdict())),
                "predicted_yield": float(getattr(row, "predicted_yield")),
            }
        )
    return rows


def _load_feature_run_metrics(model_key: str, spec: dict[str, Any]) -> dict[str, Any] | None:
    run_dir = Path(spec["run_dir"])
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "config.json"
    if not metrics_path.exists():
        return None
    metrics = pd.read_csv(metrics_path)
    row = metrics[metrics["model"] == model_key].copy()
    if row.empty:
        row = metrics.head(1).copy()
    if row.empty:
        return None
    record = row.iloc[0].to_dict()
    config: dict[str, Any] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "key": model_key,
        "label": str(spec.get("label", model_key)),
        "run_dir": str(run_dir),
        "config_path": str(config_path) if config_path.exists() else None,
        "checkpoint_path": record.get("checkpoint_path"),
        "history_path": record.get("history_path"),
        "loss_curve_path": record.get("loss_curve_path"),
        "physics_curve_path": record.get("physics_curve_path"),
        "predictions_path": record.get("predictions_path"),
        "trainable_parameters": int(record["trainable_parameters"]) if pd.notna(record.get("trainable_parameters")) else None,
        "total_parameters": int(record["total_parameters"]) if pd.notna(record.get("total_parameters")) else None,
        "train_loss": float(record["train_loss"]) if pd.notna(record.get("train_loss")) else None,
        "val_loss": float(record["val_loss"]) if pd.notna(record.get("val_loss")) else None,
        "physics_loss": float(record["physics_loss"]) if pd.notna(record.get("physics_loss")) else None,
        "physics_latent_loss": float(record["physics_latent_loss"]) if pd.notna(record.get("physics_latent_loss")) else None,
        "physics_ag_loss": float(record["physics_ag_loss"]) if pd.notna(record.get("physics_ag_loss")) else None,
        "physics_ndvi_loss": float(record["physics_ndvi_loss"]) if pd.notna(record.get("physics_ndvi_loss")) else None,
        "physics_weather_loss": float(record["physics_weather_loss"]) if pd.notna(record.get("physics_weather_loss")) else None,
        "physics_weather_identity_loss": float(record["physics_weather_identity_loss"]) if pd.notna(record.get("physics_weather_identity_loss")) else None,
        "physics_weather_threshold_loss": float(record["physics_weather_threshold_loss"]) if pd.notna(record.get("physics_weather_threshold_loss")) else None,
        "physics_weather_drought_loss": float(record["physics_weather_drought_loss"]) if pd.notna(record.get("physics_weather_drought_loss")) else None,
        "physics_weather_bounded_loss": float(record["physics_weather_bounded_loss"]) if pd.notna(record.get("physics_weather_bounded_loss")) else None,
        "physics_consistency_loss": float(record["physics_consistency_loss"]) if pd.notna(record.get("physics_consistency_loss")) else None,
        "physics_growth_loss": float(record["physics_growth_loss"]) if pd.notna(record.get("physics_growth_loss")) else None,
        "physics_phenology_loss": float(record["physics_phenology_loss"]) if pd.notna(record.get("physics_phenology_loss")) else None,
        "physics_water_loss": float(record["physics_water_loss"]) if pd.notna(record.get("physics_water_loss")) else None,
        "rmse": float(record["rmse"]) if pd.notna(record.get("rmse")) else None,
        "mae": float(record["mae"]) if pd.notna(record.get("mae")) else None,
        "mse": float(record["mse"]) if pd.notna(record.get("mse")) else None,
        "r2": float(record["r2"]) if pd.notna(record.get("r2")) else None,
        "val_rmse": float(record["val_rmse"]) if pd.notna(record.get("val_rmse")) else None,
        "val_mae": float(record["val_mae"]) if pd.notna(record.get("val_mae")) else None,
        "val_mse": float(record["val_mse"]) if pd.notna(record.get("val_mse")) else None,
        "val_r2": float(record["val_r2"]) if pd.notna(record.get("val_r2")) else None,
        "status": str(record.get("status") or ""),
        "target_mode": str(config.get("target_mode", "")) if config else None,
        "seq_len": int(config["seq_len"]) if config.get("seq_len") is not None else None,
        "hidden_size": int(config["hidden_size"]) if config.get("hidden_size") is not None else None,
        "num_layers": int(config["num_layers"]) if config.get("num_layers") is not None else None,
        "dropout": float(config["dropout"]) if config.get("dropout") is not None else None,
        "learning_rate": float(config["learning_rate"]) if config.get("learning_rate") is not None else None,
        "weight_decay": float(config["weight_decay"]) if config.get("weight_decay") is not None else None,
        "physics_weight": float(config["physics_weight"]) if config.get("physics_weight") is not None else None,
        "physics_warmup_epochs": int(config["physics_warmup_epochs"]) if config.get("physics_warmup_epochs") is not None else None,
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
        self._reference_model = joblib.load(self.model_path)
        self._yield_models = self._load_yield_models()
        self._metadata = self._load_json(self.metadata_path)
        self._model = self._reference_model
        self._feature_model_runs = self._load_feature_model_runs()
        benchmark = self._load_benchmark(self.benchmark_path)
        self._summary = YieldModelSummary(
            model_name=str(
                self._metadata.get("best_trainable_model")
                or self._metadata.get("best_model_name")
                or self._metadata.get("best_overall_model")
                or self._reference_model.named_steps["model"].__class__.__name__
            ),
            feature_count=len(getattr(self._reference_model, "feature_names_in_", [])),
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
    def _load_yield_models() -> dict[str, dict[str, Any]]:
        models: dict[str, dict[str, Any]] = {}
        for key, spec in YIELD_MODEL_VARIANTS.items():
            path = Path(spec["path"])
            if not path.exists():
                continue
            models[key] = {
                "label": spec["label"],
                "model": joblib.load(path),
            }
        return models

    @staticmethod
    def _load_feature_model_runs() -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for key, spec in FEATURE_MODEL_RUNS.items():
            loaded = _load_feature_run_metrics(key, spec)
            if loaded is not None:
                runs.append(loaded)
        runs.sort(
            key=lambda item: (
                -(item.get("r2") if item.get("r2") is not None else float("-inf")),
                item.get("rmse") if item.get("rmse") is not None else float("inf"),
                item.get("label") or item.get("key"),
            )
        )
        return runs

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
        names = getattr(self._reference_model, "feature_names_in_", None)
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

    def prepare_monthly_frame(self, monthly_frame: pd.DataFrame) -> pd.DataFrame:
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

        return prepared.sort_values(["county_id", "crop_type", "year", "month"]).reset_index(drop=True)

    def _predict_from_prepared_monthly_frame(self, prepared: pd.DataFrame) -> dict[str, Any]:
        feature_names = self.feature_names or [str(name) for name in getattr(self._model, "feature_names_in_", [])]
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

        prediction_rows = _predict_yield_rows(self._model, prediction_frame, feature_names)
        yield_series_by_model: dict[str, list[dict[str, Any]]] = {
            "current": prediction_rows,
        }
        yield_model_payload = [{"key": "current", "label": self.model_name, "series": prediction_rows}]
        for model_key, model_info in self._yield_models.items():
            model_rows = _predict_yield_rows(model_info["model"], prediction_frame, feature_names)
            yield_series_by_model[model_key] = model_rows
            yield_model_payload.append(
                {
                    "key": model_key,
                    "label": str(model_info["label"]),
                    "series": model_rows,
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
        feature_model_best = self._feature_model_runs[0] if self._feature_model_runs else None
        return {
            "summary": self._summary.to_payload(),
            "benchmark_summary": self._summary.to_payload(),
            "headline": headline,
            "prediction_rows": prediction_rows,
            "yield_series": prediction_rows,
            "yield_series_by_model": yield_series_by_model,
            "yield_models": yield_model_payload,
            "feature_model_runs": self._feature_model_runs,
            "feature_model_best": feature_model_best,
            "feature_groups": feature_group_payload,
            "monthly_features": prediction_frame.assign(
                month_label=prediction_frame[["year", "month"]].apply(_month_label, axis=1)
            ).replace({np.nan: None}).to_dict(orient="records"),
            "feature_importance": importance_payload,
            "drivers": drivers,
        }

    def predict_from_monthly_frame(self, monthly_frame: pd.DataFrame) -> dict[str, Any]:
        prepared = self.prepare_monthly_frame(monthly_frame)
        return self._predict_from_prepared_monthly_frame(prepared)

    def predict_from_directory(
        self,
        root_dir: str | Path,
        *,
        county_id: str,
        crop_type: str,
        progress: ProgressCallback | None = None,
        preprocessed: tuple[pd.DataFrame, list[Any]] | None = None,
        prepared_monthly_frame: pd.DataFrame | None = None,
        feature_forecasts: Any | None = None,
    ) -> dict[str, Any]:
        if preprocessed is None:
            monthly_features, source_files = self.build_monthly_frame(
                root_dir,
                county_id=county_id,
                crop_type=crop_type,
                progress=progress,
            )
        else:
            monthly_features, source_files = preprocessed
        if prepared_monthly_frame is None:
            prepared_monthly_frame = self.prepare_monthly_frame(monthly_features)
        payload = self._predict_from_prepared_monthly_frame(prepared_monthly_frame)
        if feature_forecasts is None:
            feature_forecasts = build_forecast_from_monthly_features(
                monthly_features,
                source_files,
                county_id=county_id,
                crop_type=crop_type,
                checkpoint_path=DEFAULT_FEATURE_FORECAST_CHECKPOINT,
                scaler_path=DEFAULT_FEATURE_FORECAST_SCALER,
                config_path=DEFAULT_FEATURE_FORECAST_CONFIG,
                progress=progress,
            )
        future_feature_frame = None
        for model_key in ("tiny_mamba_ssm", "transformer_encoder", "gru", "lstm"):
            candidate = feature_forecasts.forecast_by_model.get(model_key)
            if candidate is not None and not candidate.empty:
                future_feature_frame = candidate
                break
        if future_feature_frame is None or future_feature_frame.empty:
            future_feature_frame = next(iter(feature_forecasts.forecast_by_model.values()), pd.DataFrame())
        yield_trajectory_by_model: dict[str, list[dict[str, Any]]] = {}
        if not future_feature_frame.empty:
            for model_key in ("lstm", "gru", "tiny_mamba_ssm", "transformer_encoder"):
                candidate = feature_forecasts.forecast_by_model.get(model_key)
                if candidate is None or candidate.empty:
                    continue
                candidate = candidate.copy()
                if "county_id" not in candidate.columns:
                    candidate["county_id"] = str(county_id).zfill(5)
                if "crop_type" not in candidate.columns:
                    candidate["crop_type"] = str(crop_type)
                candidate = candidate.sort_values(["county_id", "crop_type", "year", "month"]).reset_index(drop=True)
                candidate = candidate.replace({np.nan: None})
                yield_trajectory_by_model[model_key] = _predict_yield_rows(self._model, candidate, self.feature_names)
        forecast_headline = None
        current_trajectory = next(iter(yield_trajectory_by_model.values()), [])
        if current_trajectory:
            forecast_headline = dict(current_trajectory[-1])
            forecast_headline["model_name"] = self.model_name
            forecast_headline["unit"] = self._summary.target_units or "BU / ACRE"
        payload["feature_forecasts_by_model"] = {
            model_key: forecast.assign(
                month_label=forecast[["year", "month"]].apply(_month_label, axis=1)
            ).replace({np.nan: None}).to_dict(orient="records")
            for model_key, forecast in feature_forecasts.forecast_by_model.items()
        }
        payload["feature_forecast_models"] = [
            {
                "key": model_key,
                "label": {
                    "lstm": "LSTM",
                    "gru": "GRU",
                    "transformer_encoder": "Transformer Encoder",
                    "tiny_mamba_ssm": "Tiny Mamba SSM",
                }.get(
                    model_key,
                    feature_forecasts.predictor_by_model[model_key].model_name.replace("_", " ").title()
                    if model_key in feature_forecasts.predictor_by_model
                    else model_key,
                ),
                "supports_latent_state": bool(
                    getattr(feature_forecasts.predictor_by_model.get(model_key), "supports_latent_state", False)
                ),
            }
            for model_key in feature_forecasts.forecast_by_model.keys()
        ]
        payload["derived_drivers_by_model"] = {
            model_key: forecast.assign(
                month_label=forecast[["year", "month"]].apply(_month_label, axis=1)
            ).replace({np.nan: None}).to_dict(orient="records")
            for model_key, forecast in feature_forecasts.derived_drivers_by_model.items()
        }
        payload["derived_driver_models"] = [
            {
                "key": model_key,
                "label": next(
                    (
                        item["label"]
                        for item in payload["feature_forecast_models"]
                        if item["key"] == model_key
                    ),
                    model_key,
                ),
                "supports_latent_state": True,
            }
            for model_key in feature_forecasts.derived_drivers_by_model.keys()
        ]
        payload["yield_trajectory_by_model"] = yield_trajectory_by_model
        if forecast_headline is not None:
            payload["forecast_headline"] = forecast_headline
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
