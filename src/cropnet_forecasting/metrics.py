from __future__ import annotations

import numpy as np
import pandas as pd

from .features import modality_for_feature
from .scaling import FeatureScaler

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def per_feature_metrics(y_true: np.ndarray, y_pred: np.ndarray, feature_names: list[str], scaler: FeatureScaler | None = None) -> pd.DataFrame:
    records = []
    for idx, feature_name in enumerate(feature_names):
        feature_true = y_true[:, idx]
        feature_pred = y_pred[:, idx]
        record = {
            "feature": feature_name,
            "modality": modality_for_feature(feature_name),
            "rmse": rmse(feature_true, feature_pred),
            "mae": mae(feature_true, feature_pred),
        }
        if scaler is not None:
            feature_std = float(scaler.subset([feature_name]).stds[0])
            if feature_std > 0:
                record["nrmse_std"] = record["rmse"] / feature_std
        records.append(record)
    return pd.DataFrame(records)

def summarize_by_modality(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    return metrics_frame.groupby("modality", as_index=False)[["rmse", "mae"]].mean()
