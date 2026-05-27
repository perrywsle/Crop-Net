from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

@dataclass(slots=True)
class FeatureScaler:
    feature_names: list[str]
    means: np.ndarray
    stds: np.ndarray

    @classmethod
    def from_csv(cls, path: str | Path) -> "FeatureScaler":
        frame = pd.read_csv(path)
        required = {"feature", "mean", "std"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Scaler csv must contain columns {sorted(required)}")
        return cls(
            feature_names=frame["feature"].astype(str).tolist(),
            means=frame["mean"].to_numpy(dtype=float),
            stds=frame["std"].replace(0, 1.0).fillna(1.0).to_numpy(dtype=float),
        )

    def subset(self, selected_features: list[str]) -> "FeatureScaler":
        index = [self.feature_names.index(name) for name in selected_features]
        return FeatureScaler(list(selected_features), self.means[index], self.stds[index])

    def transform_array(self, values: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
        feature_names = feature_names or self.feature_names
        subset = self if feature_names == self.feature_names else self.subset(feature_names)
        return (np.asarray(values, dtype=float) - subset.means) / subset.stds

    def inverse_transform_array(self, values: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
        feature_names = feature_names or self.feature_names
        subset = self if feature_names == self.feature_names else self.subset(feature_names)
        return np.asarray(values, dtype=float) * subset.stds + subset.means

    def transform_frame(self, frame: pd.DataFrame, feature_names: list[str] | None = None) -> pd.DataFrame:
        feature_names = feature_names or self.feature_names
        out = frame.copy()
        out[feature_names] = self.transform_array(out[feature_names].to_numpy(), feature_names)
        return out

    def inverse_transform_frame(self, frame: pd.DataFrame, feature_names: list[str] | None = None) -> pd.DataFrame:
        feature_names = feature_names or self.feature_names
        out = frame.copy()
        out[feature_names] = self.inverse_transform_array(out[feature_names].to_numpy(), feature_names)
        return out
