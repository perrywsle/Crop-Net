from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .scaling import FeatureScaler


DEFAULT_PHYSICS_CONFIG: dict[str, Any] = {
    "latent": {
        "weight": 0.10,
        "warmup_epochs": 5,
        "loss": "combined",
    },
    "ag": {
        "weight": 0.30,
        "growth_rate": 0.35,
        "senescence_rate": 0.18,
        "closure_weight": 0.10,
        "smooth_weight": 0.08,
        "feature_weights": {
            "ag_green_pixel_ratio": 1.6,
            "ag_vegetation_area_percent": 1.6,
            "ag_brown_yellow_pixel_ratio": 1.2,
            "ag_soil_exposure_ratio": 1.1,
            "ag_shadow_cloud_ratio": 0.9,
            "ag_mean_brightness": 0.4,
            "ag_texture_entropy": 0.4,
            "ag_field_uniformity_score": 0.4,
        },
    },
    "ndvi": {
        "weight": 0.40,
        "up_center": 0.25,
        "down_center": 0.75,
        "up_rate": 8.0,
        "down_rate": 8.0,
        "shape_weight": 0.18,
        "ordering_weight": 0.12,
        "bounds_weight": 0.10,
        "smooth_weight": 0.08,
        "feature_weights": {
            "ndvi_mean": 1.8,
            "ndvi_median": 1.4,
            "ndvi_max": 1.2,
            "ndvi_std": 0.8,
            "ndvi_cv": 0.7,
            "ndvi_p25": 0.8,
            "ndvi_p75": 0.8,
            "ndvi_above_0_3_ratio": 1.0,
            "ndvi_above_0_5_ratio": 1.0,
            "ndvi_above_0_7_ratio": 0.9,
            "ndvi_low_ratio": 0.8,
            "ndvi_valid_coverage_ratio": 0.6,
        },
    },
    "weather": {
        "weight": 0.30,
        "gdd_base_c": 10.0,
        "gdd_scale": 30.0,
        "heat_threshold_c": 35.0,
        "frost_threshold_c": 0.0,
        "vpd_scale": 1.0,
        "drought_scale": 1.25,
        "threshold_weight": 0.12,
        "identity_weight": 0.22,
        "bounded_weight": 0.08,
        "feature_weights": {
            "weather_temp_mean": 1.0,
            "weather_temp_max": 1.0,
            "weather_temp_min": 1.0,
            "weather_temp_range_mean": 1.2,
            "weather_gdd": 1.6,
            "weather_heat_stress_days": 1.1,
            "weather_cold_stress_days": 1.1,
            "weather_total_precipitation": 1.2,
            "weather_precipitation_days": 0.9,
            "weather_heavy_rain_days": 0.8,
            "weather_drought_index": 1.5,
            "weather_humidity_mean": 0.8,
            "weather_wind_mean": 0.5,
            "weather_solar_radiation_mean": 0.7,
            "weather_vpd_mean": 1.5,
        },
    },
}


@dataclass(slots=True)
class PinnConfig:
    latent_state_dim: int = 3
    physics_weight: float = 0.1
    physics_warmup_epochs: int = 0
    physics_loss: str = "combined"


def load_physics_config(source: str | Path | dict[str, Any] | None = None) -> dict[str, Any]:
    if source is None:
        return json.loads(json.dumps(DEFAULT_PHYSICS_CONFIG))
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            return json.loads(json.dumps(DEFAULT_PHYSICS_CONFIG))
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = dict(source)
    merged = json.loads(json.dumps(DEFAULT_PHYSICS_CONFIG))
    for group_name, group_cfg in payload.items():
        if isinstance(group_cfg, dict):
            merged.setdefault(group_name, {})
            for key, value in group_cfg.items():
                if key == "feature_weights" and isinstance(value, dict):
                    merged[group_name].setdefault("feature_weights", {})
                    merged[group_name]["feature_weights"].update(value)
                else:
                    merged[group_name][key] = value
    return merged


def _feature_indices(feature_names: list[str], candidates: list[str]) -> list[int]:
    lookup = {name: idx for idx, name in enumerate(feature_names)}
    return [lookup[name] for name in candidates if name in lookup]


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, {})
    return section if isinstance(section, dict) else {}


def _feature_weights(section: dict[str, Any], feature_names: list[str]) -> torch.Tensor:
    weights = torch.ones(len(feature_names), dtype=torch.float32)
    mapping = section.get("feature_weights", {})
    if isinstance(mapping, dict):
        for idx, name in enumerate(feature_names):
            if name in mapping:
                weights[idx] = float(mapping[name])
    return weights


def _positive(value: torch.Tensor) -> torch.Tensor:
    return torch.relu(value)


def _bounded_01(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.relu(-x) ** 2 + torch.relu(x - 1.0) ** 2)


def _second_difference_penalty(x: torch.Tensor) -> torch.Tensor:
    if x.size(1) < 3:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    second = x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]
    return torch.mean(second**2)


def _sigmoid_centered(x: torch.Tensor, center: float, scale: float) -> torch.Tensor:
    return torch.sigmoid((x - center) / max(scale, 1e-6))


def _double_logistic_template(seq_len: int, *, up_center: float, down_center: float, up_rate: float, down_rate: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, seq_len, device=device, dtype=dtype).unsqueeze(0)
    up = torch.sigmoid(up_rate * (t - up_center))
    down = torch.sigmoid(down_rate * (t - down_center))
    template = up * (1.0 - down)
    template = template / (template.max(dim=1, keepdim=True).values + 1e-6)
    return template


class CropPhysicsModule(nn.Module):
    def __init__(
        self,
        feature_names: list[str],
        latent_state_dim: int = 3,
        physics_loss: str = "combined",
        physics_config: dict[str, Any] | str | Path | None = None,
        feature_scaler: FeatureScaler | None = None,
    ) -> None:
        super().__init__()
        if latent_state_dim < 3:
            raise ValueError("latent_state_dim must be at least 3 for the crop-state PINN.")
        self.feature_names = list(feature_names)
        self.latent_state_dim = int(latent_state_dim)
        self.physics_loss = physics_loss
        self.config = load_physics_config(physics_config)
        self._feature_scale_lookup = {
            name: max(float(std), 1e-6)
            for name, std in zip(feature_scaler.feature_names, feature_scaler.stds, strict=False)
        } if feature_scaler is not None else {}

        self.ag_features = [
            "ag_green_pixel_ratio",
            "ag_vegetation_area_percent",
            "ag_brown_yellow_pixel_ratio",
            "ag_soil_exposure_ratio",
            "ag_shadow_cloud_ratio",
            "ag_mean_brightness",
            "ag_texture_entropy",
            "ag_field_uniformity_score",
        ]
        self.ndvi_features = [
            "ndvi_mean",
            "ndvi_median",
            "ndvi_max",
            "ndvi_std",
            "ndvi_cv",
            "ndvi_p25",
            "ndvi_p75",
            "ndvi_above_0_3_ratio",
            "ndvi_above_0_5_ratio",
            "ndvi_above_0_7_ratio",
            "ndvi_low_ratio",
            "ndvi_valid_coverage_ratio",
        ]
        self.weather_features = [
            "weather_temp_mean",
            "weather_temp_max",
            "weather_temp_min",
            "weather_gdd",
            "weather_heat_stress_days",
            "weather_cold_stress_days",
            "weather_total_precipitation",
            "weather_precipitation_days",
            "weather_heavy_rain_days",
            "weather_drought_index",
            "weather_humidity_mean",
            "weather_wind_mean",
            "weather_solar_radiation_mean",
            "weather_vpd_mean",
            "weather_temp_range_mean",
        ]
        self._ag_idx = _feature_indices(self.feature_names, self.ag_features)
        self._ndvi_idx = _feature_indices(self.feature_names, self.ndvi_features)
        self._weather_idx = _feature_indices(self.feature_names, self.weather_features)

        self._ag_weights = _feature_weights(_section(self.config, "ag"), self.feature_names)
        self._ndvi_weights = _feature_weights(_section(self.config, "ndvi"), self.feature_names)
        self._weather_weights = _feature_weights(_section(self.config, "weather"), self.feature_names)

        self.biomass_growth = nn.Parameter(torch.tensor(float(_section(self.config, "latent").get("biomass_growth", 0.15))))
        self.biomass_senescence = nn.Parameter(torch.tensor(float(_section(self.config, "latent").get("biomass_senescence", 0.05))))
        self.phenology_growth = nn.Parameter(torch.tensor(float(_section(self.config, "latent").get("phenology_growth", 0.15))))
        self.phenology_senescence = nn.Parameter(torch.tensor(float(_section(self.config, "latent").get("phenology_senescence", 0.05))))
        self.water_gain = nn.Parameter(torch.tensor(float(_section(self.config, "latent").get("water_gain", 0.15))))
        self.water_loss = nn.Parameter(torch.tensor(float(_section(self.config, "latent").get("water_loss", 0.08))))
        self.coupling = nn.Parameter(torch.tensor(float(_section(self.config, "latent").get("coupling", 0.05))))
        self.extra_state_penalty = nn.Parameter(torch.tensor(float(_section(self.config, "latent").get("extra_state_penalty", 0.01))))

    def _select(self, x: torch.Tensor, indices: list[int], *, transform: str = "identity") -> torch.Tensor:
        if not indices:
            return torch.zeros(x.shape[:2], device=x.device, dtype=x.dtype)
        selected = x[..., indices]
        if transform == "sigmoid":
            return torch.sigmoid(selected.mean(dim=-1))
        if transform == "bounded_ratio":
            return torch.clamp(selected.mean(dim=-1), 0.0, 1.0)
        return selected.mean(dim=-1)

    def _feature_scale(self, feature_name: str, fallback: float) -> float:
        return float(self._feature_scale_lookup.get(feature_name, fallback))

    def _latent_consistency(self, x_raw: torch.Tensor, latent_seq: torch.Tensor) -> torch.Tensor:
        biomass = latent_seq[..., 0]
        phenology = latent_seq[..., 1]
        water = latent_seq[..., 2]

        ag_proxy = self._select(x_raw, _feature_indices(self.feature_names, ["ag_green_pixel_ratio", "ag_vegetation_area_percent"]), transform="sigmoid")
        ndvi_proxy = self._select(x_raw, _feature_indices(self.feature_names, ["ndvi_mean", "ndvi_median", "ndvi_above_0_5_ratio"]), transform="sigmoid")
        weather_proxy = self._select(x_raw, _feature_indices(self.feature_names, ["weather_total_precipitation", "weather_drought_index", "weather_vpd_mean"]), transform="sigmoid")

        return (
            torch.mean((biomass - ag_proxy) ** 2)
            + torch.mean((phenology - ndvi_proxy) ** 2)
            + torch.mean((water - weather_proxy) ** 2)
        )

    def _latent_dynamics(self, x_raw: torch.Tensor, latent_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        seq_len = latent_seq.size(1)
        if seq_len < 2:
            zero = torch.zeros((), device=x_raw.device, dtype=x_raw.dtype)
            return {"growth": zero, "phenology": zero, "water": zero}

        dt = 1.0 / float(seq_len - 1)
        biomass = latent_seq[..., 0]
        phenology = latent_seq[..., 1]
        water = latent_seq[..., 2]

        gdd = self._select(x_raw, _feature_indices(self.feature_names, ["weather_gdd"]), transform="sigmoid")[:, :-1]
        solar = self._select(x_raw, _feature_indices(self.feature_names, ["weather_solar_radiation_mean"]), transform="sigmoid")[:, :-1]
        temp = self._select(x_raw, _feature_indices(self.feature_names, ["weather_temp_mean"]), transform="sigmoid")[:, :-1]
        drought = self._select(x_raw, _feature_indices(self.feature_names, ["weather_drought_index", "weather_vpd_mean"]), transform="sigmoid")[:, :-1]

        biomass_prev = biomass[:, :-1]
        phenology_prev = phenology[:, :-1]
        water_prev = water[:, :-1]

        biomass_delta = biomass[:, 1:] - biomass_prev
        phenology_delta = phenology[:, 1:] - phenology_prev
        water_delta = water[:, 1:] - water_prev

        growth_rate = _positive(self.biomass_growth)
        senescence_rate = _positive(self.biomass_senescence)
        phenology_gain = _positive(self.phenology_growth)
        phenology_senescence = _positive(self.phenology_senescence)
        water_gain = _positive(self.water_gain)
        water_loss = _positive(self.water_loss)
        coupling = _positive(self.coupling)

        growth_rhs = growth_rate * (gdd + solar + temp) * biomass_prev * (1.0 - biomass_prev)
        growth_rhs = growth_rhs + coupling * water_prev * (1.0 - biomass_prev)
        growth_rhs = growth_rhs - senescence_rate * drought * biomass_prev

        phenology_rhs = phenology_gain * biomass_prev * (1.0 - phenology_prev) * (gdd + solar)
        phenology_rhs = phenology_rhs - phenology_senescence * drought * phenology_prev

        water_rhs = water_gain * torch.relu(gdd + solar) - water_loss * drought - coupling * biomass_prev * water_prev

        return {
            "growth": torch.mean((biomass_delta - dt * growth_rhs) ** 2),
            "phenology": torch.mean((phenology_delta - dt * phenology_rhs) ** 2),
            "water": torch.mean((water_delta - dt * water_rhs) ** 2),
        }

    def _ag_loss(self, x_raw: torch.Tensor, latent_seq: torch.Tensor) -> torch.Tensor:
        section = _section(self.config, "ag")
        if not self._ag_idx:
            return torch.zeros((), device=x_raw.device, dtype=x_raw.dtype)

        weights = self._ag_weights.to(device=x_raw.device, dtype=x_raw.dtype)
        ag_values = x_raw[..., self._ag_idx]
        ag_norm = torch.sigmoid(ag_values / 2.0)
        ag_mean = (ag_norm * weights[self._ag_idx]).sum(dim=-1) / (weights[self._ag_idx].sum() + 1e-6)
        ag_biomass = latent_seq[..., 0]

        growth_rate = float(section.get("growth_rate", 0.35))
        senescence_rate = float(section.get("senescence_rate", 0.18))
        closure_weight = float(section.get("closure_weight", 0.10))
        smooth_weight = float(section.get("smooth_weight", 0.08))

        canopy = _sigmoid_centered(ag_mean, 0.45, 0.15)
        bare = _sigmoid_centered(
            self._select(x_raw, _feature_indices(self.feature_names, ["ag_brown_yellow_pixel_ratio", "ag_soil_exposure_ratio", "ag_shadow_cloud_ratio"]), transform="sigmoid"),
            0.35,
            0.15,
        )
        growth_driver = _sigmoid_centered(
            self._select(x_raw, _feature_indices(self.feature_names, ["weather_gdd", "weather_solar_radiation_mean", "weather_temp_mean"]), transform="sigmoid"),
            0.40,
            0.18,
        )
        stress_driver = _sigmoid_centered(
            self._select(x_raw, _feature_indices(self.feature_names, ["weather_drought_index", "weather_vpd_mean", "weather_heat_stress_days"]), transform="sigmoid"),
            0.50,
            0.18,
        )

        seq_len = latent_seq.size(1)
        if seq_len < 2:
            dynamic_loss = torch.zeros((), device=x_raw.device, dtype=x_raw.dtype)
        else:
            biomass_prev = ag_biomass[:, :-1]
            biomass_delta = ag_biomass[:, 1:] - biomass_prev
            rhs = growth_rate * growth_driver[:, :-1] * biomass_prev * (1.0 - biomass_prev)
            rhs = rhs + 0.15 * bare[:, :-1] * (1.0 - biomass_prev)
            rhs = rhs - senescence_rate * stress_driver[:, :-1] * biomass_prev
            dynamic_loss = torch.mean((biomass_delta - rhs / max(seq_len - 1, 1)) ** 2)

        complement = torch.relu(canopy + bare - 1.0) ** 2
        closure = torch.relu(1.0 - (canopy + bare)) ** 2
        smooth = _second_difference_penalty(torch.stack(
            [
                _sigmoid_centered(self._select(x_raw, _feature_indices(self.feature_names, ["ag_mean_brightness"]), transform="sigmoid"), 0.50, 0.20),
                _sigmoid_centered(self._select(x_raw, _feature_indices(self.feature_names, ["ag_texture_entropy"]), transform="sigmoid"), 0.50, 0.20),
                _sigmoid_centered(self._select(x_raw, _feature_indices(self.feature_names, ["ag_field_uniformity_score"]), transform="sigmoid"), 0.50, 0.20),
            ],
            dim=-1,
        ))
        return dynamic_loss + closure_weight * (complement.mean() + closure.mean()) + smooth_weight * smooth

    def _ndvi_loss(self, x_raw: torch.Tensor, latent_seq: torch.Tensor) -> torch.Tensor:
        section = _section(self.config, "ndvi")
        if not self._ndvi_idx:
            return torch.zeros((), device=x_raw.device, dtype=x_raw.dtype)

        _ = self._ndvi_weights.to(device=x_raw.device, dtype=x_raw.dtype)
        ndvi_mean = _sigmoid_centered(self._select(x_raw, _feature_indices(self.feature_names, ["ndvi_mean"]), transform="identity"), 0.50, 0.18)
        seq_len = latent_seq.size(1)

        template = _double_logistic_template(
            seq_len,
            up_center=float(section.get("up_center", 0.25)),
            down_center=float(section.get("down_center", 0.75)),
            up_rate=float(section.get("up_rate", 8.0)),
            down_rate=float(section.get("down_rate", 8.0)),
            device=x_raw.device,
            dtype=x_raw.dtype,
        )
        phenology_latent = latent_seq[..., 1]
        shape_weight = float(section.get("shape_weight", 0.18))
        ordering_weight = float(section.get("ordering_weight", 0.12))
        bounds_weight = float(section.get("bounds_weight", 0.10))
        smooth_weight = float(section.get("smooth_weight", 0.08))

        mean_proxy = _sigmoid_centered(self._select(x_raw, _feature_indices(self.feature_names, ["ndvi_mean", "ndvi_median", "ndvi_max"]), transform="sigmoid"), 0.50, 0.18)
        template_loss = torch.mean((mean_proxy - template) ** 2) + torch.mean((phenology_latent - template) ** 2) + torch.mean((ndvi_mean - template) ** 2)

        order_loss = torch.zeros((), device=x_raw.device, dtype=x_raw.dtype)
        if {"ndvi_p25", "ndvi_median", "ndvi_p75", "ndvi_max"} <= set(self.feature_names):
            p25 = self._select(x_raw, _feature_indices(self.feature_names, ["ndvi_p25"]), transform="sigmoid")
            med = self._select(x_raw, _feature_indices(self.feature_names, ["ndvi_median"]), transform="sigmoid")
            p75 = self._select(x_raw, _feature_indices(self.feature_names, ["ndvi_p75"]), transform="sigmoid")
            mx = self._select(x_raw, _feature_indices(self.feature_names, ["ndvi_max"]), transform="sigmoid")
            order_loss = torch.mean(torch.relu(p25 - med) ** 2 + torch.relu(med - p75) ** 2 + torch.relu(p75 - mx) ** 2)

        bounds_loss = torch.zeros((), device=x_raw.device, dtype=x_raw.dtype)
        for name in ["ndvi_above_0_3_ratio", "ndvi_above_0_5_ratio", "ndvi_above_0_7_ratio", "ndvi_low_ratio", "ndvi_valid_coverage_ratio"]:
            if name in self.feature_names:
                values = _sigmoid_centered(self._select(x_raw, _feature_indices(self.feature_names, [name]), transform="identity"), 0.50, 0.30)
                bounds_loss = bounds_loss + _bounded_01(values)

        smooth_loss = _second_difference_penalty(mean_proxy)
        return shape_weight * template_loss + ordering_weight * order_loss + bounds_weight * bounds_loss + smooth_weight * smooth_loss

    def _weather_loss(self, x_raw: torch.Tensor, latent_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        section = _section(self.config, "weather")
        if not self._weather_idx:
            zero = torch.zeros((), device=x_raw.device, dtype=x_raw.dtype)
            return {
                "total": zero,
                "identity": zero,
                "threshold": zero,
                "drought": zero,
                "bounded": zero,
                "gdd": zero,
                "vpd": zero,
                "temp_range": zero,
                "heat_days": zero,
                "cold_days": zero,
                "precip_days": zero,
                "heavy_days": zero,
            }

        temp_mean = self._select(x_raw, _feature_indices(self.feature_names, ["weather_temp_mean"]), transform="identity")
        temp_max = self._select(x_raw, _feature_indices(self.feature_names, ["weather_temp_max"]), transform="identity")
        temp_min = self._select(x_raw, _feature_indices(self.feature_names, ["weather_temp_min"]), transform="identity")
        precip = self._select(x_raw, _feature_indices(self.feature_names, ["weather_total_precipitation"]), transform="identity")
        humidity = self._select(x_raw, _feature_indices(self.feature_names, ["weather_humidity_mean"]), transform="identity")
        gdd = self._select(x_raw, _feature_indices(self.feature_names, ["weather_gdd"]), transform="identity")
        vpd = self._select(x_raw, _feature_indices(self.feature_names, ["weather_vpd_mean"]), transform="identity")
        drought = self._select(x_raw, _feature_indices(self.feature_names, ["weather_drought_index"]), transform="identity")
        heat_days = self._select(x_raw, _feature_indices(self.feature_names, ["weather_heat_stress_days"]), transform="identity")
        cold_days = self._select(x_raw, _feature_indices(self.feature_names, ["weather_cold_stress_days"]), transform="identity")
        precip_days = self._select(x_raw, _feature_indices(self.feature_names, ["weather_precipitation_days"]), transform="identity")
        heavy_days = self._select(x_raw, _feature_indices(self.feature_names, ["weather_heavy_rain_days"]), transform="identity")
        temp_range = self._select(x_raw, _feature_indices(self.feature_names, ["weather_temp_range_mean"]), transform="identity")

        gdd_base = float(section.get("gdd_base_c", 10.0))
        heat_threshold = float(section.get("heat_threshold_c", 35.0))
        frost_threshold = float(section.get("frost_threshold_c", 0.0))
        drought_scale = float(section.get("drought_scale", 1.25))
        days_per_month = 30.4375

        es = 0.6108 * torch.exp((17.27 * temp_mean) / (temp_mean + 237.3))
        humidity_frac = torch.clamp(humidity / 100.0, 0.0, 1.0)
        vpd_proxy = torch.relu(es * (1.0 - humidity_frac))
        gdd_proxy = torch.relu(temp_mean - gdd_base) * days_per_month
        temp_range_proxy = temp_max - temp_min
        drought_proxy = torch.sigmoid(drought_scale * (vpd_proxy / (precip + 1.0) - 0.5))

        count_scale = days_per_month
        heat_proxy = count_scale * torch.sigmoid((temp_max - heat_threshold) / 2.0)
        cold_proxy = count_scale * torch.sigmoid((frost_threshold - temp_min) / 2.0)
        rain_proxy = count_scale * torch.sigmoid((precip - 1.0) / 4.0)
        heavy_proxy = count_scale * torch.sigmoid((precip - 10.0) / 4.0)

        gdd_loss = F.smooth_l1_loss(
            torch.log1p(torch.clamp(gdd, min=0.0)),
            torch.log1p(torch.clamp(gdd_proxy, min=0.0)),
            reduction="mean",
        )
        vpd_loss = F.smooth_l1_loss(
            (vpd - vpd_proxy) / self._feature_scale("weather_vpd_mean", 1.0),
            torch.zeros_like(vpd),
            reduction="mean",
        )
        temp_range_loss = F.smooth_l1_loss(
            (temp_range - temp_range_proxy) / self._feature_scale("weather_temp_range_mean", 5.0),
            torch.zeros_like(temp_range),
            reduction="mean",
        )
        identity_loss = gdd_loss + vpd_loss + temp_range_loss

        heat_days_loss = F.smooth_l1_loss(
            (heat_days - heat_proxy) / self._feature_scale("weather_heat_stress_days", days_per_month),
            torch.zeros_like(heat_days),
            reduction="mean",
        )
        cold_days_loss = F.smooth_l1_loss(
            (cold_days - cold_proxy) / self._feature_scale("weather_cold_stress_days", days_per_month),
            torch.zeros_like(cold_days),
            reduction="mean",
        )
        precip_days_loss = F.smooth_l1_loss(
            (precip_days - rain_proxy) / self._feature_scale("weather_precipitation_days", days_per_month),
            torch.zeros_like(precip_days),
            reduction="mean",
        )
        heavy_days_loss = F.smooth_l1_loss(
            (heavy_days - heavy_proxy) / self._feature_scale("weather_heavy_rain_days", days_per_month),
            torch.zeros_like(heavy_days),
            reduction="mean",
        )
        threshold_loss = (
            heat_days_loss
            + cold_days_loss
            + precip_days_loss
            + heavy_days_loss
            + F.smooth_l1_loss(
                torch.relu(heavy_days - precip_days) / self._feature_scale("weather_heavy_rain_days", days_per_month),
                torch.zeros_like(heavy_days),
                reduction="mean",
            )
            + F.smooth_l1_loss(
                torch.relu(-precip_days) / self._feature_scale("weather_precipitation_days", days_per_month),
                torch.zeros_like(precip_days),
                reduction="mean",
            )
            + F.smooth_l1_loss(
                torch.relu(precip_days - count_scale) / self._feature_scale("weather_precipitation_days", days_per_month),
                torch.zeros_like(precip_days),
                reduction="mean",
            )
        )

        drought_loss = torch.mean((torch.sigmoid(drought) - drought_proxy) ** 2)
        bounded_loss = _bounded_01(torch.sigmoid((humidity - 50.0) / 20.0)) + _bounded_01(torch.sigmoid((precip - 10.0) / 20.0))
        total = (
            float(section.get("identity_weight", 0.22)) * identity_loss
            + float(section.get("threshold_weight", 0.12)) * threshold_loss
            + float(section.get("bounded_weight", 0.08)) * (drought_loss + bounded_loss)
        )
        return {
            "total": total,
            "identity": identity_loss,
            "threshold": threshold_loss,
            "drought": drought_loss,
            "bounded": bounded_loss,
            "gdd": gdd_loss,
            "vpd": vpd_loss,
            "temp_range": temp_range_loss,
            "heat_days": heat_days_loss,
            "cold_days": cold_days_loss,
            "precip_days": precip_days_loss,
            "heavy_days": heavy_days_loss,
        }

    def forward(self, x_raw: torch.Tensor, latent_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        if latent_seq.ndim != 3:
            raise ValueError("latent_seq must have shape [batch, seq_len, latent_dim].")
        if latent_seq.size(-1) < 3:
            raise ValueError("latent_seq must include at least 3 latent crop states.")

        latent_loss = self._latent_consistency(x_raw, latent_seq)
        latent_dynamics = self._latent_dynamics(x_raw, latent_seq)
        ag_loss = self._ag_loss(x_raw, latent_seq)
        ndvi_loss = self._ndvi_loss(x_raw, latent_seq)
        weather_loss = self._weather_loss(x_raw, latent_seq)

        latent_cfg = _section(self.config, "latent")
        latent_total = latent_loss
        if self.physics_loss == "growth":
            latent_total = latent_total + latent_dynamics["growth"]
        elif self.physics_loss == "phenology":
            latent_total = latent_total + latent_dynamics["phenology"]
        elif self.physics_loss == "water":
            latent_total = latent_total + latent_dynamics["water"]
        else:
            latent_total = latent_total + latent_dynamics["growth"] + latent_dynamics["phenology"] + latent_dynamics["water"]
        if latent_seq.size(-1) > 3:
            latent_total = latent_total + _positive(self.extra_state_penalty) * latent_seq[..., 3:].pow(2).mean()

        total = (
            float(_section(self.config, "ag").get("weight", 0.30)) * ag_loss
            + float(_section(self.config, "ndvi").get("weight", 0.40)) * ndvi_loss
            + float(_section(self.config, "weather").get("weight", 0.30)) * weather_loss["total"]
            + float(latent_cfg.get("weight", 0.10)) * latent_total
        )
        return {
            "total": total,
            "latent": latent_total,
            "ag": ag_loss,
            "ndvi": ndvi_loss,
            "weather": weather_loss["total"],
            "weather_identity": weather_loss["identity"],
            "weather_threshold": weather_loss["threshold"],
            "weather_drought": weather_loss["drought"],
            "weather_bounded": weather_loss["bounded"],
            "weather_gdd": weather_loss["gdd"],
            "weather_vpd": weather_loss["vpd"],
            "weather_temp_range": weather_loss["temp_range"],
            "weather_heat_days": weather_loss["heat_days"],
            "weather_cold_days": weather_loss["cold_days"],
            "weather_precip_days": weather_loss["precip_days"],
            "weather_heavy_days": weather_loss["heavy_days"],
            "consistency": latent_loss,
            "growth": latent_dynamics["growth"],
            "phenology": latent_dynamics["phenology"],
            "water": latent_dynamics["water"],
        }


class PINNForecaster(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_names: list[str],
        *,
        input_dim: int,
        latent_state_dim: int = 3,
        dropout: float = 0.0,
        physics_loss: str = "combined",
        physics_config: dict[str, Any] | str | Path | None = None,
        feature_scaler: FeatureScaler | None = None,
    ) -> None:
        super().__init__()
        if not hasattr(backbone, "encode"):
            raise TypeError("Backbone must expose an encode(x) method.")
        hidden_size = getattr(backbone, "hidden_size", None)
        if hidden_size is None:
            raise TypeError("Backbone must expose a hidden_size attribute.")
        self.backbone = backbone
        self.hidden_size = int(hidden_size)
        self.latent_state_dim = int(latent_state_dim)
        self.latent_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.latent_state_dim),
        )
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size + self.latent_state_dim),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size + self.latent_state_dim, input_dim),
        )
        self.physics = CropPhysicsModule(
            feature_names,
            latent_state_dim=latent_state_dim,
            physics_loss=physics_loss,
            physics_config=physics_config,
            feature_scaler=feature_scaler,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(x)

    def forward_with_latents(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode(x)
        latent = torch.sigmoid(self.latent_head(hidden))
        forecast_input = torch.cat([hidden[:, -1, :], latent[:, -1, :]], dim=-1)
        pred = self.forecast_head(forecast_input)
        return pred, latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred, _ = self.forward_with_latents(x)
        return pred
