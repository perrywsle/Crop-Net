"""Preprocessing utilities for CropNet modality feature extraction."""

from crop_fusion_ai.preprocessing.ag import (
    derive_ag_time_series_features,
    extract_ag_features,
)
from crop_fusion_ai.preprocessing.common import combine_modality_feature_frames
from crop_fusion_ai.preprocessing.common import aggregate_monthly_feature_frame
from crop_fusion_ai.preprocessing.ndvi import (
    derive_ndvi_time_series_features,
    extract_ndvi_features,
)
from crop_fusion_ai.preprocessing.usda_dataset import (
    build_usda_record,
    build_usda_records_from_frame,
    filter_usda_rows,
    infer_usda_split,
    normalize_crop_type,
    parse_usda_remote_path,
    resolve_usda_target_column,
    resolve_usda_target_unit,
    select_usda_remote_files,
    write_jsonl_records,
    write_jsonl_splits,
)
from crop_fusion_ai.preprocessing.weather import (
    derive_weather_time_series_features,
    extract_weather_features,
)

__all__ = [
    "combine_modality_feature_frames",
    "aggregate_monthly_feature_frame",
    "build_usda_record",
    "build_usda_records_from_frame",
    "build_usda_records_from_dataframe",
    "build_usda_records_from_snapshot",
    "build_modality_allow_patterns",
    "build_modality_record_from_sentinel_tensor",
    "build_modality_record_from_weather_frame",
    "build_modality_records_by_split",
    "collect_usda_anchor_records",
    "derive_ag_time_series_features",
    "derive_ndvi_time_series_features",
    "derive_weather_time_series_features",
    "extract_ag_features",
    "extract_ndvi_features",
    "extract_weather_features",
    "filter_usda_rows",
    "infer_usda_split",
    "normalize_crop_type",
    "parse_usda_remote_path",
    "resolve_usda_target_column",
    "resolve_usda_target_unit",
    "select_usda_remote_files",
    "write_jsonl_records",
    "write_jsonl_splits",
]

def _missing_cropnet_export(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise ModuleNotFoundError(
        "h5py is required for cropnet_export helpers"
    )

try:  # pragma: no cover - optional dependency path
    from crop_fusion_ai.preprocessing.cropnet_export import (
        build_modality_allow_patterns,
        build_modality_record_from_sentinel_tensor,
        build_modality_record_from_weather_frame,
        build_modality_records_by_split,
        build_usda_records_from_dataframe,
        build_usda_records_from_snapshot,
        collect_usda_anchor_records,
        extract_county_ids_from_usda_records,
        get_state_abbr,
        normalize_fips_code,
        write_modality_jsonl_splits,
    )
except ModuleNotFoundError:
    build_modality_allow_patterns = _missing_cropnet_export
    build_modality_record_from_sentinel_tensor = _missing_cropnet_export
    build_modality_record_from_weather_frame = _missing_cropnet_export
    build_modality_records_by_split = _missing_cropnet_export
    build_usda_records_from_dataframe = _missing_cropnet_export
    build_usda_records_from_snapshot = _missing_cropnet_export
    collect_usda_anchor_records = _missing_cropnet_export
    extract_county_ids_from_usda_records = _missing_cropnet_export
    get_state_abbr = _missing_cropnet_export
    normalize_fips_code = _missing_cropnet_export
    write_modality_jsonl_splits = _missing_cropnet_export
else:
    __all__.extend(
        [
            "build_modality_allow_patterns",
            "build_modality_record_from_sentinel_tensor",
            "build_modality_record_from_weather_frame",
            "build_modality_records_by_split",
            "build_usda_records_from_dataframe",
            "build_usda_records_from_snapshot",
            "collect_usda_anchor_records",
            "extract_county_ids_from_usda_records",
            "get_state_abbr",
            "normalize_fips_code",
            "write_modality_jsonl_splits",
        ]
    )
