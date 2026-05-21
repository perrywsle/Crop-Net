"""Tests for optional MobileNet dependency handling."""

import pytest

from crop_fusion_ai.models.mobilenet_feature_extractor import (
    MobileNetDependencyError,
    MobileNetFeatureExtractor,
)


def test_mobilenet_feature_extractor_reports_missing_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing PyTorch stack should produce an actionable setup error."""

    def missing_torch_stack() -> tuple[object, object, object]:
        raise MobileNetDependencyError("missing optional torch stack")

    monkeypatch.setattr(
        MobileNetFeatureExtractor,
        "_import_torch_stack",
        staticmethod(missing_torch_stack),
    )

    with pytest.raises(MobileNetDependencyError, match="missing optional torch stack"):
        MobileNetFeatureExtractor()
