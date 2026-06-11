"""Tests for extraction-only feature preprocessing mode."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_server_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "research"
        / "cropnet_feature_forecasting_v12_server.py"
    )
    spec = importlib.util.spec_from_file_location("cropnet_feature_server_for_tests", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_only_config_skips_forecasting_even_with_full_year_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit extraction-only flag should not depend on split inference."""
    server = _load_server_module()
    output_dir = tmp_path / "extract_only"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cropnet_feature_forecasting_v12_server.py",
            "--full-run",
            "--extract-only",
            "--output-dir",
            str(output_dir),
            "--state-codes",
            "IA",
            "--crop",
            "Corn",
            "--years",
            "2017",
            "2018",
            "2019",
            "2020",
            "2021",
            "2022",
            "--max-counties",
            "5",
        ],
    )

    args = server.parse_args()
    cfg = server.build_config(args)

    assert cfg.extract_only is True
    assert cfg.run_forecasting is True
    assert cfg.output_dir == output_dir.resolve()


def test_extract_only_rejects_blank_fill_eval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extraction-only mode must not be mixed with generated blank-fill outputs."""
    server = _load_server_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cropnet_feature_forecasting_v12_server.py",
            "--full-run",
            "--extract-only",
            "--run-blank-fill-eval",
            "--output-dir",
            str(tmp_path / "bad"),
            "--years",
            "2021",
            "2022",
        ],
    )

    args = server.parse_args()
    with pytest.raises(ValueError, match="extract-only"):
        server.build_config(args)


def test_extract_only_environment_logging_does_not_require_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extraction-only environment logging should tolerate the Torch placeholder."""
    server = _load_server_module()
    monkeypatch.setattr(server, "np", np)
    monkeypatch.setattr(server, "torch", types.SimpleNamespace())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cropnet_feature_forecasting_v12_server.py",
            "--full-run",
            "--extract-only",
            "--output-dir",
            str(tmp_path / "extract_only"),
            "--years",
            "2021",
            "2022",
        ],
    )
    cfg = server.build_config(server.parse_args())

    server.print_environment(cfg)


def test_extract_only_pipeline_entry_does_not_seed_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The extraction-only pipeline entry should not call Torch before returning."""
    server = _load_server_module()
    monkeypatch.setattr(server, "np", np)
    monkeypatch.setattr(server, "torch", types.SimpleNamespace())
    monkeypatch.setattr(server, "ensure_repo_extractors", lambda cfg: None)
    monkeypatch.setattr(
        server,
        "try_load_existing_monthly_artifacts",
        lambda cfg: (
            pd.DataFrame({"county_id": ["19001"], "year": [2022], "month": [4], "ag_green_pixel_ratio": [1.0]}),
            ["ag_green_pixel_ratio"],
            {},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cropnet_feature_forecasting_v12_server.py",
            "--full-run",
            "--extract-only",
            "--output-dir",
            str(tmp_path / "extract_only"),
            "--years",
            "2021",
            "2022",
        ],
    )
    cfg = server.build_config(server.parse_args())

    server.run_pipeline(cfg)
