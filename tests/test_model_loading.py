"""Tests for dynamic model loading and legacy checkpoint compatibility."""

from __future__ import annotations

from pathlib import Path

from cropnet_forecasting.models import load_legacy_module


def test_load_legacy_module_registers_dataclass_module(tmp_path: Path) -> None:
    """The legacy loader should register the module before executing dataclasses."""
    legacy_script = tmp_path / "legacy_dataclass_module.py"
    legacy_script.write_text(
        "\n".join(
            [
                "from dataclasses import dataclass",
                "",
                "@dataclass",
                "class ExampleConfig:",
                "    value: int",
                "",
                "def sentinel() -> int:",
                "    return ExampleConfig(3).value",
                "",
            ]
        ),
        encoding="utf-8",
    )

    module = load_legacy_module(legacy_script)

    assert module is not None
    assert module.ExampleConfig(7).value == 7
    assert module.sentinel() == 3
