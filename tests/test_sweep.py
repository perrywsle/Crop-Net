from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.sweep import _args_from_mapping, _build_jobs, _expand_grid, _slug_combo


def test_sweep_grid_expansion_and_slugs() -> None:
    config = {
        "name": "pinn",
        "models": ["gru"],
        "base_args": {
            "target_mode": "raw",
            "physics_weight": 0.03,
        },
        "grid": {
            "hidden_size": [64, 96],
            "learning_rate": [0.001, 0.0005],
        },
    }

    jobs = _build_jobs(config)
    assert len(jobs) == 1
    job = jobs[0]
    combos = _expand_grid(job.grid)
    assert len(combos) == 4
    assert _slug_combo({"hidden_size": 96, "learning_rate": 0.001}) == "hidden-size-96__learning-rate-0p001"


def test_args_from_mapping_handles_lists_and_flags() -> None:
    args = _args_from_mapping(
        {
            "models": ["lstm", "gru"],
            "overwrite": True,
            "physics_weight": 0.03,
            "dropout": 0.05,
            "disabled": False,
        }
    )
    assert args == [
        "--models",
        "lstm",
        "gru",
        "--overwrite",
        "--physics-weight",
        "0.03",
        "--dropout",
        "0.05",
    ]
