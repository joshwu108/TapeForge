"""Shared experiment configurations for the experiment, report, replay and adapter tests.

``synthetic_mapping`` returns the TOML-shaped mapping of a small synthetic
experiment (three short days, two train and one validation) so each test
can tweak one key and build an ``ExperimentConfig`` from it.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from tapeforge.execution.experiment import (
    EXPERIMENT_CONFIG_SCHEMA,
    ExperimentConfig,
    config_from_mapping,
)

DATES = ("2024-01-01", "2024-01-02", "2024-01-03")


def synthetic_mapping(**overrides: Any) -> dict[str, Any]:
    """Return the mapping of the synthetic smoke experiment with top-level ``overrides`` applied."""
    mapping: dict[str, Any] = {
        "schema": EXPERIMENT_CONFIG_SCHEMA,
        "name": "synthetic-test",
        "label": "smoke",
        "description": "three synthetic days for the test suite",
        "data": {
            "source": "synthetic",
            "market": "synthetic",
            "synthetic": {
                "n_buckets": 300,
                "profile": "uniform",
                "days": [{"date": d, "seed": i + 1} for i, d in enumerate(DATES)],
            },
        },
        "split": {"train": [DATES[0], DATES[1]], "validation": [DATES[2]], "test": []},
        "episodes": {
            "seeds": [1, 2],
            "episodes_per_day": 4,
            "rho_choices_bps": [200, 500, 1000],
            "evaluate_roles": ["train", "validation"],
        },
        "policies": {"names": ["twap", "immediate", "cap_aware_twap"]},
        "template": {"horizon_buckets": 12, "arrival_lookback": 4},
        "output_dir": "runs",
    }
    mapping.update(deepcopy(overrides))
    return mapping


def synthetic_config(**overrides: Any) -> ExperimentConfig:
    """Return the ``ExperimentConfig`` of ``synthetic_mapping(**overrides)``."""
    return config_from_mapping(synthetic_mapping(**overrides))
