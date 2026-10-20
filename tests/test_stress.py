"""Week 4: robustness across the published stress cases (PRD section 7)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("n_records", [100_000, 1_000_000])
@pytest.mark.parametrize("profile", ["sparse", "uniform", "burst"])
def test_no_incorrect_features_at_scale(n_records: int, profile: str) -> None:
    pytest.skip("week 4: enable once reference + kernel are implemented")


def test_integer_overflow_bound_is_respected_at_extremes() -> None:
    pytest.skip("week 4: enable once the student fixes the accumulation dtype")
