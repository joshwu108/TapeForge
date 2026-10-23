"""Normalisation statistics come from the training split only."""

from __future__ import annotations

import numpy as np
import pytest

from tapeforge.contract import DEFAULT_BUCKET_WIDTH_US as W
from tapeforge.execution.baselines import twap
from tapeforge.execution.days import DayTape, synthetic_day
from tapeforge.execution.normalization import (
    ObservationStatistics,
    normalize,
    observation_statistics,
)
from tapeforge.execution.runner import run_episode
from tapeforge.execution.sampling import materialize, sample_episodes
from tapeforge.execution.simulator import OBSERVATION_FIELDS
from tapeforge.execution.spec import EpisodeSpec
from tapeforge.execution.splits import SplitDates, Splits, build_splits
from test_execution.tapes import DEEP, P0, slice_from_rows

SPEC = EpisodeSpec(side=1, target_qty_ticks=1, horizon_buckets=5, arrival_lookback=3)
DATES = ("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04")


def _splits(validation_seed: int = 50, test_seed: int = 60) -> Splits:
    days = [
        synthetic_day(1, DATES[0], 80, start_us=0),
        synthetic_day(2, DATES[1], 80, start_us=80 * W),
        synthetic_day(validation_seed, DATES[2], 80, start_us=160 * W, profile="burst"),
        synthetic_day(test_seed, DATES[3], 80, start_us=240 * W, profile="sparse"),
    ]
    return build_splits(days, SplitDates((DATES[0], DATES[1]), (DATES[2],), (DATES[3],)), SPEC)


def test_statistics_are_per_field_mean_and_population_std_of_the_visited_observations() -> None:
    splits = _splits()
    draws = sample_episodes(splits.record, "train", 6, seed=3)
    stats = observation_statistics(splits, draws, SPEC, twap, "twap")
    rows = []
    for draw in draws:
        spec, market = materialize(splits, draw, SPEC)
        rows.extend(run_episode(spec, market, twap).observations)
    matrix = np.vstack(rows)
    assert stats.fields == OBSERVATION_FIELDS
    assert stats.n_observations == matrix.shape[0] and stats.n_episodes == 6
    np.testing.assert_allclose(stats.mean, matrix.mean(axis=0))
    np.testing.assert_allclose(stats.std, matrix.std(axis=0))
    assert stats.policy == "twap" and stats.seed == 3
    assert isinstance(stats, ObservationStatistics)


def test_changing_validation_and_test_data_leaves_the_statistics_unchanged() -> None:
    baseline = _splits()
    draws = sample_episodes(baseline.record, "train", 8, seed=5)
    reference = observation_statistics(baseline, draws, SPEC, twap, "twap")
    for validation_seed, test_seed in [(51, 60), (50, 61), (99, 98)]:
        perturbed = _splits(validation_seed, test_seed)
        # Same geometry and ranges; only the validation/test day data (and sources) differ.
        assert perturbed.record.segments == baseline.record.segments
        assert perturbed.record.days[:2] == baseline.record.days[:2]
        other = observation_statistics(perturbed, draws, SPEC, twap, "twap")
        assert other == reference
    # Changing a training day does change them (the statistic is not constant).
    days = [
        synthetic_day(7, DATES[0], 80, start_us=0),
        synthetic_day(2, DATES[1], 80, start_us=80 * W),
        synthetic_day(50, DATES[2], 80, start_us=160 * W, profile="burst"),
        synthetic_day(60, DATES[3], 80, start_us=240 * W, profile="sparse"),
    ]
    changed = build_splits(days, SplitDates((DATES[0], DATES[1]), (DATES[2],), (DATES[3],)), SPEC)
    assert observation_statistics(changed, draws, SPEC, twap, "twap") != reference


def test_statistics_refuse_non_training_draws_and_empty_input() -> None:
    splits = _splits()
    validation = sample_episodes(splits.record, "validation", 2, seed=1)
    with pytest.raises(ValueError, match="training split only"):
        observation_statistics(splits, validation, SPEC, twap, "twap")
    train = sample_episodes(splits.record, "train", 1, seed=1)
    with pytest.raises(ValueError, match="training split only"):
        observation_statistics(splits, train + validation[:1], SPEC, twap, "twap")
    with pytest.raises(ValueError, match="at least one"):
        observation_statistics(splits, (), SPEC, twap, "twap")


def test_normalize_centres_and_scales_with_a_unit_divisor_for_constant_fields() -> None:
    stats = ObservationStatistics(
        fields=("a", "b", "c"), mean=(1.0, 2.0, 3.0), std=(2.0, 0.0, 0.5),
        n_observations=4, n_episodes=1, policy="twap", seed=0,
    )
    np.testing.assert_array_equal(normalize(np.array([3.0, 2.0, 2.0]), stats), [1.0, 0.0, -2.0])
    np.testing.assert_array_equal(normalize(np.array([1.0, 5.0, 3.0]), stats), [0.0, 3.0, 0.0])
    with pytest.raises(ValueError, match="statistics cover 3 fields"):
        normalize(np.zeros(4), stats)
    with pytest.raises(ValueError, match="same length"):
        ObservationStatistics(("a",), (0.0, 1.0), (1.0,), 1, 1, "twap", 0)


def test_statistics_are_json_serialisable_and_round_trip() -> None:
    import json

    splits = _splits()
    stats = observation_statistics(splits, sample_episodes(splits.record, "train", 3, 2), SPEC,
                                   twap, "twap")
    loaded = json.loads(json.dumps(stats.to_json()))
    assert loaded["fields"] == list(OBSERVATION_FIELDS) and len(loaded["mean"]) == 10
    assert loaded["policy"] == "twap" and loaded["n_episodes"] == 3


def test_normalized_training_observations_have_zero_mean_and_unit_variance() -> None:
    splits = _splits()
    draws = sample_episodes(splits.record, "train", 6, seed=3)
    stats = observation_statistics(splits, draws, SPEC, twap, "twap")
    rows = []
    for draw in draws:
        spec, market = materialize(splits, draw, SPEC)
        rows.extend(normalize(o, stats) for o in run_episode(spec, market, twap).observations)
    matrix = np.vstack(rows)
    np.testing.assert_allclose(matrix.mean(axis=0), 0.0, atol=1e-9)
    varying = np.array(stats.std) > 0
    np.testing.assert_allclose(matrix.std(axis=0)[varying], 1.0, atol=1e-9)
    assert varying.sum() >= 5   # most fields vary across the sampled episodes


def test_a_constant_tape_gives_zero_std_everywhere_but_time() -> None:
    day = DayTape(DATES[0], 0, W, slice_from_rows([], [(DEEP, P0)] * 60), "flat")
    splits = build_splits([day], SplitDates((DATES[0],), ()), SPEC)
    draws = sample_episodes(splits.record, "train", 4, seed=1)
    stats = observation_statistics(splits, draws, SPEC, lambda _: 0.0, "zero")
    by_name = dict(zip(stats.fields, stats.std, strict=True))
    assert by_name["mark_return"] == 0.0 and by_name["bucket_return"] == 0.0
    assert by_name["time_remaining_fraction"] > 0.0
