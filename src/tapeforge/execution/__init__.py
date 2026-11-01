"""Reference execution simulator: CPU-only, torch-free, exact integer accounting.

Submodules and their responsibilities:

* ``spec``: episode parameters, units and the int64 bound check.
* ``state``: immutable data carried through an episode (tape slice, state,
  step record).
* ``simulator``: ``reset`` / ``observe`` / ``step`` transition semantics, the
  fill rule and the reward decomposition.
* ``episodes``: the tape window an episode needs and target sizing from the
  reference volume.
* ``days``: whole-day tapes, synthetic or from an aggTrades archive.
* ``splits``: chronological day splits with embargo and purge controls.
* ``sampling``: seeded, index-addressable episode draws.
* ``runner``: policy replay, summaries and JSON run records.
* ``baselines``: signal-free reference policies and the zero-action
  diagnostic.
* ``normalization``: train-only observation statistics and their versioned
  files.
* ``datasets``: archive acquisition with checksums, a budget and a manifest.
* ``experiment``: TOML-configured baseline experiments and run directories.
* ``report``: per-policy, per-day and paired statistics over a run.
* ``replay``: step-by-step replay of one recorded episode.
* ``sensitivity``: cross-run comparison of terminal-rule settings on
  identical episodes.
* ``evaluation``: fixed evaluation sets, the external-policy interface,
  strict training configuration and checkpoint provenance.
* ``gym_env`` and ``env_check``: Gymnasium adapter and its checker command;
  both need the optional ``rl`` extra and are not imported here.
"""

from __future__ import annotations

from tapeforge.execution.baselines import (
    BASELINES,
    DIAGNOSTIC_POLICIES,
    cap_aware_twap,
    immediate,
    twap,
    zero_action,
)
from tapeforge.execution.days import (
    DayTape,
    concat_tapes,
    cut_slice,
    day_from_archive,
    day_from_trades,
    episode_at,
    synthetic_day,
)
from tapeforge.execution.episodes import (
    build_market_slice,
    make_episode,
    reference_volume_ticks,
    required_window,
    target_from_reference_volume,
)
from tapeforge.execution.evaluation import (
    TrainingConfig,
    checkpoint_provenance,
    evaluate_policy,
    evaluation_manifest,
    load_training_config,
    validate_checkpoint_provenance,
    vector_policy,
)
from tapeforge.execution.experiment import (
    ExperimentConfig,
    load_config,
    run_experiment,
)
from tapeforge.execution.normalization import (
    ObservationStatistics,
    load_statistics,
    normalize,
    observation_statistics,
    save_statistics,
)
from tapeforge.execution.replay import replay_episode
from tapeforge.execution.report import build_report, write_report
from tapeforge.execution.runner import (
    EpisodeSummary,
    Observation,
    Policy,
    SetSummary,
    Trajectory,
    run_episode,
    run_policy,
    summarize_episode,
    summarize_set,
    write_run_record,
)
from tapeforge.execution.sampling import EpisodeDraw, draw_episode, materialize, sample_episodes
from tapeforge.execution.sensitivity import build_sensitivity, write_sensitivity
from tapeforge.execution.simulator import (
    OBSERVATION_FIELDS,
    SIMULATOR_VERSION,
    EpisodeOver,
    action_for_quantity,
    fill_shortfall,
    implementation_shortfall,
    is_done,
    observe,
    remaining_qty_ticks,
    reset,
    settlement_shortfall,
    shortfall_bps,
    step,
    terminal_mark,
)
from tapeforge.execution.spec import (
    FILL_RULE_BUCKET_VWAP_CAPPED,
    SIDE_BUY,
    SIDE_SELL,
    TERMINAL_RULE_RESIDUAL_PEROLD,
    TERMINAL_RULE_SETTLEMENT,
    EpisodeSpec,
    check_episode_bounds,
    fee_ticks,
    participation_cap_ticks,
    settlement_price_ticks,
)
from tapeforge.execution.splits import (
    ROLES,
    SplitDates,
    SplitRecord,
    Splits,
    build_splits,
    embargo_buckets,
    purge_window,
)
from tapeforge.execution.state import MarketSlice, State, StepResult

__all__ = [
    "BASELINES",
    "DIAGNOSTIC_POLICIES",
    "FILL_RULE_BUCKET_VWAP_CAPPED",
    "OBSERVATION_FIELDS",
    "ROLES",
    "SIDE_BUY",
    "SIDE_SELL",
    "SIMULATOR_VERSION",
    "TERMINAL_RULE_RESIDUAL_PEROLD",
    "TERMINAL_RULE_SETTLEMENT",
    "DayTape",
    "EpisodeDraw",
    "EpisodeOver",
    "EpisodeSpec",
    "EpisodeSummary",
    "ExperimentConfig",
    "MarketSlice",
    "Observation",
    "ObservationStatistics",
    "Policy",
    "SetSummary",
    "SplitDates",
    "SplitRecord",
    "Splits",
    "State",
    "StepResult",
    "TrainingConfig",
    "Trajectory",
    "action_for_quantity",
    "build_market_slice",
    "build_report",
    "build_sensitivity",
    "build_splits",
    "cap_aware_twap",
    "check_episode_bounds",
    "checkpoint_provenance",
    "concat_tapes",
    "cut_slice",
    "day_from_archive",
    "day_from_trades",
    "draw_episode",
    "embargo_buckets",
    "episode_at",
    "evaluate_policy",
    "evaluation_manifest",
    "fee_ticks",
    "fill_shortfall",
    "immediate",
    "implementation_shortfall",
    "is_done",
    "load_config",
    "load_statistics",
    "load_training_config",
    "make_episode",
    "materialize",
    "normalize",
    "observation_statistics",
    "observe",
    "participation_cap_ticks",
    "purge_window",
    "reference_volume_ticks",
    "remaining_qty_ticks",
    "replay_episode",
    "required_window",
    "reset",
    "run_episode",
    "run_experiment",
    "run_policy",
    "sample_episodes",
    "save_statistics",
    "settlement_price_ticks",
    "settlement_shortfall",
    "shortfall_bps",
    "step",
    "summarize_episode",
    "summarize_set",
    "synthetic_day",
    "target_from_reference_volume",
    "terminal_mark",
    "twap",
    "validate_checkpoint_provenance",
    "vector_policy",
    "write_report",
    "write_run_record",
    "write_sensitivity",
    "zero_action",
]
