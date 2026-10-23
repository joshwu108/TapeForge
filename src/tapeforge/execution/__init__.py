"""Reference execution simulator (CPU, torch-free, exact integer accounting).

See ``simulator`` for the transition semantics, ``spec`` for the episode
parameters and units, ``state`` for the frozen data carried through an
episode, ``episodes`` for cutting a tape into episodes and sizing targets,
``days`` for whole-day tapes (synthetic or from an archive), ``splits`` for
chronological day splits with embargo and purge, ``sampling`` for seeded
episode draws, ``runner`` for replaying a policy and recording runs,
``baselines`` for the signal-free policies and ``normalization`` for
train-only observation statistics.
"""

from __future__ import annotations

from tapeforge.execution.baselines import BASELINES, cap_aware_twap, immediate, twap
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
from tapeforge.execution.normalization import (
    ObservationStatistics,
    normalize,
    observation_statistics,
)
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
from tapeforge.execution.simulator import (
    OBSERVATION_FIELDS,
    EpisodeOver,
    action_for_quantity,
    implementation_shortfall,
    is_done,
    observe,
    remaining_qty_ticks,
    reset,
    shortfall_bps,
    step,
    terminal_mark,
)
from tapeforge.execution.spec import (
    FILL_RULE_BUCKET_VWAP_CAPPED,
    SIDE_BUY,
    SIDE_SELL,
    EpisodeSpec,
    check_episode_bounds,
    fee_ticks,
    participation_cap_ticks,
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
    "FILL_RULE_BUCKET_VWAP_CAPPED",
    "OBSERVATION_FIELDS",
    "ROLES",
    "SIDE_BUY",
    "SIDE_SELL",
    "DayTape",
    "EpisodeDraw",
    "EpisodeOver",
    "EpisodeSpec",
    "EpisodeSummary",
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
    "Trajectory",
    "action_for_quantity",
    "build_market_slice",
    "build_splits",
    "cap_aware_twap",
    "check_episode_bounds",
    "concat_tapes",
    "cut_slice",
    "day_from_archive",
    "day_from_trades",
    "draw_episode",
    "embargo_buckets",
    "episode_at",
    "fee_ticks",
    "immediate",
    "implementation_shortfall",
    "is_done",
    "make_episode",
    "materialize",
    "normalize",
    "observation_statistics",
    "observe",
    "participation_cap_ticks",
    "purge_window",
    "reference_volume_ticks",
    "remaining_qty_ticks",
    "required_window",
    "reset",
    "run_episode",
    "run_policy",
    "sample_episodes",
    "shortfall_bps",
    "step",
    "summarize_episode",
    "summarize_set",
    "synthetic_day",
    "target_from_reference_volume",
    "terminal_mark",
    "twap",
    "write_run_record",
]
