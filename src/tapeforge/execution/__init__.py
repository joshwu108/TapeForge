"""Reference execution simulator (CPU, torch-free, exact integer accounting).

See ``simulator`` for the transition semantics, ``spec`` for the episode
parameters and units, ``state`` for the frozen data carried through an
episode and ``episodes`` for cutting a tape into episodes and sizing targets.
"""

from __future__ import annotations

from tapeforge.execution.episodes import (
    build_market_slice,
    make_episode,
    reference_volume_ticks,
    required_window,
    target_from_reference_volume,
)
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
from tapeforge.execution.state import MarketSlice, State, StepResult

__all__ = [
    "FILL_RULE_BUCKET_VWAP_CAPPED",
    "OBSERVATION_FIELDS",
    "SIDE_BUY",
    "SIDE_SELL",
    "EpisodeOver",
    "EpisodeSpec",
    "MarketSlice",
    "State",
    "StepResult",
    "action_for_quantity",
    "build_market_slice",
    "check_episode_bounds",
    "fee_ticks",
    "implementation_shortfall",
    "is_done",
    "make_episode",
    "observe",
    "participation_cap_ticks",
    "reference_volume_ticks",
    "remaining_qty_ticks",
    "required_window",
    "reset",
    "shortfall_bps",
    "step",
    "target_from_reference_volume",
    "terminal_mark",
]
