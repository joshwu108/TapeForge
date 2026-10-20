"""Public aggTrades loading, normalization and validation (TF-02).

Loads one symbol / one day of Binance spot ``aggTrades`` from the public
archive, normalizes timestamps to ``contract.TIME_UNIT``, converts price and
quantity to scaled integers, and records dataset provenance.

Never trust the archive: validate the column count, the timestamp unit
(microseconds from 2025-01-01, milliseconds before -- PRD section 3), the
sort order, and the checksum before any feature code sees the rows.

Downloaded archives land in ``data/`` and are git-ignored (PRD section 10:
do not commit large downloaded files).

CLI:
    python -m tapeforge.ingest --symbol BTCUSDT --date 2025-01-02
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"


@dataclass(frozen=True)
class Provenance:
    """Where a dataset came from, recorded next to every benchmark result."""

    symbol: str
    date: str          # YYYY-MM-DD
    url: str
    sha256: str
    n_records: int
    timestamp_unit: str


def archive_url(symbol: str, date: str) -> str:
    """Return the public ZIP URL for one symbol-day."""
    return f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{date}.zip"


def load_agg_trades(path: str) -> tuple[np.ndarray, Provenance]:
    """Parse a downloaded aggTrades ZIP/CSV into ``TRADE_DTYPE`` + provenance.

    Note (PRD section 3): a row is an *aggregate* trade record, not an
    individual fill. Record counts must be described as aggregate-trade
    records in every benchmark report.
    """
    raise NotImplementedError("week 1: implement after tests/test_ingest.py is RED")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: download one symbol-day, verify checksum, report schema."""
    parser = argparse.ArgumentParser(prog="python -m tapeforge.ingest")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--date", default="2025-01-02")
    parser.add_argument("--out-dir", default="data")
    parser.parse_args(argv)
    raise NotImplementedError("week 1: implement after tests/test_ingest.py is RED")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
