"""Public aggTrades loading, normalization and validation (TF-02).

Loads one symbol-day of Binance spot ``aggTrades`` from the public archive,
normalizes timestamps to ``contract.TIME_UNIT``, converts price and quantity
to exactly scaled integers, and records dataset provenance.

A row is an *aggregate* trade record, not an individual fill. Record counts
in this project always mean aggregate-trade records

CLI:
    python -m tapeforge.ingest --symbol BTCUSDT --date 2025-01-02
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from tapeforge.contract import (
    PRICE_SCALE,
    QUANTITY_SCALE,
    TRADE_DTYPE,
    US_PER_SECOND,
)

BASE_URL: Final[str] = "https://data.binance.vision/data/spot/daily/aggTrades"

EXPECTED_COLUMNS: Final[tuple[int, ...]] = (7, 8)
COL_AGG_TRADE_ID: Final[int] = 0
COL_PRICE: Final[int] = 1
COL_QUANTITY: Final[int] = 2
COL_TRANSACT_TIME: Final[int] = 5
COL_IS_BUYER_MAKER: Final[int] = 6

US_DETECTION_THRESHOLD: Final[int] = 10**14

# Plausibility window for a normalized timestamp, in microseconds:
# 2015-01-01 to 2100-01-01. Outside this, the unit detection was wrong.
MIN_PLAUSIBLE_US: Final[int] = 1_420_070_400 * US_PER_SECOND
MAX_PLAUSIBLE_US: Final[int] = 4_102_444_800 * US_PER_SECOND

TRUE_STRINGS: Final[frozenset[str]] = frozenset({"true", "True", "TRUE", "1"})
FALSE_STRINGS: Final[frozenset[str]] = frozenset({"false", "False", "FALSE", "0"})


@dataclass(frozen=True)
class Provenance:
    """Where a dataset came from, recorded next to every benchmark result."""
    symbol: str
    date: str          # YYYY-MM-DD
    url: str
    sha256: str
    n_records: int
    timestamp_unit: str
    was_resorted: bool = False


def archive_url(symbol: str, date: str) -> str:
    """Return the public ZIP URL for one symbol-day."""
    return f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{date}.zip"


def _scale_digits(scale: int) -> int:
    """Number of decimal places a power-of-ten scale can represent exactly."""
    digits = len(str(scale)) - 1
    if 10**digits != scale:
        raise ValueError(f"scale {scale} is not a power of ten")
    return digits


def decimal_strings_to_ticks(values: np.ndarray, scale: int) -> np.ndarray:
    """Convert decimal strings to exactly scaled int64 ticks.

    Raises:
        ValueError: on a negative value, scientific notation, more than one
            decimal point, or precision beyond ``scale``.
    """
    digits = _scale_digits(scale)
    text = np.asarray(values, dtype=np.str_)
    if text.size == 0:
        return np.zeros(0, dtype=np.int64)

    if np.any(np.char.find(np.char.lower(text), "e") >= 0):
        raise ValueError("scientific notation is not supported; expected plain decimals")
    # Leading sign only: a "-" elsewhere belongs to an exponent, caught above.
    if np.any(np.char.startswith(np.char.strip(text), "-")):
        raise ValueError("negative price or quantity in the archive; refusing to continue")
    if np.any(np.char.count(text, ".") > 1):
        raise ValueError("value with more than one decimal point")

    whole, _, frac = (np.char.partition(text, ".")[:, i] for i in range(3))

    # The archive pads every decimal to 8 places ("94591.78000000"), so only
    # *significant* digits count against the scale. Trailing zeros carry no
    # information and must not trigger a precision error.
    significant = np.char.rstrip(frac, "0")
    if np.any(np.char.str_len(significant) > digits):
        worst = text[np.argmax(np.char.str_len(significant))]
        raise ValueError(
            f"value {worst!r} carries more precision than the {digits}-digit scale "
            f"{scale} can represent exactly; change the scale rather than truncating. "
            "This symbol needs per-symbol scales (see contract.py, D-003)."
        )

    whole = np.where(whole == "", "0", whole)
    padded = np.char.ljust(significant, digits, "0")
    padded = np.where(padded == "", "0" * digits, padded)
    return whole.astype(np.int64) * scale + padded.astype(np.int64)


def detect_timestamp_unit(raw: np.ndarray) -> str:
    """Return "us" or "ms" from the magnitude of the raw timestamps."""
    if raw.size == 0:
        raise ValueError("no records: cannot detect the timestamp unit")
    return "us" if int(raw.max()) > US_DETECTION_THRESHOLD else "ms"


def _booleans(values: np.ndarray) -> np.ndarray:
    """Parse the ``is_buyer_maker`` column strictly: no silent truthiness."""
    text = np.asarray(values, dtype=np.str_)
    is_true = np.isin(text, list(TRUE_STRINGS))
    is_false = np.isin(text, list(FALSE_STRINGS))
    unknown = ~(is_true | is_false)
    if np.any(unknown):
        raise ValueError(f"unparseable is_buyer_maker value {text[unknown][0]!r}")
    return is_true


def _read_rows(path: Path) -> tuple[list[list[str]], bytes]:
    """Return the CSV rows (header dropped) and the raw bytes for checksumming.

    Accepts a ZIP exactly as the archive ships it, or a bare CSV.
    """

    raw = path.read_bytes()
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError(f"expected exactly one CSV inside {path.name}, found {names}")
            text = archive.read(names[0]).decode("utf-8")
    else:
        text = raw.decode("utf-8")

    rows = [row for row in csv.reader(io.StringIO(text)) if row]
    if not rows:
        raise ValueError(f"{path.name} contains no records")
    if len(rows[0]) not in EXPECTED_COLUMNS:
        raise ValueError(
            f"{path.name} has {len(rows[0])} columns; expected one of {EXPECTED_COLUMNS}. "
            "Is this an aggTrades file?"
        )
    if not rows[0][COL_AGG_TRADE_ID].lstrip("-").isdigit():
        rows = rows[1:]  # header row (2025-onward layout)
    if not rows:
        raise ValueError(f"{path.name} contains no records after the header")
    return rows, raw


def _symbol_and_date(path: Path) -> tuple[str, str]:
    """Best-effort provenance from the archive's documented filename pattern."""
    parts = path.stem.split("-")
    if len(parts) >= 5 and parts[1] == "aggTrades":
        return parts[0], "-".join(parts[2:5])
    return "unknown", "unknown"


def load_agg_trades(path: str | Path) -> tuple[np.ndarray, Provenance]:
    """Parse a downloaded aggTrades ZIP/CSV into ``TRADE_DTYPE`` + provenance.

    Returns a timestamp-sorted array. Duplicate timestamps are preserved:
    several aggregate records legitimately share a millisecond.
    """
    path = Path(path)
    rows, raw = _read_rows(path)
    columns = list(zip(*rows, strict=False))

    # detect the time in the trade record
    raw_time = np.array(columns[COL_TRANSACT_TIME], dtype=np.int64)
    unit = detect_timestamp_unit(raw_time)
    timestamps = raw_time if unit == "us" else raw_time * 1_000
    out_of_range = (timestamps < MIN_PLAUSIBLE_US) | (timestamps > MAX_PLAUSIBLE_US)
    if np.any(out_of_range):
        raise ValueError(
            f"timestamp {int(timestamps[out_of_range][0])} is outside the plausible range "
            f"after assuming {unit}; the unit detection is wrong for {path.name}"
        )

    # set column quantities extracted from the csv
    trades = np.zeros(len(rows), dtype=TRADE_DTYPE)
    trades["timestamp"] = timestamps
    trades["price"] = decimal_strings_to_ticks(np.array(columns[COL_PRICE]), PRICE_SCALE)
    trades["quantity"] = decimal_strings_to_ticks(
        np.array(columns[COL_QUANTITY]), QUANTITY_SCALE
    )
    trades["buyer_is_maker"] = _booleans(np.array(columns[COL_IS_BUYER_MAKER]))
    trades["agg_trade_id"] = np.array(columns[COL_AGG_TRADE_ID], dtype=np.int64)

    was_resorted = bool(np.any(np.diff(trades["timestamp"]) < 0))
    if was_resorted:
        trades = trades[np.argsort(trades["timestamp"], kind="stable")]

    symbol, date = _symbol_and_date(path)
    return trades, Provenance(
        symbol=symbol,
        date=date,
        url=archive_url(symbol, date),
        sha256=hashlib.sha256(raw).hexdigest(),
        n_records=len(rows),
        timestamp_unit=unit,
        was_resorted=was_resorted,
    )


def download(symbol: str, date: str, out_dir: str | Path = "data") -> Path:
    """Fetch one symbol-day ZIP and verify it against the published CHECKSUM.

    Network access lives here and nowhere else, so the test suite stays
    offline and reproducible.
    """
    from urllib.request import urlopen

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    url = archive_url(symbol, date)
    target = out_dir / Path(url).name

    if not target.exists():
        print(f"downloading {url}")
        with urlopen(url) as response, target.open("wb") as handle:  # noqa: S310
            handle.write(response.read())

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    try:
        with urlopen(f"{url}.CHECKSUM") as response:  # noqa: S310
            published = response.read().decode().split()[0]
    except OSError as exc:
        print(f"WARNING: could not fetch the published checksum ({exc}); local sha256 {digest}")
        return target
    if published != digest:
        target.unlink()
        raise ValueError(
            f"checksum mismatch for {target.name}: published {published}, got {digest}. "
            "The partial download was deleted."
        )
    print(f"checksum verified: {digest}")
    return target


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: download one symbol-day, verify checksum, report schema."""
    parser = argparse.ArgumentParser(prog="python -m tapeforge.ingest")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--date", default="2025-01-02")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--path", default=None, help="skip the download, read this file")
    args = parser.parse_args(argv)

    path = Path(args.path) if args.path else download(args.symbol, args.date, args.out_dir)
    trades, prov = load_agg_trades(path)

    span_s = (int(trades["timestamp"][-1]) - int(trades["timestamp"][0])) / US_PER_SECOND
    print(f"{prov.symbol} {prov.date}: {prov.n_records} aggregate-trade records")
    print(f"  sha256:    {prov.sha256}")
    print(f"  timestamps: {prov.timestamp_unit} in the file, normalized to us")
    print(f"  span:       {span_s:.1f} s, resorted: {prov.was_resorted}")
    print(f"  price:      {trades['price'].min() / PRICE_SCALE:.8f}"
          f" .. {trades['price'].max() / PRICE_SCALE:.8f}")
    print(f"  quantity:   {trades['quantity'].min() / QUANTITY_SCALE:.8f}"
          f" .. {trades['quantity'].max() / QUANTITY_SCALE:.8f}")
    print(f"  buyer_is_maker true: {int(trades['buyer_is_maker'].sum())}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
