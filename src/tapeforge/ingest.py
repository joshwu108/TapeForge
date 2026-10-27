"""Public aggTrades loading, normalization and validation.

``load_agg_trades`` reads one symbol-day of Binance Vision ``aggTrades``
(the spot and USD-M futures archives share the 7-column layout; 2025-onward
files add a header), normalizes timestamps to ``contract.TIME_UNIT``,
converts price and quantity to exactly scaled integer ticks, parses
``is_buyer_maker`` strictly, sorts by timestamp when the file is not already
sorted, and records dataset provenance (sha256 of the raw file, record
count, timestamp unit).

Data contract:
    A row is an aggregate trade record, not an individual fill. Record counts
    throughout the project mean aggregate-trade records.

Invariants:
    Values that the fixed scales cannot represent exactly are rejected, never
    truncated. A timestamp unit that fails the plausibility check raises
    instead of shifting every record by a factor of 1,000.

Not covered:
    Network access exists only in ``download`` (spot archive URL). Every
    other function reads local files, so the test suite stays offline.

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

EXPECTED_COLUMNS: Final[tuple[int, ...]] = (7, 8)  # Column counts of the known layouts.
COL_AGG_TRADE_ID: Final[int] = 0
COL_PRICE: Final[int] = 1
COL_QUANTITY: Final[int] = 2
COL_TRANSACT_TIME: Final[int] = 5
COL_IS_BUYER_MAKER: Final[int] = 6

# A raw timestamp above this magnitude is in microseconds; otherwise milliseconds.
US_DETECTION_THRESHOLD: Final[int] = 10**14

# Plausibility window for a normalized timestamp, in microseconds:
# 2015-01-01 to 2100-01-01. A value outside it means the unit detection was wrong.
MIN_PLAUSIBLE_US: Final[int] = 1_420_070_400 * US_PER_SECOND
MAX_PLAUSIBLE_US: Final[int] = 4_102_444_800 * US_PER_SECOND

TRUE_STRINGS: Final[frozenset[str]] = frozenset({"true", "True", "TRUE", "1"})
FALSE_STRINGS: Final[frozenset[str]] = frozenset({"false", "False", "FALSE", "0"})


@dataclass(frozen=True)
class Provenance:
    """Origin of a loaded dataset, recorded next to every benchmark result.

    Attributes:
        symbol: Exchange symbol, or ``"unknown"`` when the filename does not
            follow the archive pattern.
        date: Trading day as ``YYYY-MM-DD``, or ``"unknown"``.
        url: Public archive URL for the symbol-day.
        sha256: Hex digest of the raw file bytes (ZIP or CSV, as read).
        n_records: Number of aggregate-trade records.
        timestamp_unit: ``"us"`` or ``"ms"``, as detected in the file.
        was_resorted: Whether the file was out of timestamp order and a
            stable sort was applied.
    """
    symbol: str
    date: str
    url: str
    sha256: str
    n_records: int
    timestamp_unit: str
    was_resorted: bool = False


def archive_url(symbol: str, date: str) -> str:
    """Return the public ZIP URL for one symbol-day."""
    return f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{date}.zip"


def _scale_digits(scale: int) -> int:
    """Return the number of decimal places a power-of-ten scale represents exactly.

    Raises:
        ValueError: ``scale`` is not a power of ten.
    """
    digits = len(str(scale)) - 1
    if 10**digits != scale:
        raise ValueError(f"scale {scale} is not a power of ten")
    return digits


def decimal_strings_to_ticks(values: np.ndarray, scale: int) -> np.ndarray:
    """Convert decimal strings to exactly scaled int64 ticks.

    Args:
        values: Array of plain decimal strings such as ``"94591.78000000"``.
        scale: Power-of-ten ticks per unit (``PRICE_SCALE`` or
            ``QUANTITY_SCALE``).

    Returns:
        int64 ticks of the same length as ``values``.

    Raises:
        ValueError: A value is negative, uses scientific notation, has more
            than one decimal point, or carries more significant decimals
            than ``scale`` can represent.
    """
    digits = _scale_digits(scale)
    text = np.asarray(values, dtype=np.str_)
    if text.size == 0:
        return np.zeros(0, dtype=np.int64)

    if np.any(np.char.find(np.char.lower(text), "e") >= 0):
        raise ValueError("scientific notation is not supported; expected plain decimals")
    # Only a leading sign is checked: a "-" elsewhere belongs to an exponent,
    # which the scientific-notation check above already rejected.
    if np.any(np.char.startswith(np.char.strip(text), "-")):
        raise ValueError("negative price or quantity in the archive; refusing to continue")
    if np.any(np.char.count(text, ".") > 1):
        raise ValueError("value with more than one decimal point")

    whole, _, frac = (np.char.partition(text, ".")[:, i] for i in range(3))

    # The archive pads every decimal to 8 places ("94591.78000000"), so only
    # significant digits count against the scale. Trailing zeros carry no
    # information and must not trigger a precision error.
    significant = np.char.rstrip(frac, "0")
    if np.any(np.char.str_len(significant) > digits):
        worst = text[np.argmax(np.char.str_len(significant))]
        raise ValueError(
            f"value {worst!r} carries more precision than the {digits}-digit scale "
            f"{scale} can represent exactly. Truncating would corrupt the integer sums; "
            "this symbol needs a larger scale (contract.PRICE_SCALE / QUANTITY_SCALE are "
            "fixed to BTCUSDT's 2 and 5 decimals; per-symbol scales are not implemented)."
        )

    whole = np.where(whole == "", "0", whole)
    padded = np.char.ljust(significant, digits, "0")
    padded = np.where(padded == "", "0" * digits, padded)
    return whole.astype(np.int64) * scale + padded.astype(np.int64)


def detect_timestamp_unit(raw: np.ndarray) -> str:
    """Return ``"us"`` or ``"ms"`` from the magnitude of the raw timestamps.

    Raises:
        ValueError: ``raw`` is empty.
    """
    if raw.size == 0:
        raise ValueError("no records: cannot detect the timestamp unit")
    return "us" if int(raw.max()) > US_DETECTION_THRESHOLD else "ms"


def _booleans(values: np.ndarray) -> np.ndarray:
    """Parse the ``is_buyer_maker`` column strictly; no silent truthiness.

    Raises:
        ValueError: A value is in neither ``TRUE_STRINGS`` nor ``FALSE_STRINGS``.
    """
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

    Raises:
        ValueError: The ZIP does not hold exactly one CSV, the file has no
            records, or the column count matches no known layout.
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
        rows = rows[1:]  # A non-numeric first id marks the 2025-onward header row.
    if not rows:
        raise ValueError(f"{path.name} contains no records after the header")
    return rows, raw


def _symbol_and_date(path: Path) -> tuple[str, str]:
    """Return best-effort provenance from the archive's documented filename pattern."""
    parts = path.stem.split("-")
    if len(parts) >= 5 and parts[1] == "aggTrades":
        return parts[0], "-".join(parts[2:5])
    return "unknown", "unknown"


def load_agg_trades(path: str | Path) -> tuple[np.ndarray, Provenance]:
    """Parse a downloaded aggTrades ZIP or CSV into ``TRADE_DTYPE`` records and provenance.

    The returned array is timestamp-sorted. Duplicate timestamps are
    preserved, since several aggregate records legitimately share a
    millisecond. A stable sort is applied only when the file is out of
    order, and ``Provenance.was_resorted`` records that it happened.

    Args:
        path: Local ZIP as shipped by the archive, or a bare CSV.

    Returns:
        The sorted ``TRADE_DTYPE`` array and its ``Provenance``.

    Raises:
        ValueError: The archive is unreadable, has the wrong column count or
            no records, holds an implausible timestamp after unit detection,
            a price or quantity the scales cannot represent, or an
            unparseable boolean.
    """
    path = Path(path)
    rows, raw = _read_rows(path)
    columns = list(zip(*rows, strict=False))

    # The archive's timestamp column switched from milliseconds to microseconds
    # in 2025. The unit is inferred from magnitude and then checked against a
    # plausible date range, so a wrong guess fails loudly instead of shifting
    # every record by a factor of 1,000.
    raw_time = np.array(columns[COL_TRANSACT_TIME], dtype=np.int64)
    unit = detect_timestamp_unit(raw_time)
    timestamps = raw_time if unit == "us" else raw_time * 1_000
    out_of_range = (timestamps < MIN_PLAUSIBLE_US) | (timestamps > MAX_PLAUSIBLE_US)
    if np.any(out_of_range):
        raise ValueError(
            f"timestamp {int(timestamps[out_of_range][0])} is outside the plausible range "
            f"after assuming {unit}; the unit detection is wrong for {path.name}"
        )

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

    An existing file is not re-downloaded but is still verified. When the
    checksum cannot be fetched, a warning is printed and the local digest is
    reported instead. This is the only function in the package that touches
    the network.

    Args:
        symbol: Exchange symbol, for example ``"BTCUSDT"``.
        date: Trading day as ``YYYY-MM-DD``.
        out_dir: Directory that receives the ZIP; created if missing.

    Returns:
        Path of the verified ZIP.

    Raises:
        ValueError: The published checksum does not match. The file is
            deleted before raising.
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
    """Download one symbol-day (or read ``--path``), verify it and print a schema summary.

    Returns:
        Process exit status; 0 on success.
    """
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
