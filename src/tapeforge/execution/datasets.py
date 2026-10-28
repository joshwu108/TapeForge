"""Historical aggTrades acquisition for experiments: cache, checksums, budget and manifest.

An experiment names one market, one symbol and a list of UTC days.
``ensure_archive`` makes each day's daily archive available under
``<data_dir>/<market>/aggTrades/`` and returns an ``ArchiveRecord`` with the
file's sha256, size and origin, so an experiment manifest can state exactly
which bytes it read. Archives already present and verified against their
published checksum are reused; nothing is re-downloaded.

Acquisition rules:
    * A download happens only when ``allow_download`` is set and the
      ``DownloadBudget`` has room for one more archive of the size the server
      reports. Every attempt, including a retry after a failed transfer,
      counts against the byte budget.
    * The published ``.CHECKSUM`` is fetched first; a day without a published
      checksum is not acquired, because an unverified file cannot be cited.
    * The archive streams into a ``.part`` file beside its final location,
      is hashed while streaming, and is renamed into place only after the
      digest matches. A mismatch deletes the partial file and raises.
    * A local file that disagrees with its checksum sidecar is reported, not
      deleted and not reused: the user decides what to do with their data.

Markets:
    Spot and USD-M futures archives share the layout but are different
    instruments; the market is part of the local path and of every record, so
    one experiment cannot read both as one history.

Licensing:
    Binance Vision data is used under its own terms; downloaded files stay
    under the Git-ignored ``data/`` directory and are never redistributed.
    ``LICENSE_NOTE`` is copied into every record.

Not covered:
    No parsing. ``days.day_from_archive`` reads the file this module
    delivers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

from tapeforge import ingest

__all__ = [
    "DATASET_MANIFEST_SCHEMA",
    "DEFAULT_MAX_DOWNLOAD_ARCHIVES",
    "DEFAULT_MAX_DOWNLOAD_BYTES",
    "LICENSE_NOTE",
    "ArchiveRecord",
    "ArchiveSpec",
    "DatasetError",
    "DownloadBudget",
    "ensure_archive",
    "ensure_archives",
    "sha256_of_file",
    "write_manifest",
]

DATASET_MANIFEST_SCHEMA: Final[str] = "tapeforge.execution.dataset-manifest/1"

# Bounds of one acquisition session: how many archives and how many bytes
# (every attempt counted) a run may fetch. A smoke experiment stays within
# these by construction; research configurations set their own limits.
DEFAULT_MAX_DOWNLOAD_ARCHIVES: Final[int] = 3
DEFAULT_MAX_DOWNLOAD_BYTES: Final[int] = 500 * 1024 * 1024

LICENSE_NOTE: Final[str] = (
    "Binance Vision public market data, used under Binance's terms of use; cached locally "
    "under the Git-ignored data/ directory and never redistributed"
)

_CHUNK_BYTES: Final[int] = 1 << 20  # Streaming read size for downloads.
_HEX_DIGEST: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


class DatasetError(RuntimeError):
    """A day could not be acquired or verified; the message names the file, URL and reason."""


@dataclass(frozen=True)
class ArchiveSpec:
    """One daily aggTrades archive: where it lives on the server and under ``data/``.

    Attributes:
        market: Key of ``ingest.MARKET_BASE_URLS`` (``"spot"`` or
            ``"futures-um"``).
        symbol: Exchange symbol, for example ``"BTCUSDT"``.
        date: UTC day, ``YYYY-MM-DD``.
    """

    market: str
    symbol: str
    date: str

    def __post_init__(self) -> None:
        """Validate the market, the symbol and the date.

        Raises:
            ValueError: Unknown market, an empty or non-alphanumeric symbol,
                or a date that is not an ISO calendar date.
        """
        if self.market not in ingest.MARKET_BASE_URLS:
            raise ValueError(
                f"unknown market {self.market!r}; expected one of {sorted(ingest.MARKET_BASE_URLS)}"
            )
        if not self.symbol or not self.symbol.isalnum():
            raise ValueError(f"symbol must be alphanumeric, got {self.symbol!r}")
        date.fromisoformat(self.date)

    @property
    def filename(self) -> str:
        """Return the archive's file name, ``SYMBOL-aggTrades-YYYY-MM-DD.zip``."""
        return f"{self.symbol}-aggTrades-{self.date}.zip"

    @property
    def url(self) -> str:
        """Return the public archive URL."""
        return ingest.archive_url(self.symbol, self.date, self.market)

    @property
    def checksum_url(self) -> str:
        """Return the URL of the published sha256 sidecar."""
        return self.url + ".CHECKSUM"

    def local_path(self, data_dir: str | Path) -> Path:
        """Return ``<data_dir>/<market>/aggTrades/<filename>``."""
        return Path(data_dir) / self.market / "aggTrades" / self.filename


@dataclass
class DownloadBudget:
    """Mutable tally of what an acquisition session may still fetch.

    Attributes:
        max_archives: Archives the session may start downloading.
        max_bytes: Bytes the session may transfer, every attempt included.
        archives_started: Downloads started so far.
        bytes_reserved: Bytes reserved so far; a reservation is made from the
            server's ``Content-Length`` before the transfer starts.
    """

    max_archives: int = DEFAULT_MAX_DOWNLOAD_ARCHIVES
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES
    archives_started: int = 0
    bytes_reserved: int = 0

    def __post_init__(self) -> None:
        """Reject negative limits (``ValueError``)."""
        if self.max_archives < 0 or self.max_bytes < 0:
            raise ValueError("download limits must be non-negative")

    @property
    def bytes_remaining(self) -> int:
        """Return how many bytes may still be transferred."""
        return self.max_bytes - self.bytes_reserved

    def reserve(self, spec: ArchiveSpec, n_bytes: int) -> None:
        """Charge one archive of ``n_bytes`` to the budget.

        Raises:
            DatasetError: The archive or byte limit would be exceeded; the
                message carries the limits and what was already used.
        """
        if self.archives_started + 1 > self.max_archives:
            raise DatasetError(
                f"refusing to download {spec.filename}: the budget allows "
                f"{self.max_archives} archive(s) per run and {self.archives_started} "
                "were already started"
            )
        if n_bytes > self.bytes_remaining:
            raise DatasetError(
                f"refusing to download {spec.filename}: {n_bytes} bytes would exceed the "
                f"remaining budget of {self.bytes_remaining} bytes "
                f"({self.bytes_reserved} of {self.max_bytes} already reserved)"
            )
        self.archives_started += 1
        self.bytes_reserved += n_bytes

    def to_json(self) -> dict[str, Any]:
        """Return the budget and its usage as plain values."""
        return asdict(self)


@dataclass(frozen=True)
class ArchiveRecord:
    """Provenance of one archive an experiment reads.

    Attributes:
        market: As in ``ArchiveSpec``.
        symbol: As in ``ArchiveSpec``.
        date: As in ``ArchiveSpec``.
        path: Local path of the verified ZIP.
        url: Public URL the file came from.
        sha256: Hex digest of the file bytes; equals the published checksum.
        n_bytes: File size.
        obtained: ``"cached"`` when a verified local copy was reused,
            ``"downloaded"`` when this session fetched it.
        verified_utc: When the digest was checked, ISO 8601.
        downloaded_utc: When the file was downloaded, if this session did
            it; otherwise the sidecar's value or ``None``.
        license_note: ``LICENSE_NOTE``.
    """

    market: str
    symbol: str
    date: str
    path: str
    url: str
    sha256: str
    n_bytes: int
    obtained: str
    verified_utc: str
    downloaded_utc: str | None = None
    license_note: str = field(default=LICENSE_NOTE)

    def to_json(self) -> dict[str, Any]:
        """Return a plain dict for manifests."""
        return asdict(self)


def sha256_of_file(path: str | Path) -> str:
    """Return the hex sha256 of a file, streamed so large archives do not load into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_checksum_text(text: str, source: str) -> str:
    """Return the digest in a ``sha256sum``-style line (``<hex>  <name>``).

    Raises:
        DatasetError: The text does not start with a 64-character hex digest.
    """
    token = text.strip().split()[0].lower() if text.strip() else ""
    if not _HEX_DIGEST.match(token):
        raise DatasetError(f"{source} does not contain a sha256 digest: {text.strip()[:80]!r}")
    return token


def _now_utc() -> str:
    """Return the current UTC time as an ISO 8601 string with second precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sidecar_paths(target: Path) -> tuple[Path, Path]:
    """Return the checksum sidecar and the download metadata sidecar of ``target``."""
    return target.with_name(target.name + ".CHECKSUM"), target.with_name(target.name + ".meta.json")


def _verify_local(spec: ArchiveSpec, target: Path) -> ArchiveRecord | None:
    """Return a record for a verified local copy, or ``None`` when no file exists.

    Raises:
        DatasetError: The file exists without a checksum sidecar, or its
            digest differs from the sidecar; the file is left untouched.
    """
    if not target.exists():
        return None
    checksum_path, meta_path = _sidecar_paths(target)
    if not checksum_path.exists():
        raise DatasetError(
            f"{target} exists but {checksum_path.name} is missing beside it; the file cannot "
            "be verified. Fetch the published checksum (python -m tapeforge.execution.datasets "
            f"fetch --market {spec.market} --symbol {spec.symbol} --date {spec.date}) or "
            "remove the file"
        )
    expected = _parse_checksum_text(checksum_path.read_text(), str(checksum_path))
    actual = sha256_of_file(target)
    if actual != expected:
        raise DatasetError(
            f"{target} has sha256 {actual} but {checksum_path.name} says {expected}; "
            "the local copy is corrupt or stale. It was not deleted and will not be used"
        )
    downloaded: str | None = None
    if meta_path.exists():
        try:
            downloaded = json.loads(meta_path.read_text()).get("downloaded_utc")
        except (OSError, ValueError):
            downloaded = None
    return ArchiveRecord(
        market=spec.market,
        symbol=spec.symbol,
        date=spec.date,
        path=str(target),
        url=spec.url,
        sha256=actual,
        n_bytes=target.stat().st_size,
        obtained="cached",
        verified_utc=_now_utc(),
        downloaded_utc=downloaded,
    )


def _fetch_text(url: str) -> str:
    """Return the body of a small text resource.

    Raises:
        DatasetError: The request failed; the message carries the URL and
            the underlying error.
    """
    from urllib.request import urlopen

    try:
        with urlopen(url) as response:  # noqa: S310 - fixed https host
            return str(response.read().decode("utf-8", errors="replace"))
    except OSError as exc:
        raise DatasetError(f"could not fetch {url}: {exc}") from exc


def _content_length(url: str) -> int | None:
    """Return the ``Content-Length`` a HEAD request reports, or ``None`` when unknown.

    Raises:
        DatasetError: The request failed.
    """
    from urllib.request import Request, urlopen

    try:
        with urlopen(Request(url, method="HEAD")) as response:  # noqa: S310 - fixed host
            header = response.headers.get("Content-Length")
    except OSError as exc:
        raise DatasetError(f"could not query the size of {url}: {exc}") from exc
    try:
        return int(header) if header is not None else None
    except ValueError:
        return None


def _stream_download(url: str, part: Path, limit: int) -> tuple[str, int]:
    """Stream ``url`` into ``part``, hashing as it goes, and return ``(sha256, n_bytes)``.

    Raises:
        DatasetError: The transfer failed or exceeded ``limit`` bytes; the
            partial file is deleted.
    """
    from urllib.request import urlopen

    digest = hashlib.sha256()
    received = 0
    try:
        with urlopen(url) as response, part.open("wb") as handle:  # noqa: S310 - fixed host
            for chunk in iter(lambda: response.read(_CHUNK_BYTES), b""):
                received += len(chunk)
                if received > limit:
                    raise DatasetError(
                        f"transfer of {url} exceeded the {limit}-byte budget after "
                        f"{received} bytes; the partial file was deleted"
                    )
                digest.update(chunk)
                handle.write(chunk)
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise DatasetError(f"transfer of {url} failed after {received} bytes: {exc}") from exc
    except DatasetError:
        part.unlink(missing_ok=True)
        raise
    return digest.hexdigest(), received


def ensure_archive(
    spec: ArchiveSpec,
    data_dir: str | Path,
    budget: DownloadBudget | None = None,
    allow_download: bool = True,
    retries: int = 1,
) -> ArchiveRecord:
    """Return a verified local copy of ``spec``, downloading it within ``budget`` if needed.

    Args:
        spec: Which archive.
        data_dir: Root of the local cache; the file lands under
            ``<data_dir>/<market>/aggTrades/``.
        budget: Limits for this session; ``None`` uses the defaults. Shared
            between calls to bound a whole run.
        allow_download: When false, a missing file is an error rather than a
            download.
        retries: Additional transfer attempts after a failed or corrupt one.
            Each attempt reserves the archive's size again.

    Returns:
        The record of the verified file.

    Raises:
        DatasetError: The local copy fails verification, downloads are
            disabled, the budget is exhausted, the checksum or the archive
            cannot be fetched, or every attempt produced a wrong digest.
    """
    target = spec.local_path(data_dir)
    cached = _verify_local(spec, target)
    if cached is not None:
        return cached
    if not allow_download:
        raise DatasetError(
            f"{target} is not present and downloads are disabled for this run; fetch it with "
            f"python -m tapeforge.execution.datasets fetch --market {spec.market} "
            f"--symbol {spec.symbol} --date {spec.date}"
        )
    budget = DownloadBudget() if budget is None else budget
    expected = _parse_checksum_text(_fetch_text(spec.checksum_url), spec.checksum_url)
    size = _content_length(spec.url)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    failures: list[str] = []
    for _attempt in range(1 + max(0, retries)):
        # The reservation precedes the transfer so a server that lies about
        # the size cannot exceed the budget: the stream itself is capped at
        # what remains after reserving the announced size.
        announced = size if size is not None else budget.bytes_remaining
        budget.reserve(spec, announced)
        limit = announced if size is not None else budget.bytes_remaining + announced
        try:
            actual, received = _stream_download(spec.url, part, limit)
        except DatasetError as exc:
            failures.append(str(exc))
            continue
        if actual != expected:
            part.unlink(missing_ok=True)
            failures.append(
                f"sha256 {actual} of the downloaded {spec.filename} differs from the published "
                f"{expected} ({received} bytes received); the partial file was deleted"
            )
            continue
        os.replace(part, target)
        checksum_path, meta_path = _sidecar_paths(target)
        now = _now_utc()
        checksum_path.write_text(f"{expected}  {spec.filename}\n")
        meta_path.write_text(
            json.dumps(
                {
                    "url": spec.url,
                    "downloaded_utc": now,
                    "content_length": size,
                    "n_bytes": received,
                    "sha256": actual,
                    "license_note": LICENSE_NOTE,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return ArchiveRecord(
            market=spec.market,
            symbol=spec.symbol,
            date=spec.date,
            path=str(target),
            url=spec.url,
            sha256=actual,
            n_bytes=received,
            obtained="downloaded",
            verified_utc=now,
            downloaded_utc=now,
        )
    raise DatasetError(
        f"could not acquire {spec.filename} after {len(failures)} attempt(s):\n  "
        + "\n  ".join(failures)
    )


def ensure_archives(
    specs: Sequence[ArchiveSpec],
    data_dir: str | Path,
    budget: DownloadBudget | None = None,
    allow_download: bool = True,
) -> tuple[ArchiveRecord, ...]:
    """Acquire several archives of one market with a shared budget, in order.

    Raises:
        ValueError: The specs name more than one market or symbol.
        DatasetError: As ``ensure_archive``; nothing after the failing day is
            attempted.
    """
    markets = {s.market for s in specs}
    symbols = {s.symbol for s in specs}
    if len(markets) > 1 or len(symbols) > 1:
        raise ValueError(
            f"one dataset covers one market and one symbol; got markets {sorted(markets)} and "
            f"symbols {sorted(symbols)}"
        )
    budget = DownloadBudget() if budget is None else budget
    return tuple(ensure_archive(spec, data_dir, budget, allow_download) for spec in specs)


def write_manifest(records: Sequence[ArchiveRecord], path: str | Path) -> Path:
    """Write the records as a JSON manifest (``DATASET_MANIFEST_SCHEMA``) and return its path."""
    payload = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "written_utc": _now_utc(),
        "archives": [r.to_json() for r in records],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out)
    return out


def main(argv: list[str] | None = None) -> int:
    """Acquire or verify daily archives from the command line.

    ``fetch`` acquires each ``--date`` within the budget and writes
    ``manifest.json`` beside the files; ``verify`` only checks local copies.

    Returns:
        0 on success, 1 when a day could not be acquired or verified.
    """
    parser = argparse.ArgumentParser(prog="python -m tapeforge.execution.datasets")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--market", required=True, choices=sorted(ingest.MARKET_BASE_URLS))
        command.add_argument("--symbol", required=True)
        command.add_argument("--date", action="append", dest="dates", required=True)
        command.add_argument("--data-dir", default="data")
        if name == "fetch":
            command.add_argument("--max-archives", type=int, default=DEFAULT_MAX_DOWNLOAD_ARCHIVES)
            command.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_DOWNLOAD_BYTES)
    args = parser.parse_args(argv)

    specs = [ArchiveSpec(args.market, args.symbol, d) for d in args.dates]
    budget = (
        DownloadBudget(args.max_archives, args.max_bytes) if args.command == "fetch"
        else DownloadBudget(0, 0)
    )
    try:
        records = ensure_archives(
            specs, args.data_dir, budget, allow_download=args.command == "fetch"
        )
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1
    for record in records:
        print(
            f"{record.obtained:10s} {record.path}  sha256={record.sha256}  "
            f"{record.n_bytes} bytes"
        )
    if args.command == "fetch":
        manifest = write_manifest(
            records, specs[0].local_path(args.data_dir).parent / "manifest.json"
        )
        print(f"manifest: {manifest}")
        print(
            f"budget: {budget.archives_started} archive(s), {budget.bytes_reserved} bytes reserved"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
