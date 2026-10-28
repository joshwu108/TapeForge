"""Tests for tapeforge.execution.datasets: archive paths per market, checksum-verified downloads
through a fake ``urlopen``, cache reuse, the download budget, failure reporting and the manifest.

No test touches the network: ``_fake_network`` serves the committed five-record
fixture archive and answers HEAD requests with its size.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from tapeforge.execution import datasets
from tapeforge.execution.datasets import (
    DATASET_MANIFEST_SCHEMA,
    ArchiveSpec,
    DatasetError,
    DownloadBudget,
    ensure_archive,
    ensure_archives,
    write_manifest,
)

FIXTURE_ZIP = Path(__file__).parent.parent / "fixtures" / "BTCUSDT-aggTrades-2025-01-02.zip"
PAYLOAD = FIXTURE_ZIP.read_bytes()
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
SPEC = ArchiveSpec("futures-um", "BTCUSDT", "2025-01-02")


class _Response:
    """Minimal stand-in for an ``http.client.HTTPResponse``."""

    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        self._stream = io.BytesIO(body)
        self.headers = headers or {}

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


def _fake_network(monkeypatch: pytest.MonkeyPatch, *, payload: bytes = PAYLOAD,
                  checksum: str = DIGEST, content_length: int | None = None,
                  fail_checksum: bool = False, calls: list[str] | None = None) -> None:
    """Install a ``urlopen`` that serves the fixture, its checksum and HEAD size."""
    length = len(payload) if content_length is None else content_length

    @contextlib.contextmanager
    def fake_urlopen(request: Any):  # noqa: ANN202
        url = request if isinstance(request, str) else request.full_url
        method = "GET" if isinstance(request, str) else request.get_method()
        if calls is not None:
            calls.append(f"{method} {url}")
        if url.endswith(".CHECKSUM"):
            if fail_checksum:
                raise OSError("checksum unavailable")
            yield _Response(f"{checksum}  {SPEC.filename}\n".encode())
        elif method == "HEAD":
            yield _Response(b"", {"Content-Length": str(length)})
        else:
            yield _Response(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a ``urlopen`` that fails every request."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the network must not be used")

    monkeypatch.setattr("urllib.request.urlopen", refuse)


# --- specs and budget ------------------------------------------------------------------


def test_archive_spec_names_market_url_and_local_path() -> None:
    assert SPEC.filename == "BTCUSDT-aggTrades-2025-01-02.zip"
    assert SPEC.url == (
        "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/"
        "BTCUSDT-aggTrades-2025-01-02.zip"
    )
    assert SPEC.checksum_url == SPEC.url + ".CHECKSUM"
    assert SPEC.local_path("data") == Path(
        "data/futures-um/aggTrades/BTCUSDT-aggTrades-2025-01-02.zip"
    )
    spot = ArchiveSpec("spot", "BTCUSDT", "2025-01-02")
    assert "/data/spot/daily/" in spot.url and spot.local_path("d").parts[1] == "spot"


def test_archive_spec_rejects_unknown_market_bad_symbol_and_bad_date() -> None:
    with pytest.raises(ValueError, match="unknown market"):
        ArchiveSpec("futures-cm", "BTCUSDT", "2025-01-02")
    with pytest.raises(ValueError, match="alphanumeric"):
        ArchiveSpec("spot", "BTC/USDT", "2025-01-02")
    with pytest.raises(ValueError):
        ArchiveSpec("spot", "BTCUSDT", "2025-13-02")


def test_budget_counts_archives_and_bytes_including_every_attempt() -> None:
    budget = DownloadBudget(max_archives=2, max_bytes=100)
    budget.reserve(SPEC, 60)
    assert budget.archives_started == 1 and budget.bytes_remaining == 40
    with pytest.raises(DatasetError, match="exceed the remaining budget"):
        budget.reserve(SPEC, 41)
    budget.reserve(SPEC, 40)
    with pytest.raises(DatasetError, match="2 archive"):
        budget.reserve(SPEC, 0)
    with pytest.raises(ValueError):
        DownloadBudget(max_archives=-1)


# --- acquisition --------------------------------------------------------------------------


def test_download_verifies_the_checksum_writes_sidecars_and_records_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _fake_network(monkeypatch, calls=calls)
    budget = DownloadBudget(max_archives=1, max_bytes=len(PAYLOAD))
    record = ensure_archive(SPEC, tmp_path, budget)
    target = SPEC.local_path(tmp_path)
    assert record.obtained == "downloaded" and record.sha256 == DIGEST
    assert record.n_bytes == len(PAYLOAD) and record.path == str(target)
    assert record.market == "futures-um" and "never redistributed" in record.license_note
    assert target.read_bytes() == PAYLOAD
    assert (target.parent / (target.name + ".CHECKSUM")).read_text().split()[0] == DIGEST
    meta = json.loads((target.parent / (target.name + ".meta.json")).read_text())
    assert meta["sha256"] == DIGEST and meta["url"] == SPEC.url
    assert not list(target.parent.glob("*.part"))
    # Checksum first, then the size, then the transfer; each exactly once.
    assert calls == [f"GET {SPEC.checksum_url}", f"HEAD {SPEC.url}", f"GET {SPEC.url}"]
    assert budget.archives_started == 1 and budget.bytes_reserved == len(PAYLOAD)


def test_a_verified_local_copy_is_reused_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_network(monkeypatch)
    first = ensure_archive(SPEC, tmp_path)
    _no_network(monkeypatch)
    second = ensure_archive(SPEC, tmp_path, DownloadBudget(0, 0), allow_download=False)
    assert second.obtained == "cached" and second.sha256 == first.sha256
    assert second.downloaded_utc == first.downloaded_utc


def test_a_wrong_digest_deletes_the_partial_file_and_reports_every_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_network(monkeypatch, checksum="0" * 64)
    budget = DownloadBudget(max_archives=5, max_bytes=10 * len(PAYLOAD))
    with pytest.raises(DatasetError, match="after 2 attempt") as info:
        ensure_archive(SPEC, tmp_path, budget, retries=1)
    assert "differs from the published" in str(info.value)
    assert not SPEC.local_path(tmp_path).exists()
    assert not list(SPEC.local_path(tmp_path).parent.glob("*"))
    assert budget.archives_started == 2 and budget.bytes_reserved == 2 * len(PAYLOAD)


def test_downloads_disabled_reports_the_missing_file_and_the_fetch_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_network(monkeypatch)
    with pytest.raises(DatasetError, match="downloads are disabled") as info:
        ensure_archive(SPEC, tmp_path, allow_download=False)
    assert "--date 2025-01-02" in str(info.value)


def test_the_budget_refuses_before_any_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _fake_network(monkeypatch, calls=calls)
    with pytest.raises(DatasetError, match="0 archive"):
        ensure_archive(SPEC, tmp_path, DownloadBudget(max_archives=0))
    assert not any(c == f"GET {SPEC.url}" for c in calls)
    calls.clear()
    with pytest.raises(DatasetError, match="exceed the remaining budget"):
        ensure_archive(SPEC, tmp_path, DownloadBudget(max_archives=1, max_bytes=len(PAYLOAD) - 1))
    assert not any(c == f"GET {SPEC.url}" for c in calls)
    assert not SPEC.local_path(tmp_path).exists()


def test_a_server_sending_more_than_announced_is_cut_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_network(monkeypatch, content_length=10)
    with pytest.raises(DatasetError, match="exceeded the 10-byte budget"):
        ensure_archive(SPEC, tmp_path, DownloadBudget(max_archives=1, max_bytes=10), retries=0)
    assert not list(SPEC.local_path(tmp_path).parent.glob("*"))


def test_an_unavailable_published_checksum_is_an_error_not_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_network(monkeypatch, fail_checksum=True)
    with pytest.raises(DatasetError, match="could not fetch"):
        ensure_archive(SPEC, tmp_path)
    assert not SPEC.local_path(tmp_path).exists()


def test_a_local_file_without_or_against_its_sidecar_is_refused_and_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_network(monkeypatch)
    target = SPEC.local_path(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(PAYLOAD)
    with pytest.raises(DatasetError, match="CHECKSUM is missing"):
        ensure_archive(SPEC, tmp_path)
    (target.parent / (target.name + ".CHECKSUM")).write_text(f"{'f' * 64}  {target.name}\n")
    with pytest.raises(DatasetError, match="corrupt or stale"):
        ensure_archive(SPEC, tmp_path)
    assert target.read_bytes() == PAYLOAD


def test_ensure_archives_refuses_to_mix_markets_or_symbols(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one market and one symbol"):
        ensure_archives([SPEC, ArchiveSpec("spot", "BTCUSDT", "2025-01-02")], tmp_path)
    with pytest.raises(ValueError, match="one market and one symbol"):
        ensure_archives([SPEC, ArchiveSpec("futures-um", "ETHUSDT", "2025-01-02")], tmp_path)


def test_manifest_lists_every_record_under_its_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_network(monkeypatch)
    records = ensure_archives([SPEC], tmp_path)
    path = write_manifest(records, tmp_path / "futures-um" / "aggTrades" / "manifest.json")
    manifest = json.loads(path.read_text())
    assert manifest["schema"] == DATASET_MANIFEST_SCHEMA
    assert manifest["archives"] == [records[0].to_json()]


def test_cli_verify_reports_cached_files_and_fetch_writes_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_network(monkeypatch)
    argv = ["--market", "futures-um", "--symbol", "BTCUSDT", "--date", "2025-01-02",
            "--data-dir", str(tmp_path)]
    assert datasets.main(["verify", *argv]) == 1   # Nothing local yet.
    assert "downloads are disabled" in capsys.readouterr().out
    assert datasets.main(["fetch", *argv]) == 0
    out = capsys.readouterr().out
    assert "downloaded" in out and "manifest:" in out and "1 archive(s)" in out
    _no_network(monkeypatch)
    assert datasets.main(["verify", *argv]) == 0
    assert "cached" in capsys.readouterr().out
