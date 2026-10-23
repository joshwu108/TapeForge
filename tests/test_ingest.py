"""Public aggTrades loading, normalization and validation.

No network access: every case runs against the committed fixtures in
tests/fixtures/, which are 5-row excerpts in the two layouts the archive has
shipped -- 2025-onward (header row, microsecond timestamps) and legacy
(no header, millisecond timestamps).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tapeforge import contract, ingest

FIXTURES = Path(__file__).parent / "fixtures"
CSV_US = FIXTURES / "aggTrades_2025_us_header.csv"
CSV_MS = FIXTURES / "aggTrades_2024_ms_noheader.csv"
ZIP_US = FIXTURES / "BTCUSDT-aggTrades-2025-01-02.zip"


# --- exact decimal -> scaled integer ------------------------------------

@pytest.mark.parametrize(
    ("text", "scale", "expected"),
    [
        # Archive formatting: everything is padded to 8 decimals, so the
        # trailing zeros must not count against the scale.
        ("94591.78000000", contract.PRICE_SCALE, 9_459_178),
        ("94591.79000000", contract.PRICE_SCALE, 9_459_179),
        ("100", contract.PRICE_SCALE, 100 * contract.PRICE_SCALE),
        ("0.00100000", contract.QUANTITY_SCALE, 100),
        ("0.00001000", contract.QUANTITY_SCALE, 1),
        ("3.00000000", contract.QUANTITY_SCALE, 3 * contract.QUANTITY_SCALE),
        ("18.89042000", contract.QUANTITY_SCALE, 1_889_042),
    ],
)
def test_decimal_strings_convert_exactly(text: str, scale: int, expected: int) -> None:
    """Integer string arithmetic, not a float round-trip: never a tick off."""
    got = ingest.decimal_strings_to_ticks(np.array([text]), scale)
    assert got.dtype == np.int64
    assert got[0] == expected


def test_precision_beyond_the_scale_is_rejected_not_truncated() -> None:
    with pytest.raises(ValueError, match="precision"):
        ingest.decimal_strings_to_ticks(np.array(["1.123"]), contract.PRICE_SCALE)


def test_trailing_zeros_are_not_precision() -> None:
    """"1.10000000" is two significant decimals, not eight."""
    got = ingest.decimal_strings_to_ticks(np.array(["1.10000000"]), contract.PRICE_SCALE)
    assert got[0] == 110


def test_a_symbol_the_scales_cannot_represent_fails_loudly(tmp_path: Path) -> None:
    """The fixed-scale limitation is enforced by refusing, never by truncating.

    SHIBUSDT-style prices carry 8 significant decimals. The chosen BTCUSDT
    scales cannot hold them, and the error says what to do about it.
    """
    with pytest.raises(ValueError, match="per-symbol scales"):
        ingest.load_agg_trades(FIXTURES / "aggTrades_overprecise_price.csv")


def test_negative_price_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        ingest.decimal_strings_to_ticks(np.array(["-1.5"]), contract.PRICE_SCALE)


# --- timestamp unit detection -------------------------------------------

def test_microsecond_file_is_detected_and_left_in_microseconds() -> None:
    trades, prov = ingest.load_agg_trades(CSV_US)
    assert prov.timestamp_unit == "us"
    assert trades["timestamp"][0] == 1_735_776_000_100_000


def test_millisecond_file_is_detected_and_scaled_up() -> None:
    """Legacy archives are ms; everything downstream assumes us."""
    trades, prov = ingest.load_agg_trades(CSV_MS)
    assert prov.timestamp_unit == "ms"
    assert trades["timestamp"][0] == 1_704_153_600_100 * 1_000


def test_both_layouts_agree_on_the_relative_timeline() -> None:
    """Same 5 records, two eras: the within-day offsets must match."""
    us, _ = ingest.load_agg_trades(CSV_US)
    ms, _ = ingest.load_agg_trades(CSV_MS)
    np.testing.assert_array_equal(
        us["timestamp"] - us["timestamp"][0], ms["timestamp"] - ms["timestamp"][0]
    )


# --- schema and values ---------------------------------------------------

def test_output_matches_the_trade_dtype() -> None:
    trades, _ = ingest.load_agg_trades(CSV_US)
    assert trades.dtype == contract.TRADE_DTYPE
    assert trades.size == 5


def test_prices_and_quantities_are_scaled_integers() -> None:
    trades, _ = ingest.load_agg_trades(CSV_US)
    assert trades["price"][0] == 9_459_178          # 94591.78
    assert trades["quantity"][0] == 2 * contract.QUANTITY_SCALE
    assert trades["quantity"][3] == 100            # 0.001 units at 1e5


def test_buyer_is_maker_parses_as_bool_both_ways() -> None:
    trades, _ = ingest.load_agg_trades(CSV_US)
    assert trades["buyer_is_maker"].dtype == np.bool_
    assert not bool(trades["buyer_is_maker"][0])
    assert bool(trades["buyer_is_maker"][1])


def test_agg_trade_ids_are_preserved_for_deterministic_replay() -> None:
    trades, _ = ingest.load_agg_trades(CSV_US)
    np.testing.assert_array_equal(trades["agg_trade_id"], [1001, 1002, 1003, 1004, 1005])


def test_duplicate_timestamps_are_preserved_not_deduplicated() -> None:
    """Two records sharing a microsecond is normal; dropping one is a bug."""
    trades, _ = ingest.load_agg_trades(CSV_US)
    assert np.sum(np.diff(trades["timestamp"]) == 0) == 2


def test_timestamps_come_out_sorted() -> None:
    trades, _ = ingest.load_agg_trades(CSV_US)
    assert np.all(np.diff(trades["timestamp"]) >= 0)


# --- provenance ----------------------------------------------------------

def test_provenance_records_checksum_and_count() -> None:
    _, prov = ingest.load_agg_trades(ZIP_US)
    assert len(prov.sha256) == 64
    assert prov.n_records == 5
    assert prov.symbol == "BTCUSDT"
    assert prov.date == "2025-01-02"


def test_checksum_is_stable_across_reads() -> None:
    assert ingest.load_agg_trades(ZIP_US)[1].sha256 == ingest.load_agg_trades(ZIP_US)[1].sha256


def test_zip_and_csv_paths_produce_identical_records() -> None:
    from_zip, _ = ingest.load_agg_trades(ZIP_US)
    from_csv, _ = ingest.load_agg_trades(CSV_US)
    np.testing.assert_array_equal(from_zip, from_csv)


# --- validation ----------------------------------------------------------

def test_wrong_column_count_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("1,2,3\n4,5,6\n")
    with pytest.raises(ValueError, match="column"):
        ingest.load_agg_trades(bad)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ValueError, match="no records"):
        ingest.load_agg_trades(empty)


def test_unsorted_input_is_sorted_and_flagged(tmp_path: Path) -> None:
    """The archive is sorted; if it is not, say so rather than trusting it."""
    rows = CSV_US.read_text().splitlines()
    shuffled = tmp_path / "shuffled.csv"
    shuffled.write_text("\n".join([rows[0], rows[3], rows[1], rows[2], rows[4], rows[5]]) + "\n")
    trades, prov = ingest.load_agg_trades(shuffled)
    assert np.all(np.diff(trades["timestamp"]) >= 0)
    assert prov.was_resorted is True


def test_url_is_the_documented_public_path() -> None:
    assert ingest.archive_url("BTCUSDT", "2025-01-02") == (
        "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/"
        "BTCUSDT-aggTrades-2025-01-02.zip"
    )


# --- more validation -----------------------------------------------------

def test_scientific_notation_is_rejected() -> None:
    with pytest.raises(ValueError, match="scientific notation"):
        ingest.decimal_strings_to_ticks(np.array(["1e-8"]), contract.PRICE_SCALE)


def test_two_decimal_points_is_rejected() -> None:
    with pytest.raises(ValueError, match="decimal point"):
        ingest.decimal_strings_to_ticks(np.array(["1.2.3"]), contract.PRICE_SCALE)


def test_empty_input_converts_to_an_empty_int64_array() -> None:
    got = ingest.decimal_strings_to_ticks(np.array([], dtype=np.str_), contract.PRICE_SCALE)
    assert got.dtype == np.int64 and got.size == 0


def test_a_scale_that_is_not_a_power_of_ten_is_rejected() -> None:
    with pytest.raises(ValueError, match="power of ten"):
        ingest.decimal_strings_to_ticks(np.array(["1.0"]), 1234)


def test_leading_decimal_point_is_handled() -> None:
    assert ingest.decimal_strings_to_ticks(np.array([".5"]), contract.PRICE_SCALE)[0] == (
        contract.PRICE_SCALE // 2
    )


def test_real_archive_values_round_trip_exactly() -> None:
    """Rows copied verbatim from BTCUSDT 2025-01-02, the canonical day."""
    prices = np.array(["94591.78000000", "97839.50000000", "94392.00000000"])
    quantities = np.array(["0.00181000", "18.89042000", "0.00001000"])
    np.testing.assert_array_equal(
        ingest.decimal_strings_to_ticks(prices, contract.PRICE_SCALE),
        [9_459_178, 9_783_950, 9_439_200],
    )
    np.testing.assert_array_equal(
        ingest.decimal_strings_to_ticks(quantities, contract.QUANTITY_SCALE),
        [181, 1_889_042, 1],
    )


def test_unparseable_buyer_is_maker_is_rejected(tmp_path: Path) -> None:
    rows = CSV_US.read_text().splitlines()
    broken = rows[:2] + [rows[2].replace(",true,", ",maybe,")]
    bad = tmp_path / "bad_bool.csv"
    bad.write_text("\n".join(broken) + "\n")
    with pytest.raises(ValueError, match="unparseable is_buyer_maker"):
        ingest.load_agg_trades(bad)


def test_implausible_timestamp_reports_a_unit_detection_failure(tmp_path: Path) -> None:
    """A timestamp that is neither plausible ms nor us must not be guessed at."""
    header, first = CSV_US.read_text().splitlines()[:2]
    fields = first.split(",")
    fields[5] = "999"
    bad = tmp_path / "bad_time.csv"
    bad.write_text(header + "\n" + ",".join(fields) + "\n")
    with pytest.raises(ValueError, match="unit detection is wrong"):
        ingest.load_agg_trades(bad)


def test_header_only_file_is_rejected(tmp_path: Path) -> None:
    only = tmp_path / "header_only.csv"
    only.write_text(CSV_US.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="no records after the header"):
        ingest.load_agg_trades(only)


def test_zip_with_several_csvs_is_rejected(tmp_path: Path) -> None:
    import zipfile

    multi = tmp_path / "multi.zip"
    with zipfile.ZipFile(multi, "w") as archive:
        archive.writestr("a.csv", CSV_US.read_text())
        archive.writestr("b.csv", CSV_US.read_text())
    with pytest.raises(ValueError, match="exactly one CSV"):
        ingest.load_agg_trades(multi)


def test_unit_detection_needs_at_least_one_record() -> None:
    with pytest.raises(ValueError, match="no records"):
        ingest.detect_timestamp_unit(np.array([], dtype=np.int64))


def test_provenance_falls_back_when_the_filename_is_not_the_archive_pattern(
    tmp_path: Path,
) -> None:
    renamed = tmp_path / "whatever.csv"
    renamed.write_text(CSV_US.read_text())
    _, prov = ingest.load_agg_trades(renamed)
    assert prov.symbol == "unknown" and prov.date == "unknown"


# --- download safety (no real network) -----------------------------------

def test_download_deletes_the_file_on_a_checksum_mismatch(tmp_path, monkeypatch) -> None:
    """A corrupt download must not be left behind to be silently reused."""
    import contextlib
    import io as _io

    payload = b"not-the-real-archive"

    @contextlib.contextmanager
    def fake_urlopen(url: str):  # noqa: ANN202
        yield _io.BytesIO(b"0" * 64 + b"  filename" if url.endswith(".CHECKSUM") else payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ValueError, match="checksum mismatch"):
        ingest.download("BTCUSDT", "2025-01-02", tmp_path)
    assert not (tmp_path / "BTCUSDT-aggTrades-2025-01-02.zip").exists()


def test_download_accepts_a_matching_checksum(tmp_path, monkeypatch) -> None:
    import contextlib
    import hashlib
    import io as _io

    payload = ZIP_US.read_bytes()
    digest = hashlib.sha256(payload).hexdigest().encode()

    @contextlib.contextmanager
    def fake_urlopen(url: str):  # noqa: ANN202
        yield _io.BytesIO(digest + b"  filename" if url.endswith(".CHECKSUM") else payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    target = ingest.download("BTCUSDT", "2025-01-02", tmp_path)
    assert target.exists()
    trades, prov = ingest.load_agg_trades(target)
    assert trades.size == 5 and prov.n_records == 5


def test_cli_reports_the_schema_without_network(capsys, monkeypatch) -> None:
    assert ingest.main(["--path", str(ZIP_US)]) == 0
    printed = capsys.readouterr().out
    assert "aggregate-trade records" in printed
    assert "sha256" in printed
    assert "normalized to us" in printed
