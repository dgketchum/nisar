"""Tests for the Mesonet VWC QC machinery: the physical-bounds rule and its exact
endpoints, the consecutive-run context helper, and the apply path -- the failure CSV as
recovery record, the parquet edit, and the verification that refuses a partial edit.

The point of this script is that nothing vanishes silently: the CSV must hold the
original value of every value it blanks, the NaN count must rise by exactly the number
of flagged observations, in-range values must be untouched, and ``--no-apply`` must leave
the archive bit-for-bit as it was.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from qc_mesonet_vwc import (
    RULES,
    apply_nan,
    build,
    find_failures,
    load_long,
    out_of_range,
    run_lengths,
    station_context,
)


def test_out_of_range_flags_below_zero_and_above_one():
    df = pd.DataFrame({"value": [-0.696, 1.846, 2.147e9, 283.0]})
    assert list(out_of_range(df)) == [True, True, True, True]


def test_out_of_range_passes_the_endpoints_exactly():
    """0.0 and 1.0 m3/m3 are physically attainable and must survive."""
    df = pd.DataFrame({"value": [0.0, 1.0, 0.25]})
    assert list(out_of_range(df)) == [False, False, False]


def test_rules_registry_holds_the_single_implemented_rule():
    assert list(RULES) == ["out_of_range_0_1"]
    assert RULES["out_of_range_0_1"] is out_of_range


def test_run_lengths_separates_a_sustained_run_from_isolated_days():
    dates = pd.Series(
        pd.to_datetime(
            [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",  # 3-day block
                "2020-02-01",  # isolated
                "2020-03-01",
                "2020-03-02",  # 2-day block
            ]
        )
    )
    assert list(run_lengths(dates)) == [3, 3, 3, 1, 2, 2]


def test_run_lengths_is_order_independent():
    """The archive is not guaranteed sorted; each row still gets its own block's size."""
    dates = pd.Series(
        pd.to_datetime(["2020-03-02", "2020-01-01", "2020-02-01", "2020-01-02"])
    )
    assert list(run_lengths(dates)) == [1, 2, 1, 2]


def test_run_lengths_of_a_single_date():
    assert list(run_lengths(pd.Series(pd.to_datetime(["2020-01-01"])))) == [1]


ARCHIVE_ROWS = [
    # station, date, element, depth_cm, value, units
    ("blmcapit", "2020-06-01", "soil_vwc_0005", 5.0, 0.24, "m3/m3"),
    ("blmcapit", "2020-06-02", "soil_vwc_0005", 5.0, 283.0, "m3/m3"),  # scaling failure
    ("blmcapit", "2020-06-03", "soil_vwc_0005", 5.0, 1354.0, "m3/m3"),  # same run
    ("blmcapit", "2020-06-04", "soil_vwc_0005", 5.0, 0.26, "m3/m3"),
    ("arskeose", "2021-07-01", "soil_vwc_0091", 91.0, 2.147e9, "m3/m3"),  # int32 max
    ("ftbentcb", "2022-05-01", "soil_vwc_0005", 5.0, -0.696, "m3/m3"),
    ("ftbentcb", "2022-05-02", "soil_vwc_0005", 5.0, 0.0, "m3/m3"),  # endpoint, keep
    ("ftbentcb", "2022-05-03", "soil_vwc_0005", 5.0, 1.0, "m3/m3"),  # endpoint, keep
    ("ftbentcb", "2022-05-04", "soil_vwc_0005", 5.0, 1.846, "m3/m3"),  # one-off spike
    ("ftbentcb", "2022-05-01", "precip", np.nan, 25.0, "mm"),  # not VWC, untouched
]


def write_archive(tmp_path: Path, rows=ARCHIVE_ROWS, stations: bool = True) -> Path:
    """Write a synthetic daily-long parquet (and station table) under tmp_path."""
    df = pd.DataFrame(
        rows, columns=["station", "date", "element", "depth_cm", "value", "units"]
    )
    df["date"] = pd.to_datetime(df["date"])
    out = tmp_path / "mesonet/mt_mesonet_daily_long.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    if stations:
        pd.DataFrame(
            {
                "station": ["blmcapit", "arskeose", "ftbentcb"],
                "name": ["Capitol", "Keogh East", "Fort Benton"],
                "sub_network": ["HydroMet"] * 3,
                "county": ["Carter", "Rosebud", "Chouteau"],
                "date_installed": ["2017-01-01"] * 3,
                "elevation": [1200, 800, 900],
            }
        ).to_csv(tmp_path / "mesonet/mt_mesonet_stations.csv", index=False)
    return tmp_path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_long_raises_on_unexpected_vwc_units(tmp_path):
    rows = [("s1", "2020-06-01", "soil_vwc_0005", 5.0, 24.0, "percent")]
    with pytest.raises(ValueError, match="unexpected VWC units"):
        load_long(write_archive(tmp_path, rows=rows, stations=False))


def test_station_context_is_optional(tmp_path):
    write_archive(tmp_path, stations=False)
    assert station_context(tmp_path) is None


def test_find_failures_preserves_originals_and_adds_rule_and_run_length(tmp_path):
    data_dir = write_archive(tmp_path)
    _, df = load_long(data_dir)
    fail = find_failures(df, station_context(data_dir))

    assert len(fail) == 5
    assert set(fail["rule"]) == {"out_of_range_0_1"}
    assert set(fail["station"]) == {"blmcapit", "arskeose", "ftbentcb"}
    assert 2.147e9 in set(fail["value"])  # the original, not a NaN
    assert "sub_network" in fail.columns  # station context merged

    runs = fail.set_index(["station", "date"])["run_length"]
    assert runs[("blmcapit", pd.Timestamp("2020-06-02"))] == 2  # 2-day scaling run
    assert runs[("ftbentcb", pd.Timestamp("2022-05-01"))] == 1  # isolated
    assert runs[("ftbentcb", pd.Timestamp("2022-05-04"))] == 1


def test_find_failures_ignores_non_vwc_elements(tmp_path):
    data_dir = write_archive(tmp_path)
    _, df = load_long(data_dir)
    fail = find_failures(df, None)

    assert "precip" not in set(fail["element"])


def test_find_failures_returns_an_empty_typed_frame_when_all_values_pass(tmp_path):
    rows = [
        ("s1", "2020-06-01", "soil_vwc_0005", 5.0, 0.0, "m3/m3"),
        ("s1", "2020-06-02", "soil_vwc_0005", 5.0, 1.0, "m3/m3"),
    ]
    data_dir = write_archive(tmp_path, rows=rows, stations=False)
    _, df = load_long(data_dir)
    fail = find_failures(df, None)

    assert fail.empty
    assert "run_length" in fail.columns


def test_build_no_apply_leaves_the_parquet_byte_identical(tmp_path):
    data_dir = write_archive(tmp_path)
    path = data_dir / "mesonet/mt_mesonet_daily_long.parquet"
    before = digest(path)

    assert build(data_dir, apply=False, suspicious=False) == 0

    assert digest(path) == before
    csv = pd.read_csv(data_dir / "mesonet/mt_mesonet_vwc_qc_failures.csv")
    assert len(csv) == 5


def test_build_apply_nans_exactly_the_failing_values(tmp_path):
    data_dir = write_archive(tmp_path)
    path = data_dir / "mesonet/mt_mesonet_daily_long.parquet"
    before = pd.read_parquet(path)

    assert build(data_dir, apply=True, suspicious=False) == 0

    after = pd.read_parquet(path)
    csv = pd.read_csv(data_dir / "mesonet/mt_mesonet_vwc_qc_failures.csv")

    assert len(after) == len(before)  # rows kept, values blanked
    n_gained = int(after["value"].isna().sum()) - int(before["value"].isna().sum())
    assert n_gained == len(csv)

    vwc = after[after["element"].str.startswith("soil_vwc")]
    assert not ((vwc["value"] < 0.0) | (vwc["value"] > 1.0)).any()

    kept = after.set_index(["station", "date", "element"])["value"]
    assert kept[("ftbentcb", pd.Timestamp("2022-05-02"), "soil_vwc_0005")] == 0.0
    assert kept[("ftbentcb", pd.Timestamp("2022-05-03"), "soil_vwc_0005")] == 1.0
    assert kept[
        ("blmcapit", pd.Timestamp("2020-06-01"), "soil_vwc_0005")
    ] == pytest.approx(0.24)
    assert kept[("ftbentcb", pd.Timestamp("2022-05-01"), "precip")] == 25.0


def test_build_apply_recovery_csv_holds_every_blanked_original(tmp_path):
    data_dir = write_archive(tmp_path)
    path = data_dir / "mesonet/mt_mesonet_daily_long.parquet"
    before = pd.read_parquet(path)
    build(data_dir, apply=True, suspicious=False)
    after = pd.read_parquet(path)

    keys = ["station", "date", "element"]
    csv = pd.read_csv(data_dir / "mesonet/mt_mesonet_vwc_qc_failures.csv")
    csv["date"] = pd.to_datetime(csv["date"])
    blanked = after[after["value"].isna()].merge(before, on=keys, suffixes=("_new", ""))

    recovered = csv.set_index(keys)["value"]
    for row in blanked.itertuples():
        original = recovered[(row.station, row.date, row.element)]
        assert original == pytest.approx(row.value)


def test_build_apply_is_idempotent(tmp_path):
    """A second pass finds nothing to do and does not touch the parquet again."""
    data_dir = write_archive(tmp_path)
    path = data_dir / "mesonet/mt_mesonet_daily_long.parquet"
    build(data_dir, apply=True, suspicious=False)
    after_first = digest(path)

    assert build(data_dir, apply=True, suspicious=False) == 0

    assert digest(path) == after_first
    csv = pd.read_csv(data_dir / "mesonet/mt_mesonet_vwc_qc_failures.csv")
    assert csv.empty


def test_apply_nan_raises_when_a_bad_value_survives_the_edit(tmp_path):
    """The verify step reads the parquet back from disk, so an incomplete failure set is
    caught after the write rather than passing silently."""
    data_dir = write_archive(tmp_path)
    path, df = load_long(data_dir)
    fail = find_failures(df, None)
    partial = fail[fail["station"] == "blmcapit"]

    with pytest.raises(ValueError, match="out-of-range soil_vwc values survived"):
        apply_nan(path, df, partial)


def test_build_reports_suspicious_without_modifying_anything(tmp_path, capsys):
    """A 40-day constant in-range run and a one-day in-range spike are reported, never
    flagged: the failure CSV stays empty and the parquet is untouched."""
    dates = pd.date_range("2020-06-01", periods=40, freq="D")
    rows = [
        ("stuck", d.strftime("%Y-%m-%d"), "soil_vwc_0005", 5.0, 0.31, "m3/m3")
        for d in dates
    ]
    spike = pd.date_range("2021-06-01", periods=3, freq="D")
    rows += [
        ("spiky", d.strftime("%Y-%m-%d"), "soil_vwc_0005", 5.0, v, "m3/m3")
        for d, v in zip(spike, [0.20, 0.75, 0.21], strict=True)
    ]
    data_dir = write_archive(tmp_path, rows=rows, stations=False)
    path = data_dir / "mesonet/mt_mesonet_daily_long.parquet"
    before = digest(path)

    assert build(data_dir, apply=True, suspicious=True) == 0

    out = capsys.readouterr().out
    assert "stuck" in out
    assert "spiky" in out
    assert "Nothing above was modified." in out

    assert digest(path) == before
    csv = pd.read_csv(data_dir / "mesonet/mt_mesonet_vwc_qc_failures.csv")
    assert csv.empty
