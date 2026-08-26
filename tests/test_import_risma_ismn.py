"""Tests for the RISMA .stm parser: header/data parsing, flag handling, and the
N-way (not pairwise) replicate mean.

Waukomis context: swap-stress's ``ismn.py::_daily_series`` fell back to the all-data
mean on days with no good observation, and its multi-sensor merge folded replicates in
pairwise rather than N-way. ``import_risma_ismn.py`` was written to avoid both bugs;
these tests pin that behavior down.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from import_risma_ismn import (
    build_long,
    daily_sensor,
    flag_census,
    read_stm,
)


def write_stm(
    path: Path,
    station: str = "MB1",
    lat: float = 49.5,
    lon: float = -98.5,
    elev: float = 300.0,
    depth_from_m: float = 0.00,
    depth_to_m: float = 0.05,
    sensor: str = "Hydraprobe II (digital)",
    rows: list[tuple[str, str, float, str]] = (),
) -> Path:
    """Write one synthetic ISMN .stm file: header line + fixed 5-field data rows."""
    header = (
        f"RISMA {station} {station} {lat} {lon} {elev} "
        f"{depth_from_m:.2f} {depth_to_m:.2f} {sensor}\n"
    )
    lines = [header]
    for day, hhmm, value, flag in rows:
        lines.append(f"{day} {hhmm} {value} {flag} {flag}\n")
    path.write_text("".join(lines))
    return path


def test_read_stm_parses_header_fields(tmp_path):
    path = write_stm(
        tmp_path / "RISMA_MB1_sm_0.000000_0.050000_Hydraprobe_20250601_20250602.stm",
        rows=[("2025/06/01", "00:00", 0.250, "G")],
    )
    meta, df = read_stm(path)

    assert meta["station"] == "RISMA:MB1"
    assert meta["latitude"] == pytest.approx(49.5)
    assert meta["longitude"] == pytest.approx(-98.5)
    assert meta["depth_from_cm"] == pytest.approx(0.0)
    assert meta["depth_to_cm"] == pytest.approx(5.0)
    assert meta["sensor"] == "Hydraprobe II (digital)"
    assert list(df.columns) == ["datetime", "value", "ismn_flag"]
    assert df.loc[0, "value"] == pytest.approx(0.250)
    assert df.loc[0, "datetime"] == pd.Timestamp("2025-06-01 00:00")


def test_read_stm_parses_point_depth(tmp_path):
    """A point sensor (depth_from == depth_to) at 20 cm, not the 0-5 cm layer probe."""
    path = write_stm(
        tmp_path / "RISMA_MB1_sm_0.200000_0.200000_Hydraprobe_20250601_20250602.stm",
        depth_from_m=0.20,
        depth_to_m=0.20,
        rows=[("2025/06/01", "00:00", 0.300, "G")],
    )
    meta, _ = read_stm(path)
    assert meta["depth_from_cm"] == pytest.approx(20.0)
    assert meta["depth_to_cm"] == pytest.approx(20.0)


def test_daily_sensor_drops_non_good_flags(tmp_path):
    """Only G rows contribute; a day with nothing good yields no row at all."""
    path = write_stm(
        tmp_path / "RISMA_MB1_sm_0.000000_0.050000_Hydraprobe_20250601_20250603.stm",
        rows=[
            ("2025/06/01", "00:00", 0.10, "G"),
            ("2025/06/01", "12:00", 0.90, "C"),  # bad reading, must not pull the mean
            ("2025/06/02", "00:00", 0.20, "D01"),  # frozen soil, whole day dropped
        ],
    )
    _, df = read_stm(path)
    daily = daily_sensor(df)

    assert len(daily) == 1
    assert daily.loc[0, "date"] == pd.Timestamp("2025-06-01")
    assert daily.loc[0, "value"] == pytest.approx(0.10)
    assert daily.loc[0, "n_obs"] == 1


def test_flag_census_counts_by_category_not_full_string():
    """A combined flag like 'D01,D02' is counted under 'D', and only 'G' is kept."""
    df = pd.DataFrame(
        {
            "ismn_flag": ["G", "G", "D01,D02", "C02"],
            "value": [0.1, 0.2, 0.3, 0.4],
        }
    )
    census = flag_census([df]).set_index("flag_category")

    assert census.loc["G", "n_hourly"] == 2
    assert census.loc["G", "kept"]
    assert census.loc["D", "n_hourly"] == 1
    assert not census.loc["D", "kept"]
    assert census.loc["C", "n_hourly"] == 1
    assert census["pct"].sum() == pytest.approx(100.0)


def test_build_long_uses_true_nway_mean_not_pairwise(tmp_path):
    """Three replicate sensors on the same station/date/depth average as one N-way
    mean. A pairwise fold-in (the other half of the Waukomis bug) would instead give
    ((0.1 + 0.2) / 2 + 0.6) / 2 == 0.375, not the true mean of 0.3.
    """
    paths = [
        write_stm(
            tmp_path / f"RISMA_MB1_sm_0.000000_0.050000_Sensor{letter}_20250601.stm",
            sensor=f"Hydraprobe {letter}",
            rows=[("2025/06/01", "00:00", value, "G")],
        )
        for letter, value in zip("ABC", [0.1, 0.2, 0.6], strict=True)
    ]

    long, _, _ = build_long(paths)

    assert len(long) == 1
    row = long.iloc[0]
    assert row["value"] == pytest.approx(0.3)
    assert row["value"] != pytest.approx(0.375)  # would be the pairwise-fold result
    assert row["n_sensors"] == 3
    assert row["n_obs"] == 3
    assert row["depth_cm"] == pytest.approx(2.5)
    assert row["depth_from_cm"] == pytest.approx(0.0)
    assert row["depth_to_cm"] == pytest.approx(5.0)


def test_build_long_raises_when_nothing_is_good(tmp_path):
    path = write_stm(
        tmp_path / "RISMA_MB1_sm_0.000000_0.050000_Hydraprobe_20250601.stm",
        rows=[("2025/06/01", "00:00", 0.10, "C")],
    )
    with pytest.raises(RuntimeError, match="no good-flagged"):
        build_long([path])
