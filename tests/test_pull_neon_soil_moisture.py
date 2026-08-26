"""Tests for the pure, network-free pieces of the NEON soil-moisture puller: sensor
depth resolution (including the duplicate-window guard), filename parsing, and the
half-hourly -> daily QC-aware aggregation.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pull_neon_soil_moisture import daily_long, depth_lookup, parse_hor_ver


def test_parse_hor_ver_reads_positions_after_product_code():
    name = (
        "NEON.D10.STER.DP1.00094.001.002.505.030.SWS_30_minute."
        "2026-06.basic.20260801T000000Z.csv"
    )
    assert parse_hor_ver(name) == (2, 505)


def make_depths(rows):
    df = pd.DataFrame(rows, columns=["site", "hor", "ver", "depth_cm", "start", "end"])
    df["start"] = pd.to_datetime(df["start"], utc=True)
    df["end"] = pd.to_datetime(df["end"], utc=True)
    return df


def test_depth_lookup_selects_window_valid_for_the_month():
    depths = make_depths(
        [
            ["STER", 1, 501, 6.0, "2017-01-01", pd.NaT],
            ["STER", 1, 502, 16.0, "2017-01-01", "2026-01-01"],
            ["STER", 1, 502, 20.0, "2026-01-01", pd.NaT],
        ]
    )
    lut = depth_lookup(depths, "STER", "2026-06")

    assert lut[(1, 501)] == pytest.approx(6.0)
    assert lut[(1, 502)] == pytest.approx(20.0)


def test_depth_lookup_raises_on_overlapping_windows():
    depths = make_depths(
        [
            ["STER", 1, 501, 6.0, "2017-01-01", pd.NaT],
            ["STER", 1, 501, 7.0, "2020-01-01", pd.NaT],
        ]
    )
    with pytest.raises(RuntimeError, match=">1 valid depth"):
        depth_lookup(depths, "STER", "2026-06")


def test_daily_long_excludes_flagged_from_mean_but_counts_it():
    thirty = pd.DataFrame(
        {
            "station": ["STER_SP1"] * 2,
            "datetime_utc": pd.to_datetime(
                ["2026-06-01 12:00", "2026-06-01 12:30"], utc=True
            ),
            "depth_cm": [6.0, 6.0],
            "vwc": [0.20, 0.90],
            "vwc_qf": [0, 1],
        }
    )
    daily = daily_long(thirty)

    assert len(daily) == 1
    row = daily.iloc[0]
    assert row["value"] == pytest.approx(0.20)
    assert row["n_obs"] == 1
    assert row["n_flagged"] == 1
    assert row["element"] == "soil_vwc_0006"


def test_daily_long_drops_a_day_with_no_passing_observation():
    thirty = pd.DataFrame(
        {
            "station": ["STER_SP1"],
            "datetime_utc": pd.to_datetime(["2026-06-01 12:00"], utc=True),
            "depth_cm": [6.0],
            "vwc": [0.90],
            "vwc_qf": [1],
        }
    )
    daily = daily_long(thirty)

    assert daily.empty


def test_daily_long_assigns_calendar_day_by_site_local_timezone():
    # 05:30 UTC is 23:30 the prior evening in America/Denver (MDT, UTC-6).
    thirty = pd.DataFrame(
        {
            "station": ["STER_SP1"],
            "datetime_utc": pd.to_datetime(["2026-06-02 05:30"], utc=True),
            "depth_cm": [6.0],
            "vwc": [0.20],
            "vwc_qf": [0],
        }
    )
    daily = daily_long(thirty)

    assert daily.loc[0, "date"] == pd.Timestamp("2026-06-01")
