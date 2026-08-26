"""Tests for the pure, network-free pieces of the ARM BNF STAMP puller: filename/
timezone parsing, daily aggregation, the station summary, and both of ARM Live's
credential-rejection shapes.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pull_arm_bnf_stamp import (
    check_auth,
    daily_precip,
    daily_vwc,
    file_day,
    local_day,
    station_table,
)


def test_file_day_extracts_yyyymmdd():
    name = "bnfstampS20.b1.20250601.000000.cdf"
    assert file_day(name) == "20250601"


def test_file_day_raises_on_unexpected_name():
    with pytest.raises(RuntimeError, match="unexpected ARM file name"):
        file_day("not_an_arm_file.txt")


def test_local_day_converts_utc_to_chicago_calendar_day():
    # 2025-06-01 04:30 UTC is still 2025-05-31 23:30 in America/Chicago (CDT, UTC-5).
    utc = pd.Series(pd.to_datetime(["2025-06-01 04:30:00"], utc=True))
    day = local_day(utc)
    assert day.iloc[0] == pd.Timestamp("2025-05-31")


def test_daily_vwc_averages_half_hourly_by_key():
    long = pd.DataFrame(
        {
            "station": ["BNF_S20"] * 3,
            "datetime_utc": pd.to_datetime(
                ["2025-06-01 06:00", "2025-06-01 06:30", "2025-06-01 07:00"], utc=True
            ),
            "profile": ["west"] * 3,
            "element": ["soil_specific_water_content"] * 3,
            "depth_cm": [5.0, 5.0, 5.0],
            "value": [0.20, 0.30, 0.40],
            "units": ["m3/m3"] * 3,
            "qc": [0, 0, 0],
        }
    )
    daily = daily_vwc(long)

    assert len(daily) == 1
    row = daily.iloc[0]
    assert row["value"] == pytest.approx(0.30)
    assert row["n_obs"] == 3
    assert row["units"] == "m3/m3"


def test_daily_precip_sums_minutes_by_local_day():
    minutes = pd.DataFrame(
        {
            "station": ["BNF_S20"] * 2,
            "datetime_utc": pd.to_datetime(
                ["2025-06-01 06:00", "2025-06-01 06:01"], utc=True
            ),
            "precip_mm": [0.5, 1.0],
        }
    )
    daily = daily_precip(minutes)

    assert len(daily) == 1
    row = daily.iloc[0]
    assert row["precip_mm"] == pytest.approx(1.5)
    assert row["n_minutes"] == 2
    assert row["units"] == "mm"


def test_check_auth_raises_on_401_status():
    response = requests.Response()
    response.status_code = 401
    with pytest.raises(SystemExit, match="rejected the credentials"):
        check_auth(response)


def test_check_auth_raises_on_invalid_username_body():
    response = requests.Response()
    response.status_code = 200
    with pytest.raises(SystemExit, match="rejected the credentials"):
        check_auth(response, body="Invalid username.")


def test_check_auth_passes_on_good_response():
    response = requests.Response()
    response.status_code = 200
    check_auth(response, body='{"status": "success", "files": []}')


def test_station_table_without_daily_has_one_row_per_site():
    table = station_table(None)

    assert len(table) == 3
    assert set(table["facility"]) == {"S20", "S30", "S40"}
    assert "n_obs" not in table.columns


def test_station_table_summarizes_5cm_coverage():
    daily = pd.DataFrame(
        {
            "station": ["BNF_S20", "BNF_S20", "BNF_S20"],
            "date": pd.to_datetime(["2025-06-01", "2025-06-02", "2025-06-01"]),
            "profile": ["west", "west", "west"],
            "element": ["soil_specific_water_content"] * 3,
            "depth_cm": [5.0, 5.0, 20.0],
            "value": [0.20, 0.25, 0.35],
            "units": ["m3/m3"] * 3,
        }
    )
    table = station_table(daily)
    row = table.set_index("facility").loc["S20"]

    assert row["n_obs_5cm"] == 2
    assert row["n_depths"] == 2
    assert row["n_out_of_range"] == 0
