"""Tests for the pure helpers in examples/10_MT_Mesonet/evaluate.py.

``station_wide``, ``build_pairs``, and ``score_pairs`` run on plain pandas — the
swimrs/Example 8 machinery is deferred, so no swim-rs environment is required.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EVAL_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "10_MT_Mesonet" / "evaluate.py"
)

spec = importlib.util.spec_from_file_location("evaluate10", EVAL_PATH)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def _long_frame(station, dates, depth_values):
    rows = []
    for depth, vals in depth_values.items():
        for d, v in zip(dates, vals):
            rows.append(
                {
                    "station": station,
                    "date": d,
                    "element": f"soil_vwc_{int(depth):04d}",
                    "depth_cm": float(depth),
                    "value": v,
                    "units": "m3/m3",
                }
            )
    return pd.DataFrame(rows)


def test_build_pairs_prefers_shallowest_sensor():
    pairs = ev.build_pairs(
        ["soil_vwc_10", "soil_vwc_50", "soil_vwc_91", "rootzone_theta"]
    )
    surface = {m: y for m, y, _ in pairs}["surface_sm_proxy"]
    assert surface == "soil_vwc_10"

    pairs = ev.build_pairs(["soil_vwc_5", "soil_vwc_10", "soil_vwc_50"])
    surface = {m: y for m, y, _ in pairs}["surface_sm_proxy"]
    assert surface == "soil_vwc_5"


def test_station_wide_pivots_and_adds_profile_mean():
    dates = pd.date_range("2024-06-01", periods=3, freq="D")
    long = _long_frame(
        "blmtest", dates, {5: [0.10, 0.12, 0.14], 50: [0.20, 0.20, 0.22]}
    )
    wide = ev.station_wide(long, "blmtest")

    assert list(wide.columns) == ["soil_vwc_5", "soil_vwc_50", "profile_mean_theta"]
    assert wide.loc[dates[0], "soil_vwc_5"] == 0.10
    assert wide.loc[dates[2], "profile_mean_theta"] == pytest.approx((0.14 + 0.22) / 2)


def test_station_wide_unknown_station_empty():
    dates = pd.date_range("2024-06-01", periods=2, freq="D")
    long = _long_frame("blmtest", dates, {5: [0.1, 0.1]})
    assert ev.station_wide(long, "blmnosuch").empty


def test_station_wide_raises_on_out_of_range():
    dates = pd.date_range("2024-06-01", periods=2, freq="D")
    long = _long_frame("blmtest", dates, {5: [0.1, 283.5]})
    with pytest.raises(SystemExit, match="outside"):
        ev.station_wide(long, "blmtest")


def _model_obs(n_years, r_seed=0):
    """Synthetic multi-year daily pair with a shared seasonal + shared anomaly signal."""
    rng = np.random.default_rng(r_seed)
    idx = pd.date_range("2020-01-01", periods=365 * n_years, freq="D")
    season = 0.1 * np.sin(2 * np.pi * idx.dayofyear / 365)
    shared = rng.normal(0, 0.03, len(idx))
    obs = pd.DataFrame(
        {
            "soil_vwc_10": 0.2 + season + shared,
            "rootzone_theta": 0.25 + season + shared,
        },
        index=idx,
    )
    obs["profile_mean_theta"] = obs["soil_vwc_10"]
    mdf = pd.DataFrame(
        {
            "theta_avail": 0.15 + season + shared + rng.normal(0, 0.005, len(idx)),
            "surface_sm_proxy": -50 + 100 * (season + shared),
        },
        index=idx,
    )
    return mdf, obs


def test_score_pairs_recovers_correlation_and_pairings():
    mdf, obs = _model_obs(n_years=4)
    rows = pd.DataFrame(ev.score_pairs("blmtest", mdf, obs))

    assert set(rows["pairing"]) == {
        "rootzone depth-wtd",
        "surface shallowest (SMAP/NISAR analog)",
        "unwtd profile mean",
    }
    root = rows[rows.pairing == "rootzone depth-wtd"].iloc[0]
    assert root["pearson"] > 0.9 and root["anom_r"] > 0.9
    # Growing-season only: no Nov-Mar dates in the pair count.
    assert root["n"] < len(mdf)


def test_score_pairs_anomaly_gated_on_short_span():
    mdf, obs = _model_obs(n_years=1)
    rows = pd.DataFrame(ev.score_pairs("blmtest", mdf, obs))
    root = rows[rows.pairing == "rootzone depth-wtd"].iloc[0]
    assert np.isnan(root["anom_r"])  # < 2-year span: climatology is the data
    assert root["pearson"] > 0.9  # plain correlation still reported


def test_score_pairs_too_few_days_skipped():
    mdf, obs = _model_obs(n_years=1)
    rows = ev.score_pairs("blmtest", mdf.iloc[:200], obs.iloc[190:210])
    assert rows == []
