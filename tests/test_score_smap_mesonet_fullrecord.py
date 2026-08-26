"""Tests for the metric machinery in the full-record SMAP scoring: the circular
day-of-year climatology, the anomaly r and its two-year span gate, and the per-station
ubRMSE.

The anomaly r exists because a raw r over a Montana annual cycle is largely scoring the
seasonal cycle both series share; the case below is built so that is literally true --
two series whose only common signal is the seasonal cycle score a high raw r and an
anomaly r of essentially zero.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from score_smap_mesonet_fullrecord import (
    DOY_SLOTS,
    MIN_ANOM_SPAN_DAYS,
    anomaly_r,
    doy_climatology,
    score,
)


def test_doy_climatology_of_a_constant_series_is_constant():
    doys = np.arange(1, DOY_SLOTS + 1)
    clim = doy_climatology(doys, np.full(DOY_SLOTS, 0.4))

    assert len(clim) == DOY_SLOTS
    assert not np.isnan(clim).any()
    assert clim == pytest.approx(np.full(DOY_SLOTS, 0.4))


def test_doy_climatology_window_wraps_the_year_boundary():
    """Jan 1 and Dec 31 sit in each other's window: with data only on doy 1 and doy 365,
    both slots -- and the leap slot 366 between them -- average the two."""
    clim = doy_climatology(np.array([1, 365]), np.array([1.0, 3.0]))

    assert clim[0] == pytest.approx(2.0)  # doy 1 sees doy 365
    assert clim[364] == pytest.approx(2.0)  # doy 365 sees doy 1
    assert clim[365] == pytest.approx(2.0)  # doy 366 sees both


def test_doy_climatology_window_is_finite_not_global():
    """The wrap must not turn the mean into a whole-record mean: a slot more than half a
    window from either observation stays NaN."""
    clim = doy_climatology(np.array([1, 365]), np.array([1.0, 3.0]))

    assert np.isnan(clim[179])  # doy 180, mid-year
    assert np.isnan(clim[29])  # doy 30, past the window from doy 1
    assert clim[15] == pytest.approx(1.0)  # doy 16, last slot doy 1 still reaches


def seasonal_pair(n: int = 1098) -> pd.DataFrame:
    """Two series sharing one seasonal cycle and nothing else.

    The departures from the cycle are a period-2 and a period-3 square wave, exactly
    orthogonal over any multiple of 6 days, so the shared seasonal cycle is the whole of
    the raw correlation.
    """
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    doy = dates.dayofyear.to_numpy()
    seasonal = 0.25 + 0.15 * np.sin(2 * np.pi * doy / 365.0)
    i = np.arange(n)
    return pd.DataFrame(
        {
            "date": dates,
            "smap_sm": seasonal + 0.03 * np.where(i % 2 == 0, 1.0, -1.0),
            "insitu_vwc": seasonal + 0.03 * np.tile([1.0, 1.0, -2.0], n // 3),
        }
    )


def test_anomaly_r_removes_a_shared_seasonal_cycle():
    paired = seasonal_pair()
    raw_r = paired["smap_sm"].corr(paired["insitu_vwc"])
    r_anom, n_anom = anomaly_r(paired)

    assert raw_r > 0.85
    assert abs(r_anom) < 0.05
    assert n_anom == len(paired)


def test_anomaly_r_returns_nan_below_the_two_year_span():
    dates = pd.date_range("2020-01-01", periods=MIN_ANOM_SPAN_DAYS, freq="D")
    values = np.linspace(0.1, 0.4, len(dates))
    paired = pd.DataFrame({"date": dates, "smap_sm": values, "insitu_vwc": values})

    assert (dates.max() - dates.min()).days == MIN_ANOM_SPAN_DAYS - 1
    r_anom, n_anom = anomaly_r(paired)
    assert np.isnan(r_anom)
    assert n_anom == 0


def test_anomaly_r_computes_at_exactly_the_two_year_span():
    paired = seasonal_pair().iloc[: MIN_ANOM_SPAN_DAYS + 1]
    dates = paired["date"]

    assert (dates.max() - dates.min()).days == MIN_ANOM_SPAN_DAYS
    r_anom, n_anom = anomaly_r(paired)
    assert not np.isnan(r_anom)
    assert n_anom == len(dates)


def test_anomaly_r_returns_nan_when_a_long_span_holds_too_few_pairs():
    """Span alone is not enough: four pairs spread over three years clear the gate and
    are still refused."""
    paired = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-01", "2020-06-01", "2021-06-01", "2022-06-01"]
            ),
            "smap_sm": [0.10, 0.20, 0.30, 0.40],
            "insitu_vwc": [0.10, 0.20, 0.30, 0.50],
        }
    )
    r_anom, n_anom = anomaly_r(paired)

    assert np.isnan(r_anom)
    assert n_anom == 4


def scoring_frames(smap_by_station: dict, insitu: list) -> tuple:
    """SMAP and in-situ frames sharing dates, one entry per station."""
    dates = pd.date_range("2020-06-01", periods=len(insitu), freq="D")
    smap = pd.concat(
        [
            pd.DataFrame({"station": stn, "date": dates, "smap_sm": vals})
            for stn, vals in smap_by_station.items()
        ],
        ignore_index=True,
    )
    ins = pd.concat(
        [
            pd.DataFrame({"station": stn, "date": dates, "insitu_vwc": insitu})
            for stn in smap_by_station
        ],
        ignore_index=True,
    )
    return smap, ins


def test_score_ubrmse_is_rmse_after_removing_the_mean_bias():
    """Differences of [0.1, 0.1, 0.1, 0.1, 0.3]: bias 0.14, RMSE sqrt(0.026), and
    ubRMSE exactly 0.08 -- so bias and ubRMSE cannot be confused for one another."""
    insitu = [0.10, 0.15, 0.20, 0.25, 0.30]
    smap, ins = scoring_frames(
        {"biased": [0.20, 0.25, 0.30, 0.35, 0.60]},
        insitu,
    )
    scores = score(smap, ins, pd.Series(dtype="int64")).set_index("station")
    row = scores.loc["biased"]

    assert row["n_pairs"] == 5
    assert row["bias"] == pytest.approx(0.14)
    assert row["rmse"] == pytest.approx(np.sqrt(0.026))
    assert row["ubrmse"] == pytest.approx(0.08)
    assert not row["meets_ubrmse_goal"]


def test_score_constant_offset_has_zero_ubrmse_and_meets_the_goal():
    insitu = [0.10, 0.15, 0.20, 0.25, 0.30]
    smap, ins = scoring_frames({"offset": [v + 0.10 for v in insitu]}, insitu)
    scores = score(smap, ins, pd.Series(dtype="int64")).set_index("station")
    row = scores.loc["offset"]

    assert row["bias"] == pytest.approx(0.10)
    assert row["ubrmse"] == pytest.approx(0.0)
    assert row["rmse"] == pytest.approx(0.10)
    assert row["r"] == pytest.approx(1.0)
    assert row["meets_ubrmse_goal"]


def test_score_drops_a_station_below_the_minimum_pairs_and_carries_voided_counts():
    insitu = [0.10, 0.15, 0.20, 0.25, 0.30]
    smap, ins = scoring_frames(
        {"kept": [v + 0.10 for v in insitu], "thin": [v + 0.10 for v in insitu]},
        insitu,
    )
    smap = smap[~((smap["station"] == "thin") & (smap["date"].dt.day > 3))]
    voided = pd.Series({"kept": 7})

    scores = score(smap, ins, voided)

    assert list(scores["station"]) == ["kept"]
    assert scores.loc[0, "n_insitu_qc_voided"] == 7
    assert scores.loc[0, "span_days"] == 4
    assert np.isnan(scores.loc[0, "r_anom"])  # five days is no two-year span
    assert scores.loc[0, "n_anom"] == 0
