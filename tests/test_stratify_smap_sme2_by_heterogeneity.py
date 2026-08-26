"""Tests for the land-cover classification in the SMAP-vs-SME2 heterogeneity
stratification: the derived heterogeneity columns, the panel-1 representativeness rule
(dominant-group match plus a small cultivated contrast, boundary included), and the
panel-2 candidate-pixel shortlist thresholds.

Both classifications are quoted as station properties in the proposal panels, so each
threshold is exercised on both sides of its edge -- a station that is mixed-agricultural
but not representative must not reach the shortlist, and vice versa.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from stratify_smap_sme2_by_heterogeneity import (
    dominant_group,
    group_columns,
    load_landcover,
    shortlist,
)

GROUPS = ["smallgrain", "grass", "forest"]


def pixel_row(
    station: str,
    cult_f_100: float,
    cult_f_9km: float,
    dom_100: str = "smallgrain",
    dom_9km: str = "smallgrain",
    irr_f_9km: float = 0.10,
    dom_frac: float = 0.6,
) -> dict:
    """One synthetic sample_smap_pixel_landcover.py row.

    The CDL group fractions and the cultivated/irrigated means come from different
    reducers upstream, so they are set independently here too.
    """
    rec = {
        "station": station,
        "cell": f"cell_{station}",
        "longitude": -110.0,
        "latitude": 46.0,
        "cult_f_pt": cult_f_100,
        "cult_f_100": cult_f_100,
        "cult_f_9km": cult_f_9km,
        "irr_f_9km": irr_f_9km,
        "entropy_100": 1.0,
        "entropy_9km": 1.2,
    }
    for suffix, dom in (("100", dom_100), ("9km", dom_9km)):
        rest = (1.0 - dom_frac) / (len(GROUPS) - 1)
        for group in GROUPS:
            rec[f"f_{group}_{suffix}"] = dom_frac if group == dom else rest
    return rec


def write_landcover(tmp_path: Path, rows: list) -> Path:
    """Write the synthetic pixel land-cover CSV where load_landcover expects it."""
    out = tmp_path / "reference/smap_pixel_landcover.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return tmp_path


def test_group_columns_excludes_the_cultivated_mean_columns():
    """``cult_f_9km`` is a mean, not a CDL group fraction, and must not be a candidate
    for the dominant group."""
    df = pd.DataFrame(
        columns=[
            "f_grass_9km",
            "f_forest_9km",
            "cult_f_9km",
            "irr_f_9km",
            "f_grass_100",
        ]
    )
    assert group_columns(df, "9km") == ["f_grass_9km", "f_forest_9km"]


def test_dominant_group_strips_prefix_and_suffix():
    df = pd.DataFrame(
        {
            "f_grass_9km": [0.2, 0.7],
            "f_forest_9km": [0.8, 0.3],
            "cult_f_9km": [0.9, 0.9],
        }
    )
    assert list(dominant_group(df, "9km")) == ["forest", "grass"]


def test_load_landcover_derives_contrast_and_mix(tmp_path):
    data_dir = write_landcover(
        tmp_path, [pixel_row("a", cult_f_100=0.90, cult_f_9km=0.50, dom_frac=0.6)]
    )
    pix = load_landcover(data_dir)
    row = pix.iloc[0]

    assert row["cult_contrast"] == pytest.approx(0.40)
    assert row["mix_9km"] == pytest.approx(0.40)  # 1 - dominant group fraction
    assert row["dom_group_100"] == "smallgrain"
    assert row["dom_group_9km"] == "smallgrain"


def test_load_landcover_contrast_is_absolute():
    """|100 m - cell|: a probe drier in cultivation than its pixel counts the same as
    a probe wetter in it."""
    df = pd.DataFrame([pixel_row("a", cult_f_100=0.20, cult_f_9km=0.60)])
    assert (df["cult_f_100"] - df["cult_f_9km"]).abs().iloc[0] == pytest.approx(0.40)


def test_representative_requires_both_match_and_small_contrast(tmp_path):
    data_dir = write_landcover(
        tmp_path,
        [
            pixel_row("both_ok", cult_f_100=0.55, cult_f_9km=0.50),
            pixel_row(
                "dom_mismatch", cult_f_100=0.55, cult_f_9km=0.50, dom_100="grass"
            ),
            pixel_row("big_contrast", cult_f_100=0.90, cult_f_9km=0.50),
        ],
    )
    pix = load_landcover(data_dir).set_index("station")

    assert pix.loc["both_ok", "dom_group_match"]
    assert pix.loc["both_ok", "representative"]

    assert not pix.loc["dom_mismatch", "dom_group_match"]
    assert not pix.loc["dom_mismatch", "representative"]
    assert pix.loc["dom_mismatch", "cult_contrast"] == pytest.approx(0.05)

    assert pix.loc["big_contrast", "dom_group_match"]
    assert not pix.loc["big_contrast", "representative"]


def test_representative_includes_a_contrast_of_exactly_the_maximum(tmp_path):
    """0.15 is inside the class (``<=``); the pair 0.30/0.15 differs by exactly the
    float 0.15, so this pins the boundary itself rather than a value near it."""
    data_dir = write_landcover(
        tmp_path,
        [
            pixel_row("at_max", cult_f_100=0.30, cult_f_9km=0.15),
            pixel_row("over_max", cult_f_100=0.31, cult_f_9km=0.15),
        ],
    )
    pix = load_landcover(data_dir).set_index("station")

    assert pix.loc["at_max", "cult_contrast"] == 0.15
    assert pix.loc["at_max", "representative"]
    assert pix.loc["over_max", "cult_contrast"] > 0.15
    assert not pix.loc["over_max", "representative"]


def shortlist_input(tmp_path, rows):
    """load_landcover output plus the head-to-head skill columns shortlist() carries."""
    pix = load_landcover(write_landcover(tmp_path, rows))
    pix["r_smap"] = 0.5
    pix["n_paired_smap"] = pd.Series([200] * len(pix), dtype="Int64")
    return pix


def test_shortlist_requires_a_representative_station(tmp_path):
    """A genuinely mixed agricultural cell is not a venue if the probe in it does not
    stand for the pixel."""
    pix = shortlist_input(
        tmp_path,
        [
            pixel_row("rep", cult_f_100=0.50, cult_f_9km=0.50),
            pixel_row("not_rep", cult_f_100=0.50, cult_f_9km=0.50, dom_100="forest"),
        ],
    )
    assert list(shortlist(pix)["station"]) == ["rep"]


def test_shortlist_cultivated_fraction_range_is_inclusive(tmp_path):
    pix = shortlist_input(
        tmp_path,
        [
            pixel_row("at_lo", cult_f_100=0.20, cult_f_9km=0.20),
            pixel_row("below_lo", cult_f_100=0.19, cult_f_9km=0.19),
            pixel_row("at_hi", cult_f_100=0.80, cult_f_9km=0.80),
            pixel_row("above_hi", cult_f_100=0.81, cult_f_9km=0.81),
        ],
    )
    assert set(shortlist(pix)["station"]) == {"at_lo", "at_hi"}


def test_shortlist_irrigation_minimum_is_inclusive(tmp_path):
    pix = shortlist_input(
        tmp_path,
        [
            pixel_row("at_min", cult_f_100=0.50, cult_f_9km=0.50, irr_f_9km=0.05),
            pixel_row("below_min", cult_f_100=0.50, cult_f_9km=0.50, irr_f_9km=0.049),
        ],
    )
    assert list(shortlist(pix)["station"]) == ["at_min"]


def test_shortlist_sorts_by_irrigated_fraction_descending(tmp_path):
    pix = shortlist_input(
        tmp_path,
        [
            pixel_row("low_irr", cult_f_100=0.50, cult_f_9km=0.50, irr_f_9km=0.06),
            pixel_row("high_irr", cult_f_100=0.50, cult_f_9km=0.50, irr_f_9km=0.40),
            pixel_row("mid_irr", cult_f_100=0.50, cult_f_9km=0.50, irr_f_9km=0.20),
        ],
    )
    short = shortlist(pix)

    assert list(short["station"]) == ["high_irr", "mid_irr", "low_irr"]
    assert list(short.columns)[:2] == ["cell", "station"]
