"""Tests for the pure file-in/file-out helpers in examples/10_MT_Mesonet/prep.py.

The module defers its swimrs/Example 5 imports, so ``select_sites``,
``prepare_properties``, and ``_guard_shapefile`` are exercised here with stub configs
and synthetic inputs — no swim-rs environment required.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

PREP_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "10_MT_Mesonet" / "prep.py"
)

spec = importlib.util.spec_from_file_location("prep10", PREP_PATH)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

SITES = ["blmalpha", "blmbravo", "acecharl"]


@pytest.fixture
def fields_cfg(tmp_path):
    """A three-site 100 m-buffer geometry and the cfg attrs select_sites reads."""
    gis_dir = tmp_path / "gis"
    gis_dir.mkdir()
    gdf = gpd.GeoDataFrame(
        {"site_id": SITES, "state": "MT"},
        geometry=gpd.points_from_xy([0, 1000, 2000], [0, 0, 0]),
        crs="EPSG:5071",
    )
    gdf["geometry"] = gdf.geometry.buffer(100.0)
    src = gis_dir / "mesonet_fields_100m.fgb"
    gdf.to_file(src, driver="FlatGeobuf", engine="fiona")
    return SimpleNamespace(
        fields_shapefile=str(src), feature_id_col="site_id", gis_dir=str(gis_dir)
    )


def test_select_sites_excludes_and_writes_subset(fields_cfg):
    src = fields_cfg.fields_shapefile
    kept = prep.select_sites(fields_cfg, ["blmbravo"])

    # FlatGeobuf round-trips in spatial-index (Hilbert) order, not insertion
    # order, so compare membership; the returned order must match the derived
    # geometry the container is then built from.
    assert set(kept) == {"blmalpha", "acecharl"}
    out = Path(fields_cfg.gis_dir) / "build" / "mesonet_fields_100m_n2.fgb"
    assert fields_cfg.fields_shapefile == str(out)
    derived = gpd.read_file(out, engine="fiona")
    assert derived["site_id"].tolist() == kept
    # The source geometry is never modified.
    assert set(gpd.read_file(src, engine="fiona")["site_id"]) == set(SITES)


def test_select_sites_empty_exclusion_keeps_all(fields_cfg):
    kept = prep.select_sites(fields_cfg, [])
    assert set(kept) == set(SITES)
    assert fields_cfg.fields_shapefile.endswith("_n3.fgb")


def test_select_sites_unknown_site_raises(fields_cfg):
    with pytest.raises(SystemExit, match="blmnosuch"):
        prep.select_sites(fields_cfg, ["blmnosuch"])


@pytest.fixture
def props_cfg(tmp_path):
    """Property CSVs covering a superset of sites, with empty LAT/LON columns."""
    src_dir = tmp_path / "properties" / "getinfo"
    src_dir.mkdir(parents=True)
    all_sites = SITES + ["blmextra"]
    for name, cols in (
        ("mt_mesonet_landcover.csv", {"modis_lc_2020": [10, 10, 12, 10]}),
        ("mt_mesonet_ssurgo.csv", {"awc": [0.11, 0.12, 0.13, 0.14]}),
        ("mt_mesonet_irr.csv", {"irr_2020": [0.0, 0.0, 0.5, 0.0]}),
    ):
        df = pd.DataFrame({"site_id": all_sites, **cols})
        df["LAT"] = np.nan
        df["LON"] = np.nan
        df.to_csv(src_dir / name, index=False)
    return SimpleNamespace(
        properties_dir=str(tmp_path / "properties"),
        feature_id_col="site_id",
        lulc_csv="/anywhere/mt_mesonet_landcover.csv",
        ssurgo_csv="/anywhere/mt_mesonet_ssurgo.csv",
        irr_csv="/anywhere/mt_mesonet_irr.csv",
    )


def test_prepare_properties_subsets_orders_and_drops_latlon(props_cfg):
    # Request an order different from the CSVs' row order to prove reordering.
    sites = ["acecharl", "blmalpha"]
    prepared = prep.prepare_properties(props_cfg, sites)

    assert set(prepared) == {"lulc", "ssurgo", "irr"}
    for path in prepared.values():
        df = pd.read_csv(path)
        assert df["site_id"].tolist() == sites
        assert "LAT" not in df.columns and "LON" not in df.columns
    irr = pd.read_csv(prepared["irr"]).set_index("site_id")
    assert irr.loc["acecharl", "irr_2020"] == 0.5

    # Sources untouched: still 4 sites, LAT/LON still present.
    src = Path(props_cfg.properties_dir) / "getinfo" / "mt_mesonet_irr.csv"
    src_df = pd.read_csv(src)
    assert len(src_df) == 4 and "LAT" in src_df.columns


def test_prepare_properties_empty_latlon_do_not_trip_null_check(props_cfg):
    # LAT/LON are all-NaN in the sources; the drop must precede the null gate.
    prep.prepare_properties(props_cfg, SITES)


def test_prepare_properties_missing_site_raises(props_cfg):
    with pytest.raises(SystemExit, match="blmnosuch"):
        prep.prepare_properties(props_cfg, SITES + ["blmnosuch"])


def test_prepare_properties_null_data_raises(props_cfg):
    src = Path(props_cfg.properties_dir) / "getinfo" / "mt_mesonet_ssurgo.csv"
    df = pd.read_csv(src)
    df.loc[df["site_id"] == "blmbravo", "awc"] = np.nan
    df.to_csv(src, index=False)
    with pytest.raises(SystemExit, match="null values"):
        prep.prepare_properties(props_cfg, SITES)


def test_guard_shapefile(fields_cfg, tmp_path):
    prep._guard_shapefile(fields_cfg)  # exists: no raise
    missing = SimpleNamespace(fields_shapefile=str(tmp_path / "nope.fgb"))
    with pytest.raises(SystemExit, match="not found"):
        prep._guard_shapefile(missing)
