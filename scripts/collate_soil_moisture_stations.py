"""Collate every soil-moisture station this project has actual pulled records for into one
point layer, for visual inspection.

Five disjoint sources, normalized to a common schema and stacked -- no re-scoring, no new
extraction, just gathering what already exists on disk:

* ``validation/conus/conus_nisar_station_scores.csv`` -- the 969-station CONUS panel
  (SNOTEL, MTMESONET, SCAN, USCRN, SOILSCAPE, CW3E, ARM/SGP, iRON), already scored against
  SME2 under the ``summer_qf`` headline config, with irrigation class and CDL cohort. Its
  MTMESONET ``lon``/``lat`` are rounded to 2 decimals upstream, so those are overridden from
  ``mesonet/mt_mesonet_stations.csv`` (see ``_mt_mesonet_precise_coords``).
* ``arm/bnf_stamp_stations.csv`` + ``validation/sme2_arm_bnf_station_scores.csv`` -- the 3
  ARM Bankhead National Forest cropland sites, scored this session.
* ``neon/neon_soil_plots.csv`` -- 11 NEON STER/KONA soil plots, not yet scored against SME2.
* ``ameriflux/ameriflux_sites.csv`` -- 14 SMARTFARM/Mead sites, not yet scored (per
  ``notes/literature_network_leads_2026-08-25.md``, most have ~zero NISAR-era overlap).
* ``risma/risma_stations.csv`` -- 24 RISMA stations; every one has ``n_days_nisar_era == 0``
  (archives end 2019-2020), so none are scorable against SME2 as pulled here.

Deliberately excluded: ``mesonet/umrb_station_panel.csv`` (443 NDAWN/SD Mesonet/NE
Mesonet/WY candidates) -- that is a siting *screen* for future expansion, not a network we
have actual soil-moisture records for yet.

Usage:
    uv run python scripts/collate_soil_moisture_stations.py /data/ssd2/nisar/
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

UBRMSE_GOAL = 0.06
MT_MESONET_NETWORK = "MTMESONET"
COLUMNS = [
    "station",
    "network",
    "source_dataset",
    "longitude",
    "latitude",
    "n_obs",
    "obs_start",
    "obs_end",
    "depths_cm",
    "scored",
    "n_pairs",
    "bias",
    "rmse",
    "ubrmse",
    "r",
    "meets_ubrmse_goal",
    "irrigation_class",
    "land_cover",
    "notes",
]
NUMERIC_COLUMNS = ["n_obs", "n_pairs", "bias", "rmse", "ubrmse", "r"]
BOOL_COLUMNS = ["scored", "meets_ubrmse_goal"]
STRING_COLUMNS = [
    c
    for c in COLUMNS
    if c not in NUMERIC_COLUMNS + BOOL_COLUMNS + ["longitude", "latitude"]
]


def _frame(records: list) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[COLUMNS]


def _mt_mesonet_precise_coords(
    data_dir: Path, df: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Return the CONUS panel's ``lon``/``lat`` with MT Mesonet coordinates de-rounded.

    ``conus_nisar_station_scores.csv`` carries ``lon``/``lat`` populated upstream from the
    2-decimal ``longitude_join``/``latitude_join`` pair (~1 km error) for ``MTMESONET`` only --
    every other network in that file has 3-5 decimal precision. Full-precision coordinates for
    those stations live in ``mesonet/mt_mesonet_stations.csv``, keyed by ``station_uid``
    (``MTMESONET:<id>``), which matches the panel's ``station`` verbatim. That upstream defect is
    not repaired here; this only overrides the coordinates on the way into the collated layer.
    """
    lon = df["lon"].astype(float).copy()
    lat = df["lat"].astype(float).copy()
    is_mt = df["network"] == MT_MESONET_NETWORK
    if not is_mt.any():
        return lon, lat

    precise = pd.read_csv(data_dir / "mesonet/mt_mesonet_stations.csv").set_index(
        "station_uid"
    )
    keys = df.loc[is_mt, "station"]
    missing = sorted(set(keys) - set(precise.index))
    if missing:
        raise ValueError(
            f"{len(missing)} {MT_MESONET_NETWORK} station(s) absent from "
            f"mt_mesonet_stations.csv, cannot recover precise coordinates: {missing}"
        )

    mt_lon = keys.map(precise["longitude"])
    mt_lat = keys.map(precise["latitude"])
    unlocated = sorted(keys[mt_lon.isna() | mt_lat.isna()])
    if unlocated:
        raise ValueError(
            f"{len(unlocated)} {MT_MESONET_NETWORK} station(s) have null coordinates in "
            f"mt_mesonet_stations.csv: {unlocated}"
        )
    lon.loc[is_mt] = mt_lon
    lat.loc[is_mt] = mt_lat
    print(
        f"Replaced rounded join coordinates for {int(is_mt.sum())} {MT_MESONET_NETWORK} stations"
    )
    return lon, lat


def conus_panel(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "validation/conus/conus_nisar_station_scores.csv")
    ubrmse = df["ubrmse_summer_qf"]
    longitude, latitude = _mt_mesonet_precise_coords(data_dir, df)
    return _frame(
        {
            "station": df["station"],
            "network": df["network"],
            "source_dataset": "conus_panel",
            "longitude": longitude,
            "latitude": latitude,
            "n_obs": df["n_pairs_qf"],
            "obs_start": df["extract_first"],
            "obs_end": df["extract_last"],
            "scored": df["n_summer_qf"].notna(),
            "n_pairs": df["n_summer_qf"],
            "bias": df["bias_summer_qf"],
            "rmse": df["rmse_summer_qf"],
            "ubrmse": ubrmse,
            "r": df["r_summer_qf"],
            "meets_ubrmse_goal": ubrmse <= UBRMSE_GOAL,
            "irrigation_class": df["irr_class"],
            "land_cover": df["cdl_cohort"],
        }
    )


def arm_bnf(data_dir: Path) -> pd.DataFrame:
    stations = pd.read_csv(data_dir / "arm/bnf_stamp_stations.csv")
    scores = pd.read_csv(data_dir / "validation/sme2_arm_bnf_station_scores.csv")
    df = stations.merge(scores, on="station", how="left")
    ubrmse = df["ubrmse"]
    return _frame(
        {
            "station": df["station"],
            "network": df["network"],
            "source_dataset": "arm_bnf_stamp",
            "longitude": df["longitude"],
            "latitude": df["latitude"],
            "n_obs": df["n_obs"],
            "obs_start": df["obs_start"],
            "obs_end": df["obs_end"],
            "depths_cm": df["vwc_depths_cm"],
            "scored": df["n_paired"].notna(),
            "n_pairs": df["n_paired"],
            "bias": df["bias"],
            "rmse": df["rmse"],
            "ubrmse": ubrmse,
            "r": df["r"],
            "meets_ubrmse_goal": ubrmse <= UBRMSE_GOAL,
            "irrigation_class": "unknown",
            "land_cover": "cropland (cotton/corn/soy; AmeriFlux CRO registration, no CDL join)",
            "notes": "no confirmed irrigation status or metered events; pulled for "
            "retrieval-accuracy, not irrigation-detection",
        }
    )


def neon(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "neon/neon_soil_plots.csv")
    return _frame(
        {
            "station": df["station"],
            "network": df["network"],
            "source_dataset": "neon",
            "longitude": df["longitude"],
            "latitude": df["latitude"],
            "n_obs": df["n_obs_30min"],
            "obs_start": df["obs_start"],
            "obs_end": df["obs_end"],
            "depths_cm": df["depths_cm"],
            "scored": False,
            "notes": "NEON site "
            + df["site"]
            + ", plot "
            + df["plot"].astype(str)
            + "; not yet scored against SME2",
        }
    )


def ameriflux_smartfarm_mead(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "ameriflux/ameriflux_sites.csv")
    obs_start = df["swc_year_start"].dropna().astype(int).astype(str) + "-01-01"
    obs_end = df["swc_year_end"].dropna().astype(int).astype(str) + "-12-31"
    return _frame(
        {
            "station": df["station"],
            "network": df["network"],
            "source_dataset": "ameriflux_smartfarm_mead",
            "longitude": df["longitude"],
            "latitude": df["latitude"],
            "obs_start": obs_start.reindex(df.index),
            "obs_end": obs_end.reindex(df.index),
            "depths_cm": df["n_swc_vars"].astype("Int64").astype(str)
            + " SWC var(s): "
            + df["swc_vars"].fillna(""),
            "scored": False,
            "irrigation_class": df["irrigation"],
            "land_cover": df["cohort"] + " / " + df["crop"],
            "notes": "correction 2026-08-25: near-zero NISAR-era temporal overlap for most "
            "SMARTFARM/Mead BASE records (see notes/literature_network_leads_2026-08-25.md); "
            "not yet scored against SME2",
        }
    )


def risma(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "risma/risma_stations.csv")
    return _frame(
        {
            "station": df["station"],
            "network": df["network"],
            "source_dataset": "risma",
            "longitude": df["longitude"],
            "latitude": df["latitude"],
            "n_obs": df["n_obs"],
            "obs_start": df["obs_start"],
            "obs_end": df["obs_end"],
            "depths_cm": df["depths_cm"],
            "scored": False,
            "notes": "0 NISAR-era days for every RISMA station (archive ends "
            + df["obs_end"]
            + ", before the 2025-01-01 SME2 window); not scorable as pulled",
        }
    )


def build(data_dir: Path) -> gpd.GeoDataFrame:
    parts = [
        conus_panel(data_dir),
        arm_bnf(data_dir),
        neon(data_dir),
        ameriflux_smartfarm_mead(data_dir),
        risma(data_dir),
    ]
    combined = pd.concat(parts, ignore_index=True)
    for col in NUMERIC_COLUMNS:
        combined[col] = pd.to_numeric(combined[col], errors="raise")
    for col in BOOL_COLUMNS:
        combined[col] = combined[col].astype("boolean")
    for col in STRING_COLUMNS:
        combined[col] = combined[col].astype("string")

    geometry = [Point(xy) for xy in zip(combined["longitude"], combined["latitude"])]
    gdf = gpd.GeoDataFrame(combined, geometry=geometry, crs="EPSG:4326")

    print(
        f"Collated {len(gdf)} stations across {gdf['source_dataset'].nunique()} sources:"
    )
    print(gdf.groupby("source_dataset")["station"].count().to_string())
    print(f"\nScored against SME2: {int(gdf['scored'].sum())}/{len(gdf)}")
    return gdf


if __name__ == "__main__":
    data_dir = Path(sys.argv[1])
    gdf = build(data_dir)
    out_path = data_dir / "reference/all_soil_moisture_stations.fgb"
    gdf.to_file(out_path, driver="FlatGeobuf", engine="fiona")
    print(f"\nWrote {out_path}")
