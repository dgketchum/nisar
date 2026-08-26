"""Assign in-situ stations to the NISAR track/frame footprints that cover them.

Generalizes the join that produced ``reference/mt_mesonet_station_frames.csv`` for the
Montana Mesonet so it can be run for any station table. The output schema is identical
to the Montana files, because ``validate_mesonet_sme2.py`` and
``pull_mesonet_frames_sme2.py`` read them by column name:

* ``<prefix>_station_frames.csv`` -- one row per station x intersecting frame::

      station, station_name, track, frame, passDirection, longitude, latitude

* ``<prefix>_track_frames.csv`` -- the deduplicated combos to stream::

      track, frame, passDirection, pass_code

The second file is the thing that actually sizes an SME2 pull: a station adds no cost
if it falls in a frame some other station already put on the list. For the Montana
Mesonet, 231 stations collapsed to 40 combos.

Frames come from ``reference/nisar_frames_conus.fgb`` (built by
``build_nisar_frame_grid.py``), in EPSG:4326. Stations are joined with a plain
point-in-polygon predicate: a frame footprint is the area the L3 product is gridded
over, so a station either falls inside it or it does not. Frames overlap heavily
between adjacent tracks and between passes, which is why one station routinely lands
in three to six of them.

Only frames that actually produce the soil-moisture product are eligible. The frame
layer carries ``produceSMST``; a frame with ``produceSMST`` false will never yield an
SME2 granule, so joining against it would inflate the combo count with combos that
cannot be pulled. Stations that match no producing frame are reported, not dropped
silently -- a station outside NISAR soil-moisture coverage is a finding about the
network's usefulness here, not a data error.

Usage:
    uv run python scripts/station_nisar_frames.py <stations_csv> <prefix>
    uv run python scripts/station_nisar_frames.py ndawn.csv ndawn --data-dir /data/ssd2/nisar
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

DATA_DIR = Path("/data/ssd2/nisar")
FRAMES_FGB = "reference/nisar_frames_conus.fgb"
PASS_CODE = {"Ascending": "A", "Descending": "D"}

STATION_FRAME_COLUMNS = [
    "station",
    "station_name",
    "track",
    "frame",
    "passDirection",
    "longitude",
    "latitude",
]


def load_frames(data_dir: Path, sm_only: bool = True) -> gpd.GeoDataFrame:
    """The CONUS NISAR frame footprints, optionally restricted to SME2 producers."""
    path = data_dir / FRAMES_FGB
    gdf = gpd.read_file(path, engine="fiona")
    keep = ["track", "frame", "passDirection", "produceSMST", "geometry"]
    missing = [c for c in keep if c not in gdf.columns]
    if missing:
        raise RuntimeError(f"{path} is missing expected columns: {missing}")
    gdf = gdf[keep]
    if sm_only:
        n_all = len(gdf)
        gdf = gdf[gdf["produceSMST"].astype(bool)].copy()
        print(f"frames: {len(gdf)} of {n_all} produce the soil-moisture product")
    return gdf.reset_index(drop=True)


def as_points(stations: pd.DataFrame) -> gpd.GeoDataFrame:
    """Station table -> WGS84 points, with the coordinate columns kept as attributes."""
    for col in ("station", "longitude", "latitude"):
        if col not in stations.columns:
            raise RuntimeError(f"station table is missing required column '{col}'")
    if "station_name" not in stations.columns:
        stations = stations.assign(station_name=stations["station"])
    # A station with no coordinate cannot be joined and must not be quietly counted as
    # "outside coverage" -- it is a metadata gap in the source network, so it is
    # separated out here and reported by the caller.
    bad = stations["longitude"].isna() | stations["latitude"].isna()
    if bad.any():
        raise RuntimeError(
            f"{int(bad.sum())} stations have no coordinate: "
            f"{sorted(stations.loc[bad, 'station'])[:10]}"
        )
    return gpd.GeoDataFrame(
        stations.copy(),
        geometry=gpd.points_from_xy(stations["longitude"], stations["latitude"]),
        crs="EPSG:4326",
    )


def join(stations: pd.DataFrame, frames: gpd.GeoDataFrame) -> pd.DataFrame:
    pts = as_points(stations)
    hits = gpd.sjoin(pts, frames, how="inner", predicate="within")
    out = hits[STATION_FRAME_COLUMNS].copy()
    out["track"] = out["track"].astype(int)
    out["frame"] = out["frame"].astype(int)
    return (
        out.drop_duplicates()
        .sort_values(["station", "track", "frame"])
        .reset_index(drop=True)
    )


def track_frames(station_frames: pd.DataFrame) -> pd.DataFrame:
    combos = (
        station_frames[["track", "frame", "passDirection"]]
        .drop_duplicates()
        .sort_values(["track", "frame"])
        .reset_index(drop=True)
    )
    combos["pass_code"] = combos["passDirection"].map(PASS_CODE)
    if combos["pass_code"].isna().any():
        unknown = sorted(set(combos.loc[combos["pass_code"].isna(), "passDirection"]))
        raise RuntimeError(f"unmapped passDirection values: {unknown}")
    return combos


def build(
    stations_csv: Path,
    prefix: str,
    data_dir: Path,
    out_dir: Path | None = None,
    sm_only: bool = True,
) -> int:
    out_dir = out_dir or data_dir / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    stations = pd.read_csv(stations_csv)
    frames = load_frames(data_dir, sm_only=sm_only)
    sf = join(stations, frames)
    combos = track_frames(sf)

    sf_path = out_dir / f"{prefix}_station_frames.csv"
    tf_path = out_dir / f"{prefix}_track_frames.csv"
    sf.to_csv(sf_path, index=False)
    combos.to_csv(tf_path, index=False)

    uncovered = sorted(set(stations["station"]) - set(sf["station"]))
    per_station = sf.groupby("station").size()

    print(f"\n--- {prefix}: station x NISAR frame join ---")
    print(f"stations in:            {len(stations)}")
    print(f"stations in coverage:   {sf['station'].nunique()}")
    print(f"station-frame rows:     {len(sf)}")
    if len(per_station):
        print(
            f"frames per station:     min {per_station.min()}, "
            f"median {per_station.median():.0f}, max {per_station.max()}"
        )
    print(f"unique track/frames:    {len(combos)}")
    print(f"  ascending:            {int((combos['pass_code'] == 'A').sum())}")
    print(f"  descending:           {int((combos['pass_code'] == 'D').sum())}")
    if uncovered:
        print(f"stations outside coverage ({len(uncovered)}): {uncovered}")
    else:
        print("stations outside coverage: none")
    print(f"\nwrote {sf_path}")
    print(f"wrote {tf_path}")
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "stations_csv", type=Path, help="csv with station,longitude,latitude"
    )
    p.add_argument("prefix", help="output filename prefix, e.g. 'ndawn'")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where to write the two csvs (default <data-dir>/reference)",
    )
    p.add_argument(
        "--all-frames",
        action="store_true",
        help="join against every frame, not just soil-moisture producers",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args(sys.argv[1:])
    sys.exit(
        build(
            a.stations_csv,
            a.prefix,
            a.data_dir,
            out_dir=a.out_dir,
            sm_only=not a.all_frames,
        )
    )
