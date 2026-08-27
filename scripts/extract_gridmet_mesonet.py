"""Extract gridMET meteorology for SWIM-RS at MT Mesonet stations.

The zoran leg of the SWIM input extraction (see
notes/zh_swim_extraction_instructions_2026-08-26.md): everything except the
bias-corrected reference ET, which comes from OpenET via the Earth Engine leg.
Builds the same 100 m buffer polygons the EE leg simulates, maps each to its
canonical gridMET cell (GFID) using the CONUS centroid master, and downloads
one daily met parquet per unique cell through the swim-rs THREDDS client.

Must run with the swim-rs venv (needs swimrs, rasterstats, tqdm):
    /home/dgketchum/code/swim-rs/.venv/bin/python scripts/extract_gridmet_mesonet.py

Outputs under --out-dir (default /data/ssd2/nisar/swim/mt_mesonet/):
    gis/mesonet_fields_100m.shp|.fgb   100 m station buffers (site_id, state)
    gis/mesonet_fields_gfid.shp|.fgb   buffers joined to GFID/LAT/LON/ELEV
    meteorology/gridmet/{GFID}.parquet daily tmin/tmax/eto/etr/prcp/srad/u2/ea/elev

No correction-factor JSON is produced: eto/etr here are raw gridMET (spinup
only); the corrected reference ET the model runs on is the OpenET extraction.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from swimrs.data_extraction.gridmet.gridmet import assign_gridmet_ids, download_gridmet

STATIONS = "/data/ssd2/nisar/mesonet/mt_mesonet_stations.csv"
CENTROIDS = "/nas/swim/gridmet/gridmet_centroids.shp"
OUT_DIR = "/data/ssd2/nisar/swim/mt_mesonet"

# Top ten Mesonet sites by 9 km cell land-cover homogeneity (ascending
# entropy_9km in reference/smap_pixel_landcover.csv, 2026-08-26) — the pilot
# cohort Hoylman's machine is extracting the EE-side SWIM inputs for.
TOP10_HOMOGENEOUS = [
    "blmbelfr",
    "blmbattl",
    "blmhardi",
    "blmwarre",
    "aceweldo",
    "blmglnor",
    "blmterry",
    "acemilli",
    "blmcapit",
    "blmrubyc",
]

BUFFER_M = 100.0
FIELDS_CRS = "EPSG:5071"


def build_fields(stations_csv: str, sites: list[str], gis_dir: Path) -> Path:
    """Write 100 m buffer polygons around the precise station coordinates."""
    df = pd.read_csv(stations_csv)
    df = df[df["station"].isin(sites)].copy()
    missing = set(sites) - set(df["station"])
    if missing:
        raise ValueError(f"stations not found in {stations_csv}: {sorted(missing)}")

    gdf = gpd.GeoDataFrame(
        {"site_id": df["station"].values, "state": "MT"},
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    ).to_crs(FIELDS_CRS)
    gdf["geometry"] = gdf.geometry.buffer(BUFFER_M)

    shp = gis_dir / "mesonet_fields_100m.shp"
    gdf.to_file(shp, engine="fiona")
    gdf.to_file(
        gis_dir / "mesonet_fields_100m.fgb", driver="FlatGeobuf", engine="fiona"
    )
    print(f"wrote {shp} ({len(gdf)} fields)")
    return shp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stations", default=STATIONS)
    ap.add_argument("--sites", nargs="+", default=TOP10_HOMOGENEOUS)
    ap.add_argument("--centroids", default=CENTROIDS)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--start", default="2010-01-01")
    # End must not exceed the last day in the THREDDS aggregation (typically
    # lags 1-2 days behind present) or the swimrs client raises a time-dim
    # mismatch for every variable.
    ap.add_argument("--end", default="2026-08-24")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    gis_dir = out_dir / "gis"
    # The run configs point met at {data}/meteorology/gridmet — write there so a
    # network-wide pull appends to the pilot cohort's parquets instead of forking
    # a second store under met/.
    met_dir = out_dir / "meteorology" / "gridmet"
    gis_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    fields_shp = build_fields(args.stations, args.sites, gis_dir)

    join_shp = gis_dir / "mesonet_fields_gfid.shp"
    joined = assign_gridmet_ids(
        str(fields_shp),
        str(join_shp),
        gridmet_points=args.centroids,
        feature_id="site_id",
    )
    joined.to_file(
        gis_dir / "mesonet_fields_gfid.fgb", driver="FlatGeobuf", engine="fiona"
    )
    print(joined[["site_id", "GFID", "LAT", "LON", "ELEV"]].to_string(index=False))

    download_gridmet(
        str(join_shp),
        None,
        str(met_dir),
        start=args.start,
        end=args.end,
        feature_id="site_id",
        append=True,
    )
    print(f"gridMET parquet files in {met_dir}")


if __name__ == "__main__":
    main()
