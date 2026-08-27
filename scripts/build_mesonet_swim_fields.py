"""Build the SWIM-RS field polygon layer for the Montana Mesonet stations.

SWIM-RS ingests *polygons*, not points: every extractor in
``swimrs.data_extraction.ee`` reduces an image over a feature geometry
(``ndvi_export.sparse_sample_ndvi`` documents its input as "path to polygon features"),
and ``ndvi_export`` has no buffer argument of its own. The Mesonet record is a point per
station, so the polygons have to be made here, once, and handed to every extractor so
NDVI, ETf, SNODAS and the property tables all describe the identical footprint.

Radius is 100 m, matching ``LOCAL_RADIUS_M`` in ``sample_smap_pixel_landcover.py``. That
is the same neighborhood the SMAP-pixel land-cover contrast already characterizes, so the
SWIM-RS inputs land on a footprint we have a cover description for, rather than a third
scale needing its own characterization. At 100 m a footprint is ~3 ha -- a handful of
Landsat pixels, enough for a stable zonal mean without pulling in cover the probe never
saw.

The buffer is taken in EPSG:5070 (CONUS Albers equal-area) so 100 m is 100 m on the
ground, then the layer is written back in EPSG:4326. ``as_ee_feature_collection``
reprojects to 4326 anyway; writing it that way keeps what the collaborator opens and what
Earth Engine receives identical.

``state`` is written as a constant 'MT' because the sparse extractors route the irrigation
mask on it (IrrMapper west of the Plains, LANID east). Every Mesonet station is in
Montana, so the column is uniform -- but it must exist, or ``state_col`` lookups fail.

Usage:
    python scripts/build_mesonet_swim_fields.py
    python scripts/build_mesonet_swim_fields.py --radius 150 --out /tmp/fields.shp
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

DATA_DIR = Path("/data/ssd2/nisar")
DEFAULT_STATIONS = DATA_DIR / "zh" / "mt_mesonet_smap_stations.csv"
DEFAULT_OUT = DATA_DIR / "zh" / "swim_mesonet" / "gis" / "mt_mesonet_100m.shp"

EQUAL_AREA_CRS = "EPSG:5070"
OUTPUT_CRS = "EPSG:4326"
DEFAULT_RADIUS_M = 100
STATE = "MT"


def build_fields(
    stations: pd.DataFrame, radius_m: int = DEFAULT_RADIUS_M, state: str = STATE
) -> gpd.GeoDataFrame:
    """Buffer station points into equal-area circles and return them in EPSG:4326."""
    missing = {"station", "longitude", "latitude"} - set(stations.columns)
    if missing:
        raise ValueError(f"stations CSV is missing required columns: {sorted(missing)}")

    dupes = stations["station"][stations["station"].duplicated()].tolist()
    if dupes:
        raise ValueError(
            f"duplicate station IDs would collide as feature IDs: {sorted(set(dupes))}"
        )

    points = gpd.GeoDataFrame(
        {"site_id": stations["station"].astype(str), "state": state},
        geometry=[
            Point(float(lon), float(lat))
            for lon, lat in zip(stations["longitude"], stations["latitude"])
        ],
        crs=OUTPUT_CRS,
    )
    buffered = points.to_crs(EQUAL_AREA_CRS)
    buffered["geometry"] = buffered.geometry.buffer(radius_m)
    return buffered.to_crs(OUTPUT_CRS)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stations", type=Path, default=DEFAULT_STATIONS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--radius", type=int, default=DEFAULT_RADIUS_M, help="meters")
    p.add_argument("--state", default=STATE)
    args = p.parse_args(argv)

    stations = pd.read_csv(args.stations)
    fields = build_fields(stations, radius_m=args.radius, state=args.state)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields.to_file(args.out, engine="fiona")
    # .fgb alongside the .shp: the swim exporters read the shapefile, but FlatGeobuf
    # carries the full column names and no sidecar sprawl for everything downstream.
    fgb = args.out.with_suffix(".fgb")
    fields.to_file(fgb, driver="FlatGeobuf", engine="fiona")
    print(
        f"wrote {len(fields)} {args.radius} m field polygons -> {args.out}\n"
        f"  also:    {fgb}\n"
        f"  columns: {list(fields.columns)}\n"
        f"  bounds:  {[round(v, 4) for v in fields.total_bounds]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
