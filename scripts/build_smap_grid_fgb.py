"""Write the EASE2_G9km SMAP grid over the Mesonet domain as a polygon FlatGeobuf.

For choosing the panel-2 pixel-fields coverage visually (see
notes/proposal_prelim_panel_2026-08-26.md): every SMAP L3 9 km cell in a padded window
around the Montana Mesonet stations, from the same fixed EASE2_G9km grid definition the
SMAP pull and the pixel land-cover sampler index against -- so a cell chosen off this
layer is, by construction, the cell scored in the head-to-head and characterized in
smap_pixel_landcover.csv.

Attributes: ``cell`` (row_col id), ``row``/``col``, the count and ids of Mesonet stations
inside each cell, and -- when reference/smap_pixel_landcover.csv exists -- the sampled
cell cover (cultivated/irrigated fraction, dominant-group entropy). Cells are built as
exact rectangles in EASE2 (EPSG:6933), edge-densified, and written in EPSG:4326 for
overlay with field layers.

Usage:
    uv run python scripts/build_smap_grid_fgb.py /data/ssd2/nisar/
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from ease2_grid import bounding_window, cell_rectangle, station_grid_cells
from shapely.geometry import box

WINDOW_PAD_CELLS = 3
DENSIFY_M = 900.0  # vertices every ~900 m so cell edges reproject as true curves
COVER_COLS = [
    "cult_f_9km",
    "irr_f_9km",
    "entropy_9km",
    "cdl_mode_9km",
    "cdl_mode_frac_9km",
]


def cell_geometry(row: int, col: int):
    return box(*cell_rectangle(row, col)).segmentize(DENSIFY_M)


def build(data_dir: Path) -> int:
    stations = station_grid_cells(data_dir / "mesonet/mt_mesonet_stations.csv")
    row_sl, col_sl = bounding_window(stations, pad=WINDOW_PAD_CELLS)

    records = [
        {"cell": f"{r}_{c}", "row": r, "col": c, "geometry": cell_geometry(r, c)}
        for r in range(row_sl.start, row_sl.stop)
        for c in range(col_sl.start, col_sl.stop)
    ]
    gdf = gpd.GeoDataFrame(records, crs="EPSG:6933").to_crs("EPSG:4326")

    stations["cell"] = stations["row"].astype(str) + "_" + stations["col"].astype(str)
    per_cell = stations.groupby("cell")["station"].agg(
        n_stations="count", stations=lambda s: ";".join(sorted(s))
    )
    gdf = gdf.merge(per_cell, on="cell", how="left")
    gdf["n_stations"] = gdf["n_stations"].fillna(0).astype(int)

    cover_path = data_dir / "reference/smap_pixel_landcover.csv"
    if cover_path.exists():
        cover = (
            pd.read_csv(cover_path)
            .drop_duplicates("cell")
            .set_index("cell")[COVER_COLS]
        )
        gdf = gdf.merge(cover, on="cell", how="left")
        n_covered = gdf["cult_f_9km"].notna().sum()
        print(
            f"Joined sampled cover for {n_covered}/{len(gdf)} cells from {cover_path}"
        )
    else:
        print(
            f"{cover_path} not found -- writing geometry and station attribution only; "
            f"rerun after sample_smap_pixel_landcover.py to attach cell cover"
        )

    out_path = data_dir / "reference/smap_ease2_9km_grid.fgb"
    gdf.to_file(out_path, driver="FlatGeobuf", engine="fiona")
    print(
        f"Wrote {out_path}: {len(gdf)} cells "
        f"(rows {row_sl.start}:{row_sl.stop}, cols {col_sl.start}:{col_sl.stop}), "
        f"{int((gdf['n_stations'] > 0).sum())} holding Mesonet stations"
    )
    return 0


if __name__ == "__main__":
    sys.exit(build(Path(sys.argv[1])))
