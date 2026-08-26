"""Fixed EASE2_G9km global grid definition and station indexing.

Single source of truth for the SMAP L3 9 km grid math: the SMAP pull
(pull_smap_mesonet.py), the pixel land-cover sampler
(sample_smap_pixel_landcover.py), and the grid polygon builder
(build_smap_grid_fgb.py) all import from here, so the cell a station is assigned
to is identical across scoring, characterization, and visualization by
construction. Grid constants are the fixed NSIDC EASE2_G9km definition, verified
against granule geolocation in pull_smap_ancillary.py.

Kept dependency-light (pandas + pyproj only) so a collaborator running just the
Earth Engine extraction does not need the SMAP streaming stack (earthaccess,
h5py) installed.
"""

from pathlib import Path

import pandas as pd
from pyproj import Transformer

EASE2_EPSG = 6933
PIXEL_SIZE = 9008.055210
GRID_X_MIN = -17367530.45
GRID_Y_MAX = 7314540.83
GRID_NCOLS = 3856
GRID_NROWS = 1624


def station_grid_cells(stations_csv: Path) -> pd.DataFrame:
    """Station -> (row, col) in the fixed EASE2_G9km global grid."""
    stations = pd.read_csv(stations_csv)[["station", "longitude", "latitude"]]
    transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{EASE2_EPSG}", always_xy=True
    )
    x, y = transformer.transform(
        stations["longitude"].to_numpy(), stations["latitude"].to_numpy()
    )
    stations["col"] = ((x - GRID_X_MIN) / PIXEL_SIZE).astype(int)
    stations["row"] = ((GRID_Y_MAX - y) / PIXEL_SIZE).astype(int)
    bad = (
        (stations["col"] < 0)
        | (stations["col"] >= GRID_NCOLS)
        | (stations["row"] < 0)
        | (stations["row"] >= GRID_NROWS)
    )
    if bad.any():
        raise ValueError(f"{bad.sum()} stations fall outside the EASE2_G9km grid")
    return stations


def bounding_window(cells: pd.DataFrame, pad: int = 2) -> tuple[slice, slice]:
    """One row/col window covering every station, with a small pad."""
    row_lo = max(int(cells["row"].min()) - pad, 0)
    row_hi = min(int(cells["row"].max()) + pad + 1, GRID_NROWS)
    col_lo = max(int(cells["col"].min()) - pad, 0)
    col_hi = min(int(cells["col"].max()) + pad + 1, GRID_NCOLS)
    return slice(row_lo, row_hi), slice(col_lo, col_hi)


def cell_rectangle(row: int, col: int) -> list:
    """[xmin, ymin, xmax, ymax] of one EASE2_G9km cell, EPSG:6933 meters."""
    xmin = GRID_X_MIN + col * PIXEL_SIZE
    xmax = xmin + PIXEL_SIZE
    ymax = GRID_Y_MAX - row * PIXEL_SIZE
    ymin = ymax - PIXEL_SIZE
    return [xmin, ymin, xmax, ymax]
