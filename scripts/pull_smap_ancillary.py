"""Pull the SMAP SPL3SMP_E (9 km EASE-Grid 2.0) soil moisture scene closest in time
to a given date, cropped to an AOI bounding box. Streams only the AOI-cropped slice
of the AM soil_moisture / retrieval_qual_flag arrays — the row/col window is computed
by projecting the AOI into the fixed EASE2_G9km grid definition, so the full
1624x3856 global grid is never pulled.

Usage:
    uv run python scripts/pull_smap_ancillary.py <target_date YYYY-MM-DD> <west> <south> <east> <north> /data/ssd2/nisar/ancillary/
    uv run python scripts/pull_smap_ancillary.py 2026-06-25 -110.30 44.99 -106.12 47.85 /data/ssd2/nisar/ancillary/
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import earthaccess
import h5py
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

SHORT_NAME = "SPL3SMP_E"
GROUP = "Soil_Moisture_Retrieval_Data_AM"
EASE2_EPSG = 6933
PIXEL_SIZE = 9008.055210
# Fixed EASE2_G9km global grid definition (NSIDC), confirmed against this file's own
# per-pixel latitude/longitude values. The data-dependent lat/lon datasets carry a
# -9999 fill wherever there's no land retrieval, so they can't be used to locate a
# row/col window directly -- the grid geometry is derived from this fixed definition
# instead, and the AOI is projected into it.
GRID_X_MIN = -17367530.45
GRID_Y_MAX = 7314540.83
GRID_NCOLS = 3856
GRID_NROWS = 1624
SEARCH_WINDOW_DAYS = 8


def find_closest_granule(target_date: str):
    earthaccess.login(strategy="netrc")
    target = datetime.strptime(target_date, "%Y-%m-%d")  # noqa: DTZ007 - calendar date only, no tz
    start = (target - timedelta(days=SEARCH_WINDOW_DAYS)).strftime("%Y-%m-%d")
    end = (target + timedelta(days=SEARCH_WINDOW_DAYS)).strftime("%Y-%m-%d")
    granules = earthaccess.search_data(
        short_name=SHORT_NAME, temporal=(start, end), count=50
    )
    if not granules:
        raise RuntimeError(f"No {SHORT_NAME} granules found within {start}..{end}")

    def granule_date(g):
        native_id = g["meta"]["native-id"]
        date_str = native_id.split("_")[5]
        return datetime.strptime(date_str, "%Y%m%d")  # noqa: DTZ007 - calendar date only, no tz

    return min(granules, key=lambda g: abs((granule_date(g) - target).days))


def find_window(
    west: float, south: float, east: float, north: float, pad: int = 2
) -> tuple[slice, slice]:
    """Project the AOI corners into the fixed EASE2_G9km grid to get a row/col window."""
    transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{EASE2_EPSG}", always_xy=True
    )
    x_west, y_north = transformer.transform(west, north)
    x_east, y_south = transformer.transform(east, south)

    col_lo = max(int((x_west - GRID_X_MIN) / PIXEL_SIZE) - pad, 0)
    col_hi = min(int((x_east - GRID_X_MIN) / PIXEL_SIZE) + pad, GRID_NCOLS)
    row_lo = max(int((GRID_Y_MAX - y_north) / PIXEL_SIZE) - pad, 0)
    row_hi = min(int((GRID_Y_MAX - y_south) / PIXEL_SIZE) + pad, GRID_NROWS)

    return slice(row_lo, row_hi), slice(col_lo, col_hi)


def stream_smap_crop(
    granule, west: float, south: float, east: float, north: float
) -> tuple[dict, dict]:
    row_sl, col_sl = find_window(west, south, east, north)
    fobj = earthaccess.open([granule])[0]
    with h5py.File(fobj, "r") as f:
        g = f[GROUP]
        data = {
            "soil_moisture": g["soil_moisture"][row_sl, col_sl],
            "retrieval_qf": g["retrieval_qual_flag"][row_sl, col_sl],
            "row_start": row_sl.start,
            "col_start": col_sl.start,
        }
    native_id = granule["meta"]["native-id"]
    date_str = native_id.split("_")[5]
    meta = {"native_id": native_id, "date": date_str}
    return data, meta


def _grid_transform(data: dict):
    west = GRID_X_MIN + data["col_start"] * PIXEL_SIZE
    north = GRID_Y_MAX - data["row_start"] * PIXEL_SIZE
    return from_origin(west, north, PIXEL_SIZE, PIXEL_SIZE)


def write_smap_geotiff(data: dict, meta: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    sm = data["soil_moisture"]
    qf = data["retrieval_qf"]
    recommended = (sm != -9999.0) & ((qf & 1) == 0)
    sm_masked = np.where(recommended, sm, np.nan).astype("float32")

    transform = _grid_transform(data)
    profile = {
        "driver": "GTiff",
        "height": sm.shape[0],
        "width": sm.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": f"EPSG:{EASE2_EPSG}",
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
    }
    out_path = out_dir / f"smap_l3_e_{meta['date']}_soil_moisture.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(sm_masked, 1)
        dst.set_band_description(1, "soil_moisture_AM_recommended_m3m3")
    return out_path


if __name__ == "__main__":
    target_date = sys.argv[1]
    west, south, east, north = (float(v) for v in sys.argv[2:6])
    out_dir = Path(sys.argv[6])

    granule = find_closest_granule(target_date)
    print(f"Closest SPL3SMP_E granule: {granule['meta']['native-id']}")
    data, meta = stream_smap_crop(granule, west, south, east, north)
    print(f"Cropped grid shape: {data['soil_moisture'].shape}")
    out_path = write_smap_geotiff(data, meta, out_dir)
    print(f"Wrote {out_path}")
