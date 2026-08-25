"""Build a CONUS-filtered NISAR track/frame footprint layer.

Source: the NASA/JPL NISAR reference observation plan Track-Frame DataBase (TFDB) KMZ,
published at https://science.nasa.gov/mission/nisar/data/ and kept verbatim under
`/data/ssd2/nisar/reference/nisar_frames/`. Each Placemark is one track/frame cell (named
`T<track>_F<frame>`) of NISAR's fixed 12-day-repeat planning grid; its KML description carries
the TFDB attribute record (track, frame, passDirection, swath geometry, product flags, ...)
plus a per-frame product list of planned acquisition dates and radar-mode mnemonics.

CONUS boundary comes from the US Census cartographic boundary file cb_2020_us_state_20m,
dissolved over the 48 contiguous states + DC.

Usage:
    uv run python scripts/build_nisar_frame_grid.py
"""

import argparse
import html
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

DATA_DIR = Path("/data/ssd2/nisar")
DEFAULT_KMZ = (
    DATA_DIR
    / "reference/nisar_frames/NISAR_ROP358_TFDB_ObservationPlan_CY2026-20260305.kmz"
)
DEFAULT_OUT = DATA_DIR / "reference/nisar_frames_conus.fgb"
CENSUS_STATES_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_20m.zip"
)

KML_NS = "{http://www.opengis.net/kml/2.2}"
NON_CONUS = {"AK", "HI", "PR", "VI", "GU", "MP", "AS"}

# The TFDB attribute table renders as <tr><th>key</th><td>value</td></tr>. The per-frame
# product list renders its rows either the same way or as <tr><td>date</td><td>mode</td></tr>,
# depending on the frame, so both forms are matched.
ATTR_ROW_RE = re.compile(r"<tr><th[^>]*>([^<]*)</th><td[^>]*>([^<]*)</td></tr>")
PRODUCT_ROW_RE = re.compile(
    r"<tr><td[^>]*>(\d{4}-\d{2}-\d{2})</td><td[^>]*>([^<]*)</td></tr>"
)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TABLE_HEADERS = {
    "Attributes",
    "Product List",
    "Acquisition Date",
    "Radar Mode Mnemonic",
}


def parse_description(desc):
    """Split a Placemark description into (TFDB attributes, acquisition summary)."""
    attrs = {}
    dates, modes = [], set()

    for key, value in ATTR_ROW_RE.findall(desc):
        key = html.unescape(key).strip()
        value = html.unescape(value).strip()
        if DATE_RE.match(key):
            dates.append(key)
            modes.add(value)
        elif key not in TABLE_HEADERS:
            attrs[key] = value

    for date, mode in PRODUCT_ROW_RE.findall(desc):
        dates.append(date)
        modes.add(html.unescape(mode).strip())

    summary = {
        "n_acquisitions": len(dates),
        "first_acq": min(dates) if dates else None,
        "last_acq": max(dates) if dates else None,
        # An empty token means the plan lists an acquisition with no radar-mode mnemonic;
        # it is preserved rather than dropped so the gap stays visible.
        "radar_modes": ";".join(sorted(modes)) if modes else None,
    }
    return attrs, summary


def coerce(series):
    """Cast a text column to bool or numeric when every value permits it."""
    values = set(series.dropna().unique())
    if values and values <= {"True", "False"}:
        return series.map({"True": True, "False": False})
    try:
        return pd.to_numeric(series)
    except (ValueError, TypeError):
        return series


def read_frame_grid(kmz_path):
    """Stream the TFDB KMZ into a GeoDataFrame of track/frame footprints."""
    with zipfile.ZipFile(kmz_path) as zf:
        kml_name = next(n for n in zf.namelist() if n.lower().endswith(".kml"))
        with zf.open(kml_name) as fh:
            records = list(_iter_placemarks(fh))

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    for col in gdf.columns:
        if col == "geometry":
            continue
        if not pd.api.types.is_numeric_dtype(
            gdf[col]
        ) and not pd.api.types.is_bool_dtype(gdf[col]):
            gdf[col] = coerce(gdf[col])
    return gdf


def _iter_placemarks(fh):
    folder = None
    for event, elem in ET.iterparse(io.BufferedReader(fh), events=("start", "end")):
        if event == "start":
            if elem.tag == f"{KML_NS}Folder":
                folder = None
            continue
        if elem.tag == f"{KML_NS}name" and folder is None:
            # The first <name> to close inside a Folder is the folder's own label
            # (Ascending / Descending / legend); Placemark names close after it.
            folder = elem.text
            continue
        if elem.tag != f"{KML_NS}Placemark":
            continue

        name_el = elem.find(f"{KML_NS}name")
        desc_el = elem.find(f"{KML_NS}description")
        coord_el = elem.find(f".//{KML_NS}coordinates")
        snippet_el = elem.find(f"{KML_NS}Snippet")
        if name_el is None or desc_el is None or coord_el is None:
            # Legend graphics carry no footprint or attribute table.
            elem.clear()
            continue

        ring = []
        for token in coord_el.text.split():
            lon, lat, *_ = token.split(",")
            ring.append((float(lon), float(lat)))

        attrs, summary = parse_description(desc_el.text)
        record = {
            "name": name_el.text,
            "folder": folder,
            "snippet": snippet_el.text if snippet_el is not None else None,
            **attrs,
            **summary,
            "geometry": Polygon(ring),
        }
        elem.clear()
        yield record


def conus_boundary(states_path=None):
    """Dissolved polygon of the 48 contiguous states + DC."""
    source = states_path or f"zip+{CENSUS_STATES_URL}"
    states = gpd.read_file(source, engine="fiona")
    conus = states[~states["STUSPS"].isin(NON_CONUS)]
    return conus.to_crs("EPSG:4326").union_all(), len(conus)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kmz", type=Path, default=DEFAULT_KMZ)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--states",
        type=Path,
        default=None,
        help="local cb_2020_us_state_20m shapefile; downloaded from Census if omitted",
    )
    args = ap.parse_args()

    gdf = read_frame_grid(args.kmz)
    print(f"global track/frame cells: {len(gdf)}")
    print(f"pass direction: {gdf['passDirection'].value_counts().to_dict()}")
    print(f"invalid geometries: {(~gdf.geometry.is_valid).sum()}")

    boundary, n_states = conus_boundary(args.states)
    print(f"conus states + DC: {n_states}")

    conus_frames = gdf[gdf.geometry.intersects(boundary)].copy()
    print(f"conus-intersecting cells: {len(conus_frames)}")
    print(f"pass direction: {conus_frames['passDirection'].value_counts().to_dict()}")
    print(
        f"distinct tracks: {conus_frames['track'].nunique()}, "
        f"distinct frames: {conus_frames['frame'].nunique()}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    conus_frames.to_file(args.out, driver="FlatGeobuf", engine="fiona")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
