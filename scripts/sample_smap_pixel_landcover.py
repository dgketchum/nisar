"""Sample land cover over each station's EASE2_G9km SMAP pixel, via Earth Engine.

Why this exists (see notes/proposal_prelim_panel_2026-08-26.md): panel 1's
representativeness filter and panel 3's keystone figure both need to know what the 9 km
SMAP L3 cell *containing* each Montana Mesonet station is made of -- not what a buffer
around the probe is made of. The existing CDL sampling (label_conus_cdl_cohort.py,
screen_station_siting.py) characterizes the probe's neighborhood at 100/500 m; this
characterizes the satellite pixel SMAP actually reported. The cell is computed with the
identical fixed EASE2_G9km grid definition the SMAP pull indexed against
(ease2_grid.station_grid_cells), so the cell characterized here is the cell scored in
sme2_vs_smap_comparison.csv by construction.

Local cover is re-sampled here too, at the point and a 100 m buffer, from the *precise*
station coordinates in mt_mesonet_stations.csv. The existing per-station labels in
conus_station_landcover_cdl.csv were computed on the CONUS panel's 2-decimal Mesonet
coordinates (up to ~700 m off), so a station-vs-pixel contrast built on them would inherit
that defect; sampling both sides fresh from one coordinate source removes it.

Cells shared by several stations are sampled once and joined back. Reducers follow the
existing scripts exactly: the screen_station_siting._stack() image (CDL `cultivated` 2023
+ IrrMapper/LANID irrigation, west/east routed) reduced to means, and the CDL 2025
crop-class frequency histogram regrouped with the same GROUPS. Entropy over the group
fractions is written alongside so downstream heterogeneity metrics have a ready-made
scalar.

Two-step design, so the EE work can run on a collaborator's quota (zh_commands.sh
pattern, as in swim-mtdnrc):

* submit (default) -- submits five table-export tasks to GCS
  (gs://<bucket>/<fn-prefix>/<table>.csv) and exits; no local data dir needed. This is
  what the collaborator runs, with --project <their EE project>.
* ``--assemble <dir>`` -- no Earth Engine at all: reads the five exported CSVs
  (downloaded from the bucket) and assembles the finished per-station CSV locally.

Usage:
    python scripts/sample_smap_pixel_landcover.py --stations zh/mt_mesonet_smap_stations.csv \
        --bucket wudr --project ee-hoylman
    uv run python scripts/sample_smap_pixel_landcover.py /data/ssd2/nisar/ \
        --assemble /data/ssd2/nisar/ee_exports/smap_pixel_landcover
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
from ease2_grid import cell_rectangle, station_grid_cells
from screen_station_siting import CDL, CROP_YEAR, EE_PROJECT, GROUPS, SCALE_M, _stack

EASE2_CRS = "EPSG:6933"
LOCAL_RADIUS_M = 100
TILE_SCALE = 4

# The five exported tables and their key columns.
TABLES = {
    "pt_means": "station",
    "buf100_means": "station",
    "cell_means": "cell",
    "buf100_hist": "station",
    "cell_hist": "cell",
}
MEAN_RENAMES = {
    "pt_means": {"cult": "cult_f_pt", "irr": "irr_f_pt"},
    "buf100_means": {"cult": "cult_f_100", "irr": "irr_f_100"},
    "cell_means": {"cult": "cult_f_9km", "irr": "irr_f_9km"},
}
HIST_SUFFIXES = {"buf100_hist": "100", "cell_hist": "9km"}


def _entropy(fracs: dict) -> float:
    return -sum(p * math.log(p) for p in fracs.values() if p > 0)


def _histogram_record(props: dict, key: str, suffix: str) -> dict:
    """Regroup one frequencyHistogram result into GROUPS fractions, mode and entropy."""
    hist = props.get("histogram") or {}
    total = sum(hist.values())
    rec = {key: props[key], f"cdl_n_px_{suffix}": total}
    if not total:
        raise ValueError(
            f"empty CDL histogram for {key}={props[key]!r} -- geometry missed the CDL "
            f"domain; investigate before writing a land-cover record"
        )
    fracs = {int(float(k)): v / total for k, v in hist.items()}
    rec[f"cdl_mode_{suffix}"] = max(fracs, key=fracs.get)
    rec[f"cdl_mode_frac_{suffix}"] = fracs[rec[f"cdl_mode_{suffix}"]]
    group_fracs = {
        name: sum(v for k, v in fracs.items() if k in classes)
        for name, classes in GROUPS.items()
    }
    for name, frac in group_fracs.items():
        rec[f"f_{name}_{suffix}"] = frac
    rec[f"entropy_{suffix}"] = _entropy(group_fracs)
    rec[f"cdl_hist_{suffix}"] = json.dumps(
        {k: round(v, 4) for k, v in sorted(fracs.items()) if v > 0}
    )
    return rec


def submit_exports(
    stations: pd.DataFrame, project: str, bucket: str, fn_prefix: str
) -> None:
    """Submit the five reduceRegions tables as EE export tasks to GCS.

    Each lands at gs://<bucket>/<fn_prefix>/<table>.csv; download the directory and
    finish with --assemble. Runs on whichever EE project is passed, so a collaborator
    can execute this leg on their quota (see zh_commands.sh).
    """
    import ee

    ee.Initialize(project=project)
    stack = _stack()
    crop = ee.Image(f"{CDL}/{CROP_YEAR}").select("cropland")
    mean = ee.Reducer.mean()
    hist = ee.Reducer.frequencyHistogram()

    pts = [
        ee.Feature(
            ee.Geometry.Point([float(r.longitude), float(r.latitude)]),
            {"station": r.station},
        )
        for r in stations.itertuples()
    ]
    bufs = [f.buffer(LOCAL_RADIUS_M) for f in pts]
    cells = stations[["row", "col"]].drop_duplicates()
    print(f"{len(stations)} stations in {len(cells)} unique EASE2_G9km cells")
    cell_feats = [
        ee.Feature(
            ee.Geometry.Rectangle(cell_rectangle(r.row, r.col), EASE2_CRS, False),
            {"cell": f"{r.row}_{r.col}"},
        )
        for r in cells.itertuples()
    ]

    jobs = {
        "pt_means": (stack, pts, mean, ["station", "cult", "irr"]),
        "buf100_means": (stack, bufs, mean, ["station", "cult", "irr"]),
        "cell_means": (stack, cell_feats, mean, ["cell", "cult", "irr"]),
        "buf100_hist": (crop, bufs, hist, ["station", "histogram"]),
        "cell_hist": (crop, cell_feats, hist, ["cell", "histogram"]),
    }
    for name, (image, features, reducer, selectors) in jobs.items():
        data = image.reduceRegions(
            collection=ee.FeatureCollection(features),
            reducer=reducer,
            scale=SCALE_M,
            tileScale=TILE_SCALE,
        )
        task = ee.batch.Export.table.toCloudStorage(
            data,
            description=f"smap_pixlc_{name}",
            bucket=bucket,
            fileNamePrefix=f"{fn_prefix}/{name}",
            fileFormat="CSV",
            selectors=selectors,
        )
        task.start()
        print(f"  started smap_pixlc_{name} -> gs://{bucket}/{fn_prefix}/{name}.csv")
    print(
        f"\n5 export tasks submitted on project {project!r}; monitor at "
        f"https://code.earthengine.google.com/tasks"
    )


def _parse_histogram(raw, key_val, suffix: str) -> dict:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            f"missing histogram for {key_val!r} ({suffix}) in exported CSV -- the "
            f"geometry missed the CDL domain; investigate before assembling"
        )
    return json.loads(raw)


def assemble_exports(stations: pd.DataFrame, exports_dir: Path) -> pd.DataFrame:
    """Build the finished CSV from the five bucket-exported tables. No Earth Engine."""
    raw = {}
    for name in TABLES:
        path = exports_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- download the exports first, e.g. "
                f"gsutil -m cp 'gs://<bucket>/<fn-prefix>/*.csv' {exports_dir}/"
            )
        raw[name] = pd.read_csv(path)

    means = {
        name: raw[name].rename(columns=MEAN_RENAMES[name]) for name in MEAN_RENAMES
    }
    hists = {}
    for name, suffix in HIST_SUFFIXES.items():
        key = TABLES[name]
        hists[name] = pd.DataFrame(
            [
                _histogram_record(
                    {
                        key: r[key],
                        "histogram": _parse_histogram(r["histogram"], r[key], suffix),
                    },
                    key,
                    suffix,
                )
                for r in raw[name].to_dict("records")
            ]
        )

    out = stations.copy()
    out["cell"] = out["row"].astype(str) + "_" + out["col"].astype(str)
    for part in (means["pt_means"], means["buf100_means"], hists["buf100_hist"]):
        out = out.merge(part, on="station", how="left", validate="1:1")
    for part in (means["cell_means"], hists["cell_hist"]):
        out = out.merge(part, on="cell", how="left", validate="m:1")

    fraction_cols = [c for c in out.columns if c.startswith(("cult_f_", "irr_f_"))]
    nulls = out[fraction_cols].isna().any(axis=1)
    if nulls.any():
        raise ValueError(
            f"{int(nulls.sum())} station(s) came back with null cover fractions -- "
            f"Earth Engine returned no value, which means a geometry missed the "
            f"CDL/irrigation domain, not that the cover is zero: "
            f"{sorted(out.loc[nulls, 'station'])}"
        )
    return out


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "data_dir",
        type=Path,
        nargs="?",
        default=None,
        help="data root; optional for the submit step when --stations is given",
    )
    p.add_argument(
        "--stations",
        type=Path,
        default=None,
        help="station CSV with station/longitude/latitude "
        "(default: <data_dir>/mesonet/mt_mesonet_stations.csv)",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--project", default=EE_PROJECT, help="EE project ID")
    p.add_argument("--bucket", default="wudr")
    p.add_argument("--fn-prefix", default="nisar/smap_pixel_landcover")
    p.add_argument(
        "--assemble",
        type=Path,
        default=None,
        help="directory holding the five bucket-exported CSVs; assembles the "
        "finished CSV locally with no Earth Engine calls",
    )
    return p.parse_args(argv)


def main(argv) -> int:
    a = parse_args(argv)
    stations_csv = a.stations or (
        a.data_dir / "mesonet/mt_mesonet_stations.csv" if a.data_dir else None
    )
    if stations_csv is None:
        raise SystemExit("provide data_dir or --stations")
    stations = station_grid_cells(stations_csv)

    if a.assemble is None:
        submit_exports(stations, a.project, a.bucket, a.fn_prefix)
        return 0

    result = assemble_exports(stations, a.assemble)
    out_path = a.out or (
        a.data_dir / "reference/smap_pixel_landcover.csv" if a.data_dir else None
    )
    if out_path is None:
        raise SystemExit("provide data_dir or --out to write the assembled CSV")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(result)} stations)")
    print(
        f"9 km cells: cultivated fraction median "
        f"{result['cult_f_9km'].median():.2f}, irrigated fraction median "
        f"{result['irr_f_9km'].median():.3f}, entropy median "
        f"{result['entropy_9km'].median():.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
