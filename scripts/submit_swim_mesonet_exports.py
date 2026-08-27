"""Submit the SWIM-RS input exports for the Montana Mesonet stations to Earth Engine.

SUPERSEDED by ``extract_swim_mesonet.py``, which follows Example 5's synchronous getInfo
approach. Read that script's header before using this one: the ETf leg here is ~7,400
batch tasks per model for this panel, and the getInfo path needs no batch tasks at all.
This remains only for the legs where a server-side batch export is genuinely wanted.

Handoff leg. This is the script a collaborator runs on their own Earth Engine project and
quota; it writes every table to ``gs://<bucket>/<file-prefix>/`` and exits as soon as the
batch tasks are submitted. Nothing is read back here -- retrieval and assembly happen on
our side, from the bucket.

Why this exists rather than ``swim extract``: swim-rs pins the Earth Engine project in
code, not in configuration. ``ee_utils.is_authorized()`` defaults to
``project="ee-dgketchum"`` and both ``swimrs/cli.py`` and the bundled examples call it
bare, while the ``[earth_engine]`` TOML section carries only ``bucket`` -- there is no
project key to override. A collaborator running the stock CLI would therefore submit
against a project they have no access to. Here ``--project`` is required and is the only
thing that ever initializes Earth Engine.

``ee.Authenticate()`` is deliberately never called. On a workstation holding a
collaborator's credential file it would overwrite that refresh token; the caller is
expected to have authenticated already, out of band.

Inputs are the four SWIM-RS remote-sensing legs. Meteorology is not here: GridMET comes
from THREDDS, needs no Earth Engine and no quota, and is run on our side.

* properties -- CDL, irrigation fraction, SSURGO, land cover (one table each)
* snodas    -- daily SWE
* ndvi      -- Landsat, plus Sentinel-2 from 2017 (its record start)
* etf       -- one OpenET model per pass

Geometry is the 100 m station footprint layer from ``build_mesonet_swim_fields.py``;
``buffer`` is left None because the polygons are already buffered.

``check_dir`` is None throughout. It is a *local* skip-if-present check, and the
collaborator has no local mirror of our outputs; on a machine without authenticated
gcloud a bucket-side check is not possible either (see the handoff README).

Usage:
    python scripts/submit_swim_mesonet_exports.py --project <ee-project> --bucket <bucket>
    python scripts/submit_swim_mesonet_exports.py --project <ee-project> --bucket <bucket> \
        --steps properties,snodas
"""

import argparse
import sys
from pathlib import Path

DATA_DIR = Path("/data/ssd2/nisar")
DEFAULT_SHAPEFILE = DATA_DIR / "zh" / "swim_mesonet" / "gis" / "mt_mesonet_100m.shp"

FEATURE_ID = "site_id"
STATE_COL = "state"
PROJECT_NAME = "mt_mesonet"
DEFAULT_PREFIX = "nisar/mt_mesonet_swim"
DEFAULT_START_YEAR = 2010
DEFAULT_END_YEAR = 2025
SENTINEL_START_YEAR = 2017

DEFAULT_MODELS = "disalexi,eemetric,ensemble,geesebal,ptjpl,sims,ssebop"
DEFAULT_MASKS = "no_mask"
STEPS = ("properties", "snodas", "ndvi", "etf")


def _chunk_stations(shapefile: Path, chunk: str | None) -> list[str] | None:
    """Return the i-th of n contiguous station-ID chunks, or None for every station.

    Chunking is on the sorted ID list, so the same ``i/n`` always names the same stations
    -- a session that dies partway can be repeated without guessing what it covered.
    """
    if chunk is None:
        return None

    import geopandas as gpd

    try:
        index, total = (int(v) for v in chunk.split("/"))
    except ValueError:
        raise SystemExit(f"--chunk must look like '2/7', got {chunk!r}") from None
    if not 1 <= index <= total:
        raise SystemExit(f"--chunk index {index} out of range for {total} chunks")

    ids = sorted(gpd.read_file(shapefile)[FEATURE_ID].astype(str))
    size = -(-len(ids) // total)
    selected = ids[(index - 1) * size : index * size]
    print(f"chunk {index}/{total}: {len(selected)} of {len(ids)} stations")
    return selected


def submit_properties(args) -> None:
    """CDL, irrigation fraction, SSURGO and land cover -- four small table exports."""
    from swimrs.data_extraction.ee.ee_props import (
        get_cdl,
        get_irrigation,
        get_landcover,
        get_ssurgo,
    )

    if args.select is not None:
        # get_cdl takes no `select`, so chunking these four would split three tables and
        # not the fourth. They are one small task each; submit them whole or not at all.
        print("  properties: --chunk does not apply here, submitting all stations")

    shp = str(args.shapefile)
    common = {
        "selector": FEATURE_ID,
        "dest": "bucket",
        "bucket": args.bucket,
        "file_prefix": args.file_prefix,
    }
    get_cdl(shp, f"{PROJECT_NAME}_cdl", **common)
    get_irrigation(shp, f"{PROJECT_NAME}_irr", lanid=True, **common)
    get_ssurgo(shp, f"{PROJECT_NAME}_ssurgo", **common)
    get_landcover(shp, f"{PROJECT_NAME}_landcover", out_fmt="CSV", **common)
    print("  properties: 4 tasks submitted (cdl, irr, ssurgo, landcover)")


def submit_snodas(args) -> None:
    from swimrs.data_extraction.ee.snodas_export import sample_snodas_swe

    sample_snodas_swe(
        feature_coll=str(args.shapefile),
        bucket=args.bucket,
        check_dir=None,
        start_yr=args.start_year,
        end_yr=args.end_year,
        feature_id=FEATURE_ID,
        select=args.select,
        dest="bucket",
        file_prefix=args.file_prefix,
    )
    print("  snodas: tasks submitted")


def submit_ndvi(args) -> None:
    """Landsat over the full range, Sentinel-2 from its 2017 record start."""
    from swimrs.data_extraction.ee.ndvi_export import sparse_sample_ndvi

    for mask in args.masks:
        for satellite, start in (
            ("landsat", args.start_year),
            ("sentinel", max(SENTINEL_START_YEAR, args.start_year)),
        ):
            sparse_sample_ndvi(
                str(args.shapefile),
                bucket=args.bucket,
                dest="bucket",
                mask_type=mask,
                check_dir=None,
                start_yr=start,
                end_yr=args.end_year,
                feature_id=FEATURE_ID,
                satellite=satellite,
                state_col=STATE_COL,
                select=args.select,
                file_prefix=args.file_prefix,
            )
            print(f"  ndvi: {satellite} {mask} {start}-{args.end_year} submitted")


def submit_etf(args) -> None:
    """One pass per OpenET model.

    Sparse mode submits one task per field-year (``etf_export._export_etf_sparse``) and
    issues a blocking scene query per field-year before each one, so a full ensemble pass
    over every station is tens of thousands of tasks -- run it one model at a time, and
    use --chunk to stay under the project's queued-task ceiling. Clustered mode discovers
    scenes once per year over the union of all stations and batches 30 scenes per task,
    which is ~20x fewer tasks but reduces statewide imagery per task.
    """
    from swimrs.data_extraction.ee.etf_export import export_etf

    for model in args.models:
        for mask in args.masks:
            export_etf(
                str(args.shapefile),
                model=model,
                feature_id=FEATURE_ID,
                select=args.select,
                start_yr=args.start_year,
                end_yr=args.end_year,
                mask_type=mask,
                check_dir=None,
                state_col=STATE_COL,
                buffer=None,
                dest="bucket",
                bucket=args.bucket,
                file_prefix=args.file_prefix,
                clustered=args.clustered,
                source=args.etf_source,
            )
            print(f"  etf: {model} {mask} {args.start_year}-{args.end_year} submitted")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--project", required=True, help="Earth Engine project to bill")
    p.add_argument("--bucket", required=True, help="GCS bucket for outputs")
    p.add_argument("--shapefile", type=Path, default=DEFAULT_SHAPEFILE)
    p.add_argument("--file-prefix", default=DEFAULT_PREFIX)
    p.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    p.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    p.add_argument(
        "--models", default=DEFAULT_MODELS, help="comma-separated OpenET models"
    )
    p.add_argument("--masks", default=DEFAULT_MASKS, help="no_mask, irr and/or inv_irr")
    p.add_argument(
        "--etf-source",
        default="foss",
        choices=("foss", "asset"),
        help="'foss' computes ETf with the openet-* packages; 'asset' reads cached "
        "ImageCollections under projects/ee-dgketchum, which a collaborator project "
        "cannot read unless that collection has been shared with them",
    )
    p.add_argument(
        "--clustered",
        action="store_true",
        help="one ETf task per year over all stations at once, instead of one per "
        "field-year; ~20x fewer tasks, but each task reduces statewide imagery",
    )
    p.add_argument(
        "--chunk",
        default=None,
        help="submit only the i-th of n station chunks, e.g. '2/7', to stay under the "
        "project's queued-task ceiling across sessions",
    )
    p.add_argument(
        "--steps", default="all", help=f"comma-separated: {', '.join(STEPS)}"
    )
    args = p.parse_args(argv)

    if not args.shapefile.exists():
        raise SystemExit(
            f"{args.shapefile} not found -- build it with build_mesonet_swim_fields.py, "
            f"or point --shapefile at the copy shipped in this handoff"
        )

    steps = (
        STEPS
        if args.steps == "all"
        else tuple(s.strip() for s in args.steps.split(","))
    )
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise SystemExit(
            f"unknown step(s) {sorted(unknown)}; choose from {list(STEPS)}"
        )

    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.masks = [m.strip() for m in args.masks.split(",") if m.strip()]
    args.select = _chunk_stations(args.shapefile, args.chunk)

    # The only Earth Engine initialization in this script, and never Authenticate().
    import ee

    ee.Initialize(project=args.project)
    print(
        f"submitting on project {args.project!r} -> gs://{args.bucket}/{args.file_prefix}/"
    )

    runners = {
        "properties": submit_properties,
        "snodas": submit_snodas,
        "ndvi": submit_ndvi,
        "etf": submit_etf,
    }
    for step in steps:
        print(f"{step}:")
        runners[step](args)

    print(
        f"\nsubmitted on project {args.project!r}; monitor at "
        f"https://code.earthengine.google.com/tasks"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
