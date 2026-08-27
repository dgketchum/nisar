"""Extract the SWIM-RS input set for the Montana Mesonet stations, via synchronous getInfo.

This follows Example 5 (``swim-rs/examples/5_Flux_Ensemble/data_extract.py``) rather than
the batch-export path, because batch export does not scale to this panel. The sparse ETf
exporter submits one task per station-year and issues a blocking scene query before each:
231 stations x 16 years x ~2 scene batches is ~7,400 tasks for ONE model, ~45,000 for the
six-model ensemble -- past the queued-task ceiling of a single project, and many hours
merely to submit.

Example 5's approach removes the batch layer entirely. Each site-year is a stacked
``reduceRegion`` resolved with ``.getInfo()``, written straight to a local CSV. Its own
docstring measures ~0.7 s per site-year for the fraction models (ssebop, sims, eemetric)
and ~2 s for the ET-denominated ones (ptjpl, geesebal, disalexi), which puts this panel at
roughly 45 min and 2 h per model respectively -- call it 8-9 hours for all six, single
threaded, with no batch tasks at all. ``extract_model`` checkpoints to the output CSV and
merges on reload, so a run that dies resumes where it stopped rather than restarting.

Two further consequences, both good:

* ETf comes from the public OpenET v2.1 collections (``projects/openet/assets/...``), not
  the cached ``projects/ee-dgketchum/assets/openet_etf/v2_1`` mirror. Nothing needs to be
  shared with anyone, and the ``openet`` FOSS packages are not required either.
* Output is local CSVs, so no bucket and no authenticated ``gsutil`` are involved.

Note the ensemble is six models, not seven: ``ensemble`` is derived from the members, not
a source collection, so it is computed downstream rather than extracted here.

Earth Engine project: every ``extract_*`` in Example 5 calls ``is_authorized()`` bare, and
that helper defaults to ``project="ee-dgketchum"`` (``ee_utils.py``), which would silently
re-initialize onto a project the caller may not be able to read. ``--project`` is required
here and is patched over that helper, so the project actually used is the one passed.
``ee.Authenticate()`` is never called; the caller is expected to have authenticated out of
band.

Usage:
    python scripts/extract_swim_mesonet.py --project <ee-project> --steps properties
    python scripts/extract_swim_mesonet.py --project <ee-project> --steps etf \
        --models ssebop
"""

import argparse
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATA_DIR = Path("/data/ssd2/nisar")
DEFAULT_CONFIG = DATA_DIR / "swim" / "mt_mesonet" / "mt_mesonet.toml"
DEFAULT_SWIM_RS = Path.home() / "code" / "swim-rs"
EXAMPLE_5 = "examples/5_Flux_Ensemble"

# 'ensemble' is intentionally absent: it is computed from the members downstream.
ALL_MODELS = ("ssebop", "sims", "eemetric", "geesebal", "ptjpl", "disalexi")
ETF_OUT_DIRNAME = "etf_v21_openet_eto"
STEPS = ("properties", "snodas", "ndvi", "etf", "refet")


def load_example_5(swim_rs: Path, project: str):
    """Import Example 5's extraction module, with Earth Engine pinned to `project`.

    The module is not importable as a package -- it is a script inside the examples tree
    -- so its directory goes on sys.path the way swim-rs's own zh_commands.sh does it.

    ``is_authorized`` is replaced rather than called: Example 5 invokes it at the top of
    every extract step, and the stock helper hardcodes a default project. Patching the
    name in the module namespace makes `--project` authoritative for every step without
    editing swim-rs.
    """
    example_dir = swim_rs / EXAMPLE_5
    if not example_dir.is_dir():
        raise SystemExit(
            f"{example_dir} not found -- pass --swim-rs with the path to the swim-rs repo"
        )
    sys.path.insert(0, str(example_dir))

    import data_extract as ex5
    import ee

    ee.Initialize(project=project)

    def _authorized(*_args, **_kwargs) -> bool:
        ee.Initialize(project=project)
        return True

    ex5.is_authorized = _authorized
    return ex5


def _site_year(ex5, model, coll_path, geometry, year, max_retries):
    """One site-year, retried, raising on final failure instead of returning {}.

    Example 5's ``_with_retries`` swallows the last exception and returns an empty dict,
    so a throttled site-year lands in the checkpoint as NaN and is indistinguishable
    from genuinely absent imagery. That is tolerable single-threaded and dangerous at
    concurrency, where throttling is the expected failure -- silent gaps would read as
    real data. Here the final failure propagates so it can be counted and reported.
    """
    fn = (
        ex5._extract_et_model if model in ex5.ET_MODELS else ex5._extract_fraction_model
    )
    for attempt in range(max_retries):
        try:
            return fn(coll_path, geometry, year)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    return {}


def extract_ndvi_threaded(cfg, ex5, sites, threads, max_retries, get_sentinel=True):
    """NDVI per site-year, fanned across threads.

    Example 5's ``extract_ndvi`` walks sites sequentially, which is the dominant cost of
    a whole-network run -- roughly 6 minutes per site across both instruments, so 221
    stations is about a day of wall clock. It writes one CSV per site, so unlike the ETf
    checkpoint there is no shared file to race on: each thread owns its own output and
    the layout below matches ``extract_ndvi`` exactly (rows=site, columns=scene IDs,
    existing values merged and re-extracted values winning) so
    ``container.ingest.ndvi()`` reads either one identically.
    """
    import pandas as pd

    geometries, fids = ex5._load_polygons(
        cfg.fields_shapefile, feature_id=cfg.feature_id_col, select=sites
    )
    instruments = ["landsat"] + (["sentinel"] if get_sentinel else [])
    lock = threading.Lock()

    for instrument in instruments:
        base_dir = (
            cfg.landsat_dir if instrument == "landsat" else ex5._sentinel_dir(cfg)
        )
        start_yr = (
            cfg.start_dt.year
            if instrument == "landsat"
            else max(ex5.SENTINEL_START_YEAR, cfg.start_dt.year)
        )
        years = list(range(start_yr, cfg.end_dt.year + 1))
        out_dir = Path(base_dir) / "getinfo" / "ndvi" / "no_mask"
        out_dir.mkdir(parents=True, exist_ok=True)
        done, failures = [0], []
        t0 = time.time()
        print(
            f"  {instrument}: {len(fids)} sites x {len(years)} years on {threads} threads"
        )

        def _one_site(
            fid,
            _inst=instrument,
            _years=years,
            _out=out_dir,
            _done=done,
            _failures=failures,
            _t0=t0,
        ):
            out_path = _out / f"ndvi_{fid}_no_mask.csv"
            site_values = {}
            if out_path.exists():
                existing = pd.read_csv(out_path, index_col=0)
                if len(existing):
                    site_values = existing.iloc[0].dropna().to_dict()
            for year in _years:
                for attempt in range(max_retries):
                    try:
                        site_values.update(
                            ex5._extract_ndvi_site_year(geometries[fid], year, _inst)
                        )
                        break
                    except Exception as e:  # noqa: BLE001 - EE throttling raises many types
                        if attempt == max_retries - 1:
                            with lock:
                                _failures.append((fid, year, str(e)[:80]))
                        else:
                            time.sleep(2 ** (attempt + 1))
            df = pd.DataFrame(
                [site_values], index=pd.Index([fid], name=cfg.feature_id_col)
            )
            df = df[sorted(df.columns)]
            df.to_csv(out_path)
            with lock:
                _done[0] += 1
                if _done[0] % 20 == 0 or _done[0] == len(fids):
                    print(f"    {_done[0]}/{len(fids)} sites  {time.time() - _t0:.0f}s")
            return int(df.notna().sum().sum())

        with ThreadPoolExecutor(max_workers=threads) as pool:
            n_values = sum(
                f.result()
                for f in as_completed([pool.submit(_one_site, s) for s in fids])
            )

        elapsed = time.time() - t0
        print(
            f"  {instrument}: {elapsed:.0f}s "
            f"({elapsed / (len(fids) * len(years)):.2f}s per site-year), "
            f"{n_values:,} values, {len(failures)} failed site-years"
        )
        for fid, year, err in failures[:5]:
            print(f"    FAILED {fid} {year}: {err}")


def extract_etf(cfg, ex5, models, sites, out_dir: Path, start_yr, threads, max_retries):
    """ETf for each OpenET v2.1 model, one CSV per model, checkpointed.

    Example 5's own ``extract_etf_v21`` is not reused: it writes into the example's
    directory, starts each model at its earliest available year (1999), and walks sites
    sequentially. These calls are ~100% network wait, so sites are fanned out across
    threads -- threads rather than processes, so one ``ee.Initialize`` is shared and the
    parent stays the only writer. Workers return per-site dicts; nothing but the parent
    touches the CSV, so concurrent writers cannot interleave rows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    geometries, fids = ex5._load_polygons(
        cfg.fields_shapefile, feature_id=cfg.feature_id_col, select=sites
    )
    years = list(range(start_yr, cfg.end_dt.year + 1))
    lock = threading.Lock()

    for model in models:
        coll_path = ex5.OPENET_V21[model]
        checkpoint = out_dir / f"{model}_etf_no_mask.csv"
        failures, done = [], [0]
        t0 = time.time()
        print(
            f"  {model}: {len(fids)} sites x {len(years)} years "
            f"({years[0]}-{years[-1]}) on {threads} threads"
        )

        def _one_site(
            fid, _model=model, _coll=coll_path, _done=done, _failures=failures, _t0=t0
        ):
            values = {}
            for year in years:
                try:
                    values.update(
                        _site_year(
                            ex5, _model, _coll, geometries[fid], year, max_retries
                        )
                    )
                except Exception as e:  # noqa: BLE001 - EE throttling raises many types
                    with lock:
                        _failures.append((fid, year, str(e)[:80]))
            values = ex5._resolve_duplicates(values)
            with lock:
                _done[0] += 1
                if _done[0] % 5 == 0 or _done[0] == len(fids):
                    print(f"    {_done[0]}/{len(fids)} sites  {time.time() - _t0:.0f}s")
            return fid, {
                k: v
                for k, v in values.items()
                if v is not None and 0 < v <= ex5.MAX_VALID_ETF
            }

        results = {}
        with ThreadPoolExecutor(max_workers=threads) as pool:
            for fut in as_completed([pool.submit(_one_site, f) for f in fids]):
                fid, vals = fut.result()
                results[fid] = vals

        import pandas as pd

        df = pd.DataFrame.from_dict(results, orient="index")
        df.index.name = cfg.feature_id_col
        df = df.reindex(fids)
        df = df[sorted(df.columns)]
        df = ex5._merge_checkpoint(df, str(checkpoint))
        df.to_csv(checkpoint)

        elapsed = time.time() - t0
        n_ops = len(fids) * len(years)
        print(
            f"  {model}: {elapsed:.0f}s ({elapsed / n_ops:.2f}s per site-year), "
            f"{int(df.notna().sum().sum()):,} values, {len(failures)} failed site-years"
        )
        if failures:
            for fid, year, err in failures[:5]:
                print(f"    FAILED {fid} {year}: {err}")

        summary = {
            "model": model,
            "n_sites": len(fids),
            "n_dates": int(df.shape[1]),
            "n_values": int(df.notna().sum().sum()),
            "date_range": [df.columns.min(), df.columns.max()] if df.shape[1] else [],
            "start_yr": cfg.start_dt.year,
            "end_yr": cfg.end_dt.year,
            "et_denominated": model in ex5.ET_MODELS,
        }
        (out_dir / f"{model}_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"  saved {checkpoint} ({summary['n_values']:,} values)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--project", required=True, help="Earth Engine project to run under")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--swim-rs", type=Path, default=DEFAULT_SWIM_RS)
    p.add_argument(
        "--steps", default="all", help=f"comma-separated: {', '.join(STEPS)}"
    )
    p.add_argument("--models", default=",".join(ALL_MODELS), help="OpenET v2.1 models")
    p.add_argument("--sites", default=None, help="comma-separated station IDs")
    p.add_argument(
        "--sites-file",
        type=Path,
        default=None,
        help="file of station IDs, one per line; avoids a 200-station command line",
    )
    p.add_argument(
        "--threads", type=int, default=8, help="concurrent EE requests for ETf"
    )
    p.add_argument(
        "--etf-start-year",
        type=int,
        default=2016,
        help="ETf start year; the TOML date_range governs met, NDVI and SNODAS",
    )
    p.add_argument("--max-retries", type=int, default=6, help="per site-year attempts")
    args = p.parse_args(argv)

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

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in models if m not in ALL_MODELS]
    if bad:
        raise SystemExit(f"unknown model(s) {bad}; choose from {list(ALL_MODELS)}")

    ex5 = load_example_5(args.swim_rs, args.project)

    from swimrs.swim.config import ProjectConfig

    cfg = ProjectConfig()
    cfg.read_config(str(args.config))

    import geopandas as gpd

    gdf = gpd.read_file(cfg.fields_shapefile, engine="fiona")
    all_sites = gdf[cfg.feature_id_col].astype(str).to_list()
    if args.sites and args.sites_file:
        raise SystemExit("pass --sites or --sites-file, not both")
    if args.sites_file:
        sites = [s.strip() for s in args.sites_file.read_text().split() if s.strip()]
    elif args.sites:
        sites = [s.strip() for s in args.sites.split(",")]
    else:
        sites = all_sites
    missing = [s for s in sites if s not in all_sites]
    if missing:
        raise SystemExit(f"stations not in {cfg.fields_shapefile}: {missing}")

    print(
        f"{cfg.project_name}: {len(sites)} stations, "
        f"{cfg.start_dt.year}-{cfg.end_dt.year}, project {args.project!r}"
    )

    if "properties" in steps:
        print("properties:")
        ex5.extract_properties(cfg)
    if "snodas" in steps:
        print("snodas:")
        ex5.extract_snodas(cfg, sites)
    if "ndvi" in steps:
        print("ndvi:")
        extract_ndvi_threaded(cfg, ex5, sites, args.threads, args.max_retries)
    if "refet" in steps:
        print("refet:")
        ex5.extract_openet_refet(cfg, sites)
        # extract_openet_refet hardcodes its output to Path(__file__).parent/"data"/
        # "openet_refet" -- i.e. into the swim-rs example directory, ignoring cfg. Same
        # defect as extract_etf_v21, which is why that one is reimplemented here. Rather
        # than fork a second extractor, relocate what it wrote into this project.
        src = Path(ex5.__file__).resolve().parent / "data" / "openet_refet"
        dst = Path(cfg.data_dir) / "openet_refet"
        moved = 0
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            for csv in src.glob("openet_*.csv"):
                # shutil.move, not Path.replace: the swim-rs checkout and this project's
                # data root are on different filesystems, and os.replace cannot cross
                # devices (OSError 18).
                shutil.move(str(csv), str(dst / csv.name))
                moved += 1
        print(f"  relocated {moved} refet CSVs -> {dst}")
    if "etf" in steps:
        print(f"etf: {', '.join(models)}")
        extract_etf(
            cfg,
            ex5,
            models,
            sites,
            Path(cfg.data_dir) / ETF_OUT_DIRNAME,
            start_yr=args.etf_start_year,
            threads=args.threads,
            max_retries=args.max_retries,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
