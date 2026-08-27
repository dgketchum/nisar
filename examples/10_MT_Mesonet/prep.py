"""Container prep for mt_mesonet (NISAR L-band soil-moisture line), reusing swim-rs Example 5.

Example-8-style wrapper: repoints Example 5's ``container_prep``/``build_container``
pipeline at an external project config (``/data/ssd2/nisar/swim/mt_mesonet/mt_mesonet.toml``)
and replaces ``build_shapefile`` with a guard — the 100 m Mesonet station buffers are an
input, not an artifact. Everything else (GridMET mapping, container creation, met/NDVI/snow
ingest, merged NDVI, OpenET v2.1 ETf, OpenET reference ET, mean ensemble, dynamics) is
Example 5's code unchanged.

This lives in the nisar repo (mirroring swim-rs's ``examples/`` numbering) rather than in
swim-rs itself: the NISAR line is exploratory and stays out of that repo. Nothing in
swim-rs is modified — Example 5's modules are imported from ``SWIMRS_EXAMPLES`` and two of
their symbols reassigned at runtime.

Deviations from Example 5, all forced by this project's delivered layout:
  * inputs follow the ``build_container.py`` getInfo orientation (wide site x YYYYMMDD),
    so ETf/refET come from ``{data}/etf_v21_openet_eto`` and ``{data}/openet_refet``;
  * the ETf observation window is cut at 2015-01-01 while the container spans
    2010-01-01 -> 2026-08-24 so 2010-2014 is spinup;
  * the delivered property CSVs cover all 231 Mesonet stations, so they are subset to the
    container's sites (and ``LAT``/``LON`` are dropped from the irrigation table, which
    would otherwise enter ``_ingest_irrigation``'s mean-over-numeric-columns) into
    ``{properties}/prepared/``;
  * SNODAS arrived as one wide ``swe.csv`` under ``snow/snodas/getinfo`` rather than
    monthly ``extracts/swe_{YYYY}_{MM}.csv``, which the getInfo ingest path reads as-is.

``blmcapit`` is excluded by default: its in-situ VWC record is corrupted by calibration
drift. Exclusion is build-time only — no input data is moved or modified. The container is
built from a derived 9-site geometry written to ``{gis}/build/``.

Must run with the swim-rs venv (needs swimrs and the Example 5 modules):
    /home/dgketchum/code/swim-rs/.venv/bin/python examples/10_MT_Mesonet/prep.py --overwrite
"""

import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

SWIMRS_EXAMPLES = Path("/home/dgketchum/code/swim-rs/examples")
E5 = SWIMRS_EXAMPLES / "5_Flux_Ensemble"

CONFIG = "/data/ssd2/nisar/swim/mt_mesonet/mt_mesonet.toml"
EXCLUDE_SITES = ["blmcapit"]  # in-situ VWC corrupted by calibration drift
ETF_START = "2015-01-01"


def _example5():
    """Import Example 5's pipeline modules and wire this project's overrides in.

    Deferred (not module-level) so the pure file-in/file-out helpers here stay
    importable — and testable — without swimrs on the path.
    """
    if str(E5) not in sys.path:
        sys.path.insert(0, str(E5))
    import build_container as b5
    import container_prep as e2

    # Repoint config + neutralize the flux-cohort shapefile builder.
    e2._load_config = _load_config
    e2.build_shapefile = _guard_shapefile
    return e2, b5


def _load_config(conf: str = CONFIG, calibrate: bool = False):
    from swimrs.swim.config import ProjectConfig

    cfg = ProjectConfig()
    cfg.read_config(conf, calibrate=calibrate)
    return cfg


def _guard_shapefile(cfg, overwrite=False, exclude_sites=None):
    """The station buffers are prebuilt; never regenerate from flux-tower footprints."""
    if not os.path.exists(cfg.fields_shapefile):
        raise SystemExit(f"Fields geometry not found: {cfg.fields_shapefile}")
    print(f"Using prebuilt fields geometry: {cfg.fields_shapefile}")


def select_sites(cfg, exclude: list[str]) -> list[str]:
    """Point ``cfg.fields_shapefile`` at a derived geometry with ``exclude`` dropped.

    The source geometry is left untouched; the subset is written to ``{gis}/build/``.
    Returns the retained site IDs in geometry order.
    """
    gdf = gpd.read_file(cfg.fields_shapefile, engine="fiona")
    uid = cfg.feature_id_col
    missing = sorted(set(exclude) - set(gdf[uid]))
    if missing:
        raise SystemExit(
            f"Sites to exclude not present in {cfg.fields_shapefile}: {missing}"
        )

    keep = gdf[~gdf[uid].isin(exclude)].reset_index(drop=True)
    build_dir = Path(cfg.gis_dir) / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    out = build_dir / f"{Path(cfg.fields_shapefile).stem}_n{len(keep)}.fgb"
    keep.to_file(out, driver="FlatGeobuf", engine="fiona")

    cfg.fields_shapefile = str(out)
    print(f"Excluded {exclude}; building from {len(keep)} sites -> {out}")
    return keep[uid].tolist()


def prepare_properties(cfg, sites: list[str]) -> dict[str, str]:
    """Subset the 231-station property CSVs to ``sites`` and drop empty LAT/LON.

    ``_ingest_irrigation``/``_ingest_lulc`` average every numeric column, so LAT/LON
    must not reach them. Writes to ``{properties}/prepared/``; sources are untouched.
    """
    src_dir = Path(cfg.properties_dir) / "getinfo"
    out_dir = Path(cfg.properties_dir) / "prepared"
    out_dir.mkdir(parents=True, exist_ok=True)
    uid = cfg.feature_id_col

    prepared = {}
    for key, csv in (
        ("lulc", cfg.lulc_csv),
        ("ssurgo", cfg.ssurgo_csv),
        ("irr", cfg.irr_csv),
    ):
        src = src_dir / Path(csv).name
        df = pd.read_csv(src).set_index(uid)
        absent = [s for s in sites if s not in df.index]
        if absent:
            raise SystemExit(f"{src.name} is missing container sites: {absent}")
        df = df.loc[sites].drop(columns=["LAT", "LON"], errors="ignore")
        if df.isna().any().any():
            raise SystemExit(
                f"{src.name} has null values for the container sites; investigate"
            )
        dst = out_dir / src.name
        df.to_csv(dst)
        prepared[key] = str(dst)
        print(f"  {src.name}: {len(df)} sites, {len(df.columns)} cols -> {dst}")

    return prepared


def build(cfg, sites: list[str], overwrite: bool) -> str:
    """Create the container and run every ingest/compute stage in dependency order."""
    e2, b5 = _example5()
    e2.build_gridmet_mapping(cfg, overwrite=False)
    container = e2.create_project_container(cfg, overwrite=overwrite)
    e2.ingest_meteorology(container, cfg, overwrite=overwrite)
    e2.ingest_remote_sensing(
        container,
        cfg,
        sites=sites,
        overwrite=overwrite,
        add_sentinel=True,
        getinfo=True,
    )
    e2.ingest_snow(container, cfg, overwrite=overwrite, getinfo=True)

    print("\n=== Ingesting Properties (subset, LAT/LON dropped) ===")
    prepared = prepare_properties(cfg, sites)
    container.ingest.properties(
        lulc_csv=prepared["lulc"],
        soils_csv=prepared["ssurgo"],
        irr_csv=prepared["irr"],
        uid_column=cfg.feature_id_col,
        overwrite=overwrite,
    )

    e2.compute_fused_ndvi(container, overwrite=overwrite)
    path = container.path
    container.close()

    # ETf/refET use the getInfo (wide site x YYYYMMDD) readers from Example 5's
    # build_container, which open the container by path. Dynamics needs the ETf
    # ensemble, so these must run before it.
    b5.MODELS = tuple(cfg.etf_ensemble_members)
    b5.ETF_START, b5.ETF_END = ETF_START, cfg.end_dt.strftime("%Y-%m-%d")

    print(
        f"\n=== Ingesting ETf ({len(b5.MODELS)} models, {b5.ETF_START}..{b5.ETF_END}) ==="
    )
    b5.ingest_new_etf(path, os.path.join(cfg.data_dir, "etf_v21_openet_eto"))

    print("\n=== Ingesting OpenET reference ETo/ETr -> {var}_corr ===")
    b5.ingest_openet_eto(path, os.path.join(cfg.data_dir, "openet_refet"))

    print("\n=== Computing mean ETf ensemble ===")
    b5.compute_mean_ensemble(path)

    print("\n=== Computing Dynamics (Example 5 settings) ===")
    b5.compute_dynamics(path, cfg)

    print("\n=== Validation Summary ===")
    b5.validate(path)
    return path


def verify(cfg, path: str) -> bool:
    """Run PestBuilder spinup (NaN end-state guard) and the container health check."""
    from swimrs.calibrate import PestBuilder
    from swimrs.container import open_container
    from swimrs.container.health import health_report_output_dir

    print("\n=== Spinup ===")
    cal_cfg = _load_config(cfg.config_path, calibrate=True)
    container = open_container(path, mode="r")
    builder = PestBuilder(cal_cfg, container, use_existing=False)
    builder.spinup(overwrite=True)
    print(f"Spinup state written: {cal_cfg.spinup}")
    container.close()

    print("\n=== Health Check ===")
    container = open_container(path, mode="r")
    report = container.report(
        # mask_mode "no_mask" rather than the config's "none": the two are
        # equivalent for the ETf rules, but only "no_mask" activates the NDVI
        # field-coverage rule, which is a gate worth having.
        config={
            "mask_mode": "no_mask",
            "etf_target_model": cfg.etf_target_model,
            "etf_ensemble_members": cfg.etf_ensemble_members,
            "met_source": cfg.met_source,
            "snow_source": cfg.snow_source,
        },
        output_dir=str(health_report_output_dir(path)),
        health_profile="calibration",
    )
    container.close()
    return report.passed


def main() -> None:
    p = argparse.ArgumentParser(description="Container prep for mt_mesonet")
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--exclude-sites",
        default=",".join(EXCLUDE_SITES),
        help="Comma-separated site IDs to leave out of the container (build-time only)",
    )
    p.add_argument(
        "--skip-verify", action="store_true", help="Skip spinup + health check"
    )
    args = p.parse_args()

    exclude = [s.strip() for s in args.exclude_sites.split(",") if s.strip()]
    cfg = _load_config(args.config)
    _guard_shapefile(cfg)
    sites = select_sites(cfg, exclude)

    path = build(cfg, sites, args.overwrite)
    ok = True if args.skip_verify else verify(cfg, path)

    print(f"\nContainer ready: {path}")
    if not ok:
        raise SystemExit("Health check reported failures")


if __name__ == "__main__":
    main()
