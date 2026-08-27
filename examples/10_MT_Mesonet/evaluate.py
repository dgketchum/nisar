"""Soil-moisture evaluation for the mt_mesonet baseline: modeled theta vs in-situ VWC.

The Example 8 evaluation, repointed at this project: forward-run the calibrated model
(ensemble-median posterior parameters via Example 5's ``parse_pest_params``), convert the
kernel's plant-available water to a volumetric content over the max rooting depth
(``theta_avail``, Example 8's like-for-like quantity), and score it against the QC'd
Montana Mesonet VWC record — which stayed out of calibration entirely (validation-only).

Pairings follow Example 8: the root-zone bucket against a depth-weighted profile mean of
the ``soil_vwc_*`` sensors; the surface evap layer (``-depl_ze``) against the 5 cm sensor
(the SMAP/NISAR analog); the unweighted profile mean and the 50 cm sensor kept for
reference. Metrics are scale-invariant (Pearson, Spearman, anomaly-r after removing the
DOY climatology), growing season Apr-Oct, matching Example 8 so this cohort's numbers sit
next to the SCAN ones — this is the ET-only-baseline side of the "soil column
underdetermined" comparison.

Observed VWC: ``/data/ssd2/nisar/mesonet/mt_mesonet_daily_long.parquet`` (m3/m3, MCO
bounds failures already voided by scripts/qc_mesonet_vwc.py; this script raises on any
surviving out-of-range value rather than clipping).

Must run with the swim-rs venv:
    /home/dgketchum/code/swim-rs/.venv/bin/python examples/10_MT_Mesonet/evaluate.py
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SWIMRS_EXAMPLES = Path("/home/dgketchum/code/swim-rs/examples")
E5 = SWIMRS_EXAMPLES / "5_Flux_Ensemble"
E8 = SWIMRS_EXAMPLES / "8_Soil_Moisture"

PROJECT = Path("/data/ssd2/nisar/swim/mt_mesonet")
CONFIG = str(PROJECT / "mt_mesonet.toml")
CONTAINER = str(PROJECT / "mt_mesonet.swim")
PAR_CSV = str(PROJECT / "results" / "run0_baseline" / "mt_mesonet.3.par.csv")
FIELDS_FGB = str(PROJECT / "gis" / "build" / "mesonet_fields_100m_n9.fgb")
OBS_PARQUET = "/data/ssd2/nisar/mesonet/mt_mesonet_daily_long.parquet"

GROW_MONTHS = range(4, 11)  # Apr-Oct, Example 8 convention
MAX_SENSOR_DEPTH_CM = 100.0  # deepest Mesonet VWC sensor
MIN_PAIRED_DAYS = 30
MIN_ANOM_SPAN_DAYS = (
    730  # anomaly-r needs >= 2 years or the DOY climatology is the data
)


def build_pairs(obs_cols) -> list[tuple[str, str, str]]:
    """Site-specific pairings: the surface proxy targets the shallowest sensor present.

    The ACE stations carry a 5 cm sensor; the BLM sub-network's shallowest is 10 cm.
    The label stays fixed so cross-site medians group; ``obs_var`` records the actual
    sensor. ``std_ratio`` is not interpretable for the surface pairing (the proxy is
    ``-depl_ze`` in mm, not a VWC) — the scale-invariant metrics are the ones to read.
    """
    import re

    sensors = sorted(
        (c for c in obs_cols if re.fullmatch(r"soil_vwc_\d+", c)),
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    pairs = [("theta_avail", "rootzone_theta", "rootzone depth-wtd")]
    if sensors:
        pairs.append(
            ("surface_sm_proxy", sensors[0], "surface shallowest (SMAP/NISAR analog)")
        )
    pairs += [
        ("theta_avail", "profile_mean_theta", "unwtd profile mean"),
        ("theta_avail", "soil_vwc_50", "deep sensor 50cm"),
    ]
    return pairs


def _example8():
    """Import Example 8's evaluate module (deferred: needs swimrs).

    Example 5's ``evaluate`` must be importable as the module named ``evaluate``
    first — Example 8's module does ``from evaluate import parse_pest_params``
    against it.
    """
    if str(E5) not in sys.path:
        sys.path.insert(0, str(E5))
    import evaluate  # noqa: F401  (E5's, claims the module name E8 expects)

    spec = importlib.util.spec_from_file_location("e8_evaluate", E8 / "evaluate.py")
    e8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e8)
    return e8


def _load_config(conf: str = CONFIG):
    from swimrs.swim.config import ProjectConfig

    cfg = ProjectConfig()
    cfg.read_config(conf)
    return cfg


def station_wide(obs_long: pd.DataFrame, station: str) -> pd.DataFrame:
    """Pivot the long QC'd VWC record to one wide frame for ``station``.

    Columns ``soil_vwc_{depth_cm}`` (m3/m3) plus ``profile_mean_theta`` (unweighted
    mean over whatever sensors report that day). Raises on out-of-range survivors —
    those mean the QC step was skipped or the archive was re-pulled without it.
    """
    df = obs_long[
        (obs_long["station"] == station)
        & obs_long["element"].str.startswith("soil_vwc")
        & obs_long["depth_cm"].notna()
    ]
    if df.empty:
        return pd.DataFrame()
    bad = df["value"].dropna()
    bad = bad[(bad < 0.0) | (bad > 1.0)]
    if len(bad):
        raise SystemExit(
            f"{station}: {len(bad)} VWC values outside [0, 1] — re-run scripts/qc_mesonet_vwc.py"
        )
    wide = df.pivot_table(
        index="date", columns="depth_cm", values="value", aggfunc="mean"
    )
    wide.columns = [f"soil_vwc_{int(d)}" for d in wide.columns]
    wide = wide.sort_index()
    wide["profile_mean_theta"] = wide.mean(axis=1)
    return wide


def _deseasonalize(s: pd.Series) -> pd.Series:
    return s - s.groupby(s.index.dayofyear).transform("mean")


def score_pairs(fid: str, mdf: pd.DataFrame, obs: pd.DataFrame) -> list[dict]:
    """Example 8's pairing metrics for one site: growing-season, scale-invariant.

    Anomaly-r is gated on a >= 2-year paired span (matching the full-record SMAP
    scorer): on a shorter record the DOY climatology *is* the data and the
    deseasonalized correlation is degenerate.
    """
    rows = []
    for mcol, ycol, label in build_pairs(obs.columns):
        if mcol not in mdf.columns or ycol not in obs.columns:
            continue
        df = mdf[[mcol]].join(obs[[ycol]], how="inner")
        d = df[df.index.month.isin(GROW_MONTHS)].dropna()
        if len(d) < MIN_PAIRED_DAYS:
            continue
        span_days = (d.index.max() - d.index.min()).days
        anom_r = np.nan
        if span_days >= MIN_ANOM_SPAN_DAYS:
            an_o = _deseasonalize(d[ycol]).dropna()
            an_m = _deseasonalize(d[mcol]).reindex(an_o.index).dropna()
            an_o = an_o.reindex(an_m.index)
            if len(an_m) > MIN_PAIRED_DAYS and an_o.std() > 0 and an_m.std() > 0:
                anom_r = round(pearsonr(an_o, an_m)[0], 3)
        rows.append(
            {
                "site_id": fid,
                "pairing": label,
                "model_var": mcol,
                "obs_var": ycol,
                "n": len(d),
                "first": d.index.min().date(),
                "last": d.index.max().date(),
                "pearson": round(pearsonr(d[ycol], d[mcol])[0], 3),
                "spearman": round(spearmanr(d[ycol], d[mcol])[0], 3),
                "anom_r": anom_r,
                "std_ratio": round(d[mcol].std() / d[ycol].std(), 3)
                if d[ycol].std() > 0
                else np.nan,
            }
        )
    return rows


def plot_site(fid, mdf, obs, fig_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _nrm(x):
        return (x - x.min()) / (x.max() - x.min())

    surface_obs = {m: y for m, y, _ in build_pairs(obs.columns)}.get("surface_sm_proxy")
    panels = [("theta_avail", "rootzone_theta", "root zone (depth-wtd obs)")]
    if surface_obs:
        panels.append(
            ("surface_sm_proxy", surface_obs, f"surface layer vs {surface_obs}")
        )
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(13, 3.5 * len(panels)), sharex=True, squeeze=False
    )
    drew = False
    for ax, (mcol, ycol, ttl) in zip(axes.ravel(), panels):
        if ycol not in obs.columns:
            continue
        d = mdf[[mcol]].join(obs[[ycol]], how="inner").dropna()
        d = d[d.index.month.isin(GROW_MONTHS)]
        if len(d) < MIN_PAIRED_DAYS:
            continue
        ax.plot(d.index, _nrm(d[ycol]), lw=0.6, label=f"observed {ycol} (norm)")
        ax.plot(d.index, _nrm(d[mcol]), lw=0.6, alpha=0.8, label=f"SWIM {mcol} (norm)")
        r = pearsonr(d[ycol], d[mcol])[0]
        ax.set_title(f"{fid} {ttl}: n={len(d)}  r={r:.2f}", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        drew = True
    if drew:
        fig.tight_layout()
        fig.savefig(fig_dir / f"{fid}_theta.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate mt_mesonet baseline vs in-situ VWC"
    )
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--container", default=CONTAINER)
    ap.add_argument("--par-csv", default=PAR_CSV)
    ap.add_argument("--fields", default=FIELDS_FGB)
    ap.add_argument("--obs", default=OBS_PARQUET)
    ap.add_argument("--out-dir", default=str(Path(PAR_CSV).parent / "evaluation"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    import geopandas as gpd

    e8 = _example8()
    cfg = _load_config(args.config)
    fids = gpd.read_file(args.fields, engine="fiona")["site_id"].tolist()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "theta_figures"
    if not args.no_figures:
        fig_dir.mkdir(exist_ok=True)

    container = e8.SwimContainer.open(args.container, mode="r")
    print(f"Forward-running {len(fids)} sites with median posterior of {args.par_csv}")
    model = e8.run_model(cfg, container, args.par_csv, fids)

    obs_long = pd.read_parquet(args.obs)
    rows = []
    for fid in fids:
        obs = station_wide(obs_long, fid)
        if obs.empty:
            print(f"  {fid}: no observed VWC, skipping")
            continue
        obs["rootzone_theta"] = e8.depth_weighted_rootzone(
            obs, max_depth_cm=MAX_SENSOR_DEPTH_CM
        )
        rows.extend(score_pairs(fid, model[fid], obs))
        if not args.no_figures:
            plot_site(fid, model[fid], obs, fig_dir)

    res = pd.DataFrame(rows)
    out_csv = out_dir / "mesonet_theta_correlations.csv"
    res.to_csv(out_csv, index=False)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(f"\n=== modeled vs observed Mesonet theta ({len(res)} site-pairing rows) ===")
    if not res.empty:
        print(res.to_string(index=False))
        print("\n--- median across sites, by pairing ---")
        for label in res["pairing"].unique():
            sub = res[res.pairing == label]
            med = sub[["pearson", "spearman", "anom_r", "std_ratio"]].median().round(3)
            print(
                f"  [{label:<32}] n_sites={len(sub)}  pearson={med.pearson}  "
                f"spearman={med.spearman}  anom_r={med.anom_r}  std_ratio={med.std_ratio}"
            )
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
