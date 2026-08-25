"""Consolidate the CONUS NISAR SME2 vs in-situ validation into this repo's data tree.

The validation itself ran in swap-stress (research/sensors/*) and its outputs live
on /nas. Nothing here recomputes the retrieval scores; this reads the finished
per-station tables, joins them to station metadata (coords, IrrMapper/LANID
irrigation label, CDL land-cover fractions), reproduces the cohort labeling and
the cohort/network summary tables, and writes a compact station-level inventory
under /data/ssd2/nisar/validation/conus/ so this repo's evidence base does not
depend on an unreproducible one-off copy.

Two things must travel with any restatement of these numbers:
  * The "no irrigated-land penalty" claim is RETRACTED. The irrigated cohort was
    contaminated by Montana Mesonet rangeland-pad stations sitting inside
    irrigated-valley footprints; cleaned, it is 4-5 stations and establishes
    nothing in either direction.
  * Rain, not irrigation, explains the detected wetting: ~85% of wetting events
    outright and ~96-98% once the gridMET day-boundary is resolved. The change-
    detection result is a wetting-detection result, never an irrigation result.

One station (ARM:Waukomis) is scored upstream against an in-situ record that is
physically impossible and is excluded here; see insitu_qc() for the mechanism and
notes/conus_nisar_validation_import.md for the dated write-up.

Source (read-only): /nas/soils/vwc_timeseries/nisar_val/
Written by:         swap-stress research/sensors/{nisar_sme2_ismn,nisar_val_report,
                    station_irrigation,station_landcover,nisar_delta_detection}.py

Usage:
    uv run python scripts/import_conus_nisar_validation.py
    uv run python scripts/import_conus_nisar_validation.py <src_dir> <out_dir>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DEFAULT = Path("/nas/soils/vwc_timeseries/nisar_val")
OUT_DEFAULT = Path("/data/ssd2/nisar/validation/conus")

CONFIGS = ["results_all", "results_qf", "results_summer_all", "results_summer_qf"]
HEADLINE_CONFIG = "results_summer_qf"  # see swap-stress nisar README section 3
FULL_RECORD_CONFIG = "results_qf"

UBRMSE_GOAL = 0.06  # NISAR mission accuracy goal, m3/m3
MIN_PAIRS = 10  # SMAP-convention scoring threshold used throughout the panel

# In-situ QC bounds on paired 5 cm VWC, m3/m3. The exclusion bound is the only
# one that needs no soil assumption: water cannot occupy more than the bulk
# volume that contains it, so theta > 1.0 (or < 0) is impossible in any medium
# and marks instrument/fill values, not wet soil. The suspect bound is ordinary
# mineral-soil porosity -- above it a reading is unlikely but not impossible
# (organic horizons, swelling clay at saturation), so it is flagged, not dropped.
VWC_IMPOSSIBLE_HI = 1.0
VWC_SUSPECT_HI = 0.6

# CDL cohort thresholds (swap-stress nisar_cohort_report.py, README section 2c)
CROP_T = 0.5  # cult_frac at or above this is cropland
RANGE_T = 0.2  # cult_frac below this is rangeland / non-ag
GRP_T = 0.3  # crop-group fraction at or above this names the sub-cohort

LC_COLS = [
    "cult_frac",
    "cdl_mode",
    "cdl_mode_frac",
    "f_smallgrain",
    "f_fallow",
    "f_rowcrop",
    "f_hay",
    "f_grass",
    "f_shrub",
    "f_forest",
    "f_developed",
    "f_water",
    "f_barren",
    "f_wetland",
    "f_crop_any",
    "f_dryland_grain",
]

METRIC_COLS = ["bias", "rmse", "ubrmse", "r", "insitu_sd", "nisar_sd"]


def cdl_cohort(cult_frac: pd.Series) -> pd.Series:
    """Label each station cropland / mixed / rangeland from the CDL cultivated fraction.

    Left as NaN where no CDL label exists rather than defaulted -- the missing
    labels are a real coverage gap (station_landcover.py ran over
    station_coords_combined.csv, which omits part of the extraction pool), not a
    value to be patched over.
    """
    out = pd.Series(pd.NA, index=cult_frac.index, dtype="object")
    known = cult_frac.notna()
    out[known & (cult_frac >= CROP_T)] = "cropland"
    out[known & (cult_frac < RANGE_T)] = "rangeland"
    out[known & (cult_frac >= RANGE_T) & (cult_frac < CROP_T)] = "mixed"
    return out


def load_station_coords(src: Path) -> pd.DataFrame:
    """Union the two coordinate files; neither covers the whole extraction pool alone."""
    frames = [
        pd.read_csv(src / "station_coords_combined.csv"),
        pd.read_csv(src / "station_coords_v2.csv"),
    ]
    coords = pd.concat(frames, ignore_index=True)
    return coords.drop_duplicates(subset="station", keep="first")


def load_extraction_pool(src: Path) -> pd.DataFrame:
    """Per-station summary of the raw NISAR station-day extractions."""
    ext = pd.read_parquet(src / "nisar_sme2_provisional_conus_extractions.parquet")
    pool = ext.groupby("station").agg(
        n_extract_days=("nisar_sm", "size"),
        extract_first=("date", "min"),
        extract_last=("date", "max"),
        frac_recommended_mean=("frac_recommended", "mean"),
    )
    return pool.reset_index()


def load_config_pairs(src: Path) -> pd.DataFrame:
    """Paired-day counts per station for each of the four screening configs."""
    counts = []
    for cfg in CONFIGS:
        pairs = pd.read_parquet(src / cfg / "nisar_pairs.parquet")
        counts.append(pairs.groupby("station").size().rename(f"n_pairs_{_short(cfg)}"))
    return pd.concat(counts, axis=1).reset_index(names="station")


def _short(cfg: str) -> str:
    return cfg.replace("results_", "")


def load_config_metrics(src: Path, cfg: str) -> pd.DataFrame:
    """Per-station level-accuracy metrics for one config (stations with >= 10 pairs)."""
    m = pd.read_csv(src / cfg / "nisar_station_metrics.csv")
    keep = ["station", "n"] + METRIC_COLS
    m = m[keep].copy()
    suffix = _short(cfg)
    return m.rename(columns={c: f"{c}_{suffix}" for c in keep if c != "station"})


def load_delta_detection(src: Path) -> pd.DataFrame:
    """Per-station change-detection skill from the headline (summer_qf) deltas."""
    d = pd.read_csv(src / HEADLINE_CONFIG / "nisar_delta_detection.csv")
    d = d.rename(columns={"n": "n_delta"}).drop(columns=["irr", "network"])
    d["pod"] = np.where(d["n_wet"] > 0, d["hits"] / d["n_wet"], np.nan)
    d["far"] = np.where(d["n_dry"] > 0, d["fa"] / d["n_dry"], np.nan)
    d["pss"] = d["pod"] - d["far"]
    return d


def insitu_qc(src: Path) -> pd.DataFrame:
    """Range-check the in-situ 5 cm VWC that actually entered the paired scoring.

    Checked against the pairs, not the whole probe archive, because only paired
    days reach a metric. Reported per station as the observed range plus a verdict.

    The one station this excludes is ARM:Waukomis, and the cause is upstream and
    understood rather than merely anomalous. ARM SGP STAMP runs three Stevens
    Hydraprobe profiles per depth at that site; profile 1_3 at 5 cm sat at exactly
    9.88 m3/m3 -- an out-of-range instrument rail, not a reading -- from 2026-05-29
    to 2026-07-17, and ISMN flags every one of those hours C02,C03 (above 0.60,
    above saturation). swap-stress swapstress/sources/ismn.py::_daily_series takes
    the good-flagged daily mean but falls back to the all-data mean on days with no
    good observation, which is exactly a fully-railed day, so the rail is readmitted
    precisely where the flag was doing work. The per-depth sensor merge then folds
    sensors in pairwise rather than as one N-way mean, giving the rail 1/4 weight
    instead of 1/3: ~0.25 * 9.88 = 2.5, which is the ~2.7 m3/m3 in the record.

    The fix belongs in swapstress/sources/ismn.py, not here -- profiles 1_1 and 1_2
    at 5 cm are good-flagged throughout, so honouring the flag recovers a usable
    Waukomis record (~0.10-0.13 m3/m3 through the window) rather than losing the
    station. Until that runs, its scores are excluded here and the reason is carried
    in the CSV.
    """
    pairs = pd.concat(
        [pd.read_parquet(src / cfg / "nisar_pairs.parquet") for cfg in CONFIGS],
        ignore_index=True,
    )
    qc = pairs.groupby("station")["insitu"].agg(
        insitu_paired_min="min", insitu_paired_max="max"
    )
    impossible = (qc["insitu_paired_max"] > VWC_IMPOSSIBLE_HI) | (
        qc["insitu_paired_min"] < 0
    )
    suspect = ~impossible & (qc["insitu_paired_max"] > VWC_SUSPECT_HI)
    qc["insitu_qc"] = "ok"
    qc.loc[suspect, "insitu_qc"] = "suspect_high_vwc_scored"
    qc.loc[impossible, "insitu_qc"] = "excluded_impossible_vwc"
    qc["qc_excluded"] = impossible
    return qc.reset_index()


def drop_excluded_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Blank the scores of QC-excluded stations, keeping the row and the reason.

    The row stays so the exclusion is visible and auditable in the CSV; the metrics
    go so no downstream group-by can silently pick the corrupt values back up. Pair
    counts and every station label are untouched -- the NISAR side is fine, it is
    the in-situ side that failed.
    """
    score_cols = [
        f"{c}_{_short(cfg)}"
        for cfg in (HEADLINE_CONFIG, FULL_RECORD_CONFIG)
        for c in ["n"] + METRIC_COLS
    ]
    score_cols += [
        "n_delta",
        "r_delta",
        "n_wet",
        "hits",
        "n_dry",
        "fa",
        "med_gap",
        "pod",
        "far",
        "pss",
    ]
    df.loc[df["qc_excluded"], [c for c in score_cols if c in df.columns]] = np.nan
    return df


def build_station_table(src: Path) -> pd.DataFrame:
    """One row per station in the NISAR extraction pool, with every label and score."""
    df = load_extraction_pool(src)
    df["network"] = df["station"].str.split(":").str[0]

    df = df.merge(load_station_coords(src), on="station", how="left")

    irr = pd.read_csv(src / "station_irrigation.csv")
    df = df.merge(
        irr[
            ["station", "irr_mode", "irr_frac", "irr_class", "is_east", "source"]
        ].rename(columns={"source": "irr_source"}),
        on="station",
        how="left",
    )

    lc = pd.read_csv(src / "station_landcover.csv")
    df = df.merge(lc[["station"] + LC_COLS], on="station", how="left")
    df["cdl_cohort"] = cdl_cohort(df["cult_frac"])
    # Crop-group flags are land-cover only and are NOT conditioned on irrigation
    # status; the published "dryland cropland" sub-cohorts additionally require
    # irr_class == "dryland", which _cohort_groups() applies.
    df["is_small_grain_fallow"] = (df["cdl_cohort"] == "cropland") & (
        df["f_dryland_grain"] >= GRP_T
    )
    df["is_row_crop"] = (df["cdl_cohort"] == "cropland") & (df["f_rowcrop"] >= GRP_T)

    df = df.merge(load_config_pairs(src), on="station", how="left")
    for cfg in [HEADLINE_CONFIG, FULL_RECORD_CONFIG]:
        df = df.merge(load_config_metrics(src, cfg), on="station", how="left")
    df = df.merge(load_delta_detection(src), on="station", how="left")

    df = df.merge(insitu_qc(src), on="station", how="left")
    # Stations that were extracted but never paired have no in-situ to check.
    df["insitu_qc"] = df["insitu_qc"].fillna("no_pairs")
    df["qc_excluded"] = df["qc_excluded"].fillna(False).astype(bool)
    df = drop_excluded_scores(df)

    return df.sort_values("station").reset_index(drop=True)


def _cohort_groups(df: pd.DataFrame) -> list:
    """The cohort partition used in the swap-stress report, reproduced here.

    Sub-cohorts are subsets of dryland cropland and can overlap; that is how the
    published tables are built, so it is kept.
    """
    d = df[df["cult_frac"].notna()]
    dry = d[d["irr_class"] == "dryland"]
    crop = dry[dry["cult_frac"] >= CROP_T]
    return [
        ("cohort", "all scored", d),
        ("cohort", "dryland, all", dry),
        ("cohort", "dryland cropland", crop),
        ("cohort", "  small grain / fallow", crop[crop["f_dryland_grain"] >= GRP_T]),
        ("cohort", "  row crop", crop[crop["f_rowcrop"] >= GRP_T]),
        ("cohort", "rangeland / non-ag", dry[dry["cult_frac"] < RANGE_T]),
        ("cohort", "irrigated (mask)", d[d["irr_class"] == "irrigated"]),
    ]


def _irr_groups(df: pd.DataFrame) -> list:
    return [("irr_class", cls, g) for cls, g in df.groupby("irr_class", dropna=True)]


def _network_groups(df: pd.DataFrame) -> list:
    return [("network", net, g) for net, g in df.groupby("network")]


def level_summary(stations: pd.DataFrame, cfg: str) -> pd.DataFrame:
    """Median ubRMSE / r / bias and goal attainment per group, one screening config."""
    suffix = _short(cfg)
    scored = stations[stations[f"n_{suffix}"].notna()].copy()
    scored = scored.rename(columns={f"{c}_{suffix}": c for c in ["n"] + METRIC_COLS})
    rows = []
    groups = [("all", "ALL", scored)]
    groups += _cohort_groups(scored) + _irr_groups(scored) + _network_groups(scored)
    for group_type, label, g in groups:
        if not len(g):
            continue
        rows.append(
            {
                "table": "level",
                "config": suffix,
                "group_type": group_type,
                "group": label,
                "n_stations": len(g),
                "n_pairs": int(g["n"].sum()),
                "ubrmse": round(g["ubrmse"].median(), 4),
                "pct_meet_goal": round(100 * (g["ubrmse"] < UBRMSE_GOAL).mean(), 1),
                "r": round(g["r"].median(), 3),
                "bias": round(g["bias"].median(), 4),
            }
        )
    return pd.DataFrame(rows)


def delta_summary(stations: pd.DataFrame) -> pd.DataFrame:
    """Pooled POD / FAR / PSS and median delta-correlation per group."""
    d = stations[stations["n_delta"].notna()].copy()
    rows = []
    groups = [("all", "ALL", d)]
    groups += _cohort_groups(d) + _irr_groups(d) + _network_groups(d)
    for group_type, label, g in groups:
        n_wet, n_dry = g["n_wet"].sum(), g["n_dry"].sum()
        if not len(g) or n_dry == 0:
            continue
        # POD is undefined where a group saw no in-situ wetting events at all
        # (CW3E in the summer window); left NaN rather than scored as zero.
        pod = g["hits"].sum() / n_wet if n_wet else np.nan
        far = g["fa"].sum() / n_dry
        rows.append(
            {
                "table": "delta",
                "config": _short(HEADLINE_CONFIG),
                "group_type": group_type,
                "group": label,
                "n_stations": len(g),
                "n_deltas": int(g["n_delta"].sum()),
                "n_wet_events": int(n_wet),
                "n_dry_intervals": int(n_dry),
                "med_gap_days": g["med_gap"].median(),
                "r_delta": round(g["r_delta"].median(), 3),
                "POD": round(pod, 3),
                "FAR": round(far, 3),
                "PSS": round(pod - far, 3),
            }
        )
    return pd.DataFrame(rows)


def headline_summary(src: Path, stations: pd.DataFrame) -> pd.DataFrame:
    """The 2x2 of screening choices: quality flag on/off x summer-only/full-record."""
    dropped = set(stations.loc[stations["qc_excluded"], "station"])
    rows = []
    for cfg in CONFIGS:
        pairs = pd.read_parquet(src / cfg / "nisar_pairs.parquet")
        metrics = pd.read_csv(src / cfg / "nisar_station_metrics.csv")
        # Same QC exclusion as the station table, or the pooled statistics keep
        # the impossible in-situ values the per-station table just dropped.
        pairs = pairs[~pairs["station"].isin(dropped)]
        metrics = metrics[~metrics["station"].isin(dropped)]
        diff = pairs["nisar"] - pairs["insitu"]
        bias = diff.mean()
        rows.append(
            {
                "config": _short(cfg),
                "is_headline": cfg == HEADLINE_CONFIG,
                "pairs": len(pairs),
                "stations_with_pairs": pairs["station"].nunique(),
                "stations_scored": len(metrics),
                "min_pairs": MIN_PAIRS,
                "date_min": pairs["date"].min(),
                "date_max": pairs["date"].max(),
                "pooled_bias": round(bias, 4),
                "pooled_ubrmse": round(float(np.sqrt(((diff - bias) ** 2).mean())), 4),
                "pooled_r": round(pairs["nisar"].corr(pairs["insitu"]), 3),
                "median_station_ubrmse": round(metrics["ubrmse"].median(), 4),
                "median_station_r": round(metrics["r"].median(), 3),
                "dyn_range_ratio": round(
                    (metrics["nisar_sd"] / metrics["insitu_sd"]).median(), 2
                ),
                "n_meet_goal": int((metrics["ubrmse"] < UBRMSE_GOAL).sum()),
                "pct_meet_goal": round(
                    100 * (metrics["ubrmse"] < UBRMSE_GOAL).mean(), 1
                ),
            }
        )
    hl = pd.DataFrame(rows)
    hl["extraction_pool_stations"] = len(stations)
    return hl


def report(stations: pd.DataFrame) -> None:
    suffix = _short(HEADLINE_CONFIG)
    print(f"\nExtraction pool: {len(stations)} stations")
    print(stations["network"].value_counts().to_string())
    n_no_cdl = stations["cult_frac"].isna().sum()
    print(
        f"\nStations with no CDL land-cover label: {n_no_cdl} "
        f"(absent from station_coords_combined.csv, which station_landcover.py read)"
    )
    print("\nCDL cohort (extraction pool):")
    print(stations["cdl_cohort"].value_counts(dropna=False).to_string())
    scored = stations[stations[f"n_{suffix}"].notna()]
    print(f"\n{HEADLINE_CONFIG}: {len(scored)} stations scored at >= {MIN_PAIRS} pairs")
    print(
        f"  median ubRMSE {scored[f'ubrmse_{suffix}'].median():.4f}  "
        f"median r {scored[f'r_{suffix}'].median():.3f}  "
        f"meet goal {(scored[f'ubrmse_{suffix}'] < UBRMSE_GOAL).sum()}/{len(scored)}"
    )
    dropped = stations[stations["qc_excluded"]]
    print(f"\nIn-situ QC: {len(dropped)} station(s) excluded, scores blanked")
    if len(dropped):
        print(
            dropped[
                ["station", "network", "insitu_paired_min", "insitu_paired_max"]
            ].to_string(index=False)
        )
    flagged = stations[stations["insitu_qc"] == "suspect_high_vwc_scored"]
    if len(flagged):
        print(
            f"\n  Flagged but kept ({len(flagged)}): paired in-situ above "
            f"{VWC_SUSPECT_HI} m3/m3, still physically possible"
        )
        print(
            flagged[["station", "insitu_paired_max", f"ubrmse_{suffix}"]].to_string(
                index=False
            )
        )
    bad = scored[scored[f"ubrmse_{suffix}"] > 0.5]
    if len(bad):
        print("\n  Scored stations with ubRMSE > 0.5 remaining:")
        print(
            bad[
                ["station", f"n_{suffix}", f"bias_{suffix}", f"ubrmse_{suffix}"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC_DEFAULT
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_DEFAULT
    if not src.exists():
        raise SystemExit(f"Source tree not reachable: {src}")
    out.mkdir(parents=True, exist_ok=True)

    stations = build_station_table(src)
    summary = pd.concat(
        [
            level_summary(stations, HEADLINE_CONFIG),
            level_summary(stations, FULL_RECORD_CONFIG),
            delta_summary(stations),
        ],
        ignore_index=True,
    )
    headline = headline_summary(src, stations)

    stations.to_csv(out / "conus_nisar_station_scores.csv", index=False)
    summary.to_csv(out / "conus_nisar_cohort_summary.csv", index=False)
    headline.to_csv(out / "conus_nisar_headline.csv", index=False)

    # The raw station-day extractions are only ~330 KB and cost an earthaccess
    # streaming pass to rebuild, so they come along; the per-station in-situ
    # parquet tree stays on /nas.
    ext = pd.read_parquet(src / "nisar_sme2_provisional_conus_extractions.parquet")
    ext.to_parquet(out / "conus_nisar_sme2_extractions.parquet", index=False)

    for name in (
        "conus_nisar_station_scores.csv",
        "conus_nisar_cohort_summary.csv",
        "conus_nisar_headline.csv",
        "conus_nisar_sme2_extractions.parquet",
    ):
        print(f"Wrote {out / name}")

    report(stations)
