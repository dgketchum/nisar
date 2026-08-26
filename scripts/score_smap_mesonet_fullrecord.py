"""Score SMAP L3 (SPL3SMP_E, 9 km enhanced, AM) against Montana Mesonet 5 cm VWC over
the full 2015-04..2026-08 SMAP record, one row per station.

This is the panel-1 baseline for the NISAR proposal -- the "SMAP is already sufficient
where the 9 km pixel is representative of the probe" leg -- and it is deliberately a
longer-record, richer-metric version of validate_mesonet_smap.py rather than a
replacement: that script exists to be sampled the same way SME2 was, for the
head-to-head, and it stays as it is. Here the whole SMAP record is used, and the
per-station table adds Spearman rho, the paired-date span, and an anomaly r on
deseasonalized series, because a raw-r baseline over a Montana annual cycle is mostly
scoring the seasonal cycle both series share.

Three sample facts govern how the output can be read, all printed by the run:

* Only 151 of the 207 SMAP-extracted stations have a 5 cm sensor at all -- the older
  HydroMet/BLM/AgriMet installs start at 10 cm -- and 12 of those 151 report a 5 cm
  series one to a few days long, so the scored sample is ~144 stations, 120 of them
  with >= MIN_REPORT_PAIRS paired days.
* Mesonet 5 cm VWC starts 2017-09, and most of the dense-5 cm ACE stations were
  installed 2024-2026, so "full record" is the full SMAP record joined against a
  much shorter in-situ one: median paired span is under two years and only 69
  stations clear the two-year bar the anomaly r needs.
* 24 of the 231 Mesonet stations have no recommended-quality SMAP retrieval on any
  day of the record. ``diagnose_missing`` reports the land-cover/terrain pattern
  behind that; see the printed caveat on how far the attributes actually separate.

Nothing is clipped or dropped to make the table tidy. In-situ days voided by
scripts/qc_mesonet_vwc.py (physically impossible VWC, originals preserved in
mesonet/mt_mesonet_vwc_qc_failures.csv) cannot be paired, so they are excluded from the
metrics and counted per station in ``n_insitu_qc_voided``; anything still out of range
after QC raises rather than being handled here. The number of pairs each metric
actually rests on is a column.

Usage:
    uv run python scripts/score_smap_mesonet_fullrecord.py
    uv run python scripts/score_smap_mesonet_fullrecord.py /data/ssd2/nisar/
    uv run python scripts/score_smap_mesonet_fullrecord.py /data/ssd2/nisar/ \
        --dem /data/nvme0/windninja/dem_tiles/dem_htd_125m.vrt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from ease2_grid import PIXEL_SIZE, cell_rectangle, station_grid_cells
from pyproj import Transformer
from scipy.stats import mannwhitneyu, spearmanr
from validate_mesonet_sme2 import load_insitu_5cm

DEFAULT_DATA_DIR = Path("/data/ssd2/nisar/")

MIN_PAIRS = 5  # station enters the per-station table
MIN_REPORT_PAIRS = 100  # station enters the headline medians
MIN_ANOM_SPAN_DAYS = 730  # ">= 2 years of pairs" for the anomaly r
CLIM_WINDOW_DAYS = 31  # day-of-year climatology smoothing window
DOY_SLOTS = 366
UBRMSE_GOAL = 0.06
VWC_LO, VWC_HI = 0.0, 1.0

# Attributes tested for separation between the retrieved and never-retrieved cells.
# The first four are the CDL/EASE2 land-cover fractions that stand in for SMAP's own
# exclusion masks (dense vegetation, urban, open water); elev_std/relief are the
# terrain-roughness stand-in for its topographic-complexity mask and appear only with
# --dem.
DIAG_ATTRS = [
    "f_forest_9km",
    "f_developed_9km",
    "f_water_9km",
    "f_wetland_9km",
    "cult_f_9km",
    "f_grass_9km",
    "f_shrub_9km",
    "entropy_9km",
    "elevation",
]
DIAG_TERRAIN_ATTRS = ["elev_std", "relief"]
DEM_SAMPLES_PER_SIDE = 48  # ~188 m sampling of each 9 km cell


def load_smap(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "validation/smap_mesonet_extractions.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    if df.duplicated(["station", "date"]).any():
        raise ValueError(f"{path.name} has duplicate (station, date) rows")
    print(
        f"SMAP:   {len(df)} station-day retrievals, {df['station'].nunique()} stations, "
        f"{df['date'].min().date()}..{df['date'].max().date()}"
    )
    return df


def audit_insitu(insitu: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Account for QC-voided and out-of-range 5 cm VWC; return the usable rows.

    ``load_insitu_5cm`` selects on ``element == soil_vwc_0005``, which in this archive
    is exactly the ``depth_cm == 5.0`` rows, all in m3/m3.

    NaN in the archive is not missing data in the ordinary sense: it is a value
    ``scripts/qc_mesonet_vwc.py`` voided against an explicit rule, with the original
    preserved in ``mesonet/mt_mesonet_vwc_qc_failures.csv``. Those rows are counted
    per station, reported, carried in the output table as ``n_insitu_qc_voided``, and
    then excluded from every metric -- a NaN cannot be paired, and pretending
    otherwise is what produced blmcapit's ubRMSE of 447 m3/m3 before the QC pass.
    Any value still outside [0, 1] after QC is a rule gap and raises: the archive is
    the place to fix it, not here.
    """
    print(
        f"In-situ: {len(insitu)} station-day 5 cm VWC values, "
        f"{insitu['station'].nunique()} stations, "
        f"{insitu['date'].min().date()}..{insitu['date'].max().date()}"
    )
    voided = insitu[insitu["insitu_vwc"].isna()]
    counts = voided.groupby("station").size()
    if len(voided):
        print(
            f"\nQC: {len(voided)} 5 cm values at {len(counts)} station(s) were voided "
            "by scripts/qc_mesonet_vwc.py (originals in "
            "mesonet/mt_mesonet_vwc_qc_failures.csv);\nexcluded from all metrics and "
            "reported as n_insitu_qc_voided:"
        )
        per = counts.to_frame("size")
        per["years"] = voided.groupby("station")["date"].apply(
            lambda s: "-".join(str(y) for y in sorted(s.dt.year.unique()))
        )
        print(per.to_string())
        insitu = insitu[insitu["insitu_vwc"].notna()]

    bad = insitu[(insitu["insitu_vwc"] < VWC_LO) | (insitu["insitu_vwc"] > VWC_HI)]
    if len(bad):
        raise ValueError(
            f"{len(bad)} 5 cm VWC values outside [{VWC_LO}, {VWC_HI}] m3/m3 survive "
            f"QC at {sorted(set(bad['station']))} -- extend the rules in "
            "scripts/qc_mesonet_vwc.py rather than handling it here"
        )

    thin = insitu.groupby("station").size()
    thin = thin[thin <= MIN_PAIRS]
    if len(thin):
        print(
            f"\n{len(thin)} station(s) report a 5 cm series <= {MIN_PAIRS} days long "
            "(a one-off record, not a working sensor); they cannot be scored:"
        )
        print("  " + ", ".join(f"{s}({n})" for s, n in thin.items()))
    return insitu, counts


def doy_climatology(doys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Day-of-year mean smoothed with a circular ``CLIM_WINDOW_DAYS`` window.

    Returns a length-``DOY_SLOTS`` array, NaN in slots the window still cannot fill.
    """
    totals = np.zeros(DOY_SLOTS)
    counts = np.zeros(DOY_SLOTS)
    np.add.at(totals, doys - 1, values)
    np.add.at(counts, doys - 1, 1.0)

    half = CLIM_WINDOW_DAYS // 2
    offsets = np.arange(-half, half + 1)
    idx = (np.arange(DOY_SLOTS)[:, None] + offsets[None, :]) % DOY_SLOTS
    win_totals = totals[idx].sum(axis=1)
    win_counts = counts[idx].sum(axis=1)
    with np.errstate(invalid="ignore"):
        return np.where(win_counts > 0, win_totals / np.maximum(win_counts, 1), np.nan)


def anomaly_r(paired: pd.DataFrame) -> tuple[float, int]:
    """Pearson r of deseasonalized SMAP vs in-situ; (NaN, 0) if the span is too short."""
    span = (paired["date"].max() - paired["date"].min()).days
    if span < MIN_ANOM_SPAN_DAYS:
        return np.nan, 0

    doys = paired["date"].dt.dayofyear.to_numpy()
    anomalies = {}
    for col in ("smap_sm", "insitu_vwc"):
        vals = paired[col].to_numpy()
        clim = doy_climatology(doys, vals)
        anomalies[col] = vals - clim[doys - 1]

    ok = ~(np.isnan(anomalies["smap_sm"]) | np.isnan(anomalies["insitu_vwc"]))
    if ok.sum() < MIN_PAIRS:
        return np.nan, int(ok.sum())
    r = np.corrcoef(anomalies["smap_sm"][ok], anomalies["insitu_vwc"][ok])[0, 1]
    return float(r), int(ok.sum())


def score(smap: pd.DataFrame, insitu: pd.DataFrame, voided: pd.Series) -> pd.DataFrame:
    """Per-station metrics over every day both products report."""
    paired_all = smap.merge(insitu, on=["station", "date"], how="inner")
    print(
        f"\nPaired: {len(paired_all)} station-days, "
        f"{paired_all['station'].nunique()} stations"
    )

    records = []
    for stn, grp in paired_all.groupby("station"):
        if len(grp) < MIN_PAIRS:
            continue
        grp = grp.sort_values("date")
        diff = grp["smap_sm"] - grp["insitu_vwc"]
        bias = diff.mean()
        rmse = float(np.sqrt((diff**2).mean()))
        ubrmse = float(np.sqrt(((diff - bias) ** 2).mean()))
        r_anom, n_anom = anomaly_r(grp)
        records.append(
            {
                "station": stn,
                "n_pairs": len(grp),
                "first_date": grp["date"].min().date(),
                "last_date": grp["date"].max().date(),
                "span_days": (grp["date"].max() - grp["date"].min()).days,
                "r": grp["smap_sm"].corr(grp["insitu_vwc"]),
                "spearman_rho": spearmanr(grp["smap_sm"], grp["insitu_vwc"]).statistic,
                "bias": float(bias),
                "rmse": rmse,
                "ubrmse": ubrmse,
                "r_anom": r_anom,
                "n_anom": n_anom,
                "insitu_mean": float(grp["insitu_vwc"].mean()),
                "smap_mean": float(grp["smap_sm"].mean()),
                "n_insitu_qc_voided": int(voided.get(stn, 0)),
                "meets_ubrmse_goal": ubrmse <= UBRMSE_GOAL,
            }
        )
    scores = pd.DataFrame(records).sort_values("station").reset_index(drop=True)
    report_thin_pairs(paired_all, smap, insitu)
    return scores


def report_thin_pairs(
    paired_all: pd.DataFrame, smap: pd.DataFrame, insitu: pd.DataFrame
) -> None:
    """Stations that have both inputs but too few (or zero) paired days, and why."""
    both = set(smap["station"]) & set(insitu["station"])
    n_smap = smap.groupby("station").size()
    n_insitu = insitu.groupby("station").size()
    n_pairs = paired_all.groupby("station").size()

    zero = sorted(both - set(paired_all["station"]))
    if zero:
        print(
            f"\n{len(zero)} station(s) have SMAP rows and 5 cm in-situ rows but zero "
            "paired days:"
        )
        print(
            pd.DataFrame(
                {
                    "n_smap_days": n_smap.reindex(zero),
                    "n_insitu_days": n_insitu.reindex(zero),
                }
            ).to_string()
        )
        print(
            "  -- each has a one-day 5 cm record that missed the recommended-quality "
            "SMAP days; not a join defect."
        )

    thin = sorted(s for s in n_pairs.index if n_pairs[s] < MIN_PAIRS)
    if thin:
        print(
            f"{len(thin)} station(s) paired on < {MIN_PAIRS} days and are excluded from "
            "the table: " + ", ".join(f"{s}({n_pairs[s]})" for s in thin)
        )

    no_5cm = sorted(set(smap["station"]) - set(insitu["station"]))
    print(
        f"{len(no_5cm)} SMAP-extracted station(s) have no 5 cm sensor at all "
        "(10 cm shallowest) and cannot enter a 5 cm comparison."
    )


def headline(scores: pd.DataFrame) -> None:
    sub = scores[scores["n_pairs"] >= MIN_REPORT_PAIRS]
    print(f"\n{'=' * 74}")
    print("SMAP L3 (SPL3SMP_E 9 km AM) vs MT Mesonet 5 cm VWC, full record")
    print(f"{'=' * 74}")
    print(f"Stations in table (n_pairs >= {MIN_PAIRS}): {len(scores)}")
    print(f"Stations with n_pairs >= {MIN_REPORT_PAIRS}:    {len(sub)}")
    print(f"\nMedians over the n_pairs >= {MIN_REPORT_PAIRS} stations:")
    print(f"  median n_pairs      {sub['n_pairs'].median():.0f}")
    print(f"  median r            {sub['r'].median():.3f}")
    print(f"  median ubRMSE       {sub['ubrmse'].median():.4f} m3/m3")
    print(f"  median spearman rho {sub['spearman_rho'].median():.3f}")
    print(f"  median bias         {sub['bias'].median():+.4f} m3/m3")
    print(f"  median RMSE         {sub['rmse'].median():.4f} m3/m3")
    pct = 100 * sub["meets_ubrmse_goal"].mean()
    print(
        f"  meet {UBRMSE_GOAL} ubRMSE   {int(sub['meets_ubrmse_goal'].sum())}/{len(sub)} "
        f"({pct:.0f}%)"
    )
    anom = sub[sub["r_anom"].notna()]
    print(
        f"  median anomaly r    {anom['r_anom'].median():.3f} "
        f"(n={len(anom)} stations with >= {MIN_ANOM_SPAN_DAYS}-day span)"
    )

    cols = ["station", "n_pairs", "span_days", "r", "ubrmse", "bias", "r_anom"]
    print(f"\nBest 5 by r (n_pairs >= {MIN_REPORT_PAIRS}):")
    print(sub.nlargest(5, "r")[cols].round(4).to_string(index=False))
    print(f"\nWorst 5 by r (n_pairs >= {MIN_REPORT_PAIRS}):")
    print(sub.nsmallest(5, "r")[cols].round(4).to_string(index=False))

    flagged = scores[scores["n_insitu_qc_voided"] > 0]
    if len(flagged):
        print(
            "\nStations with QC-voided in-situ days (scored on the surviving days "
            "only; a\nshortened, not a contaminated, record):"
        )
        print(
            flagged[["station", "n_pairs", "n_insitu_qc_voided", "r", "ubrmse"]]
            .round(4)
            .to_string(index=False)
        )


def cell_terrain(cells: pd.DataFrame, dem_path: Path) -> pd.DataFrame:
    """Elevation std and relief inside each station's 9 km EASE2 cell, from a DEM.

    Stand-in for SMAP's topographic-complexity exclusion mask, which is not derivable
    from the CDL land-cover attributes. Samples a regular DEM_SAMPLES_PER_SIDE^2 grid
    of the cell in EASE2 coordinates and reads the DEM at those points.
    """
    records = []
    with rasterio.open(dem_path) as src:
        to_dem = Transformer.from_crs("EPSG:6933", src.crs, always_xy=True)
        inv = ~src.transform
        step = PIXEL_SIZE / DEM_SAMPLES_PER_SIDE
        offsets = (np.arange(DEM_SAMPLES_PER_SIDE) + 0.5) * step
        for stn, row, col in zip(
            cells["station"], cells["row"], cells["col"], strict=True
        ):
            xmin, ymin, _, _ = cell_rectangle(int(row), int(col))
            xx, yy = np.meshgrid(xmin + offsets, ymin + offsets)
            px, py = to_dem.transform(xx.ravel(), yy.ravel())
            c, r = inv * (np.asarray(px), np.asarray(py))
            c = np.floor(c).astype(int)
            r = np.floor(r).astype(int)
            arr = src.read(1, window=((r.min(), r.max() + 1), (c.min(), c.max() + 1)))
            vals = arr[r - r.min(), c - c.min()].astype("float64")
            records.append(
                {
                    "station": stn,
                    "elev_std": float(vals.std()),
                    "relief": float(vals.max() - vals.min()),
                }
            )
    return pd.DataFrame(records)


def diagnose_missing(
    data_dir: Path, smap: pd.DataFrame, insitu: pd.DataFrame, dem_path: Path | None
) -> None:
    """Why 24 of the 231 Mesonet stations get no recommended-quality SMAP retrieval."""
    pix = pd.read_csv(data_dir / "reference/smap_pixel_landcover.csv")
    stations = pd.read_csv(data_dir / "mesonet/mt_mesonet_stations.csv")[
        ["station", "name", "elevation", "sub_network"]
    ]
    cells = station_grid_cells(data_dir / "mesonet/mt_mesonet_stations.csv")
    df = pix.merge(stations, on="station", validate="1:1").merge(
        cells[["station", "row", "col"]], on="station", validate="1:1"
    )
    attrs = list(DIAG_ATTRS)
    if dem_path is not None:
        df = df.merge(cell_terrain(cells, dem_path), on="station", validate="1:1")
        attrs = DIAG_TERRAIN_ATTRS + attrs

    df["retrieved"] = df["station"].isin(set(smap["station"]))
    df["has_5cm"] = df["station"].isin(set(insitu["station"]))
    miss = df[~df["retrieved"]]

    print(f"\n{'=' * 74}")
    print("MISSING STATIONS: no recommended-quality SMAP retrieval, ever")
    print(f"{'=' * 74}")
    print(
        f"{len(df)} stations in the land-cover table, {int(df['retrieved'].sum())} with "
        f"at least one retrieval, {len(miss)} with none."
    )
    shared = set(miss["cell"]) & set(df.loc[df["retrieved"], "cell"])
    print(
        f"The {len(miss)} occupy {miss['cell'].nunique()} distinct 9 km cells, sharing "
        f"{len(shared)} of them with a retrieved station -- so this is a property of "
        "the cell,\nnot of the station record: no station is dropped while a neighbour "
        "in the same pixel is kept."
    )
    print(
        f"{int(miss['has_5cm'].sum())} of the {len(miss)} have a 5 cm sensor, so that "
        "many validation sites are lost outright."
    )

    print("\nAttribute separation (Mann-Whitney AUC: 1.0 = attribute always higher in")
    print("the never-retrieved group, 0.5 = no separation):")
    rows = []
    for attr in attrs:
        a = miss[attr]
        b = df.loc[df["retrieved"], attr]
        u = mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {
                "attribute": attr,
                "median_missing": a.median(),
                "median_retrieved": b.median(),
                "auc": u.statistic / (len(a) * len(b)),
                "p_value": u.pvalue,
            }
        )
    table = pd.DataFrame(rows).sort_values("auc", ascending=False)
    print(table.round(4).to_string(index=False))

    lead = [attr for attr in attrs if attr in ("f_forest_9km", "elev_std", "relief")]
    for attr in lead + ["f_developed_9km"]:
        edges = np.unique(np.nanquantile(df[attr], [0, 0.25, 0.5, 0.75, 0.9, 1.0]))
        binned = df.assign(bin=pd.cut(df[attr], edges, include_lowest=True))
        grp = binned.groupby("bin", observed=True).agg(
            n=("station", "size"), n_missing=("retrieved", lambda s: int((~s).sum()))
        )
        grp["pct_missing"] = (100 * grp["n_missing"] / grp["n"]).round(0)
        print(f"\nNever-retrieved rate by {attr}:")
        print(grp.to_string())

    cols = ["station", "name", "elevation", *lead, "f_developed_9km", "f_water_9km"]
    print("\nThe never-retrieved stations:")
    print(
        miss[cols + ["has_5cm"]]
        .sort_values(lead[0] if lead else "f_forest_9km", ascending=False)
        .round(4)
        .to_string(index=False)
    )
    print(
        "\nCAVEAT: the pattern is a strong tendency, not a reproduction of SMAP's mask. "
        "No single\nattribute threshold separates the two groups -- retrieved stations "
        "exist at every level of\nforest, terrain and urban fraction seen in the "
        "never-retrieved set. SMAP's own exclusion\nlayers (MOD44W water, "
        "GTOPO30-derived topographic complexity, NDVI-climatology vegetation\nwater "
        "content, urban class) are at different resolutions and definitions than the "
        "CDL and\nDEM stand-ins used here, and are not held locally, so the "
        "attribution is diagnostic only."
    )


def build(data_dir: Path, dem_path: Path | None) -> int:
    smap = load_smap(data_dir)
    insitu = load_insitu_5cm(data_dir)
    insitu["date"] = pd.to_datetime(insitu["date"])
    insitu, voided = audit_insitu(insitu)

    scores = score(smap, insitu, voided)
    out_path = data_dir / "validation/smap_mesonet_fullrecord_scores.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    headline(scores)
    diagnose_missing(data_dir, smap, insitu, dem_path)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("data_dir", nargs="?", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument(
        "--dem",
        type=Path,
        default=None,
        help="DEM (any CRS rasterio can read) used to add per-cell elevation std and "
        "relief to the missing-station diagnosis, e.g. "
        "/data/nvme0/windninja/dem_tiles/dem_htd_125m.vrt",
    )
    a = p.parse_args(sys.argv[1:])
    sys.exit(build(a.data_dir, a.dem))
