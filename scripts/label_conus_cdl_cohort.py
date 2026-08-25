"""Label every station in the CONUS NISAR extraction pool with a CDL land-cover cohort.

Why this exists: the land-cover labels that shipped with the consolidated panel came
from swap-stress ``research/sensors/station_landcover.py``, which read
``station_coords_combined.csv``. That file omits 169 of the 969 stations actually in
the extraction pool, so those stations carry no cohort label at all. The gap is not
neutral for the rainfed-agriculture thread: a genuine dryland cropland site that was
never labeled is indistinguishable, in the summary tables, from one that was labeled
and found to be rangeland.

This reproduces the swap-stress method exactly -- same two CDL layers, same 100 m
buffer, same 30 m reduction scale, same class groupings -- but runs it over the full
pool taken from the panel itself, so the whole panel is labeled by one method rather
than by two.

Method (verified against station_landcover.py, not just the note):
  * ``cultivated`` from CDL 2023, the last year the band is published. It is a
    multi-year derived band, so a fallow season does not flip a field to
    non-cultivated. Read as ``eq(2)`` (1 = non-cultivated, 2 = cultivated), mean over
    the buffer -> ``cult_frac``.
  * ``cropland`` from CDL 2025, the latest published year, as a full frequency
    histogram so any regrouping is possible downstream without re-running EE.
  * 100 m radius buffer at 30 m scale, matching station_irrigation.py so the
    irrigation and land-cover labels are geometrically comparable, and matching the
    ~200 m NISAR SME2 posting.

Cohort thresholds come from ``import_conus_nisar_validation.py`` (CROP_T / RANGE_T /
GRP_T), which is also where the cohort and delta summary tables are built; this script
imports them rather than restating them.

Usage:
    uv run python scripts/label_conus_cdl_cohort.py
    uv run python scripts/label_conus_cdl_cohort.py --limit 12   # pilot
"""

import argparse
import json
import sys
from pathlib import Path

import ee
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from import_conus_nisar_validation import (
    CROP_T,
    FULL_RECORD_CONFIG,
    GRP_T,
    HEADLINE_CONFIG,
    LC_COLS,
    RANGE_T,
    cdl_cohort,
    delta_summary,
    level_summary,
)

OUT_DEFAULT = Path("/data/ssd2/nisar/validation/conus")
PANEL = "conus_nisar_station_scores.csv"

CDL = "USDA/NASS/CDL"
CULTIVATED_YEAR = 2023  # last year the `cultivated` band is published
CROP_YEAR = 2025  # latest published CDL
RADIUS_M = 100  # matches station_irrigation.py and the 200 m posting
SCALE_M = 30  # CDL native

# CDL class groupings, copied verbatim from station_landcover.py so the two runs are
# comparable. Kept explicit rather than clever -- these get argued about, and a
# reviewer should be able to read the list.
SMALL_GRAIN = {21, 22, 23, 24, 25, 27, 28, 29, 205}  # barley/wheat/rye/oats/millet
FALLOW = {61}  # fallow/idle -- the other half of a dryland rotation
ROW_CROP = {
    1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 31, 32, 33, 34, 35,
    38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
}  # fmt: skip
HAY = {36, 37}  # alfalfa, other hay
GRASS = {176}  # grassland/pasture
SHRUB = {64, 152}
FOREST = {63, 141, 142, 143}
DEVELOPED = {82, 121, 122, 123, 124}
WATER = {83, 92, 111, 112}
BARREN = {65, 131}
WETLAND = {87, 190, 195}

GROUPS = {
    "smallgrain": SMALL_GRAIN,
    "fallow": FALLOW,
    "rowcrop": ROW_CROP,
    "hay": HAY,
    "grass": GRASS,
    "shrub": SHRUB,
    "forest": FOREST,
    "developed": DEVELOPED,
    "water": WATER,
    "barren": BARREN,
    "wetland": WETLAND,
}


def _stack():
    cult = (
        ee.Image(f"{CDL}/{CULTIVATED_YEAR}")
        .select("cultivated")
        .eq(2)  # 1 = non-cultivated, 2 = cultivated
        .unmask(0)
        .rename("cult")
    )
    crop = ee.Image(f"{CDL}/{CROP_YEAR}").select("cropland").rename("cdl")
    return cult, crop


def sample_cdl(
    coords: pd.DataFrame, radius=RADIUS_M, chunk=150, project="ee-dgketchum"
):
    """Per-station CDL cultivated fraction and crop-class histogram over the buffer."""
    ee.Initialize(project=project)
    cult, crop = _stack()

    recs = []
    for i in range(0, len(coords), chunk):
        part = coords.iloc[i : i + chunk]
        feats = [
            ee.Feature(
                ee.Geometry.Point([float(r.lon), float(r.lat)]).buffer(radius),
                {"station": r.station},
            )
            for r in part.itertuples()
        ]
        fc = ee.FeatureCollection(feats)

        cult_out = cult.reduceRegions(
            collection=fc, reducer=ee.Reducer.mean(), scale=SCALE_M
        ).getInfo()
        crop_out = crop.reduceRegions(
            collection=fc, reducer=ee.Reducer.frequencyHistogram(), scale=SCALE_M
        ).getInfo()

        cult_by_stn = {
            f["properties"]["station"]: f["properties"].get("mean")
            for f in cult_out["features"]
        }

        for f in crop_out["features"]:
            p = f["properties"]
            stn = p["station"]
            hist = p.get("histogram") or {}
            total = sum(hist.values())
            rec = {"station": stn, "cult_frac": cult_by_stn.get(stn), "cdl_n_px": total}
            if total:
                fracs = {int(float(k)): v / total for k, v in hist.items()}
                rec["cdl_mode"] = max(fracs, key=fracs.get)
                rec["cdl_mode_frac"] = fracs[rec["cdl_mode"]]
                for name, classes in GROUPS.items():
                    rec[f"f_{name}"] = sum(v for k, v in fracs.items() if k in classes)
                rec["cdl_hist"] = json.dumps(
                    {k: round(v, 4) for k, v in sorted(fracs.items()) if v > 0}
                )
            recs.append(rec)
        print(f"  {min(i + chunk, len(coords))}/{len(coords)} stations", flush=True)

    out = pd.DataFrame(recs)
    out["f_crop_any"] = out[["f_smallgrain", "f_fallow", "f_rowcrop", "f_hay"]].sum(
        axis=1
    )
    out["f_dryland_grain"] = out[["f_smallgrain", "f_fallow"]].sum(axis=1)
    return out


def apply_cohorts(panel: pd.DataFrame, lc: pd.DataFrame) -> pd.DataFrame:
    """Replace the panel's land-cover columns and cohort flags with the new labels.

    The old columns are dropped outright rather than filled in only where NaN: mixing
    two runs of the same method would make the cohort counts unattributable, which is
    the whole reason for relabeling the full pool.
    """
    df = panel.drop(columns=[c for c in LC_COLS if c in panel.columns])
    df = df.drop(
        columns=[
            c
            for c in ("cdl_cohort", "is_small_grain_fallow", "is_row_crop")
            if c in df.columns
        ]
    )
    df = df.merge(lc[["station"] + LC_COLS], on="station", how="left")

    df["cdl_cohort"] = cdl_cohort(df["cult_frac"])
    # Crop-group flags are land-cover only and are NOT conditioned on irrigation
    # status; the published "dryland cropland" sub-cohorts additionally require
    # irr_class == "dryland".
    df["is_small_grain_fallow"] = (df["cdl_cohort"] == "cropland") & (
        df["f_dryland_grain"] >= GRP_T
    )
    df["is_row_crop"] = (df["cdl_cohort"] == "cropland") & (df["f_rowcrop"] >= GRP_T)
    return df


def report(before: pd.DataFrame, after: pd.DataFrame, min_pairs_col: str) -> None:
    """Before/after station counts for the cohorts the rainfed-ag thread depends on."""

    def counts(df):
        scored = df[df[min_pairs_col].notna()]
        dry = scored[(scored["irr_class"] == "dryland") & scored["cult_frac"].notna()]
        crop = dry[dry["cult_frac"] >= CROP_T]
        return {
            "labeled (pool)": int(df["cult_frac"].notna().sum()),
            "scored (>=10 pairs)": len(scored),
            "dryland, all": len(dry),
            "dryland cropland": len(crop),
            "  small grain / fallow": int((crop["f_dryland_grain"] >= GRP_T).sum()),
            "  row crop": int((crop["f_rowcrop"] >= GRP_T).sum()),
            "rangeland / non-ag": int((dry["cult_frac"] < RANGE_T).sum()),
        }

    b, a = counts(before), counts(after)
    print(f"\nCohort station counts, scored at >= 10 pairs ({min_pairs_col}):")
    print(f"  {'group':<24} {'before':>8} {'after':>8} {'delta':>8}")
    for k in b:
        print(f"  {k:<24} {b[k]:>8} {a[k]:>8} {a[k] - b[k]:>+8}")

    print("\nCDL cohort over the full extraction pool:")
    print(
        pd.DataFrame(
            {
                "before": before["cdl_cohort"].value_counts(dropna=False),
                "after": after["cdl_cohort"].value_counts(dropna=False),
            }
        ).to_string()
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--panel", type=Path, help=f"default <out-dir>/{PANEL}")
    ap.add_argument("--radius", type=float, default=RADIUS_M)
    ap.add_argument("--project", default="ee-dgketchum")
    ap.add_argument("--limit", type=int, help="pilot on the first N stations only")
    a = ap.parse_args()

    panel = pd.read_csv(a.panel or a.out_dir / PANEL)
    coords = panel[["station", "lat", "lon"]].dropna()
    if len(coords) != len(panel):
        raise SystemExit(
            f"{len(panel) - len(coords)} panel stations have no coordinates; "
            "fix the coordinate join before labeling"
        )
    if a.limit:
        coords = coords.head(a.limit)

    print(
        f"Sampling CDL {CULTIVATED_YEAR}/{CROP_YEAR} over {len(coords)} station buffers "
        f"({a.radius:g} m radius, {SCALE_M} m scale)"
    )
    lc = sample_cdl(coords, radius=a.radius, project=a.project)

    lc_path = a.out_dir / "conus_station_landcover_cdl.csv"
    lc.to_csv(lc_path, index=False)
    print(f"Wrote {lc_path}  ({len(lc)} stations)")

    if a.limit:
        raise SystemExit("pilot run: land-cover table only, panel not rewritten")

    after = apply_cohorts(panel, lc)
    summary = pd.concat(
        [
            level_summary(after, HEADLINE_CONFIG),
            level_summary(after, FULL_RECORD_CONFIG),
            delta_summary(after),
        ],
        ignore_index=True,
    )

    scores_path = a.out_dir / "conus_nisar_station_scores_cdl.csv"
    summary_path = a.out_dir / "conus_nisar_cohort_summary_cdl.csv"
    after.to_csv(scores_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {scores_path}\nWrote {summary_path}")

    report(panel, after, f"n_{HEADLINE_CONFIG.replace('results_', '')}")
