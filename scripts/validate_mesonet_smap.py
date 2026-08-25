"""Validate SMAP L3 (SPL3SMP_E, 9 km enhanced, AM) soil moisture against Montana
Mesonet 5 cm VWC, using the same per-station bias/RMSE/ubRMSE/r methodology as
validate_mesonet_sme2.py, then print a head-to-head against the NISAR SME2 scores.

SMAP is scored across its own full available sample (every Mesonet station with a
5 cm sensor and >= MIN_PAIRS paired days), matching how SME2 was scored. The
head-to-head is restricted to stations both products scored.

Caveat carried into the printed output: SMAP is daily, SME2's revisit is sparse, so
the two products' per-station sample sizes are very different and the comparison is
not a like-for-like sampling of the same days.

Usage:
    uv run python scripts/validate_mesonet_smap.py /data/ssd2/nisar/
"""

import sys
from pathlib import Path

import pandas as pd
from validate_mesonet_sme2 import load_insitu_5cm, score, summarize

SMAP_LABEL = "SMAP L3 (SPL3SMP_E, 9 km AM) vs Montana Mesonet 5 cm VWC"


def load_extractions(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "validation/smap_mesonet_extractions.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    print(
        f"Loaded {len(df)} station-day SMAP SM values "
        f"({df['station'].nunique()} stations, "
        f"{df['date'].dt.date.nunique()} days) from {path.name}"
    )
    return df


def compare_products(sme2_scores: pd.DataFrame, smap_scores: pd.DataFrame):
    """Head-to-head on stations both products scored; returns the merged table."""
    merged = sme2_scores.merge(smap_scores, on="station", suffixes=("_nisar", "_smap"))

    print(f"\n{'=' * 70}")
    print("NISAR L3 SME2 (200 m) vs SMAP L3 (9 km AM) — Montana Mesonet 5 cm")
    print(f"{'=' * 70}\n")

    print(f"{'Metric':<25s} {'SMAP L3':>12s} {'NISAR SME2':>12s}")
    print("-" * 50)
    for metric in ["rmse", "ubrmse", "r", "bias"]:
        print(
            f"Median {metric:<18s} {merged[f'{metric}_smap'].median():>12.4f} "
            f"{merged[f'{metric}_nisar'].median():>12.4f}"
        )
    print(
        f"{'Stations scored (all)':<25s} {len(smap_scores):>12d} {len(sme2_scores):>12d}"
    )
    print(
        f"{'Median paired obs':<25s} {merged['n_paired_smap'].median():>12.0f} "
        f"{merged['n_paired_nisar'].median():>12.0f}"
    )
    print(
        f"{'Meet 0.06 ubRMSE goal':<25s} "
        f"{100 * merged['meets_goal_smap'].mean():>11.0f}% "
        f"{100 * merged['meets_goal_nisar'].mean():>11.0f}%"
    )

    print(f"\nStations in both: {len(merged)}")
    d_ub = merged["ubrmse_smap"] - merged["ubrmse_nisar"]
    d_r = merged["r_nisar"] - merged["r_smap"]
    print(
        f"  ubRMSE: NISAR lower at {(d_ub > 0).sum()}/{len(merged)} "
        f"({100 * (d_ub > 0).mean():.0f}%)"
    )
    print(
        f"  r:      NISAR higher at {(d_r > 0).sum()}/{len(merged)} "
        f"({100 * (d_r > 0).mean():.0f}%)"
    )
    print(f"  Median ubRMSE improvement (SMAP - NISAR): {d_ub.median():+.4f} m3/m3")
    print(f"  Median r improvement (NISAR - SMAP):      {d_r.median():+.4f}")

    print(
        "\nCAVEAT: these two columns are not equally sampled. SMAP is a daily 9 km "
        f"product\n(median {merged['n_paired_smap'].median():.0f} paired days per "
        "station here); NISAR SME2's revisit over this\nsample is far sparser "
        f"(median {merged['n_paired_nisar'].median():.0f} paired days). The NISAR "
        "per-station metrics rest on\nan order of magnitude fewer observations and "
        "on a different, much smaller set of\ndates, so a per-station metric "
        "difference is not a controlled head-to-head."
    )

    merged["d_ubrmse_smap_minus_nisar"] = d_ub
    merged["d_r_nisar_minus_smap"] = d_r
    return merged


if __name__ == "__main__":
    data_dir = Path(sys.argv[1])
    out_dir = data_dir / "validation"

    extractions = load_extractions(data_dir)
    insitu = load_insitu_5cm(data_dir)
    smap_scores = score(extractions, insitu, sat_col="smap_sm")

    smap_path = out_dir / "smap_mesonet_station_scores.csv"
    smap_scores.to_csv(smap_path, index=False)
    print(f"\nWrote {smap_path}")
    summarize(smap_scores, label=SMAP_LABEL)

    sme2_path = out_dir / "sme2_mesonet_station_scores.csv"
    if sme2_path.exists():
        merged = compare_products(pd.read_csv(sme2_path), smap_scores)
        cmp_path = out_dir / "sme2_vs_smap_comparison.csv"
        merged.to_csv(cmp_path, index=False)
        print(f"\nWrote {cmp_path}")
    else:
        print(f"\n{sme2_path} not found — skipping head-to-head comparison")
