"""Stratify the SMAP-vs-SME2 head-to-head by sub-9km land-cover heterogeneity.

Tests the keystone prediction in notes/proposal_prelim_panel_2026-08-26.md (panel 3):
NISAR SME2's standing relative to SMAP L3 should improve where the SMAP pixel is
land-cover-heterogeneous, because a homogeneous pixel is representative of the probe and
SMAP's revisit/record advantages dominate there, while a mixed pixel is exactly where
200 m resolution carries information 9 km cannot. If the gradient is absent at current
record length, that is the finding -- know it before writing the proposal, not after.

Three heterogeneity metrics per station, all from sample_smap_pixel_landcover.py output
(both sides sampled from precise coordinates, so none inherit the 2-decimal Mesonet
coordinate defect):

* ``entropy_9km`` -- Shannon entropy of the CDL group fractions over the 9 km cell
* ``mix_9km`` -- 1 minus the dominant group's fraction over the cell
* ``cult_contrast`` -- |cultivated fraction at 100 m minus over the cell|, the
  probe-vs-pixel disagreement specifically

Also written, because the same join answers them (plan work items 2-3):

* the panel-1 representativeness classification -- a station is ``representative`` when
  its local dominant CDL group matches the cell's and the cultivated contrast is small,
  i.e. the probe's ground plausibly stands for the pixel SMAP reported
* the panel-2 candidate-pixel shortlist -- cells holding a representative station whose
  composition is genuinely mixed agricultural (cultivated fraction mid-range, irrigation
  present), the venue for the sub-pixel heterogeneity demo

The head-to-head r values ride on very unequal samples (median 12 NISAR pairs vs 219
SMAP); every statistic is therefore reported for all stations and for the
``n_paired_nisar >= 10`` subset, and neither is quotable without that caveat.

Usage:
    uv run python scripts/stratify_smap_sme2_by_heterogeneity.py /data/ssd2/nisar/
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

MIN_NISAR_PAIRS = 10
CULT_CONTRAST_MAX = (
    0.15  # probe-vs-pixel cultivated disagreement a "representative" site may carry
)
SHORTLIST_CULT_RANGE = (
    0.2,
    0.8,
)  # mid-range cultivated fraction = genuinely mixed cell
SHORTLIST_IRR_MIN = 0.05
DELTA_COLS = ["d_r_nisar_minus_smap", "d_ubrmse_smap_minus_nisar"]
HET_COLS = ["entropy_9km", "mix_9km", "cult_contrast"]


def group_columns(df: pd.DataFrame, suffix: str) -> list:
    return [c for c in df.columns if c.startswith("f_") and c.endswith(f"_{suffix}")]


def dominant_group(df: pd.DataFrame, suffix: str) -> pd.Series:
    cols = group_columns(df, suffix)
    return df[cols].idxmax(axis=1).str.removeprefix("f_").str.removesuffix(f"_{suffix}")


def load(data_dir: Path) -> pd.DataFrame:
    cmp = pd.read_csv(data_dir / "validation/sme2_vs_smap_comparison.csv")
    pix = pd.read_csv(data_dir / "reference/smap_pixel_landcover.csv")
    missing = sorted(set(cmp["station"]) - set(pix["station"]))
    if missing:
        raise ValueError(
            f"{len(missing)} head-to-head station(s) absent from the pixel land-cover "
            f"sample -- rerun sample_smap_pixel_landcover.py before stratifying: {missing}"
        )
    df = cmp.merge(pix, on="station", how="left", validate="1:1")

    df["mix_9km"] = 1.0 - df[group_columns(df, "9km")].max(axis=1)
    df["cult_contrast"] = (df["cult_f_100"] - df["cult_f_9km"]).abs()
    df["dom_group_100"] = dominant_group(df, "100")
    df["dom_group_9km"] = dominant_group(df, "9km")
    df["dom_group_match"] = df["dom_group_100"] == df["dom_group_9km"]
    df["representative"] = df["dom_group_match"] & (
        df["cult_contrast"] <= CULT_CONTRAST_MAX
    )
    return df


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    subsets = {
        "all": df,
        f">={MIN_NISAR_PAIRS}_nisar_pairs": df[df["n_paired_nisar"] >= MIN_NISAR_PAIRS],
    }
    for name, sub in subsets.items():
        for delta in DELTA_COLS:
            for het in HET_COLS:
                rho, p = spearmanr(sub[delta], sub[het])
                records.append(
                    {
                        "subset": name,
                        "n": len(sub),
                        "delta": delta,
                        "heterogeneity": het,
                        "spearman_rho": rho,
                        "p_value": p,
                    }
                )
    return pd.DataFrame(records)


def tercile_table(df: pd.DataFrame, het: str = "entropy_9km") -> pd.DataFrame:
    binned = df.copy()
    binned["tercile"] = pd.qcut(binned[het], 3, labels=["low", "mid", "high"])
    return (
        binned.groupby("tercile", observed=True)
        .agg(
            n=("station", "count"),
            het_median=(het, "median"),
            d_r_median=("d_r_nisar_minus_smap", "median"),
            d_ubrmse_median=("d_ubrmse_smap_minus_nisar", "median"),
            r_nisar_median=("r_nisar", "median"),
            r_smap_median=("r_smap", "median"),
        )
        .reset_index()
    )


def figure(df: pd.DataFrame, fig_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, het, label in zip(
        axes,
        ["entropy_9km", "cult_contrast"],
        ["9 km cell CDL-group entropy", "|cultivated: 100 m - 9 km cell|"],
    ):
        thin = df["n_paired_nisar"] < MIN_NISAR_PAIRS
        ax.scatter(
            df.loc[thin, het],
            df.loc[thin, "d_r_nisar_minus_smap"],
            s=18,
            alpha=0.4,
            color="gray",
            label=f"<{MIN_NISAR_PAIRS} NISAR pairs",
        )
        ax.scatter(
            df.loc[~thin, het],
            df.loc[~thin, "d_r_nisar_minus_smap"],
            s=24,
            alpha=0.8,
            color="tab:blue",
            label=f">={MIN_NISAR_PAIRS} NISAR pairs",
        )
        binned = df.copy()
        binned["bin"] = pd.qcut(binned[het], 3, duplicates="drop")
        med = binned.groupby("bin", observed=True).agg(
            x=(het, "median"), y=("d_r_nisar_minus_smap", "median")
        )
        ax.plot(med["x"], med["y"], "o-", color="tab:red", label="tercile median")
        ax.axhline(0, lw=0.8, color="k")
        ax.set_xlabel(label)
    axes[0].set_ylabel("r(NISAR) - r(SMAP)")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "SMAP L3 (9 km) vs NISAR SME2 (200 m) at MT Mesonet, by sub-pixel heterogeneity"
    )
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200)
    print(f"Wrote {fig_path}")


def shortlist(df: pd.DataFrame) -> pd.DataFrame:
    """Panel-2 candidate cells: representative station in a mixed agricultural pixel."""
    lo, hi = SHORTLIST_CULT_RANGE
    ok = df[
        df["representative"]
        & df["cult_f_9km"].between(lo, hi)
        & (df["irr_f_9km"] >= SHORTLIST_IRR_MIN)
    ]
    cols = [
        "cell",
        "station",
        "longitude",
        "latitude",
        "cult_f_9km",
        "irr_f_9km",
        "entropy_9km",
        "dom_group_9km",
        "r_smap",
        "n_paired_smap",
    ]
    return ok[cols].sort_values("irr_f_9km", ascending=False)


def build(data_dir: Path) -> int:
    df = load(data_dir)

    corr = correlation_table(df)
    terc = tercile_table(df)
    corr_path = data_dir / "validation/smap_sme2_heterogeneity_correlations.csv"
    terc_path = data_dir / "validation/smap_sme2_heterogeneity_terciles.csv"
    rep_path = data_dir / "reference/mesonet_smap_representativeness.csv"
    short_path = data_dir / "reference/panel2_candidate_pixels.csv"
    corr.to_csv(corr_path, index=False)
    terc.to_csv(terc_path, index=False)
    rep_cols = [
        "station",
        "longitude",
        "latitude",
        "cell",
        "cult_f_pt",
        "cult_f_100",
        "cult_f_9km",
        "irr_f_9km",
        "cult_contrast",
        "dom_group_100",
        "dom_group_9km",
        "dom_group_match",
        "entropy_9km",
        "mix_9km",
        "representative",
    ]
    df[rep_cols].to_csv(rep_path, index=False)
    short = shortlist(df)
    short.to_csv(short_path, index=False)
    figure(df, data_dir / "figs/smap_sme2_heterogeneity.png")

    print(
        f"\nWrote {corr_path}\nWrote {terc_path}\nWrote {rep_path}\nWrote {short_path}"
    )
    print(f"\nStations: {len(df)}; representative: {int(df['representative'].sum())}")
    print(f"Panel-2 candidate cells: {short['cell'].nunique()}")
    print("\nSpearman rho, d_r vs heterogeneity:")
    print(
        corr[corr["delta"] == "d_r_nisar_minus_smap"]
        .pivot(index="heterogeneity", columns="subset", values="spearman_rho")
        .round(3)
        .to_string()
    )
    print("\nEntropy terciles:")
    print(terc.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(build(Path(sys.argv[1])))
