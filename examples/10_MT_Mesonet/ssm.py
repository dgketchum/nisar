"""Build the SMAP L3 SSM conditioning parquet for the mt_mesonet run-1 calibration.

Transforms the network-wide extraction parquet (station, date, smap_sm) into the
long (date, site_id, smap_l3_sm) table swim-rs's PestBuilder reads when
``ssm_calibration = true`` (see swimrs.calibrate.ssm_obs — the WP-C4 pathway).
Restricted to the SWIM cohort sites; raises if any site has no SMAP rows, because a
silently absent series is exactly how blmrubyc went missing the first time.

The SSM target is the SMAP satellite product only — in-situ Mesonet VWC never enters
calibration (validation-only, see evaluate.py).
"""

import argparse

import pandas as pd

EXTRACTIONS = "/data/ssd2/nisar/validation/smap_mesonet_extractions.parquet"
FIELDS_FGB = "/data/ssd2/nisar/swim/mt_mesonet/gis/build/mesonet_fields_100m_n9.fgb"
OUT_PARQUET = "/data/ssd2/nisar/swim/mt_mesonet/ssm/smap_l3_sites.parquet"


def build_ssm_table(
    extractions: pd.DataFrame, sites: list[str], skip_missing: bool = False
) -> pd.DataFrame:
    """Subset the extraction table to ``sites`` and rename to the PestBuilder schema.

    ``skip_missing`` is the full-network mode: ~24 stations have zero SMAP retrievals
    by construction (QC: water-adjacent, urban, dense-veg/mountainous) and are dropped
    with a loud report instead of raising. They calibrate as pure ETf+SWE — PestBuilder
    gives sites absent from the parquet zero weighted SSM obs. Leave it off for cohort
    runs, where a silently absent series is a bug (how blmrubyc went missing).
    """
    df = extractions[extractions["station"].isin(sites)]
    have = set(df["station"].unique())
    missing = [s for s in sites if s not in have]
    if missing and skip_missing:
        print(
            f"WARNING: dropping {len(missing)}/{len(sites)} sites with no SMAP rows "
            f"(zero-retrieval stations, excluded from the ablation contrast):\n"
            f"  {sorted(missing)}"
        )
    elif missing:
        raise SystemExit(
            f"no SMAP rows for {missing} — check the extraction parquet "
            "(a flagged cell needs a targeted --allow-flagged pull)"
        )
    out = df.rename(columns={"station": "site_id", "smap_sm": "smap_l3_sm"})
    out = out[["date", "site_id", "smap_l3_sm"]].sort_values(["site_id", "date"])
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build SMAP SSM conditioning parquet")
    ap.add_argument("--extractions", default=EXTRACTIONS)
    ap.add_argument("--fields", default=FIELDS_FGB)
    ap.add_argument("--out", default=OUT_PARQUET)
    ap.add_argument(
        "--skip-missing",
        action="store_true",
        help="Drop (with a report) sites that have no SMAP rows instead of raising "
        "— full-network mode, where ~24 zero-retrieval stations are expected",
    )
    args = ap.parse_args()

    import geopandas as gpd

    sites = gpd.read_file(args.fields, engine="fiona")["site_id"].tolist()
    table = build_ssm_table(
        pd.read_parquet(args.extractions), sites, skip_missing=args.skip_missing
    )

    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    span = table.groupby("site_id")["date"].agg(["size", "min", "max"])
    print(span.to_string())
    print(
        f"wrote {args.out} ({len(table)} rows, "
        f"{table['site_id'].nunique()}/{len(sites)} sites covered)"
    )


if __name__ == "__main__":
    main()
