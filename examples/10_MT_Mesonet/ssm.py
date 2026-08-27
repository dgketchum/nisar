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


def build_ssm_table(extractions: pd.DataFrame, sites: list[str]) -> pd.DataFrame:
    """Subset the extraction table to ``sites`` and rename to the PestBuilder schema."""
    df = extractions[extractions["station"].isin(sites)]
    have = set(df["station"].unique())
    missing = [s for s in sites if s not in have]
    if missing:
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
    args = ap.parse_args()

    import geopandas as gpd

    sites = gpd.read_file(args.fields, engine="fiona")["site_id"].tolist()
    table = build_ssm_table(pd.read_parquet(args.extractions), sites)

    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    span = table.groupby("site_id")["date"].agg(["size", "min", "max"])
    print(span.to_string())
    print(f"wrote {args.out} ({len(table)} rows, {len(sites)} sites)")


if __name__ == "__main__":
    main()
