"""Validate NISAR L3 SME2 soil moisture against ARM BNF STAMP 5 cm VWC.

Reuses ``validate_mesonet_sme2``'s extraction and scoring (``extract_all``, ``score``,
``summarize``) unchanged -- point-sampling a GeoTIFF at a station and computing
bias/RMSE/ubRMSE/r doesn't depend on which network the station belongs to. What differs
is entirely on the in-situ side:

* ``arm_bnf_station_frames.csv`` (from ``station_nisar_frames.py``) already carries
  ``longitude``/``latitude``, so the lookup here is built directly from one file, no
  second merge against a stations table.
* STAMP publishes two water-content calibrations per profile
  (``soil_specific_water_content``, the soil-type-specific primary variable, and
  ``loam_soil_water_content``, the loam-equivalent). ``ELEMENT`` picks
  ``soil_specific_water_content`` rather than silently defaulting; pass
  ``--element loam_soil_water_content`` to score the other one.
* Each site runs three profiles (west/south/east) at every depth. The 5 cm in-situ value
  is the mean across whichever profiles reported that station-day -- a true N-way mean,
  not one profile picked arbitrarily or a pairwise fold (see ``import_risma_ismn.py`` for
  why that distinction matters).

Usage:
    uv run python scripts/validate_arm_bnf_sme2.py /data/ssd2/nisar/
    uv run python scripts/validate_arm_bnf_sme2.py /data/ssd2/nisar/ --element loam_soil_water_content
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from validate_mesonet_sme2 import extract_all, score, summarize

DEPTH_CM = 5.0
ELEMENT = "soil_specific_water_content"


def load_station_frame_lookup(data_dir: Path) -> dict:
    """{(track, frame, pass_code): [(station, lon, lat), ...]}, for the ARM BNF sites."""
    frames = pd.read_csv(data_dir / "reference/arm_bnf_station_frames.csv")
    frames["pass_code"] = frames["passDirection"].map(
        {"Ascending": "A", "Descending": "D"}
    )
    lookup = {}
    for (track, frame, pass_code), grp in frames.groupby(
        ["track", "frame", "pass_code"]
    ):
        lookup[(int(track), int(frame), pass_code)] = list(
            zip(grp["station"], grp["longitude"], grp["latitude"], strict=True)
        )
    return lookup


def load_insitu_5cm(data_dir: Path, element: str = ELEMENT) -> pd.DataFrame:
    """Daily 5 cm VWC, averaged across whichever profiles reported that station-day."""
    long_df = pd.read_parquet(data_dir / "arm/bnf_stamp_daily_long.parquet")
    at_5cm = long_df[
        (long_df["element"] == element) & (long_df["depth_cm"] == DEPTH_CM)
    ]
    if at_5cm.empty:
        raise RuntimeError(
            f"no rows at depth_cm={DEPTH_CM} for element={element!r}; "
            f"elements present: {sorted(long_df['element'].unique())}"
        )
    daily = at_5cm.groupby(["station", "date"], as_index=False).agg(
        insitu_vwc=("value", "mean"), n_profiles=("profile", "nunique")
    )
    return daily[["station", "date", "insitu_vwc", "n_profiles"]]


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("data_dir", type=Path)
    p.add_argument(
        "--element",
        default=ELEMENT,
        choices=["soil_specific_water_content", "loam_soil_water_content"],
        help="which STAMP water-content calibration to score against (default: %(default)s)",
    )
    return p.parse_args(argv)


def build(data_dir: Path, element: str) -> int:
    lookup = load_station_frame_lookup(data_dir)
    extractions = extract_all(data_dir, lookup=lookup)
    insitu = load_insitu_5cm(data_dir, element)
    result = score(extractions, insitu)

    out_dir = data_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    extractions.to_parquet(out_dir / "sme2_arm_bnf_extractions.parquet", index=False)
    result.to_csv(out_dir / "sme2_arm_bnf_station_scores.csv", index=False)
    print(f"\nWrote {out_dir / 'sme2_arm_bnf_extractions.parquet'}")
    print(f"Wrote {out_dir / 'sme2_arm_bnf_station_scores.csv'}")

    summarize(result, label=f"NISAR L3 SME2 vs ARM BNF STAMP 5 cm VWC ({element})")
    return 0


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    sys.exit(build(args.data_dir, args.element))
