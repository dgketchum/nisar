"""Validate NISAR L3 SME2 soil moisture against Montana Mesonet 5 cm VWC.

Unlike the CONUS-wide validation in swap-stress (which streams HDF5 remotely
per station), this reuses the GeoTIFFs already pulled locally by
pull_mesonet_frames_sme2.py: point-samples each dated, track/frame-stamped
scene at every Mesonet station inside its footprint, merges against the
bulk-archive daily 5 cm VWC, and scores per-station bias/RMSE/ubRMSE/r —
the same metrics used in the existing evidence base, for comparability.

Usage:
    uv run python scripts/validate_mesonet_sme2.py /data/ssd2/nisar/
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

STEM_RE = re.compile(r"sme2_t(\d{3})f(\d{3})([AD])_(\d{8})_soil_moisture\.tif$")
UBRMSE_GOAL = 0.06
MIN_PAIRS = 5


def load_station_frame_lookup(data_dir: Path) -> dict:
    """{(track, frame, pass_code): [(station, lon, lat), ...]}"""
    frames = pd.read_csv(data_dir / "reference/mt_mesonet_station_frames.csv")
    frames["pass_code"] = frames["passDirection"].map(
        {"Ascending": "A", "Descending": "D"}
    )

    stations = pd.read_csv(data_dir / "mesonet/mt_mesonet_stations.csv")[
        ["station", "longitude", "latitude"]
    ]
    frames = frames.merge(stations, on="station", suffixes=("_join", ""))

    lookup = {}
    for (track, frame, pass_code), grp in frames.groupby(
        ["track", "frame", "pass_code"]
    ):
        lookup[(int(track), int(frame), pass_code)] = list(
            zip(grp["station"], grp["longitude"], grp["latitude"], strict=True)
        )
    return lookup


def extract_scene(tif_path: Path, stations: list, transformer: Transformer) -> list:
    """Point-sample one SME2 GeoTIFF at every station in its footprint."""
    m = STEM_RE.search(tif_path.name)
    date_str = m.group(4)
    records = []
    with rasterio.open(tif_path) as src:
        band = src.read(1)
        for station, lon, lat in stations:
            x, y = transformer.transform(lon, lat)
            row, col = src.index(x, y)
            if not (0 <= row < band.shape[0] and 0 <= col < band.shape[1]):
                continue
            val = band[row, col]
            if np.isnan(val):
                continue
            records.append(
                {"station": station, "date": date_str, "nisar_sm": float(val)}
            )
    return records


def extract_all(data_dir: Path) -> pd.DataFrame:
    lookup = load_station_frame_lookup(data_dir)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)

    tif_paths = sorted((data_dir / "tif").glob("sme2_t*_soil_moisture.tif"))
    print(f"Scanning {len(tif_paths)} SME2 scenes...")

    all_records = []
    n_no_lookup = 0
    for tif_path in tif_paths:
        m = STEM_RE.search(tif_path.name)
        if not m:
            continue
        track, frame, pass_code = int(m.group(1)), int(m.group(2)), m.group(3)
        stations = lookup.get((track, frame, pass_code))
        if not stations:
            n_no_lookup += 1
            continue
        all_records.extend(extract_scene(tif_path, stations, transformer))

    if n_no_lookup:
        print(f"  {n_no_lookup} scenes had no matching station lookup entry")

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    agg = df.groupby(["station", "date"], as_index=False)["nisar_sm"].mean()
    print(
        f"Extracted {len(agg)} station-day NISAR SM values ({df['station'].nunique()} stations)"
    )
    return agg


def load_insitu_5cm(data_dir: Path) -> pd.DataFrame:
    long_df = pd.read_parquet(data_dir / "mesonet/mt_mesonet_daily_long.parquet")
    vwc5 = long_df[long_df["element"] == "soil_vwc_0005"][["station", "date", "value"]]
    return vwc5.rename(columns={"value": "insitu_vwc"})


def score(
    extractions: pd.DataFrame,
    insitu: pd.DataFrame,
    min_pairs: int = MIN_PAIRS,
    sat_col: str = "nisar_sm",
) -> pd.DataFrame:
    """Per-station bias/RMSE/ubRMSE/r for any satellite SM column against 5 cm VWC."""
    mean_col = sat_col.replace("_sm", "_mean")
    records = []
    n_skip_nocol = 0
    n_skip_fewpairs = 0
    for stn, ext_grp in extractions.groupby("station"):
        ins_grp = insitu[insitu["station"] == stn]
        if ins_grp.empty:
            n_skip_nocol += 1
            continue
        paired = ext_grp.merge(ins_grp, on="date", how="inner").dropna(
            subset=[sat_col, "insitu_vwc"]
        )
        if len(paired) < min_pairs:
            n_skip_fewpairs += 1
            continue

        diff = paired[sat_col] - paired["insitu_vwc"]
        bias = diff.mean()
        rmse = np.sqrt((diff**2).mean())
        ubrmse = np.sqrt(((diff - bias) ** 2).mean())
        r = paired[sat_col].corr(paired["insitu_vwc"])

        records.append(
            {
                "station": stn,
                "n_paired": len(paired),
                "bias": bias,
                "rmse": rmse,
                "ubrmse": ubrmse,
                "r": r,
                "insitu_mean": paired["insitu_vwc"].mean(),
                mean_col: paired[sat_col].mean(),
                "meets_goal": ubrmse <= UBRMSE_GOAL,
            }
        )

    print(f"Stations scored: {len(records)}")
    print(f"  Skipped (no 5cm in-situ record): {n_skip_nocol}")
    print(f"  Skipped (<{min_pairs} paired days): {n_skip_fewpairs}")
    return pd.DataFrame(records)


def summarize(
    result: pd.DataFrame, label: str = "NISAR L3 SME2 vs Montana Mesonet 5 cm VWC"
) -> None:
    if result.empty:
        print("No stations scored.")
        return
    print(f"\n{'=' * 60}")
    print(label)
    print(f"{'=' * 60}")
    print(f"Stations scored: {len(result)}")
    print(f"Median n_paired: {result['n_paired'].median():.0f}")
    print(f"Median ubRMSE:   {result['ubrmse'].median():.4f} m3/m3")
    print(f"Median r:        {result['r'].median():.3f}")
    print(f"Median bias:     {result['bias'].median():+.4f} m3/m3")
    pct_goal = 100 * result["meets_goal"].mean()
    print(
        f"Meet 0.06 goal:  {result['meets_goal'].sum()}/{len(result)} ({pct_goal:.0f}%)"
    )


if __name__ == "__main__":
    data_dir = Path(sys.argv[1])
    extractions = extract_all(data_dir)
    insitu = load_insitu_5cm(data_dir)
    result = score(extractions, insitu)

    out_dir = data_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    extractions.to_parquet(out_dir / "sme2_mesonet_extractions.parquet", index=False)
    result.to_csv(out_dir / "sme2_mesonet_station_scores.csv", index=False)
    print(f"\nWrote {out_dir / 'sme2_mesonet_extractions.parquet'}")
    print(f"Wrote {out_dir / 'sme2_mesonet_station_scores.csv'}")

    summarize(result)
