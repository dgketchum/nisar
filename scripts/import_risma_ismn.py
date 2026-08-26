"""Import the RISMA (Agriculture and Agri-Food Canada) soil-VWC record from the local
ISMN bulk export, and join its stations to the NISAR frame footprints.

NO MANUAL DOWNLOAD IS REQUIRED. RISMA is already present in this project's local ISMN
archive at ``/nas/soils/vwc_timeseries/ismn/ismn_db/RISMA`` (23 station directories),
pulled with the rest of the ISMN bulk export in the 2026-08-19 build that
``swap-stress/swapstress/sources/ismn.py`` reads. That client works exclusively from a
locally unzipped bulk export via ``ismn.interface.ISMN_Interface`` -- ISMN has no public
REST API for bulk series -- and nothing here re-requests anything from ismn.earth.

This reads the ``.stm`` header/data files directly rather than through the ``ismn``
package, which is not a dependency of this repo and would be a heavyweight addition for
a fixed five-column text format. The parse is deliberately narrow: it validates the
network name and record shape and raises on anything unexpected, rather than adapting.

    ***  THE CRITICAL FINDING, AND THE REASON THIS IS NOT A NISAR COHORT  ***

    The ISMN RISMA holdings END 2020-03-25 (earliest station end 2017-11-18). NISAR
    SME2 does not exist before 2025. There is ZERO temporal overlap, so this cohort
    cannot be paired against SME2 from ISMN at all, no matter how the stations score
    geographically.

    RISMA itself is still collecting -- the record is current on AAFC's own portal at
    agriculture.canada.ca -- but ISMN's mirror of it stopped in 2020. Extending this
    cohort into the NISAR era therefore needs an AAFC portal pull, NOT another ISMN
    export. That is a separate puller and is out of scope here.

    Retracting nothing: the note in ``notes/literature_network_leads_2026-08-25.md``
    Tier 1 #6 says RISMA "drops into the existing ISMN tooling with no new puller."
    The tooling part is right and this script demonstrates it. The "no new puller"
    part is wrong for the NISAR window and this script's own output is the evidence.

What this script is still good for: it establishes the station geography, the frame
join, the depth/sensor inventory, and the record shape so an AAFC-portal puller has a
target schema to land in -- and it gives a 2013-2020 L-band-depth VWC record usable for
ALOS-2 PALSAR era work (RISMA Manitoba is the crop case in Lal et al. 2025).

Quality flags
    Only ISMN ``G`` (good) hourly values are kept. Everything else is dropped, and the
    drop is counted by flag category in the report rather than filled or averaged
    around. This is a deliberate departure from ``swapstress/sources/ismn.py``, whose
    ``_daily_series`` falls back to the all-data daily mean on days with no good
    observation -- exactly the path that readmitted the railed 9.88 m3/m3 Waukomis
    values documented in ``import_conus_nisar_validation.py::insitu_qc``.

    At these sites the dominant non-good flags are D01/D02/D03 (in-situ soil, in-situ
    air, or GLDAS soil temperature below 0 C), i.e. frozen ground through the Canadian
    prairie winter. Dropping them is not merely conservative: an L-band surface
    soil-moisture retrieval is not defined over frozen soil, so those hours have no
    counterpart on the satellite side either.

Replicate sensors
    RISMA runs three Hydraprobe II profiles (A/B/C) at each depth. Sensors are averaged
    as a single N-way mean over whatever replicates reported a good value that day --
    never folded in pairwise, which is the other half of the Waukomis weighting bug.

Depths
    ISMN carries RISMA at 0.00-0.05 (a layer-integrated probe), then point sensors at
    0.05 / 0.20 / 0.50 / 1.00 / 1.50 m. Both ``depth_from_cm``/``depth_to_cm`` and the
    midpoint ``depth_cm`` are written, because the 0-5 cm layer and the 5 cm point are
    different measurements and the midpoint alone (2.5 cm vs 5 cm) hides that. The 0-5
    cm layer probe is the better match to the SME2 sensing depth of the two.

Output schema (long, one row per station/date/depth), aligned with
``pull_mt_mesonet.py`` and extended with the depth-layer and replicate-count columns
that a point-only network does not need::

    station, date, element, depth_cm, depth_from_cm, depth_to_cm, value, units,
    n_sensors, n_obs

Usage:
    uv run python scripts/import_risma_ismn.py
    uv run python scripts/import_risma_ismn.py --data-dir /data/ssd2/nisar
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from station_nisar_frames import join, load_frames, track_frames

ISMN_DB = Path("/nas/soils/vwc_timeseries/ismn/ismn_db")
DATA_DIR = Path("/data/ssd2/nisar")

NETWORK = "RISMA"
GOOD_FLAG = "G"
ELEMENT = "soil_vwc"
VWC_UNITS = "m3/m3"

# ISMN .stm data rows are exactly: date, time, value, ismn_flag, provider_flag.
STM_FIELDS = 5
# ISMN .stm header is: network, station_stub, station, lat, lon, elev, depth_from,
# depth_to, then the sensor name, which itself contains spaces.
HEADER_FIXED = 8

# NISAR SME2 does not exist before this; see the module docstring.
NISAR_ERA_START = pd.Timestamp("2025-01-01")

LONG_COLUMNS = [
    "station",
    "date",
    "element",
    "depth_cm",
    "depth_from_cm",
    "depth_to_cm",
    "value",
    "units",
    "n_sensors",
    "n_obs",
]


def sm_files(network_dir: Path) -> list[Path]:
    """Every soil-moisture .stm in the export, one per station x depth x replicate."""
    paths = sorted(network_dir.glob("*/*_sm_*.stm"))
    if not paths:
        raise RuntimeError(f"no soil-moisture .stm files under {network_dir}")
    return paths


def read_stm(path: Path) -> tuple[dict, pd.DataFrame]:
    """Parse one ISMN .stm into (sensor metadata, hourly frame).

    Header fields are positional and the sensor name is the free-text remainder, so the
    split is bounded at HEADER_FIXED rather than run to the end of the line.
    """
    with path.open() as fh:
        header = fh.readline().split(maxsplit=HEADER_FIXED)
    if len(header) < HEADER_FIXED + 1:
        raise RuntimeError(f"{path.name}: header has {len(header)} fields")
    if header[0] != NETWORK:
        raise RuntimeError(
            f"{path.name}: network is {header[0]!r}, expected {NETWORK!r}"
        )

    meta = {
        "station": f"{NETWORK}:{header[2]}",
        "station_name": header[2],
        "latitude": float(header[3]),
        "longitude": float(header[4]),
        "elevation_m": float(header[5]),
        "depth_from_cm": float(header[6]) * 100.0,
        "depth_to_cm": float(header[7]) * 100.0,
        "sensor": header[8].strip(),
    }

    df = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=1,
        header=None,
        names=["day", "hhmm", "value", "ismn_flag", "provider_flag"],
        dtype={"ismn_flag": str, "provider_flag": str},
    )
    if df.shape[1] != STM_FIELDS:
        raise RuntimeError(
            f"{path.name}: {df.shape[1]} data columns, expected {STM_FIELDS}"
        )
    df["datetime"] = pd.to_datetime(
        df["day"] + " " + df["hhmm"], format="%Y/%m/%d %H:%M"
    )
    return meta, df[["datetime", "value", "ismn_flag"]]


def flag_census(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """How many hourly values each ISMN flag category accounts for, good and dropped."""
    flags = pd.concat([f["ismn_flag"] for f in frames], ignore_index=True)
    # A row can carry several flags (e.g. "D01,D02,D03"); the category is the letter,
    # which is what makes the census add up to the row count instead of over-counting.
    census = flags.str[0].value_counts().rename_axis("flag_category").reset_index()
    census.columns = ["flag_category", "n_hourly"]
    census["kept"] = census["flag_category"] == GOOD_FLAG
    census["pct"] = (100 * census["n_hourly"] / len(flags)).round(1)
    return census


def daily_sensor(df: pd.DataFrame) -> pd.DataFrame:
    """Good-flagged hourly values -> one daily mean per sensor, with its hour count.

    No fallback to unflagged data on days with no good observation: a day with nothing
    good contributes nothing, which is the whole point of honouring the flag.
    """
    good = df[df["ismn_flag"] == GOOD_FLAG]
    if good.empty:
        return pd.DataFrame(columns=["date", "value", "n_obs"])
    daily = good.set_index("datetime")["value"].resample("D").agg(["mean", "size"])
    daily = daily[daily["size"] > 0]
    return daily.rename(columns={"mean": "value", "size": "n_obs"}).reset_index(
        names="date"
    )


def build_long(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read every sensor file and collapse replicates to one row per station/date/depth."""
    per_sensor, metas, raw = [], [], []
    for path in paths:
        meta, df = read_stm(path)
        raw.append(df)
        daily = daily_sensor(df)
        metas.append(
            {
                **{k: v for k, v in meta.items() if k != "sensor"},
                "sensor": meta["sensor"],
                "n_hourly": len(df),
                "n_hourly_good": int((df["ismn_flag"] == GOOD_FLAG).sum()),
                "n_days_good": len(daily),
            }
        )
        if daily.empty:
            continue
        for key in ("station", "depth_from_cm", "depth_to_cm"):
            daily[key] = meta[key]
        daily["sensor"] = meta["sensor"]
        per_sensor.append(daily)

    census = flag_census(raw)
    sensors = pd.DataFrame(metas)
    if not per_sensor:
        raise RuntimeError("no good-flagged soil-moisture observations in the export")

    stacked = pd.concat(per_sensor, ignore_index=True)
    grouped = stacked.groupby(["station", "date", "depth_from_cm", "depth_to_cm"])
    long = grouped.agg(
        value=("value", "mean"),  # single N-way mean over the replicates present
        n_sensors=("sensor", "nunique"),
        n_obs=("n_obs", "sum"),
    ).reset_index()
    long["element"] = ELEMENT
    long["units"] = VWC_UNITS
    long["depth_cm"] = 0.5 * (long["depth_from_cm"] + long["depth_to_cm"])
    long = long[LONG_COLUMNS].sort_values(["station", "date", "depth_cm"])
    return long.reset_index(drop=True), sensors, census


def station_table(long: pd.DataFrame, sensors: pd.DataFrame) -> pd.DataFrame:
    """One row per station: coordinates, record span, depth inventory, NISAR-era count."""
    coords = (
        sensors.groupby(["station", "station_name"], as_index=False)
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            elevation_m=("elevation_m", "first"),
            n_sensor_files=("sensor", "size"),
        )
        .assign(network=NETWORK)
    )
    g = long.groupby("station")
    summary = g.agg(
        n_obs=("value", "size"),
        n_days=("date", "nunique"),
        obs_start=("date", "min"),
        obs_end=("date", "max"),
        n_depths=("depth_cm", "nunique"),
    )
    summary["depths_cm"] = g["depth_cm"].agg(
        lambda s: ";".join(f"{d:g}" for d in sorted(set(s)))
    )
    # The 0-5 cm layer probe is the closest thing RISMA has to the SME2 sensing depth.
    surface = long[long["depth_to_cm"] == 5.0]
    summary["n_days_surface"] = surface.groupby("station")["date"].nunique()
    summary["n_days_nisar_era"] = (
        long[long["date"] >= NISAR_ERA_START].groupby("station")["date"].nunique()
    )
    summary[["n_days_surface", "n_days_nisar_era"]] = (
        summary[["n_days_surface", "n_days_nisar_era"]].fillna(0).astype(int)
    )
    return coords.merge(summary.reset_index(), on="station", how="left")


def write_outputs(
    out_dir: Path,
    long: pd.DataFrame,
    stations: pd.DataFrame,
    sensors: pd.DataFrame,
    census: pd.DataFrame,
    station_frames: pd.DataFrame,
    combos: pd.DataFrame,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_dir = out_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    for stale in daily_dir.glob("risma_*_daily.parquet"):
        stale.unlink()
    for station, df in long.groupby("station"):
        name = station.replace(":", "_")
        df.reset_index(drop=True).to_parquet(
            daily_dir / f"risma_{name}_daily.parquet", index=False
        )

    written = []
    for name, df in (
        ("risma_daily_long.parquet", long),
        ("risma_stations.csv", stations),
        ("risma_sensors.csv", sensors),
        ("risma_flag_census.csv", census),
        ("risma_station_frames.csv", station_frames),
        ("risma_track_frames.csv", combos),
    ):
        path = out_dir / name
        if path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        written.append(path)

    fgb = out_dir / "risma_stations.fgb"
    gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["longitude"], stations["latitude"]),
        crs="EPSG:4326",
    ).to_file(fgb, driver="FlatGeobuf", engine="fiona")
    written.append(fgb)
    return written


def report(
    long: pd.DataFrame,
    stations: pd.DataFrame,
    census: pd.DataFrame,
    station_frames: pd.DataFrame,
    combos: pd.DataFrame,
) -> None:
    print(f"\n--- {NETWORK} (ISMN bulk export) daily VWC ---")
    print(f"stations:               {stations['station'].nunique()}")
    print(f"sensor files:           {int(stations['n_sensor_files'].sum())}")
    print(f"station-date-depth rows:{len(long):,}")
    print(
        f"date range:             {long['date'].min().date()} .. {long['date'].max().date()}"
    )
    depths = sorted(long["depth_cm"].unique())
    print(f"depth midpoints (cm):   {[f'{d:g}' for d in depths]}")
    print(f"stations with 0-5 cm:   {int((stations['n_days_surface'] > 0).sum())}")
    print("\nISMN flag census (hourly values, by category):")
    print(census.to_string(index=False))

    print("\n--- NISAR frame join ---")
    print(
        f"stations in coverage:   {station_frames['station'].nunique()} of {len(stations)}"
    )
    per_station = station_frames.groupby("station").size()
    print(
        f"frames per station:     min {per_station.min()}, "
        f"median {per_station.median():.0f}, max {per_station.max()}"
    )
    print(f"unique track/frames:    {len(combos)}")
    print(f"  ascending:            {int((combos['pass_code'] == 'A').sum())}")
    print(f"  descending:           {int((combos['pass_code'] == 'D').sum())}")
    uncovered = sorted(set(stations["station"]) - set(station_frames["station"]))
    print(f"stations outside coverage: {uncovered or 'none'}")

    n_era = int(stations["n_days_nisar_era"].sum())
    print("\n--- NISAR-era overlap ---")
    print(f"station-days on or after {NISAR_ERA_START.date()}: {n_era}")
    if n_era == 0:
        print(
            "  ZERO overlap with the NISAR record. The ISMN mirror of RISMA ends "
            f"{long['date'].max().date()}; the network itself is still collecting, so "
            "a NISAR-era RISMA cohort requires an AAFC portal pull "
            "(agriculture.canada.ca), not another ISMN export. See the module docstring."
        )


def build(data_dir: Path, ismn_db: Path) -> int:
    network_dir = ismn_db / NETWORK
    if not network_dir.is_dir():
        raise SystemExit(
            f"{NETWORK} not found in the local ISMN export at {ismn_db}. "
            "Rebuild the export (swap-stress swapstress/sources/ismn.py reads it) "
            "before running this."
        )
    paths = sm_files(network_dir)
    print(
        f"reading {len(paths)} {NETWORK} soil-moisture files from {network_dir}",
        flush=True,
    )

    long, sensors, census = build_long(paths)
    stations = station_table(long, sensors)

    frames = load_frames(data_dir)
    station_frames = join(stations, frames)
    combos = track_frames(station_frames)

    out_dir = data_dir / "risma"
    written = write_outputs(
        out_dir, long, stations, sensors, census, station_frames, combos
    )
    report(long, stations, census, station_frames, combos)
    print()
    for path in written:
        print(f"wrote {path}")
    print(
        f"wrote {long['station'].nunique()} per-station parquets in {out_dir / 'daily'}"
    )
    print(f"\nrsync -rav zoran:{out_dir}/ ~/data/nisar/risma/")
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--ismn-db", type=Path, default=ISMN_DB)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args(sys.argv[1:])
    sys.exit(build(a.data_dir, a.ismn_db))
