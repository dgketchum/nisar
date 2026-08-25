"""Build the Montana Mesonet daily soil-VWC record for the stations that fall inside the
CONUS NISAR frame footprints, from the Montana Climate Office's bulk Parquet archive.

Source: https://data2.climate.umt.edu/mesonet/data/ -- a Hive-partitioned Parquet mirror
of the full Mesonet record, rebuilt each morning, served over plain HTTPS. We read
``daily/wide/level=2/year=<YYYY>/part-*.parquet``: daily aggregates (America/Denver
calendar days), QC level 2, which the archive's own README recommends unless there is a
reason not to. HTTPS has no directory listing, so the file list comes from the archive's
``manifest.json`` rather than a glob -- and years are split into a varying number of
parts (by station, not by date), so every part listed for a year is read.

This replaces an earlier implementation that pulled the live API
(``mesonet.climate.umt.edu/api/v2``) one station and one 45-day window at a time. The
archive is the same data, already quality-controlled, and is a few dozen HTTPS reads
instead of several thousand. The live API remains the only source for the trailing ~1 day
that postdates the archive's QC watermark (``complete_through`` in the manifest); nothing
here needs that day, so nothing here calls the API.

Scope: soil VWC at every depth Mesonet publishes. Soil temperature, bulk EC,
precipitation and the met variables are deferred to a follow-up pass; they live in the
same archive files and drop into the same long schema without a schema change.

Output schema (long format, one row per station/date/element)::

    station, date, element, depth_cm, value, units

Long rather than wide because the archive's *wide* schema evolves -- different years carry
different column sets as sensors were added -- so each year is melted to long immediately
after it is read and only the long frames are concatenated. The long schema is fixed, and
the deferred non-VWC elements will drop straight into it.

``date`` is the America/Denver calendar day the daily aggregate covers. The archive keys
each day by its UTC bucket start (e.g. ``2024-01-01T07:00:00Z`` is 1 Jan 2024 MST), so the
timestamp is converted back to Denver local time before the date is taken.

Elements come from the archive's ``elements.parquet``, which defines exactly the eight
canonical ``soil_vwc_<depth>`` variables and their depths. Recent wide files also carry
undocumented replicate columns (``soil_vwc_0005_a``, ``soil_vwc_0005_b``, ...) for
stations with duplicate probes at one depth. Those are NOT defined elements and are
deliberately excluded -- sweeping them in on a name prefix would silently double up
depths. If replicate probes are wanted later (sensor-uncertainty work), they need a
deliberate schema decision, not a wider prefix match.

Units: the archive reports VWC in percent; values are divided by 100 to m3/m3, matching
the ISMN convention used elsewhere in this project. The ``units`` column records what the
stored ``value`` actually is.

Station selection is NOT recomputed here. It is read from the already-built spatial join
``<data_dir>/reference/mt_mesonet_station_frames.csv`` (one row per station x intersecting
NISAR frame). Station metadata and coordinates come from the archive's
``stations.parquet``, whose coordinates are full precision; the rounded (~0.01 deg,
~0.6 km) coordinates the frame join was computed from are carried alongside as
``longitude_join`` / ``latitude_join`` so the provenance of the frame assignment stays
visible.

Re-running simply re-derives everything from the archive, which is cheap, so there is no
resume or skip logic: the outputs are a pure function of the archive as of the manifest
tag recorded in ``mt_mesonet_archive_manifest.json``.

Usage:
    uv run python scripts/pull_mt_mesonet.py
    uv run python scripts/pull_mt_mesonet.py --data-dir /data/ssd2/nisar
"""

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

ARCHIVE = "https://data2.climate.umt.edu/mesonet/data/"
WIDE_DAILY_PREFIX = "daily/wide/level=2/"
LOCAL_TZ = "America/Denver"
NETWORK = "MTMESONET"
DATA_DIR = Path("/data/ssd2/nisar")

VWC_PREFIX = "soil_vwc_"
VWC_SCALE = 100.0
VWC_UNITS = "m3/m3"

LONG_COLUMNS = ["station", "date", "element", "depth_cm", "value", "units"]


def archive_manifest() -> dict:
    r = requests.get(f"{ARCHIVE}manifest.json", timeout=(10, 120))
    r.raise_for_status()
    return r.json()


def year_files(manifest: dict) -> dict[int, list[str]]:
    """Group the wide daily level-2 parts by year. Part counts vary by year."""
    out: dict[int, list[str]] = {}
    for entry in manifest["files"]:
        key = entry["key"]
        if not key.startswith(WIDE_DAILY_PREFIX) or not key.endswith(".parquet"):
            continue
        year = int(key.split("year=")[1].split("/")[0])
        out.setdefault(year, []).append(ARCHIVE + key)
    return {y: sorted(urls) for y, urls in sorted(out.items())}


def vwc_elements() -> pd.DataFrame:
    """The canonical soil_vwc elements and their depths, per the archive's own registry.

    Using the registry rather than a name prefix is what keeps the undocumented
    replicate columns (soil_vwc_0005_a/_b) out of the result.
    """
    el = pd.read_parquet(f"{ARCHIVE}elements.parquet")
    el = el[el["element"].astype(str).str.startswith(VWC_PREFIX)].copy()
    if el.empty:
        raise RuntimeError("no soil_vwc elements in the archive element registry")
    # elevation_cm is negative below the surface; depth is its magnitude.
    el["depth_cm"] = el["elevation_cm"].abs()
    if not (el["base_units"] == "percent").all():
        raise RuntimeError(f"unexpected VWC units: {sorted(set(el['base_units']))}")
    return el[["element", "depth_cm"]].sort_values("depth_cm").reset_index(drop=True)


def frame_stations(data_dir: Path) -> pd.DataFrame:
    """Read the pre-built station x NISAR-frame join and collapse it to one row/station."""
    path = data_dir / "reference" / "mt_mesonet_station_frames.csv"
    df = pd.read_csv(path)

    def _agg(g):
        pairs = sorted(
            {f"{t}_{f}" for t, f in zip(g["track"], g["frame"], strict=True)}
        )
        return pd.Series(
            {
                "n_frames": len(pairs),
                "track_frames": ";".join(pairs),
                "tracks": ";".join(str(t) for t in sorted(set(g["track"]))),
                "pass_directions": ";".join(sorted(set(g["passDirection"]))),
                # The coordinates the frame join was actually computed from: the live
                # API rounds to ~0.01 deg (~0.6 km), coarser than stations.parquet.
                "longitude_join": g["longitude"].iloc[0],
                "latitude_join": g["latitude"].iloc[0],
            }
        )

    return df.groupby("station", as_index=False).apply(_agg, include_groups=False)


def melt_year(url: str, elements: pd.DataFrame, keep: set[str]) -> pd.DataFrame:
    """Read one wide part-file and reshape its VWC columns into the long schema.

    Melting per file, before any concatenation, is what makes the archive's evolving
    wide schema a non-issue: a year that predates a sensor simply contributes no rows
    for that element.
    """
    wide = pd.read_parquet(url)
    wide = wide[wide["station"].isin(keep)]
    present = [c for c in elements["element"] if c in wide.columns]
    if wide.empty or not present:
        return pd.DataFrame(columns=LONG_COLUMNS)

    long = wide.melt(
        id_vars=["station", "datetime"],
        value_vars=present,
        var_name="element",
        value_name="value",
    )
    # A null is either "this station has no probe at that depth" or "QC removed the
    # value" -- the wide shape cannot tell them apart, and neither is an observation.
    long = long.dropna(subset=["value"])
    if long.empty:
        return pd.DataFrame(columns=LONG_COLUMNS)

    # The archive keys each Denver calendar day by its UTC bucket start, so convert back
    # to Denver local time before taking the date. Re-wrapped as a tz-naive midnight
    # timestamp so the column round-trips through parquet as a date rather than as
    # opaque Python date objects.
    long["date"] = pd.to_datetime(long["datetime"].dt.tz_convert(LOCAL_TZ).dt.date)
    long["value"] = long["value"] / VWC_SCALE
    long["units"] = VWC_UNITS
    long = long.merge(elements, on="element", how="left")
    if long["depth_cm"].isna().any():
        raise RuntimeError("melted an element with no depth in the archive registry")
    return long[LONG_COLUMNS]


def load_archive(elements: pd.DataFrame, keep: set[str], files: dict[int, list[str]]):
    """Read and melt every year, reporting what each contributed."""
    frames, per_year = [], []
    for year, urls in files.items():
        parts = [melt_year(u, elements, keep) for u in urls]
        year_long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        n_rows = len(year_long)
        n_stations = year_long["station"].nunique() if n_rows else 0
        per_year.append(
            {
                "year": year,
                "n_parts": len(urls),
                "n_obs": n_rows,
                "n_stations": n_stations,
            }
        )
        print(
            f"  year={year}: {len(urls)} part(s) -> {n_rows:,} VWC obs, "
            f"{n_stations} stations",
            flush=True,
        )
        if n_rows:
            frames.append(year_long)
    if not frames:
        raise RuntimeError(
            "no VWC observations found in the archive for these stations"
        )
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["station", "date", "depth_cm"]).reset_index(
        drop=True
    )
    return combined, pd.DataFrame(per_year)


def summarize(combined: pd.DataFrame) -> pd.DataFrame:
    g = combined.groupby("station")
    out = g.agg(
        n_obs=("value", "size"),
        n_days=("date", "nunique"),
        obs_start=("date", "min"),
        obs_end=("date", "max"),
        n_depths=("depth_cm", "nunique"),
    )
    out["vwc_depths_cm"] = g["depth_cm"].agg(
        lambda s: ";".join(str(int(d)) for d in sorted(set(s)))
    )
    # 5 cm is the depth NISAR L3 SME2 is validated against, and it is far from universal
    # in this network, so it gets its own column rather than hiding inside n_depths.
    out["n_obs_5cm"] = (
        combined[combined["depth_cm"] == 5.0].groupby("station")["value"].size()
    )
    out["n_obs_5cm"] = out["n_obs_5cm"].fillna(0).astype(int)
    # Kept, not dropped: flagged so out-of-range readings stay visible for a deliberate
    # QC decision rather than being silently filtered here.
    out["n_out_of_range"] = (
        combined.assign(bad=(combined["value"] <= 0) | (combined["value"] > 1.0))
        .groupby("station")["bad"]
        .sum()
    )
    return out.reset_index()


def write_station_table(table: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    csv_path = out_dir / "mt_mesonet_stations.csv"
    fgb_path = out_dir / "mt_mesonet_stations.fgb"
    table.to_csv(csv_path, index=False)
    gdf = gpd.GeoDataFrame(
        table,
        geometry=gpd.points_from_xy(table["longitude"], table["latitude"]),
        crs="EPSG:4326",
    )
    gdf.to_file(fgb_path, driver="FlatGeobuf", engine="fiona")
    return csv_path, fgb_path


def write_per_station(combined: pd.DataFrame, daily_dir: Path) -> int:
    daily_dir.mkdir(parents=True, exist_ok=True)
    for stale in daily_dir.glob("mesonet_*_daily.parquet"):
        stale.unlink()
    for station, df in combined.groupby("station"):
        df.reset_index(drop=True).to_parquet(
            daily_dir / f"mesonet_{station}_daily.parquet", index=False
        )
    return combined["station"].nunique()


def write_manifest(
    out_dir: Path, manifest: dict, files: dict[int, list[str]], elements: pd.DataFrame
) -> Path:
    """Record exactly which archive snapshot the outputs were derived from.

    The archive changes -- gaps heal and QC verdicts get revised -- so the tag is what
    makes a build reproducible (and citable) after the fact.
    """
    path = out_dir / "mt_mesonet_archive_manifest.json"
    path.write_text(
        json.dumps(
            {
                "network": NETWORK,
                "archive": ARCHIVE,
                "archive_tag": manifest.get("tag"),
                "archive_generated_at": manifest.get("generated_at"),
                "archive_complete_through": manifest.get("complete_through"),
                "path": WIDE_DAILY_PREFIX,
                "qc_level": 2,
                "elements": elements["element"].tolist(),
                "units": "archive percent, converted to m3/m3",
                "date_basis": f"{LOCAL_TZ} calendar day",
                "files": [
                    u.removeprefix(ARCHIVE) for urls in files.values() for u in urls
                ],
            },
            indent=2,
        )
    )
    return path


def build(data_dir: Path) -> int:
    out_dir = data_dir / "mesonet"
    daily_dir = out_dir / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = frame_stations(data_dir)
    keep = set(frames["station"])
    manifest = archive_manifest()
    files = year_files(manifest)
    elements = vwc_elements()
    print(
        f"archive tag {manifest.get('tag')} (complete through "
        f"{manifest.get('complete_through')}); {sum(len(v) for v in files.values())} "
        f"level-2 daily wide files over {len(files)} years; "
        f"{len(elements)} VWC elements; {len(keep)} frame-matched stations",
        flush=True,
    )

    combined, per_year = load_archive(elements, keep, files)
    stations = pd.read_parquet(f"{ARCHIVE}stations.parquet")
    summary = summarize(combined)

    table = stations.merge(frames, on="station", how="inner").merge(
        summary, on="station", how="left"
    )
    table.insert(1, "network", NETWORK)
    table.insert(2, "station_uid", NETWORK + ":" + table["station"])

    csv_path, fgb_path = write_station_table(table, out_dir)
    n_station_files = write_per_station(combined, daily_dir)
    combined_path = out_dir / "mt_mesonet_daily_long.parquet"
    combined.to_parquet(combined_path, index=False)
    manifest_path = write_manifest(out_dir, manifest, files, elements)

    no_data = sorted(keep - set(combined["station"]))
    empty_years = per_year[per_year["n_obs"] == 0]["year"].tolist()
    n_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())

    print("\n--- Montana Mesonet daily VWC (bulk archive) ---")
    print(f"archive tag:            {manifest.get('tag')}")
    print(f"stations in frame join: {len(keep)}")
    print(f"stations with VWC:      {combined['station'].nunique()}")
    print(
        f"date range:             {combined['date'].min().date()} .. "
        f"{combined['date'].max().date()}"
    )
    print(
        f"stations with 5 cm VWC: {int((summary['n_obs_5cm'] > 0).sum())} "
        f"({int(summary['n_obs_5cm'].sum()):,} obs)"
    )
    print(f"total observations:     {len(combined):,}")
    print(f"total station-days:     {int(summary['n_days'].sum()):,}")
    depths = sorted(int(d) for d in combined["depth_cm"].unique())
    print(f"depths present (cm):    {depths}")
    print(
        f"VWC out of (0, 1]:      {int(summary['n_out_of_range'].sum()):,} (retained)"
    )
    print(f"years with no VWC:      {empty_years or 'none'}")
    if no_data:
        print(f"stations with no VWC ({len(no_data)}): {no_data}")
    else:
        print("stations with no VWC:   none")
    print(f"\nwrote {csv_path}")
    print(f"wrote {fgb_path}")
    print(f"wrote {combined_path}")
    print(f"wrote {manifest_path}")
    print(f"wrote {n_station_files} per-station parquets in {daily_dir}")
    print(f"total output size:      {n_bytes / 1e6:.1f} MB")
    print(f"\nrsync -rav zoran:{out_dir}/ ~/data/nisar/mesonet/")
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(build(parse_args(sys.argv[1:]).data_dir))
