"""Pull ARM BNF STAMP soil-moisture profiles for the three Bankhead National Forest
cropland supplemental sites (S20 Courtland, S30 Falkville, S40 Double Springs, AL).

Why these three: they are Tennessee Valley row-crop (cotton/corn/soy) fields with the
probes *in the field*, not on a mowed weather pad -- the failure mode that disqualified
the UMRB mesonets. All three carry STAMP (Soil Temperature and Moisture Profiles:
three profiles x five depths, half-hourly) from 2024-10-01, i.e. the record starts
before the NISAR window rather than after it. That combination -- in-field, cropland,
5 cm, open, and NISAR-concurrent -- is what the CONUS in-situ pool has been missing.

Source: the ARM Live Data Webservice (``https://adc.arm.gov/armlive``), which is the
programmatic face of ARM Data Discovery (``adc.arm.gov/discovery``). Two endpoints:

* ``/armlive/livedata/query?user=<user>:<token>&ds=<datastream>&start=&end=&wt=json``
  -> the list of file names in the range
* ``/armlive/livedata/saveData?user=<user>:<token>&file=<name>`` -> one netCDF file

ACCESS IS NOT ANONYMOUS. ARM data is free and unembargoed, but every request needs a
free ARM account plus that account's ARM Live access token:

  1. Register at https://adc.arm.gov/armuserreg/#/new -- self-service, free, no PI
     approval and no data request; an ORCID can be linked to autofill the form.
  2. Log in at https://adc.arm.gov/armlive/home, which shows the account's API access
     token (a 16-hex-character string).
  3. Export both before running this script::

         export ARM_USERNAME=<arm user id>
         export ARM_ACCESS_TOKEN=<16-hex token>

The token is the account's, not per-datastream: nothing here is gated beyond having an
account. An unauthenticated request returns the plain-text body ``Invalid username.``
rather than an HTTP error, so that case is checked explicitly.

Datastreams (confirmed against the ARM Data Center metadata index, 2026-08-25 --
all six run 2024-10-01 to present and are updated to within a day)::

    bnfstampS20.b1     bnfstamppcpS20.b1    S20 / US-A20, Courtland      34.6538 -87.2927
    bnfstampS30.b1     bnfstamppcpS30.b1    S30 / US-A30, Falkville      34.3848 -86.9279
    bnfstampS40.b1     bnfstamppcpS40.b1    S40 / US-A40, Double Springs 34.1788 -87.4539

DOIs: 10.5439/1238260 (stamp), 10.5439/1238261 (stamp rain gauge). The BNF main-site
STAMP systems (S10/S13/S14) are forest, not cropland, and are deliberately excluded.

Scope: volumetric water content only, in both calibrations ARM publishes --
``soil_specific_water_content_<profile>`` (soil-type-specific, the primary variable)
and ``loam_soil_water_content_<profile>`` (the loam-equivalent) -- because at a site
this project has no prior experience with, which calibration to score against is a real
decision and not one to make silently here. Soil temperature, conductivity, real
dielectric permittivity and plant water availability live in the same files and drop
into the same long schema without a schema change; they are deferred, exactly as the
non-VWC elements are in ``pull_mt_mesonet.py``. Precipitation comes from the companion
``stamppcp`` datastream and is aggregated separately -- on-site rain is load-bearing
here, since the standing finding is that rain, not irrigation, explains essentially all
detected wetting in the current sample.

Output (under ``<data-dir>/arm/``)::

    bnf_stamp_stations.csv / .fgb     one row per site, written even in list-only mode
    raw/<datastream>/*.cdf            the downloaded ARM files, untouched
    bnf_stamp_halfhourly_long.parquet station, datetime_utc, profile, element,
                                      depth_cm, value, units, qc
    bnf_stamp_daily_long.parquet      station, date, profile, element, depth_cm,
                                      value, units, n_obs
    bnf_stamp_precip_daily.parquet    station, date, precip_mm, n_minutes
    bnf_stamp_manifest.json           what was queried and downloaded

Long rather than wide for the same reason as the Mesonet archive: three profiles x five
depths x two calibrations is a wide shape that would have to change the day soil
temperature is added.

``date`` is the America/Chicago calendar day (north Alabama is Central time); ARM stores
everything in UTC, so timestamps are converted to local before the day is taken, matching
the local-day convention in ``pull_mt_mesonet.py``.

ARM reports water content in percent; values are divided by 100 to m3/m3, the ISMN
convention used everywhere else in this project. ``units`` records what ``value`` is.

QC: STAMP writes ``-9999`` where a depth was never installed or the logger dropped the
record, and a bit-packed ``qc_<var>`` companion (bit 1 missing, bit 2 below valid_min,
bit 3 above valid_max). Fill values are dropped -- they are not observations, and the
file cannot tell "no probe here" from "no record here". Out-of-range values are KEPT
with their ``qc`` code carried alongside, so the screening decision stays visible
downstream instead of being made silently in the puller.

ARM Live only serves files that are on spinning disk; anything aged off to HPSS has to
go through the ordering (staging) workflow in Data Discovery instead. The BNF record
starts 2024-10-01 and is updated to within a day, so all of it should be online -- but a
query that comes back short of the expected day count is that, not a data gap.

Bulk-pull discipline: the full record is roughly 700 days x 6 datastreams ~ 4,200 files.
The default run only queries and reports the inventory. Pass ``--download`` to actually
fetch, and ``--limit`` to cap a first pilot pull. Re-running skips files already on disk,
so an interrupted pull resumes.

Usage:
    export ARM_USERNAME=... ARM_ACCESS_TOKEN=...
    uv run python scripts/pull_arm_bnf_stamp.py                       # inventory only
    uv run python scripts/pull_arm_bnf_stamp.py --download --limit 10 # pilot
    uv run python scripts/pull_arm_bnf_stamp.py --download            # full record
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

ARMLIVE = "https://adc.arm.gov/armlive/livedata"
REGISTER_URL = "https://adc.arm.gov/armuserreg/#/new"
TOKEN_URL = "https://adc.arm.gov/armlive/home"

NETWORK = "ARMBNF"
DATA_DIR = Path("/data/ssd2/nisar")
LOCAL_TZ = "America/Chicago"

# The three BNF supplemental sites registered in AmeriFlux as cropland (CRO).
# Coordinates and elevations from arm.gov/capabilities/observatories/bnf/locations.
SITES = {
    "S20": {
        "station_name": "Courtland",
        "ameriflux": "US-A20",
        "longitude": -87.292676,
        "latitude": 34.653784,
        "elevation_m": 178,
    },
    "S30": {
        "station_name": "Falkville",
        "ameriflux": "US-A30",
        "longitude": -86.927905,
        "latitude": 34.384829,
        "elevation_m": 183,
    },
    "S40": {
        "station_name": "Double Springs",
        "ameriflux": "US-A40",
        "longitude": -87.453905,
        "latitude": 34.178796,
        "elevation_m": 236,
    },
}

SOIL_DS = "bnfstamp{facility}.b1"
PCP_DS = "bnfstamppcp{facility}.b1"
SOIL_DOI = "10.5439/1238260"
PCP_DOI = "10.5439/1238261"

RECORD_START = "2024-10-01"

PROFILES = ("west", "south", "east")
# The two water-content calibrations ARM publishes for every profile.
VWC_VARS = ("soil_specific_water_content", "loam_soil_water_content")
PCP_VAR = "precip"

FILL = -9999.0
VWC_SCALE = 100.0  # ARM reports percent; this project works in m3/m3.
VWC_UNITS = "m3/m3"
PCP_UNITS = "mm"

REQUEST_PAUSE = 0.2  # polite spacing between saveData calls
TIMEOUT = (10, 300)

FILE_DATE_RE = re.compile(r"\.(\d{8})\.(\d{6})\.(?:cdf|nc)$")

HALFHOURLY_COLUMNS = [
    "station",
    "datetime_utc",
    "profile",
    "element",
    "depth_cm",
    "value",
    "units",
    "qc",
]
DAILY_COLUMNS = [
    "station",
    "date",
    "profile",
    "element",
    "depth_cm",
    "value",
    "units",
    "n_obs",
]


def credentials() -> str:
    """The ``user=<id>:<token>`` parameter value, from the environment.

    Deliberately not read from a file or prompted for: the token is a credential, and
    the only thing this script should ever do with a missing one is refuse and say how
    to get it.
    """
    user = os.environ.get("ARM_USERNAME")
    token = os.environ.get("ARM_ACCESS_TOKEN")
    if not user or not token:
        raise SystemExit(
            "ARM_USERNAME and ARM_ACCESS_TOKEN must both be set.\n"
            f"  1. Register a free ARM account (self-service): {REGISTER_URL}\n"
            f"  2. Log in and copy the API access token: {TOKEN_URL}\n"
            "  3. export ARM_USERNAME=<user> ARM_ACCESS_TOKEN=<token>"
        )
    return f"{user}:{token}"


def check_auth(response: requests.Response, body: str | None = None) -> None:
    """Turn an ARM Live credential rejection into an actionable message.

    The service signals a bad credential two different ways depending on how the
    request was formed -- a 401, or a 200 whose body is the plain text
    ``Invalid username.`` -- so both are checked before ``raise_for_status`` would
    surface a bare HTTPError with no hint about what to fix.
    """
    head = (body if body is not None else "").strip()[:80]
    if response.status_code in (401, 403) or head.startswith(
        ("Invalid username", "Invalid token", "Invalid user")
    ):
        detail = f" ({head!r})" if head else f" (HTTP {response.status_code})"
        raise SystemExit(
            f"ARM Live rejected the credentials{detail}.\n"
            f"  Register (free, self-service): {REGISTER_URL}\n"
            f"  Copy the API access token:     {TOKEN_URL}\n"
            "  Then re-export ARM_USERNAME and ARM_ACCESS_TOKEN."
        )


def query_files(session: requests.Session, user: str, ds: str, start: str, end: str):
    """The file names ARM holds for one datastream over a date range.

    ``end`` is pushed to the last second of its day: the service compares against file
    timestamps, so a bare ``YYYY-MM-DD`` end date silently drops that day's file.
    """
    r = session.get(
        f"{ARMLIVE}/query",
        params={
            "user": user,
            "ds": ds,
            "start": f"{start}T00:00:00",
            "end": f"{end}T23:59:59",
            "wt": "json",
        },
        timeout=TIMEOUT,
    )
    check_auth(r, r.text)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"{ds}: ARM Live query failed: {payload}")
    return sorted(payload.get("files", []))


def download_file(
    session: requests.Session, user: str, name: str, dest_dir: Path
) -> tuple[Path, bool]:
    """Fetch one ARM file; returns (path, downloaded_now). Existing files are reused."""
    path = dest_dir / name
    if path.exists() and path.stat().st_size > 0:
        return path, False
    dest_dir.mkdir(parents=True, exist_ok=True)
    r = session.get(
        f"{ARMLIVE}/saveData",
        params={"user": user, "file": name},
        timeout=TIMEOUT,
        stream=True,
    )
    check_auth(r)
    r.raise_for_status()
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("wb") as fh:
        for chunk in r.iter_content(chunk_size=1 << 16):
            fh.write(chunk)
    if tmp.stat().st_size == 0:
        tmp.unlink()
        raise RuntimeError(f"{name}: ARM Live returned an empty file")
    tmp.replace(path)
    return path, True


def open_arm(path: Path):
    """Open an ARM netCDF file.

    Imported here rather than at module scope so the inventory-only run -- the one a
    user makes first, to check credentials and see how big the pull is -- does not
    require the netCDF stack to be installed.
    """
    from netCDF4 import Dataset

    return Dataset(path, "r")


def arm_time_index(ds) -> pd.DatetimeIndex:
    """ARM stores time as base_time (epoch seconds) + time_offset (seconds)."""
    base = int(ds.variables["base_time"][...])
    offset = ds.variables["time_offset"][...].astype("float64")
    return pd.to_datetime(base + offset, unit="s", utc=True)


def read_stamp_file(path: Path, station: str) -> pd.DataFrame:
    """One bnfstamp*.b1 file -> the long half-hourly VWC schema.

    Depths are read from the file's own ``depth`` coordinate rather than assumed. The
    handbook's nominal profile is 5/10/20/50/100 cm, but the deepest level varies by
    facility, and a hardcoded ladder would mislabel every value at a site that differs.
    """
    with open_arm(path) as ds:
        times = arm_time_index(ds)
        depths = ds.variables["depth"][...].astype("float64")
        blocks = []
        for var in VWC_VARS:
            for profile in PROFILES:
                name = f"{var}_{profile}"
                if name not in ds.variables:
                    continue
                values = ds.variables[name][...].astype("float64")
                qc_name = f"qc_{name}"
                qc = (
                    ds.variables[qc_name][...].astype("int64")
                    if qc_name in ds.variables
                    else None
                )
                block = pd.DataFrame(
                    {
                        "station": station,
                        "datetime_utc": times.repeat(len(depths)),
                        "profile": profile,
                        "element": var,
                        "depth_cm": list(depths) * len(times),
                        "value": values.reshape(-1),
                        "qc": (
                            qc.reshape(-1)
                            if qc is not None
                            else pd.Series(0, index=range(values.size))
                        ),
                    }
                )
                blocks.append(block)

    if not blocks:
        raise RuntimeError(f"{path.name}: no STAMP water-content variables present")

    long = pd.concat(blocks, ignore_index=True)
    # -9999 is "no probe at this depth" or "no record for this timestamp" -- the file
    # cannot distinguish them, and neither is an observation. Range-flagged values are
    # kept, with their qc code, so that screening stays a downstream decision.
    long = long[long["value"] != FILL].copy()
    long["value"] = long["value"] / VWC_SCALE
    long["units"] = VWC_UNITS
    return long[HALFHOURLY_COLUMNS]


def read_pcp_file(path: Path, station: str) -> pd.DataFrame:
    """One bnfstamppcp*.b1 file -> one-minute precipitation totals."""
    with open_arm(path) as ds:
        if PCP_VAR not in ds.variables:
            raise RuntimeError(f"{path.name}: no '{PCP_VAR}' variable")
        times = arm_time_index(ds)
        values = ds.variables[PCP_VAR][...].astype("float64")
    df = pd.DataFrame({"station": station, "datetime_utc": times, "precip_mm": values})
    return df[df["precip_mm"] != FILL].copy()


def local_day(utc: pd.Series) -> pd.Series:
    """UTC timestamps -> tz-naive midnight of the America/Chicago calendar day."""
    return pd.to_datetime(utc.dt.tz_convert(LOCAL_TZ).dt.date)


def daily_vwc(long: pd.DataFrame) -> pd.DataFrame:
    """Half-hourly VWC -> daily means, per station/profile/element/depth."""
    df = long.copy()
    df["date"] = local_day(df["datetime_utc"])
    keys = ["station", "date", "profile", "element", "depth_cm"]
    daily = df.groupby(keys, as_index=False).agg(
        value=("value", "mean"), n_obs=("value", "size")
    )
    daily["units"] = VWC_UNITS
    return daily[DAILY_COLUMNS].sort_values(keys).reset_index(drop=True)


def daily_precip(minutes: pd.DataFrame) -> pd.DataFrame:
    """One-minute accumulations -> daily totals on the same local-day basis."""
    df = minutes.copy()
    df["date"] = local_day(df["datetime_utc"])
    daily = df.groupby(["station", "date"], as_index=False).agg(
        precip_mm=("precip_mm", "sum"), n_minutes=("precip_mm", "size")
    )
    daily["units"] = PCP_UNITS
    return daily.sort_values(["station", "date"]).reset_index(drop=True)


def station_table(daily: pd.DataFrame | None) -> pd.DataFrame:
    """One row per BNF cropland site, with the columns station_nisar_frames.py needs.

    Written whether or not any data was downloaded, so the NISAR frame join can be run
    on these three sites before committing to a bulk pull.
    """
    rows = []
    for facility, meta in SITES.items():
        station = f"BNF_{facility}"
        rows.append(
            {
                "station": station,
                "network": NETWORK,
                "station_uid": f"{NETWORK}:{station}",
                "station_name": f"BNF {meta['station_name']} ({facility})",
                "facility": facility,
                "ameriflux_id": meta["ameriflux"],
                "longitude": meta["longitude"],
                "latitude": meta["latitude"],
                "elevation_m": meta["elevation_m"],
                "soil_datastream": SOIL_DS.format(facility=facility),
                "precip_datastream": PCP_DS.format(facility=facility),
                "soil_doi": SOIL_DOI,
                "precip_doi": PCP_DOI,
            }
        )
    table = pd.DataFrame(rows)
    if daily is None or daily.empty:
        return table

    g = daily.groupby("station")
    summary = g.agg(
        n_obs=("value", "size"),
        n_days=("date", "nunique"),
        obs_start=("date", "min"),
        obs_end=("date", "max"),
        n_depths=("depth_cm", "nunique"),
    )
    summary["vwc_depths_cm"] = g["depth_cm"].agg(
        lambda s: ";".join(str(int(d)) for d in sorted(set(s)))
    )
    # 5 cm is the depth SME2 is scored against, so it gets its own column.
    at_5 = daily[daily["depth_cm"] == 5.0]
    summary["n_obs_5cm"] = at_5.groupby("station")["value"].size()
    summary["n_obs_5cm"] = summary["n_obs_5cm"].fillna(0).astype(int)
    summary["n_out_of_range"] = (
        daily.assign(bad=(daily["value"] <= 0) | (daily["value"] > 1.0))
        .groupby("station")["bad"]
        .sum()
    )
    return table.merge(summary.reset_index(), on="station", how="left")


def write_station_table(table: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    csv_path = out_dir / "bnf_stamp_stations.csv"
    fgb_path = out_dir / "bnf_stamp_stations.fgb"
    table.to_csv(csv_path, index=False)
    gdf = gpd.GeoDataFrame(
        table,
        geometry=gpd.points_from_xy(table["longitude"], table["latitude"]),
        crs="EPSG:4326",
    )
    gdf.to_file(fgb_path, driver="FlatGeobuf", engine="fiona")
    return csv_path, fgb_path


def file_day(name: str) -> str:
    m = FILE_DATE_RE.search(name)
    if m is None:
        raise RuntimeError(f"unexpected ARM file name: {name}")
    return m.group(1)


def collect(
    session: requests.Session,
    user: str,
    args,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Query, optionally download, and parse every datastream for the three sites."""
    raw_root = out_dir / "raw"
    vwc_frames, pcp_frames = [], []
    inventory = {}

    for facility in SITES:
        station = f"BNF_{facility}"
        for kind, template, reader in (
            ("soil", SOIL_DS, read_stamp_file),
            ("precip", PCP_DS, read_pcp_file),
        ):
            ds = template.format(facility=facility)
            names = query_files(session, user, ds, args.start, args.end)
            days = sorted({file_day(n) for n in names})
            inventory[ds] = {
                "kind": kind,
                "n_files": len(names),
                "first_day": days[0] if days else None,
                "last_day": days[-1] if days else None,
            }
            print(
                f"{ds}: {len(names)} files "
                f"({days[0] if days else '-'} .. {days[-1] if days else '-'})",
                flush=True,
            )
            if not args.download or not names:
                continue

            selected = names[: args.limit] if args.limit else names
            dest = raw_root / ds
            n_new = n_reused = 0
            for i, name in enumerate(selected, start=1):
                path, fetched = download_file(session, user, name, dest)
                if fetched:
                    n_new += 1
                    time.sleep(REQUEST_PAUSE)
                else:
                    n_reused += 1
                frame = reader(path, station)
                (vwc_frames if kind == "soil" else pcp_frames).append(frame)
                if i % 50 == 0:
                    print(f"  {ds}: {i}/{len(selected)} files", flush=True)
            inventory[ds]["n_downloaded"] = n_new
            inventory[ds]["n_already_present"] = n_reused
            print(
                f"  {ds}: {n_new} downloaded, {n_reused} already on disk, "
                f"{len(selected)} parsed",
                flush=True,
            )

    vwc = (
        pd.concat(vwc_frames, ignore_index=True)
        if vwc_frames
        else pd.DataFrame(columns=HALFHOURLY_COLUMNS)
    )
    pcp = (
        pd.concat(pcp_frames, ignore_index=True)
        if pcp_frames
        else pd.DataFrame(columns=["station", "datetime_utc", "precip_mm"])
    )
    return vwc, pcp, inventory


def write_manifest(out_dir: Path, args, inventory: dict) -> Path:
    path = out_dir / "bnf_stamp_manifest.json"
    path.write_text(
        json.dumps(
            {
                "network": NETWORK,
                "service": ARMLIVE,
                "sites": {f: SITES[f]["ameriflux"] for f in SITES},
                "requested_range": [args.start, args.end],
                "downloaded": bool(args.download),
                "file_limit": args.limit,
                "datastreams": inventory,
                "dois": {"stamp": SOIL_DOI, "stamppcp": PCP_DOI},
                "units": "ARM percent water content, converted to m3/m3",
                "date_basis": f"{LOCAL_TZ} calendar day",
            },
            indent=2,
        )
    )
    return path


def build(args) -> int:
    user = credentials()
    out_dir = args.data_dir / "arm"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    vwc, pcp, inventory = collect(session, user, args, out_dir)

    daily = daily_vwc(vwc) if not vwc.empty else pd.DataFrame(columns=DAILY_COLUMNS)
    precip = daily_precip(pcp) if not pcp.empty else pd.DataFrame()

    table = station_table(daily if not daily.empty else None)
    csv_path, fgb_path = write_station_table(table, out_dir)
    manifest_path = write_manifest(out_dir, args, inventory)

    written = [csv_path, fgb_path, manifest_path]
    if not vwc.empty:
        hh_path = out_dir / "bnf_stamp_halfhourly_long.parquet"
        vwc.sort_values(["station", "datetime_utc", "profile", "depth_cm"]).to_parquet(
            hh_path, index=False
        )
        daily_path = out_dir / "bnf_stamp_daily_long.parquet"
        daily.to_parquet(daily_path, index=False)
        written += [hh_path, daily_path]
    if not precip.empty:
        pcp_path = out_dir / "bnf_stamp_precip_daily.parquet"
        precip.to_parquet(pcp_path, index=False)
        written.append(pcp_path)

    print("\n--- ARM BNF STAMP cropland sites ---")
    total_files = sum(v["n_files"] for v in inventory.values())
    print(f"datastreams queried:    {len(inventory)}")
    print(f"files available:        {total_files:,}")
    if vwc.empty:
        print("\nInventory only -- nothing downloaded.")
        print("Re-run with --download (and --limit N for a pilot) to fetch.")
    else:
        print(f"half-hourly VWC obs:    {len(vwc):,}")
        print(f"daily station-rows:     {len(daily):,}")
        print(
            f"date range:             {daily['date'].min().date()} .. "
            f"{daily['date'].max().date()}"
        )
        depths = sorted(int(d) for d in daily["depth_cm"].unique())
        print(f"depths present (cm):    {depths}")
        print(f"profiles:               {sorted(daily['profile'].unique())}")
        n_flagged = int((vwc["qc"] != 0).sum())
        print(f"qc-flagged (retained):  {n_flagged:,}")
        if not precip.empty:
            print(f"precip station-days:    {len(precip):,}")

    for p in written:
        print(f"wrote {p}")
    print(f"\nrsync -rav zoran:{out_dir}/ ~/data/nisar/arm/")
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--start", default=RECORD_START)
    p.add_argument("--end", default=pd.Timestamp.now(tz=LOCAL_TZ).strftime("%Y-%m-%d"))
    p.add_argument(
        "--download",
        action="store_true",
        help="actually fetch files; without it the run only reports the inventory",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="cap files fetched per datastream (0 = no cap); use for a pilot pull",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(build(parse_args(sys.argv[1:])))
