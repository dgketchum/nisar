"""Pull NEON soil water content (DP1.00094.001) at the in-field cropland sites STER and KONA.

Why these two sites: every UMRB state mesonet sites its probes on a maintained grass pad,
so the existing panel has almost no probe that is actually inside a farmed field. NEON's
STER (North Sterling, CO -- winter wheat/fallow dryland rotation) and KONA (Konza Prairie
Agroecosystem, Riley Co. KS -- wheat/corn/milo/soy/alfalfa/oats) put five soil plots each
inside the cultivated field, and both are rainfed. See
``notes/literature_network_leads_2026-08-25.md`` sections 2 and 3.

Source: the NEON Data API (``data.neonscience.org/api/v0``). There is an official Python
port of ``neonUtilities`` on PyPI (``neonutilities``, NEONScience/NEON-utilities-python),
but this pull is four REST calls' worth of logic against a documented API, so it is
hand-rolled with ``requests`` rather than taking the dependency -- the file selection, the
release pinning and the depth correction below all need to be explicit anyway.

Auth: the metadata endpoints (``/sites``, ``/locations``, ``/products``, ``/documents``)
are open, but ``/data/{product}/{site}/{month}`` returns 403 Access Denied without an API
token. A free token is read from ``$NEON_TOKEN`` or ``~/neon_token.txt`` and sent as
``X-API-Token``. The token also lifts the rate limit from 200 to 2000 requests/hour. The
per-file download URLs the API hands back are pre-signed Google Cloud Storage links and
need no header of their own.

What is pulled: the ``SWS_30_minute`` **basic** package only -- 5 soil plots x 8 depths =
40 CSVs per site-month, ~8 MB. The full month package is ~940 MB because it also carries
1-minute and expanded variants; those are not read. 30-minute (not daily) is the point:
NISAR's overpass is at a fixed local time, so the comparison has to be made at that time
of day, which a daily mean would destroy.

**Sensor depths do not come from the download package.** NEON has a standing, published
erratum: the ``zOffset`` in ``sensor_positions`` is wrong for this product at every
terrestrial site (it reports the co-located soil *temperature* profile's depth ladder,
which is offset by one level -- e.g. STER VER 501 shows 0.00 m where the sensor is at
0.06 m). Depths are therefore taken from NEON's corrected
``/documents/swc_depthsV3`` table, keyed on (site, HOR, VER) with validity dates. The
script cross-checks that the package's own sensor_positions disagrees exactly as the
erratum describes and fails loudly if the erratum is ever fixed upstream, so this
correction cannot silently go stale.

Release pinning: months covered by a numbered NEON release (RELEASE-2026 at time of
writing) are requested from that release; the trailing months are PROVISIONAL and change.
The release actually served for each month is recorded in the manifest.

Outputs, under ``<data-dir>/neon/``:

* ``monthly/{site}_{YYYY-MM}_swc30.parquet`` -- one resumable chunk per site-month.
* ``neon_swc_30min.parquet`` -- concatenated half-hourly record::

      station, datetime_utc, depth_cm, vwc, vwc_min, vwc_max, vwc_exp_uncert,
      vwc_n_pts, vwc_qf, vsic, vsic_qf

* ``neon_daily_long.parquet`` -- daily aggregate in the same long schema
  ``pull_mt_mesonet.py`` writes, so the validation scripts can read it by column name::

      station, date, element, depth_cm, value, units, n_obs, n_flagged

* ``neon_soil_plots.csv`` / ``.fgb`` -- per-plot metadata: coordinates, depths, record
  span, and the NISAR track/frame join.
* ``neon_pull_manifest.json`` -- product, releases served, months, file counts, bytes.

Rows are never dropped for QC here. ``vwc_qf`` is NEON's ``VSWCFinalQF`` (0 pass, 1 fail)
and is carried through; only the *daily* aggregate restricts to ``vwc_qf == 0``, and it
records ``n_obs`` and ``n_flagged`` so the exclusion stays visible.

Usage:
    # what a pull would cost, without downloading anything
    uv run python scripts/pull_neon_soil_moisture.py --estimate
    # pilot: one month, both sites
    uv run python scripts/pull_neon_soil_moisture.py --start 2026-06 --end 2026-06
    # full record (confirm the estimate first)
    uv run python scripts/pull_neon_soil_moisture.py
"""

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from station_nisar_frames import join, load_frames, track_frames

API = "https://data.neonscience.org/api/v0"
PRODUCT = "DP1.00094.001"
SWC_DEPTHS_DOC = f"{API}/documents/swc_depthsV3"
NETWORK = "NEON"
PREFIX = "neon"
DATA_DIR = Path("/data/ssd2/nisar")

SITES = ("STER", "KONA")
# NEON reports every timestamp in UTC; the local zone is only used to assign a calendar
# day to the daily aggregate, matching how the Mesonet daily record is keyed.
SITE_TZ = {"STER": "America/Denver", "KONA": "America/Chicago"}

TABLE = "SWS_30_minute"
PACKAGE = "basic"
VWC_UNITS = "m3/m3"
PROVISIONAL = "PROVISIONAL"

TOKEN_FILE = Path.home() / "neon_token.txt"
RETRIES = 4
BACKOFF = 3.0
WORKERS = 8

# NEON's published erratum: sensor_positions reports the soil temperature ladder, which
# sits one level shallower than the water content ladder. VER 501 is the tell -- it points
# at the profile assembly (~0 m) instead of the ~0.06 m shallowest moisture sensor.
ERRATUM_VER = 501
ERRATUM_MAX_ABS_Z = 0.02

THIRTY_MIN_COLUMNS = [
    "station",
    "datetime_utc",
    "depth_cm",
    "vwc",
    "vwc_min",
    "vwc_max",
    "vwc_exp_uncert",
    "vwc_n_pts",
    "vwc_qf",
    "vsic",
    "vsic_qf",
]
CSV_RENAME = {
    "VSWCMean": "vwc",
    "VSWCMinimum": "vwc_min",
    "VSWCMaximum": "vwc_max",
    "VSWCExpUncert": "vwc_exp_uncert",
    "VSWCNumPts": "vwc_n_pts",
    "VSWCFinalQF": "vwc_qf",
    "VSICMean": "vsic",
    "VSICFinalQF": "vsic_qf",
}
DAILY_COLUMNS = [
    "station",
    "date",
    "element",
    "depth_cm",
    "value",
    "units",
    "n_obs",
    "n_flagged",
]


def token() -> str:
    tok = os.environ.get("NEON_TOKEN")
    if not tok and TOKEN_FILE.exists():
        tok = TOKEN_FILE.read_text().strip()
    if not tok:
        raise RuntimeError(
            f"the NEON /data endpoint returns 403 without a token; set $NEON_TOKEN or "
            f"put one in {TOKEN_FILE} (free, from data.neonscience.org/myaccount)"
        )
    return tok


def get(
    url: str, tok: str | None = None, params: dict | None = None, raw: bool = False
):
    """One GET with retry. ``raw`` returns the response for pre-signed file URLs."""
    headers = {"X-API-Token": tok} if tok else {}
    last = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=(10, 300))
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
            elif r.status_code >= 400:
                # A 4xx is a bad request, not a transient failure -- retrying it just
                # burns rate limit and hides the real message.
                raise RuntimeError(f"GET {url} -> HTTP {r.status_code}: {r.text[:300]}")
            else:
                return r if raw else r.json()["data"]
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(BACKOFF * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {RETRIES} attempts: {last}")


def station_id(site: str, hor: int) -> str:
    return f"{site}_SP{hor}"


def site_month_releases(site: str, tok: str) -> dict[str, str]:
    """Month -> the release to request, preferring a numbered release over PROVISIONAL.

    A month that has been folded into a numbered release is frozen and citable; the
    trailing PROVISIONAL months are still being revised. Requesting the numbered release
    where one exists is what makes a rebuild reproducible.
    """
    data = get(f"{API}/sites/{site}", tok)
    prods = [p for p in data["dataProducts"] if p["dataProductCode"] == PRODUCT]
    if not prods:
        raise RuntimeError(f"{site} does not carry {PRODUCT}")
    out: dict[str, str] = {}
    # PROVISIONAL first, then numbered releases ascending, so the newest numbered release
    # wins for any month that appears in more than one.
    ordered = sorted(
        prods[0]["availableReleases"],
        key=lambda r: (r["release"] != PROVISIONAL, r["release"]),
    )
    for rel in ordered:
        for m in rel["availableMonths"]:
            out[m] = rel["release"]
    return dict(sorted(out.items()))


def swc_depths() -> pd.DataFrame:
    """NEON's corrected soil water content installation depths, per the erratum."""
    r = get(SWC_DEPTHS_DOC, raw=True)
    df = pd.read_csv(io.StringIO(r.text))
    df = df[df["siteID"].isin(SITES)].copy()
    if df.empty:
        raise RuntimeError(f"swc_depthsV3 has no rows for {SITES}")
    df = df.rename(
        columns={
            "siteID": "site",
            "horizontalPosition.HOR": "hor",
            "verticalPosition.VER": "ver",
        }
    )
    df["depth_cm"] = (-df["sensorDepth"] * 100.0).round(1)
    if (df["depth_cm"] <= 0).any():
        raise RuntimeError("swc_depthsV3 reports a non-positive installation depth")
    df["start"] = pd.to_datetime(df["startDateTime"], utc=True)
    df["end"] = pd.to_datetime(df["endDateTime"], utc=True)
    return df[["site", "hor", "ver", "depth_cm", "start", "end"]].reset_index(drop=True)


def depth_lookup(
    depths: pd.DataFrame, site: str, month: str
) -> dict[tuple[int, int], float]:
    """(HOR, VER) -> depth in cm, for the sensor configuration valid during ``month``."""
    at = pd.Timestamp(f"{month}-01", tz="UTC")
    sub = depths[depths["site"] == site]
    sub = sub[(sub["start"] <= at) & (sub["end"].isna() | (sub["end"] > at))]
    dup = sub.duplicated(["hor", "ver"], keep=False)
    if dup.any():
        raise RuntimeError(
            f"{site} {month}: swc_depthsV3 gives >1 valid depth for "
            f"{sorted(set(zip(sub.loc[dup, 'hor'], sub.loc[dup, 'ver'], strict=True)))}"
        )
    return {
        (int(h), int(v)): float(d)
        for h, v, d in zip(sub["hor"], sub["ver"], sub["depth_cm"], strict=True)
    }


def check_erratum(files: list[dict], site: str, month: str) -> None:
    """Confirm the download package's sensor_positions is still the erratum's wrong one.

    If NEON fixes the product, ``zOffset`` at VER 501 becomes a real depth and this
    raises -- at which point the corrected-depths document should be dropped in favour of
    the package's own file. Failing loudly is the point: a silently stale correction would
    mislabel every depth in the panel.
    """
    sp = [f for f in files if "sensor_positions" in f["name"]]
    if not sp:
        raise RuntimeError(f"{site} {month}: no sensor_positions file in the package")
    df = pd.read_csv(
        io.StringIO(get(sp[0]["url"], raw=True).text), dtype={"HOR.VER": str}
    )
    ver501 = df[df["HOR.VER"].str.endswith(f".{ERRATUM_VER}")]
    if ver501.empty:
        raise RuntimeError(
            f"{site} {month}: sensor_positions has no VER {ERRATUM_VER} row"
        )
    if (ver501["zOffset"].abs() > ERRATUM_MAX_ABS_Z).all():
        raise RuntimeError(
            f"{site} {month}: sensor_positions VER {ERRATUM_VER} zOffset is no longer ~0 "
            f"({sorted(set(ver501['zOffset']))}) -- the NEON depth erratum may be fixed; "
            "re-check whether swc_depthsV3 is still the right depth source"
        )


def month_files(
    site: str, month: str, release: str, tok: str
) -> tuple[list[dict], dict]:
    """The 30-minute basic CSVs for one site-month, plus the raw package listing.

    PROVISIONAL is the API's default and is rejected as an explicit ``release`` value
    (HTTP 400), so it is passed only for numbered releases.
    """
    params = {"release": release} if release != PROVISIONAL else None
    data = get(f"{API}/data/{PRODUCT}/{site}/{month}", tok, params=params)
    files = data["files"]
    wanted = [
        f
        for f in files
        if TABLE in f["name"]
        and f".{PACKAGE}." in f["name"]
        and f["name"].endswith(".csv")
    ]
    return wanted, data


def parse_hor_ver(name: str) -> tuple[int, int]:
    """``NEON.D10.STER.DP1.00094.001.002.505.030.SWS_30_minute...`` -> (2, 505)."""
    parts = name.split(".")
    i = parts.index("00094")
    return int(parts[i + 2]), int(parts[i + 3])


def read_file(url: str, station: str, depth_cm: float) -> pd.DataFrame:
    text = get(url, raw=True).text
    df = pd.read_csv(io.StringIO(text))
    missing = [c for c in CSV_RENAME if c not in df.columns]
    if missing:
        raise RuntimeError(f"{url} is missing expected columns: {missing}")
    df = df.rename(columns=CSV_RENAME)
    df["station"] = station
    df["depth_cm"] = depth_cm
    df["datetime_utc"] = pd.to_datetime(df["startDateTime"], utc=True)
    return df[THIRTY_MIN_COLUMNS]


def pull_month(
    site: str,
    month: str,
    release: str,
    depths: pd.DataFrame,
    tok: str,
    out_dir: Path,
    overwrite: bool,
    verify_depths: bool,
) -> tuple[pd.DataFrame, dict]:
    chunk = out_dir / f"{site}_{month}_swc30.parquet"
    if chunk.exists() and not overwrite:
        df = pd.read_parquet(chunk)
        return df, {"site": site, "month": month, "cached": True, "n_rows": len(df)}

    wanted, data = month_files(site, month, release, tok)
    if verify_depths:
        check_erratum(data["files"], site, month)
    lut = depth_lookup(depths, site, month)

    jobs, n_bytes = [], 0
    for f in wanted:
        hor, ver = parse_hor_ver(f["name"])
        if (hor, ver) not in lut:
            raise RuntimeError(
                f"{site} {month}: no corrected depth for HOR {hor} VER {ver}; "
                "swc_depthsV3 does not cover this sensor position"
            )
        jobs.append((f["url"], station_id(site, hor), lut[(hor, ver)]))
        n_bytes += f["size"]

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        frames = list(pool.map(lambda j: read_file(*j), jobs))

    if not frames:
        raise RuntimeError(f"{site} {month}: package has no {TABLE} {PACKAGE} files")
    df = pd.concat(frames, ignore_index=True).sort_values(
        ["station", "datetime_utc", "depth_cm"]
    )
    df = df.reset_index(drop=True)
    chunk.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(chunk, index=False)
    return df, {
        "site": site,
        "month": month,
        "cached": False,
        "release": data.get("release", release),
        "n_files": len(wanted),
        "n_bytes_csv": n_bytes,
        "n_rows": len(df),
    }


def daily_long(thirty: pd.DataFrame) -> pd.DataFrame:
    """Half-hourly -> daily mean on the site's local calendar day, mesonet long schema."""
    site = thirty["station"].str.split("_").str[0]
    tz = site.map(SITE_TZ)
    if tz.isna().any():
        raise RuntimeError(f"no timezone for sites {sorted(set(site[tz.isna()]))}")
    local_day = pd.Series(pd.NaT, index=thirty.index, dtype="datetime64[ns]")
    for zone in sorted(tz.unique()):
        idx = tz.index[tz == zone]
        conv = thirty.loc[idx, "datetime_utc"].dt.tz_convert(zone)
        local_day.loc[idx] = pd.to_datetime(conv.dt.date)

    work = thirty.assign(date=local_day)
    work["flagged"] = work["vwc_qf"].fillna(1) != 0
    # A flagged half-hour is not an observation, so it is excluded from the daily mean --
    # but the count of what was excluded travels with the row rather than vanishing.
    work["vwc_ok"] = work["vwc"].where(~work["flagged"])
    out = work.groupby(["station", "date", "depth_cm"], as_index=False).agg(
        value=("vwc_ok", "mean"),
        n_obs=("vwc_ok", "count"),
        n_flagged=("flagged", "sum"),
    )
    # A day whose every half-hour failed QC has no mean; it is a real gap in the record,
    # not a value to impute, so it is dropped here and counted by the caller.
    out = out[out["n_obs"] > 0].copy()
    # astype(str) rather than .map("{:04d}".format): on an empty Series (a site-month
    # where every observation failed QC) .map() returns int64, not object, and the
    # concatenation below raises instead of producing an empty frame.
    out["element"] = "soil_vwc_" + out["depth_cm"].round().astype(int).astype(
        str
    ).str.zfill(4)
    out["units"] = VWC_UNITS
    out["n_flagged"] = out["n_flagged"].astype(int)
    return (
        out[DAILY_COLUMNS]
        .sort_values(["station", "date", "depth_cm"])
        .reset_index(drop=True)
    )


def soil_plots(site: str) -> pd.DataFrame:
    """Per-plot coordinates from the NEON locations tree (SITE -> SOIL_ARRAY -> SOIL_PLOT).

    Each soil plot publishes a ~10 m corner polygon; the plot point is its centroid.
    """
    site_loc = get(f"{API}/locations/{site}")
    arrays = [c for c in site_loc["locationChildren"] if c.startswith("SOILAR")]
    if len(arrays) != 1:
        raise RuntimeError(f"{site}: expected one SOIL_ARRAY, found {arrays}")
    array = get(f"{API}/locations/{arrays[0]}")

    rows = []
    for name in sorted(array["locationChildren"]):
        loc = get(f"{API}/locations/{name}")
        desc = loc["locationDescription"]
        tag = desc.rsplit("SP", 1)[-1].strip()
        if not tag.isdigit():
            raise RuntimeError(f"{site}: cannot read a plot number from '{desc}'")
        hor = int(tag)
        coords = loc["locationPolygon"]["coordinates"]
        # The polygon closes on its first corner; drop the repeat before averaging.
        uniq = coords[:-1] if coords[0] == coords[-1] else coords
        rows.append(
            {
                "station": station_id(site, hor),
                "station_name": desc,
                "network": NETWORK,
                "station_uid": f"{NETWORK}:{station_id(site, hor)}",
                "site": site,
                "plot": hor,
                "location_id": name,
                "longitude": sum(c["longitude"] for c in uniq) / len(uniq),
                "latitude": sum(c["latitude"] for c in uniq) / len(uniq),
                "elevation_m": sum(c["elevation"] for c in uniq) / len(uniq),
                "timezone": SITE_TZ[site],
                "site_latitude": site_loc["locationDecimalLatitude"],
                "site_longitude": site_loc["locationDecimalLongitude"],
            }
        )
    return pd.DataFrame(rows)


def frame_join(plots: pd.DataFrame, data_dir: Path, ref_dir: Path) -> pd.DataFrame:
    """Join plots to the NISAR SME2-producing frames, reusing station_nisar_frames."""
    frames = load_frames(data_dir)
    sf = join(plots[["station", "station_name", "longitude", "latitude"]], frames)
    combos = track_frames(sf)
    ref_dir.mkdir(parents=True, exist_ok=True)
    sf.to_csv(ref_dir / f"{PREFIX}_station_frames.csv", index=False)
    combos.to_csv(ref_dir / f"{PREFIX}_track_frames.csv", index=False)

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
            }
        )

    print(
        f"NISAR frame join: {sf['station'].nunique()}/{len(plots)} plots in SME2 coverage, "
        f"{len(combos)} unique track/frames"
    )
    return sf.groupby("station", as_index=False).apply(_agg, include_groups=False)


def summarize(thirty: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    g = thirty.groupby("station")
    out = g.agg(
        n_obs_30min=("vwc", "size"),
        obs_start=("datetime_utc", "min"),
        obs_end=("datetime_utc", "max"),
        n_depths=("depth_cm", "nunique"),
    )
    out["depths_cm"] = g["depth_cm"].agg(
        lambda s: ";".join(f"{d:g}" for d in sorted(set(s)))
    )
    out["n_qc_fail"] = g["vwc_qf"].agg(lambda s: int((s.fillna(1) != 0).sum()))
    # Kept, not filtered: an out-of-range reading is a QC question for the panel build,
    # not something to hide at pull time.
    out["n_out_of_range"] = g["vwc"].agg(lambda s: int(((s <= 0) | (s > 1.0)).sum()))
    out["n_days"] = daily.groupby("station")["date"].nunique()
    # The shallowest sensor differs by plot (5.0-6.5 cm at KONA, 6-7 cm at STER), so it is
    # taken per station rather than from a single global minimum.
    is_shallow = daily["depth_cm"] == daily.groupby("station")["depth_cm"].transform(
        "min"
    )
    shallow = daily[is_shallow]
    out["depth_cm_shallowest"] = shallow.groupby("station")["depth_cm"].min()
    out["n_days_shallowest"] = shallow.groupby("station")["date"].nunique()
    # NEON's final QF fires far more often on the deep sensors than on the shallow one,
    # and it is the shallow one NISAR is compared against, so it gets its own rate.
    top = thirty[
        thirty["depth_cm"] == thirty.groupby("station")["depth_cm"].transform("min")
    ]
    out["qc_fail_frac_shallowest"] = (
        top.assign(fail=top["vwc_qf"].fillna(1) != 0)
        .groupby("station")["fail"]
        .mean()
        .round(3)
    )
    return out.reset_index()


def estimate(
    sites: tuple[str, ...], months_by_site: dict[str, dict[str, str]], tok: str
) -> int:
    """Price a full pull from one probe month per site -- no bulk download."""
    print(f"--- {PRODUCT} pull estimate ({TABLE}, {PACKAGE} package) ---")
    total_bytes = total_files = total_months = 0
    for site in sites:
        months = months_by_site[site]
        span = list(months)
        probe = span[-1]
        wanted, data = month_files(site, probe, months[probe], tok)
        per_month = sum(f["size"] for f in wanted)
        full = sum(f["size"] for f in data["files"])
        print(
            f"{site}: {len(months)} months {span[0]}..{probe}; "
            f"probe {probe} -> {len(wanted)} files, {per_month / 1e6:.1f} MB "
            f"(full package would be {full / 1e6:.0f} MB)"
        )
        print(
            f"      projected: {len(months) * len(wanted):,} files, "
            f"{len(months) * per_month / 1e9:.2f} GB CSV downloaded"
        )
        total_bytes += len(months) * per_month
        total_files += len(months) * len(wanted)
        total_months += len(months)
    print(
        f"total: {total_months} site-months, {total_files:,} file GETs, "
        f"{total_bytes / 1e9:.2f} GB CSV; {total_months} API calls against a "
        "2000/hour token limit"
    )
    return 0


def build(
    data_dir: Path,
    sites: tuple[str, ...],
    start: str | None,
    end: str | None,
    do_estimate: bool,
    overwrite: bool,
    verify_depths: bool,
) -> int:
    tok = token()
    months_by_site = {s: site_month_releases(s, tok) for s in sites}
    for site, months in months_by_site.items():
        if start:
            months = {m: r for m, r in months.items() if m >= start}
        if end:
            months = {m: r for m, r in months.items() if m <= end}
        if not months:
            raise RuntimeError(f"{site}: no {PRODUCT} months in {start}..{end}")
        months_by_site[site] = months

    if do_estimate:
        return estimate(sites, months_by_site, tok)

    out_dir = data_dir / PREFIX
    monthly_dir = out_dir / "monthly"
    out_dir.mkdir(parents=True, exist_ok=True)

    depths = swc_depths()
    ladder = "; ".join(
        f"{s} {sorted(set(depths.loc[depths['site'] == s, 'depth_cm']))}" for s in sites
    )
    print(f"corrected depths (swc_depthsV3): {ladder}", flush=True)

    chunks, log = [], []
    for site in sites:
        months = months_by_site[site]
        span = list(months)
        print(f"\n{site}: {len(months)} month(s) {span[0]}..{span[-1]}", flush=True)
        for i, (month, release) in enumerate(months.items(), 1):
            df, rec = pull_month(
                site, month, release, depths, tok, monthly_dir, overwrite, verify_depths
            )
            chunks.append(df)
            log.append(rec)
            print(
                f"  [{i}/{len(months)}] {month} {release}: {len(df):,} rows"
                + (
                    " (cached)"
                    if rec["cached"]
                    else f", {rec['n_bytes_csv'] / 1e6:.1f} MB"
                ),
                flush=True,
            )

    thirty = pd.concat(chunks, ignore_index=True)
    thirty = thirty.sort_values(["station", "datetime_utc", "depth_cm"]).reset_index(
        drop=True
    )
    daily = daily_long(thirty)

    plots = pd.concat([soil_plots(s) for s in sites], ignore_index=True)
    plots = plots.sort_values(["site", "plot"]).reset_index(drop=True)
    plots = plots.merge(
        frame_join(plots, data_dir, data_dir / "reference"), on="station", how="left"
    )
    plots = plots.merge(summarize(thirty, daily), on="station", how="left")

    thirty_path = out_dir / f"{PREFIX}_swc_30min.parquet"
    daily_path = out_dir / f"{PREFIX}_daily_long.parquet"
    plots_csv = out_dir / f"{PREFIX}_soil_plots.csv"
    plots_fgb = out_dir / f"{PREFIX}_soil_plots.fgb"
    manifest_path = out_dir / f"{PREFIX}_pull_manifest.json"

    thirty.to_parquet(thirty_path, index=False)
    daily.to_parquet(daily_path, index=False)
    plots.to_csv(plots_csv, index=False)
    gpd.GeoDataFrame(
        plots,
        geometry=gpd.points_from_xy(plots["longitude"], plots["latitude"]),
        crs="EPSG:4326",
    ).to_file(plots_fgb, driver="FlatGeobuf", engine="fiona")
    manifest_path.write_text(
        json.dumps(
            {
                "network": NETWORK,
                "product": PRODUCT,
                "api": API,
                "table": TABLE,
                "package": PACKAGE,
                "depth_source": SWC_DEPTHS_DOC,
                "depth_note": (
                    "sensor_positions zOffset is wrong for DP1.00094.001 (NEON erratum); "
                    "depths taken from the corrected swc_depthsV3 document"
                ),
                "pulled_at": datetime.now(UTC).isoformat(),
                "sites": list(sites),
                "months": {s: list(months_by_site[s]) for s in sites},
                "releases": sorted(
                    {r for m in months_by_site.values() for r in m.values()}
                ),
                "site_months": log,
            },
            indent=2,
        )
    )

    n_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(f"\n--- NEON {PRODUCT} soil water content ---")
    print(f"sites:                  {', '.join(sites)}")
    print(f"soil plots:             {plots['station'].nunique()}")
    print(f"half-hourly obs:        {len(thirty):,}")
    n_fail = int((thirty["vwc_qf"].fillna(1) != 0).sum())
    print(f"  QC-failed (retained): {n_fail:,} ({n_fail / len(thirty):.1%})")
    print(
        f"  shallowest sensor:    "
        f"{plots['qc_fail_frac_shallowest'].min():.1%}-"
        f"{plots['qc_fail_frac_shallowest'].max():.1%} QC-failed by plot"
    )
    print(f"  null VWC:             {int(thirty['vwc'].isna().sum()):,}")
    print(f"daily station-day-depth:{len(daily):,}")
    print(
        f"datetime range (UTC):   {thirty['datetime_utc'].min()} .. {thirty['datetime_utc'].max()}"
    )
    print(f"depths present (cm):    {sorted(set(thirty['depth_cm']))}")
    print(f"\nwrote {thirty_path}")
    print(f"wrote {daily_path}")
    print(f"wrote {plots_csv}")
    print(f"wrote {plots_fgb}")
    print(f"wrote {manifest_path}")
    print(f"wrote {len(log)} monthly chunk(s) in {monthly_dir}")
    print(f"total output size:      {n_bytes / 1e6:.1f} MB")
    print(f"\nrsync -rav zoran:{out_dir}/ ~/data/nisar/{PREFIX}/")
    print(f"rsync -rav zoran:{data_dir}/reference/ ~/data/nisar/reference/")
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--sites", nargs="+", default=list(SITES), choices=list(SITES))
    p.add_argument("--start", help="first month, YYYY-MM (default: site's first)")
    p.add_argument("--end", help="last month, YYYY-MM (default: site's last)")
    p.add_argument(
        "--estimate",
        action="store_true",
        help="report projected file count and volume from one probe month; download nothing",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="re-download site-months already cached",
    )
    p.add_argument(
        "--no-verify-depths",
        action="store_true",
        help="skip the per-month check that NEON's depth erratum still holds",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args(sys.argv[1:])
    sys.exit(
        build(
            a.data_dir,
            tuple(a.sites),
            a.start,
            a.end,
            a.estimate,
            a.overwrite,
            not a.no_verify_depths,
        )
    )
