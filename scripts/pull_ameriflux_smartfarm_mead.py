"""Build the AmeriFlux soil-VWC record for the two in-field cropland cohorts named in
``notes/literature_network_leads_2026-08-25.md`` sections 1 and 5: the ARPA-E
SMARTFARM / Arva Intelligence towers on commercially farmed fields, and the Mead,
Nebraska irrigated/rainfed maize triad.

Why these two: a flux tower is sited *in* the field it characterizes, which is the
structural opposite of the mesonet grass pad that contaminated the earlier irrigated
cohort. The California rice sites are flood irrigated -- the largest, least ambiguous
surface-soil-water forcing in CONUS -- and Mead is the only co-located
irrigated-vs-rainfed pair of working maize fields in the country.

Two stages, because they have different access requirements.

``catalog`` (no credentials, runs today)
    Reads the public, unauthenticated AmeriFlux web services on ``amfcdn.lbl.gov``:
    the site map, the CC-BY-4.0 site list, and BASE data-year availability; plus the
    ``amerifluxr`` project's published BASE variable-availability summary, which is what
    tells us which sites actually report SWC and in which years. Writes a station table
    (CSV + FlatGeobuf) in the same shape as ``mt_mesonet_stations.csv`` so the NISAR
    track/frame join can be run against it exactly as it was for the Mesonet.

``download`` (needs an AmeriFlux account -- see below)
    BASE data itself is NOT served unauthenticated. There is no anonymous bulk URL: the
    download web service is a POST that requires a registered AmeriFlux ``user_id`` and
    the account's e-mail, and it returns per-site signed archive URLs. All thirteen sites
    here are **CC-BY-4.0**, which is the permissive AmeriFlux policy -- it requires
    citation of each site's data-product DOI and acknowledgement of DOE Office of Science
    funding, but unlike the AmeriFlux LEGACY policy it does *not* require contacting the
    site PI before publishing. The registration itself is still mandatory.

    To unblock this stage:
      1. Register at https://ameriflux-data.lbl.gov/Pages/RequestAccount.aspx
      2. Note the assigned username; no API token is issued -- the username and the
         account e-mail *are* the credential.
      3. Export them:
             export AMERIFLUX_USER_ID=<username>
             export AMERIFLUX_USER_EMAIL=<account email>
      4. The CC-BY-4.0 terms are acknowledged by passing ``--agree-policy``, which is the
         programmatic equivalent of the portal's click-through. Read the terms first:
         https://ameriflux.lbl.gov/data/data-policy/

    **The download stage is UNTESTED.** It was written against the documented request
    contract (the ``amerifluxr`` R client, which is AmeriFlux's own reference
    implementation) but has never been executed, because no AmeriFlux credential exists
    on this machine. Treat its first run as a pilot: it defaults to a single site.

Output schema for the download stage (long format, matching pull_mt_mesonet.py)::

    station, date, element, depth_cm, value, units

``station`` is the AmeriFlux SITE_ID (e.g. ``US-RGB``). ``element`` is the AmeriFlux
variable name (``SWC_1_1_1``, ``P_1_1_1``, ...) -- the name carries only a
horizontal/vertical/replicate index, NOT a physical depth, so ``depth_cm`` is resolved
from the BADM (BIF) file shipped alongside each BASE file, whose ``VAR_INFO_HEIGHT`` is
metres with below-ground negative. A variable whose depth cannot be resolved is kept with
a null ``depth_cm`` and reported, not silently dropped -- an unresolved depth is a
metadata gap worth seeing, and for SWC it decides whether the record is comparable to the
5 cm depth everything else in this project is scored at.

Units: BASE reports SWC in percent (0-100) and P in mm. SWC is divided by 100 to m3/m3 to
match the ISMN convention used elsewhere; P is left in mm. Precipitation is pulled
alongside soil moisture deliberately: the standing finding is that rain, not irrigation,
explains essentially all detected wetting in the current CONUS sample, so a tower rain
gauge in the same field is what makes an irrigation attribution at these sites testable
at all.

``date`` is the local-standard-time calendar day. AmeriFlux BASE timestamps are local
standard time with no daylight saving, so no timezone conversion is applied -- the
half-hourly/hourly record is aggregated straight to the day its TIMESTAMP_START falls in.

Usage:
    uv run python scripts/pull_ameriflux_smartfarm_mead.py catalog
    uv run python scripts/pull_ameriflux_smartfarm_mead.py download --sites US-RGB \\
        --agree-policy
    uv run python scripts/pull_ameriflux_smartfarm_mead.py download --all --agree-policy
"""

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

DATA_DIR = Path("/data/ssd2/nisar")
NETWORK = "AMERIFLUX"

API = "https://amfcdn.lbl.gov/api/v1"
SITEMAP_URL = f"{API}/site_display/AmeriFlux"
CCBY4_URL = f"{API}/site_availability/AmeriFlux/BIF/CCBY4.0"
DATA_YEAR_URL = f"{API}/data_availability/AmeriFlux/BASE-BADM/CCBY4.0"
DOWNLOAD_URL = f"{API}/data_download"
# The amerifluxr project publishes a periodically rebuilt summary of which BASE variables
# each site reports, by year. There is no equivalent web service, and this is the file
# AmeriFlux's own R client uses for the same question.
VAR_AVAIL_URL = (
    "https://raw.githubusercontent.com/chuhousen/amerifluxr/master/data-summary/"
    "AMF_AA-Flx_BASE_VARIABLE-AVAILABILITY_LATEST.csv"
)

DATA_PRODUCT = "BASE-BADM"
DATA_POLICY = "CCBY4.0"
INTENDED_USE = "Research - Remote sensing"
INTENDED_USE_TEXT = (
    "NISAR L3 SME2 (200 m L-band surface soil moisture) validation over in-field "
    "cropland, and irrigation-event detection skill at flood- and pivot-irrigated sites."
)

# Cohort definitions and management attributes. These are curated from
# notes/literature_network_leads_2026-08-25.md sections 1 and 5, not read from BADM --
# AmeriFlux BADM has no irrigation flag, and irrigation status is the whole reason these
# sites were selected, so it is recorded explicitly rather than inferred.
SITES = {
    "US-RGA": ("SMARTFARM", "corn-soy", "irrigated", None),
    "US-RGW": ("SMARTFARM", "rice", "flood", None),
    "US-RGB": ("SMARTFARM", "rice", "flood", 30.0),
    "US-RGo": ("SMARTFARM", "organic rice", "flood", 7.0),
    "US-RGF": ("SMARTFARM", "corn-wheat silage", "irrigated", 31.3),
    "US-AV1": ("SMARTFARM", "row crop", "unknown", None),
    "US-AV2": ("SMARTFARM", "row crop", "unknown", None),
    "US-AV3": ("SMARTFARM", "row crop", "unknown", None),
    "US-AV4": ("SMARTFARM", "row crop", "unknown", None),
    "US-AV5": ("SMARTFARM", "row crop", "unknown", None),
    "US-Ne1": ("MEAD", "continuous maize", "irrigated", None),
    "US-Ne2": ("MEAD", "maize-soy rotation", "irrigated", None),
    "US-Ne3": ("MEAD", "maize-soy rotation", "rainfed", None),
}

# BASE basenames to keep. SWC is the target; P is the rain gauge that makes an
# irrigation-vs-rain attribution possible at the same tower.
BASENAMES = ("SWC", "P")
PERCENT_TO_FRACTION = {"SWC": 100.0}
UNITS = {"SWC": "m3/m3", "P": "mm"}
# Gap-filled and PI-provided aggregate variants (SWC_PI_F_1, SWC_PI_1_SD, ...) are
# excluded: they are the PI's own processing, not the measured half-hourly record, and
# mixing them with SWC_1_1_1 would double-count depths.
EXCLUDE_TOKENS = ("_PI_",)

LONG_COLUMNS = ["station", "date", "element", "depth_cm", "value", "units"]
BASE_FILL = -9999
BASE_HEADER_ROWS = 2
TIMEOUT = (10, 300)


def _get_json(url: str):
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def site_catalog() -> pd.DataFrame:
    """Site metadata + data policy + published BASE years, from the public services."""
    sitemap = {s["SITE_ID"]: s for s in _get_json(SITEMAP_URL)}
    ccby4 = {row[0] for row in _get_json(CCBY4_URL)}
    years = {d["SITE_ID"]: d.get("publish_years", []) for d in _get_json(DATA_YEAR_URL)}

    missing = sorted(set(SITES) - set(sitemap))
    if missing:
        raise RuntimeError(f"site IDs not in the AmeriFlux site map: {missing}")

    rows = []
    for site_id, (cohort, crop, irrigation, field_ha) in SITES.items():
        s = sitemap[site_id]
        loc = s["GRP_LOCATION"]
        clim = s["GRP_CLIM_AVG"]
        published = sorted(years.get(site_id, []))
        rows.append(
            {
                "station": site_id,
                "network": NETWORK,
                "station_uid": f"{NETWORK}:{site_id}",
                "site_name": s["SITE_NAME"],
                "cohort": cohort,
                "crop": crop,
                "irrigation": irrigation,
                "field_ha": field_ha,
                "state": s.get("STATE"),
                "igbp": s.get("IGBP"),
                "longitude": float(loc["LOCATION_LONG"]),
                "latitude": float(loc["LOCATION_LAT"]),
                "elevation_m": loc.get("LOCATION_ELEV"),
                "tower_began": s.get("TOWER_BEGAN"),
                "tower_end": s.get("TOWER_END"),
                "data_policy": "CCBY4.0" if site_id in ccby4 else "LEGACY",
                "climate_koeppen": clim.get("CLIMATE_KOEPPEN"),
                "mat_c": clim.get("MAT"),
                "map_mm": clim.get("MAP"),
                "base_years": ";".join(str(y) for y in published),
                "base_year_start": published[0] if published else None,
                "base_year_end": published[-1] if published else None,
                "url": s.get("URL_AMERIFLUX"),
            }
        )
    return pd.DataFrame(rows)


def keep_variable(variable: str, basename: str) -> bool:
    return basename in BASENAMES and not any(t in variable for t in EXCLUDE_TOKENS)


def swc_availability() -> pd.DataFrame:
    """Which of our sites report which BASE variables, and over which years.

    Answers the "does this tower actually measure soil moisture" question before any
    download is attempted -- a flux tower is not obliged to report SWC.
    """
    raw = requests.get(VAR_AVAIL_URL, timeout=TIMEOUT)
    raw.raise_for_status()
    text = raw.content.decode("utf-8")
    # First line is a bare "create_date:YYYY-MM-DD" stamp ahead of the real header.
    stamp, _, body = text.partition("\n")
    df = pd.read_csv(io.StringIO(body))
    df = df[df["SITE_ID"].isin(SITES)]
    df = df[
        [
            keep_variable(v, b)
            for v, b in zip(df["VARIABLE"], df["BASENAME"], strict=True)
        ]
    ]
    year_cols = [c for c in df.columns if c.startswith("Y") and c[1:].isdigit()]

    rows = []
    for site_id in SITES:
        sub = df[df["SITE_ID"] == site_id]
        rec = {"station": site_id, "var_summary_stamp": stamp.strip()}
        for basename in BASENAMES:
            b = sub[sub["BASENAME"] == basename]
            present = sorted(b["VARIABLE"].unique())
            covered = [int(c[1:]) for c in year_cols if b[c].fillna(0).sum() > 0]
            key = basename.lower()
            rec[f"n_{key}_vars"] = len(present)
            rec[f"{key}_vars"] = ";".join(present)
            rec[f"{key}_year_start"] = covered[0] if covered else None
            rec[f"{key}_year_end"] = covered[-1] if covered else None
        rows.append(rec)
    return pd.DataFrame(rows)


def write_catalog(data_dir: Path) -> Path:
    out_dir = data_dir / "ameriflux"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = site_catalog().merge(swc_availability(), on="station", how="left")

    csv_path = out_dir / "ameriflux_sites.csv"
    fgb_path = out_dir / "ameriflux_sites.fgb"
    table.to_csv(csv_path, index=False)
    gdf = gpd.GeoDataFrame(
        table,
        geometry=gpd.points_from_xy(table["longitude"], table["latitude"]),
        crs="EPSG:4326",
    )
    gdf.to_file(fgb_path, driver="FlatGeobuf", engine="fiona")

    print(f"--- AmeriFlux SMARTFARM + Mead catalog ({len(table)} sites) ---")
    for _, r in table.iterrows():
        swc = r["n_swc_vars"] or 0
        span = (
            f"{int(r['swc_year_start'])}-{int(r['swc_year_end'])}"
            if swc
            else "no SWC in BASE"
        )
        print(
            f"  {r['station']} | {r['cohort']:9} | {r['irrigation']:9} | "
            f"{r['data_policy']:8} | SWC vars {int(swc):2} | {span}"
        )
    no_base = table[table["base_years"] == ""]["station"].tolist()
    no_swc = table[(table["n_swc_vars"].fillna(0) == 0)]["station"].tolist()
    legacy = table[table["data_policy"] != "CCBY4.0"]["station"].tolist()
    print(f"\nsites with no published BASE data: {no_base or 'none'}")
    print(f"sites with no SWC in BASE:         {no_swc or 'none'}")
    print(f"sites NOT under CC-BY-4.0:         {legacy or 'none'}")
    print(f"\nwrote {csv_path}")
    print(f"wrote {fgb_path}")
    print(f"\nrsync -rav zoran:{out_dir}/ ~/data/nisar/ameriflux/")
    return csv_path


def credentials() -> tuple[str, str]:
    user_id = os.environ.get("AMERIFLUX_USER_ID")
    user_email = os.environ.get("AMERIFLUX_USER_EMAIL")
    if not user_id or not user_email:
        raise SystemExit(
            "AmeriFlux BASE downloads require a registered account; there is no "
            "anonymous bulk-download route.\n"
            "  1. Register: https://ameriflux-data.lbl.gov/Pages/RequestAccount.aspx\n"
            "  2. export AMERIFLUX_USER_ID=<username>\n"
            "     export AMERIFLUX_USER_EMAIL=<account email>\n"
            "No API token is issued -- the username plus account e-mail is the "
            "credential the download service checks."
        )
    return user_id, user_email


def request_download_urls(sites: list[str], user_id: str, user_email: str) -> list[str]:
    payload = {
        "user_id": user_id,
        "user_email": user_email,
        "data_product": DATA_PRODUCT,
        "data_policy": DATA_POLICY,
        "site_ids": sites,
        "intended_use": INTENDED_USE,
        "description": INTENDED_USE_TEXT,
        "is_test": "",
    }
    r = requests.post(
        DOWNLOAD_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"download service returned HTTP {r.status_code}: {r.text[:400]}"
        )
    urls = [d["url"] for d in r.json().get("data_urls", [])]
    if not urls:
        raise RuntimeError(f"download service returned no URLs for {sites}")
    # The service has historically handed back ftp:// URLs on a host that also serves
    # https. Outbound FTP is commonly blocked; https on the same path is not.
    return [u.replace("ftp://", "https://", 1) for u in urls]


def fetch_zip(url: str, zip_dir: Path) -> Path:
    name = url.split("/")[-1].split("?")[0]
    path = zip_dir / name
    if path.exists():
        print(f"  {name}: already present, skipping fetch")
        return path
    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with path.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    print(f"  {name}: {path.stat().st_size / 1e6:.1f} MB")
    return path


def badm_depths(zf: zipfile.ZipFile) -> dict[str, float]:
    """variable name -> depth in cm, from the BADM GRP_VAR_INFO group.

    VAR_INFO_HEIGHT is metres, below ground negative, so depth_cm is its magnitude in
    centimetres. Variables measured above ground (positive height, e.g. the rain gauge)
    get no depth.
    """
    names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
    if not names:
        return {}
    with zf.open(names[0]) as fh:
        bif = pd.read_excel(io.BytesIO(fh.read()), engine="openpyxl")
    need = {"GROUP_ID", "VARIABLE", "DATAVALUE"}
    if not need.issubset(bif.columns):
        raise RuntimeError(
            f"unexpected BADM columns in {names[0]}: {list(bif.columns)}"
        )

    depths: dict[str, float] = {}
    for _, grp in bif.groupby("GROUP_ID"):
        kv = dict(zip(grp["VARIABLE"], grp["DATAVALUE"], strict=True))
        varname = kv.get("VAR_INFO_VARNAME")
        height = kv.get("VAR_INFO_HEIGHT")
        if varname is None or height is None:
            continue
        h = float(height)
        if h < 0:
            # A variable can be re-registered across BADM date entries; depths agree, so
            # last write wins.
            depths[str(varname)] = round(-h * 100.0, 1)
    return depths


def base_to_long(zf: zipfile.ZipFile, site_id: str) -> pd.DataFrame:
    names = [n for n in zf.namelist() if "_BASE_" in n and n.lower().endswith(".csv")]
    if len(names) != 1:
        raise RuntimeError(
            f"expected one BASE csv in the {site_id} archive, got {names}"
        )
    with zf.open(names[0]) as fh:
        wide = pd.read_csv(
            io.BytesIO(fh.read()),
            skiprows=BASE_HEADER_ROWS,
            na_values=[BASE_FILL, str(BASE_FILL)],
        )

    keep = [
        c
        for c in wide.columns
        if not c.startswith("TIMESTAMP") and keep_variable(c, c.split("_")[0])
    ]
    if not keep:
        return pd.DataFrame(columns=LONG_COLUMNS)

    # BASE timestamps are local standard time (no DST); YYYYMMDDHHMM.
    stamp = pd.to_datetime(
        wide["TIMESTAMP_START"].astype("int64").astype(str), format="%Y%m%d%H%M"
    )
    long = (
        wide[keep]
        .assign(date=stamp.dt.normalize())
        .melt(id_vars="date", var_name="element", value_name="value")
    )
    # A null here is a gap in the half-hourly record, not a value to patch.
    long = long.dropna(subset=["value"])
    if long.empty:
        return pd.DataFrame(columns=LONG_COLUMNS)

    long["basename"] = long["element"].str.split("_").str[0]
    # Daily aggregate depends on what the variable is: a state variable (SWC) averages
    # over the day, an accumulating flux (P) sums.
    keys = ["date", "element", "basename"]
    parts = []
    state = long[long["basename"] != "P"]
    if not state.empty:
        parts.append(state.groupby(keys, as_index=False)["value"].mean())
    rain = long[long["basename"] == "P"]
    if not rain.empty:
        parts.append(rain.groupby(keys, as_index=False)["value"].sum())
    daily = pd.concat(parts, ignore_index=True)

    depths = badm_depths(zf)
    daily["station"] = site_id
    daily["depth_cm"] = daily["element"].map(depths)
    daily["units"] = daily["basename"].map(UNITS)
    scale = daily["basename"].map(PERCENT_TO_FRACTION).fillna(1.0)
    daily["value"] = daily["value"] / scale

    unresolved = sorted(
        daily.loc[
            daily["depth_cm"].isna() & (daily["basename"] == "SWC"), "element"
        ].unique()
    )
    if unresolved:
        print(
            f"  {site_id}: no BADM depth for SWC variable(s) {unresolved} -- "
            f"retained with null depth_cm"
        )
    return (
        daily[LONG_COLUMNS]
        .sort_values(["station", "date", "element"])
        .reset_index(drop=True)
    )


def summarize(combined: pd.DataFrame) -> pd.DataFrame:
    swc = combined[combined["element"].str.startswith("SWC")]
    g = combined.groupby("station")
    out = g.agg(
        n_obs=("value", "size"),
        n_days=("date", "nunique"),
        obs_start=("date", "min"),
        obs_end=("date", "max"),
    )
    out["n_obs_swc"] = swc.groupby("station")["value"].size()
    out["swc_depths_cm"] = swc.groupby("station")["depth_cm"].agg(
        lambda s: ";".join(str(d) for d in sorted(set(s.dropna())))
    )
    # 5 cm is the depth NISAR L3 SME2 is scored against everywhere else in this project;
    # nothing in these two cohorts is guaranteed to have it.
    shallow = swc[swc["depth_cm"] <= 10.0]
    out["n_obs_swc_le10cm"] = shallow.groupby("station")["value"].size()
    return out.fillna(0).reset_index()


def run_download(data_dir: Path, sites: list[str], agree_policy: bool) -> int:
    if not agree_policy:
        raise SystemExit(
            "Refusing to download without --agree-policy.\n"
            "All sites here are CC-BY-4.0 (https://ameriflux.lbl.gov/data/data-policy/):\n"
            "  (1) free to share and adapt for any purpose;\n"
            "  (2) cite each site's data-product DOI and/or recommended publication;\n"
            "  (3) acknowledge DOE Office of Science support for the AmeriFlux data "
            "portal.\n"
            "Unlike the AmeriFlux LEGACY policy, CC-BY-4.0 does not require contacting "
            "the site PI before publication."
        )
    user_id, user_email = credentials()

    out_dir = data_dir / "ameriflux"
    zip_dir = out_dir / "zip"
    daily_dir = out_dir / "daily"
    zip_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"requesting {DATA_PRODUCT} / {DATA_POLICY} for {len(sites)} site(s): {sites}"
    )
    urls = request_download_urls(sites, user_id, user_email)
    print(f"service returned {len(urls)} archive URL(s)")

    frames = []
    for url in urls:
        path = fetch_zip(url, zip_dir)
        site_id = path.name.split("_")[1]
        with zipfile.ZipFile(path) as zf:
            long = base_to_long(zf, site_id)
        if long.empty:
            print(f"  {site_id}: no SWC or P observations in BASE")
            continue
        long.to_parquet(daily_dir / f"ameriflux_{site_id}_daily.parquet", index=False)
        frames.append(long)

    if not frames:
        raise RuntimeError("no observations recovered from any archive")
    combined = pd.concat(frames, ignore_index=True)
    combined_path = out_dir / "ameriflux_daily_long.parquet"
    combined.to_parquet(combined_path, index=False)

    summary = summarize(combined)
    summary_path = out_dir / "ameriflux_daily_summary.csv"
    summary.to_csv(summary_path, index=False)
    (out_dir / "ameriflux_download_manifest.json").write_text(
        json.dumps(
            {
                "network": NETWORK,
                "api": DOWNLOAD_URL,
                "data_product": DATA_PRODUCT,
                "data_policy": DATA_POLICY,
                "sites": sites,
                "archives": [u.split("/")[-1].split("?")[0] for u in urls],
                "basenames": list(BASENAMES),
                "units": UNITS,
                "date_basis": "local standard time calendar day (BASE convention)",
            },
            indent=2,
        )
    )

    print("\n--- AmeriFlux daily long record ---")
    print(summary.to_string(index=False))
    n_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(f"\nwrote {combined_path} ({len(combined):,} rows)")
    print(f"wrote {summary_path}")
    print(f"total output size: {n_bytes / 1e6:.1f} MB")
    print(f"\nrsync -rav zoran:{out_dir}/ ~/data/nisar/ameriflux/")
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=["catalog", "download"])
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument(
        "--sites",
        nargs="+",
        default=["US-RGB"],
        help="site IDs for the download stage (default: US-RGB, the pilot site)",
    )
    p.add_argument(
        "--all", action="store_true", help="download every site in the two cohorts"
    )
    p.add_argument(
        "--agree-policy",
        action="store_true",
        help="acknowledge the AmeriFlux CC-BY-4.0 data-use terms (required to download)",
    )
    return p.parse_args(argv)


def main(argv) -> int:
    args = parse_args(argv)
    if args.stage == "catalog":
        write_catalog(args.data_dir)
        return 0
    sites = sorted(SITES) if args.all else args.sites
    unknown = sorted(set(sites) - set(SITES))
    if unknown:
        raise SystemExit(f"not part of these cohorts: {unknown}")
    return run_download(args.data_dir, sites, args.agree_policy)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
