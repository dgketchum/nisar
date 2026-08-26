"""Consolidate the four new USACE Upper Missouri mesonets into one NISAR validation panel.

Montana is deliberately not in here. It already has its own tables
(``mt_mesonet_*``) and its own pull; it appears only as the baseline the net-new
track/frame count is differenced against.

Scope
-----
Station *metadata* only. This builds the panel -- who the stations are, where they are
to what precision, which carry 5 cm VWC, whether the NISAR footprint over them is
cropland, and which NISAR track/frames cover them. It deliberately does not fetch a
single soil-moisture observation, because as of this writing none of the four new
networks has an open bulk route to one:

============  ==========================================================================
NDAWN (ND)    No public historical VWC endpoint at all. ``table.csv`` serves the met
              variables but every soil-moisture variable code returns the export-error
              page; ``/soil-moisture.html`` is current-hour only. On top of that,
              ``robots.txt`` sets ``Disallow: /*.csv`` for all agents and separately
              blocklists Claude-Code by name -- so an automated puller is not merely
              unimplemented here, it is not permitted. Bulk history is an email request
              to NDAWN.
SD Mesonet    A per-station monthly archive JSON (``archive_json_url`` in the raw table)
              does work unauthenticated, and is the only endpoint of the four that
              returns real history. It is still not a route: the Terms of Use say
              "automated collection ('scraping') or redistribution of Mesonet data is
              prohibited." Bulk history is a written agreement with SDSU -- which also
              gets the 5-minute data the web tools never expose.
Nebraska      ``/api/v1/data/historical`` exists but returns HTTP 500 on every parameter
              combination; ``robots.txt`` disallows ``/api/``; the Terms of Use forbid
              scraping and redistribution outright. Bulk history is a reCAPTCHA-gated
              request form plus, in practice, a redistribution agreement.
Wyoming       ``wrds.uwyo.edu/Mesonet/data/`` returns 401 Basic-auth. The unauthenticated
              path is a rolling 24 h HTML table. Bulk history is archive credentials or
              a Synoptic token.
============  ==========================================================================

So the raw station tables are inputs here, not outputs: they were assembled from each
network's public metadata surfaces during discovery and are carried under
``<data-dir>/mesonet/raw/``. Re-running this script re-derives the panel from them
without touching the networks.

What it produces
----------------
``<data-dir>/mesonet/umrb_station_panel.csv`` / ``.fgb``
    One row per 5 cm-VWC station across the four networks, normalized to a common
    schema, with the siting verdict and the NISAR frame assignment joined on.
``<data-dir>/reference/<network>_station_frames.csv`` / ``_track_frames.csv``
    Per network, in the schema ``pull_mesonet_frames_sme2.py`` already reads.
``<data-dir>/reference/umrb_track_frames.csv``
    The deduplicated union across the verified cohort, minus the track/frames the
    Montana pull already covered -- i.e. exactly what a follow-up SME2 stream would
    cost. **Nothing streams it. That is a separate, approved step.**

The siting verdict comes from ``screen_station_siting.py`` and must be run first; this
script joins its output rather than recomputing it, so the Earth Engine pass happens
once.

Usage:
    uv run python scripts/build_umrb_station_panel.py
    uv run python scripts/build_umrb_station_panel.py --data-dir /data/ssd2/nisar
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from station_nisar_frames import join, load_frames, track_frames

DATA_DIR = Path("/data/ssd2/nisar")

# Per-network normalization. Each entry says how to get from that network's raw table to
# the common schema: the id column, the prefix that makes station ids unique across
# networks, the columns that carry name / coordinates / VWC flag / coordinate precision,
# and the value of the VWC flag that means "has a 5 cm probe".
#
# ``id_prefix`` is carried explicitly rather than reusing the dict key, because the
# siting-screen outputs are keyed on it and re-keying them would mean re-running the
# Earth Engine pass. Whatever it is, it has to stay stable.
NETWORKS = {
    "ndawn": {
        "id_prefix": "ndawn",
        "label": "NDAWN (North Dakota Agricultural Weather Network)",
        "operator": "NDSU",
        "states": "ND/MN/MT",
        "raw": "ndawn_stations_raw.csv",
        "id": "station_id",
        "name": "station_name",
        "vwc_flag": ("shallow_5cm_vwc", "Y"),
        "depths": "vwc_depths_cm",
        "decimals": ("lat_decimals", "lon_decimals"),
        "extra": {"ag_district": "district", "sponsors": "sponsors"},
        "photos": "site_photos_url",
        "access": "email request; robots.txt blocks automated CSV",
    },
    "sdmesonet": {
        "id_prefix": "sd",
        "label": "South Dakota Mesonet",
        "operator": "SDSU",
        "states": "SD",
        "raw": "sdmesonet_stations_raw.csv",
        "id": "station_id",
        "name": "station_name",
        "vwc_flag": ("has_5cm_vwc", "Y"),
        "depths": "vwc_depths_cm",
        "decimals": ("lat_decimals", "lon_decimals"),
        "extra": {
            "county": "county",
            "soil_probe_model": "probe",
            "land_cover_published": "documented_land_cover",
        },
        "photos": None,
        "access": "archive JSON works but ToU forbids scraping",
    },
    "nemesonet": {
        "id_prefix": "ne",
        "label": "Nebraska Mesonet",
        "operator": "Nebraska State Climate Office, UNL",
        "states": "NE",
        "raw": "nemesonet_stations_raw.csv",
        "id": "station_id",
        "name": "station_name",
        "vwc_flag": ("has_vwc_2in_5cm", "yes"),
        "depths": "soil_sensor_depths_cm",
        "decimals": ("lat_effective_decimals", "lon_effective_decimals"),
        "extra": {
            "county": "county",
            "usace_iija_station": "usace_iija",
            "network_documented_land_cover": "documented_land_cover",
        },
        "photos": None,
        "access": "request form; ToU forbids scraping/redistribution",
    },
    "wy": {
        "id_prefix": "wy",
        "label": "Wyoming Mesonet (UMRB + WACNet)",
        "operator": "WRDS, University of Wyoming",
        "states": "WY",
        "raw": "wy_stations_raw.csv",
        "id": "station_id",
        "name": "station_name",
        "vwc_flag": ("has_soil_vwc", "yes"),
        "depths": "vwc_depths_cm",
        "decimals": ("coord_decimals", None),
        "extra": {"county": "county", "network": "subnetwork"},
        "photos": "site_photos",
        "access": "401 archive; Synoptic token or credentials",
    },
}

PANEL_COLUMNS = [
    "station",
    "network",
    "network_label",
    "station_id",
    "station_name",
    "longitude",
    "latitude",
    "coord_decimals",
    "coord_unc_m",
    "vwc_depths_cm",
    "field_verified",
    "verdict_reason",
    "cult_f_pt",
    "cult_f_100",
    "cult_f_500",
    "irr_f_100",
    "cdl_mode",
    "cdl_mode_frac",
    "f_smallgrain",
    "f_fallow",
    "f_rowcrop",
    "f_hay",
    "f_grass",
    "n_frames",
    "track_frames",
    "pass_directions",
    "bulk_access",
    "site_photos",
]


def normalize(prefix: str, spec: dict, raw_dir: Path) -> pd.DataFrame:
    """One network's raw table -> the common schema, VWC stations only."""
    df = pd.read_csv(raw_dir / spec["raw"])
    flag_col, flag_val = spec["vwc_flag"]
    df = df[df[flag_col].astype(str).str.lower() == str(flag_val).lower()].copy()

    lat_dec, lon_dec = spec["decimals"]
    decimals = df[lat_dec] if lon_dec is None else df[[lat_dec, lon_dec]].min(axis=1)

    out = pd.DataFrame(
        {
            "station": spec["id_prefix"] + "_" + df[spec["id"]].astype(str),
            "network": prefix,
            "network_label": spec["label"],
            "station_id": df[spec["id"]].astype(str),
            "station_name": df[spec["name"]],
            "longitude": df["longitude"],
            "latitude": df["latitude"],
            "coord_decimals": decimals,
            "vwc_depths_cm": df[spec["depths"]],
            "bulk_access": spec["access"],
            "site_photos": df[spec["photos"]] if spec["photos"] else None,
        }
    )
    for src, dst in spec["extra"].items():
        out[dst] = df[src]

    # A station with no coordinate cannot be placed in a frame and cannot be screened.
    # It is a metadata gap in the source network, so it is surfaced, not dropped quietly.
    missing = out["longitude"].isna() | out["latitude"].isna()
    if missing.any():
        print(
            f"  {prefix}: {int(missing.sum())} VWC stations have no coordinate, excluded"
        )
        out = out[~missing]
    return out.reset_index(drop=True)


def frame_summary(sf: pd.DataFrame) -> pd.DataFrame:
    """Collapse the station x frame join to one row per station, matching the shape
    ``pull_mt_mesonet.py`` builds for Montana."""

    def _agg(g):
        pairs = sorted(
            {f"{t}_{f}" for t, f in zip(g["track"], g["frame"], strict=True)}
        )
        return pd.Series(
            {
                "n_frames": len(pairs),
                "track_frames": ";".join(pairs),
                "pass_directions": ";".join(sorted(set(g["passDirection"]))),
            }
        )

    return sf.groupby("station", as_index=False).apply(_agg, include_groups=False)


def build(data_dir: Path, cohort: str = "pass") -> int:
    raw_dir = data_dir / "mesonet" / "raw"
    mesonet_dir = data_dir / "mesonet"
    ref_dir = data_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    frames = load_frames(data_dir)
    mt_done = pd.read_csv(ref_dir / "mt_mesonet_track_frames.csv")
    done = set(map(tuple, mt_done[["track", "frame", "pass_code"]].values))

    panels, rows = [], []
    verified_keys: set = set()

    for prefix, spec in NETWORKS.items():
        stations = normalize(prefix, spec, raw_dir)
        screen_path = mesonet_dir / f"{prefix}_siting_screen.csv"
        if not screen_path.exists():
            raise RuntimeError(
                f"{screen_path} not found -- run screen_station_siting.py for "
                f"'{prefix}' before building the panel"
            )
        screen = pd.read_csv(screen_path)
        keep = [
            c for c in screen.columns if c not in stations.columns or c == "station"
        ]
        merged = stations.merge(screen[keep], on="station", how="left")
        if merged["field_verified"].isna().any():
            n = int(merged["field_verified"].isna().sum())
            raise RuntimeError(
                f"{prefix}: {n} stations have no siting verdict -- the screen was run "
                f"on a different station set; re-run it on this panel"
            )

        sf = join(
            merged[["station", "station_name", "longitude", "latitude"]].copy(), frames
        )
        merged = merged.merge(frame_summary(sf), on="station", how="left")

        sf.to_csv(ref_dir / f"{prefix}_station_frames.csv", index=False)
        track_frames(sf).to_csv(ref_dir / f"{prefix}_track_frames.csv", index=False)

        sel = merged[merged["field_verified"] == cohort]
        if not sel.empty:
            keys = track_frames(sf[sf["station"].isin(sel["station"])])
            verified_keys |= set(
                map(tuple, keys[["track", "frame", "pass_code"]].values)
            )

        counts = merged["field_verified"].value_counts()
        rows.append(
            {
                "network": prefix,
                "label": spec["label"],
                "states": spec["states"],
                "vwc_stations": len(merged),
                "pass": int(counts.get("pass", 0)),
                "ambiguous": int(counts.get("ambiguous", 0)),
                "fail": int(counts.get("fail", 0)),
                "in_nisar_coverage": int(merged["n_frames"].notna().sum()),
                "track_frames": len(track_frames(sf)),
                "bulk_access": spec["access"],
            }
        )
        panels.append(merged)

    panel = pd.concat(panels, ignore_index=True)
    for col in PANEL_COLUMNS:
        if col not in panel.columns:
            panel[col] = None
    ordered = PANEL_COLUMNS + [c for c in panel.columns if c not in PANEL_COLUMNS]
    panel = panel[ordered]

    csv_path = mesonet_dir / "umrb_station_panel.csv"
    fgb_path = mesonet_dir / "umrb_station_panel.fgb"
    panel.to_csv(csv_path, index=False)
    gpd.GeoDataFrame(
        panel,
        geometry=gpd.points_from_xy(panel["longitude"], panel["latitude"]),
        crs="EPSG:4326",
    ).to_file(fgb_path, driver="FlatGeobuf", engine="fiona")

    net_new = sorted(verified_keys - done)
    combos = pd.DataFrame(net_new, columns=["track", "frame", "pass_code"])
    combos["passDirection"] = combos["pass_code"].map(
        {"A": "Ascending", "D": "Descending"}
    )
    combos = combos[["track", "frame", "passDirection", "pass_code"]]
    combos_path = ref_dir / "umrb_track_frames.csv"
    combos.to_csv(combos_path, index=False)

    summary = pd.DataFrame(rows)
    summary_path = mesonet_dir / "umrb_network_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n--- USACE Upper Missouri mesonet panel ---")
    print(summary.drop(columns=["bulk_access"]).to_string(index=False))
    print(f"\ncohort used for the pull estimate: field_verified == '{cohort}'")
    print(f"track/frames over that cohort:     {len(verified_keys)}")
    print(f"already covered by Montana pull:   {len(verified_keys & done)}")
    print(f"NET NEW track/frames to stream:    {len(net_new)}")
    print("\nNo SME2 granules were streamed. That is a separate, approved step.")
    print(f"\nwrote {csv_path}")
    print(f"wrote {fgb_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {combos_path}")
    print(f"\nrsync -rav zoran:{mesonet_dir}/ ~/data/nisar/mesonet/")
    print(f"rsync -rav zoran:{ref_dir}/ ~/data/nisar/reference/")
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument(
        "--cohort",
        default="pass",
        choices=["pass", "ambiguous", "fail"],
        help="siting verdict whose stations size the pull estimate",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args(sys.argv[1:])
    sys.exit(build(a.data_dir, cohort=a.cohort))
