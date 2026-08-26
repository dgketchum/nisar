"""Point-extract SMAP L3 (SPL3SMP_E, 9 km enhanced, AM/descending) soil moisture at
every Montana Mesonet station, one granule per day over a date range.

Streams only the Montana-cropped row/col window of the AM soil_moisture /
retrieval_qual_flag arrays -- the window is computed once by projecting all station
coordinates into the fixed EASE2_G9km grid definition (see pull_smap_ancillary.py for
why the file's own latitude/longitude datasets can't be used for this). One remote
read per day covers all stations; station values are indexed from their local offset
within the cropped window.

Output is stats-only: a tidy parquet of (station, date, smap_sm), no per-day GeoTIFFs.
Re-running skips dates already present in the parquet.

With --h5-dir, granules are read from a local archive (the swap-stress NAS mirror
holds the full period of record at /nas/soils/smap/SPL3SMP_E/hdf5) instead of
streamed -- same window logic, same recommended-quality screening, no network.

Usage:
    uv run python scripts/pull_smap_mesonet.py /data/ssd2/nisar/
    uv run python scripts/pull_smap_mesonet.py /data/ssd2/nisar/ 2025-01-01 2026-08-25
    uv run python scripts/pull_smap_mesonet.py /data/ssd2/nisar/ 2015-04-01 2026-08-26 \
        --h5-dir /nas/soils/smap/SPL3SMP_E/hdf5
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import earthaccess
import h5py
import pandas as pd

# Grid constants and station indexing live in ease2_grid.py (shared with the EE
# pixel land-cover sampler and the grid fgb builder); re-exported here for callers.
from ease2_grid import (  # noqa: F401 - re-exports
    GRID_NCOLS,
    GRID_NROWS,
    GRID_X_MIN,
    GRID_Y_MAX,
    PIXEL_SIZE,
    bounding_window,
    station_grid_cells,
)

SHORT_NAME = "SPL3SMP_E"
GROUP = "Soil_Moisture_Retrieval_Data_AM"

FILL = -9999.0
DEFAULT_START = "2025-01-01"
PROGRESS_EVERY = 25
FLUSH_EVERY = 25
MAX_CONSECUTIVE_FAILURES = 20


def granule_date(granule) -> str:
    """YYYYMMDD from the native id, e.g. SMAP_L3_SM_P_E_20250101_R19240_001.h5."""
    if isinstance(granule, Path):
        return granule.name.split("_")[5]
    return granule["meta"]["native-id"].split("_")[5]


def find_daily_granules(start: str, end: str) -> dict:
    """{YYYYMMDD: granule} for the range, one search rather than one per day."""
    n_days = (
        datetime.strptime(end, "%Y-%m-%d")  # noqa: DTZ007 - calendar date only, no tz
        - datetime.strptime(start, "%Y-%m-%d")  # noqa: DTZ007 - calendar date only, no tz
    ).days + 1
    granules = earthaccess.search_data(
        short_name=SHORT_NAME, temporal=(start, end), count=n_days + 100
    )
    by_date = {}
    for g in granules:
        # A reprocessed date can appear more than once; last one wins (highest version).
        by_date[granule_date(g)] = g
    return by_date


def local_daily_granules(h5_dir: Path) -> dict:
    """{YYYYMMDD: Path} from a local granule archive (e.g. the swap-stress NAS
    mirror at /nas/soils/smap/SPL3SMP_E/hdf5) -- no earthaccess, no streaming.
    Sorted so a reprocessed date resolves to the highest version, matching
    find_daily_granules."""
    paths = sorted(h5_dir.glob("SMAP_L3_SM_P_E_*.h5"))
    if not paths:
        raise FileNotFoundError(f"no SMAP_L3_SM_P_E_*.h5 granules under {h5_dir}")
    return {granule_date(p): p for p in paths}


def extract_day(granule, cells: pd.DataFrame, window: tuple[slice, slice]) -> list:
    """Read the cropped AM window once and pull each station's recommended-quality value."""
    row_sl, col_sl = window
    fobj = granule if isinstance(granule, Path) else earthaccess.open([granule])[0]
    with h5py.File(fobj, "r") as f:
        g = f[GROUP]
        sm = g["soil_moisture"][row_sl, col_sl]
        qf = g["retrieval_qual_flag"][row_sl, col_sl]

    recommended = (sm != FILL) & ((qf & 1) == 0)
    date_str = granule_date(granule)
    records = []
    for station, row, col in zip(
        cells["station"], cells["row"], cells["col"], strict=True
    ):
        r, c = row - row_sl.start, col - col_sl.start
        if not recommended[r, c]:
            continue
        records.append(
            {"station": station, "date": date_str, "smap_sm": float(sm[r, c])}
        )
    return records


def load_existing(out_path: Path) -> tuple[pd.DataFrame, set]:
    if not out_path.exists():
        return pd.DataFrame(columns=["station", "date", "smap_sm"]), set()
    existing = pd.read_parquet(out_path)
    done = set(pd.to_datetime(existing["date"]).dt.strftime("%Y%m%d"))
    return existing, done


def write_output(existing: pd.DataFrame, new_records: list, out_path: Path) -> int:
    frames = [existing] if len(existing) else []
    if new_records:
        fresh = pd.DataFrame(new_records)
        fresh["date"] = pd.to_datetime(fresh["date"], format="%Y%m%d")
        frames.append(fresh)
    if not frames:
        return 0
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["station", "date"], keep="last")
    combined = combined.sort_values(["station", "date"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    return len(combined)


def run(data_dir: Path, start: str, end: str, h5_dir: Path | None = None) -> None:
    if h5_dir is None:
        earthaccess.login(strategy="netrc")

    cells = station_grid_cells(data_dir / "mesonet/mt_mesonet_stations.csv")
    window = bounding_window(cells)
    row_sl, col_sl = window
    print(f"{len(cells)} Mesonet stations")
    print(
        f"Cropped EASE2_G9km window: rows {row_sl.start}:{row_sl.stop}, "
        f"cols {col_sl.start}:{col_sl.stop} "
        f"({row_sl.stop - row_sl.start}x{col_sl.stop - col_sl.start} cells)"
    )

    out_path = data_dir / "validation/smap_mesonet_extractions.parquet"
    existing, done = load_existing(out_path)
    if done:
        print(f"Resuming: {len(done)} dates already in {out_path.name}")

    if h5_dir is not None:
        by_date = local_daily_granules(h5_dir)
        print(f"Found {len(by_date)} local {SHORT_NAME} granules under {h5_dir}")
    else:
        by_date = find_daily_granules(start, end)
        print(f"Found {len(by_date)} {SHORT_NAME} granules in {start}..{end}")

    d0 = datetime.strptime(start, "%Y-%m-%d").date()  # noqa: DTZ007 - calendar date only, no tz
    d1 = datetime.strptime(end, "%Y-%m-%d").date()  # noqa: DTZ007 - calendar date only, no tz
    all_days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]

    new_records = []
    n_done = n_skipped = n_missing = n_failed = 0
    consecutive_failures = 0

    for i, day in enumerate(all_days):
        date_str = day.strftime("%Y%m%d")
        if date_str in done:
            n_skipped += 1
            continue
        granule = by_date.get(date_str)
        if granule is None:
            n_missing += 1
            print(f"  {date_str}: no granule (data gap), skipping")
            continue
        try:
            new_records.extend(extract_day(granule, cells, window))
            n_done += 1
            consecutive_failures = 0
        except Exception as exc:  # noqa: BLE001 - keep the run going past transient errors
            n_failed += 1
            consecutive_failures += 1
            print(f"  {date_str}: FAILED ({type(exc).__name__}: {exc})")
            if consecutive_failures > MAX_CONSECUTIVE_FAILURES:
                print(
                    f"\nStopping: {consecutive_failures} consecutive failures "
                    f"at {date_str}."
                )
                break

        if n_done and n_done % FLUSH_EVERY == 0 and new_records:
            total = write_output(existing, new_records, out_path)
            existing, _ = load_existing(out_path)
            new_records = []
            print(f"  flushed -> {total} station-day rows")

        if (i + 1) % PROGRESS_EVERY == 0:
            print(
                f"[{i + 1}/{len(all_days)}] {date_str} | pulled {n_done} "
                f"| resumed-skip {n_skipped} | gaps {n_missing} | failed {n_failed}"
            )

    total = write_output(existing, new_records, out_path)
    print(f"\nDays pulled this run: {n_done}")
    print(f"Days already present (skipped): {n_skipped}")
    print(f"Days with no granule (data gap): {n_missing}")
    print(f"Days failed: {n_failed}")
    print(f"Wrote {out_path} ({total} station-day rows)")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("data_dir", type=Path)
    p.add_argument("start", nargs="?", default=DEFAULT_START)
    p.add_argument(
        "end",
        nargs="?",
        default=date.today().strftime("%Y-%m-%d"),  # noqa: DTZ011 - local calendar date
    )
    p.add_argument(
        "--h5-dir",
        type=Path,
        default=None,
        help="local granule archive (e.g. /nas/soils/smap/SPL3SMP_E/hdf5); "
        "reads granules from disk instead of streaming via earthaccess",
    )
    a = p.parse_args(sys.argv[1:])
    run(a.data_dir, a.start, a.end, h5_dir=a.h5_dir)
