"""Stream just the soilMoisture (+ QF, coords) arrays for every available NISAR L3
SME2 granule on one track/frame, across the full mission to date. No full-file
download, no scene-stack/ancillary bands — soil moisture GeoTIFFs only.

Usage:
    uv run python scripts/track_frame_sm_stream.py <track> <frame> <pass_direction> /data/ssd2/nisar/
    uv run python scripts/track_frame_sm_stream.py 157 65 D /data/ssd2/nisar/
"""

import sys
from pathlib import Path

import earthaccess
import h5py
from eda_sme2_scene import GRIDS, write_soil_moisture_geotiff

SHORT_NAMES = ["NISAR_L3_SME2_PROVISIONAL_V1", "NISAR_L3_SME2_BETA_V1"]

_LOGGED_IN = False


def login() -> None:
    """Authenticate once per process (earthaccess.login is not free to repeat 40x)."""
    global _LOGGED_IN
    if not _LOGGED_IN:
        earthaccess.login(strategy="netrc")
        _LOGGED_IN = True


def granule_stem(track: int, frame: int, pass_direction: str, date_str: str) -> str:
    """Collision-safe output stem: unique per (track, frame, pass, acquisition date).

    Multiple track/frame combos can be acquired on the same calendar date, so the bare
    `sme2_<YYYYMMDD>` stem used by the original single-combo run silently overwrote.
    """
    return f"sme2_t{track:03d}f{frame:03d}{pass_direction.upper()[0]}_{date_str}"


def find_granules(track: int, frame: int, pass_direction: str) -> list:
    """Search both SME2 collections for one track/frame/pass, provisional first.

    A given acquisition can appear in both the provisional and beta collections; the
    provisional version wins and the duplicate beta granule is dropped so the two do
    not fight over the same output path.
    """
    login()
    pattern = f"*_{track:03d}_{pass_direction}_{frame:03d}_*"
    granules, seen = [], set()
    for short_name in SHORT_NAMES:
        found = earthaccess.search_data(
            short_name=short_name, granule_name=pattern, count=2000
        )
        kept = []
        for granule in found:
            key = granule_date(granule)
            if key is not None and key in seen:
                continue
            if key is not None:
                seen.add(key)
            kept.append(granule)
        print(f"  {short_name}: {len(found)} found, {len(kept)} kept")
        granules.extend(kept)
    return granules


def granule_date(granule) -> str | None:
    """YYYYMMDD from CMR temporal metadata, so a granule can be skipped without opening it."""
    try:
        begin = granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
    except (KeyError, TypeError):
        return None
    return begin[:10].replace("-", "")


def stream_soil_moisture(granule) -> tuple[dict, dict]:
    """Read only soilMoisture, retrievalQualityFlag, and grid coords remotely."""
    fobj = earthaccess.open([granule])[0]
    with h5py.File(fobj, "r") as f:
        g = f[GRIDS]
        data = {
            "soil_moisture": g["soilMoisture"][:],
            "retrieval_qf": g["retrievalQualityFlag"][:],
            "x": g["xCoordinates"][:],
            "y": g["yCoordinates"][:],
            "dx": g["xCoordinateSpacing"][()],
            "dy": g["yCoordinateSpacing"][()],
        }
        ident = f["science/LSAR/identification"]
        meta = {
            "granule_id": ident["granuleId"][()].decode(),
            "start_time": ident["zeroDopplerStartTime"][()].decode(),
        }
    return data, meta


def process(
    track: int, frame: int, pass_direction: str, out_dir: Path, verbose: bool = True
) -> dict:
    """Stream every granule for one combo. Returns {found, written, skipped, failed}."""
    granules = find_granules(track, frame, pass_direction)
    tif_dir = out_dir / "tif"
    counts = {"found": len(granules), "written": 0, "skipped": 0, "failed": 0}
    failures = []
    print(
        f"Found {len(granules)} granules for track={track} frame={frame} pass={pass_direction}"
    )

    for i, granule in enumerate(granules):
        native_id = granule["meta"]["native-id"]
        date_str = granule_date(granule)
        if date_str is not None:
            stem = granule_stem(track, frame, pass_direction, date_str)
            if (tif_dir / f"{stem}_soil_moisture.tif").exists():
                counts["skipped"] += 1
                if verbose:
                    print(f"[{i + 1}/{len(granules)}] {native_id}\n  exists, skipping")
                continue
        if verbose:
            print(f"[{i + 1}/{len(granules)}] {native_id}")
        try:
            data, meta = stream_soil_moisture(granule)
        except Exception as e:  # noqa: BLE001 - one bad granule (network/HDF5) shouldn't kill the batch
            counts["failed"] += 1
            failures.append((native_id, repr(e)))
            print(f"  FAILED: {e}")
            continue
        date_str = meta["start_time"][:10].replace("-", "")
        stem = granule_stem(track, frame, pass_direction, date_str)
        out_path = write_soil_moisture_geotiff(data, meta, out_dir, stem=stem)
        counts["written"] += 1
        if verbose:
            print(f"  wrote {out_path.name}")

    counts["failures"] = failures
    return counts


if __name__ == "__main__":
    track = int(sys.argv[1])
    frame = int(sys.argv[2])
    pass_direction = sys.argv[3]
    out_dir = Path(sys.argv[4])
    process(track, frame, pass_direction, out_dir)
