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


def find_granules(track: int, frame: int, pass_direction: str) -> list:
    earthaccess.login(strategy="netrc")
    pattern = f"*_{track:03d}_{pass_direction}_{frame:03d}_*"
    granules = []
    for short_name in SHORT_NAMES:
        found = earthaccess.search_data(
            short_name=short_name, granule_name=pattern, count=2000
        )
        print(f"  {short_name}: {len(found)}")
        granules.extend(found)
    return granules


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


def process(track: int, frame: int, pass_direction: str, out_dir: Path) -> None:
    granules = find_granules(track, frame, pass_direction)
    print(
        f"Found {len(granules)} granules for track={track} frame={frame} pass={pass_direction}"
    )

    for i, granule in enumerate(granules):
        native_id = granule["meta"]["native-id"]
        print(f"[{i + 1}/{len(granules)}] {native_id}")
        try:
            data, meta = stream_soil_moisture(granule)
        except Exception as e:  # noqa: BLE001 - one bad granule (network/HDF5) shouldn't kill the batch
            print(f"  failed: {e}")
            continue
        date_str = meta["start_time"][:10].replace("-", "")
        stem = f"sme2_{date_str}"
        out_path = write_soil_moisture_geotiff(data, meta, out_dir, stem=stem)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    track = int(sys.argv[1])
    frame = int(sys.argv[2])
    pass_direction = sys.argv[3]
    out_dir = Path(sys.argv[4])
    process(track, frame, pass_direction, out_dir)
