"""Download every available NISAR L3 SME2 granule for one track/frame and write
each as GeoTIFFs (via eda_sme2_scene.write_geotiffs).

Usage:
    uv run python scripts/track_frame_tifs.py <track> <frame> <pass_direction> /data/ssd2/nisar/
    uv run python scripts/track_frame_tifs.py 157 65 D /data/ssd2/nisar/
"""

import sys
import tempfile
from pathlib import Path

import earthaccess
from eda_sme2_scene import load_scene, write_geotiffs

SHORT_NAME = "NISAR_L3_SME2_PROVISIONAL_V1"


def find_granules(track: int, frame: int, pass_direction: str) -> list:
    earthaccess.login(strategy="netrc")
    pattern = f"*_{track:03d}_{pass_direction}_{frame:03d}_*"
    return earthaccess.search_data(
        short_name=SHORT_NAME, granule_name=pattern, count=2000
    )


def process(track: int, frame: int, pass_direction: str, out_dir: Path) -> None:
    granules = find_granules(track, frame, pass_direction)
    print(
        f"Found {len(granules)} granules for track={track} frame={frame} pass={pass_direction}"
    )

    with tempfile.TemporaryDirectory() as tmp:
        for i, granule in enumerate(granules):
            native_id = granule["meta"]["native-id"]
            print(f"[{i + 1}/{len(granules)}] {native_id}")
            local = earthaccess.download([granule], tmp)[0]
            data, meta = load_scene(local)
            date_str = meta["start_time"][:10].replace("-", "")
            stem = f"sme2_{date_str}"
            write_geotiffs(data, meta, out_dir, stem=stem)
            print(f"  wrote {stem}_soil_moisture.tif, {stem}_scene_stack.tif")


if __name__ == "__main__":
    track = int(sys.argv[1])
    frame = int(sys.argv[2])
    pass_direction = sys.argv[3]
    out_dir = Path(sys.argv[4])
    process(track, frame, pass_direction, out_dir)
