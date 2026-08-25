"""Stream NISAR L3 SME2 soil-moisture GeoTIFFs for every track/frame/pass that covers
a Montana Mesonet station.

Reads the precomputed combo list (track, frame, passDirection, pass_code) and runs the
single-combo streaming pull from track_frame_sm_stream over each row. Output stems are
unique per (track, frame, pass, date), and a granule whose GeoTIFF already exists is
skipped, so the run is resumable.

Usage:
    uv run python scripts/pull_mesonet_frames_sme2.py
    uv run python scripts/pull_mesonet_frames_sme2.py <combos_csv> <out_dir>
"""

import sys
import traceback
from pathlib import Path

import pandas as pd
from track_frame_sm_stream import process

DEFAULT_CSV = Path("/data/ssd2/nisar/reference/mt_mesonet_track_frames.csv")
DEFAULT_OUT = Path("/data/ssd2/nisar")


def run(combos_csv: Path, out_dir: Path) -> None:
    combos = pd.read_csv(combos_csv)
    n = len(combos)
    print(f"{n} track/frame/pass combos from {combos_csv}\n")

    totals = {"found": 0, "written": 0, "skipped": 0, "failed": 0}
    combo_failures, granule_failures = [], []

    for i, row in enumerate(combos.itertuples(index=False), start=1):
        track, frame = int(row.track), int(row.frame)
        pass_code = str(row.pass_code).strip().upper()[0]
        label = f"t{track:03d} f{frame:03d} {pass_code}"
        print(f"=== [{i}/{n}] {label} ===")
        try:
            counts = process(track, frame, pass_code, out_dir, verbose=True)
        except Exception as e:  # noqa: BLE001 - a dead combo shouldn't kill the batch
            combo_failures.append((label, repr(e)))
            print(f"  COMBO FAILED: {e}")
            traceback.print_exc()
            continue
        for key in totals:
            totals[key] += counts[key]
        granule_failures.extend((label, nid, err) for nid, err in counts["failures"])
        print(
            f"  -> {label}: {counts['found']} granules, {counts['written']} new, "
            f"{counts['skipped']} already present, {counts['failed']} failed\n"
        )

    print("=" * 60)
    print(f"Combos processed : {n - len(combo_failures)} / {n}")
    print(f"Granules found   : {totals['found']}")
    print(f"GeoTIFFs written : {totals['written']}")
    print(f"Already present  : {totals['skipped']}")
    print(f"Granules failed  : {totals['failed']}")
    if combo_failures:
        print(f"\nFailed combos ({len(combo_failures)}):")
        for label, err in combo_failures:
            print(f"  {label}: {err}")
    if granule_failures:
        print(f"\nFailed granules ({len(granule_failures)}):")
        for label, nid, err in granule_failures:
            print(f"  {label} {nid}: {err}")

    tif_dir = out_dir / "tif"
    tifs = sorted(tif_dir.glob("*_soil_moisture.tif"))
    size_gb = sum(p.stat().st_size for p in tifs) / 1e9
    print(f"\n{len(tifs)} soil-moisture GeoTIFFs in {tif_dir} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    combos_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    run(combos_csv, out_dir)
