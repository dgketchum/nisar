"""Tests for the QC screening in pull_smap_mesonet.py.

The default path keeps only recommended-quality retrievals (qual bit 0 == 0); the
``allow_flagged`` whitelist is the targeted relaxation used to recover stations whose
cell is permanently flagged for a reason an in-situ check has shown to be conservative
(blmrubyc: mountainous-terrain flag, r_slow ~0.83 vs its 10 cm sensor). Fill values
never survive either path, and station-scoped resume lets a targeted re-run revisit
dates the full-network pull already covered for everyone else.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pull_smap_mesonet import FILL, GROUP, extract_day, load_existing

GRANULE_NAME = "SMAP_L3_SM_P_E_20200715_R19240_001.h5"


def _granule(tmp_path):
    """3x3 window granule: (0,0) recommended, (1,1) flagged, (2,2) fill."""
    sm = np.full((3, 3), FILL, dtype=np.float32)
    qf = np.zeros((3, 3), dtype=np.uint16)
    sm[0, 0], qf[0, 0] = 0.20, 0  # recommended
    sm[1, 1], qf[1, 1] = 0.15, 1  # retrieved, not recommended
    qf[2, 2] = 1  # no retrieval at all
    path = tmp_path / GRANULE_NAME
    with h5py.File(path, "w") as f:
        g = f.create_group(GROUP)
        g.create_dataset("soil_moisture", data=sm)
        g.create_dataset("retrieval_qual_flag", data=qf)
    return path


CELLS = pd.DataFrame(
    {
        "station": ["ok_station", "flagged_station", "fill_station"],
        "row": [0, 1, 2],
        "col": [0, 1, 2],
    }
)
WINDOW = (slice(0, 3), slice(0, 3))


def test_default_keeps_only_recommended(tmp_path):
    recs = extract_day(_granule(tmp_path), CELLS, WINDOW)
    assert [r["station"] for r in recs] == ["ok_station"]
    assert recs[0]["smap_sm"] == np.float32(0.20)
    assert recs[0]["qual_flag"] == 0


def test_allow_flagged_recovers_station_but_not_fill(tmp_path):
    recs = extract_day(
        _granule(tmp_path),
        CELLS,
        WINDOW,
        allow_flagged=frozenset({"flagged_station", "fill_station"}),
    )
    by_station = {r["station"]: r for r in recs}
    assert set(by_station) == {"ok_station", "flagged_station"}
    assert by_station["flagged_station"]["smap_sm"] == np.float32(0.15)
    assert by_station["flagged_station"]["qual_flag"] == 1  # provenance preserved


def test_load_existing_done_dates_scoped_to_stations(tmp_path):
    out = tmp_path / "extractions.parquet"
    pd.DataFrame(
        {
            "station": ["a", "a", "b"],
            "date": pd.to_datetime(["2020-07-01", "2020-07-02", "2020-07-01"]),
            "smap_sm": [0.1, 0.2, 0.3],
        }
    ).to_parquet(out, index=False)

    _, done_all = load_existing(out)
    assert done_all == {"20200701", "20200702"}
    # station-scoped: a targeted 'b' re-run still owes 07-02
    _, done_b = load_existing(out, stations={"b"})
    assert done_b == {"20200701"}
    # a station with no rows yet owes everything
    _, done_c = load_existing(out, stations={"c"})
    assert done_c == set()
