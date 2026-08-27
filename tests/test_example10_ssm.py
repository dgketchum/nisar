"""Tests for the SSM conditioning-table builder in examples/10_MT_Mesonet/ssm.py."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SSM_PATH = Path(__file__).resolve().parents[1] / "examples" / "10_MT_Mesonet" / "ssm.py"

spec = importlib.util.spec_from_file_location("ssm10", SSM_PATH)
ssm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ssm)


def _extractions():
    return pd.DataFrame(
        {
            "station": ["blma", "blma", "blmb", "othersite"],
            "date": pd.to_datetime(
                ["2020-07-02", "2020-07-01", "2020-07-01", "2020-07-01"]
            ),
            "smap_sm": [0.2, 0.1, 0.3, 0.9],
            "qual_flag": [0, 0, 0, 0],
        }
    )


def test_build_ssm_table_subsets_renames_and_sorts():
    out = ssm.build_ssm_table(_extractions(), ["blma", "blmb"])
    assert list(out.columns) == ["date", "site_id", "smap_l3_sm"]
    assert set(out["site_id"]) == {"blma", "blmb"}  # cohort only
    blma = out[out.site_id == "blma"]
    assert blma["date"].is_monotonic_increasing
    assert blma["smap_l3_sm"].tolist() == [0.1, 0.2]


def test_build_ssm_table_raises_on_site_with_no_rows():
    with pytest.raises(SystemExit, match="blmmissing"):
        ssm.build_ssm_table(_extractions(), ["blma", "blmmissing"])
