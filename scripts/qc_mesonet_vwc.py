"""QC the Montana Mesonet in-situ soil-VWC record and NaN the failing values.

Full-record SMAP scoring (score_smap_mesonet_fullrecord.py) surfaced physically
impossible in-situ VWC in the bulk archive: a units/scaling failure at blmcapit
(values of 283-1354 m3/m3 through 2020-21, mirrored at 70 and 91 cm), a
sentinel-like 2.147e9 (int32 max) at arskeose 91 cm, a run of marginally
negative 5 cm values at ftbentcb, and a handful of one-off 1.846 / -0.696
spikes. Those are archive defects, not soil physics, and every metric computed
against them is meaningless -- but they must not simply vanish either, so this
script splits the job in two:

* every failing observation is written out in full to
  ``mesonet/mt_mesonet_vwc_qc_failures.csv``, original value included. That CSV
  is the recovery record: the parquet edit is reversible from it.
* only then, and only with ``--apply``, are those values set to NaN in
  ``mesonet/mt_mesonet_daily_long.parquet`` in place. Rows are kept -- the
  station-day still exists, its value is simply unknown.

The rule set is a dict so it can grow, but exactly one rule is implemented:
physical bounds, value < 0 or value > 1 m3/m3. Nothing else is touched. Runs of
suspicious-but-in-range values (long constants, single-day spikes inside [0, 1])
are deliberately left alone; ``--report-suspicious`` prints candidates for
future rules without acting on them.

Usage:
    uv run python scripts/qc_mesonet_vwc.py                   # scan + apply
    uv run python scripts/qc_mesonet_vwc.py --no-apply        # read-only scan
    uv run python scripts/qc_mesonet_vwc.py /data/ssd2/nisar/ --report-suspicious
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path("/data/ssd2/nisar/")

VWC_PREFIX = "soil_vwc"
VWC_LO, VWC_HI = 0.0, 1.0

# Thresholds for the report-only suspicious scan. Nothing here fails a value.
CONSTANT_RUN_DAYS = 30  # bit-identical daily value this long is a stuck sensor
SPIKE_JUMP = 0.20  # m3/m3 day-over-day up-and-back-down excursion


def out_of_range(df: pd.DataFrame) -> pd.Series:
    """Physically impossible volumetric water content."""
    return (df["value"] < VWC_LO) | (df["value"] > VWC_HI)


# name -> predicate returning a boolean failure mask over the VWC rows. Add rules
# here; each contributes its own rows to the failure CSV, so a value failing two
# rules appears twice and both reasons survive.
RULES = {
    "out_of_range_0_1": out_of_range,
}


def load_long(data_dir: Path) -> tuple[Path, pd.DataFrame]:
    path = data_dir / "mesonet/mt_mesonet_daily_long.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    is_vwc = df["element"].str.startswith(VWC_PREFIX)
    print(
        f"Archive: {len(df)} rows, {int(is_vwc.sum())} soil VWC rows, "
        f"{df.loc[is_vwc, 'station'].nunique()} stations with VWC, "
        f"{sorted(df.loc[is_vwc, 'element'].unique())}"
    )
    units = sorted(df.loc[is_vwc, "units"].unique())
    if units != ["m3/m3"]:
        raise ValueError(f"unexpected VWC units in archive: {units}")
    return path, df


def station_context(data_dir: Path) -> pd.DataFrame | None:
    """sub_network / county / name per station, if the station table is present."""
    path = data_dir / "mesonet/mt_mesonet_stations.csv"
    if not path.exists():
        print(f"NOTE: {path} absent; failure CSV will carry no station attributes")
        return None
    cols = ["station", "name", "sub_network", "county", "date_installed"]
    return pd.read_csv(path)[cols]


def run_lengths(dates: pd.Series) -> np.ndarray:
    """Length of the consecutive-daily block of flagged dates each row sits in.

    Separates a sustained archive failure (blmcapit's 103-day scaling run) from an
    isolated one-day spike without needing a second pass over the data.
    """
    days = dates.to_numpy("datetime64[D]").astype("int64")
    order = np.argsort(days, kind="stable")
    sorted_days = days[order]
    breaks = np.diff(sorted_days) > 1
    block = np.concatenate([[0], np.cumsum(breaks)])
    sizes = np.bincount(block)
    out = np.empty(len(days), dtype="int64")
    out[order] = sizes[block]
    return out


def find_failures(df: pd.DataFrame, context: pd.DataFrame | None) -> pd.DataFrame:
    """One row per (failing observation, rule), original value preserved."""
    vwc = df[df["element"].str.startswith(VWC_PREFIX)]
    frames = []
    for rule, predicate in RULES.items():
        mask = predicate(vwc)
        if not mask.any():
            print(f"rule {rule}: no failures")
            continue
        hit = vwc[mask].copy()
        hit["rule"] = rule
        print(f"rule {rule}: {len(hit)} failing observations")
        frames.append(hit)

    if not frames:
        return pd.DataFrame(
            columns=[
                "station",
                "date",
                "element",
                "depth_cm",
                "value",
                "units",
                "rule",
                "run_length",
            ]
        )

    fail = pd.concat(frames, ignore_index=True)
    fail["run_length"] = 0
    for keys, grp in fail.groupby(["station", "element", "rule"], sort=False):
        del keys
        fail.loc[grp.index, "run_length"] = run_lengths(grp["date"])

    if context is not None:
        fail = fail.merge(context, on="station", how="left")
    return fail.sort_values(["station", "element", "date"]).reset_index(drop=True)


def summarize(fail: pd.DataFrame) -> None:
    if fail.empty:
        print("\nNo QC failures.")
        return
    print(f"\n{'=' * 78}")
    print("QC FAILURES BY STATION")
    print(f"{'=' * 78}")
    per = fail.groupby("station").agg(
        n_bad=("value", "size"),
        elements=("element", lambda s: ",".join(sorted(s.unique()))),
        value_min=("value", "min"),
        value_max=("value", "max"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        max_run=("run_length", "max"),
    )
    per["first_date"] = per["first_date"].dt.date
    per["last_date"] = per["last_date"].dt.date
    print(per.to_string())

    print("\nBY STATION AND ELEMENT")
    per_el = fail.groupby(["station", "element", "rule"]).agg(
        n_bad=("value", "size"),
        value_min=("value", "min"),
        value_max=("value", "max"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        max_run=("run_length", "max"),
    )
    per_el["first_date"] = per_el["first_date"].dt.date
    per_el["last_date"] = per_el["last_date"].dt.date
    print(per_el.to_string())
    print(
        f"\nTotal: {len(fail)} failing observations, {fail['station'].nunique()} "
        f"station(s), {fail['element'].nunique()} element(s)"
    )


def apply_nan(path: Path, df: pd.DataFrame, fail: pd.DataFrame) -> None:
    """Set the failing values to NaN in the parquet in place, then verify.

    The rows stay; only ``value`` is blanked. Verification is on the re-read file,
    not the in-memory frame, so the check covers the round trip.
    """
    keys = ["station", "date", "element"]
    targets = fail[keys].drop_duplicates()
    idx = df.reset_index().merge(targets, on=keys, how="inner")["index"]
    n_before = int(df["value"].isna().sum())
    df.loc[idx, "value"] = np.nan
    df.to_parquet(path, index=False)
    print(f"\nWrote {path} ({len(idx)} values set to NaN)")

    check = pd.read_parquet(path)
    vwc = check[check["element"].str.startswith(VWC_PREFIX)]
    still_bad = int(((vwc["value"] < VWC_LO) | (vwc["value"] > VWC_HI)).sum())
    n_after = int(check["value"].isna().sum())
    print("VERIFY (re-read from disk):")
    print(
        f"  rows                        {len(check)} (unchanged: {len(check) == len(df)})"
    )
    print(f"  soil_vwc outside [0, 1]     {still_bad}")
    print(f"  NaN value count before      {n_before}")
    print(f"  NaN value count after       {n_after}")
    print(f"  NaN increase                {n_after - n_before}")
    print(f"  failure CSV rows            {len(fail)}")
    print(f"  distinct failing obs        {len(targets)}")
    if still_bad:
        raise ValueError(f"{still_bad} out-of-range soil_vwc values survived the edit")
    if n_after - n_before != len(targets):
        raise ValueError(
            f"NaN increase {n_after - n_before} != {len(targets)} flagged observations"
        )


def report_suspicious(df: pd.DataFrame) -> None:
    """In-range patterns that look wrong. REPORTED ONLY -- never flagged, never NaN'd."""
    vwc = df[df["element"].str.startswith(VWC_PREFIX)]
    vwc = vwc[(vwc["value"] >= VWC_LO) & (vwc["value"] <= VWC_HI)]

    print(f"\n{'=' * 78}")
    print("SUSPICIOUS BUT IN RANGE -- candidates for future rules, NOT acted on")
    print(f"{'=' * 78}")

    const_rows = []
    spike_rows = []
    for (stn, el), grp in vwc.groupby(["station", "element"], sort=False):
        grp = grp.sort_values("date")
        vals = grp["value"].to_numpy()
        days = grp["date"].to_numpy("datetime64[D]").astype("int64")
        contiguous = np.concatenate([[True], np.diff(days) == 1])
        same = np.concatenate([[False], vals[1:] == vals[:-1]]) & contiguous
        block = np.cumsum(~same)
        sizes = np.bincount(block)
        longest = int(sizes.max()) if len(sizes) else 0
        if longest >= CONSTANT_RUN_DAYS:
            at = int(np.argmax(sizes))
            where = grp["date"].to_numpy()[block == at]
            const_rows.append(
                {
                    "station": stn,
                    "element": el,
                    "run_days": longest,
                    "value": float(vals[block == at][0]),
                    "start": pd.Timestamp(where[0]).date(),
                    "end": pd.Timestamp(where[-1]).date(),
                }
            )

        if len(vals) >= 3:
            up = vals[1:-1] - vals[:-2]
            down = vals[1:-1] - vals[2:]
            step = np.concatenate([[True], np.diff(days) == 1])
            ok = step[1:-1] & step[2:] if len(step) > 2 else np.zeros(0, bool)
            spikes = (np.abs(up) > SPIKE_JUMP) & (np.abs(down) > SPIKE_JUMP)
            spikes &= np.sign(up) == np.sign(down)
            spikes &= ok
            if spikes.any():
                spike_rows.append(
                    {
                        "station": stn,
                        "element": el,
                        "n_spikes": int(spikes.sum()),
                        "max_excursion": float(
                            np.minimum(np.abs(up), np.abs(down))[spikes].max()
                        ),
                    }
                )

    if const_rows:
        tab = pd.DataFrame(const_rows).sort_values("run_days", ascending=False)
        print(
            f"\nConstant daily value for >= {CONSTANT_RUN_DAYS} consecutive days "
            f"({len(tab)} station-elements; longest run shown per series):"
        )
        print(tab.head(30).to_string(index=False))
    else:
        print(f"\nNo constant runs >= {CONSTANT_RUN_DAYS} days.")

    if spike_rows:
        tab = pd.DataFrame(spike_rows).sort_values("max_excursion", ascending=False)
        print(
            f"\nOne-day up-and-back spikes > {SPIKE_JUMP} m3/m3 "
            f"({len(tab)} station-elements, {int(tab['n_spikes'].sum())} days):"
        )
        print(tab.head(30).to_string(index=False))
    else:
        print(f"\nNo one-day spikes > {SPIKE_JUMP} m3/m3.")
    print("\nNothing above was modified.")


def build(data_dir: Path, apply: bool, suspicious: bool) -> int:
    path, df = load_long(data_dir)
    context = station_context(data_dir)
    fail = find_failures(df, context)

    out_path = data_dir / "mesonet/mt_mesonet_vwc_qc_failures.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fail.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(fail)} rows)")

    summarize(fail)
    if suspicious:
        report_suspicious(df)

    if not apply:
        print("\n--no-apply: parquet left untouched.")
    elif fail.empty:
        print("\nNothing to NaN.")
    else:
        apply_nan(path, df, fail)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("data_dir", nargs="?", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument(
        "--apply",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="NaN the failing values in the parquet in place (default). "
        "--no-apply scans and writes the CSV only.",
    )
    p.add_argument(
        "--report-suspicious",
        action="store_true",
        help="also print in-range patterns that look wrong (stuck sensors, "
        "one-day spikes) as candidates for future rules; nothing is modified",
    )
    a = p.parse_args(sys.argv[1:])
    sys.exit(build(a.data_dir, a.apply, a.report_suspicious))
