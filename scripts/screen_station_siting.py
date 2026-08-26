"""Screen in-situ stations for genuine in-field agricultural siting.

Why this exists
---------------
The Montana Mesonet taught this the hard way. Its stations are frequently sited on a
maintained grass or gravel pad set inside, or beside, a field -- so a land-cover or
irrigation label read from the station coordinate describes the *neighbourhood*, not
the ground under the probe. Montana compounds it by publishing coordinates rounded to
two decimal degrees (about 550 m of slack), which is coarser than the 200 m NISAR SME2
posting: at that precision the "point" pixel is not the probe's pixel at all. NDAWN's
own documentation makes the same concession, that station siting "may not be similar
with nearby agricultural fields."

So no station in the USACE Upper Missouri buildout is treated as a valid rainfed
cropland site merely because it exists. Each one has to earn the label.

What this screen can and cannot decide
--------------------------------------
Read the verdict as a statement about the **NISAR footprint**, not about the probe.
CDL is a 30 m raster and these networks fence a station pad of roughly 6 x 8 m, so a
pad is a quarter of one CDL pixel and cannot be resolved: a pad inside a wheat field
reads as wheat. ``pass`` therefore means "the 200 m cell NISAR integrates over is
cropland", which is the right question to ask of the *satellite* side of the pair.

It is not the right question for the *probe* side, and for three of these networks the
operators have already answered that one, in writing, against us. NDAWN: "Our goal is
to maintain grass cover in the area, except for a 4- to 5-foot square area that is kept
bare for soil temperature measurement." Nebraska (Shulski et al. 2018): "Locations are
sited on native or planted vegetation (grass), as opposed to cropland." South Dakota
documents sod/natural cover network-wide. Their probes are on grass pads *by design* --
so the Montana pad-in-field problem is not an occasional defect to screen out here, it
is the standing configuration of the whole buildout.

That makes ``pad_in_cropland`` a *lower bound* on the mismatch, not a census of it, and
it means a ``pass`` still needs per-station photo verification before anyone calls the
probe in-field. The photo URLs are carried in the station tables for exactly that.

What is measured
----------------
One Earth Engine pass per chunk of stations, three geometries per station -- the bare
point, a 100 m buffer (the NISAR 200 m footprint), and a 500 m buffer (the
neighbourhood) -- over:

* ``cultivated`` from CDL 2023, the last year the band is published. It is a
  multi-year derived band, so it does not flip to non-cultivated in a fallow year --
  which matters enormously here, because fallow is half of a Northern Plains dryland
  small-grain rotation and the single-year crop class would throw those sites away.
* ``cropland``, the actual crop class from the latest CDL year, as a full frequency
  histogram over the 100 m buffer so any regrouping can be recomputed downstream
  without re-running Earth Engine.
* the combined irrigation mask -- IrrMapper in the 11 western states, LANID in the
  eastern 38 -- because "rainfed" is a claim that needs its own evidence, especially
  in Nebraska.

The point/100 m/500 m nesting is the whole diagnostic. A probe on a grass pad inside
cropland reads as cultivated at 100 m and 500 m but *not* at the point, and that
contrast is what separates it from a genuine in-field probe.

The verdict
-----------
Three-valued, never two. ``field_verified`` is one of:

``pass``
    Cultivated at the point and over the footprint. The NISAR cell is cropland -- which,
    per the section above, is a claim about the footprint and not yet a claim that the
    probe is in the crop.
``fail``
    Not cultivated anywhere in the nesting -- rangeland, forest, developed. Not an
    agricultural site at all.
``ambiguous``
    Everything else, and the reason is recorded in ``verdict_reason``:

    * ``pad_in_cropland`` -- the Montana case: point not cultivated, footprint is.
      Either a pad inside a field, or a coordinate that missed. Not silently dropped,
      because whether it is usable depends on what the probe is actually in.
    * ``field_in_nonag`` -- point cultivated, footprint not. A small field in a
      rangeland matrix; the 200 m NISAR pixel will be mostly not-field.
    * ``coord_precision`` -- the coordinate is rounded coarser than the footprint, so
      the point reading is not the probe's pixel and *no* point-based verdict is
      trustworthy. This outranks a ``pass``: a station cannot be verified in-field on
      a coordinate that cannot resolve the field.
    * ``site_metadata`` -- the network's own site description says grass, turf, sod or
      similar. The operator's word beats the raster.

Nothing is dropped. Every station comes out with a verdict, and the counts of each are
reported, because "how many stations survived screening" is the number that decides
whether a network is worth a NISAR pull.

Input CSV needs ``station``, ``longitude``, ``latitude``. Optional and used if present:
``station_name``, ``coord_decimals`` (decimal places the network publishes), and
``site_description`` (free text land-cover/vegetation metadata).

Usage:
    uv run python scripts/screen_station_siting.py <stations_csv> <prefix>
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path("/data/ssd2/nisar")
EE_PROJECT = "ee-dgketchum"

CDL = "USDA/NASS/CDL"
CULTIVATED_YEAR = 2023  # last year the `cultivated` band is published
CROP_YEAR = 2025  # latest published CDL crop class
IRR = "projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp"
EAST_FC = "users/dgketchum/boundaries/eastern_38_dissolved"
LANID = "users/xyhuwmir/LANID/update/LANID2018-2020"
IRRMAPPER_YEAR = 2025
LANID_YEAR = 2020

SCALE_M = 30  # CDL / IrrMapper native
RADII = {"pt": 0, "100": 100, "500": 500}  # 100 m ~ the 200 m SME2 posting
CHUNK = 100

CULT_THRESHOLD = 0.5
# A coordinate no better than this cannot place a probe inside the 200 m footprint,
# so a point-based verdict built on it is not evidence.
COORD_UNC_LIMIT_M = 200.0
PAD_WORDS = re.compile(
    r"\b(grass|turf|sod|lawn|mow|mown|mowed|pasture|rangeland|range|gravel|"
    r"parking|roof|airport|park|campus|school|yard)\b",
    re.IGNORECASE,
)

# CDL class groupings, same lists as swap-stress station_landcover.py so the two
# panels stay comparable. Explicit rather than clever -- these get argued about.
SMALL_GRAIN = {21, 22, 23, 24, 25, 27, 28, 29, 205}
FALLOW = {61}
ROW_CROP = {
    1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 31, 32, 33, 34, 35, 38, 39, 41, 42, 43, 44,
    45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
}  # fmt: skip
HAY = {36, 37}
GRASS = {176}
SHRUB = {64, 152}
FOREST = {63, 141, 142, 143}
DEVELOPED = {82, 121, 122, 123, 124}
WATER = {83, 92, 111, 112}
BARREN = {65, 131}
WETLAND = {87, 190, 195}

GROUPS = {
    "smallgrain": SMALL_GRAIN,
    "fallow": FALLOW,
    "rowcrop": ROW_CROP,
    "hay": HAY,
    "grass": GRASS,
    "shrub": SHRUB,
    "forest": FOREST,
    "developed": DEVELOPED,
    "water": WATER,
    "barren": BARREN,
    "wetland": WETLAND,
}


def _stack():
    """cultivated + irrigation as one image; the crop class is reduced separately
    because it needs a frequency histogram rather than a mean."""
    import ee

    cult = (
        ee.Image(f"{CDL}/{CULTIVATED_YEAR}")
        .select("cultivated")
        .eq(2)  # 1 = non-cultivated, 2 = cultivated
        .unmask(0)
        .rename("cult")
    )
    west = (
        ee.ImageCollection(IRR)
        .filterDate(f"{IRRMAPPER_YEAR}-01-01", f"{IRRMAPPER_YEAR}-12-31")
        .select("classification")
        .mosaic()
        .lt(1)
        .unmask(0)
    )
    east = ee.Image(LANID).select([f"irMap{str(LANID_YEAR)[-2:]}"]).unmask(0)
    east_mask = ee.Image.constant(0).paint(ee.FeatureCollection(EAST_FC), 1)
    irr = west.where(east_mask, east).rename("irr")
    return cult.addBands(irr)


def _feature(row, radius):
    import ee

    pt = ee.Geometry.Point([float(row.longitude), float(row.latitude)])
    geom = pt if radius == 0 else pt.buffer(radius)
    return ee.Feature(geom, {"station": row.station})


def sample_cdl(df: pd.DataFrame, project: str = EE_PROJECT) -> pd.DataFrame:
    """Cultivated + irrigated fraction at each radius, and the 100 m crop histogram."""
    import ee

    ee.Initialize(project=project)
    stack = _stack()
    crop = ee.Image(f"{CDL}/{CROP_YEAR}").select("cropland")

    out = pd.DataFrame({"station": df["station"].values}).set_index("station")

    for tag, radius in RADII.items():
        vals = {}
        for i in range(0, len(df), CHUNK):
            part = df.iloc[i : i + CHUNK]
            fc = ee.FeatureCollection([_feature(r, radius) for r in part.itertuples()])
            res = stack.reduceRegions(
                collection=fc, reducer=ee.Reducer.mean(), scale=SCALE_M
            ).getInfo()
            for f in res["features"]:
                p = f["properties"]
                vals[p["station"]] = (p.get("cult"), p.get("irr"))
        out[f"cult_f_{tag}"] = pd.Series({k: v[0] for k, v in vals.items()})
        out[f"irr_f_{tag}"] = pd.Series({k: v[1] for k, v in vals.items()})
        print(f"  cultivated/irrigated fraction done: {tag}", flush=True)

    recs = []
    for i in range(0, len(df), CHUNK):
        part = df.iloc[i : i + CHUNK]
        fc = ee.FeatureCollection(
            [_feature(r, RADII["100"]) for r in part.itertuples()]
        )
        res = crop.reduceRegions(
            collection=fc, reducer=ee.Reducer.frequencyHistogram(), scale=SCALE_M
        ).getInfo()
        for f in res["features"]:
            p = f["properties"]
            hist = p.get("histogram") or {}
            total = sum(hist.values())
            rec = {"station": p["station"], "cdl_n_px": total}
            if total:
                fracs = {int(float(k)): v / total for k, v in hist.items()}
                rec["cdl_mode"] = max(fracs, key=fracs.get)
                rec["cdl_mode_frac"] = fracs[rec["cdl_mode"]]
                for name, classes in GROUPS.items():
                    rec[f"f_{name}"] = sum(v for k, v in fracs.items() if k in classes)
                rec["cdl_hist"] = json.dumps(
                    {k: round(v, 4) for k, v in sorted(fracs.items()) if v > 0}
                )
            recs.append(rec)
        print(f"  crop histogram: {min(i + CHUNK, len(df))}/{len(df)}", flush=True)

    hist_df = pd.DataFrame(recs).set_index("station")
    return out.join(hist_df).reset_index()


def coord_uncertainty_m(df: pd.DataFrame) -> pd.Series:
    """Half-cell distance implied by the published coordinate precision.

    ``coord_decimals`` is how many decimal places the network actually publishes, not
    how many the CSV happens to print, so it has to be supplied by the puller. Absent
    it, precision is inferred from the string form of the coordinates -- an inference,
    and one that reads high if a value happens to end in a zero, so it is only a
    fallback.
    """
    if "coord_decimals" in df.columns and df["coord_decimals"].notna().all():
        dec = df["coord_decimals"].astype(int)
    else:

        def _dp(v):
            s = f"{v}".rstrip("0")
            return len(s.split(".")[1]) if "." in s else 0

        dec = df[["longitude", "latitude"]].map(_dp).min(axis=1)
    step = 10.0 ** (-dec)
    half = step / 2.0
    ns = half * 111_320.0
    ew = half * 111_320.0 * df["latitude"].map(lambda y: math.cos(math.radians(y)))
    return (ns**2 + ew**2) ** 0.5


def verdict(df: pd.DataFrame) -> pd.DataFrame:
    """Three-valued siting verdict plus the reason it came out that way."""
    pt = df["cult_f_pt"]
    fp = df["cult_f_100"]
    if pt.isna().any() or fp.isna().any():
        raise RuntimeError(
            "cultivated fraction is null for some stations -- Earth Engine returned no "
            "value, which means the point missed the CDL domain rather than that the "
            "site is uncultivated; investigate before labelling"
        )

    pt_ag = pt >= CULT_THRESHOLD
    fp_ag = fp >= CULT_THRESHOLD

    res = pd.Series("ambiguous", index=df.index)
    reason = pd.Series("", index=df.index)

    res[pt_ag & fp_ag] = "pass"
    reason[pt_ag & fp_ag] = "point_and_footprint_cultivated"
    res[~pt_ag & ~fp_ag] = "fail"
    reason[~pt_ag & ~fp_ag] = "not_cultivated"
    reason[~pt_ag & fp_ag] = "pad_in_cropland"
    reason[pt_ag & ~fp_ag] = "field_in_nonag"

    # The operator's own site description outranks the raster: if the network says the
    # sensor sits on turf, a cultivated CDL pixel is describing the field next door.
    if "site_description" in df.columns:
        pad = df["site_description"].fillna("").str.contains(PAD_WORDS) & (
            res == "pass"
        )
        res[pad] = "ambiguous"
        reason[pad] = "site_metadata"

    # A coordinate that cannot resolve the footprint cannot verify a point, so this
    # outranks a pass. It does not rescue a fail: a station in the middle of
    # uncultivated ground at 100 m and 500 m is not agricultural whatever the rounding.
    # Applied last and only to what is still a pass, so a station that already failed
    # for a substantive reason keeps that reason rather than having it overwritten.
    coarse = (df["coord_unc_m"] > COORD_UNC_LIMIT_M) & (res == "pass")
    res[coarse] = "ambiguous"
    reason[coarse] = "coord_precision"

    out = df.copy()
    out["field_verified"] = res
    out["verdict_reason"] = reason
    return out


def build(
    stations_csv: Path,
    prefix: str,
    out_dir: Path,
    project: str = EE_PROJECT,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(stations_csv)
    for col in ("station", "longitude", "latitude"):
        if col not in df.columns:
            raise RuntimeError(f"station table is missing required column '{col}'")

    df["coord_unc_m"] = coord_uncertainty_m(df)
    print(f"{prefix}: screening {len(df)} stations", flush=True)
    print(
        f"  coordinate uncertainty: median {df['coord_unc_m'].median():.0f} m, "
        f"max {df['coord_unc_m'].max():.0f} m",
        flush=True,
    )

    sampled = sample_cdl(df, project=project)
    merged = df.merge(sampled, on="station", how="left")
    scored = verdict(merged)

    path = out_dir / f"{prefix}_siting_screen.csv"
    scored.to_csv(path, index=False)

    counts = scored["field_verified"].value_counts()
    print(f"\n--- {prefix}: siting screen ---")
    for label in ("pass", "ambiguous", "fail"):
        print(f"{label:>10}: {int(counts.get(label, 0))}")
    print("\nreasons:")
    for reason, n in scored["verdict_reason"].value_counts().items():
        print(f"  {reason}: {n}")
    ok = scored[scored["field_verified"] == "pass"]
    if len(ok):
        print(
            f"\npass cohort: median dryland-grain fraction "
            f"{(ok['f_smallgrain'] + ok['f_fallow']).median():.2f}, "
            f"median irrigated fraction {ok['irr_f_100'].median():.2f}"
        )
    print(f"\nwrote {path}")
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stations_csv", type=Path)
    p.add_argument("prefix", help="output filename prefix, e.g. 'ndawn'")
    p.add_argument("--out-dir", type=Path, default=DATA_DIR / "mesonet")
    p.add_argument("--project", default=EE_PROJECT)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args(sys.argv[1:])
    sys.exit(build(a.stations_csv, a.prefix, a.out_dir, project=a.project))
