"""EDA on a single NISAR L3 SME2 granule: soil moisture image plus the fields
needed to interpret it (quality flags, backscatter, incidence angle, land cover).

Usage:
    uv run python scripts/eda_sme2_scene.py <path_to_h5> outputs/
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_origin

GRIDS = "science/LSAR/SME2/grids"
SM_FILL = -9999.0
EASE2_EPSG = 6933


def load_scene(h5_path: str) -> dict:
    with h5py.File(h5_path, "r") as f:
        g = f[GRIDS]
        data = {
            "soil_moisture": g["soilMoisture"][:],
            "sm_uncertainty": g["soilMoistureUncertainty"][:],
            "retrieval_qf": g["retrievalQualityFlag"][:],
            "surface_qf": g["surfaceQualityFlag"][:],
            "land_cover": g["ancillaryData/landCover"][:],
            "incidence_angle": g["ancillaryData/localIncidenceAngle"][:],
            "waterbody_frac": g["ancillaryData/waterbodyFraction"][:],
            "sigma0_hh_a": g["radarData/frequencyA/sigma0HH"][:],
            "sigma0_hv_a": g["radarData/frequencyA/sigma0HV"][:],
            "lat": g["latitude"][:],
            "lon": g["longitude"][:],
            "x": g["xCoordinates"][:],
            "y": g["yCoordinates"][:],
            "dx": g["xCoordinateSpacing"][()],
            "dy": g["yCoordinateSpacing"][()],
        }
        ident = f["science/LSAR/identification"]
        meta = {
            "granule_id": ident["granuleId"][()].decode(),
            "start_time": ident["zeroDopplerStartTime"][()].decode(),
            "track": int(ident["trackNumber"][()]),
            "frame": int(ident["frameNumber"][()]),
            "pass_direction": ident["orbitPassDirection"][()].decode(),
        }
        proc = f["science/LSAR/SME2/metadata/processingInformation"]
        meta["l2_gcov_inputs"] = [
            v.decode().strip()
            for v in proc["inputs/l2GcovGranules"][:]
            if v.decode().strip().lower() != "none"
        ]
    return data, meta


def summarize(data: dict, meta: dict) -> str:
    sm = data["soil_moisture"]
    qf = data["retrieval_qf"]
    valid = sm != SM_FILL
    recommended = valid & ((qf & 1) == 0)

    lines = [
        f"Granule: {meta['granule_id']}",
        f"Acquired: {meta['start_time']}  track={meta['track']} frame={meta['frame']} pass={meta['pass_direction']}",
        f"L2 GCOV input(s): {meta['l2_gcov_inputs']}",
        f"Grid shape: {sm.shape}  (200 m EASE-Grid 2.0)",
        f"Valid (non-fill) pixels: {valid.sum()} / {sm.size} ({100 * valid.mean():.1f}%)",
        (
            f"Recommended (qf bit0==0) pixels: {recommended.sum()} ({100 * recommended.mean():.1f}% of all)"
            f", {100 * recommended.sum() / max(valid.sum(), 1):.1f}% of valid"
        ),
        f"Soil moisture range (recommended): {np.nanmin(sm[recommended]):.3f} - {np.nanmax(sm[recommended]):.3f} m3/m3",
        f"Soil moisture mean/median (recommended): {np.nanmean(sm[recommended]):.3f} / {np.nanmedian(sm[recommended]):.3f}",
        f"Incidence angle range: {np.nanmin(data['incidence_angle']):.1f} - {np.nanmax(data['incidence_angle']):.1f} deg",
        f"Land cover classes present: {sorted(set(np.unique(data['land_cover'])))}",
        f"Waterbody fraction > 0.1: {100 * (data['waterbody_frac'] > 0.1).mean():.2f}% of pixels",
    ]
    return "\n".join(lines)


def make_figures(data: dict, meta: dict, out_dir: Path) -> None:
    out_dir = out_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    sm = np.where(data["soil_moisture"] == SM_FILL, np.nan, data["soil_moisture"])
    qf = data["retrieval_qf"]
    recommended = (data["soil_moisture"] != SM_FILL) & ((qf & 1) == 0)
    sm_masked = np.where(recommended, sm, np.nan)

    extent = [
        data["lon"].min(),
        data["lon"].max(),
        data["lat"].min(),
        data["lat"].max(),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(meta["granule_id"], fontsize=9)

    im0 = axes[0, 0].imshow(
        sm, extent=extent, origin="upper", cmap="YlGnBu", vmin=0, vmax=0.5
    )
    axes[0, 0].set_title("soilMoisture (all, incl. non-recommended)")
    plt.colorbar(im0, ax=axes[0, 0], label="m3/m3")

    im1 = axes[0, 1].imshow(
        sm_masked, extent=extent, origin="upper", cmap="YlGnBu", vmin=0, vmax=0.5
    )
    axes[0, 1].set_title("soilMoisture (qf bit0==0 only)")
    plt.colorbar(im1, ax=axes[0, 1], label="m3/m3")

    qf_class = np.full(qf.shape, -1, dtype=int)
    qf_class[qf == -9999] = 0  # fill
    qf_class[(qf != -9999) & ((qf & 1) == 0)] = 1  # recommended
    qf_class[(qf != -9999) & ((qf & 1) == 1)] = 2  # not recommended
    im2 = axes[0, 2].imshow(
        qf_class, extent=extent, origin="upper", cmap="RdYlGn", vmin=0, vmax=2
    )
    axes[0, 2].set_title("retrievalQualityFlag bit0 (0=fill,1=ok,2=flagged)")
    plt.colorbar(im2, ax=axes[0, 2], ticks=[0, 1, 2])

    sigma0_db = np.where(
        data["sigma0_hh_a"] > 0, 10 * np.log10(data["sigma0_hh_a"]), np.nan
    )
    im3 = axes[1, 0].imshow(
        sigma0_db, extent=extent, origin="upper", cmap="gray", vmin=-25, vmax=0
    )
    axes[1, 0].set_title("sigma0 HH (freqA, dB)")
    plt.colorbar(im3, ax=axes[1, 0], label="dB")

    ia = np.where(data["incidence_angle"] >= 0, data["incidence_angle"], np.nan)
    im4 = axes[1, 1].imshow(ia, extent=extent, origin="upper", cmap="viridis")
    axes[1, 1].set_title("localIncidenceAngle (deg)")
    plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(
        data["land_cover"], extent=extent, origin="upper", cmap="tab20"
    )
    axes[1, 2].set_title("landCover (class code)")
    plt.colorbar(im5, ax=axes[1, 2])

    for ax in axes.flat:
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

    fig.tight_layout()
    fig.savefig(out_dir / "sme2_scene_overview.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sm_masked[~np.isnan(sm_masked)].ravel(), bins=60, color="#2b6cb0")
    ax.set_xlabel("soil moisture (m3/m3)")
    ax.set_ylabel("pixel count")
    ax.set_title("Recommended-retrieval soil moisture histogram")
    fig.tight_layout()
    fig.savefig(out_dir / "sme2_sm_histogram.png", dpi=140)
    plt.close(fig)


def write_geotiffs(data: dict, meta: dict, out_dir: Path) -> None:
    """Write QGIS-renderable GeoTIFFs on the native EASE-Grid 2.0 (EPSG:6933) grid."""
    out_dir = out_dir / "tif"
    out_dir.mkdir(parents=True, exist_ok=True)
    dx = float(data["dx"])
    dy = float(data["dy"])
    west = float(data["x"][0]) - dx / 2
    north = float(data["y"][0]) - dy / 2
    transform = from_origin(west, north, dx, abs(dy))

    sm = data["soil_moisture"]
    qf = data["retrieval_qf"]
    recommended = (sm != SM_FILL) & ((qf & 1) == 0)
    sm_masked = np.where(recommended, sm, np.nan).astype("float32")

    profile = {
        "driver": "GTiff",
        "height": sm.shape[0],
        "width": sm.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": f"EPSG:{EASE2_EPSG}",
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
    }
    with rasterio.open(out_dir / "sme2_soil_moisture.tif", "w", **profile) as dst:
        dst.write(sm_masked, 1)
        dst.set_band_description(1, "soilMoisture_recommended_m3m3")

    sigma0_db = np.where(
        data["sigma0_hh_a"] > 0, 10 * np.log10(data["sigma0_hh_a"]), np.nan
    ).astype("float32")
    ia = np.where(data["incidence_angle"] >= 0, data["incidence_angle"], np.nan).astype(
        "float32"
    )
    stack = np.stack([sm_masked, sigma0_db, ia, data["land_cover"].astype("float32")])
    stack_profile = dict(profile, count=stack.shape[0])
    with rasterio.open(out_dir / "sme2_scene_stack.tif", "w", **stack_profile) as dst:
        dst.write(stack)
        for i, name in enumerate(
            [
                "soilMoisture_recommended_m3m3",
                "sigma0HH_dB",
                "localIncidenceAngle_deg",
                "landCover_class",
            ],
            start=1,
        ):
            dst.set_band_description(i, name)


if __name__ == "__main__":
    h5_path = sys.argv[1]
    out_dir = Path(sys.argv[2])
    data, meta = load_scene(h5_path)
    print(summarize(data, meta))
    make_figures(data, meta, out_dir)
    write_geotiffs(data, meta, out_dir)
    print(f"\nFigures and GeoTIFFs written to {out_dir}")
