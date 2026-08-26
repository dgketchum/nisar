#!/bin/bash
# SMAP 9 km pixel land-cover extraction for the MT Mesonet SMAP/NISAR head-to-head.
# Submits 5 small Earth Engine table-export tasks (231 Mesonet points + 100 m buffers,
# 211 unique EASE2 9 km cells; CDL cultivated + IrrMapper/LANID irrigation means, and
# CDL 2025 crop-class histograms) to gs://wudr/nisar/smap_pixel_landcover/.
# Server-side batch exports -- the command returns as soon as the tasks are submitted
# (watch them at https://code.earthengine.google.com/tasks); we take it from the bucket.
#
# Run from the repo root. Setup once: uv sync --all-extras
# (or: pip install earthengine-api pandas pyproj)

PYTHONPATH=scripts python scripts/sample_smap_pixel_landcover.py \
    --stations zh/mt_mesonet_smap_stations.csv \
    --bucket wudr \
    --project ee-hoylman
