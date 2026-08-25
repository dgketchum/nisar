#!/usr/bin/env python3
"""Add / refresh NISAR SME2 soil-moisture rasters and the frame-footprint layer in a QGIS .qgz.

Idempotent and reusable. New dated SME2 GeoTIFFs are discovered by globbing the tif dir; each is
loaded as its own raster layer (date in the layer name), all inside one group with a single shared
0-0.9 m3/m3 YlGnBu pseudocolor ramp so the time series is directly comparable. Ancillary soil-
moisture rasters (e.g. SMAP) in the ancillary dir are loaded into their own group with the same
ramp for cross-sensor comparison. The frame footprints are added as an outline-only polygon layer.
Re-running updates the managed groups in place without creating duplicates and without touching any
of the project's other layers.

The QGIS project XML (.qgs, no <?xml?> declaration, custom DOCTYPE) is edited as text via precise,
targeted string operations rather than a full ElementTree round-trip, because ElementTree does not
preserve the CDATA sections in other layers' definitions. ElementTree is used only to validate the
result. Managed elements are identified deterministically (SME2 group name, SME2 datasource pattern,
footprint datasource) so they can be found and replaced on every run.

Usage:
    python scripts/qgz_add_nisar_layers.py \
        --qgz ~/data/nisar.qgz \
        --tif-dir ~/data/nisar/tif \
        --footprint ~/data/nisar/reference/nisar_frames_conus.fgb

When new tifs arrive in the tif dir, just re-run with the same arguments.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

DOCTYPE = "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>"

SME2_GROUP_NAME = "NISAR SME2 soil moisture"
SME2_DATASOURCE_RE = re.compile(r"\./nisar/tif/[^<]*_soil_moisture\.tif")
# The undated file must NOT survive as its own layer; it is byte-identical to the 2026-06-25 scene.
UNDATED_STEM = "sme2_soil_moisture"
DATE_RE = re.compile(r"(\d{8})")

FOOTPRINT_DATASOURCE = "./nisar/reference/nisar_frames_conus.fgb"
FOOTPRINT_NAME = "NISAR frames (CONUS)"

# Ancillary soil-moisture rasters (e.g. SMAP) live in a sibling dir and share the SM ramp so they
# are directly comparable to the SME2 series. Discovered by globbing the ancillary dir for *.tif.
ANC_GROUP_NAME = "Ancillary soil moisture"
ANC_DATASOURCE_RE = re.compile(r"\./nisar/ancillary/[^<]*\.tif")

# WKT / proj strings pulled verbatim from the existing project so QGIS recognizes the CRS cleanly.
EASE_WKT = (
    'PROJCRS["WGS 84 / NSIDC EASE-Grid 2.0 Global",BASEGEOGCRS["WGS 84",'
    'ENSEMBLE["World Geodetic System 1984 ensemble",'
    'MEMBER["World Geodetic System 1984 (Transit)"],'
    'MEMBER["World Geodetic System 1984 (G730)"],'
    'MEMBER["World Geodetic System 1984 (G873)"],'
    'MEMBER["World Geodetic System 1984 (G1150)"],'
    'MEMBER["World Geodetic System 1984 (G1674)"],'
    'MEMBER["World Geodetic System 1984 (G1762)"],'
    'MEMBER["World Geodetic System 1984 (G2139)"],'
    'MEMBER["World Geodetic System 1984 (G2296)"],'
    'ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],'
    "ENSEMBLEACCURACY[2.0]],"
    'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]],'
    'CONVERSION["US NSIDC EASE-Grid 2.0 Global",'
    'METHOD["Lambert Cylindrical Equal Area",ID["EPSG",9835]],'
    'PARAMETER["Latitude of 1st standard parallel",30,'
    'ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8823]],'
    'PARAMETER["Longitude of natural origin",0,'
    'ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]],'
    'PARAMETER["False easting",0,LENGTHUNIT["metre",1],ID["EPSG",8806]],'
    'PARAMETER["False northing",0,LENGTHUNIT["metre",1],ID["EPSG",8807]]],'
    'CS[Cartesian,2],AXIS["easting (X)",east,ORDER[1],LENGTHUNIT["metre",1]],'
    'AXIS["northing (Y)",north,ORDER[2],LENGTHUNIT["metre",1]],'
    'USAGE[SCOPE["Environmental science - used as basis for EASE grid."],'
    'AREA["World between 86°S and 86°N."],BBOX[-86,-180,86,180]],ID["EPSG",6933]]'
)
EASE_PROJ4 = (
    "+proj=cea +lat_ts=30 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)

WGS84_WKT = (
    'GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",'
    'MEMBER["World Geodetic System 1984 (Transit)"],'
    'MEMBER["World Geodetic System 1984 (G730)"],'
    'MEMBER["World Geodetic System 1984 (G873)"],'
    'MEMBER["World Geodetic System 1984 (G1150)"],'
    'MEMBER["World Geodetic System 1984 (G1674)"],'
    'MEMBER["World Geodetic System 1984 (G1762)"],'
    'MEMBER["World Geodetic System 1984 (G2139)"],'
    'MEMBER["World Geodetic System 1984 (G2296)"],'
    'ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],'
    "ENSEMBLEACCURACY[2.0]],"
    'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],'
    'CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],'
    'ANGLEUNIT["degree",0.0174532925199433]],'
    'AXIS["geodetic longitude (Lon)",east,ORDER[2],'
    'ANGLEUNIT["degree",0.0174532925199433]],'
    'USAGE[SCOPE["Horizontal component of 3D system."],AREA["World."],'
    'BBOX[-90,-180,90,180]],ID["EPSG",4326]]'
)
WGS84_PROJ4 = "+proj=longlat +datum=WGS84 +no_defs"

# Shared soil-moisture pipe: fixed 0-0.9 m3/m3 YlGnBu pseudocolor, identical for every SM layer
# (SME2 and ancillary alike) so all soil-moisture images read on one common, comparable scale.
SM_PIPE = """      <pipe>
        <provider>
          <resampling zoomedOutResamplingMethod="nearestNeighbour" zoomedInResamplingMethod="nearestNeighbour" maxOversampling="2" enabled="false"/>
        </provider>
        <rasterrenderer band="1" nodataColor="" opacity="1" alphaBand="-1" type="singlebandpseudocolor" classificationMin="0" classificationMax="0.9">
          <rasterTransparency/>
          <minMaxOrigin>
            <limits>None</limits>
            <extent>WholeRaster</extent>
            <statAccuracy>Estimated</statAccuracy>
            <cumulativeCutLower>0.02</cumulativeCutLower>
            <cumulativeCutUpper>0.98</cumulativeCutUpper>
            <stdDevFactor>2</stdDevFactor>
          </minMaxOrigin>
          <rastershader>
            <colorrampshader maximumValue="0.9" minimumValue="0" colorRampType="INTERPOLATED" classificationMode="1" clip="0" labelPrecision="2">
              <colorramp name="[source]" type="gradient">
                <Option type="Map">
                  <Option name="color1" value="255,255,217,255" type="QString"/>
                  <Option name="color2" value="8,29,88,255" type="QString"/>
                  <Option name="discrete" value="0" type="QString"/>
                  <Option name="rampType" value="gradient" type="QString"/>
                  <Option name="stops" value="0.125;237,248,177,255:0.25;199,233,180,255:0.375;127,205,187,255:0.5;65,182,196,255:0.625;29,145,192,255:0.75;34,94,168,255:0.875;37,52,148,255" type="QString"/>
                </Option>
              </colorramp>
              <item value="0" color="#ffffd9" label="0.00" alpha="255"/>
              <item value="0.1125" color="#edf8b1" label="0.11" alpha="255"/>
              <item value="0.225" color="#c7e9b4" label="0.23" alpha="255"/>
              <item value="0.3375" color="#7fcdbb" label="0.34" alpha="255"/>
              <item value="0.45" color="#41b6c4" label="0.45" alpha="255"/>
              <item value="0.5625" color="#1d91c0" label="0.56" alpha="255"/>
              <item value="0.675" color="#225ea8" label="0.68" alpha="255"/>
              <item value="0.7875" color="#253494" label="0.79" alpha="255"/>
              <item value="0.9" color="#081d58" label="0.90" alpha="255"/>
            </colorrampshader>
          </rastershader>
        </rasterrenderer>
        <brightnesscontrast gamma="1" brightness="0" contrast="0"/>
        <huesaturation invertColors="0" colorizeRed="255" saturation="0" grayscaleMode="0" colorizeBlue="128" colorizeStrength="100" colorizeGreen="128" colorizeOn="0"/>
        <rasterresampler maxOversampling="2"/>
        <resamplingStage>resamplingFilter</resamplingStage>
      </pipe>"""


def stable_id(stem: str) -> str:
    """Deterministic layer id from a file stem, so re-runs reuse the same id (idempotent)."""
    h = hashlib.md5(stem.encode("utf-8")).hexdigest()[:32]
    parts = [h[0:8], h[8:12], h[12:16], h[16:20], h[20:32]]
    return f"{stem}_{'_'.join(parts)}"


def raster_extents(path: str):
    """Return (native_bounds, wgs84_bounds) for a raster."""
    with rasterio.open(path) as ds:
        b = ds.bounds
        wgs = transform_bounds(ds.crs, "EPSG:4326", *b, densify_pts=21)
    return b, wgs


def make_sm_maplayer(stem: str, layer_id: str, datasource: str, native, wgs) -> str:
    """Build a full <maplayer> for one SME2 soil-moisture raster with the shared pseudocolor pipe."""
    return f"""    <maplayer autoRefreshMode="Disabled" refreshOnNotifyEnabled="0" minScale="1e+08" autoRefreshTime="0" maxScale="0" refreshOnNotifyMessage="" styleCategories="AllStyleCategories" hasScaleBasedVisibilityFlag="0" legendPlaceholderImage="" type="raster">
      <extent>
        <xmin>{native.left!r}</xmin>
        <ymin>{native.bottom!r}</ymin>
        <xmax>{native.right!r}</xmax>
        <ymax>{native.top!r}</ymax>
      </extent>
      <wgs84extent>
        <xmin>{wgs[0]!r}</xmin>
        <ymin>{wgs[1]!r}</ymin>
        <xmax>{wgs[2]!r}</xmax>
        <ymax>{wgs[3]!r}</ymax>
      </wgs84extent>
      <id>{layer_id}</id>
      <datasource>{datasource}</datasource>
      <layername>{stem}</layername>
      <srs>
        <spatialrefsys nativeFormat="Wkt">
          <wkt>{EASE_WKT}</wkt>
          <proj4>{EASE_PROJ4}</proj4>
          <srsid>28242</srsid>
          <srid>6933</srid>
          <authid>EPSG:6933</authid>
          <description>WGS 84 / NSIDC EASE-Grid 2.0 Global</description>
          <projectionacronym>cea</projectionacronym>
          <ellipsoidacronym>EPSG:7030</ellipsoidacronym>
          <geographicflag>false</geographicflag>
        </spatialrefsys>
      </srs>
      <resourceMetadata>
        <identifier></identifier>
        <parentidentifier></parentidentifier>
        <language></language>
        <type>dataset</type>
        <title></title>
        <abstract></abstract>
        <links/>
        <dates/>
        <fees></fees>
        <encoding></encoding>
        <crs>
          <spatialrefsys nativeFormat="Wkt">
            <wkt>{EASE_WKT}</wkt>
            <proj4>{EASE_PROJ4}</proj4>
            <srsid>28242</srsid>
            <srid>6933</srid>
            <authid>EPSG:6933</authid>
            <description>WGS 84 / NSIDC EASE-Grid 2.0 Global</description>
            <projectionacronym>cea</projectionacronym>
            <ellipsoidacronym>EPSG:7030</ellipsoidacronym>
            <geographicflag>false</geographicflag>
          </spatialrefsys>
        </crs>
        <extent/>
      </resourceMetadata>
      <provider>gdal</provider>
      <noData>
        <noDataList bandNo="1" useSrcNoData="1"/>
      </noData>
      <map-layer-style-manager current="default">
        <map-layer-style name="default"/>
      </map-layer-style-manager>
      <flags>
        <Identifiable>1</Identifiable>
        <Removable>1</Removable>
        <Searchable>1</Searchable>
        <Private>0</Private>
      </flags>
      <temporal fetchMode="0" enabled="0" bandNumber="1" mode="0">
        <fixedRange>
          <start></start>
          <end></end>
        </fixedRange>
      </temporal>
      <elevation symbology="Line" band="1" zscale="1" enabled="0" mode="RepresentsElevationSurface" zoffset="0">
        <data-defined-properties>
          <Option type="Map">
            <Option name="name" value="" type="QString"/>
            <Option name="properties"/>
            <Option name="type" value="collection" type="QString"/>
          </Option>
        </data-defined-properties>
      </elevation>
      <customproperties>
        <Option type="Map">
          <Option name="identify/format" value="Value" type="QString"/>
        </Option>
      </customproperties>
      <mapTip enabled="1"></mapTip>
      <pipe-data-defined-properties>
        <Option type="Map">
          <Option name="name" value="" type="QString"/>
          <Option name="properties"/>
          <Option name="type" value="collection" type="QString"/>
        </Option>
      </pipe-data-defined-properties>
{SM_PIPE}
      <blendMode>0</blendMode>
    </maplayer>
"""


def make_footprint_maplayer(layer_id: str, datasource: str, extent) -> str:
    """Build a <maplayer> for the footprints: outline-only (SimpleFill style=no), EPSG:4326."""
    xmin, ymin, xmax, ymax = extent
    return f"""    <maplayer autoRefreshMode="Disabled" minScale="100000000" simplifyDrawingHints="1" refreshOnNotifyEnabled="0" labelsEnabled="0" simplifyLocal="1" simplifyAlgorithm="0" readOnly="0" geometry="Polygon" hasScaleBasedVisibilityFlag="0" legendPlaceholderImage="" refreshOnNotifyMessage="" styleCategories="AllStyleCategories" simplifyDrawingTol="1" symbologyReferenceScale="-1" maxScale="0" autoRefreshTime="0" type="vector" simplifyMaxScale="1" wkbType="Polygon">
      <extent>
        <xmin>{xmin!r}</xmin>
        <ymin>{ymin!r}</ymin>
        <xmax>{xmax!r}</xmax>
        <ymax>{ymax!r}</ymax>
      </extent>
      <wgs84extent>
        <xmin>{xmin!r}</xmin>
        <ymin>{ymin!r}</ymin>
        <xmax>{xmax!r}</xmax>
        <ymax>{ymax!r}</ymax>
      </wgs84extent>
      <id>{layer_id}</id>
      <datasource>{datasource}</datasource>
      <layername>{FOOTPRINT_NAME}</layername>
      <srs>
        <spatialrefsys nativeFormat="Wkt">
          <wkt>{WGS84_WKT}</wkt>
          <proj4>{WGS84_PROJ4}</proj4>
          <srsid>3452</srsid>
          <srid>4326</srid>
          <authid>EPSG:4326</authid>
          <description>WGS 84</description>
          <projectionacronym>longlat</projectionacronym>
          <ellipsoidacronym>EPSG:7030</ellipsoidacronym>
          <geographicflag>true</geographicflag>
        </spatialrefsys>
      </srs>
      <resourceMetadata>
        <identifier></identifier>
        <parentidentifier></parentidentifier>
        <language></language>
        <type>dataset</type>
        <title></title>
        <abstract></abstract>
        <links/>
        <dates/>
        <fees></fees>
        <encoding></encoding>
        <crs>
          <spatialrefsys nativeFormat="Wkt">
            <wkt>{WGS84_WKT}</wkt>
            <proj4>{WGS84_PROJ4}</proj4>
            <srsid>3452</srsid>
            <srid>4326</srid>
            <authid>EPSG:4326</authid>
            <description>WGS 84</description>
            <projectionacronym>longlat</projectionacronym>
            <ellipsoidacronym>EPSG:7030</ellipsoidacronym>
            <geographicflag>true</geographicflag>
          </spatialrefsys>
        </crs>
        <extent/>
      </resourceMetadata>
      <provider encoding="UTF-8">ogr</provider>
      <vectorjoins/>
      <layerDependencies/>
      <dataDependencies/>
      <expressionfields/>
      <map-layer-style-manager current="default">
        <map-layer-style name="default"/>
      </map-layer-style-manager>
      <auxiliaryLayer/>
      <flags>
        <Identifiable>1</Identifiable>
        <Removable>1</Removable>
        <Searchable>1</Searchable>
        <Private>0</Private>
      </flags>
      <temporal startExpression="" limitMode="0" startField="" endField="" fixedDuration="0" durationField="" enabled="0" durationUnit="min" mode="0" accumulate="0" endExpression="">
        <fixedRange>
          <start></start>
          <end></end>
        </fixedRange>
      </temporal>
      <elevation extrusionEnabled="0" extrusion="0" symbology="Line" zscale="1" showMarkerSymbolInSurfacePlots="0" binding="Centroid" respectLayerSymbol="1" customToleranceEnabled="0" zoffset="0" clamping="Terrain" type="IndividualFeatures">
        <data-defined-properties>
          <Option type="Map">
            <Option name="name" value="" type="QString"/>
            <Option name="properties"/>
            <Option name="type" value="collection" type="QString"/>
          </Option>
        </data-defined-properties>
      </elevation>
      <renderer-v2 symbollevels="0" forceraster="0" enableorderby="0" referencescale="-1" type="singleSymbol">
        <symbols>
          <symbol force_rhr="0" clip_to_extent="1" frame_rate="10" name="0" alpha="1" is_animated="0" type="fill">
            <data_defined_properties>
              <Option type="Map">
                <Option name="name" value="" type="QString"/>
                <Option name="properties"/>
                <Option name="type" value="collection" type="QString"/>
              </Option>
            </data_defined_properties>
            <layer class="SimpleFill" id="{{footprint-fill-0000-0000-000000000000}}" locked="0" enabled="1" pass="0">
              <Option type="Map">
                <Option name="border_width_map_unit_scale" value="3x:0,0,0,0,0,0" type="QString"/>
                <Option name="color" value="255,128,0,0,rgb:1,0.5019608,0,0" type="QString"/>
                <Option name="joinstyle" value="bevel" type="QString"/>
                <Option name="offset" value="0,0" type="QString"/>
                <Option name="offset_map_unit_scale" value="3x:0,0,0,0,0,0" type="QString"/>
                <Option name="offset_unit" value="MM" type="QString"/>
                <Option name="outline_color" value="255,127,0,255,rgb:1,0.4980392,0,1" type="QString"/>
                <Option name="outline_style" value="solid" type="QString"/>
                <Option name="outline_width" value="0.4" type="QString"/>
                <Option name="outline_width_unit" value="MM" type="QString"/>
                <Option name="style" value="no" type="QString"/>
              </Option>
              <data_defined_properties>
                <Option type="Map">
                  <Option name="name" value="" type="QString"/>
                  <Option name="properties"/>
                  <Option name="type" value="collection" type="QString"/>
                </Option>
              </data_defined_properties>
            </layer>
          </symbol>
        </symbols>
        <rotation/>
        <sizescale/>
        <data-defined-properties>
          <Option type="Map">
            <Option name="name" value="" type="QString"/>
            <Option name="properties"/>
            <Option name="type" value="collection" type="QString"/>
          </Option>
        </data-defined-properties>
      </renderer-v2>
      <selection mode="Default">
        <selectionColor invalid="1"/>
      </selection>
      <customproperties>
        <Option type="Map">
          <Option name="embeddedWidgets/count" value="0" type="int"/>
          <Option name="variableNames" type="invalid"/>
          <Option name="variableValues" type="invalid"/>
        </Option>
      </customproperties>
      <blendMode>0</blendMode>
      <featureBlendMode>0</featureBlendMode>
      <layerOpacity>1</layerOpacity>
      <geometryOptions geometryPrecision="0" removeDuplicateNodes="0">
        <activeChecks/>
        <checkConfiguration/>
      </geometryOptions>
      <legend showLabelLegend="0" type="default-vector"/>
      <referencedLayers/>
      <referencingLayers/>
      <fieldConfiguration/>
      <aliases/>
      <splitPolicies/>
      <duplicatePolicies/>
      <defaults/>
      <constraints/>
      <constraintExpressions/>
      <expressionfields/>
      <attributeactions/>
      <attributetableconfig actionWidgetStyle="dropDown" sortExpression="" sortOrder="0"/>
      <conditionalstyles>
        <rowstyles/>
        <fieldstyles/>
      </conditionalstyles>
      <storedexpressions/>
      <editform tolerant="1"></editform>
      <editforminit/>
      <editforminitcodesource>0</editforminitcodesource>
      <editforminitfilepath></editforminitfilepath>
      <editforminitcode></editforminitcode>
      <featformsuppress>0</featformsuppress>
      <editorlayout>generatedlayout</editorlayout>
      <editable/>
      <labelOnTop/>
      <reuseLastValue/>
      <dataDefinedFieldProperties/>
      <widgets/>
      <previewExpression></previewExpression>
      <mapTip enabled="1"></mapTip>
    </maplayer>
"""


def remove_managed(qgs: str) -> str:
    """Strip all previously-managed content so a fresh run does not accumulate duplicates."""
    qgs = _remove_group_from_tree(qgs, SME2_GROUP_NAME, SME2_DATASOURCE_RE)
    qgs = _remove_group_from_tree(qgs, ANC_GROUP_NAME, ANC_DATASOURCE_RE)
    qgs = _remove_footprint_tree_layer(qgs)
    qgs = _remove_managed_maplayers(qgs)
    qgs = _remove_managed_legend(qgs)
    qgs = _remove_managed_order_refs(qgs)
    return qgs


def _remove_block(text: str, open_re: re.Pattern, close_tag: str) -> str:
    """Remove the first element matching open_re through its matching close_tag (non-nested)."""
    m = open_re.search(text)
    while m:
        end = text.find(close_tag, m.end())
        if end == -1:
            break
        end += len(close_tag)
        # swallow trailing newline + indentation up to next tag start
        nl = text.find("\n", end)
        if nl != -1 and text[end:nl].strip() == "":
            end = nl + 1
        text = text[: m.start()] + text[end:]
        m = open_re.search(text)
    return text


def _remove_group_from_tree(qgs: str, group_name: str, source_re: re.Pattern) -> str:
    """Remove a managed <layer-tree-group name="..."> ... </layer-tree-group> and any stray
    top-level <layer-tree-layer> pointing at a datasource matching source_re (e.g. the original
    undated single SME2 layer, or layers a prior version placed outside the group).
    """
    grp_open = re.compile(
        r'[ \t]*<layer-tree-group[^>]*name="' + re.escape(group_name) + r'"[^>]*>'
    )
    qgs = _remove_block(qgs, grp_open, "</layer-tree-group>")

    # self-closing form (empty group) safety
    grp_selfclose = re.compile(
        r'[ \t]*<layer-tree-group[^>]*name="' + re.escape(group_name) + r'"[^>]*/>\n?'
    )
    qgs = grp_selfclose.sub("", qgs)

    # any stray layer-tree-layer entries (with or without child content)
    qgs = _remove_tree_layers_by_source(qgs, source_re)
    return qgs


def _remove_tree_layers_by_source(qgs: str, source_re: re.Pattern) -> str:
    """Remove <layer-tree-layer ... source="<matches>" ...> plus its optional children."""
    pattern = re.compile(r"[ \t]*<layer-tree-layer\b[^>]*?/?>")
    out = qgs
    while True:
        changed = False
        for m in pattern.finditer(out):
            tag = m.group(0)
            src_m = re.search(r'source="([^"]*)"', tag)
            if not src_m or not source_re.search(src_m.group(1)):
                continue
            if tag.rstrip().endswith("/>"):
                end = m.end()
                nl = out.find("\n", end)
                if nl != -1 and out[end:nl].strip() == "":
                    end = nl + 1
                out = out[: m.start()] + out[end:]
            else:
                close = "</layer-tree-layer>"
                end = out.find(close, m.end())
                if end == -1:
                    continue
                end += len(close)
                nl = out.find("\n", end)
                if nl != -1 and out[end:nl].strip() == "":
                    end = nl + 1
                out = out[: m.start()] + out[end:]
            changed = True
            break
        if not changed:
            return out


def _remove_footprint_tree_layer(qgs: str) -> str:
    src_re = re.compile(re.escape(FOOTPRINT_DATASOURCE))
    return _remove_tree_layers_by_source(qgs, src_re)


def _remove_managed_maplayers(qgs: str) -> str:
    """Remove <maplayer> blocks whose <datasource> is an SME2 tif, ancillary tif, or footprint fgb."""
    pattern = re.compile(r"[ \t]*<maplayer\b")
    out = qgs
    while True:
        changed = False
        for m in pattern.finditer(out):
            close = "</maplayer>"
            end = out.find(close, m.end())
            if end == -1:
                break
            end += len(close)
            block = out[m.start() : end]
            ds_m = re.search(r"<datasource>([^<]*)</datasource>", block)
            ds = ds_m.group(1) if ds_m else ""
            if (
                SME2_DATASOURCE_RE.search(ds)
                or ANC_DATASOURCE_RE.search(ds)
                or ds == FOOTPRINT_DATASOURCE
            ):
                nl = out.find("\n", end)
                if nl != -1 and out[end:nl].strip() == "":
                    end = nl + 1
                out = out[: m.start()] + out[end:]
                changed = True
                break
        if not changed:
            return out


def _remove_managed_legend(qgs: str) -> str:
    """Remove legend entries for managed layers: the SME2 / ancillary legendgroups and footprint."""
    for group_name in (SME2_GROUP_NAME, ANC_GROUP_NAME):
        grp_open = re.compile(
            r'[ \t]*<legendgroup[^>]*name="' + re.escape(group_name) + r'"[^>]*>'
        )
        qgs = _remove_block(qgs, grp_open, "</legendgroup>")
        grp_selfclose = re.compile(
            r'[ \t]*<legendgroup[^>]*name="' + re.escape(group_name) + r'"[^>]*/>\n?'
        )
        qgs = grp_selfclose.sub("", qgs)

    # footprint legendlayer + any stray SME2 legendlayer (matched by name / layerid)
    qgs = _remove_legendlayers_matching(qgs)
    return qgs


def _remove_legendlayers_matching(qgs: str) -> str:
    pattern = re.compile(r"[ \t]*<legendlayer\b[^>]*>")
    out = qgs
    while True:
        changed = False
        for m in pattern.finditer(out):
            close = "</legendlayer>"
            end = out.find(close, m.end())
            if end == -1:
                break
            end += len(close)
            block = out[m.start() : end]
            name_m = re.search(r'name="([^"]*)"', block)
            name = name_m.group(1) if name_m else ""
            is_footprint = name == FOOTPRINT_NAME
            is_undated = name == UNDATED_STEM
            is_sme2 = bool(re.fullmatch(r"sme2_(\w+_)?\d{8}_soil_moisture", name))
            if is_footprint or is_undated or is_sme2:
                nl = out.find("\n", end)
                if nl != -1 and out[end:nl].strip() == "":
                    end = nl + 1
                out = out[: m.start()] + out[end:]
                changed = True
                break
        if not changed:
            return out


def _remove_managed_order_refs(qgs: str) -> str:
    """Drop draw-order references that no longer resolve to a maplayer.

    Called after the managed maplayers have been stripped, so their <item>ID</item> (custom-order)
    and <layer id="ID"/> (layerorder) entries are now dangling and get pruned. References to the
    project's other (still-present) layers are kept. This is stem-agnostic, so it also cleans up
    ancillary layers whose stems are not known ahead of time.
    """
    valid_ids = set(re.findall(r"<id>([^<]+)</id>", qgs))

    def drop_items(text, open_lit, close_lit):
        pat = re.compile(r"[ \t]*" + open_lit + r"([^<]*)" + close_lit + r"\n?")
        return pat.sub(lambda mm: mm.group(0) if mm.group(1) in valid_ids else "", text)

    qgs = drop_items(qgs, r"<item>", r"</item>")

    # <layer id="..."/> lines in <layerorder>
    lo_pat = re.compile(r'[ \t]*<layer id="([^"]*)"/>\n?')
    qgs = lo_pat.sub(lambda mm: mm.group(0) if mm.group(1) in valid_ids else "", qgs)
    return qgs


def build_tree_group(group_name, entries) -> str:
    """Build a <layer-tree-group> of raster tree-layers (used for the SME2 and ancillary groups)."""
    lines = [
        f'    <layer-tree-group groupLayer="" name="{group_name}" expanded="1" checked="Qt::Checked">',
        "      <customproperties>",
        "        <Option/>",
        "      </customproperties>",
    ]
    for stem, layer_id, datasource in entries:
        lines.append(
            f'      <layer-tree-layer providerKey="gdal" source="{datasource}" '
            f'name="{stem}" id="{layer_id}" legend_exp="" legend_split_behavior="0" '
            f'expanded="1" checked="Qt::Checked" patch_size="-1,-1">'
        )
        lines.append("        <customproperties>")
        lines.append("          <Option/>")
        lines.append("        </customproperties>")
        lines.append("      </layer-tree-layer>")
    lines.append("    </layer-tree-group>")
    return "\n".join(lines) + "\n"


def build_footprint_tree_layer(layer_id: str, datasource: str) -> str:
    return (
        f'    <layer-tree-layer providerKey="ogr" source="{datasource}" '
        f'name="{FOOTPRINT_NAME}" id="{layer_id}" legend_exp="" legend_split_behavior="0" '
        f'expanded="0" checked="Qt::Checked" patch_size="-1,-1">\n'
        "      <customproperties>\n"
        "        <Option/>\n"
        "      </customproperties>\n"
        "    </layer-tree-layer>\n"
    )


def build_legend_group(group_name, entries) -> str:
    lines = [f'    <legendgroup name="{group_name}" open="true" checked="Qt::Checked">']
    for stem, layer_id, _ in entries:
        lines.append(
            f'      <legendlayer showFeatureCount="0" drawingOrder="-1" name="{stem}" open="false" checked="Qt::Checked">'
        )
        lines.append('        <filegroup open="false" hidden="false">')
        lines.append(
            f'          <legendlayerfile visible="1" isInOverview="0" layerid="{layer_id}"/>'
        )
        lines.append("        </filegroup>")
        lines.append("      </legendlayer>")
    lines.append("    </legendgroup>")
    return "\n".join(lines) + "\n"


def build_footprint_legend(layer_id: str) -> str:
    return (
        f'    <legendlayer showFeatureCount="0" drawingOrder="-1" name="{FOOTPRINT_NAME}" open="false" checked="Qt::Checked">\n'
        '      <filegroup open="false" hidden="false">\n'
        f'        <legendlayerfile visible="1" isInOverview="0" layerid="{layer_id}"/>\n'
        "      </filegroup>\n"
        "    </legendlayer>\n"
    )


def insert_before(text: str, anchor: str, payload: str) -> str:
    idx = text.find(anchor)
    if idx == -1:
        raise RuntimeError(f"anchor not found: {anchor!r}")
    return text[:idx] + payload + text[idx:]


def discover_sme2(tif_dir: str):
    """Return chronologically-sorted list of (stem, layer_id, datasource, path) for dated tifs.

    The undated sme2_soil_moisture.tif is skipped: it is byte-identical to the 2026-06-25 scene and
    must not become its own layer.
    """
    entries = []
    for path in glob.glob(os.path.join(tif_dir, "*_soil_moisture.tif")):
        stem = Path(path).stem
        if stem == UNDATED_STEM:
            continue
        dm = DATE_RE.search(stem)
        if not dm:
            print(f"  ! skipping (no YYYYMMDD in name): {stem}", file=sys.stderr)
            continue
        entries.append((dm.group(1), stem, path))
    entries.sort(key=lambda e: e[0])
    return [
        (stem, stable_id(stem), f"./nisar/tif/{Path(path).name}", path)
        for _, stem, path in entries
    ]


def discover_ancillary(anc_dir: str):
    """Return sorted (stem, layer_id, datasource, path) for ancillary SM rasters (e.g. SMAP).

    Any *.tif in the ancillary dir is treated as a comparison soil-moisture image and styled with
    the shared SM ramp. These are expected to be on EPSG:6933 like the SME2 rasters.
    """
    entries = []
    for path in sorted(glob.glob(os.path.join(anc_dir, "*.tif"))):
        stem = Path(path).stem
        entries.append(
            (stem, stable_id(stem), f"./nisar/ancillary/{Path(path).name}", path)
        )
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qgz", default="~/data/nisar.qgz")
    ap.add_argument("--tif-dir", default="~/data/nisar/tif")
    ap.add_argument(
        "--footprint", default="~/data/nisar/reference/nisar_frames_conus.fgb"
    )
    ap.add_argument("--ancillary-dir", default="~/data/nisar/ancillary")
    args = ap.parse_args()

    qgz = Path(os.path.expanduser(args.qgz))
    tif_dir = os.path.expanduser(args.tif_dir)
    footprint = Path(os.path.expanduser(args.footprint))
    anc_dir = os.path.expanduser(args.ancillary_dir)

    if not qgz.exists():
        sys.exit(f"qgz not found: {qgz}")

    # --- backup (never overwrite an existing backup) ---
    bak = qgz.with_suffix(qgz.suffix + ".bak")
    n = 2
    while bak.exists():
        bak = qgz.with_suffix(qgz.suffix + f".bak{n}")
        n += 1
    shutil.copy2(qgz, bak)
    print(f"backup: {bak}")

    # --- read the .qgz (zip of nisar.qgs + jGSAaD_styles.db) ---
    with zipfile.ZipFile(qgz) as z:
        names = z.namelist()
        qgs_name = next(nm for nm in names if nm.endswith(".qgs"))
        payload = {nm: z.read(nm) for nm in names}
    qgs = payload[qgs_name].decode("utf-8")

    # --- discover inputs ---
    sme2 = discover_sme2(tif_dir)
    if not sme2:
        sys.exit(f"no dated SME2 tifs found in {tif_dir}")
    print(f"discovered {len(sme2)} dated SME2 tifs")

    anc = discover_ancillary(anc_dir) if os.path.isdir(anc_dir) else []
    print(f"discovered {len(anc)} ancillary rasters in {anc_dir}")

    fp_id = stable_id("nisar_frames_conus")

    # footprint extent (fgb has no CRS; coords are lon/lat -> EPSG:4326)
    ogrinfo = os.path.expanduser("~/Applications/QGIS-LTR.app/Contents/MacOS/ogrinfo")
    fp_extent = _footprint_extent(footprint, ogrinfo)

    # --- strip prior managed content, then insert fresh ---
    qgs = remove_managed(qgs)

    # 1) tree groups (insert as first children of root layer-tree-group, before existing children)
    tree_blocks = build_tree_group(SME2_GROUP_NAME, [(s, i, d) for s, i, d, _ in sme2])
    if anc:
        tree_blocks += build_tree_group(
            ANC_GROUP_NAME, [(s, i, d) for s, i, d, _ in anc]
        )
    tree_blocks += build_footprint_tree_layer(fp_id, FOOTPRINT_DATASOURCE)
    anchor_tree = "    <layer-tree-layer providerKey="
    # place the managed groups + footprint layer right before the first existing tree layer
    qgs = insert_before(qgs, anchor_tree, tree_blocks)

    # 2) maplayers (insert just before </projectlayers>)
    sm_layers = "".join(
        make_sm_maplayer(stem, lid, ds, *raster_extents(path))
        for stem, lid, ds, path in sme2
    )
    anc_layers = "".join(
        make_sm_maplayer(stem, lid, ds, *raster_extents(path))
        for stem, lid, ds, path in anc
    )
    fp_layer = make_footprint_maplayer(fp_id, FOOTPRINT_DATASOURCE, fp_extent)
    qgs = insert_before(qgs, "  </projectlayers>", sm_layers + anc_layers + fp_layer)

    # 3) legend (insert before </legend>)
    legend = build_legend_group(SME2_GROUP_NAME, [(s, i, d) for s, i, d, _ in sme2])
    if anc:
        legend += build_legend_group(ANC_GROUP_NAME, [(s, i, d) for s, i, d, _ in anc])
    legend += build_footprint_legend(fp_id)
    qgs = insert_before(qgs, "  </legend>", legend)

    # 4) layerorder (insert before </layerorder>): footprint on top, then ancillary, then rasters
    lo = [f'    <layer id="{fp_id}"/>\n']
    lo += [f'    <layer id="{lid}"/>\n' for _, lid, _, _ in reversed(anc)]
    lo += [f'    <layer id="{lid}"/>\n' for _, lid, _, _ in reversed(sme2)]
    qgs = insert_before(qgs, "  </layerorder>", "".join(lo))

    # 5) custom-order (append items before </custom-order>) for full consistency
    co = [f"      <item>{fp_id}</item>\n"]
    co += [f"      <item>{lid}</item>\n" for _, lid, _, _ in anc]
    co += [f"      <item>{lid}</item>\n" for _, lid, _, _ in sme2]
    qgs = insert_before(qgs, "    </custom-order>", "".join(co))

    # --- ensure DOCTYPE is the first line ---
    if not qgs.lstrip().startswith("<!DOCTYPE"):
        qgs = DOCTYPE + "\n" + qgs

    # --- write repacked zip preserving jGSAaD_styles.db ---
    payload[qgs_name] = qgs.encode("utf-8")
    tmp = qgz.with_suffix(qgz.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for nm in names:
            z.writestr(nm, payload[nm])
    os.replace(tmp, qgz)
    print(f"wrote: {qgz}")
    print(
        f"  SME2 layers: {len(sme2)}  ancillary layers: {len(anc)}  "
        f"+ footprint: {FOOTPRINT_NAME}"
    )


def _footprint_extent(footprint: Path, ogrinfo: str):
    """Read the footprint extent via ogrinfo. Extent is cosmetic; QGIS recomputes it on load."""
    try:
        out = subprocess.run(
            [ogrinfo, "-so", "-al", str(footprint)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        m = re.search(
            r"Extent:\s*\(([-\d.]+),\s*([-\d.]+)\)\s*-\s*\(([-\d.]+),\s*([-\d.]+)\)",
            out,
        )
        if m:
            xmin, ymin, xmax, ymax = (float(g) for g in m.groups())
            return xmin, ymin, xmax, ymax
    except (OSError, ValueError) as exc:
        print(
            f"  ! could not read footprint extent ({exc}); using CONUS default",
            file=sys.stderr,
        )
    return -128.019119, 23.408210, -63.524069, 51.762307


if __name__ == "__main__":
    main()
