"""Render a LoD2.2 + terrain PROMETHEUS nadir plate for any configured city.

    python examples/city_lod2_pt.py zurich
    python examples/city_lod2_pt.py luxembourg --probe

Reuses osm_city_pt_3d wholesale (light-free ground palette, PT light field,
modulation) and swaps the heightfield, the building colour and the poster
furniture. Square canvas so the disc fills the sheet.

Three things this encodes that the first cut of these plates got wrong:

1. NORTH IS UP. osm_city_demo.localize_feature rotates every feature by
   SCENE_ROTATION_DEG = -180, so the original plates were SOUTH-UP - north
   pointed down and east pointed left. That is invisible until you try to put
   an honest north arrow on the sheet. These plates cancel the rotation and
   keep the terrain, the roof DSM and the overlay in that one frame.

2. NO SILENT COVERAGE HOLES. A cell with no LoD2 roof used to fall to height
   zero while the overlay still painted a building there, so a missing map
   sheet read as a district of flat buildings rather than as missing data.
   Prague was missing the sheet containing its own centre and Lyon was missing
   most of its communes. Coverage is now completed from source AND backstopped
   by OSM heights, with the fallback share printed every run.

3. HEIGHT IS READABLE. Roofs are classed by true metre height from the same
   DSM the tracer lit (see city_lod2_plate), not by an OSM tag bin.

V8 is the current default look: V7's geometry with the shadow grade retuned
compose-side (shadow_floor/gain 0.72/0.42, cool_shadow 0.14, SHADE_SOFTEN_PX
1.0 - each value measured in the 2026-07-31 sweep; see the constants). V7
plates are kept as {city}_pt_v7.png.

V7 was one coupled change, not five:
TRUE VERTICAL SCALE (1.00x, WORLD_UNITS_PER_M 0.025) with the wall reveal held
at 0.348 by PERSPECTIVE_FOV_DEG 38.4, the sun lifted to 60 deg so the shadows
stop being 4.5x a building's height, the terrain held at its old world scale
via TERRAIN_SCALE_COMPENSATION, and the edge stroke cut to 1 px now that the
walls - not a black rule - separate the blocks. Each constant carries the
measurement that set it; the fov and the exaggeration must be re-solved
together, since reveal/height = exaggeration * tan(fov/2).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from PIL.PngImagePlugin import PngInfo
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import city_lod2_common as C  # noqa: E402
import city_lod2_perspective as persp  # noqa: E402
import city_lod2_plate as plate  # noqa: E402
import osm_city_pt_3d as pt  # noqa: E402

ROOT = Path("C:/tmp/citylod2")
LYON_GML = ROOT / "lyon" / "citygml" / "gml"

# Grand Lyon publishes one archive per commune, and the axis-order probe runs on
# the FIRST path, so the list is ordered by how much of the disc each commune
# covers - probing on a commune that barely clips the disc is a needless risk.
LYON_COMMUNES = ("LYON_5EME", "LYON_7EME", "LYON_2EME", "LYON_3EME", "LYON_1ER",
                 "LYON_6EME", "LYON_9EME", "LYON_4EME", "LA_MULATIERE",
                 "STE_FOY_LES_LYON")

OSM_CREDIT = "Base map \u00a9 OpenStreetMap contributors"
AUTHOR = "\u00a9 2026 Milos Popovic  \u00b7  milosgis.nl"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "docs" / "city-lod2-launch-manifest.json"


def _load_launch_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        raise RuntimeError(f"missing authoritative city-poster manifest: {MANIFEST_PATH}")
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    designs = data.get("designs", [])
    if data.get("design_count") != len(designs) or data.get("sku_count") != len(designs):
        raise ValueError("city-poster manifest counts do not match its design records")
    if len({d.get("sku") for d in designs}) != len(designs):
        raise ValueError("city-poster manifest contains duplicate SKUs")
    return data


LAUNCH_MANIFEST = _load_launch_manifest()
LAUNCH_RECORDS = {
    (item["config_key"], item["edition"]): item for item in LAUNCH_MANIFEST["designs"]
}


def launch_record(city_key: str, edition: str) -> dict | None:
    return LAUNCH_RECORDS.get((city_key, edition))


def _validate_canonical_launch_args(a: argparse.Namespace, record: dict | None) -> None:
    """Prevent an untagged parameter sweep from masquerading as a SKU."""
    if record is None or a.tag or a.probe:
        return
    expected_fov = float(record["camera"]["fov_deg"])
    expected_light = LAUNCH_MANIFEST["city_lighting"][record["city"]]
    errors: list[str] = []
    if tuple(a.size) != (7200, 7200):
        errors.append("size must be 7200x7200")
    if (int(a.grid_max), int(a.frame), int(a.tiles)) != (4096, 900, 4):
        errors.append("grid-max/frame/tiles must be 4096/900/4")
    if a.fov is not None and abs(float(a.fov) - expected_fov) > 1e-6:
        errors.append(f"fov must be {expected_fov:g}")
    if a.sun_azimuth is not None and abs(float(a.sun_azimuth) - float(expected_light["azimuth_deg"])) > 1e-6:
        errors.append("sun azimuth differs from manifest")
    if a.sun_elevation is not None and abs(float(a.sun_elevation) - float(expected_light["elevation_deg"])) > 1e-6:
        errors.append("sun elevation differs from manifest")
    if a.terrain_scale is not None or a.relief is not None or a.exaggeration is not None:
        errors.append("terrain/vertical overrides require --tag")
    if a.shade_soften is not None and abs(float(a.shade_soften) - SHADE_SOFTEN_PX) > 1e-6:
        errors.append("shade-soften differs from launch setting")
    if a.edge_stroke is not None and int(a.edge_stroke) != EDGE_PX_DEFAULT:
        errors.append("edge-stroke differs from launch setting")
    if a.no_road_casing or a.legend:
        errors.append("road-casing/legend overrides require --tag")
    expected_ortho_shade = float(LAUNCH_MANIFEST["orthophoto_transform"]["pt_modulation"]["ortho_shade"])
    if a.ortho_shade is not None and abs(float(a.ortho_shade) - expected_ortho_shade) > 1e-6:
        errors.append("ortho-shade differs from launch setting")
    if a.env_intensity is not None or a.sun_intensity is not None:
        errors.append("light-intensity overrides require --tag")
    if errors:
        raise SystemExit("canonical launch output would drift: " + "; ".join(errors))

CITIES = {
    "luxembourg": dict(
        title="Luxembourg", lon=6.13000, lat=49.61160, src_epsg="EPSG:2169",
        gml=[ROOT / "luxembourg" / "luxembourg.gml"],
        rings=ROOT / "luxembourg" / "lux_disc2km.npz",
        dem_url=("/vsicurl/https://download.data.public.lu/resources/"
                 "bd-l-lidar2024-releve-3d-du-territoire-luxembourgeois/"
                 "20241223-093912/MNT_Lidar2024.tif"),
        dem_tiles=None, terrain_scale=0.35, relief=4.5,
        sources=f"Buildings BD-L-BATI3D \u00b7 Terrain LiDAR 2024 MNT, ACT Luxembourg \u00b7 {OSM_CREDIT}",
        # Official ACT orthophoto. The service speaks EPSG:25832 (ETRS89/UTM32N),
        # the same projection as the scene's EPSG:32632 to well under a pixel.
        ortho=dict(url="https://wms.geoportail.lu/opendata/service",
                   layer="ortho_2025", crs="EPSG:25832")),
    "zurich": dict(
        # WARNING: the CityGML under C:/tmp/citylod2/zurich/citygml no longer
        # exists, so this glob is EMPTY and the city only builds because
        # C.extract short-circuits on the cached zurich_disc2000.npz. Lose that
        # npz and Zurich cannot be re-extracted - which is exactly how Lyon's
        # source was lost. Re-download from the Stadt Zurich open-data portal
        # (Gebaeudemodell LoD2) before relying on a rebuild.
        title="Zurich", lon=8.5410, lat=47.3690, src_epsg="EPSG:2056",
        gml=sorted((ROOT / "zurich" / "citygml").glob("*.gml")),
        dem_url=None, dem_tiles=sorted((ROOT / "zurich" / "alti").glob("*.tif")),
        # 184 m of relief here vs Luxembourg's 121, so damp terrain harder or it
        # eats the vertical budget and flattens the roofs.
        terrain_scale=0.26, relief=4.5,
        sources=f"Buildings LoD2 Stadt Z\u00fcrich (GeoZ) \u00b7 Terrain swissALTI3D, swisstopo \u00b7 {OSM_CREDIT}",
        # SWISSIMAGE via the federal WMS, which speaks the scene's EPSG:32632
        # directly (probed 2026-07-31: real imagery, not the blank-white frame).
        ortho=dict(url="https://wms.geo.admin.ch/",
                   layer="ch.swisstopo.swissimage", crs="EPSG:32632")),
    "lyon": dict(
        # EPSG:3946 (RGF93 / CC46), NOT Lambert-93 - the GML's own srsName.
        title="Lyon", lon=4.8320, lat=45.7578, src_epsg="EPSG:3946",
        gml=[p for name in LYON_COMMUNES
             for p in (LYON_GML / f"{name}_BATI_2012.gml",
                       LYON_GML / f"{name}_BATI_REMARQUABLE_2012.gml") if p.exists()],
        dem_url=None, dem_tiles=None,
        # IGN serves RGE ALTI over WMS as BIL32 and supports EPSG:32631, so the
        # DEM arrives already in the scene's frame - no warp, no reprojection.
        dem_wms=dict(url="https://data.geopf.fr/wms-r/wms",
                     layer="ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"),
        terrain_scale=0.32, relief=4.5,
        sources=f"Buildings CityGML 2012, M\u00e9tropole de Lyon \u00b7 Terrain RGE ALTI, IGN \u00b7 {OSM_CREDIT}",
        # BD ORTHO HR (20 cm) from the same Geoplateforme WMS as the DEM; it
        # supports the scene's EPSG:32631 natively.
        ortho=dict(url="https://data.geopf.fr/wms-r/wms",
                   layer="HR.ORTHOIMAGERY.ORTHOPHOTOS", crs="EPSG:32631")),
    "prague": dict(
        # No CityGML from Prague - PolygonZ shapefiles, pre-extracted by
        # examples/prague_shp_extract.py into the shared rings npz format.
        title="Prague", lon=14.4205, lat=50.0875, src_epsg="EPSG:5514",
        gml=[], dem_url=None, dem_tiles=None,
        rings=ROOT / "prague" / "prague_disc2000.npz",
        dem_arcgis=dict(url="https://ags.cuzk.cz/arcgis2/rest/services/dmr5g/ImageServer"),
        terrain_scale=0.34, relief=4.5,
        # CUZK = \u010c\u00daZK. Note \u00da (Latin U-acute), NOT \u0417, which is
        # Cyrillic Ze and renders the national mapping agency as "CU3K".
        sources=f"Buildings 3D model, IPR Praha \u00b7 Terrain DMR 5G, \u010c\u00daZK \u00b7 {OSM_CREDIT}",
        # National \u010c\u00daZK ortofoto. NOT the IPR Praha city WMS
        # (gs-pub.praha.eu), which only speaks S-JTSK (EPSG:5514) - wms_ortho
        # requests the bbox in scene coordinates with no warp, so the service
        # must speak EPSG:32633; arcgis1/ORTOFOTO does (layer "0").
        ortho=dict(url="https://ags.cuzk.gov.cz/arcgis1/services/ORTOFOTO/MapServer/WMSServer",
                   layer="0", crs="EPSG:32633")),
    "sanfrancisco": dict(
        # The only city here whose roofs are RECONSTRUCTED rather than surveyed:
        # no US city publishes LoD2, so roofer builds it from 2023 USGS LiDAR
        # (~44 pts/m2) against DataSF footprints. EPSG:7131 = San Francisco CS13.
        title="San Francisco", lon=-122.4100, lat=37.7950, src_epsg="EPSG:7131",
        gml=[], dem_url=None, dem_tiles=None,
        rings=ROOT / "sanfrancisco" / "sf_disc2000.npz",
        dem_arcgis=dict(url="https://elevation.nationalmap.gov/arcgis/rest/"
                            "services/3DEPElevation/ImageServer"),
        terrain_scale=0.30, relief=4.5,
        sources=f"Buildings reconstructed from USGS 2023 LiDAR \u00b7 Terrain 3DEP, USGS \u00b7 {OSM_CREDIT}"),
    "amsterdam": dict(
        # 3D BAG LoD2.2 CityJSON, the same surveyed-roof-plane class of source as
        # Zurich and Luxembourg. Authored in EPSG:7415 (RD New + NAP); the
        # HORIZONTAL half of that compound code is EPSG:28992, which is what the
        # ring transform needs. z is NAP absolute, cross-checked against the DEM
        # over 841 buildings (median offset +0.15 m, no negative heights).
        title="Amsterdam", lon=4.8952, lat=52.3702, src_epsg="EPSG:28992",
        gml=[], rings=ROOT / "amsterdam" / "amsterdam_disc2000.npz",
        dem_url=None,
        # AHN4 DTM 0.5 m. The PDOK AHN *WMS* cannot be used as dem_wms - it
        # serves no image/x-bil;bits=32, only PNG/JPEG - so the coverage was
        # pulled via WCS and mosaicked to one local GeoTIFF instead. The raw DTM
        # is only 48% valid (nodata under every roof); the holes are nearest-
        # filled, otherwise terrain_grid's nanmedian would flatten the city.
        dem_tiles=[ROOT / "amsterdam" / "dem" / "ahn4_dtm_05m_amsterdam_rd.tif"],
        terrain_scale=0.30, relief=4.5,
        sources=f"Buildings 3D BAG (TU Delft 3D geoinformation) \u00b7 Terrain AHN4, PDOK \u00b7 {OSM_CREDIT}",
        # PDOK Luchtfoto RGB. The service does NOT support EPSG:32631
        # (LayerNotDefined); EPSG:25831 is ETRS89-UTM31 and differs from the
        # scene's WGS84-UTM31 by 1.3e-4 m here, far under a pixel. The year is
        # pinned: "Actueel"/2025 is a leaf-OFF flight (probe std 40.4) while
        # 2024 is leaf-on (std 46.2), and the plates want foliage.
        ortho=dict(url="https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0",
                   layer="2024_orthoHR", crs="EPSG:25831")),
    "paris": dict(
        # NOT LoD2. Paris publishes no openly-licensed roof-plane model: IGN's
        # batiment3d.ign.fr LoD2 viewer is 3DTiles with no export and no stated
        # licence, and the WFS GetCapabilities carries no lod/bati3d layer at
        # all. BD TOPO is explicitly a "boite a chaussures" - every vertex of a
        # building gets the SAME roof altitude - so what ships here is a prism
        # model whose roof height was re-measured from the IGN LiDAR HD MNS
        # (p90 inside each footprint), the same reconstruct-from-LiDAR class as
        # San Francisco. Cross-check: the LiDAR p90 sits +2.62 m above BD TOPO's
        # eaves, against an independently measured ridge-eaves of +2.30 m.
        # BD TOPO also carries a -1000 "altitude unknown" sentinel on 6.0% of
        # in-disc rings, which a max-z DSM would have swallowed silently.
        title="Paris", lon=2.3480, lat=48.8560, src_epsg="EPSG:2154",
        gml=[], rings=ROOT / "paris" / "paris_disc2000.npz",
        dem_url=None, dem_tiles=None,
        # Same Geoplateforme WMS as Lyon, re-probed for this bbox: BIL32, exactly
        # 262144 float32, zero nodata. Relief 36.2 m (Montagne Sainte-Genevieve;
        # Montmartre falls outside the disc).
        dem_wms=dict(url="https://data.geopf.fr/wms-r/wms",
                     layer="ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"),
        terrain_scale=0.30, relief=4.5,
        sources=f"Buildings BD TOPO® + LiDAR HD, IGN · Terrain RGE ALTI, IGN · {OSM_CREDIT}",
        ortho=dict(url="https://data.geopf.fr/wms-r/wms",
                   layer="HR.ORTHOIMAGERY.ORTHOPHOTOS", crs="EPSG:32631")),
    "paris_eiffel": dict(
        # The Eiffel-centred companion to "paris". Same sources and the same
        # BD TOPO + LiDAR HD ingest, re-centred 3.9 km west so the tower is in
        # the disc at all; rings built by
        # tools/paris_eiffel_day/build_eiffel_lod2.py.
        #
        # Cross-check against the independently built Notre-Dame disc: LiDAR p90
        # sits +2.51 m above BD TOPO's eaves here vs +2.62 m there, and the
        # fabric measures median 21.0 m / p95 29.9 m above terrain (Haussmann).
        #
        # These are PRISM heights, LiDAR-surveyed -- not roof planes. Paris has
        # no openly-licensed LoD2; see C:/tmp/citylod2/paris/RECON.md.
        #
        # The tower reaches 286.5 m above terrain only because of the landmark
        # rule in the builder: its authored altitude_minimale_toit is the FIRST
        # PLATFORM (~58 m) and a LiDAR p90 inside an open lattice measures the
        # park below, so both ordinary paths put it at a few tens of metres.
        title="Paris", lon=2.2945, lat=48.8584, src_epsg="EPSG:2154",
        gml=[], rings=Path("C:/tmp/citylod2/paris_eiffel/paris_eiffel_disc2000.npz"),
        dem_url=None, dem_tiles=None,
        dem_wms=dict(url="https://data.geopf.fr/wms-r/wms",
                     layer="ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"),
        # Flatter than the Notre-Dame disc (no Sainte-Genevieve here), so the
        # terrain can take a little more scale without eating the roof budget.
        terrain_scale=0.34, relief=4.5,
        sources=f"Buildings BD TOPO® + LiDAR HD, IGN · Terrain RGE ALTI, IGN · {OSM_CREDIT}",
        ortho=dict(url="https://data.geopf.fr/wms-r/wms",
                   layer="HR.ORTHOIMAGERY.ORTHOPHOTOS", crs="EPSG:32631")),
    "london": dict(
        # No open LoD2 exists for London, and every ready-made 3D model with
        # absolute roof heights is closed or non-commercial (OS MasterMap
        # Building Height Attribute is premium, City of London Tall Buildings is
        # INSPIRE non-commercial, GlobalBuildingAtlas is CC BY-NC). So roofs are
        # RECONSTRUCTED like San Francisco's: the Environment Agency 2022 LIDAR
        # Composite first-return DSM plane-fitted inside each OSM footprint.
        # Validated against OSM's own height tags: r = 0.942, median bias -0.1 m.
        # Rings are stored already in EPSG:32630 so they register to the OSM base
        # map by construction rather than through an OSTN15 datum shift.
        title="London", lon=-0.1180, lat=51.5080, src_epsg="EPSG:32630",
        gml=[], rings=ROOT / "london" / "london_disc2000.npz",
        # The EA GeoTIFFs ship nodata = -3.4e38 (float32 -FLT_MAX), which
        # rasterio.merge refuses and then returns an ALL-ZERO mosaic for, with no
        # exception - London would have rendered dead flat and said nothing. The
        # tiles here are re-tagged to -9999.0; originals kept in raw/.
        dem_url=None, dem_tiles=sorted((ROOT / "london" / "dem").glob("*.tif")),
        terrain_scale=0.30, relief=4.5,
        sources=f"Buildings + terrain from Environment Agency 2022 LiDAR (OGL v3) · {OSM_CREDIT}"),
        # NO `ortho` KEY ON PURPOSE. Every high-resolution London orthophoto
        # traces back to APGB / Getmapping / Bluesky or OS MasterMap Imagery, all
        # of which forbid third-party commercial reproduction, and these plates
        # are sold. The EA Vertical Aerial Photography WMS serves only tile-index
        # POLYGONS, not pixels (probed: blank white, std 0.0000, and byte-
        # identical output for 32630 and 27700 bboxes, so it does not even honour
        # the CRS). London therefore ships palette-only.
    "barcelona": dict(
        # LOD1, not LoD2: CartoBCN's own metadata says "LOD1 de teulades planes".
        # But the extrusion runs onto the municipal Digital Surface Model, so
        # every flat top is a SURVEYED absolute elevation (0.025 m altimetric),
        # and the cartography is finely subdivided - 114715 separate volumes in
        # the disc, median roof 17 m2 - so rooftop bodies and set-back storeys
        # are modelled individually. Barcelona's roofs really are flat, so this
        # is close to physically correct rather than a simplification.
        #
        # src_epsg is EPSG:3857: the I3S/SLPK store is authored in Web Mercator.
        # Only X and Y are inflated (1.333x at this latitude); Z is a true
        # gravity-related height (vcsWkid 5782, Alicante), which is why the disc
        # had to be filtered in EPSG:32631 rather than in the source frame.
        title="Barcelona", lon=2.1734, lat=41.3851, src_epsg="EPSG:3857",
        gml=[], dem_url=None,
        rings=ROOT / "barcelona" / "barcelona_disc2000.npz",
        # PNOA MDT02 (2 m), one tile, zero nodata over the window. 141 m of
        # relief - Montjuic's north flank against the port basin - which is the
        # second most of any city here after Zurich, hence the damped scale.
        dem_tiles=sorted((ROOT / "barcelona" / "dem").glob("*.tif")),
        terrain_scale=0.32, relief=4.5,
        sources=f"Buildings 3D city model, Ajuntament de Barcelona · Terrain PNOA MDT02, IGN · {OSM_CREDIT}",
        ortho=dict(url="https://geoserveis.icgc.cat/servei/catalunya/orto-territorial/wms",
                   layer="ortofoto_color_vigent", crs="EPSG:32631")),
    "berlin": dict(
        # True surveyed LoD2, the strongest source in the series alongside
        # Zurich. CityGML 1.0 (not 2.0), EPSG:25833, axis order probed as (E,N)
        # normal - 2.7 km from centre direct vs 7676.7 km swapped.
        #
        # gml= is kept populated even though rings= short-circuits it, so the
        # model can be re-extracted; Zurich is the cautionary tale of an npz
        # whose source was lost.
        title="Berlin", lon=13.4050, lat=52.5200, src_epsg="EPSG:25833",
        gml=sorted((ROOT / "berlin" / "citygml").glob("*.gml")),
        rings=ROOT / "berlin" / "berlin_disc2000.npz",
        # ATKIS DGM1, converted from ASCII XYZ locally: the dgm1 WMS offers no
        # image/x-bil;bits=32, so dem_wms is not available here.
        dem_url=None, dem_tiles=sorted((ROOT / "berlin" / "dem").glob("*.tif")),
        # 0.30, NOT the 0.40 the relief figure alone suggests. Raw relief is
        # 55.7 m but p1-p99 is 21.8 m: Mitte is genuinely flat and the maximum is
        # the artificial Bunkerberg clipping the disc's east edge. terrain_scale
        # is a DAMPING knob - it exists so tall terrain cannot eat the roofs'
        # vertical budget - so pushing it above the family maximum to make a flat
        # city read hillier would invert its purpose and break comparability
        # with the other plates, which is the whole point of shared breaks.
        terrain_scale=0.30, relief=4.5,
        sources=f"Buildings LoD2, Geoportal Berlin · Terrain ATKIS DGM1, Geoportal Berlin · {OSM_CREDIT}",
        # TrueDOP20 summer 2025 - a TRUE ortho, so roofs sit on their footprints
        # with no building lean, which matters more here than anywhere else
        # because the plate modulates imagery by a nadir light field.
        ortho=dict(url="https://gdi.berlin.de/services/wms/truedop_2025_sommer",
                   layer="truedop_2025_sommer_rgb", crs="EPSG:25833")),
    "warsaw": dict(
        # True surveyed LoD2 from the national GUGiK model (2017 vintage - newer
        # towers may be missing while the orthophoto is current).
        #
        # gml= is EMPTY and must stay empty: GUGiK writes every vertex as its own
        # <gml:pos>, and city_lod2_common._rings reads only <gml:posList>
        # (measured on this data: 0 posList, 222835 pos). Handing the .gml to the
        # streamer fails loudly with "no posList found to test axis order". The
        # npz was produced by the pipeline's own extract() with just the ring
        # READER swapped, so its layout is exactly what extract would have
        # written. Axis order probed as (E,N) normal despite EPSG:2180's
        # authority order being (N,E): 489 m from centre direct vs 211 km swapped.
        title="Warsaw", lon=21.0122, lat=52.2297, src_epsg="EPSG:2180",
        gml=[], rings=ROOT / "warsaw" / "warsaw_disc2000.npz",
        # GUGiK NMT 1 m via WCS. The service declares axisLabels="y x" but reads
        # y=northing, x=easting; getting it backwards returns HTTP 200 and a
        # valid GeoTIFF of the WRONG PART OF POLAND, never an error.
        dem_url=None, dem_tiles=sorted((ROOT / "warsaw" / "dem").glob("*.tif")),
        terrain_scale=0.30, relief=4.5,
        sources=f"Buildings LoD2 · Terrain NMT 1 m — materiały PZGiK, GUGiK (CC BY 4.0) · {OSM_CREDIT}",
        # No ortho service ADVERTISES EPSG:32634 (High resolution advertises only
        # CRS:84/2180/4326), so the working response was not taken on trust: the
        # same patch requested in 32634 vs 2180 differs by a best-fit -1.50deg
        # against the -1.581deg of meridian convergence the two projections
        # predict, correlation 0.980. A service silently ignoring the CRS cannot
        # reproduce its own convergence angle. Both endpoints 404 intermittently
        # - retry before blaming the CRS.
        ortho=dict(url="https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/HighResolution",
                   layer="Raster", crs="EPSG:32634")),
    # ROME IS DELIBERATELY ABSENT. It has no open building model at all: Roma
    # Capitale's CKAN carries no building cartography, Regione Lazio's DBGT omits
    # the EDIFC class entirely (its only 3D classes are canopies, median ring
    # 47 m2), and the PST-A 1 m LiDAR covers 54.4% of the disc and serves no
    # invertible float. That leaves only the OSM backstop, and just 18.0% of the
    # disc's 10390 OSM buildings carry height or building:levels - the rest would
    # take infer_building_height's flat 12.0 m, piling the plate into one height
    # class and breaking the shared-breaks comparison the series exists for.
    # A no-ortho entry was also ruled out: the Regione Lazio AGEA imagery probes
    # clean (EPSG:32633, std 43.0) but AGEA owns the pixels and the licence chain
    # is contradictory across regions, which fails the commercial-print gate.
    # Evidence: C:/tmp/citylod2/rome/RECON.md. No entry, so no one can render a
    # plate the data cannot support.
}

# With the scene un-rotated (north up) the light compass turns 180 degrees
# relative to the old south-up plates, so the azimuth that used to give SE
# screen shadows no longer does. Derived with --probe, never eyeballed.
SUN_AZIMUTH_NORTH_UP = 0.0

# Sun elevation above the horizon, degrees. V7 default.
#
# The plates shipped at osm_city_pt_3d's module default of 26 deg, whose cast
# shadow is cot(26) = 2.05x a building's height BEFORE the vertical
# exaggeration multiplies it. At the old 2.20x that was 4.5x the true height -
# a five-storey block threw a shadow the length of its own street, which is
# what pinned the plate's hue cue backwards (docs/city-lod2-3d-parity-
# hypotheses.md P2/P3: shortening the shadows is the only real remedy, since
# -0.214 of the inverted cue is already in the raw light-free overlay and the
# grade cannot reach it).
#
# 60 deg gives cot(60) = 0.58x, and at the new 1.00x exaggeration that is 0.58x
# the true height - a shadow that reads as a shadow instead of as a second,
# darker city. Changes the LIGHT FIELD, so it needs a full trace; --reuse-shade
# will silently keep whatever elevation the cached field was traced at.
SUN_ELEVATION_DEG = 60.0

# World units of relief per METRE of heightfield.
#
# _normalize_heightfield divides the field by its own max before the tracer
# applies RELIEF_WORLD, so a fixed RELIEF_WORLD makes the vertical scale a
# function of the single tallest thing in the disc: completing Prague's missing
# map sheet lifted its max from 83.9 m to 115.9 m, which would have SHORTENED
# every shadow on the plate by 28% - the opposite of the intended fix - purely
# because one taller building arrived. Setting RELIEF_WORLD = K * hf.max()
# cancels the normalisation, so a metre of building is the same number of world
# units in every city and on every run. K is the value the three original
# plates were effectively drawn at (4.5 / ~82 m), so the look is preserved.
#
# V7: 0.055 -> 0.025, i.e. 2.20x -> 1.00x vertical exaggeration (m_per_wu is 40
# at the shipped 2 km radius / SPAN_X 100, so exaggeration = 0.025 * 40 = 1.00).
# TRUE SCALE. The 2.20x was inherited from the very first plate and it is what
# made the shadows 2.2x too long, the roof-to-roof height steps 2.2x too coarse
# and the whole disc read as a relief model rather than as a city.
#
# The wall reveal is held CONSTANT across this change, because reveal/height =
# exaggeration * tan(fov/2) and the fov moves the opposite way: the old
# 2.20 * tan(18/2) = 0.348 is reproduced exactly by 1.00 * tan(38.4/2) = 0.348.
# See PERSPECTIVE_FOV_DEG. Change one of the two and you must re-solve the
# other or the plates stop being comparable with everything measured so far.
WORLD_UNITS_PER_M = 0.025

# Terrain is NOT rescaled with the buildings. The old plates' terrain was
# terrain_scale * 2.20 metres of world per metre of ground (Prague 0.34 * 2.20
# = 0.748), and that reading was right - the Vltava valley and Zurich's moraine
# ridges are what place these cities. Dropping the buildings to 1.00x without
# this factor would have flattened the terrain by the same 2.20x and taken the
# landform out with the exaggeration.
#
# This works because heightfield() applies tscale to the TERRAIN only, and
# _normalize_heightfield's range cancels against RELIEF_WORLD (see above), so
# per-metre world scale is exactly WORLD_UNITS_PER_M for buildings and
# WORLD_UNITS_PER_M * tscale for terrain. The two separate cleanly.
#
# Applied to each city's own shipped terrain_scale, so every city keeps the
# terrain it was tuned with rather than adopting Prague's.
TERRAIN_SCALE_COMPENSATION = 2.20

# --- shadow softness (both compose-side, so free with --reuse-shade) ---------
# The plate's shadow reads come from `scale = TERRAIN_FLOOR + TERRAIN_GAIN*shade`
# and are darken-only. Lifting the FLOOR (and dropping GAIN to keep floor+gain
# at 1.0) makes the deepest street shadow 0.70 of the overlay colour instead of
# 0.58 - the same shadow shapes, carrying less weight.
SHADOW_FLOOR = 0.70
SHADOW_GAIN = 0.30
# Gaussian blur on the resampled light field, in overlay pixels. At the 8192
# overlay over 4000 m that is 0.49 m/px, so 8 px was about 4 m of penumbra - it
# softens the shadow EDGE, which the floor alone cannot do. This is a look
# control applied after the trace, not more physical sky fill; a genuinely
# larger penumbra would need a re-trace at higher ENV_INTENSITY.
#
# Cut 8.0 -> 2.5 (about 1.2 m) for the perspective plates. 4 m of blur is wide
# enough to erase the contact darkening at the foot of a wall, which is exactly
# the cue the walls now need to sit ON the ground rather than float over it, and
# it is also what the Blender/PLATEAU reference shows crisply. The blur existed
# only because ENV_INTENSITY was refuted as the edge-softness lever; with the
# walls carrying the form, the shade no longer has to fake softness.
#
# Cut again 2.5 -> 1.0 (V8): the sweep showed 2.5 was still eating real shadow
# DEPTH, not just edges - at 1.0 the p5-p95 spread rises ~3 luma and gradient
# energy ~4% on all three cities. 5.0 was REFUTED (wall-foot contact washes
# out, worst in Zurich's fine-grained fabric).
SHADE_SOFTEN_PX = 1.0

# --- nadir PERSPECTIVE re-projection (docs/city-lod2-3d-parity-hypotheses.md P1)
# 0 keeps the historical orthographic plate. The lever is a wall reveal of
#   reveal / true height = exaggeration * tan(fov / 2)
# and 0.348 is the reference's read.
#
# V7: 18.0 -> 38.4 deg. This is the OTHER half of the exaggeration change and
# it is not an independent look choice: with WORLD_UNITS_PER_M dropped to 1.00x
# the old 18 deg would have given 1.00 * tan(9) = 0.158, less than half the
# reveal, and the walls would have stopped carrying the form. 38.4 deg restores
# it - 1.00 * tan(19.2) = 0.348, identical to the shipped 2.20 * tan(9).
#
# So the two constants are TIED: exaggeration * tan(fov/2) = 0.348. Overriding
# --exaggeration or --fov alone breaks the tie deliberately (that is what the
# flags are for), but the DEFAULTS must be re-solved together.
#
# Everything about this lever is compose-side: the light field stays orthogonal
# and stays VALID, because diffuse radiance is view-independent.
PERSPECTIVE_FOV_DEG = 38.4

# Per-building outline width in overlay px, pushed into city_lod2_plate.
#
# V7: 3 -> 1. At 2.20x the stroke was doing the separation work the walls do
# now, and 3 px of it survived the 0.686 overlay->disc downscale as a hard
# black rule around every roof - which is most of why the centre of the disc
# read as a dark mesh. 1 px still survives the BOX resample as a faint line
# (measured: it is what keeps the disc centre legible where the blocks touch),
# but it no longer competes with the walls. 0 was refuted - see the measured
# stroke-on/stroke-off table below.
EDGE_PX_DEFAULT = 1

# --- per-city look overrides -------------------------------------------------
# Measured on the delivered plate, the disc is already WARM overall (mean RGB
# 182/177/152, B/R 0.833) and the HDRI is neutral (snow_field_1k B/R 0.987) - so
# neither is what reads cold. The cold impression comes from the ROAD AND WATER
# palette, which is cool (casing B/R 1.070, minor 1.045, water 1.134) and forms
# a dense web across the whole disc, dominating the eye even though the
# area-weighted mean is warm.
#
# The dark impression is partly real (disc mean luma 67%) and partly a surround
# effect introduced by the pure-white sheet: the disc used to sit at 0.81x its
# background, and against white it sits at 0.67x. Same pixels, more contrast.
#
# So "lighter and warmer" is a floor lift plus a warm-grey road network, not an
# HDRI swap - switching to forge3d_neutral_daylight_v1 would make it COLDER
# (B/R 1.156) despite the name.
#
# What follows was tuned on Prague and then applied to all three: the class
# breaks are deliberately shared so the plates can be compared, and a palette
# that differed per city would defeat that just as thoroughly as per-city
# breaks would. LOOK below carries only genuine per-city deltas.
BASE_LOOK: dict = dict(
        # LIGHTER has to come from the PALETTE, not the modulation. `scale =
        # floor + gain*shade` is <= 1, so the plate can never be brighter than
        # the colours it multiplies - measured, the ramp's mean luma was 171 and
        # the finished disc 170, i.e. already pinned at the ceiling. Raising the
        # floor 0.70 -> 0.78 bought only ~2%, so the ramp itself is lifted ~13%.
        # Ramp spread widened at the TOP: the two lightest classes were only 14
        # luma apart, so the low-rise fabric had almost no separation. Now
        # 251/230/213/193/169/148/113 - deltas 21/17/20/24/21/35.
        height_ramp=((0xFF, 0xFB, 0xF4), (0xF6, 0xE5, 0xC8), (0xEE, 0xD3, 0xA6),
                     (0xE4, 0xBC, 0x86), (0xD6, 0xA2, 0x68), (0xC6, 0x8C, 0x52),
                     (0xA5, 0x66, 0x3F)),
        park=(0xA8, 0xBA, 0x8E),               # lifted from (142,166,122)
        # The MODULATION CEILING is unpinned. floor + gain was exactly 1.0, so
        # sunlight could only ever fail to darken, never actually brighten, and
        # the measured lit-vs-shadow separation was 5.1% (182.4 vs 169.4 / 255).
        # Keeping that 1.14 ceiling, the V8 sweep moved weight from floor to
        # gain: 0.72 + 0.42 widens the p5-p95 luma spread a further +14/+17/+11
        # (prague/zurich/lyon) over 0.80/0.34 for only ~4 luma of mean, keeps
        # p95 unclipped at 245, and holds the dark share (<100 luma) under the
        # reference's 3.8% everywhere (worst: lyon 2.72%). The lighter
        # 0.88/0.26 direction was REFUTED - it flattens all three plates.
        shadow_floor=0.72, shadow_gain=0.42,
        highlight_knee=0.82,                   # required once the ceiling > 1.0
        unify_contrast=1.045,
        warm_balance=(1.085, 1.005, 0.885),
        # Chromatic depth. Measured, the shadows were WARMER than the lit areas
        # (B/R 0.652 vs 0.740) - backwards, since shadow here is sky-lit. A
        # little cool in the deepest shadow plus more of the sky's own chroma
        # restores the cue, and it costs no luminance at all; warm_balance is
        # pushed slightly further to keep the plate's overall warmth. 0.07 was
        # a partial correction; the V8 sweep doubled it and the dark-quartile
        # B/R rose 0.467->0.505 / 0.573->0.637 / 0.468->0.508 with no measured
        # cost to brightness or spread. 0.00 was REFUTED (shadows go warm).
        cool_shadow=0.14,
        # ...and the differential that actually flips the cue. Swept on the real
        # modulation inputs (overlay + shade + tint cached from a production
        # run, so a variant costs milliseconds instead of a 7-minute compose):
        #   split  0.00   hue -0.138   lit 0.814 / shadow 0.676   clip 0.00%
        #          0.14   hue -0.064   lit 0.770 / shadow 0.706   clip 0.00%
        #          0.28   hue +0.010   lit 0.725 / shadow 0.735   clip 0.00%
        # 0.28 is the measured point where the sign flips on the exact
        # modulation arrays, and it costs nothing: lit/shadow LUMA separation
        # moves 16.99% -> 16.86% and mean luma 199.9 -> 199.7.
        #
        # WATER IS HELD OUT of the split - see _modulate_overlay. Unprotected,
        # 0.28 pushed the Vltava from B/R 1.147 to 1.053 with GREEN as its top
        # channel. The exemption costs almost nothing: hue -0.138 -> -0.009
        # instead of -0.138 -> +0.010, and water returns to its solved value
        # exactly.
        #
        # On the FINISHED plates it is a partial correction, not a cure. All
        # three, hue separation before -> after, with LUMA separation alongside
        # to show the lever costs no relief:
        #   Prague  -0.270 -> -0.198   luma 18.52% -> 18.42%
        #   Zurich  -0.265 -> -0.182   luma 17.22% -> 17.11%
        #   Lyon    -0.257 -> -0.191   luma 18.64% -> 18.52%
        # The rest
        # is not reachable from the grade at all, because -0.214 of it is
        # already present in the RAW light-free overlay (see the note on
        # WARM_SHADE_SPLIT in osm_city_pt_3d). Pushing the split far enough to
        # flip the finished plate would need ~0.85, which turns the ochre roof
        # classes orange. Shortening the shadows is the real remedy - which is
        # exactly what docs/city-lod2-3d-parity-hypotheses.md P2/P3 propose.
        #
        # REFUTED on the way here: rolling the highlight shoulder off on
        # LUMINANCE instead of per channel. The mechanism was plausible - the
        # per-channel knee compresses whichever channel warm_balance
        # over-drives, and that is red, exactly where the sun is - but measured
        # it moved the cue only -0.138 -> -0.129 while minting 4.4% clipped
        # pixels, because the per-channel knee is what was keeping the
        # over-driven channels under 1.0 in the first place.
        warm_shade_split=0.28,
        light_tint_strength=0.20,
        sat_pull=0.0,                          # keep the chroma; 0.07 muddied it
        road_casing=(0xB9, 0xB2, 0xA7),        # warm greys: B/R 0.908 (was 1.070)
        road_minor=(0xD4, 0xCD, 0xC1),         # B/R 0.910 (was 1.045)
        road_major=(0xEA, 0xE4, 0xDA),         # B/R 0.930 (was 1.032)
        # Water is PRE-COMPENSATED for the grade, not picked by eye. warm_balance
        # multiplies blue by 0.885 and red by 1.085, which inverts the hue of any
        # plausible water colour: the previous (198,216,220) (B/R 1.111) came out
        # of the chain at B/R 0.904 with GREEN the highest channel - measured on
        # the Zurichsee at (235.5, 237.4, 225.3), a pale green field, not a lake.
        # This value is the solved inverse of the real per-channel chain (scale ->
        # highlight shoulder -> unify_contrast) for a target of (186,203,219),
        # and forward-checks to (185.6, 203.2, 219.3), B/R 1.18. Blue is held
        # just under 250 to leave headroom for WATER_BLUE_LIFT.
        water=(0xAD, 0xCC, 0xF4),
)

# Per-city deltas layered ON TOP of BASE_LOOK. Empty means the house style is
# used unchanged - which is the intended state, since a city needing its own
# palette is a city that can no longer be compared with the others.
LOOK: dict[str, dict] = {}


_LOOK_ATTRS = (("warm_balance", pt, "WARM_BALANCE"),
               ("cool_shadow", pt, "COOL_SHADOW"),
               ("warm_shade_split", pt, "WARM_SHADE_SPLIT"),
               ("light_tint_strength", pt, "LIGHT_TINT_STRENGTH"),
               ("unify_contrast", pt, "UNIFY_CONTRAST"),
               ("sat_pull", pt, "SAT_PULL"),
               ("road_casing", pt, "ROAD_CASING_RGB"),
               ("road_minor", pt, "ROAD_MINOR_RGB"),
               ("road_major", pt, "ROAD_MAJOR_RGB"),
               ("water", pt, "WATER_OVERLAY_RGB"),
               ("park", pt, "PARK_OVERLAY_RGB"),
               ("highlight_knee", pt, "HIGHLIGHT_KNEE"),
               ("height_ramp", plate, "HEIGHT_RAMP_RGB"))
# Pristine values, snapshotted before anything can overwrite them.
_LOOK_DEFAULTS = [(mod, attr, getattr(mod, attr)) for _, mod, attr in _LOOK_ATTRS]


def apply_look(city_key: str) -> dict:
    """Push a city's look overrides into the osm_city_pt_3d module globals.

    Resets to the snapshotted defaults FIRST. Applying only the keys a city
    names would leave the previous city's palette in place for any city
    processed afterwards in the same process - today each runs in its own
    process, so it would not bite, which is exactly what makes it worth
    closing now rather than after someone loops over the cities in one run.
    """
    for mod, attr, value in _LOOK_DEFAULTS:
        setattr(mod, attr, value)
    look = {**BASE_LOOK, **LOOK.get(city_key, {})}
    for key, mod, attr in _LOOK_ATTRS:
        if key in look:
            setattr(mod, attr, look[key])
    if "height_ramp" in look and len(look["height_ramp"]) != len(plate.HEIGHT_BREAKS_M) + 1:
        raise SystemExit(
            f"{city_key}: height_ramp has {len(look['height_ramp'])} colours but "
            f"{len(plate.HEIGHT_BREAKS_M) + 1} classes are defined")
    return look


def _printed_provenance(record: dict, *, edition: str) -> str:
    """Keep the footer concise; the manifest carries the complete record."""
    p = record["provenance"]
    base_map = ("© OpenStreetMap contributors"
                if p["base_map_source"].startswith("OpenStreetMap")
                else p["base_map_source"])
    second = (f"Aerial {p['aerial_provider']} · Base map {base_map}"
              if edition == "orthophoto"
              else f"Base map {base_map}")
    return (f"Buildings {p['building_provider']} · "
            f"Terrain {p['terrain_provider']}\n{second}")

# LoD2 coverage is decided on a coarse grid, not per pixel. Per-pixel mixing
# would drop an OSM box into every gap BETWEEN LoD2 roofs - light wells, roof
# terraces, the sliver between two blocks - and re-inflate the buildings the
# LoD2 model deliberately carves apart. One coarse cell is about a city block.
#
# The cell is sized in METRES, not pixels: the same mask is built at the 768
# probe grid and at the 8192 trace grid, and a fixed pixel count would silently
# mean 667 m in one and 62 m in the other - a different decision each time.
COVER_CELL_M = 64.0
COVER_MIN_FRAC = 0.02


def _blur_f32(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a float32 field.

    NOT via PIL: `Image.fromarray(a, "F").filter(GaussianBlur(...))` raises
    "image has wrong mode" - PIL's blur filters reject mode "F" outright, and
    the failure only surfaces at modulation time, minutes into a compose. scipy
    handles float natively; the fallback is three box passes, which is a well
    known gaussian approximation and keeps this working without scipy.
    """
    a = np.ascontiguousarray(a, dtype=np.float32)
    if sigma <= 0.0:
        return a
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(a, sigma, mode="nearest").astype(np.float32)
    except ImportError:
        pass
    # Three box passes of width w have variance 3*(w^2-1)/12 = (w^2-1)/4, so
    # matching a gaussian of the requested sigma needs w = sqrt(4*sigma^2 + 1).
    # The boxes below are 2r wide, hence r = w/2. Guessing the radius instead
    # lands nowhere near the requested sigma - verified against scipy.
    w = math.sqrt(4.0 * sigma * sigma + 1.0)
    r = max(1, int(round(w / 2.0)))
    out = a
    for _ in range(3):
        pad = np.pad(out, ((r, r), (0, 0)), mode="edge")
        c = np.cumsum(pad, axis=0, dtype=np.float32)
        out = (c[2 * r:] - c[:-2 * r]) / (2 * r)
        pad = np.pad(out, ((0, 0), (r, r)), mode="edge")
        c = np.cumsum(pad, axis=1, dtype=np.float32)
        out = (c[:, 2 * r:] - c[:, :-2 * r]) / (2 * r)
    return out.astype(np.float32)


def coarse_lod2_cover(have: np.ndarray, half_extent_m: float,
                      cell_m: float = COVER_CELL_M) -> np.ndarray:
    """Per-pixel mask of the districts that actually have LoD2 roofs."""
    grid = have.shape[0]
    cell_px = max(1.0, cell_m * grid / (2.0 * float(half_extent_m)))
    n = max(1, int(round(grid / cell_px)))
    small = Image.fromarray(have.astype(np.float32), mode="F").resize(
        (n, n), Image.Resampling.BOX)
    occupied = (np.asarray(small, dtype=np.float32) >= COVER_MIN_FRAC)
    # Dilate by one cell so a district's edge blocks stay on the LoD2 side of
    # the seam rather than getting an OSM box stacked beside their real roof.
    grown = Image.fromarray((occupied * 255).astype(np.uint8), mode="L").filter(
        ImageFilter.MaxFilter(3))
    return np.asarray(grown.resize((grid, grid), Image.Resampling.NEAREST)) > 127


def main() -> int:
    # --exaggeration rewrites this module constant in place. That is the whole
    # point: heightfield() and perspective_modulate() BOTH derive their vertical
    # scale from this one name, so rewriting it is what keeps the tracer's
    # shadows and the re-projection's walls describing the same building.
    global WORLD_UNITS_PER_M
    ap = argparse.ArgumentParser()
    ap.add_argument("city", choices=sorted(CITIES))
    ap.add_argument("--grid-max", type=int, default=4096)
    ap.add_argument("--frame", type=int, default=900)
    ap.add_argument("--tiles", type=int, default=4)
    ap.add_argument("--size", type=int, nargs=2, default=(7200, 7200))
    ap.add_argument("--radius", type=float, default=2000.0)
    ap.add_argument("--terrain-scale", type=float, default=None,
                    help="terrain metres of height per metre of DEM relief, RAW - "
                         "the per-city config value gets TERRAIN_SCALE_COMPENSATION "
                         "applied, an explicit value does not. Terrain world scale "
                         "is exaggeration * this; V7 Prague ships 1.00 * 0.748.")
    ap.add_argument("--relief", type=float, default=None)
    ap.add_argument("--reuse-shade", action="store_true")
    ap.add_argument("--reuse-legacy-shade", action="store_true",
                    help="reuse the old unverified field; output is retrace-required")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--sun-azimuth", type=float, default=None,
                    help="override the north-up sun azimuth (see --probe)")
    ap.add_argument("--env-intensity", type=float, default=None,
                    help="HDRI dome fill. The sun is a DIRECTIONAL light with no "
                         "angular size, so every soft shadow edge on the plate "
                         "comes from this dome - raising it is the only way to "
                         "widen a penumbra rather than just lighten it.")
    ap.add_argument("--sun-intensity", type=float, default=None)
    ap.add_argument("--sun-elevation", type=float, default=None,
                    help=f"sun elevation above the horizon in degrees (V7 default "
                         f"{SUN_ELEVATION_DEG:g}). Shadow length is cot(elev) x height x "
                         "the vertical exaggeration, so "
                         "this is the only lever that shortens the cast shadows "
                         "themselves rather than just lightening them. Changes the "
                         "light field, so it needs a full trace, not --reuse-shade.")
    ap.add_argument("--exaggeration", type=float, default=None,
                    help="vertical exaggeration in metres of height per metre of "
                         "ground (V7 ships 1.00; was 2.20). Rewrites WORLD_UNITS_PER_M, so it "
                         "reaches BOTH the tracer (via RELIEF_WORLD) and the "
                         "perspective re-projection, which derive from that one "
                         "constant. Shadow length scales with it, and so does the "
                         "wall reveal: reveal/height = exaggeration * tan(fov/2), so "
                         "changing this WITHOUT --fov breaks the 0.348 reveal the "
                         "defaults are solved for. Ignored when --relief is given.")
    ap.add_argument("--shade-soften", type=float, default=None,
                    help="post-trace blur on the light field, in overlay px")
    ap.add_argument("--fov", type=float, default=None,
                    help=f"nadir PERSPECTIVE re-projection field of view in degrees "
                         f"(V7 default {PERSPECTIVE_FOV_DEG:g}, tied to the exaggeration "
                         "so reveal/height stays 0.348); "
                         "0 restores the historical orthographic plate. Compose-side "
                         "only - the cached light field stays valid because diffuse "
                         "radiance is view-independent.")
    ap.add_argument("--edge-stroke", type=int, default=None,
                    help=f"per-building outline width in overlay px (V7 default "
                         f"{EDGE_PX_DEFAULT}; 0 = off, which was REFUTED - see the "
                         "measured stroke-on/off table in the source)")
    ap.add_argument("--no-road-casing", action="store_true",
                    help="drop the dark casing under every road. The reference has "
                         "none, but the casing is what makes the road network trace "
                         "at a glance on a MAP, so it is kept by default.")
    ap.add_argument("--tag", default="",
                    help="suffix for the light-field cache and output filename, "
                         "so a parameter sweep cannot overwrite the real plate")
    ap.add_argument("--legend", action="store_true",
                    help="draw the roof-height legend (off by default)")
    ap.add_argument("--ortho", action="store_true",
                    help="drape the official orthophoto instead of the palette")
    # 0.38 over the launch 0.30: the 2026-07-31 sweep (0.30/0.38/0.45 on all
    # four cities) showed the extra weight lands almost entirely in the PT
    # street shadows - the plate mean moves only ~2 luma per step because the
    # imagery's own dynamic range dominates - so 0.38 buys visible relief for
    # free. 0.45 holds up on bright imagery (Prague/Lyon/Luxembourg) but
    # pushes SWISSIMAGE's already-dark shadows toward murk (Zurich terrain
    # mean 78 vs Prague 109), so the heavier weight stays a per-run flag.
    ap.add_argument("--ortho-shade", type=float, default=0.38,
                    help="how hard the PT light field modulates the imagery")
    a = ap.parse_args()

    cfg = CITIES[a.city]
    edition = "orthophoto" if a.ortho else "clean"
    record = launch_record(a.city, edition)
    _validate_canonical_launch_args(a, record)
    if record is not None:
        centre = record["centre"]
        if abs(float(centre["lat"]) - float(cfg["lat"])) > 1e-6 or \
                abs(float(centre["lon"]) - float(cfg["lon"])) > 1e-6:
            raise ValueError(f"{a.city}/{edition}: manifest centre disagrees with renderer config")
        if abs(float(record["coverage_radius_m"]) - float(a.radius)) > 1e-6:
            raise ValueError(
                f"{a.city}/{edition}: launch manifest requires {record['coverage_radius_m']} m radius")
    work = ROOT / a.city
    work.mkdir(parents=True, exist_ok=True)
    if a.probe:
        a.grid_max, a.frame, a.tiles, a.size = 1024, 320, 1, (1500, 1500)

    spec = C.CitySpec(name=cfg["title"], lon=cfg["lon"], lat=cfg["lat"],
                      src_epsg=cfg["src_epsg"], gml_paths=list(cfg["gml"]),
                      dem_paths=list(cfg["dem_tiles"] or []), dem_url=cfg["dem_url"],
                      radius=a.radius)
    if record is not None and record["source_crs"] != spec.src_epsg:
        raise ValueError(f"{a.city}/{edition}: manifest source CRS disagrees with renderer config")

    to_src = Transformer.from_crs("EPSG:4326", spec.src_epsg, always_xy=True)
    cx_src, cy_src = to_src.transform(spec.lon, spec.lat)
    if cfg.get("rings"):
        rings = Path(cfg["rings"])          # pre-extracted (non-CityGML source)
        if not C.valid_ring_cache(rings, (cx_src, cy_src), a.radius):
            raise ValueError(f"{a.city}: pre-extracted rings do not prove the requested "
                             f"centre/radius ({rings})")
        print(f"[{spec.name}] rings: {rings.name}")
    else:
        rings = C.extract(spec, cx_src, cy_src, work / f"{a.city}_disc{int(a.radius)}.npz")

    dst_epsg = C.scene_epsg(spec.lon, spec.lat)
    manifest_render_crs = (LAUNCH_MANIFEST.get("rendering", {})
                           .get("render_crs_by_city", {}).get(cfg["title"]))
    if record is not None and manifest_render_crs != dst_epsg:
        raise ValueError(f"{a.city}/{edition}: manifest render CRS {manifest_render_crs} "
                         f"does not match runtime scene CRS {dst_epsg}")
    to_dst = Transformer.from_crs(spec.src_epsg, dst_epsg, always_xy=True)
    cx_dst, cy_dst = Transformer.from_crs("EPSG:4326", dst_epsg, always_xy=True).transform(
        spec.lon, spec.lat)
    print(f"[{spec.name}] scene CRS {dst_epsg}, centre {cx_dst:.1f} {cy_dst:.1f}")

    # --terrain-scale is taken RAW when given (it is the world-scale knob the
    # sweeps used), but the per-city config value is the pre-V7 number and gets
    # the compensation factor so the terrain survives the exaggeration drop.
    if a.terrain_scale is not None:
        tscale = a.terrain_scale
    else:
        tscale = cfg["terrain_scale"] * TERRAIN_SCALE_COMPENSATION

    # Exaggeration -> WORLD_UNITS_PER_M. m_per_wu is fixed by the grid span and
    # the disc radius, and the normal path's exaggeration is exactly
    # WORLD_UNITS_PER_M * m_per_wu (the hf range cancels out of RELIEF_WORLD),
    # so this inverts cleanly. Done BEFORE the closures below are built, and by
    # rebinding the module global rather than a local, because both of them read
    # the global by name at call time.
    m_per_wu = 2.0 * float(a.radius) / pt.SPAN_X
    if a.exaggeration is not None:
        if a.relief is not None:
            raise SystemExit("--exaggeration and --relief both set the vertical "
                             "scale; --relief wins and would silently ignore it")
        WORLD_UNITS_PER_M = float(a.exaggeration) / m_per_wu
    print(f"[{spec.name}] vertical exaggeration "
          f"{WORLD_UNITS_PER_M * m_per_wu:.2f}x buildings / "
          f"{WORLD_UNITS_PER_M * m_per_wu * tscale:.3f}x terrain "
          f"(WORLD_UNITS_PER_M {WORLD_UNITS_PER_M:.4f}, terrain_scale {tscale:.3f})")

    # --- north-up frame -----------------------------------------------------
    # Cancel osm_city_demo's -180 scene rotation instead of matching it, so the
    # plate is north-up and the north arrow can be honest. Terrain and roof DSM
    # therefore take NO 180 flip either; all three layers share one frame.
    pt.city.SCENE_ROTATION_DEG = 0.0
    pt.MAP_ROTATION_DEG = 0.0
    lighting = LAUNCH_MANIFEST.get("city_lighting", {}).get(cfg["title"], {})
    pt.SUN_AZIMUTH = float(
        a.sun_azimuth if a.sun_azimuth is not None
        else lighting.get("azimuth_deg", SUN_AZIMUTH_NORTH_UP))
    product_title = record["title"] if record is not None else cfg["title"]
    display_title = cfg["title"]
    pt.city.POSTER_TITLE = product_title

    # --- softer shadows + per-city look --------------------------------------
    look = apply_look(a.city)
    pt.TERRAIN_FLOOR = float(look.get("shadow_floor", SHADOW_FLOOR))
    pt.TERRAIN_GAIN = float(look.get("shadow_gain", SHADOW_GAIN))
    if look:
        print(f"[{spec.name}] look override: {', '.join(sorted(look))}")
    if a.env_intensity is not None:
        pt.ENV_INTENSITY = float(a.env_intensity)
    if a.sun_intensity is not None:
        pt.SUN_INTENSITY = float(a.sun_intensity)
    # Always set, not only on --sun-elevation: the V7 default lives here rather
    # than in osm_city_pt_3d, whose 26 deg other city scripts still ship at.
    pt.SUN_ELEVATION = float(
        a.sun_elevation if a.sun_elevation is not None
        else lighting.get("elevation_deg", SUN_ELEVATION_DEG))
    soften_px = SHADE_SOFTEN_PX if a.shade_soften is None else float(a.shade_soften)
    print(f"[{spec.name}] light: sun {pt.SUN_INTENSITY:.2f} / env {pt.ENV_INTENSITY:.2f} "
          f"(ratio {pt.SUN_INTENSITY/max(pt.ENV_INTENSITY,1e-6):.2f}), "
          f"elevation {pt.SUN_ELEVATION:.1f} deg "
          f"(shadow {1.0/max(math.tan(math.radians(pt.SUN_ELEVATION)),1e-6):.2f}x height), "
          f"shade soften {soften_px:.1f} px, floor {pt.TERRAIN_FLOOR:.2f}")
    _shade_on_grid = pt._shade_on_overlay_grid

    def softened_shade(rgb_field, hit, overlay_size):
        shade, tint = _shade_on_grid(rgb_field, hit, overlay_size)
        if soften_px > 0.0:
            shade = _blur_f32(shade, soften_px)
        # The TINT is deliberately left unblurred: it carries the sky's colour,
        # which varies smoothly already, and blurring it only desaturates the
        # shadow chroma that gives the plate its depth.
        return shade, tint

    pt._shade_on_overlay_grid = softened_shade

    osm_only_heightfield = pt.build_building_heightfield   # capture before patching
    # Keyed by grid size for the building heights, and by f"terrain_{grid}" for
    # the DEM at that same grid, so heightfield() can reuse both without a
    # second fetch.
    cache: dict[int | str, np.ndarray] = {}
    # Share of the built area whose height came from OSM tags rather than the
    # LoD2 survey. Disclosed on the sheet when it is not negligible: a plate
    # that mixes surveyed roofs with tag-inferred boxes and says nothing is the
    # exact failure this whole rebuild was about.
    coverage = {"fallback_share": 0.0}

    def building_heights(scene, half_extent_m: float, grid: int) -> np.ndarray:
        """Metres of building above terrain, LoD2 where surveyed, OSM elsewhere.

        Memoized because the tracer and the colour overlay must describe the
        same surface, and rebuilding the roof DSM costs 40-110 s at 8192.
        """
        if grid in cache:
            return cache[grid]
        terrain_cache = work / f"terrain_{dst_epsg.split(':')[1]}_{grid}.npy"
        wms = cfg.get("dem_wms")
        ags = cfg.get("dem_arcgis")
        if wms:
            terrain = C.wms_dem(cx_dst, cy_dst, half_extent_m, grid, crs=dst_epsg,
                                layer=wms["layer"], url=wms["url"], cache=terrain_cache)
        elif ags:
            terrain = C.arcgis_dem(cx_dst, cy_dst, half_extent_m, grid, crs=dst_epsg,
                                   url=ags["url"], cache=terrain_cache)
        else:
            terrain = C.terrain_grid(spec, cx_dst, cy_dst, half_extent_m, grid,
                                     dst_epsg, terrain_cache)
        roof = C.roof_dsm(rings, to_dst, cx_dst, cy_dst, half_extent_m, grid,
                          flip180=False)
        have = roof > -8e8
        bh = np.where(have, np.maximum(roof - terrain, 0.0), 0.0).astype(np.float32)

        covered = coarse_lod2_cover(have, half_extent_m)
        osm_bh = osm_only_heightfield(scene, half_extent_m, grid)
        bh = np.where(covered, bh, osm_bh).astype(np.float32)
        filled = (~covered) & (osm_bh > 0.0)
        built = bh > 0.0
        coverage["fallback_share"] = (
            float(filled.sum()) / float(max(int(built.sum()), 1)) * 100.0)
        print(f"[{spec.name}] built {built.mean()*100:.1f}%  "
              f"(LoD2 {(covered & built).mean()*100:.1f}%, "
              f"OSM fallback {filled.mean()*100:.1f}%)  "
              f"roof p50 {np.percentile(bh[built], 50):.1f} m "
              f"p99 {np.percentile(bh[built], 99):.1f} m  "
              f"terrain relief {terrain.max()-terrain.min():.1f} m")
        # Class histogram: a legend is only honest if the classes are populated,
        # so print the split every run rather than trusting the chosen breaks.
        cls, cbuilt = plate.classify_heights(bh)
        shares = [float((cls[cbuilt] == i).mean()) * 100 for i in range(len(plate.HEIGHT_RAMP_RGB))]
        print(f"[{spec.name}] height classes " + "  ".join(
            f"{lab}:{s:.1f}%" for lab, s in zip(plate.HEIGHT_CLASS_LABELS, shares)))
        cache[grid] = bh
        cache[f"terrain_{grid}"] = terrain
        return bh

    def heightfield(scene, half_extent_m, grid):
        bh = building_heights(scene, half_extent_m, grid)
        terrain = cache[f"terrain_{grid}"]
        hf = (terrain - terrain.min()) * tscale + bh
        if a.relief is not None:
            pt.RELIEF_WORLD = float(a.relief)
        else:
            pt.RELIEF_WORLD = WORLD_UNITS_PER_M * float(hf.max() - hf.min())
        print(f"[{spec.name}] hf {hf.min():.1f}..{hf.max():.1f} m  "
              f"relief {pt.RELIEF_WORLD:.2f} world units "
              f"({WORLD_UNITS_PER_M:.3f}/m)")
        return hf

    def height_overlay(scene, half_extent_m, size):
        return plate.build_height_overlay(
            scene, half_extent_m, size,
            building_heights(scene, half_extent_m, size))

    pt.build_building_heightfield = heightfield
    pt.build_lightfree_overlay = height_overlay

    # --- nadir perspective ---------------------------------------------------
    # Wrapping _modulate_overlay rather than the compose is deliberate: the warp
    # has to run on the LIGHT-FREE overlay and its light field SEPARATELY, so a
    # wall arrives carrying its own light and the existing modulation chain
    # (which is what the whole house style is tuned on) runs untouched over it.
    # Re-projecting an already-modulated plate would mean dividing the roof's
    # light back out of the wall pixels.
    fov = PERSPECTIVE_FOV_DEG if a.fov is None else float(a.fov)
    if record is not None and abs(fov - float(record["camera"]["fov_deg"])) > 1e-6:
        raise ValueError(f"{a.city}/{edition}: runtime FOV {fov:g} differs from manifest")
    # The per-building edge stroke is KEPT under perspective. It was switched
    # off first, on the theory that the stroke rings the ROOF while the wall
    # stands on the INNER edge, so the two would disagree by the displacement.
    # That theory is wrong: the stroke lives in the light-free overlay, and the
    # re-projection GATHERS the overlay, so the stroke travels with the roof to
    # its displaced position and outlines it correctly, with the wall inboard.
    #
    # Measured on Prague, 800 m crop, perspective on in both:
    #   stroke on   p1  88.7  range 161.9  dark 8.52%  rms@2m 24.37  rms@25m 33.82
    #   stroke off  p1 120.6  range 129.0  dark 3.86%  rms@2m 18.82  rms@25m 27.63
    # Removing it cost 20% of the dynamic range, 55% of the dark pixels and 23%
    # of the small-scale contrast - by far the largest effect of anything
    # changed in this pass, and all of it a loss.
    plate.EDGE_PX = int(a.edge_stroke if a.edge_stroke is not None else EDGE_PX_DEFAULT)
    if a.no_road_casing:
        pt.ROAD_CASING_M = pt.ROAD_MAJOR_CASING_M = 0.0
    _modulate = pt._modulate_overlay

    def perspective_modulate(overlay_img, shade, tint):
        if fov > 0.0:
            size = overlay_img.size[0]
            # building_heights memoizes by grid, and height_overlay has already
            # asked for exactly this grid, so this is a dict hit, not a rebuild.
            bh = cache[size]
            m_per_wu = 2.0 * float(a.radius) / pt.SPAN_X
            # Vertical exaggeration the TRACER used, in metres of height per
            # metre of ground. In the normal path RELIEF_WORLD = K*hf_range, so
            # the range cancels and the exaggeration is just K*m_per_wu = 2.20 -
            # which must NOT be derived from RELIEF_WORLD at run time, because
            # `--reuse-shade` never calls heightfield() and RELIEF_WORLD is then
            # still at its 6.0 module default. That read 240x and asked for 6928
            # march steps.
            if a.relief is None:
                exag = WORLD_UNITS_PER_M * m_per_wu
            else:
                hf = (cache[f"terrain_{size}"] - cache[f"terrain_{size}"].min()) * tscale + bh
                exag = float(a.relief) * m_per_wu / max(float(hf.max() - hf.min()), 1e-6)
            overlay_img, shade, tint = persp.reproject(
                overlay_img, shade, tint, bh, half_extent_m=float(a.radius),
                exaggeration=exag, sun_azimuth_deg=pt.SUN_AZIMUTH, fov_deg=fov)
        return _modulate(overlay_img, shade, tint)

    pt._modulate_overlay = perspective_modulate
    print(f"[{spec.name}] perspective fov {fov:g} deg, edge stroke {plate.EDGE_PX} px, "
          f"road casing {pt.ROAD_CASING_M:g}/{pt.ROAD_MAJOR_CASING_M:g} m")

    if a.ortho:
        o = cfg.get("ortho")
        if not o:
            raise SystemExit(f"no orthophoto service configured for {a.city}")
        ortho_policy = LAUNCH_MANIFEST["orthophoto_transform"]
        grade_policy = ortho_policy["grade"]
        wms_policy = ortho_policy["wms"]
        modulation_policy = ortho_policy["pt_modulation"]

        def ortho_overlay(scene, half_extent_m, size):
            rgb = C.wms_ortho(cx_dst, cy_dst, half_extent_m, size, crs=o["crs"],
                              layer=o["layer"], url=o["url"],
                              cache=work / f"ortho_{o['layer']}_{size}.npz",
                              tile=int(wms_policy["tile_px"]))
            return C.ortho_overlay(
                plate.grade_orthophoto(
                    rgb,
                    exposure=float(grade_policy["exposure"]),
                    neutral_strength=float(grade_policy["neutral_strength"]),
                    saturation_limit=float(grade_policy["saturation_limit"]),
                    shadow_floor=float(grade_policy["shadow_floor"]),
                    luma_percentiles=tuple(float(v) for v in grade_policy["luma_percentiles"])),
                flip180=False)

        pt.build_lightfree_overlay = ortho_overlay
        # No perspective warp on imagery: an orthophoto is already a real
        # photograph with its own camera baked in, and the height grid the warp
        # needs is never built in this mode.
        fov = 0.0
        # The imagery already carries the sun and shadows it was flown under, so
        # applying the full palette-mode modulation would double them up.
        ortho_shade = float(a.ortho_shade)
        pt.TERRAIN_FLOOR = 1.0 - ortho_shade
        pt.TERRAIN_GAIN = ortho_shade
        pt.LIGHT_TINT_STRENGTH = float(modulation_policy["light_tint_strength"])
        pt.SAT_PULL = 0.0
        # ...and no shade-driven hue split either: that lever exists to restore a
        # lit/shadow cue the flat PALETTE cannot carry, and aerial imagery already
        # carries its own.
        pt.WARM_SHADE_SPLIT = 0.0
        print(f"[{spec.name}] ortho mode: {o['layer']} via {o['crs']}, "
              f"shade floor {pt.TERRAIN_FLOOR:.2f} gain {pt.TERRAIN_GAIN:.2f}")

    # --- print-grade poster --------------------------------------------------
    north_deg = plate.north_screen_deg()
    render_crs = (LAUNCH_MANIFEST.get("rendering", {})
                  .get("render_crs_by_city", {}).get(cfg["title"], dst_epsg))
    terrain_world_scale = (LAUNCH_MANIFEST.get("rendering", {})
                           .get("terrain_world_units_per_dem_relief_m", {})
                           .get(cfg["title"]))
    hemi_ns = "N" if spec.lat >= 0 else "S"
    hemi_ew = "E" if spec.lon >= 0 else "W"
    camera_desc = (record["camera"]["projection"] if record is not None
                   else ("nadir-orthographic" if a.ortho else "nadir-perspective"))
    edition_prefix = "ORTHOPHOTO EDITION \u00b7 " if a.ortho else ""
    subtitle = (f"{edition_prefix}{a.radius/1000:g} km \u00b7 "
                f"{abs(spec.lat):.4f}\u00b0{hemi_ns} {abs(spec.lon):.4f}\u00b0{hemi_ew}")
    print(f"[{spec.name}] north points {north_deg:.1f}\u00b0 clockwise from screen-up, "
          f"sun azimuth {pt.SUN_AZIMUTH:.0f}\u00b0")

    def compose(subject, *, width, height, radius_m, **_ignored):
        share = coverage["fallback_share"]
        provenance = (_printed_provenance(record, edition=edition)
                      if record is not None else cfg["sources"])
        if record is None and a.ortho:
            provenance = f"{provenance} · Aerial imagery: source record pending"
        elif record is None and share >= 1.0:
            provenance = (f"{provenance} · {share:.0f}% of built area "
                          f"height-inferred from OSM tags")
        return plate.compose_plate(
            subject, width=width, height=height, radius_m=radius_m,
            subject_span_m=2.0 * float(radius_m), title=display_title,
            subtitle=subtitle, provenance=provenance, author="",
            north_screen_deg=north_deg, show_legend=bool(a.legend) and not a.ortho)

    pt.city.compose_poster = compose
    # compose_plate already sharpened the disc alone; a second pass over the
    # finished sheet would halo the type at 300 dpi.
    pt.UNSHARP = (1.6, 0, 2)
    pt.SAVE_KWARGS = {"dpi": (300, 300)}
    icc = pt.city.srgb_icc_profile_bytes()
    if icc:
        pt.SAVE_KWARGS["icc_profile"] = icc
    if record is not None:
        pnginfo = PngInfo()
        metadata = {
            "collection": LAUNCH_MANIFEST["collection"],
            "sku": record["sku"],
            "product_title": product_title,
            "display_title": display_title,
            "edition": edition,
            "centre_lat": f"{record['centre']['lat']:.6f}",
            "centre_lon": f"{record['centre']['lon']:.6f}",
            "coverage_radius_m": str(record["coverage_radius_m"]),
            "source_crs": record["source_crs"],
            "render_crs": render_crs,
            "camera_projection": camera_desc,
            "camera_fov_deg": f"{record['camera']['fov_deg']:g}",
            "north_direction": record["north_direction"],
            "north_screen_deg": f"{north_deg:g}",
            "scale_reference": LAUNCH_MANIFEST.get("scale_reference", plate.SCALE_REFERENCE),
            "terrain_world_units_per_dem_relief_m": "" if terrain_world_scale is None else f"{terrain_world_scale:g}",
            "manifest_sun_azimuth_deg": f"{pt.SUN_AZIMUTH:g}",
            "manifest_sun_elevation_deg": f"{pt.SUN_ELEVATION:g}",
            "provenance_status": LAUNCH_MANIFEST["provenance_status"],
        }
        if edition == "orthophoto":
            transform = LAUNCH_MANIFEST["orthophoto_transform"]
            grade = transform["grade"]
            metadata.update({
                "orthophoto_transform_version": transform["version"],
                "orthophoto_grade": (f"luma-p{grade['luma_percentiles'][0]:g}/p{grade['luma_percentiles'][1]:g}; "
                                     f"exposure={grade['exposure']:g}; neutral={grade['neutral_strength']:g}; "
                                     f"sat={grade['saturation_limit']:g}; floor={grade['shadow_floor']:g}"),
                "orthophoto_wms": (f"{transform['wms']['format']}; tile={transform['wms']['tile_px']}px; "
                                   f"{transform['wms']['resampling']}"),
                "orthophoto_pt_modulation": json.dumps(transform["pt_modulation"], sort_keys=True),
            })
        for key, value in metadata.items():
            pnginfo.add_text(key, value)
        pt.SAVE_KWARGS["pnginfo"] = pnginfo

    out_dir = Path(__file__).resolve().parent / "out" / "city_lod2"
    out_dir.mkdir(parents=True, exist_ok=True)
    # A sweep writes to its own cache dir and filename; the fingerprinted light
    # cache is an additional guard against stale fields within that directory.
    output_tag = a.tag or ("ortho" if a.ortho else "")
    pt.OUT_DIR = out_dir / (f"{a.city}_{output_tag}" if output_tag else a.city)
    pt.OUT_DIR.mkdir(parents=True, exist_ok=True)

    out = out_dir / (f"{a.city}_pt_{output_tag}.png" if output_tag else f"{a.city}_pt.png")
    sys.argv = ["osm_city_pt_3d", "--lon", str(spec.lon), "--lat", str(spec.lat),
                "--radius", str(a.radius), "--grid-max", str(a.grid_max),
                "--frame", str(a.frame), "--tiles", str(a.tiles),
                "--size", str(a.size[0]), str(a.size[1]),
                "--overlay-size", "8192" if max(a.size) >= 6000 else "4096",
                "--output", str(out)]
    if a.reuse_shade:
        sys.argv.append("--reuse-shade")
    if a.reuse_legacy_shade:
        sys.argv.append("--reuse-legacy-shade")
    if a.probe:
        sys.argv.append("--probe")
    return pt.main()


if __name__ == "__main__":
    raise SystemExit(main())
