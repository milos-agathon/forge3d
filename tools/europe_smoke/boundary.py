# tools/europe_smoke/boundary.py
"""The sentinel boundary the engine's DEM builder consumes, plus NE countries.

build_base_map() takes no bbox. The polygon written here is the ONLY thing
that defines the map extent and the DEM's valid mask, and the engine's default
is California relation 165475 -- so this file, and the _download_osm_boundary
stub in basemap.py, are load-bearing (§1.4).
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

from . import config, provenance

NE_URL = ("https://naturalearth.s3.amazonaws.com/10m_cultural/"
          "ne_10m_admin_0_countries.zip")


def osm_boundary_path(cache_dir: Path) -> Path:
    """Where the engine's _osm_boundary_path() will look."""
    return (Path(cache_dir) / "osm"
            / f"california_boundary_osm_r{config.SENTINEL_RELATION_ID}.geojson")


_OSM_SIDECARS = (".shp", ".shx", ".dbf", ".prj")


def prepare_osm_land(cache_dir: Path) -> Path:
    """Extract the VERIFIED OSM land ZIP into a digest-keyed cache directory.

    The manifest pins the whole archive, so every shapefile sidecar comes from
    the same verified bytes; a cache directory that lost a sidecar is a partial
    extraction and fails loudly rather than being silently reused.
    """
    entry = provenance.load_manifest()["osm_land"]
    finding = provenance.check_artifact("osm_land", entry)
    if finding.status != provenance.VERIFIED:
        raise RuntimeError(f"OSM land archive is not verified: {finding.detail}")

    dest = Path(cache_dir) / "osm" / f"land-{finding.sha256[:16]}"
    if dest.exists():
        existing = list(dest.rglob("simplified_land_polygons.shp"))
        if len(existing) != 1:
            raise RuntimeError(
                f"partial OSM extraction at {dest}; found {len(existing)} shapefiles"
            )
        shp = existing[0]
        missing = [suffix for suffix in _OSM_SIDECARS if not shp.with_suffix(suffix).exists()]
        if missing:
            raise RuntimeError(f"partial OSM extraction at {dest}; missing {missing}")
        return shp

    written = provenance.safe_extract(config.OSM_LAND_ZIP, dest)
    matches = [p for p in written if p.name == "simplified_land_polygons.shp"]
    if len(matches) != 1:
        raise RuntimeError(f"OSM archive carried {len(matches)} matching shapefiles")
    shp = matches[0]
    missing = [suffix for suffix in _OSM_SIDECARS if not shp.with_suffix(suffix).exists()]
    if missing:
        raise RuntimeError(f"OSM archive is missing required sidecars {missing}")
    return shp


def build_land_union(cache_dir: Path, force: bool = False, window=None) -> Path:
    """Union the verified OSM land polygons over the widened basemap window.

    The sentinel GeoJSON is keyed by (window, source hash) through its feature
    properties, so a boundary built for an older window or from different
    source bytes is rebuilt rather than silently reused.
    """
    window = tuple(window or config.BASEMAP_WINDOW)
    out = osm_boundary_path(cache_dir)
    source = prepare_osm_land(cache_dir)
    source_hash = provenance.check_artifact(
        "osm_land", provenance.load_manifest()["osm_land"]
    ).sha256

    wanted = {"window": list(window), "source_sha256": source_hash}
    if out.exists() and not force:
        try:
            current = json.loads(out.read_text(encoding="utf-8"))
            properties = current["features"][0].get("properties")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            properties = None
        if properties == wanted:
            return out

    west, south, east, north = window
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import box, mapping

    project = Transformer.from_crs(4326, 3857, always_xy=True).transform
    x0, y0 = project(west, south)
    x1, y1 = project(east, north)
    land = gpd.read_file(source, bbox=(x0, y0, x1, y1))
    if land.empty:
        raise RuntimeError("OSM land dataset did not cover BASEMAP_WINDOW")
    merged = land.geometry.union_all()
    geom = (gpd.GeoSeries([merged], crs=land.crs).to_crs(4326).iloc[0]
            .intersection(box(west, south, east, north)))
    if geom.is_empty:
        raise RuntimeError("land union emptied after clipping to BASEMAP_WINDOW")
    if not geom.is_valid:
        geom = geom.buffer(0)
    doc = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": wanted, "geometry": mapping(geom)
    }]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc), encoding="utf-8")
    return out


def load_countries(cache_dir: Path, force: bool = False):
    """Natural Earth 10m admin_0, clipped to the request area.

    The zip arrives over the network, so it is extracted through
    provenance.safe_extract -- never ZipFile.extractall -- and the shapefile
    parts are taken from the members that were actually written, not from a
    glob of a directory that may still hold an earlier download.
    """
    import geopandas as gpd
    from shapely.geometry import box

    zip_path = Path(cache_dir) / "ne" / "ne_10m_admin_0_countries.zip"
    if force or not zip_path.exists():
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        req = Request(NE_URL, headers={"User-Agent": config.USER_AGENT})
        with urlopen(req, timeout=600) as resp:
            zip_path.write_bytes(resp.read())

    written = provenance.safe_extract(zip_path, zip_path.parent)
    shps = [p for p in written if p.name == "ne_10m_admin_0_countries.shp"]
    if not shps:
        raise RuntimeError(
            f"{zip_path} carried no ne_10m_admin_0_countries.shp; members were "
            f"{sorted(p.name for p in written)}"
        )
    shp = shps[0]

    north, west, south, east = config.AREA
    gdf = gpd.read_file(shp)
    gdf = gdf[gdf.intersects(box(west, south, east, north))].copy()
    gdf["geometry"] = gdf.geometry.intersection(box(west, south, east, north))
    return gdf[~gdf.geometry.is_empty]
