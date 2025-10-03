# python/forge3d/adapters/geopandas_adapter.py
# M3: GeoPandas Adapter
# - Reproject GeoSeries/GeoDataFrame
# - Convert polygons to numpy rings
# - Extrude geometries to meshes via forge3d.geometry

from __future__ import annotations

from typing import List, Tuple, Optional, Any
import numpy as np

try:
    import geopandas as gpd
    import shapely.geometry as sgeom
    import shapely.ops as sops
    import pyproj
    _HAS_GPD = True
except Exception:
    _HAS_GPD = False
    gpd = None  # type: ignore


def is_geopandas_available() -> bool:
    return _HAS_GPD


def _require_geopandas():
    if not _HAS_GPD:
        raise ImportError(
            "GeoPandas/Shapely are required. Install with: pip install geopandas shapely pyproj"
        )


def reproject_geoseries(gs: "gpd.GeoSeries", dst_crs: str | Any) -> "gpd.GeoSeries":
    """Reproject a GeoSeries to dst_crs using GeoPandas .to_crs."""
    _require_geopandas()
    if not hasattr(gs, "to_crs"):
        raise ValueError("Expected GeoSeries-like object with .to_crs()")
    return gs.to_crs(dst_crs)


def _poly_to_ring_arrays(poly: "sgeom.Polygon") -> List[np.ndarray]:
    rings: List[np.ndarray] = []
    if poly.is_empty:
        return rings
    # Exterior
    ext = np.asarray(poly.exterior.coords, dtype=np.float32)
    if ext.shape[0] >= 3:
        rings.append(ext[:, :2])
    # Interiors (holes)
    for hole in poly.interiors:
        arr = np.asarray(hole.coords, dtype=np.float32)
        if arr.shape[0] >= 3:
            rings.append(arr[:, :2])
    return rings


def geoseries_to_polygons(gs: "gpd.GeoSeries") -> List[List[np.ndarray]]:
    """Convert a GeoSeries to a list of polygon ring arrays.

    Returns a list of geometries; each geometry is a list of (N,2) float32 arrays
    for the exterior and holes.
    """
    _require_geopandas()
    out: List[List[np.ndarray]] = []
    for geom in gs.geometry:  # type: ignore[attr-defined]
        if geom is None:
            continue
        if isinstance(geom, sgeom.Polygon):
            out.append(_poly_to_ring_arrays(geom))
        elif isinstance(geom, sgeom.MultiPolygon):
            parts: List[np.ndarray] = []
            for p in geom.geoms:
                parts.extend(_poly_to_ring_arrays(p))
            if parts:
                out.append(parts)
        else:
            # Attempt polygonize if LineString-like
            try:
                polys = list(sops.polygonize(geom))
                part_rings: List[np.ndarray] = []
                for p in polys:
                    part_rings.extend(_poly_to_ring_arrays(p))
                if part_rings:
                    out.append(part_rings)
            except Exception:
                continue
    return out


def extrude_geometries_to_meshes(
    gs: "gpd.GeoSeries",
    *,
    height: float = 1.0,
    cap_uv_scale: float = 1.0,
) -> List[Any]:
    """Extrude polygonal geometries to meshes using forge3d.geometry.extrude_polygon.

    Returns a list of MeshBuffers.
    """
    _require_geopandas()
    from .. import geometry as fgeo
    meshes: List[Any] = []
    polygons = geoseries_to_polygons(gs)
    for rings in polygons:
        if not rings:
            continue
        # Use exterior only for now; hole support can be added with CSG or triangle libs
        exterior = rings[0]
        try:
            mesh = fgeo.extrude_polygon(exterior.astype(np.float32), float(height), cap_uv_scale=float(cap_uv_scale))
            meshes.append(mesh)
        except Exception:
            continue
    return meshes
