# python/forge3d/crs.py
# CRS reprojection utilities with Rust proj / Python pyproj fallback
# RELEVANT FILES: src/geo/mod.rs, src/geo/reproject.rs, python/forge3d/render.py

"""
CRS (Coordinate Reference System) utilities for forge3d.

Provides reprojection of coordinates between different coordinate systems.
Uses the native Rust proj library when available, with pyproj as fallback.

Example:
    >>> from forge3d.crs import transform_coords, reproject_geom
    >>> import numpy as np
    >>>
    >>> # Transform WGS84 coordinates to UTM zone 54N
    >>> wgs84 = np.array([[138.73, 35.36], [138.74, 35.37]])
    >>> utm = transform_coords(wgs84, "EPSG:4326", "EPSG:32654")
    >>> print(utm)  # UTM coordinates in meters
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

# Try to import pyproj for fallback
# We require pyproj >= 2.0 for Transformer.from_crs() API
try:
    import pyproj
    # Check for modern API (pyproj >= 2.0)
    if hasattr(pyproj, 'Transformer'):
        HAS_PYPROJ = True
        HAS_PYPROJ_LEGACY = False
    elif hasattr(pyproj, 'Proj'):
        # Old pyproj API (< 2.0)
        HAS_PYPROJ = False
        HAS_PYPROJ_LEGACY = True
    else:
        HAS_PYPROJ = False
        HAS_PYPROJ_LEGACY = False
except ImportError:
    HAS_PYPROJ = False
    HAS_PYPROJ_LEGACY = False
    pyproj = None  # type: ignore

# Try to import native extension
try:
    from forge3d._forge3d import proj_available as _native_proj_available
    from forge3d._forge3d import reproject_coords as _native_reproject
    HAS_NATIVE_PROJ = _native_proj_available()
except (ImportError, AttributeError):
    HAS_NATIVE_PROJ = False
    _native_reproject = None


def proj_available() -> bool:
    """Check if any CRS transformation backend is available."""
    return HAS_NATIVE_PROJ or HAS_PYPROJ or HAS_PYPROJ_LEGACY


def transform_coords(
    coords: np.ndarray,
    from_crs: str,
    to_crs: str,
    *,
    always_xy: bool = True,
) -> np.ndarray:
    """Transform an array of coordinates from one CRS to another.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, 2) with [x, y] or [lon, lat] coordinates.
    from_crs : str
        Source CRS as EPSG code (e.g., "EPSG:4326") or PROJ string.
    to_crs : str
        Target CRS as EPSG code or PROJ string.
    always_xy : bool
        If True, coordinates are always interpreted as (x, y) / (lon, lat).
        This matches the pyproj always_xy=True convention. Default: True.

    Returns
    -------
    np.ndarray
        Transformed coordinates with shape (N, 2).

    Raises
    ------
    RuntimeError
        If no CRS transformation backend is available.
    ValueError
        If the CRS is invalid or transformation fails.

    Example
    -------
    >>> wgs84 = np.array([[138.73, 35.36]])  # lon, lat
    >>> utm = transform_coords(wgs84, "EPSG:4326", "EPSG:32654")
    >>> print(utm)  # [[x_utm, y_utm]]
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (N, 2), got {coords.shape}")

    if len(coords) == 0:
        return coords.copy()

    # Skip if same CRS
    if _crs_equal(from_crs, to_crs):
        return coords.copy()

    # Try native Rust implementation first
    if HAS_NATIVE_PROJ:
        try:
            coords_list = coords.tolist()
            result = _native_reproject(coords_list, from_crs, to_crs)
            return np.array(result, dtype=np.float64)
        except Exception:
            # Fall through to pyproj
            pass

    # Fallback to pyproj (modern API: pyproj >= 2.0)
    if HAS_PYPROJ:
        try:
            transformer = pyproj.Transformer.from_crs(
                from_crs, to_crs, always_xy=always_xy
            )
            x, y = transformer.transform(coords[:, 0], coords[:, 1])
            return np.column_stack([x, y])
        except Exception as e:
            raise ValueError(f"Reprojection failed: {e}") from e

    # Fallback to legacy pyproj API (pyproj < 2.0)
    if HAS_PYPROJ_LEGACY:
        try:
            # Legacy API uses pyproj.Proj and pyproj.transform
            src_proj = pyproj.Proj(init=from_crs) if from_crs.upper().startswith("EPSG:") else pyproj.Proj(from_crs)
            dst_proj = pyproj.Proj(init=to_crs) if to_crs.upper().startswith("EPSG:") else pyproj.Proj(to_crs)
            x, y = pyproj.transform(src_proj, dst_proj, coords[:, 0], coords[:, 1])
            return np.column_stack([x, y])
        except Exception as e:
            raise ValueError(f"Reprojection failed (legacy API): {e}") from e

    raise RuntimeError(
        "No CRS transformation backend available. "
        "Install pyproj (pip install pyproj) or build forge3d with proj feature."
    )


def reproject_geom(
    geom: "BaseGeometry",
    from_crs: str,
    to_crs: str,
) -> "BaseGeometry":
    """Reproject a Shapely geometry from one CRS to another.

    Parameters
    ----------
    geom : BaseGeometry
        Shapely geometry (Point, LineString, Polygon, etc.).
    from_crs : str
        Source CRS as EPSG code or PROJ string.
    to_crs : str
        Target CRS as EPSG code or PROJ string.

    Returns
    -------
    BaseGeometry
        Reprojected geometry of the same type.

    Example
    -------
    >>> from shapely.geometry import Point
    >>> pt_wgs84 = Point(138.73, 35.36)
    >>> pt_utm = reproject_geom(pt_wgs84, "EPSG:4326", "EPSG:32654")
    """
    try:
        from shapely import ops as shapely_ops
        from shapely.geometry import shape, mapping
    except ImportError as e:
        raise ImportError("shapely is required for geometry reprojection") from e

    # Skip if same CRS
    if _crs_equal(from_crs, to_crs):
        return geom

    # Use pyproj transformer if available (modern API)
    if HAS_PYPROJ:
        transformer = pyproj.Transformer.from_crs(
            from_crs, to_crs, always_xy=True
        )
        return shapely_ops.transform(transformer.transform, geom)

    # Use legacy pyproj API
    if HAS_PYPROJ_LEGACY:
        src_proj = pyproj.Proj(init=from_crs) if from_crs.upper().startswith("EPSG:") else pyproj.Proj(from_crs)
        dst_proj = pyproj.Proj(init=to_crs) if to_crs.upper().startswith("EPSG:") else pyproj.Proj(to_crs)
        def _transform_legacy(x, y):
            return pyproj.transform(src_proj, dst_proj, x, y)
        return shapely_ops.transform(_transform_legacy, geom)

    # Fallback: extract coords, transform, reconstruct
    # This only works for simple geometries
    coords = np.array(geom.coords)
    new_coords = transform_coords(coords, from_crs, to_crs)
    return type(geom)(new_coords.tolist())


def parse_crs_from_wkt(wkt: str) -> Optional[str]:
    """Parse a CRS from WKT string and return EPSG code if possible.

    Parameters
    ----------
    wkt : str
        Well-Known Text CRS definition.

    Returns
    -------
    Optional[str]
        EPSG code (e.g., "EPSG:4326") if recognized, or the original WKT.
    """
    if not HAS_PYPROJ and not HAS_PYPROJ_LEGACY:
        return None

    if HAS_PYPROJ:
        try:
            crs = pyproj.CRS.from_wkt(wkt)
            epsg = crs.to_epsg()
            if epsg:
                return f"EPSG:{epsg}"
            return wkt
        except Exception:
            return None

    # Legacy pyproj doesn't have CRS.from_wkt
    return None


def crs_to_epsg(crs: Union[str, "pyproj.CRS"]) -> Optional[int]:
    """Convert a CRS to its EPSG code if possible.

    Parameters
    ----------
    crs : str or pyproj.CRS
        CRS as string or pyproj CRS object.

    Returns
    -------
    Optional[int]
        EPSG code or None if not determinable.
    """
    if isinstance(crs, str):
        if crs.upper().startswith("EPSG:"):
            try:
                return int(crs.split(":")[1])
            except (ValueError, IndexError):
                return None

        if HAS_PYPROJ:
            try:
                return pyproj.CRS.from_user_input(crs).to_epsg()
            except Exception:
                return None
        # Legacy pyproj doesn't have CRS.from_user_input
        return None

    if HAS_PYPROJ and hasattr(crs, "to_epsg"):
        return crs.to_epsg()

    return None


def _crs_equal(crs1: str, crs2: str) -> bool:
    """Check if two CRS strings refer to the same coordinate system."""
    if crs1 == crs2:
        return True

    # Compare EPSG codes
    code1 = crs_to_epsg(crs1)
    code2 = crs_to_epsg(crs2)
    if code1 is not None and code2 is not None:
        return code1 == code2

    return False


def get_crs_from_rasterio(path: str) -> Optional[str]:
    """Get CRS from a rasterio dataset (GeoTIFF, etc.).

    Parameters
    ----------
    path : str
        Path to raster file.

    Returns
    -------
    Optional[str]
        EPSG code or WKT, or None if not determinable.
    """
    try:
        import rasterio
        with rasterio.open(path) as ds:
            if ds.crs is None:
                return None
            # Try to get EPSG code
            epsg = ds.crs.to_epsg()
            if epsg:
                return f"EPSG:{epsg}"
            # Fall back to WKT
            return ds.crs.to_wkt()
    except Exception:
        return None


def get_crs_from_geopandas(path: str, layer: Optional[str] = None) -> Optional[str]:
    """Get CRS from a geopandas-readable file.

    Parameters
    ----------
    path : str
        Path to vector file (GeoJSON, Shapefile, GeoPackage, etc.).
    layer : Optional[str]
        Layer name for multi-layer datasets.

    Returns
    -------
    Optional[str]
        EPSG code or WKT, or None if not determinable.
    """
    try:
        import geopandas as gpd
        gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
        if gdf.crs is None:
            return None
        epsg = gdf.crs.to_epsg()
        if epsg:
            return f"EPSG:{epsg}"
        return gdf.crs.to_wkt()
    except Exception:
        return None


__all__ = [
    "proj_available",
    "transform_coords",
    "reproject_geom",
    "parse_crs_from_wkt",
    "crs_to_epsg",
    "get_crs_from_rasterio",
    "get_crs_from_geopandas",
]
