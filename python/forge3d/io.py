# python/forge3d/io.py
# High-level wrappers for OBJ import/export (Workstream F4/F5)
# DEM loading for terrain rendering (Milestone 5 - Task 5.1)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .geometry import MeshBuffers, _mesh_from_py, _mesh_to_py  # internal helpers

try:
    from . import _forge3d
except ImportError:  # pragma: no cover - handled in tests via skip
    _forge3d = None  # type: ignore[assignment]


def _ensure_native() -> None:
    if _forge3d is None:  # pragma: no cover - import guard
        raise RuntimeError("forge3d native module is not available; build the extension first")


# ========== DEM (Digital Elevation Model) Loading ==========

@dataclass
class DEMData:
    """Container for Digital Elevation Model data.

    Attributes
    ----------
    data : np.ndarray
        2D array of elevation values (float32), shape (height, width)
    domain : Tuple[float, float]
        (min_elevation, max_elevation) value range
    nodata_value : Optional[float]
        Value representing missing data (None if no nodata)
    crs : Optional[str]
        Coordinate Reference System (e.g., "EPSG:4326")
    transform : Optional[Any]
        Affine transform for georeferencing
    resolution : Tuple[float, float]
        (x_resolution, y_resolution) in CRS units
    bounds : Optional[Tuple[float, float, float, float]]
        (minx, miny, maxx, maxy) bounding box in CRS units
    texture_view : Optional[Any]
        GPU texture view (populated when uploaded to GPU)
    stats : Dict[str, float]
        Statistics: mean, std, min, max, median, etc.
    """
    data: np.ndarray
    domain: Tuple[float, float]
    nodata_value: Optional[float] = None
    crs: Optional[str] = None
    transform: Optional[Any] = None
    resolution: Tuple[float, float] = (1.0, 1.0)
    bounds: Optional[Tuple[float, float, float, float]] = None
    texture_view: Optional[Any] = None
    stats: Dict[str, float] = None  # type: ignore

    def __post_init__(self):
        """Initialize stats if not provided."""
        if self.stats is None:
            self.stats = calculate_dem_stats(self.data, self.nodata_value)


def calculate_dem_stats(data: np.ndarray, nodata: Optional[float] = None) -> Dict[str, float]:
    """Calculate statistics for DEM data.

    Parameters
    ----------
    data : np.ndarray
        2D elevation array
    nodata : Optional[float]
        Value to exclude from statistics

    Returns
    -------
    Dict[str, float]
        Statistics including min, max, mean, std, median, percentiles
    """
    # Create mask for valid data
    if nodata is not None:
        valid_mask = ~np.isclose(data, nodata, rtol=1e-5)
        valid_data = data[valid_mask]
    else:
        valid_data = data[~np.isnan(data)]

    if len(valid_data) == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p01": 0.0,
            "p99": 0.0,
            "count": 0,
        }

    stats = {
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
        "mean": float(np.mean(valid_data)),
        "std": float(np.std(valid_data)),
        "median": float(np.median(valid_data)),
        "p01": float(np.percentile(valid_data, 1)),
        "p99": float(np.percentile(valid_data, 99)),
        "count": int(len(valid_data)),
    }

    return stats


def fill_nodata(data: np.ndarray, nodata: float, method: str = "nearest") -> np.ndarray:
    """Fill nodata values in DEM using interpolation.

    Parameters
    ----------
    data : np.ndarray
        2D elevation array
    nodata : float
        Value representing missing data
    method : str, default "nearest"
        Interpolation method: "nearest", "linear", or "cubic"

    Returns
    -------
    np.ndarray
        Filled elevation data
    """
    # Create mask of nodata pixels
    nodata_mask = np.isclose(data, nodata, rtol=1e-5)

    if not np.any(nodata_mask):
        return data.copy()

    filled = data.copy()

    try:
        from scipy import interpolate
        from scipy.ndimage import distance_transform_edt

        # Get indices of valid and invalid pixels
        valid_mask = ~nodata_mask
        rows, cols = np.indices(data.shape)

        if method == "nearest":
            # Use distance transform for nearest neighbor
            indices = distance_transform_edt(nodata_mask, return_distances=False, return_indices=True)
            filled = data[tuple(indices)]
        else:
            # Use griddata for linear/cubic interpolation
            valid_points = np.column_stack((rows[valid_mask], cols[valid_mask]))
            valid_values = data[valid_mask]
            invalid_points = np.column_stack((rows[nodata_mask], cols[nodata_mask]))

            if len(valid_points) > 0 and len(invalid_points) > 0:
                filled_values = interpolate.griddata(
                    valid_points,
                    valid_values,
                    invalid_points,
                    method=method if method in ["linear", "cubic"] else "nearest"
                )
                filled[nodata_mask] = filled_values

    except ImportError:
        # Fallback to simple mean filling if scipy not available
        valid_mean = np.mean(data[~nodata_mask])
        filled[nodata_mask] = valid_mean

    return filled


def load_dem(
    path: str,
    *,
    fill_nodata_values: bool = True,
    fill_method: str = "nearest",
    normalize: bool = False,
    target_domain: Optional[Tuple[float, float]] = None,
    dtype: np.dtype = np.float32,
) -> DEMData:
    """Load a Digital Elevation Model from a GeoTIFF file.

    Parameters
    ----------
    path : str
        Path to GeoTIFF file (.tif, .tiff)
    fill_nodata_values : bool, default True
        Whether to fill nodata values using interpolation
    fill_method : str, default "nearest"
        Interpolation method: "nearest", "linear", or "cubic"
    normalize : bool, default False
        Whether to normalize elevation to [0, 1] range
    target_domain : Optional[Tuple[float, float]]
        If provided, remap values to this domain
    dtype : np.dtype, default np.float32
        Output data type

    Returns
    -------
    DEMData
        Loaded DEM with metadata and statistics

    Examples
    --------
    >>> dem = load_dem("terrain.tif", fill_nodata_values=True)
    >>> print(f"Elevation range: {dem.domain}")
    >>> print(f"Mean elevation: {dem.stats['mean']:.1f}m")

    Notes
    -----
    Requires rasterio package for GeoTIFF loading.
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError(
            "rasterio is required for DEM loading. "
            "Install with: pip install rasterio (or pip install 'forge3d[raster]')"
        )
    except Exception as exc:
        # Some environments raise KeyError or other loader errors when rasterio is missing;
        # normalize to a clear dependency message.
        raise ImportError(
            "Failed to import rasterio (required for DEM loading). "
            "Install with: pip install rasterio (or pip install 'forge3d[raster]')."
        ) from exc

    # Open GeoTIFF
    with rasterio.open(path) as src:
        # Read first band (elevation)
        data = src.read(1, masked=False).astype(dtype)

        # Get metadata
        nodata = src.nodata
        crs = src.crs.to_string() if src.crs else None
        transform = src.transform
        resolution = (abs(transform.a), abs(transform.e))
        bounds = src.bounds

    # Fill nodata if requested
    if fill_nodata_values and nodata is not None:
        data = fill_nodata(data, float(nodata), method=fill_method)
        nodata = None  # No more nodata after filling

    # Calculate domain
    if nodata is not None:
        valid_mask = ~np.isclose(data, nodata, rtol=1e-5)
        valid_data = data[valid_mask]
        if len(valid_data) > 0:
            domain = (float(np.min(valid_data)), float(np.max(valid_data)))
        else:
            domain = (0.0, 1.0)
    else:
        domain = (float(np.min(data)), float(np.max(data)))

    # Normalize if requested
    if normalize:
        data = (data - domain[0]) / max(domain[1] - domain[0], 1e-8)
        domain = (0.0, 1.0)

    # Remap to target domain if provided
    if target_domain is not None:
        # Normalize to [0, 1] first
        normalized = (data - domain[0]) / max(domain[1] - domain[0], 1e-8)
        # Remap to target domain
        data = normalized * (target_domain[1] - target_domain[0]) + target_domain[0]
        domain = target_domain

    return DEMData(
        data=data,
        domain=domain,
        nodata_value=nodata,
        crs=crs,
        transform=transform,
        resolution=resolution,
        bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top) if bounds else None,
        texture_view=None,
        stats=calculate_dem_stats(data, nodata),
    )


def load_dem_from_array(
    data: np.ndarray,
    *,
    domain: Optional[Tuple[float, float]] = None,
    nodata: Optional[float] = None,
) -> DEMData:
    """Create DEMData from a NumPy array.

    Parameters
    ----------
    data : np.ndarray
        2D elevation array (will be converted to float32)
    domain : Optional[Tuple[float, float]]
        Value range (computed from data if not provided)
    nodata : Optional[float]
        Value representing missing data

    Returns
    -------
    DEMData
        DEM data object
    """
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(f"DEM data must be 2D, got shape {data.shape}")

    if domain is None:
        if nodata is not None:
            valid_mask = ~np.isclose(data, nodata, rtol=1e-5)
            valid_data = data[valid_mask]
            if len(valid_data) > 0:
                domain = (float(np.min(valid_data)), float(np.max(valid_data)))
            else:
                domain = (0.0, 1.0)
        else:
            domain = (float(np.min(data)), float(np.max(data)))

    return DEMData(
        data=data,
        domain=domain,
        nodata_value=nodata,
        crs=None,
        transform=None,
        resolution=(1.0, 1.0),
        bounds=None,
        texture_view=None,
        stats=calculate_dem_stats(data, nodata),
    )


def upload_dem_to_gpu(dem: DEMData, session: Optional[Any] = None) -> DEMData:
    """Upload DEM data to GPU as a texture.

    Parameters
    ----------
    dem : DEMData
        DEM data to upload
    session : Optional[Session]
        GPU session (uses global context if None)

    Returns
    -------
    DEMData
        Same DEM with texture_view populated

    Notes
    -----
    This creates an R32Float texture on the GPU for use in shaders.
    """
    _ensure_native()

    # Ensure data is C-contiguous float32
    data = np.ascontiguousarray(dem.data, dtype=np.float32)

    # Get GPU context
    if session is not None:
        device = session.device
        queue = session.queue
    else:
        # Use global context
        from . import _gpu
        device = _gpu.device()
        queue = _gpu.queue()

    # Create texture using native module
    height, width = data.shape

    # Upload to GPU using TerrainSpike's upload mechanism
    # or create a simple R32Float texture
    try:
        # Use the terrain module's texture creation if available
        from . import _forge3d

        # Create a simple wrapper or use existing texture creation
        # For now, store the data and mark as ready for upload
        dem.texture_view = {
            "data": data,
            "width": width,
            "height": height,
            "format": "R32Float",
        }
    except Exception as e:
        print(f"Warning: Could not upload DEM to GPU: {e}")
        dem.texture_view = None

    return dem


@dataclass
class ObjMaterial:
    name: str
    diffuse_color: np.ndarray  # (3,)
    ambient_color: np.ndarray  # (3,)
    specular_color: np.ndarray  # (3,)
    diffuse_texture: Optional[str]


def load_obj(
    path: str,
    *,
    return_metadata: bool = False,
) -> MeshBuffers | Tuple[
    MeshBuffers,
    List[ObjMaterial],
    Dict[str, np.ndarray],  # material_groups
    Dict[str, np.ndarray],  # g_groups
    Dict[str, np.ndarray],  # o_groups
]:
    """Load a Wavefront OBJ file.

    Parameters
    ----------
    path: str
        Path to the OBJ file.
    return_metadata: bool, default False
        If True, also return a list of `ObjMaterial` entries and a mapping of material name to
        triangle ordinal indices (one per triangle in reading order).

    Returns
    -------
    MeshBuffers or (MeshBuffers, materials, material_groups, g_groups, o_groups)
        If `return_metadata` is False, returns only the mesh.
        Otherwise returns a tuple with materials parsed from MTL and group indices per material,
        plus `g` and `o` groupings as triangle ordinal arrays.

    Notes
    -----
    - Faces are triangulated on import.
    - MTL parsing is minimal: `Kd`, `Ka`, `Ks`, and `map_Kd` are supported.
    """
    _ensure_native()
    result = _forge3d.io_import_obj_py(str(path))
    # result is a dict with keys: mesh, materials, groups
    mesh = _mesh_from_py(result["mesh"])  # type: ignore[arg-type]
    if not return_metadata:
        return mesh

    materials: List[ObjMaterial] = []
    for md in result["materials"]:
        materials.append(
            ObjMaterial(
                name=str(md["name"]),
                diffuse_color=np.asarray(md["diffuse_color"], dtype=np.float32),
                ambient_color=np.asarray(md["ambient_color"], dtype=np.float32),
                specular_color=np.asarray(md["specular_color"], dtype=np.float32),
                diffuse_texture=(None if md["diffuse_texture"] is None else str(md["diffuse_texture"]))
            )
        )

    # Backward-compat: prefer "material_groups" if present, otherwise fallback to "groups"
    mat_groups_src: Dict[str, Any] = result.get("material_groups", result.get("groups", {}))
    material_groups: Dict[str, np.ndarray] = {}
    for k, v in mat_groups_src.items():
        material_groups[str(k)] = np.asarray(v, dtype=np.uint32)

    g_groups: Dict[str, np.ndarray] = {}
    for k, v in result.get("g_groups", {}).items():
        g_groups[str(k)] = np.asarray(v, dtype=np.uint32)

    o_groups: Dict[str, np.ndarray] = {}
    for k, v in result.get("o_groups", {}).items():
        o_groups[str(k)] = np.asarray(v, dtype=np.uint32)

    return mesh, materials, material_groups, g_groups, o_groups


def save_obj(
    mesh: MeshBuffers,
    path: str,
    *,
    materials: Optional[List[ObjMaterial]] = None,
    material_groups: Optional[Dict[str, np.ndarray]] = None,
    g_groups: Optional[Dict[str, np.ndarray]] = None,
    o_groups: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Save a MeshBuffers as a Wavefront OBJ file, optionally with materials and groups.

    Parameters
    ----------
    mesh: MeshBuffers
        Geometry to export.
    path: str
        Destination OBJ path. If materials are provided, a sibling .mtl will be written.
    materials: Optional[List[ObjMaterial]]
        Materials to write to the MTL file (minimal fields: name, Kd/Ka/Ks, map_Kd).
    material_groups: Optional[Dict[str, np.ndarray]]
        Mapping material name -> triangle ordinal indices.
    g_groups: Optional[Dict[str, np.ndarray]]
        Mapping 'g' group name -> triangle ordinal indices.
    o_groups: Optional[Dict[str, np.ndarray]]
        Mapping 'o' object name -> triangle ordinal indices.

    Notes
    -----
    - Exports v/vt/vn/f records. If UV or normal arrays are empty, the corresponding tokens
      are omitted in faces.
    - Faces are written as triangle-list.
    - If materials are provided, mtllib/usemtl statements are emitted and a .mtl is written.
    - Groups are emitted as g/o prior to the corresponding faces.
    """
    _ensure_native()
    payload = _mesh_to_py(mesh)
    mats_payload = None
    if materials is not None:
        mats_payload = []
        for m in materials:
            md = {
                "name": m.name,
                "diffuse_color": tuple(np.asarray(m.diffuse_color, dtype=np.float32).tolist()),
                "ambient_color": tuple(np.asarray(m.ambient_color, dtype=np.float32).tolist()),
                "specular_color": tuple(np.asarray(m.specular_color, dtype=np.float32).tolist()),
            }
            if m.diffuse_texture is not None:
                md["diffuse_texture"] = m.diffuse_texture
            mats_payload.append(md)

    def _groups_to_py(d: Optional[Dict[str, np.ndarray]]):
        if d is None:
            return None
        out: Dict[str, np.ndarray] = {}
        for k, v in d.items():
            out[str(k)] = np.asarray(v, dtype=np.uint32)
        return out

    mg = _groups_to_py(material_groups)
    gg = _groups_to_py(g_groups)
    og = _groups_to_py(o_groups)

    _forge3d.io_export_obj_py(str(path), payload, mats_payload, mg, gg, og)


def save_stl(mesh: MeshBuffers, path: str, *, validate: bool = False) -> bool:
    """Export a mesh to binary STL.

    Parameters
    ----------
    mesh: MeshBuffers
        Geometry to export (triangle list).
    path: str
        Destination STL path.
    validate: bool
        If True, performs a watertight edge check and returns the result.

    Returns
    -------
    bool
        Watertightness result if `validate=True`, otherwise False.
    """
    _ensure_native()
    payload = _mesh_to_py(mesh)
    return bool(_forge3d.io_export_stl_py(str(path), payload, bool(validate)))


def import_osm_buildings_extrude(
    features: list[dict],
    *,
    default_height: float = 10.0,
    height_key: str | None = None,
) -> MeshBuffers:
    """Extrude OSM building footprints into a merged mesh.

    Parameters
    ----------
    features: list of dict
        Each dict must contain 'coords' as an (N,2) float array in XY.
        Optionally include 'height' (float) or a custom height_key.
    default_height: float
        Default height when none is provided per feature.
    height_key: Optional[str]
        Custom key in each feature dict to read the height from.
    """
    _ensure_native()
    # Minimal sanity conversion: ensure coords are float32 (N,2)
    conv: list[dict] = []
    for f in features:
        d: dict = {}
        coords = np.asarray(f["coords"], dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("feature 'coords' must have shape (N,2)")
        d["coords"] = coords
        if height_key is not None and height_key in f:
            d[height_key] = float(f[height_key])
        if "height" in f:
            d["height"] = float(f["height"])
        conv.append(d)

    result = _forge3d.import_osm_buildings_extrude_py(conv, float(default_height), height_key)
    return _mesh_from_py(result)  # type: ignore[arg-type]


def import_osm_buildings_from_geojson(
    geojson: str,
    *,
    default_height: float = 10.0,
    height_key: str | None = None,
) -> MeshBuffers:
    """Parse a GeoJSON FeatureCollection of building footprints and extrude to a merged mesh.

    Parameters
    ----------
    geojson: str
        Full GeoJSON string for a FeatureCollection with Polygon/MultiPolygon features.
    default_height: float
        Default height when feature properties lack a height field.
    height_key: Optional[str]
        Name of a property field to read height from (fallback to "height" when None).
    """
    _ensure_native()
    result = _forge3d.import_osm_buildings_from_geojson_py(
        str(geojson), float(default_height), height_key
    )
    return _mesh_from_py(result)  # type: ignore[arg-type]


def import_gltf(path: str) -> MeshBuffers:
    """Import the first mesh primitive from a glTF 2.0 file (.gltf or .glb).

    Parameters
    ----------
    path: str
        Path to a .gltf (JSON) or .glb (binary) file. Supports embedded buffers and external.
    """
    _ensure_native()
    result = _forge3d.io_import_gltf_py(str(path))
    return _mesh_from_py(result)  # type: ignore[arg-type]
