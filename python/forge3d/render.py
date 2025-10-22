"""
python/forge3d/render.py
High-level rendering entry points (rayshader-like convenience API).

Currently provides:
- render_raster: load or accept a DEM heightmap, triangulate to a mesh, auto-frame a camera,
  and render via the native GPU mesh path tracer when available, with a CPU fallback.

Dependencies: only runtime imports; safe to import in CPU-only environments.
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import importlib
import io
import math
import os as _os
import warnings

import numpy as np

try:  # Python <3.8 compatibility for typing.Literal
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal  # type: ignore

from .path_tracing import PathTracer, make_camera, render_aovs, save_aovs as _save_aovs

# -----------------------------------------------------------------------------
# Optional progress helper (enabled when FORGE3D_PROGRESS env var is set)
# -----------------------------------------------------------------------------
def _make_progress(total: int, desc: str):  # pragma: no cover
    if not _os.environ.get("FORGE3D_PROGRESS"):
        class _Noop:
            def update(self, n: int = 1):
                pass
            def write(self, msg: str):
                pass
            def close(self):
                pass
        return _Noop()
    try:
        from tqdm.auto import tqdm  # type: ignore
        class _Wrap:
            def __init__(self, total: int, desc: str) -> None:
                self._bar = tqdm(total=total, desc=desc, unit="step")
            def update(self, n: int = 1) -> None:
                self._bar.update(n)
            def write(self, msg: str) -> None:
                try:
                    from tqdm import write as _tw  # type: ignore
                    _tw(str(msg))
                except Exception:
                    print(str(msg), flush=True)
            def close(self) -> None:
                self._bar.close()
        return _Wrap(total, desc)
    except Exception:
        class _Text:
            def __init__(self, total: int, desc: str) -> None:
                self.total = total; self.n = 0; self.desc = desc
            def update(self, n: int = 1) -> None:
                self.n += n
                print(f"[{self.desc}] {self.n}/{self.total}", flush=True)
            def write(self, msg: str) -> None:
                print(msg, flush=True)
            def close(self) -> None:
                pass
        return _Text(total, desc)


# -----------------------------------------------------------------------------
# Vector file ingestion (.shp, .geojson, .gpkg, .gdb) to polygon rings
# -----------------------------------------------------------------------------
def _load_polygons_from_vector(source: Union[str, Path], layer: Optional[str] = None) -> list[dict]:
    try:
        import geopandas as gpd  # type: ignore
        from shapely.geometry import Polygon, MultiPolygon  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Geospatial support requires geopandas and shapely. "
            "Install with: pip install geopandas shapely fiona"
        ) from exc

    src = str(source)
    gdf = gpd.read_file(src, layer=layer) if layer is not None else gpd.read_file(src)
    if gdf.empty:
        raise ValueError(f"No polygon features found in {src}")

    out: list[dict] = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        if isinstance(geom, Polygon):
            polys = [geom]
        elif isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
        else:
            # Skip non-polygonal geometries
            continue
        for poly in polys:
            ext = _ensure_closed_ring(np.asarray(poly.exterior.coords, dtype=np.float64))
            holes = []
            for ring in poly.interiors:
                holes.append(_ensure_closed_ring(np.asarray(ring.coords, dtype=np.float64)))
            out.append({"exterior": ext, "holes": holes})
    if not out:
        raise ValueError(f"No polygon rings extracted from {src}")
    return out

# Optional native extension access happens lazily inside functions


def _load_dem(src_path: Path) -> tuple[np.ndarray, tuple[float, float]]:
    """Load a DEM from GeoTIFF using rasterio, returning (data, (sx, sy))."""
    # Try regular import first; some environments may shadow with a stub
    rio = None
    try:  # type: ignore[assignment]
        import rasterio as _rio  # type: ignore
        rio = _rio if hasattr(_rio, "open") else None
    except Exception:
        rio = None

    if rio is None:
        # Fallback: explicitly locate site-packages rasterio and import as a separate module
        try:
            import sys as _sys  # type: ignore
            import site as _site  # type: ignore
            from importlib.util import spec_from_file_location as _spec_loc  # type: ignore
            from importlib.util import module_from_spec as _mod_from_spec  # type: ignore
            from pathlib import Path as _P  # type: ignore

            candidates: list[str] = []
            try:
                candidates.extend(_site.getsitepackages())  # type: ignore
            except Exception:
                pass
            try:
                candidates.append(_site.getusersitepackages())  # type: ignore
            except Exception:
                pass
            real_init = None
            real_pkg_path = None
            for base in candidates:
                p = _P(base) / "rasterio" / "__init__.py"
                if p.is_file():
                    real_init = str(p)
                    real_pkg_path = str(p.parent)
                    break
            if real_init:
                spec = _spec_loc("rasterio_real", real_init, submodule_search_locations=[real_pkg_path])
                if spec and spec.loader:
                    mod = _mod_from_spec(spec)
                    _sys.modules["rasterio_real"] = mod  # register under alternate name
                    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                    if hasattr(mod, "open"):
                        rio = mod  # type: ignore[assignment]
        except Exception:
            rio = None

    if rio is None:
        raise ImportError("rasterio is required to load GeoTIFF DEMs. pip install rasterio")

    with rio.open(str(src_path)) as ds:  # type: ignore[attr-defined]
        band1 = ds.read(1, masked=True)
        data = np.array(band1.filled(np.nan), dtype=np.float32)
        if np.isnan(data).any():
            finite = data[np.isfinite(data)]
            fill_val = float(np.min(finite)) if finite.size else 0.0
            data = np.nan_to_num(data, nan=fill_val)
        try:
            sx = float(ds.transform.a)
            sy = float(-ds.transform.e)
        except Exception:
            sx, sy = 1.0, 1.0
        sx = abs(sx) or 1.0
        sy = abs(sy) or 1.0
        return data, (sx, sy)


def load_dem(src_path: Union[str, Path]) -> tuple[np.ndarray, tuple[float, float]]:
    """Public helper: load a DEM from GeoTIFF and return (heightmap32, (sx, sy)).

    This wraps the internal ``_load_dem`` to provide a stable import for examples.
    """
    return _load_dem(Path(src_path))


def _compute_vertex_colors(
    hm: np.ndarray,
    palette: Union[str, Sequence[str], np.ndarray, None] = None,
    invert_palette: bool = False,
    contrast_pct: float = 1.0,
    gamma: float = 1.1,
    equalize: bool = True,
) -> np.ndarray:
    """Compute per-vertex RGB colors from heightmap using palette mapping.
    
    Returns: (H*W, 3) float32 array of RGB colors in range [0, 1]
    """
    H, W = hm.shape
    
    # Normalize heightmap to [0, 1]
    norm01 = _normalize_robust(hm, pct=float(contrast_pct))
    
    # Apply gamma correction
    gval = float(gamma)
    if np.isfinite(gval) and abs(gval - 1.0) > 1e-3:
        norm01 = np.clip(norm01, 0.0, 1.0) ** (1.0 / gval)
    
    # Apply histogram equalization
    if bool(equalize):
        norm01 = _equalize01(norm01)
    
    # Map to palette
    palette_table = _resolve_palette(palette, invert=invert_palette, entries=256)
    idx = np.clip((norm01 * (palette_table.shape[0] - 1)).astype(np.int32), 0, palette_table.shape[0] - 1)
    rgb = palette_table[idx, :3].astype(np.float32) / 255.0
    
    # Flatten to per-vertex colors
    colors = rgb.reshape(-1, 3)
    return colors


def _heightmap_to_mesh(hm: np.ndarray, spacing: tuple[float, float], z_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a heightmap grid to positions (N,3) and indices (M,3).

    The grid is laid out so that X increases with column index and Z increases with row index,
    using the provided pixel spacing. Y is elevation * z_scale. We output positions as [X, Y, Z]
    where Y is the 'up' axis (standard coordinate system for 3D graphics).
    """
    H, W = hm.shape
    sx, sy = float(spacing[0]), float(spacing[1])
    xs = np.arange(W, dtype=np.float32) * sx
    zs = np.arange(H, dtype=np.float32) * sy
    X, Z = np.meshgrid(xs, zs)
    Y = hm.astype(np.float32) * float(z_scale)
    V = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    i = np.arange(H * W, dtype=np.uint32).reshape(H, W)
    i00 = i[:-1, :-1]  # top-left of quad
    i10 = i[1:, :-1]   # bottom-left (Z increases down)
    i01 = i[:-1, 1:]   # top-right (X increases right)
    i11 = i[1:, 1:]    # bottom-right
    # Counter-clockwise winding for Y-up coordinate system when viewed from above
    # Triangle 1: i00 → i01 → i11 (top-left → top-right → bottom-right)
    # Triangle 2: i00 → i11 → i10 (top-left → bottom-right → bottom-left)
    t0 = np.stack([i00, i01, i11], axis=-1).reshape(-1, 3)
    t1 = np.stack([i00, i11, i10], axis=-1).reshape(-1, 3)
    F = np.ascontiguousarray(np.concatenate([t0, t1], axis=0).astype(np.uint32))
    return V, F


def heightmap_to_mesh(
    heightmap: np.ndarray,
    spacing: tuple[float, float],
    *,
    vertical_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Public helper: convert a heightmap into triangle mesh vertices/indices.

    Args:
        heightmap: 2D array (H,W) float32 elevations
        spacing: (sx, sy) pixel spacing
        vertical_scale: multiplier applied to elevation values
    """
    return _heightmap_to_mesh(heightmap, spacing, z_scale=float(vertical_scale))


class RaytraceMeshCache:
    """Cache for lazily building a mesh suitable for raytracing.

    Useful for interactive viewers where (heightmap, spacing) are known and a mesh is
    only needed when the user requests a raytrace. Supports optional subsampling and
    vertical exaggeration.
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        spacing: tuple[float, float],
        *,
        subsample: int = 1,
        vertical_scale: float = 1.0,
    ) -> None:
        self._heightmap = np.asarray(heightmap)
        self._spacing = (float(spacing[0]), float(spacing[1]))
        self._subsample = max(1, int(subsample))
        self._vertical_scale = float(vertical_scale)
        self._cached: Optional[tuple[np.ndarray, np.ndarray]] = None

    def get_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        if self._cached is None:
            hm = self._heightmap[:: self._subsample, :: self._subsample]
            spacing = (
                self._spacing[0] * self._subsample,
                self._spacing[1] * self._subsample,
            )
            self._cached = heightmap_to_mesh(
                hm,
                spacing,
                vertical_scale=self._vertical_scale,
            )
        return self._cached


def _autoframe_camera(V: np.ndarray, size: tuple[int, int], *, fovy: float = 35.0) -> Dict[str, Any]:
    """Auto-frame a camera around the mesh bounds at a pleasing oblique angle."""
    W, H = int(size[0]), int(size[1])
    aspect = float(W) / float(max(1, H))
    minv = V.min(axis=0)
    maxv = V.max(axis=0)
    center = 0.5 * (minv + maxv)
    sizev = (maxv - minv)
    dir_hint = np.array([0.9, 0.6, 1.2], dtype=np.float32)
    dir_hint /= (np.linalg.norm(dir_hint) + 1e-8)
    radius = float(np.linalg.norm(sizev)) * 0.6
    eye = center + dir_hint * radius
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return make_camera(
        origin=tuple(map(float, eye.tolist())),
        look_at=tuple(map(float, center.tolist())),
        up=tuple(map(float, up.tolist())),
        fov_y=float(fovy),
        aspect=float(aspect),
        exposure=1.0,
    )


_DEFAULT_HEX_PALETTE = (
    "#AABD8A",
    "#E6CE99",
    "#D4B388",
    "#C0A181",
    "#AC8D75",
    "#9B7B62",
)


def _hex_to_rgb01(s: str) -> tuple[float, float, float]:
    s = s.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Invalid hex color: {s}")
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b)


def _interpolate_palette_rgba(hex_colors: Sequence[str], n: int = 256) -> np.ndarray:
    if n <= 0:
        raise ValueError("Palette length must be positive")
    cols = np.array([_hex_to_rgb01(h) for h in hex_colors], dtype=np.float32)
    if cols.ndim != 2 or cols.shape[0] < 2:
        raise ValueError("At least two color stops are required to build a palette")
    stops = np.linspace(0.0, 1.0, cols.shape[0], dtype=np.float32)
    t = np.linspace(0.0, 1.0, int(n), dtype=np.float32)
    r = np.interp(t, stops, cols[:, 0])
    g = np.interp(t, stops, cols[:, 1])
    b = np.interp(t, stops, cols[:, 2])
    table = np.empty((t.shape[0], 4), dtype=np.uint8)
    table[:, 0] = (r * 255.0 + 0.5).astype(np.uint8)
    table[:, 1] = (g * 255.0 + 0.5).astype(np.uint8)
    table[:, 2] = (b * 255.0 + 0.5).astype(np.uint8)
    table[:, 3] = 255
    return table


def _resolve_palette(
    palette: Union[str, Sequence[str], np.ndarray, None],
    *,
    invert: bool = False,
    entries: int = 256,
) -> np.ndarray:
    table: np.ndarray
    if palette is None:
        palette = _DEFAULT_HEX_PALETTE

    if isinstance(palette, str):
        try:
            module_name = f"{__package__}.colormap" if __package__ else "forge3d.colormap"
            colormap_mod = importlib.import_module(module_name)
            data = colormap_mod.decode_png_rgba8(palette)
            table = np.frombuffer(data, dtype=np.uint8).reshape(-1, 4)
        except Exception:
            table = _interpolate_palette_rgba(_DEFAULT_HEX_PALETTE, n=entries)
    elif isinstance(palette, np.ndarray):
        table = np.asarray(palette, dtype=np.uint8)
        if table.ndim != 2 or table.shape[1] not in (3, 4):
            raise ValueError("Palette array must have shape (N,3) or (N,4)")
        if table.shape[1] == 3:
            alpha = np.full((table.shape[0], 1), 255, dtype=np.uint8)
            table = np.concatenate([table, alpha], axis=1)
    else:
        table = _interpolate_palette_rgba(list(palette), n=entries)

    if table.shape[0] != entries:
        idx = np.linspace(0.0, table.shape[0] - 1, entries, dtype=np.float32)
        base_idx = np.arange(table.shape[0], dtype=np.float32)
        base = table.astype(np.float32)
        r = np.interp(idx, base_idx, base[:, 0])
        g = np.interp(idx, base_idx, base[:, 1])
        b = np.interp(idx, base_idx, base[:, 2])
        a = np.interp(idx, base_idx, base[:, 3])
        table = np.stack([r, g, b, a], axis=-1).astype(np.uint8)

    if invert:
        table = table[::-1, :]

    return table


def resolve_palette_argument(
    colormap: str,
    *,
    interpolate: bool = False,
    size: int = 256,
    base_colors: Optional[Sequence[str]] = None,
) -> Union[str, Sequence[str], np.ndarray]:
    """Resolve palette argument from a name for examples/CLI usage.

    If ``colormap`` is not "custom" (case-insensitive), returns the colormap name unchanged.
    If it is "custom", returns either the discrete ``base_colors`` sequence (if provided
    or falls back to a small default) or an interpolated RGBA palette (``np.ndarray(N,4)``)
    when ``interpolate=True``.
    """
    if colormap.lower() != "custom":
        return colormap

    hex_list = list(base_colors) if base_colors is not None else list(_DEFAULT_HEX_PALETTE)
    if not interpolate or len(hex_list) < 2:
        return hex_list
    return _interpolate_palette_rgba(hex_list, n=int(size))


def _write_png(path: Union[str, Path], array: np.ndarray) -> None:
    """Save an RGBA numpy array to ``path`` using Pillow when available."""
    path_obj = Path(path)
    if path_obj.suffix.lower() != ".png":
        path_obj = path_obj.with_suffix(".png")
    try:
        from PIL import Image  # type: ignore

        img = Image.fromarray(array, mode="RGBA")
        img.save(str(path_obj))
    except Exception:
        path_obj.write_bytes(array.tobytes())


def _auto_frame_mesh(
    vertices: np.ndarray,
    *,
    target: Optional[Sequence[float]],
    up: Sequence[float],
    fov_y: float,
    aspect: float,
    orbit_theta: Optional[float],
    orbit_phi: Optional[float],
    radius: Optional[float],
    zoom: Optional[float],
    eye_override: Optional[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    import numpy as _np

    verts = _np.asarray(vertices, dtype=_np.float32)
    minv = verts.min(axis=0)
    maxv = verts.max(axis=0)
    
    # For terrain: use representative surface elevation instead of geometric center
    bbox_size = maxv - minv
    x_range = float(max(bbox_size[0], 1e-6))
    y_range = float(max(bbox_size[1], 1e-6))
    z_range = float(max(bbox_size[2], 1e-6))
    horizontal_extent = math.sqrt(x_range * x_range + z_range * z_range)
    vertical_extent = y_range
    # Terrain detection: horizontal extent >> vertical (wide flat mesh)
    is_terrain_shape = horizontal_extent > 1e-3 and horizontal_extent > 2.0 * vertical_extent
    if target is None and is_terrain_shape:
        # Terrain case: use horizontal center but median elevation
        center = (minv + maxv) * 0.5
        center[1] = float(_np.median(verts[:, 1]))  # Use median Y instead of center Y
    else:
        center = (minv + maxv) * 0.5 if target is None else _np.asarray(target, dtype=_np.float32)
    dir_hint_default = _np.array([0.8, 0.6, 1.6], dtype=_np.float32)
    dir_hint = dir_hint_default
    if orbit_theta is not None or orbit_phi is not None:
        theta = math.radians(orbit_theta if orbit_theta is not None else 45.0)
        phi = math.radians(orbit_phi if orbit_phi is not None else 25.0)
        dir_hint = _np.array(
            [
                math.cos(phi) * math.sin(theta),
                math.sin(phi),
                math.cos(phi) * math.cos(theta),
            ],
            dtype=_np.float32,
        )
    dir_hint /= (float(_np.linalg.norm(dir_hint)) + 1e-8)

    if eye_override is not None:
        eye = _np.asarray(eye_override, dtype=_np.float32)
        to_eye = eye - center
        distance = float(_np.linalg.norm(to_eye))
        if distance <= 1e-6:
            distance = 1.0
        dir_hint = to_eye / distance
    else:
        horizontal_extent = math.sqrt(x_range * x_range + z_range * z_range)
        # Terrain: horizontal extent >> vertical (wide flat mesh, even with exaggeration)
        is_terrain = horizontal_extent > 1e-3 and horizontal_extent > 2.0 * vertical_extent

        fov_y_rad = math.radians(fov_y)
        fov_x_rad = 2.0 * math.atan(math.tan(fov_y_rad * 0.5) * max(aspect, 1e-6))
        half_horiz = 0.5 * horizontal_extent
        half_vert = 0.5 * vertical_extent

        # Camera distance calculation
        dist_x = half_horiz / max(math.tan(fov_x_rad * 0.5), 1e-6)
        dist_y = half_vert / max(math.tan(fov_y_rad * 0.5), 1e-6)
        
        if is_terrain:
            # For terrain: use HORIZONTAL extent only to maximize viewport coverage
            # Ignore vertical extent (exaggeration) - camera should frame horizontal extent
            distance = dist_x
        else:
            # Normal mesh: use both extents equally
            distance = max(dist_x, dist_y)
        
        # Add margin to ensure camera is outside mesh bounds
        # Use 1.35x margin for terrain (tested working value)
        margin_factor = 1.35 if is_terrain else 1.25
        distance = max(distance, 1.0) * margin_factor

        if radius is not None:
            distance = float(radius)
        if zoom is not None:
            distance *= float(zoom)
        elif is_terrain:
            distance *= 1.0  # terrain: keep full distance, don't bring closer
        eye = center + dir_hint * distance

    up_vec = _np.asarray(up, dtype=_np.float32)
    up_vec /= (float(_np.linalg.norm(up_vec)) + 1e-8)

    camera = make_camera(
        origin=tuple(map(float, eye.tolist())),
        look_at=tuple(map(float, center.tolist())),
        up=tuple(map(float, up_vec.tolist())),
        fov_y=float(fov_y),
        aspect=float(aspect),
        exposure=1.0,
    )
    return eye, center, up_vec, camera


def render_raytrace_mesh(
    mesh: Union[str, Path, tuple[np.ndarray, np.ndarray], dict],
    *,
    size: tuple[int, int] = (512, 512),
    frames: int = 8,
    seed: int = 7,
    sampling_mode: str = "sobol",
    debug_mode: int = 0,
    fov_y: float = 34.0,
    target: Optional[Sequence[float]] = None,
    up: Sequence[float] = (0.0, 1.0, 0.0),
    orbit_theta: Optional[float] = None,
    orbit_phi: Optional[float] = None,
    radius: Optional[float] = None,
    zoom: Optional[float] = None,
    eye: Optional[Sequence[float]] = None,
    camera: Optional[Dict[str, Any]] = None,
    prefer_gpu: bool = True,
    normalize: bool = True,
    normalize_scale: float = 1.6,
    denoiser: str = "off",
    svgf_iters: int = 5,
    luminance_clamp: Optional[float] = None,
    preview: bool = False,
    preview_size: int = 1,
    preview_color: Sequence[int] = (255, 160, 40),
    palette: Union[str, Sequence[str], np.ndarray, None] = None,
    invert_palette: bool = False,
    lighting_type: str = "lambertian",
    lighting_intensity: float = 1.0,
    lighting_azimuth: float = 315.0,
    lighting_elevation: float = 45.0,
    shadows: bool = True,
    shadow_intensity: float = 1.0,
    hdri_path: Optional[Union[str, Path]] = None,
    hdri_rotation_deg: float = 0.0,
    hdri_intensity: float = 1.0,
    hdri_exposure: float = 1.0,
    background_color: Optional[Sequence[float]] = None,
    save_aovs: bool = False,
    aovs: Sequence[str] = ("albedo", "normal", "depth", "visibility"),
    aov_dir: Optional[Union[str, Path]] = None,
    basename: Optional[str] = None,
    outfile: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Ray trace a triangle mesh with optional post-processing controls."""

    import numpy as _np

    background_rgb: Optional[_np.ndarray] = None
    if background_color is not None:
        bg_arr = _np.asarray(background_color, dtype=_np.float32).reshape(-1)
        if bg_arr.size != 3:
            raise ValueError("background_color must have exactly three components")
        max_val = float(_np.max(bg_arr))
        min_val = float(_np.min(bg_arr))
        if max_val > 1.0 or min_val < 0.0:
            bg_arr = bg_arr / 255.0
        background_rgb = _np.clip(bg_arr, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Mesh ingestion
    # ------------------------------------------------------------------
    if isinstance(mesh, (str, Path)):
        from .io import load_obj

        obj = load_obj(str(mesh))
        vertices = _np.asarray(obj.positions, dtype=_np.float32)
        indices = _np.asarray(obj.indices.reshape(-1, 3), dtype=_np.uint32)
        mesh_name = Path(mesh).name
    elif isinstance(mesh, tuple) and len(mesh) == 2:
        vertices = _np.asarray(mesh[0], dtype=_np.float32)
        indices = _np.ascontiguousarray(mesh[1], dtype=_np.uint32)
        mesh_name = "mesh"
    elif isinstance(mesh, dict) and "vertices" in mesh and "indices" in mesh:
        vertices = _np.asarray(mesh["vertices"], dtype=_np.float32)
        indices = _np.asarray(mesh["indices"], dtype=_np.uint32)
        mesh_name = mesh.get("name", "mesh")
    else:
        raise TypeError("mesh must be an OBJ path, (vertices, indices) tuple, or mesh dict")

    if vertices.ndim != 2 or (vertices.shape[1] != 3 and vertices.shape[1] != 8):
        raise ValueError("vertices must have shape (N,3) or (N,8) for colored vertices")
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices must have shape (M,3)")

    from .mesh import make_mesh, build_bvh_cpu, upload_mesh, validate_mesh_arrays

    # Extract positions for validation and normalization
    has_colors = vertices.shape[1] == 8
    positions = _np.ascontiguousarray(vertices[:, :3]) if has_colors else vertices
    
    validate_mesh_arrays(positions, indices)

    verts_proc = vertices.copy()
    if normalize:
        # Normalize only the position columns (0:3)
        minv = positions.min(axis=0)
        maxv = positions.max(axis=0)
        center = (minv + maxv) * 0.5
        scale = 1.0 / (max(maxv - minv) + 1e-8)
        verts_proc[:, :3] = (positions - center) * float(normalize_scale * scale)

    # For BVH construction, use only positions (N,3)
    positions_proc = _np.ascontiguousarray(verts_proc[:, :3] if has_colors else verts_proc, dtype=_np.float32)
    mesh_dict = make_mesh(positions_proc, _np.ascontiguousarray(indices, dtype=_np.uint32))
    bvh = build_bvh_cpu(mesh_dict, method="median")
    handle = upload_mesh(mesh_dict, bvh)

    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError("size must be positive")

    if camera is not None:
        cam = camera
        eye_vec = _np.asarray(cam["origin"], dtype=_np.float32)
        target_vec = _np.asarray(cam["look_at"], dtype=_np.float32)
        up_vec = _np.asarray(cam.get("up", up), dtype=_np.float32)
    else:
        aspect = float(width) / float(max(1, height))
        # Use only positions for camera framing
        verts_for_camera = verts_proc[:, :3] if has_colors else verts_proc
        eye_vec, target_vec, up_vec, cam = _auto_frame_mesh(
            verts_for_camera,
            target=target,
            up=up,
            fov_y=fov_y,
            aspect=aspect,
            orbit_theta=orbit_theta,
            orbit_phi=orbit_phi,
            radius=radius,
            zoom=zoom,
            eye_override=eye,
        )
    
    # Add lighting parameters to camera dict for GPU path
    cam["lighting_type"] = str(lighting_type)
    cam["lighting_intensity"] = float(lighting_intensity)
    cam["lighting_azimuth"] = float(lighting_azimuth)
    cam["lighting_elevation"] = float(lighting_elevation)
    cam["shadow_intensity"] = float(shadow_intensity) if shadows else 0.0

    # Pass HDRI parameters to GPU tracer via camera dict (expected by native _pt_render_gpu_mesh)
    if hdri_path is not None:
        try:
            cam["hdri_path"] = str(Path(hdri_path))
        except Exception:
            cam["hdri_path"] = str(hdri_path)
    cam["hdri_rotation_deg"] = float(hdri_rotation_deg)
    cam["hdri_intensity"] = float(hdri_intensity)
    cam["hdri_exposure"] = float(hdri_exposure)
    cam["hdri_enabled"] = 1.0 if (hdri_path is not None and float(hdri_intensity) > 0.0) else 0.0

    # Propagate OIDN mode preference to native layer (Final/Tiled/Off)
    # If Python-side denoiser is "off", request GPU OIDN=Off
    try:
        cam["oidn_mode"] = "off" if str(denoiser).lower() in ("off", "none") else "final"
    except Exception:
        cam["oidn_mode"] = "final"

    # ------------------------------------------------------------------
    # AOV request preprocessing (shared by GPU and CPU paths)
    # ------------------------------------------------------------------
    saved_aovs: Optional[Dict[str, Path]] = None
    background_mask: Optional[np.ndarray] = None
    aov_list = [str(a).strip().lower() for a in aovs if str(a).strip()]
    requested_aovs: list[str] = []
    if save_aovs:
        requested_aovs.extend(aov_list)
    if background_rgb is not None and "visibility" not in requested_aovs:
        requested_aovs.append("visibility")

    aov_name_to_bit: Dict[str, int] = {
        "albedo": 1 << 0,
        "normal": 1 << 1,
        "depth": 1 << 2,
        "direct": 1 << 3,
        "indirect": 1 << 4,
        "emission": 1 << 5,
        "visibility": 1 << 6,
    }
    aov_mask = 0
    for name in requested_aovs:
        aov_mask |= aov_name_to_bit.get(name, 0)

    gpu_used = False
    probe_status = "skipped"
    image: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # GPU attempt
    # ------------------------------------------------------------------
    if prefer_gpu:
        try:
            from . import _forge3d as _native  # type: ignore[attr-defined]

            if hasattr(_native, "device_probe"):
                try:
                    probe = _native.device_probe("metal")  # type: ignore[attr-defined]
                    probe_status = str(probe.get("status", "unknown"))
                    gpu_used = probe_status == "ok"
                except Exception as exc:  # pragma: no cover
                    probe_status = f"error:{exc!s}"[:64]

            if gpu_used:
                try:
                    V32 = _np.ascontiguousarray(verts_proc, dtype=_np.float32)
                    F32 = _np.ascontiguousarray(indices.astype(_np.uint32))
                    # Map sampling mode: rng=0, sobol=1, cmj=2
                    sampling_mode_map = {"rng": 0, "sobol": 1, "cmj": 2}
                    sampling_mode_int = sampling_mode_map.get(sampling_mode.lower(), 0)
                    image = _native._pt_render_gpu_mesh(
                        int(width),
                        int(height),
                        V32,
                        F32,
                        cam,
                        int(seed),
                        int(max(1, frames)),
                        sampling_mode_int,
                        int(max(0, debug_mode)),
                        aov_mask=int(aov_mask),
                    )
                except Exception as exc:  # pragma: no cover
                    if verbose:
                        print(f"[render_raytrace_mesh] GPU render failed ({exc!r}); falling back to CPU")
                    gpu_used = False
                    image = None
        except Exception:
            gpu_used = False

    # ------------------------------------------------------------------
    # CPU fallback path tracer
    # ------------------------------------------------------------------
    if image is None:
        tracer = PathTracer()
        image = tracer.render_rgba(
            width,
            height,
            scene=[],
            camera=cam,
            seed=int(seed),
            frames=max(1, int(frames)),
            use_gpu=False,
            mesh=handle,
            denoiser=str(denoiser),
            svgf_iters=int(svgf_iters),
            luminance_clamp=(float(luminance_clamp) if luminance_clamp is not None else None),
        )

    assert image is not None  # for type-checkers

    # ------------------------------------------------------------------
    # Optional point-cloud preview overlay (CPU path only)
    # ------------------------------------------------------------------
    if preview and not gpu_used:
        def _normalize(vec: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-8
            return vec / n

        def _look_at(eye: np.ndarray, tgt: np.ndarray, up_vec: np.ndarray) -> np.ndarray:
            fwd = _normalize(tgt - eye)
            side = _normalize(np.cross(fwd, up_vec))
            upn = np.cross(side, fwd)
            M = np.eye(4, dtype=np.float32)
            M[0, :3] = side
            M[1, :3] = upn
            M[2, :3] = -fwd
            M[0, 3] = -float(np.dot(side, eye))
            M[1, 3] = -float(np.dot(upn, eye))
            M[2, 3] = float(np.dot(fwd, eye))
            return M

        def _perspective(fy: float, asp: float) -> np.ndarray:
            f = 1.0 / math.tan(math.radians(float(fy)) * 0.5)
            M = np.zeros((4, 4), dtype=np.float32)
            M[0, 0] = f / max(asp, 1e-8)
            M[1, 1] = f
            M[2, 2] = -1.0
            M[2, 3] = -1.0
            M[3, 2] = -1.0
            return M

        def _project_points(pts: np.ndarray, view: np.ndarray, proj: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
            pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
            clip = (proj @ (view @ pts_h.T)).T
            w_comp = clip[:, 3:4]
            valid = np.abs(w_comp[:, 0]) > 1e-6
            ndc = np.zeros((pts.shape[0], 3), dtype=np.float32)
            ndc[valid] = clip[valid, :3] / w_comp[valid]
            vis = valid & np.all(np.isfinite(ndc), axis=1) & (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
            xs = np.clip((ndc[:, 0] * 0.5 + 0.5) * (w - 1), 0, w - 1).astype(np.int32)
            ys = np.clip((1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (h - 1), 0, h - 1).astype(np.int32)
            return np.stack([xs, ys], axis=1), vis

        view = _look_at(eye_vec, target_vec, up_vec)
        proj = _perspective(fov_y, float(width) / float(max(1, height)))
        pts, vis = _project_points(verts_proc, view, proj, width, height)
        preview_col = tuple(int(c) for c in preview_color)
        r = max(1, int(preview_size))
        img = np.array(image, copy=True)
        for (x, y), ok in zip(pts, vis):
            if not ok:
                continue
            x0 = max(0, x - r)
            x1 = min(width - 1, x + r)
            y0 = max(0, y - r)
            y1 = min(height - 1, y + r)
            img[y0 : y1 + 1, x0 : x1 + 1, 0] = preview_col[0]
            img[y0 : y1 + 1, x0 : x1 + 1, 1] = preview_col[1]
            img[y0 : y1 + 1, x0 : x1 + 1, 2] = preview_col[2]
            img[y0 : y1 + 1, x0 : x1 + 1, 3] = 255
        image = img
    elif preview and gpu_used and verbose:
        print("[render_raytrace_mesh] GPU render succeeded; preview overlay skipped to preserve shading.")

    # ------------------------------------------------------------------
    # Optional AOVs & visibility mask
    # ------------------------------------------------------------------
    if requested_aovs:
        unique_aovs = tuple(dict.fromkeys(requested_aovs))
        try:
            aov_map = render_aovs(
                width,
                height,
                scene=[],
                camera=cam,
                aovs=unique_aovs,
                seed=int(seed),
                frames=max(1, int(frames)),
                use_gpu=bool(gpu_used),
                mesh=handle,
            )
        except TypeError:
            aov_map = render_aovs(
                width,
                height,
                scene=[],
                camera=cam,
                aovs=unique_aovs,
                seed=int(seed),
                frames=max(1, int(frames)),
                use_gpu=bool(gpu_used),
            )

        if background_rgb is not None:
            mask_candidate: Optional[np.ndarray] = None
            vis_arr = aov_map.get("visibility")
            if vis_arr is not None:
                vis_arr_np = np.asarray(vis_arr, dtype=np.float32)
                if vis_arr_np.ndim == 3 and vis_arr_np.shape[2] == 1:
                    vis_arr_np = vis_arr_np[..., 0]
                if vis_arr_np.shape[:2] == (height, width):
                    mask_candidate = vis_arr_np <= 0.0
                    if mask_candidate is not None and not np.any(mask_candidate):
                        mask_candidate = None

            if mask_candidate is None:
                try:
                    vis_cpu = render_aovs(
                        width,
                        height,
                        scene=[],
                        camera=cam,
                        aovs=("visibility",),
                        seed=int(seed),
                        frames=1,
                        use_gpu=False,
                        mesh=handle,
                    )
                except TypeError:
                    vis_cpu = render_aovs(
                        width,
                        height,
                        scene=[],
                        camera=cam,
                        aovs=("visibility",),
                        seed=int(seed),
                        frames=1,
                        use_gpu=False,
                    )
                except Exception:
                    vis_cpu = None

                if vis_cpu and "visibility" in vis_cpu:
                    vis_cpu_arr = np.asarray(vis_cpu["visibility"], dtype=np.float32)
                    if vis_cpu_arr.ndim == 3 and vis_cpu_arr.shape[2] == 1:
                        vis_cpu_arr = vis_cpu_arr[..., 0]
                    if vis_cpu_arr.shape[:2] == (height, width):
                        mask_candidate = vis_cpu_arr <= 0.0

            if mask_candidate is not None:
                mask_candidate = np.asarray(mask_candidate, dtype=bool)
                if mask_candidate.shape[:2] == (height, width) and np.any(mask_candidate):
                    background_mask = mask_candidate

        if save_aovs and aov_list:
            export_map = {k: aov_map[k] for k in aov_list if k in aov_map}
            if export_map:
                base = basename if basename else (Path(outfile).stem if outfile else mesh_name.rsplit(".", 1)[0])
                out_dir = Path(aov_dir) if aov_dir else (Path(outfile).parent if outfile else Path("."))
                saved = _save_aovs(export_map, str(base), output_dir=str(out_dir))
                if saved:
                    saved_aovs = {k: Path(v) for k, v in saved.items()}
                    if verbose:
                        print("Saved AOVs:")
                        for key, path in saved.items():
                            print(f"  {key}: {path}")

    image_f = np.asarray(image, dtype=np.float32)
    if image_f.ndim == 2:
        image_f = image_f[..., None]
    if image_f.ndim == 3 and image_f.shape[2] == 3:
        alpha = np.ones((image_f.shape[0], image_f.shape[1], 1), dtype=np.float32)
        image_f = np.concatenate([image_f, alpha], axis=-1)
    if image_f.max(initial=0.0) > 1.0 or image_f.min(initial=0.0) < 0.0:
        image_f = np.clip(image_f / 255.0, 0.0, 1.0)
    else:
        image_f = np.clip(image_f, 0.0, 1.0)

    # For GPU renders: detect background pixels to protect from post-processing
    gpu_bg_mask = None
    if gpu_used and background_rgb is not None:
        r = image_f[..., 0]
        g = image_f[..., 1]
        b = image_f[..., 2]
        # Detect pixels matching the background color (magenta was already replaced)
        # Allow small tolerance for floating point comparison
        is_bg = (
            _np.abs(r - background_rgb[0]) < 0.02
        ) & (
            _np.abs(g - background_rgb[1]) < 0.02
        ) & (
            _np.abs(b - background_rgb[2]) < 0.02
        )
        if _np.any(is_bg):
            gpu_bg_mask = is_bg

    processed = _apply_lighting_and_palette(
        image_f,
        palette=palette,
        invert_palette=bool(invert_palette),
        lighting_type=lighting_type,
        lighting_intensity=float(lighting_intensity),
        lighting_azimuth=float(lighting_azimuth),
        lighting_elevation=float(lighting_elevation),
        shadows=bool(shadows),
        shadow_intensity=float(shadow_intensity),
        hdri_path=hdri_path,
        hdri_rotation_deg=float(hdri_rotation_deg),
        hdri_intensity=float(hdri_intensity),
        background_rgb=None,  # Don't replace - GPU already handled it
        background_mask=None,
        gpu_background_mask=gpu_bg_mask,  # Pass mask to protect GPU background
        gpu_background_color=background_rgb if gpu_used else None,
    )

    image_out = (np.clip(processed, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    if outfile is not None:
        _write_png(outfile, image_out)

    if verbose:
        gpu_msg = "GPU" if gpu_used else "CPU fallback"
        if outfile is not None:
            print(f"[render_raytrace_mesh] Saved {outfile} via {gpu_msg} path (tris={len(indices)})")
        else:
            print(f"[render_raytrace_mesh] Rendered {mesh_name} via {gpu_msg} path (tris={len(indices)})")

    meta = {
        "gpu_used": gpu_used,
        "probe_status": probe_status,
        "camera": cam,
        "aov_outputs": saved_aovs,
        "outfile": Path(outfile) if outfile is not None else None,
        "triangles": int(len(indices)),
        "postprocess": {
            "palette": palette,
            "invert_palette": bool(invert_palette),
            "lighting_type": lighting_type,
            "lighting_intensity": float(lighting_intensity),
            "lighting_azimuth": float(lighting_azimuth),
            "lighting_elevation": float(lighting_elevation),
            "shadows": bool(shadows),
            "shadow_intensity": float(shadow_intensity),
            "hdri_path": str(hdri_path) if hdri_path is not None else None,
            "hdri_rotation_deg": float(hdri_rotation_deg),
            "hdri_intensity": float(hdri_intensity),
            "hdri_exposure": float(hdri_exposure),
            "hdri_enabled": float(1.0 if (hdri_path is not None and hdri_intensity > 0.0) else 0.0),
            "background_color": background_rgb.tolist() if background_rgb is not None else None,
            "background_mask_available": bool(background_mask is not None),
        },
    }

    return image_out, meta


def _decode_radiance_hdr(path: Path) -> Optional[np.ndarray]:
    """Decode Radiance RGBE .hdr/.pic files into float32 RGB."""

    try:
        with path.open("rb") as f:
            raw = f.read()
    except Exception as exc:
        warnings.warn(f"Failed to read HDRI map '{path}': {exc}")
        return None

    stream = io.BytesIO(raw)
    header_lines: list[str] = []
    while True:
        line = stream.readline()
        if not line:
            return None
        line_str = line.decode("ascii", errors="ignore").strip()
        if not line_str:
            break
        header_lines.append(line_str)

    if not any(line.upper().startswith("FORMAT=") for line in header_lines):
        return None

    res_line = stream.readline().decode("ascii", errors="ignore").strip()
    tokens = res_line.split()
    if len(tokens) != 4:
        return None

    try:
        if tokens[0].upper().endswith("Y"):
            height = int(tokens[1])
            flip_y = tokens[0].startswith("-")
        else:
            return None
        if tokens[2].upper().endswith("X"):
            width = int(tokens[3])
            flip_x = tokens[2].startswith("-")
        else:
            return None
    except ValueError:
        return None

    if width <= 0 or height <= 0:
        return None

    data = np.empty((height, width, 4), dtype=np.uint8)

    for y in range(height):
        scan_header = stream.read(4)
        if len(scan_header) < 4:
            return None
        if scan_header[0] != 2 or scan_header[1] != 2:
            return None
        scan_width = (scan_header[2] << 8) | scan_header[3]
        if scan_width != width:
            return None

        for channel in range(4):
            x = 0
            while x < width:
                val_byte = stream.read(1)
                if not val_byte:
                    return None
                val = val_byte[0]
                if val > 128:
                    count = val - 128
                    repeat = stream.read(1)
                    if not repeat:
                        return None
                    data[y, x:x + count, channel] = repeat[0]
                    x += count
                else:
                    count = val
                    literals = stream.read(count)
                    if len(literals) != count:
                        return None
                    data[y, x:x + count, channel] = np.frombuffer(literals, dtype=np.uint8)
                    x += count

    rgbe = data.astype(np.float32)
    exponent = rgbe[..., 3].astype(np.int32)
    mantissa = rgbe[..., :3]

    rgb = np.zeros((height, width, 3), dtype=np.float32)
    mask = exponent > 0
    if np.any(mask):
        scale = np.ldexp(1.0, exponent[mask] - 128)
        rgb[mask] = (mantissa[mask] / 256.0) * scale[:, None]

    if flip_y:
        rgb = rgb[::-1, :, :]
    if flip_x:
        rgb = rgb[:, ::-1, :]

    return rgb


def _load_hdri_map(path: Union[str, Path], rotation_deg: float = 0.0) -> Optional[np.ndarray]:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    data: Optional[np.ndarray] = None

    if suffix in {".hdr", ".pic"}:
        data = _decode_radiance_hdr(path_obj)

    if data is None:
        try:
            import imageio.v2 as imageio  # type: ignore

            if suffix == ".hdr":
                data = imageio.imread(str(path_obj), format="HDR-FI")
            else:
                data = imageio.imread(str(path_obj))
        except Exception:
            try:
                from PIL import Image  # type: ignore

                with Image.open(str(path_obj)) as img:
                    data = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            except Exception as exc:
                warnings.warn(f"Failed to load HDRI map '{path_obj}': {exc}")
                return None
        else:
            data = np.asarray(data, dtype=np.float32)
            if data.ndim == 2:
                data = np.stack([data, data, data], axis=-1)
            if data.ndim == 4:
                data = data[..., :3]
            if np.max(data) > 1.0:
                data = data / float(np.max(data))

    if data.ndim != 3 or data.shape[2] < 3:
        warnings.warn(f"HDRI map '{path_obj}' has unexpected shape {data.shape}")
        return None

    width = data.shape[1]
    if width == 0:
        return None
    if rotation_deg:
        shift = int(round((float(rotation_deg) % 360.0) / 360.0 * width))
        if shift:
            data = np.roll(data, shift=shift, axis=1)

    return data[..., :3]


def _apply_lighting_and_palette(
    image: np.ndarray,
    *,
    palette: Optional[Sequence[str]],
    invert_palette: bool,
    lighting_type: str,
    lighting_intensity: float,
    lighting_azimuth: float,
    lighting_elevation: float,
    shadows: bool,
    shadow_intensity: float,
    hdri_path: Optional[Union[str, Path]],
    hdri_rotation_deg: float,
    hdri_intensity: float,
    background_rgb: Optional[np.ndarray],
    background_mask: Optional[np.ndarray],
    gpu_background_mask: Optional[np.ndarray] = None,
    gpu_background_color: Optional[np.ndarray] = None,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] < 3:
        return image

    rgb = np.clip(image[..., :3], 0.0, 1.0)
    original_rgb = rgb.copy()
    
    # Store GPU background color for restoration after all effects
    gpu_bg_preserved = None
    if gpu_background_mask is not None and gpu_background_color is not None:
        gpu_bg_preserved = gpu_background_color.copy()
    if image.shape[2] > 3:
        alpha = np.clip(image[..., 3:4], 0.0, 1.0)
    else:
        alpha = np.ones_like(rgb[..., :1])

    h, w = rgb.shape[:2]
    if h > 0 and w > 0:
        az = math.radians(float(lighting_azimuth))
        el = math.radians(float(lighting_elevation))
        light_dir = np.array(
            [
                math.cos(el) * math.sin(az),
                math.sin(el),
                math.cos(el) * math.cos(az),
            ],
            dtype=np.float32,
        )
        light_dir /= np.linalg.norm(light_dir) + 1e-8

        x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(x, y)
        pseudo_normals = np.stack([grid_x, np.ones_like(grid_x), grid_y], axis=-1)
        pseudo_normals /= np.linalg.norm(pseudo_normals, axis=-1, keepdims=True) + 1e-8
        lambert = np.clip(np.sum(pseudo_normals * light_dir[None, None, :], axis=-1), 0.0, 1.0)

        lt = str(lighting_type or "").lower()
        if lt not in {"none", "flat"}:
            light_gain = 1.0 + (lambert - 0.5) * float(lighting_intensity)
            # Don't apply lighting to GPU background
            if gpu_background_mask is not None:
                light_gain_masked = np.where(gpu_background_mask, 1.0, light_gain)
                rgb = np.clip(rgb * light_gain_masked[..., None], 0.0, 1.0)
            else:
                rgb = np.clip(rgb * light_gain[..., None], 0.0, 1.0)

        # Only apply shadows if lighting is enabled (otherwise creates unbalanced darkening)
        if shadows and float(lighting_intensity) > 0.0:
            sf = np.clip(float(shadow_intensity), 0.0, 1.0)
            shadow_term = 1.0 - sf * (1.0 - lambert)
            # Don't apply shadows to GPU background
            if gpu_background_mask is not None:
                shadow_term_masked = np.where(gpu_background_mask, 1.0, shadow_term)
                rgb = np.clip(rgb * shadow_term_masked[..., None], 0.0, 1.0)
            else:
                rgb = np.clip(rgb * shadow_term[..., None], 0.0, 1.0)

    # Apply palette BEFORE HDRI to preserve dark colors
    if palette is not None or invert_palette:
        table = _resolve_palette(palette, invert=invert_palette, entries=256)
        tablef = table.astype(np.float32) / 255.0
        luminance = np.clip(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2], 0.0, 1.0)
        idx = np.clip((luminance * (tablef.shape[0] - 1)).round().astype(np.int32), 0, tablef.shape[0] - 1)
        rgb_palette = tablef[idx, :3]
        # Don't apply palette to GPU background
        if gpu_background_mask is not None:
            rgb = np.where(gpu_background_mask[..., None], rgb, rgb_palette)
        else:
            rgb = rgb_palette
        if table.shape[1] == 4:
            alpha = tablef[idx, 3:4]

    # Apply HDRI after palette using multiplicative tinting to preserve dark colors
    if hdri_path is not None and hdri_intensity != 0.0:
        env = _load_hdri_map(hdri_path, rotation_deg=hdri_rotation_deg)
        if env is not None:
            env_rgb = env.reshape(-1, env.shape[-1])
            env_color = np.clip(env_rgb.mean(axis=0), 1e-4, None)
            env_color = env_color / float(np.max(env_color))
            # Use multiplicative tinting instead of additive blending to preserve darks
            strength = np.clip(float(hdri_intensity), 0.0, 1.0)
            tint = 1.0 + strength * (env_color[None, None, :3] - 1.0)
            # Don't apply HDRI to GPU background
            if gpu_background_mask is not None:
                rgb_tinted = np.clip(rgb * tint, 0.0, 1.0)
                rgb = np.where(gpu_background_mask[..., None], rgb, rgb_tinted)
            else:
                rgb = np.clip(rgb * tint, 0.0, 1.0)

    if background_rgb is not None:
        mask: Optional[np.ndarray]
        if background_mask is not None:
            mask_candidate = np.asarray(background_mask, dtype=bool)
            if mask_candidate.shape[:2] == rgb.shape[:2]:
                mask = mask_candidate
            else:
                mask = None
        else:
            mask = None

        if mask is None:
            mask = np.all(original_rgb <= 1e-5, axis=-1)

        if mask is not None and np.any(mask):
            rgb[mask] = background_rgb
    
    # Final restoration: ensure GPU background is exactly as rendered
    if gpu_bg_preserved is not None and gpu_background_mask is not None:
        rgb[gpu_background_mask] = gpu_bg_preserved

    out = np.concatenate([np.clip(rgb, 0.0, 1.0), np.clip(alpha, 0.0, 1.0)], axis=-1)
    return out


def _normalize_robust(hm: np.ndarray, pct: float = 2.0) -> np.ndarray:
    h = np.asarray(hm, dtype=np.float32)
    pct_clamped = float(np.clip(pct, 0.0, 50.0))
    if pct_clamped <= 0.0:
        low = float(np.nanmin(h)) if np.isnan(h).any() else float(h.min(initial=0.0))
        high = float(np.nanmax(h)) if np.isnan(h).any() else float(h.max(initial=0.0))
    else:
        low = float(np.percentile(h, pct_clamped))
        high = float(np.percentile(h, 100.0 - pct_clamped))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(h, dtype=np.float32)
    h = np.clip(h, low, high)
    return (h - low) / (high - low)


def _equalize01(a: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
    hist, bin_edges = np.histogram(x, bins=256, range=(0.0, 1.0), density=False)
    cdf = np.cumsum(hist).astype(np.float32)
    if cdf[-1] <= 0:
        return x
    cdf /= cdf[-1]
    idx = np.minimum(np.searchsorted(bin_edges[1:], x, side="right"), 255)
    y = cdf[idx]
    return y.astype(np.float32)


def _auto_exaggeration(hm: np.ndarray, spacing: tuple[float, float]) -> float:
    sx, sy = float(spacing[0] or 1.0), float(spacing[1] or 1.0)
    z = np.asarray(hm, dtype=np.float32)
    gy, gx = np.gradient(z, sy, sx)
    slope = np.sqrt(gx * gx + gy * gy)
    s95 = float(np.percentile(slope, 95.0)) if np.isfinite(slope).any() else 0.0
    if not np.isfinite(s95) or s95 <= 1e-6:
        return 6.0
    target = 1.0
    exag = target / s95
    return float(np.clip(exag, 2.0, 18.0))


def _hillshade(
    hm: np.ndarray,
    spacing: tuple[float, float],
    exaggeration: float,
    elev_deg: float = 45.0,
    azim_deg: float = 315.0,
) -> np.ndarray:
    normals = _compute_surface_normals(hm, spacing, exaggeration)
    el = math.radians(float(elev_deg))
    az = math.radians(float(azim_deg))
    lx = math.cos(el) * math.sin(az)
    ly = math.sin(el)
    lz = math.cos(el) * math.cos(az)
    shade = np.clip(
        normals[..., 0] * lx + normals[..., 1] * ly + normals[..., 2] * lz,
        0.0,
        1.0,
    )
    return shade.astype(np.float32)


def _compute_surface_normals(
    hm: np.ndarray,
    spacing: tuple[float, float],
    exaggeration: float,
) -> np.ndarray:
    sx, sy = float(spacing[0] or 1.0), float(spacing[1] or 1.0)
    z = np.asarray(hm, dtype=np.float32) * float(exaggeration)
    gy, gx = np.gradient(z, sy, sx)
    nx = -gx
    ny = -gy
    nz = np.ones_like(z, dtype=np.float32)
    norm = np.maximum(np.sqrt(nx * nx + ny * ny + nz * nz), 1e-6)
    nx /= norm
    ny /= norm
    nz /= norm
    return np.stack([nx, ny, nz], axis=-1)


def _camera_view_vector(camera_phi: float, camera_theta: float) -> np.ndarray:
    theta = float(np.clip(camera_theta, 0.0, 180.0))
    phi = float(camera_phi)
    theta_rad = math.radians(theta)
    phi_rad = math.radians(phi)
    vertical = math.sin(theta_rad)
    planar = math.cos(theta_rad)
    vx = math.cos(phi_rad) * planar
    vy = math.sin(phi_rad) * planar
    vz = vertical if abs(vertical) > 1e-6 else 1.0
    view = np.array([vx, vy, vz], dtype=np.float32)
    nrm = float(np.linalg.norm(view))
    if nrm <= 1e-6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return view / nrm


def _apply_filmic_tonemap(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e), 0.0, 1.0)


def _designer_grade(
    rgb: np.ndarray,
    lambert: Optional[np.ndarray],
    elev_norm: np.ndarray,
    normals: Optional[np.ndarray],
) -> np.ndarray:
    base = np.asarray(rgb, dtype=np.float32)
    out = base.copy()
    if lambert is None:
        return out

    # Ridge/elevation accent (neutral, no color shift)
    elev = np.clip(np.asarray(elev_norm, dtype=np.float32), 0.0, 1.0)
    ridge_accent = 0.82 + 0.35 * np.power(elev[..., None], 1.3)
    out *= ridge_accent

    # Ambient occlusion from slope (neutral, no color shift)
    if normals is not None:
        nz = np.clip(normals[..., 2], 0.0, 1.0)
        slope = np.power(1.0 - nz, 1.4)
        ao = np.clip(1.0 - 0.4 * slope, 0.55, 1.0)
        out *= ao[..., None]

    # REMOVED: Warm/cool color grading - was shifting colors
    # REMOVED: Green vegetation boost - was adding unwanted green
    # REMOVED: Saturation boost - was amplifying color shifts

    out = np.clip(out, 0.0, None)
    return out


def _get_api_module():
    for name in ("forge3d", __package__):
        if not name:
            continue
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    raise ImportError("forge3d API module is not available")


def _detect_water_mask_via_scene(
    hm: np.ndarray,
    spacing: tuple[float, float],
    level: float,
    *,
    method: str = "flat",
    smooth_iters: int = 1,
) -> np.ndarray:
    """Detect water mask using simple elevation threshold.

    Falls back to Python implementation since detect_water_from_dem is not available.
    """
    # Create mask: True where elevation is below water level
    mask = hm < float(level)

    # Apply smoothing if requested
    if smooth_iters > 0:
        try:
            from scipy import ndimage  # type: ignore
            # Use binary closing to fill small gaps, then opening to remove small islands
            for _ in range(smooth_iters):
                mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
                mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
        except Exception:
            pass  # Skip smoothing if scipy not available

    return mask


def _postprocess_water_mask(mask: np.ndarray, keep_components: int = 1, min_area_px: int = 0) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    try:
        from scipy import ndimage  # type: ignore

        lab, n = ndimage.label(m)
        if n <= 1:
            return m
        areas = ndimage.sum(np.ones_like(lab, dtype=np.int32), labels=lab, index=np.arange(1, n + 1))
        idx_sorted = np.argsort(areas)[::-1]
        keep_k = int(max(1, keep_components))
        keep_labels = [int(i + 1) for i in idx_sorted[:keep_k] if areas[i] >= max(0, min_area_px)]
        out = (lab > 0) & np.isin(lab, keep_labels)
        return out
    except Exception:
        return m


def _compute_slope_deg(hm: np.ndarray, spacing: tuple[float, float]) -> np.ndarray:
    sx, sy = float(spacing[0] or 1.0), float(spacing[1] or 1.0)
    z = np.asarray(hm, dtype=np.float32)
    gy, gx = np.gradient(z, sy, sx)
    slope = np.sqrt(gx * gx + gy * gy)
    return (np.degrees(np.arctan(slope))).astype(np.float32)


def _apply_water_overlay(
    hm: np.ndarray,
    out_rgb: np.ndarray,
    water_mask: np.ndarray,
    *,
    water_level: float,
    fixed_color: Optional[Sequence[float]] = None,
    depth_colors: Optional[tuple[Sequence[float], Sequence[float]]] = None,
    depth_max: Optional[float] = None,
    depth_gamma: float = 1.0,
) -> np.ndarray:
    out = np.array(out_rgb, dtype=np.float32, copy=True)
    m = np.asarray(water_mask, dtype=bool)
    if not np.any(m):
        return out

    if depth_colors is not None:
        shallow, deep = depth_colors
        shallow = np.clip(np.asarray(shallow, dtype=np.float32), 0.0, 1.0)
        deep = np.clip(np.asarray(deep, dtype=np.float32), 0.0, 1.0)
        depth = np.maximum(float(water_level) - np.asarray(hm, dtype=np.float32), 0.0)
        dvals = depth[m]
        if depth_max is None or not np.isfinite(depth_max) or depth_max <= 0.0:
            if dvals.size:
                dmax = float(np.percentile(dvals, 95.0))
            else:
                dmax = 1.0
            dmax = max(dmax, 1e-6)
        else:
            dmax = float(depth_max)
        t = np.clip(dvals / dmax, 0.0, 1.0)
        g = float(depth_gamma)
        if np.isfinite(g) and abs(g - 1.0) > 1e-6:
            t = t ** (1.0 / g)
        water_rgb = (shallow[None, :] * (1.0 - t[:, None]) + deep[None, :] * t[:, None]).astype(np.float32)
        out[m] = water_rgb
    else:
        if fixed_color is None:
            fixed = np.array((0.15, 0.65, 1.0), dtype=np.float32)
        else:
            fixed = np.clip(np.asarray(fixed_color, dtype=np.float32), 0.0, 1.0)
        out[m] = fixed[None, :]

    return out


def _render_hillshade_pipeline(
    hm: np.ndarray,
    spacing: tuple[float, float],
    size: tuple[int, int],
    *,
    palette: Union[str, Sequence[str], np.ndarray, None],
    invert_palette: bool,
    contrast_pct: float,
    gamma: float,
    equalize: bool,
    exaggeration: float,
    shadow_enabled: bool,
    shadow_intensity: float,
    lighting_type: str,
    lighting_intensity: float,
    lighting_azimuth: float,
    lighting_elevation: float,
    camera_distance: float,
    camera_phi: float,
    camera_theta: float,
    water_level: Optional[float],
    water_level_percentile: Optional[float],
    water_method: str,
    water_smooth: int,
    water_color: Optional[Sequence[float]],
    water_shallow: Optional[Sequence[float]],
    water_deep: Optional[Sequence[float]],
    water_depth_gamma: float,
    water_depth_max: Optional[float],
    water_keep_components: int,
    water_min_area_pct: float,
    water_morph_iter: int,
    water_max_slope_deg: float,
    water_min_depth: float,
    water_debug: bool,
) -> np.ndarray:
    hm = np.asarray(hm, dtype=np.float32)
    spacing = (float(spacing[0] or 1.0), float(spacing[1] or 1.0))

    norm01 = _normalize_robust(hm, pct=float(contrast_pct))
    gval = float(gamma)
    if np.isfinite(gval) and abs(gval - 1.0) > 1e-3:
        norm01 = np.clip(norm01, 0.0, 1.0) ** (1.0 / gval)
    if bool(equalize):
        norm01 = _equalize01(norm01)

    palette_table = _resolve_palette(palette, invert=invert_palette, entries=256)
    idx = np.clip((norm01 * (palette_table.shape[0] - 1)).astype(np.int32), 0, palette_table.shape[0] - 1)
    rgb = palette_table[idx, :3].astype(np.float32) / 255.0

    effective_lighting_azimuth = float(lighting_azimuth)
    effective_lighting_elevation = float(lighting_elevation)
    cam_theta_val = float(camera_theta)
    cam_phi_val = float(camera_phi)
    if abs(cam_theta_val - 90.0) > 1.0:
        angle_factor = (90.0 - cam_theta_val) / 90.0
        effective_lighting_elevation = lighting_elevation * (1.0 - 0.3 * angle_factor)
        effective_lighting_azimuth = (lighting_azimuth + cam_phi_val) % 360.0
    _ = float(camera_distance)

    lighting_type_lower = str(lighting_type).lower()
    needs_normals = lighting_type_lower in {"lambertian", "phong", "blinn-phong"} or shadow_enabled
    lambert = None
    normals = None
    light_vec = None
    if needs_normals:
        exag = float(exaggeration)
        if exag <= 0.0 or not np.isfinite(exag):
            exag = _auto_exaggeration(hm, spacing)
        normals = _compute_surface_normals(hm, spacing, exag)
        el = math.radians(float(effective_lighting_elevation))
        az = math.radians(float(effective_lighting_azimuth))
        light_vec = np.array([
            math.cos(el) * math.sin(az),
            math.sin(el),
            math.cos(el) * math.cos(az),
        ], dtype=np.float32)
        lambert = np.clip(
            normals[..., 0] * light_vec[0]
            + normals[..., 1] * light_vec[1]
            + normals[..., 2] * light_vec[2],
            0.0,
            1.0,
        )

    out_rgb = rgb
    if lighting_type_lower == "lambertian" and lambert is not None:
        strength = float(np.clip(shadow_intensity, 0.0, 1.0))
        if shadow_enabled:
            shade_fac = (0.4 + 0.6 * lambert)
            mix = (1.0 - strength) + strength * shade_fac
        else:
            mix = 0.3 + 0.7 * lambert
        out_rgb = rgb * np.clip(mix[..., None], 0.0, 1.5)
    elif lighting_type_lower in {"phong", "blinn-phong"} and lambert is not None and normals is not None and light_vec is not None:
        ambient_coeff = 0.3
        diffuse_coeff = 0.7
        shininess = 32.0
        specular_strength = float(np.clip(shadow_intensity, 0.0, 1.0)) * 0.5 + 0.3
        view_vec = _camera_view_vector(cam_phi_val, cam_theta_val)
        if lighting_type_lower == "phong":
            reflect_dir = 2.0 * lambert[..., None] * normals - light_vec
            reflect_norm = np.maximum(np.linalg.norm(reflect_dir, axis=2, keepdims=True), 1e-6)
            reflect_dir = reflect_dir / reflect_norm
            spec_dot = np.clip(np.sum(reflect_dir * view_vec[None, None, :], axis=2), 0.0, 1.0)
        else:
            half_vec = light_vec + view_vec
            half_norm = float(np.linalg.norm(half_vec))
            if half_norm <= 1e-6:
                half_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                half_vec = half_vec / half_norm
            spec_dot = np.clip(np.sum(normals * half_vec[None, None, :], axis=2), 0.0, 1.0)
        specular = np.power(spec_dot, shininess)
        lighting_mix = ambient_coeff + diffuse_coeff * lambert
        out_rgb = rgb * np.clip(lighting_mix[..., None], 0.0, 1.5)
        out_rgb += specular_strength * specular[..., None]

    out_rgb = out_rgb * float(lighting_intensity)
    if lambert is not None:
        out_rgb = _designer_grade(out_rgb, lambert, norm01, normals)
    out_rgb = _apply_filmic_tonemap(out_rgb)

    if (water_level is not None and np.isfinite(water_level)) or (
        water_level_percentile is not None and np.isfinite(water_level_percentile)
    ):
        try:
            if water_level_percentile is not None and np.isfinite(water_level_percentile):
                level_eff = float(np.percentile(hm, float(water_level_percentile)))
            else:
                level_eff = float(water_level)

            mask = _detect_water_mask_via_scene(
                hm,
                spacing=spacing,
                level=level_eff,
                method=str(water_method),
                smooth_iters=int(water_smooth),
            )

            mask = np.asarray(mask, dtype=bool)
            cov_raw = float(mask.mean()) * 100.0 if mask.size else 0.0

            try:
                from scipy import ndimage  # type: ignore

                iters = int(max(0, water_morph_iter))
                if iters > 0:
                    structure = np.ones((3, 3), dtype=bool)
                    mask = ndimage.binary_opening(mask, structure=structure, iterations=iters)
                    mask = ndimage.binary_closing(mask, structure=structure, iterations=iters)
            except Exception:
                pass
            cov_morph = float(mask.mean()) * 100.0 if mask.size else 0.0

            mask &= (hm <= level_eff)
            cov_hclamp = float(mask.mean()) * 100.0 if mask.size else 0.0

            depth = np.maximum(level_eff - hm, 0.0)
            min_depth = float(max(0.0, water_min_depth))
            if min_depth > 0.0:
                mask &= (depth >= min_depth)
            max_slope = float(max(0.0, water_max_slope_deg))
            if max_slope > 0.0:
                slope_deg = _compute_slope_deg(hm, spacing)
                mask &= (slope_deg <= max_slope)
            cov_filters = float(mask.mean()) * 100.0 if mask.size else 0.0

            min_area_px = int(hm.shape[0] * hm.shape[1] * max(0.0, float(water_min_area_pct)) / 100.0)
            mask = _postprocess_water_mask(mask, keep_components=int(max(1, water_keep_components)), min_area_px=min_area_px)
            cov_post = float(mask.mean()) * 100.0 if mask.size else 0.0

            if water_debug:
                print(
                    "Water level="
                    f"{level_eff:.3f} method={water_method} cov% raw={cov_raw:.3f} "
                    f"morph={cov_morph:.3f} hclamp={cov_hclamp:.3f} filters={cov_filters:.3f} "
                    f"post={cov_post:.3f} (min_depth={min_depth}, max_slope={max_slope}, "
                    f"keep={water_keep_components}, min_area_pct={water_min_area_pct})"
                )

            depth_colors = None
            if water_shallow is not None and water_deep is not None:
                depth_colors = (water_shallow, water_deep)
            fixed_color = water_color if water_color is not None and depth_colors is None else None

            out_rgb = _apply_water_overlay(
                hm,
                out_rgb,
                mask,
                water_level=float(level_eff),
                fixed_color=fixed_color,
                depth_colors=depth_colors,
                depth_max=(float(water_depth_max) if water_depth_max is not None else None),
                depth_gamma=float(water_depth_gamma),
            )
        except Exception as exc:
            if water_debug:
                print(f"Water overlay failed: {exc}")

    src_H, src_W = hm.shape
    target_W, target_H = int(size[0]), int(size[1])
    if target_W <= 0 or target_H <= 0:
        raise ValueError("Output size must be positive")

    if (src_W, src_H) != (target_W, target_H):
        try:
            from scipy import ndimage  # type: ignore

            zoom_y = target_H / src_H
            zoom_x = target_W / src_W
            out_rgb = ndimage.zoom(out_rgb, (zoom_y, zoom_x, 1), order=1)
        except Exception:
            yy = (np.linspace(0, src_H - 1, target_H)).astype(int)
            xx = (np.linspace(0, src_W - 1, target_W)).astype(int)
            out_rgb = out_rgb[yy][:, xx]

    rgba = np.empty((target_H, target_W, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(out_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgba[..., 3] = 255
    return rgba


def render_raster(
    src: Union[np.ndarray, str, Path],
    *,
    size: tuple[int, int] = (1200, 900),
    z_scale: float = 1.0,
    frames: int = 8,
    seed: int = 7,
    camera: Optional[Dict[str, Any]] = None,
    denoiser: str = "off",
    svgf_iters: int = 5,
    luminance_clamp: Optional[float] = None,
    renderer: Literal["path_tracer", "hillshade", "raster"] = "path_tracer",
    spacing: Optional[tuple[float, float]] = None,
    palette: Union[str, Sequence[str], np.ndarray, None] = None,
    invert_palette: bool = False,
    contrast_pct: float = 1.0,
    gamma: float = 1.1,
    equalize: bool = True,
    exaggeration: float = 0.0,
    shadow_enabled: bool = True,
    shadow_intensity: float = 1.0,
    lighting_type: str = "lambertian",
    lighting_intensity: float = 1.0,
    lighting_azimuth: float = 315.0,
    lighting_elevation: float = 45.0,
    camera_distance: float = 1.0,
    camera_phi: float = 0.0,
    camera_theta: float = 90.0,
    water_level: Optional[float] = None,
    water_level_percentile: Optional[float] = 30.0,
    water_method: str = "flat",
    water_smooth: int = 1,
    water_color: Optional[Sequence[float]] = None,
    water_shallow: Optional[Sequence[float]] = None,
    water_deep: Optional[Sequence[float]] = None,
    water_depth_gamma: float = 1.0,
    water_depth_max: Optional[float] = None,
    water_keep_components: int = 2,
    water_min_area_pct: float = 0.01,
    water_morph_iter: int = 1,
    water_max_slope_deg: float = 6.0,
    water_min_depth: float = 0.1,
    water_debug: bool = False,
    show_in_viewer: bool = False,
) -> np.ndarray:
    """Render a DEM heightmap as a shaded terrain image.

    Parameters
    ----------
    src : np.ndarray | str | pathlib.Path
        Either a heightmap array (H,W) float or an input GeoTIFF path.
    size : (int, int)
        Output image size (W,H).
    z_scale : float
        Vertical scale factor applied to elevation values (path tracer only).
    frames : int
        Accumulation frames for the path tracer (only used by GPU/native path).
    seed : int
        Random seed for reproducibility.
    camera : dict | None
        Camera dictionary from ``forge3d.path_tracing.make_camera``. If None, an
        auto-framed camera is generated from the mesh bounds (path tracer only).
    denoiser : {"off","svgf"}
        Optional denoiser to apply (CPU fallback path only).
    svgf_iters : int
        SVGF iterations for the CPU fallback.
    luminance_clamp : float | None
        Optional luminance clamp for CPU fallback to suppress outliers.
    renderer : {"path_tracer","hillshade","raster"}
        Select the rendering pipeline. ``"path_tracer"`` matches the historical
        behaviour. ``"hillshade"``/``"raster"`` enables the CPU hillshade pipeline
        with palette, lighting, and water controls.
    spacing : (float, float) | None
        Pixel spacing (sx, sy) when ``src`` is a numpy array. Defaults to (1,1)
        if not provided. Ignored when ``src`` is a file path.
    palette, invert_palette, contrast_pct, gamma, equalize, exaggeration,
    shadow_enabled, shadow_intensity, lighting_type, lighting_intensity,
    lighting_azimuth, lighting_elevation, camera_distance, camera_phi,
    camera_theta, water_level, water_level_percentile, water_method,
    water_smooth, water_color, water_shallow, water_deep, water_depth_gamma,
    water_depth_max, water_keep_components, water_min_area_pct, water_morph_iter,
    water_max_slope_deg, water_min_depth, water_debug : optional
        Parameters applied by the hillshade pipeline. They are ignored when
        ``renderer="path_tracer"``.

    Returns
    -------
    np.ndarray(H,W,4) uint8
        Final RGBA image.
    """
    # Load or accept heightmap
    spacing_override = spacing
    if isinstance(src, (str, Path)):
        hm, spacing_loaded = _load_dem(Path(src))
        spacing_vals = spacing_loaded
        if spacing_override is not None:
            sx, sy = spacing_override
            spacing_vals = (float(sx), float(sy))
    else:
        hm = np.asarray(src, dtype=np.float32)
        if hm.ndim != 2:
            raise ValueError("heightmap array must be 2D (H,W)")
        if spacing_override is not None:
            sx, sy = spacing_override
            spacing_vals = (float(sx), float(sy))
        else:
            spacing_vals = (1.0, 1.0)

    out_w, out_h = int(size[0]), int(size[1])
    if out_w <= 0 or out_h <= 0:
        raise ValueError("Output size must be positive")

    renderer_mode = str(renderer).lower()
    if renderer_mode in {"hillshade", "raster", "cpu"}:
        rgba = _render_hillshade_pipeline(
            hm,
            spacing_vals,
            (out_w, out_h),
            palette=palette,
            invert_palette=invert_palette,
            contrast_pct=contrast_pct,
            gamma=gamma,
            equalize=equalize,
            exaggeration=exaggeration,
            shadow_enabled=shadow_enabled,
            shadow_intensity=shadow_intensity,
            lighting_type=lighting_type,
            lighting_intensity=lighting_intensity,
            lighting_azimuth=lighting_azimuth,
            lighting_elevation=lighting_elevation,
            camera_distance=camera_distance,
            camera_phi=camera_phi,
            camera_theta=camera_theta,
            water_level=water_level,
            water_level_percentile=water_level_percentile,
            water_method=water_method,
            water_smooth=water_smooth,
            water_color=water_color,
            water_shallow=water_shallow,
            water_deep=water_deep,
            water_depth_gamma=water_depth_gamma,
            water_depth_max=water_depth_max,
            water_keep_components=water_keep_components,
            water_min_area_pct=water_min_area_pct,
            water_morph_iter=water_morph_iter,
            water_max_slope_deg=water_max_slope_deg,
            water_min_depth=water_min_depth,
            water_debug=water_debug,
        )
        if show_in_viewer:
            try:
                from ._viewer import open_viewer_image as _open_viewer_image
                _open_viewer_image(rgba, title="forge3d Raster Preview")
            except Exception:
                pass
        return rgba

    # Triangulate to mesh for path tracer
    print(f"[PT-DEBUG] Mesh generation: reduced from {hm.shape} using spacing={spacing_vals}, z_scale={z_scale}")
    V, F = _heightmap_to_mesh(hm, spacing_vals, z_scale=float(z_scale))
    if verbose:
        print(f"[PT-DEBUG] Mesh ready: {V.shape[0]} verts, {F.shape[0]} tris")
    # Compute vertex colors from palette (matching hillshade quality)
    vertex_colors = _compute_vertex_colors(
        hm,
        palette=palette,
        invert_palette=invert_palette,
        contrast_pct=contrast_pct,
        gamma=gamma,
        equalize=equalize,
    )
    
    # Package vertices with colors: [x, y, z, pad, r, g, b, pad2]
    num_verts = V.shape[0]
    V_colored = np.zeros((num_verts, 8), dtype=np.float32)
    V_colored[:, 0:3] = V  # positions
    V_colored[:, 4:7] = vertex_colors  # colors

    # Auto camera if not provided
    cam = camera if camera is not None else _autoframe_camera(V, (out_w, out_h))

    # Try native GPU mesh tracer first
    try:
        from . import _forge3d as _native  # type: ignore[attr-defined]
        try:
            if hasattr(_native, "device_probe"):
                status = _native.device_probe("metal").get("status", "unavailable")  # type: ignore
                if status == "ok":
                    V_colored32 = np.ascontiguousarray(V_colored, dtype=np.float32)
                    F32 = np.ascontiguousarray(F.astype(np.uint32))
                    img = _native._pt_render_gpu_mesh(
                        int(out_w),
                        int(out_h),
                        V_colored32,
                        F32,
                        cam,
                        int(seed),
                        int(max(1, frames)),
                        None,
                        int(max(0, debug_mode)),
                    )
                    if show_in_viewer:
                        try:
                            from ._viewer import open_viewer_image as _open_viewer_image
                            _open_viewer_image(img, title="forge3d Raster Preview")
                        except Exception:
                            pass
                    return img
        except Exception:
            pass
    except Exception:
        pass

    # CPU fallback path tracer
    tracer = PathTracer()
    image = tracer.render_rgba(
        out_w,
        out_h,
        scene=None,
        camera=cam,
        seed=int(seed),
        frames=max(1, int(frames)),
        use_gpu=False,
        mesh=None,
        denoiser=str(denoiser),
        svgf_iters=int(svgf_iters),
        luminance_clamp=(float(luminance_clamp) if luminance_clamp is not None else None),
    )

    # Show in interactive viewer when requested
    if show_in_viewer:
        try:
            from ._viewer import open_viewer_image as _open_viewer_image
            _open_viewer_image(image, title="forge3d Raster Preview")
        except Exception:
            pass

    return image


def _ensure_closed_ring(coords: np.ndarray) -> np.ndarray:
    """Ensure a 2D ring is explicitly closed by repeating the first vertex at the end."""
    c = np.asarray(coords, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError("ring must have shape (N,2)")
    if c.shape[0] == 0:
        return c
    if not np.allclose(c[0], c[-1]):
        c = np.vstack([c, c[0]])
    return np.ascontiguousarray(c)


def _compute_bounds_from_iterables(iterables: list[np.ndarray]) -> tuple[float, float, float, float]:
    """Compute min_x, min_y, max_x, max_y across a list of (N,2) arrays."""
    min_x = float("inf"); min_y = float("inf"); max_x = float("-inf"); max_y = float("-inf")
    for arr in iterables:
        if arr.size == 0:
            continue
        a = np.asarray(arr, dtype=np.float64)
        min_x = min(min_x, float(np.min(a[:, 0])))
        min_y = min(min_y, float(np.min(a[:, 1])))
        max_x = max(max_x, float(np.max(a[:, 0])))
        max_y = max(max_y, float(np.max(a[:, 1])))
    if not np.isfinite([min_x, min_y, max_x, max_y]).all():
        return -1.0, -1.0, 1.0, 1.0
    return min_x, min_y, max_x, max_y


def _world_to_pixel_transform(
    bounds: tuple[float, float, float, float], size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Return (sx, sy, tx, ty) mapping world (x,y) to pixel coords for viewport size.

    Uses uniform NDC scale 2/d with d = max(dx, dy), realized in pixel space as:
      sx_px = W / d, sy_px = H / d, tx = W/2 - sx*cx, ty = H/2 - sy*cy
    where (cx,cy) is the world bbox center.
    """
    min_x, min_y, max_x, max_y = bounds
    W, H = int(size[0]), int(size[1])
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    dx = max(max_x - min_x, 1e-6)
    dy = max(max_y - min_y, 1e-6)
    d = max(dx, dy)
    sx = float(W) / float(d)
    sy = float(H) / float(d)
    tx = float(W) * 0.5 - sx * cx
    ty = float(H) * 0.5 - sy * cy
    return sx, sy, tx, ty


def _apply_affine_to_ring(r: np.ndarray, sx: float, sy: float, tx: float, ty: float) -> np.ndarray:
    r64 = np.asarray(r, dtype=np.float64)
    out = np.empty_like(r64)
    out[:, 0] = r64[:, 0] * sx + tx
    out[:, 1] = r64[:, 1] * sy + ty
    return out


def _alpha_over(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Alpha-composite overlay over base (both HxWx4 uint8, straight alpha) and return uint8."""
    if base.shape != overlay.shape or base.ndim != 3 or base.shape[2] != 4:
        raise ValueError("base and overlay must have shape (H,W,4) and match")
    b = base.astype(np.float32) / 255.0
    o = overlay.astype(np.float32) / 255.0
    Ab = b[..., 3:4]
    Ao = o[..., 3:4]
    Cb_p = b[..., :3] * Ab
    Co_p = o[..., :3] * Ao
    Cout_p = Co_p + Cb_p * (1.0 - Ao)
    Aout = Ao + Ab * (1.0 - Ao)
    # Avoid divide-by-zero; where Aout==0 keep zeros
    Cout = np.where(Aout > 1e-6, Cout_p / Aout, 0.0)
    out = np.concatenate([Cout, Aout], axis=-1)
    out = np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
    return out


def render_polygons(
    polygons: Union[np.ndarray, dict, list, str, Path],
    *,
    size: tuple[int, int] = (1200, 900),
    fill_rgba: tuple[float, float, float, float] | None = (0.2, 0.4, 0.8, 1.0),
    stroke_rgba: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    stroke_width: float = 1.0,
    add_points: Optional[np.ndarray] = None,
    point_rgba: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
    point_size: float = 8.0,
    add_polylines: Optional[list[np.ndarray]] = None,
    transform: Optional[Tuple[float, float, float, float]] = None,
    return_pick: bool = False,
    base_pick_id: int = 1,
    show_in_viewer: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Render 2D polygons with GPU fill when available, falling back to stroke-only OIT.

    Parameters
    ----------
    polygons : array | dict | list | str | pathlib.Path
        One of:
          - np.ndarray (N,2): a single polygon exterior ring
          - dict: { 'exterior': (N,2) ndarray, 'holes': [ (M,2) ndarrays ], 'stroke_rgba'?, 'stroke_width'? }
          - list of the above
          - str | Path: path to a vector dataset
              • .shp (ESRI Shapefile)
              • .geojson (GeoJSON)
              • .gpkg (GeoPackage)
              • .gdb (FileGDB; supply layer via dict { 'path': ..., 'layer': ... })
          - dict with { 'path': str|Path, 'layer'?: str } for datasets with layers (e.g., .gdb, .gpkg)
    size : (int, int)
        Output image size (W,H)
    stroke_rgba : (r,g,b,a)
        Default stroke color in 0..1, used when per-polygon not provided
    stroke_width : float
        Default stroke width in world units
    add_points : (N,2) array, optional
        Extra points to draw via OIT overlay.
    point_size : float
        Size in pixels for added points overlay.
    add_polylines : list of (M,2) arrays, optional
        Extra polylines to draw via OIT overlay.
    transform : (sx, sy, tx, ty) | None
        Optional world->pixel affine transform used for OIT overlays and stroke fallback.
        If not provided, an auto-fit transform is computed from geometry bounds.
    return_pick : bool
        If True, also returns a pick map (R32Uint) alongside the RGBA image
    base_pick_id : int
        Base pick id used when return_pick is True

    Returns
    -------
    np.ndarray(H,W,4) uint8
      or (rgba, pick) if return_pick = True

    """
    from .vector import VectorScene

    def _maybe_show(img: np.ndarray) -> np.ndarray:
        if show_in_viewer:
            try:
                from ._viewer import open_viewer_image as _open_viewer_image
                _open_viewer_image(img, title="forge3d Polygons Preview")
            except Exception:
                pass
        return img

    def _iter_polys(src: Union[np.ndarray, dict, list, str, Path]):
        # New: accept vector file paths or a dict with {'path','layer'}
        if isinstance(src, (str, Path)):
            for item in _load_polygons_from_vector(src):
                yield item
            return
        if isinstance(src, dict) and ("path" in src):
            layer = src.get("layer")
            for item in _load_polygons_from_vector(src["path"], layer=layer):
                yield item
            return
        # Existing behaviors
        if isinstance(src, np.ndarray):
            yield {"exterior": src}
            return
        if isinstance(src, dict) and ("exterior" in src):
            yield src
            return
        if isinstance(src, (list, tuple)):
            for item in src:
                if isinstance(item, (str, Path)):
                    for sub in _load_polygons_from_vector(item):
                        yield sub
                elif isinstance(item, dict) and ("path" in item):
                    layer = item.get("layer")
                    for sub in _load_polygons_from_vector(item["path"], layer=layer):
                        yield sub
                elif isinstance(item, np.ndarray):
                    yield {"exterior": item}
                elif isinstance(item, dict) and ("exterior" in item):
                    yield item
                else:
                    raise ValueError("Unsupported polygon list element; expected ndarray/dict with 'exterior' or 'path'")
            return
        raise ValueError("polygons must be an ndarray, a dict with 'exterior' or 'path', a list of those, or a vector file path")

    # ---------------------------------------------
    # Setup and optional dataset ingestion
    # ---------------------------------------------
    W, H = int(size[0]), int(size[1])
    pbar = _make_progress(4, "render_polygons")  # parse, native/fallback, overlay, done
    pbar.write("Parsing polygon inputs...")

    loaded_geoms: Optional[dict] = None
    src_path: Optional[Union[str, Path]] = None
    src_layer: Optional[str] = None
    if isinstance(polygons, (str, Path)):
        src_path = polygons
    elif isinstance(polygons, dict) and ("path" in polygons):
        src_path = polygons["path"]
        src_layer = polygons.get("layer")

    if src_path is not None:
        try:
            import geopandas as gpd  # type: ignore
            from shapely.geometry import (
                Polygon as _ShpPolygon,
                MultiPolygon as _ShpMultiPolygon,
                LineString as _ShpLineString,
                MultiLineString as _ShpMultiLineString,
                Point as _ShpPoint,
                MultiPoint as _ShpMultiPoint,
                GeometryCollection as _ShpGeomColl,
            )  # type: ignore

            def _xy_coords(coords) -> np.ndarray:
                a2 = np.asarray(coords, dtype=np.float64)
                if a2.ndim != 2 or a2.shape[1] < 2:
                    raise ValueError("coordinate array must have shape (N,>=2)")
                return np.ascontiguousarray(a2[:, :2])  # drop Z if present

            gdf = gpd.read_file(str(src_path), layer=src_layer) if src_layer is not None else gpd.read_file(str(src_path))
            polys: list[dict] = []
            lines: list[np.ndarray] = []
            pts: list[tuple[float, float]] = []

            def _collect(geom):
                if geom is None or getattr(geom, "is_empty", False):
                    return
                if isinstance(geom, _ShpPolygon):
                    ext = _ensure_closed_ring(_xy_coords(geom.exterior.coords))
                    holes = []
                    for ring in geom.interiors:
                        holes.append(_ensure_closed_ring(_xy_coords(ring.coords)))
                    polys.append({"exterior": ext, "holes": holes})
                elif isinstance(geom, _ShpMultiPolygon):
                    for gg in geom.geoms:
                        _collect(gg)
                elif isinstance(geom, _ShpLineString):
                    arr = _xy_coords(geom.coords)
                    if arr.shape[0] >= 2:
                        lines.append(arr)
                elif isinstance(geom, _ShpMultiLineString):
                    for gg in geom.geoms:
                        _collect(gg)
                elif isinstance(geom, _ShpPoint):
                    x, y = float(geom.x), float(geom.y)
                    if np.isfinite([x, y]).all():
                        pts.append((x, y))
                elif isinstance(geom, _ShpMultiPoint):
                    for gg in geom.geoms:
                        _collect(gg)
                elif isinstance(geom, _ShpGeomColl):
                    for gg in geom.geoms:
                        _collect(gg)

            for geom in gdf.geometry:
                _collect(geom)

            loaded_geoms = {
                "polygons": polys,
                "lines": lines,
                "points": np.asarray(pts, dtype=np.float64) if pts else None,
            }
        except Exception:
            loaded_geoms = None

    # ---------------------------------------------
    # Native GPU filled polygons if requested and available
    # ---------------------------------------------
    if fill_rgba is not None:
        try:
            from . import _forge3d as _native  # type: ignore[attr-defined]
            exteriors: list[np.ndarray] = []
            holes_all: list[list[np.ndarray]] = []
            all_rings_for_bounds: list[np.ndarray] = []
            if loaded_geoms is not None and len(loaded_geoms.get("polygons", [])) > 0:
                for poly in loaded_geoms["polygons"]:
                    ext = _ensure_closed_ring(np.asarray(poly["exterior"], dtype=np.float64))
                    exteriors.append(ext)
                    all_rings_for_bounds.append(ext)
                    hlist = []
                    for h in poly.get("holes", []) or []:
                        h_arr = _ensure_closed_ring(np.asarray(h, dtype=np.float64))
                        hlist.append(h_arr)
                        all_rings_for_bounds.append(h_arr)
                    holes_all.append(hlist)
            else:
                count = 0
                for poly in _iter_polys(polygons):
                    ext = _ensure_closed_ring(np.asarray(poly["exterior"], dtype=np.float64))
                    exteriors.append(ext)
                    all_rings_for_bounds.append(ext)
                    hlist = []
                    for h in poly.get("holes", []) or []:
                        h_arr = _ensure_closed_ring(np.asarray(h, dtype=np.float64))
                        hlist.append(h_arr)
                        all_rings_for_bounds.append(h_arr)
                    holes_all.append(hlist)
                    count += 1
                if count == 0:
                    raise RuntimeError("no polygon exteriors")
            pbar.update(1)
            img = _native.vector_render_polygons_fill_py(
                int(W), int(H), exteriors, holes_all, tuple(map(float, fill_rgba)), tuple(map(float, stroke_rgba)), float(stroke_width)
            )
            # Determine whether we need to composite overlays
            need_overlay = False
            if loaded_geoms is not None:
                if (loaded_geoms.get("lines") and len(loaded_geoms["lines"]) > 0) or (loaded_geoms.get("points") is not None):
                    need_overlay = True
            if add_points is not None or (add_polylines is not None and len(add_polylines) > 0) or return_pick:
                need_overlay = True
            if need_overlay:
                scene = VectorScene()
                if transform is not None:
                    sx, sy, tx, ty = map(float, transform)
                else:
                    bounds = _compute_bounds_from_iterables(all_rings_for_bounds)
                    sx, sy, tx, ty = _world_to_pixel_transform(bounds, (W, H))
                # Dataset overlays
                if loaded_geoms is not None:
                    for path in loaded_geoms.get("lines", []) or []:
                        P = np.asarray(path, dtype=np.float64)
                        Pp = _apply_affine_to_ring(P, sx, sy, tx, ty)
                        scene.add_polyline([(float(x), float(y)) for x, y in Pp], rgba=stroke_rgba, width=stroke_width)
                    if loaded_geoms.get("points") is not None:
                        Pts = np.asarray(loaded_geoms["points"], dtype=np.float64)
                        Pp = _apply_affine_to_ring(Pts, sx, sy, tx, ty)
                        for x, y in Pp:
                            scene.add_point(float(x), float(y), rgba=point_rgba, size=float(point_size))
                # Extra overlays
                if add_polylines:
                    for path in add_polylines:
                        path = np.asarray(path, dtype=np.float64)
                        if path.ndim != 2 or path.shape[1] != 2:
                            raise ValueError("each polyline must have shape (N,2)")
                        p = _apply_affine_to_ring(path, sx, sy, tx, ty)
                        scene.add_polyline([(float(x), float(y)) for x, y in p], rgba=stroke_rgba, width=stroke_width)
                if add_points is not None:
                    P = np.asarray(add_points, dtype=np.float64)
                    if P.ndim != 2 or P.shape[1] != 2:
                        raise ValueError("add_points must have shape (N,2)")
                    Pp = _apply_affine_to_ring(P, sx, sy, tx, ty)
                    for x, y in Pp:
                        scene.add_point(float(x), float(y), rgba=point_rgba, size=float(point_size))
                if return_pick:
                    pbar.write("Compositing overlays with pick map...")
                    overlay_rgba, pick = scene.render_oit_and_pick(W, H, base_pick_id=int(base_pick_id))
                    composed = _alpha_over(img, overlay_rgba)
                    pbar.update(1); pbar.update(1); pbar.close()
                    return _maybe_show(composed), pick
                else:
                    pbar.write("Compositing overlays...")
                    overlay_rgba = scene.render_oit(W, H)
                    out = _alpha_over(img, overlay_rgba)
                    pbar.update(1); pbar.update(1); pbar.close()
                    return _maybe_show(out)
            else:
                pbar.update(2); pbar.close()
                return _maybe_show(img)
        except Exception:
            # Fall through to stroke-only OIT
            pass

    # Stroke-only fallback via VectorScene + OIT (or when dataset has only lines/points)
    pbar.write("Falling back to stroke-only OIT...")
    scene = VectorScene()
    # Compute transform to map world coords to pixel coords to fit viewport
    rings_for_bounds: list[np.ndarray] = []
    poly_rings: list[np.ndarray] = []
    if loaded_geoms is not None:
        for poly in loaded_geoms.get("polygons", []) or []:
            ext = _ensure_closed_ring(np.asarray(poly["exterior"], dtype=np.float64))
            rings_for_bounds.append(ext); poly_rings.append(ext)
            for h in poly.get("holes", []) or []:
                h_arr = _ensure_closed_ring(np.asarray(h, dtype=np.float64))
                rings_for_bounds.append(h_arr); poly_rings.append(h_arr)
        for path in loaded_geoms.get("lines", []) or []:
            arr = np.asarray(path, dtype=np.float64)
            rings_for_bounds.append(arr)
        if loaded_geoms.get("points") is not None:
            rings_for_bounds.append(np.asarray(loaded_geoms["points"], dtype=np.float64))
    else:
        for poly in _iter_polys(polygons):
            ext = _ensure_closed_ring(poly["exterior"]).astype(np.float64, copy=False)
            rings_for_bounds.append(ext); poly_rings.append(ext)
            for h in poly.get("holes", []) or []:
                h_arr = _ensure_closed_ring(h).astype(np.float64, copy=False)
                rings_for_bounds.append(h_arr); poly_rings.append(h_arr)
    if transform is not None:
        sx, sy, tx, ty = map(float, transform)
    else:
        bounds = _compute_bounds_from_iterables(rings_for_bounds)
        sx, sy, tx, ty = _world_to_pixel_transform(bounds, (W, H))


    # Optional additional overlays in fallback
    if add_polylines:
        for path in add_polylines:
            path = np.asarray(path, dtype=np.float64)
            if path.ndim != 2 or path.shape[1] != 2:
                raise ValueError("each polyline must have shape (N,2)")
            p = _apply_affine_to_ring(path, sx, sy, tx, ty)
            scene.add_polyline([(float(x), float(y)) for x, y in p], rgba=stroke_rgba, width=stroke_width)

    for r in poly_rings:
        rp = _apply_affine_to_ring(r, sx, sy, tx, ty)
        scene.add_polyline([(float(x), float(y)) for x, y in rp], rgba=stroke_rgba, width=float(stroke_width))

    # Add lines from dataset
    if loaded_geoms is not None:
        for path in loaded_geoms.get("lines", []) or []:
            arr = _apply_affine_to_ring(np.asarray(path, dtype=np.float64), sx, sy, tx, ty)
            scene.add_polyline([(float(x), float(y)) for x, y in arr], rgba=stroke_rgba, width=float(stroke_width))

    # Add points overlay if provided
    if loaded_geoms is not None and loaded_geoms.get("points") is not None:
        P = _apply_affine_to_ring(np.asarray(loaded_geoms["points"], dtype=np.float64), sx, sy, tx, ty)
        for x, y in P:
            scene.add_point(float(x), float(y), rgba=point_rgba, size=float(point_size))
    if add_points is not None:
        P2 = np.asarray(add_points, dtype=np.float64)
        if P2.ndim != 2 or P2.shape[1] != 2:
            raise ValueError("add_points must have shape (N,2)")
        Pp2 = _apply_affine_to_ring(P2, sx, sy, tx, ty)
        for x, y in Pp2:
            scene.add_point(float(x), float(y), rgba=point_rgba, size=float(point_size))

    if return_pick:
        rgba, pick = scene.render_oit_and_pick(W, H, base_pick_id=int(base_pick_id))
        pbar.update(2); pbar.close()
        return _maybe_show(rgba), pick
    else:
        rgba = scene.render_oit(W, H)
        pbar.update(2); pbar.close()
        return _maybe_show(rgba)


def render_overlay(
    base_rgba: np.ndarray,
    *,
    points: Optional[np.ndarray] = None,
    polylines: Optional[list[np.ndarray]] = None,
    polygons: Optional[Union[np.ndarray, dict, list]] = None,
    stroke_rgba: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    stroke_width: float = 1.5,
    point_rgba: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
    point_size: float = 8.0,
    fit_to_base: bool = True,
    transform: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """Composite vector OIT strokes (polylines/points) over a base RGBA image.

    Parameters
    ----------
    base_rgba : np.ndarray (H,W,4) uint8
        Base image to composite onto.
    points : (N,2) array, optional
    polylines : list of (M,2) arrays, optional
    polygons : array|dict|list, optional
        If provided, polygon outlines (exterior + holes) are added as polylines.
    fit_to_base : bool
        If True, world coordinates are fit to the base image using the same
        uniform-bbox transform as filled polygons; otherwise inputs are assumed
        to be in pixel coordinates of the base.
    transform : (sx, sy, tx, ty) | None
        If provided, overrides fit_to_base and uses this world->pixel transform.
    """
    base = np.asarray(base_rgba)
    if base.ndim != 3 or base.shape[2] != 4 or base.dtype != np.uint8:
        raise ValueError("base_rgba must be uint8 with shape (H,W,4)")
    H, W = int(base.shape[0]), int(base.shape[1])

    from .vector import VectorScene
    scene = VectorScene()

    # Gather coordinates for bounds if fitting
    rings: list[np.ndarray] = []
    if polygons is not None:
        def _iter_polys(src: Union[np.ndarray, dict, list]):
            if isinstance(src, np.ndarray):
                yield {"exterior": src}; return
            if isinstance(src, dict) and ("exterior" in src):
                yield src; return
            if isinstance(src, (list, tuple)):
                for item in src:
                    if isinstance(item, np.ndarray):
                        yield {"exterior": item}
                    elif isinstance(item, dict) and ("exterior" in item):
                        yield item
                    else:
                        raise ValueError("Unsupported polygon list element; expected ndarray or dict with 'exterior'")
                return
            raise ValueError("polygons must be an ndarray, dict with 'exterior', or a list of those")
        for poly in _iter_polys(polygons):
            ext = _ensure_closed_ring(poly["exterior"]).astype(np.float64, copy=False)
            rings.append(ext)
            for h in poly.get("holes", []) or []:
                rings.append(_ensure_closed_ring(h).astype(np.float64, copy=False))
    if polylines:
        for path in polylines:
            arr = np.asarray(path, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("each polyline must have shape (N,2)")
            rings.append(arr)
    if points is not None:
        P = np.asarray(points, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] != 2:
            raise ValueError("points must have shape (N,2)")
        rings.append(P)

    if transform is not None:
        sx, sy, tx, ty = map(float, transform)
        def to_px(a: np.ndarray) -> np.ndarray:
            return _apply_affine_to_ring(a, sx, sy, tx, ty)
    elif fit_to_base and rings:
        bounds = _compute_bounds_from_iterables(rings)
        sx, sy, tx, ty = _world_to_pixel_transform(bounds, (W, H))
        def to_px(a: np.ndarray) -> np.ndarray:
            return _apply_affine_to_ring(a, sx, sy, tx, ty)
    else:
        def to_px(a: np.ndarray) -> np.ndarray:
            return np.asarray(a, dtype=np.float64)

    # Add polygons as outlines
    if polygons is not None:
        def _iter_polys2(src: Union[np.ndarray, dict, list]):
            if isinstance(src, np.ndarray):
                yield {"exterior": src}; return
            if isinstance(src, dict) and ("exterior" in src):
                yield src; return
            if isinstance(src, (list, tuple)):
                for item in src:
                    if isinstance(item, np.ndarray):
                        yield {"exterior": item}
                    elif isinstance(item, dict) and ("exterior" in item):
                        yield item
                    else:
                        raise ValueError("Unsupported polygon list element; expected ndarray or dict with 'exterior'")
                return
            raise ValueError("polygons must be an ndarray, dict with 'exterior', or a list of those")
        for poly in _iter_polys2(polygons):
            ext = to_px(_ensure_closed_ring(poly["exterior"]))
            scene.add_polyline([(float(x), float(y)) for x, y in ext], rgba=stroke_rgba, width=float(stroke_width))
            for h in poly.get("holes", []) or []:
                hring = to_px(_ensure_closed_ring(h))
                scene.add_polyline([(float(x), float(y)) for x, y in hring], rgba=stroke_rgba, width=float(stroke_width))

    # Add explicit polylines
    if polylines:
        for path in polylines:
            arr = to_px(np.asarray(path, dtype=np.float64))
            scene.add_polyline([(float(x), float(y)) for x, y in arr], rgba=stroke_rgba, width=float(stroke_width))

    # Add points
    if points is not None:
        P = to_px(np.asarray(points, dtype=np.float64))
        for x, y in P:
            scene.add_point(float(x), float(y), rgba=point_rgba, size=float(point_size))

    overlay = scene.render_oit(W, H)
    return _alpha_over(base, overlay)


def _render_raytrace_mesh_simple(
    mesh: tuple[_np.ndarray, _np.ndarray],
    width: int,
    height: int,
    camera: dict,
    frames: int = 50,
    seed: int = 42,
    prefer_gpu: bool = True,
    denoiser: str | None = None,
    denoise_strength: float = 0.8,
    background_rgb: _np.ndarray | None = None,
    sampling_mode: str = "rng",
    verbose: bool = True,
) -> tuple[_np.ndarray, dict]:
    """Simplified render function for internal use (GPU mesh path with CPU fallback).

    Parameters
    ----------
    vertices : np.ndarray (N,3) float32/float64
    indices : np.ndarray (M,3) int/uint
    """
    V = np.asarray(mesh[0])
    F = np.asarray(mesh[1])
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("vertices must have shape (N,3)")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("indices must have shape (M,3)")

    V32 = np.ascontiguousarray(V.astype(np.float32))
    F32 = np.ascontiguousarray(F.astype(np.uint32))

    W, H = int(width), int(height)
    cam = camera

    # Try native GPU mesh tracer
    try:
        from . import _forge3d as _native  # type: ignore[attr-defined]
        try:
            status = _native.device_probe("metal").get("status", "unavailable")  # type: ignore
            if status == "ok":
                # Map sampling mode string to int: rng=0, sobol=1, cmj=2
                sampling_mode_map = {"rng": 0, "sobol": 1, "cmj": 2}
                sampling_mode_int = sampling_mode_map.get(sampling_mode.lower(), 0)
                img = _native._pt_render_gpu_mesh(
                    int(W),
                    int(H),
                    V32,
                    F32,
                    cam,
                    int(seed),
                    int(max(1, frames)),
                    sampling_mode_int,
                    int(max(0, debug_mode)),
                )
                if verbose:
                    print("Rendered with GPU")
                return img, {}
        except Exception:
            pass
    except Exception:
        pass

    # CPU fallback
    tracer = PathTracer()
    img = tracer.render_rgba(
        W,
        H,
        scene=None,
        camera=cam,
        seed=int(seed),
        frames=max(1, int(frames)),
        use_gpu=False,
        mesh=None,
        denoiser="off",
        svgf_iters=5,
        luminance_clamp=None,
    )
    if verbose:
        print("Rendered with CPU")
    return img, {}


def render_object(
    vertices: np.ndarray,
    indices: np.ndarray,
    *,
    size: tuple[int, int] = (1200, 900),
    frames: int = 8,
    seed: int = 7,
    camera: Optional[Dict[str, Any]] = None,
    show_in_viewer: bool = False,
    viewer_mode: "Literal['image','mesh']" = "image",
) -> np.ndarray:
    """Render a 3D triangle mesh object using the GPU mesh path (with CPU fallback).

    Parameters
    ----------
    vertices : np.ndarray (N,3) float32/float64
    indices : np.ndarray (M,3) int/uint
    size : (int, int)
        Output image size (W,H)
    frames : int
        Accumulation frames for the GPU tracer
    seed : int
        RNG seed
    camera : dict | None
        Optional camera from make_camera(); if None, auto-framed from bounds
    show_in_viewer : bool
        If True, opens an interactive viewer window for preview.
    viewer_mode : {"image","mesh"}
        Choose whether to preview the ray-traced image ("image") or open the
        interactive mesh viewer ("mesh"). Defaults to "image".
    """
    V = np.asarray(vertices)
    F = np.asarray(indices)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("vertices must have shape (N,3)")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("indices must have shape (M,3)")

    V32 = np.ascontiguousarray(V.astype(np.float32))
    F32 = np.ascontiguousarray(F.astype(np.uint32))

    W, H = int(size[0]), int(size[1])
    cam = camera if camera is not None else _autoframe_camera(V32, (W, H))

    # Try native GPU mesh tracer
    try:
        from . import _forge3d as _native  # type: ignore[attr-defined]
        try:
            status = _native.device_probe("metal").get("status", "unavailable")  # type: ignore
            if status == "ok":
                img = _native._pt_render_gpu_mesh(
                    int(W),
                    int(H),
                    V32,
                    F32,
                    cam,
                    int(seed),
                    int(max(1, frames)),
                    None,
                    int(max(0, debug_mode)),
                )
                if show_in_viewer:
                    if str(viewer_mode).lower() == "mesh":
                        try:
                            # Open interactive mesh viewer
                            _native.open_mesh_viewer(
                                V32,
                                F32,
                                width=int(W),
                                height=int(H),
                                title="forge3d Object Viewer",
                                vsync=True,
                                fov_deg=45.0,
                                znear=0.1,
                                zfar=1000.0,
                            )
                        except Exception:
                            # Fallback to image viewer if mesh viewer not available
                            try:
                                from ._viewer import open_viewer_image as _open_viewer_image
                                _open_viewer_image(img, title="forge3d Object Preview")
                            except Exception:
                                pass
                    else:
                        try:
                            from ._viewer import open_viewer_image as _open_viewer_image
                            _open_viewer_image(img, title="forge3d Object Preview")
                        except Exception:
                            pass
                return img
        except Exception:
            pass
    except Exception:
        pass

    # CPU fallback
    tracer = PathTracer()
    img = tracer.render_rgba(
        W,
        H,
        scene=None,
        camera=cam,
        seed=int(seed),
        frames=max(1, int(frames)),
        use_gpu=False,
        mesh=None,
        denoiser="off",
        svgf_iters=5,
        luminance_clamp=None,
    )
    if show_in_viewer:
        if str(viewer_mode).lower() == "mesh":
            try:
                from . import _forge3d as _native  # type: ignore[attr-defined]
                _native.open_mesh_viewer(
                    V32,
                    F32,
                    width=int(W),
                    height=int(H),
                    title="forge3d Object Viewer",
                    vsync=True,
                    fov_deg=45.0,
                    znear=0.1,
                    zfar=1000.0,
                )
            except Exception:
                # Fallback to image viewer
                try:
                    from ._viewer import open_viewer_image as _open_viewer_image
                    _open_viewer_image(img, title="forge3d Object Preview")
                except Exception:
                    pass
        else:
            try:
                from ._viewer import open_viewer_image as _open_viewer_image
                _open_viewer_image(img, title="forge3d Object Preview")
            except Exception:
                pass
    return img
