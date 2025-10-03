# python/forge3d/adapters/mpl_data.py
# M2: Matplotlib Adapter (Data)
# - Parse common Matplotlib artists into polygon/line point arrays
# - Extrude polygons and thicken lines to forge3d meshes via geometry API
# - Optional text-to-mesh via TextPath (SDF-like outline polygons)

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

try:
    import matplotlib
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon as MplPolygon, Rectangle, PathPatch
    from matplotlib.path import Path
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False
    Axes = object  # type: ignore


def is_matplotlib_available() -> bool:
    return _HAS_MPL


def _require_mpl():
    if not _HAS_MPL:
        raise ImportError("Matplotlib is required. Install with: pip install matplotlib")


# -----------------------
# Artist extraction (2D)
# -----------------------

def extract_lines_from_axes(ax: 'Axes') -> List[np.ndarray]:
    """Extract polylines (N,2) float32 arrays from Line2D artists on an Axes."""
    _require_mpl()
    out: List[np.ndarray] = []
    for line in ax.lines:  # type: ignore[attr-defined]
        if not isinstance(line, Line2D):
            continue
        x = np.asarray(line.get_xdata(), dtype=np.float32)
        y = np.asarray(line.get_ydata(), dtype=np.float32)
        if x.size == 0 or y.size == 0 or x.size != y.size:
            continue
        pts = np.stack([x, y], axis=-1)
        out.append(pts)
    return out


def _polygon_from_patch(patch) -> Optional[np.ndarray]:
    if isinstance(patch, MplPolygon):
        xy = np.asarray(patch.get_xy(), dtype=np.float32)
        if xy.ndim == 2 and xy.shape[1] >= 2:
            return xy[:, :2]
    if isinstance(patch, Rectangle):
        x, y = patch.get_xy()
        w, h = patch.get_width(), patch.get_height()
        return np.asarray(
            [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
                [x, y],
            ],
            dtype=np.float32,
        )
    if isinstance(patch, PathPatch):
        path: Path = patch.get_path()
        verts = np.asarray(path.vertices, dtype=np.float32)
        if verts.ndim == 2 and verts.shape[1] >= 2:
            return verts[:, :2]
    return None


def extract_polygons_from_axes(ax: 'Axes') -> List[np.ndarray]:
    """Extract polygon rings (M,2) float32 arrays from Patch artists on an Axes.

    Note: Holes are not resolved here; for complex PathPatch with multiple rings,
    the consumer should split by MOVETO/CLOSEPOLY as needed. Here we provide a
    simple exterior approximation to keep scope focused.
    """
    _require_mpl()
    out: List[np.ndarray] = []
    for patch in getattr(ax, 'patches', []):
        poly = _polygon_from_patch(patch)
        if poly is not None and poly.shape[0] >= 3:
            out.append(poly.astype(np.float32))
    return out


# -----------------------
# Text to polygon outlines
# -----------------------

def text_to_polygons(
    text: str,
    *,
    font: Optional['FontProperties'] = None,
    size: float = 12.0,
) -> List[np.ndarray]:
    """Convert a text string to a list of polygon rings using Matplotlib TextPath.

    Returns a list of (N,2) float32 point arrays in glyph space.
    """
    _require_mpl()
    if font is None:
        font = FontProperties()
    tp = TextPath((0, 0), text, size=size, prop=font)
    path: Path = tp
    polys: List[np.ndarray] = []
    verts = np.asarray(path.vertices, dtype=np.float32)
    codes = path.codes
    if codes is None:
        if verts.shape[0] >= 3:
            polys.append(verts[:, :2])
        return polys
    # Split by MOVETO/CLOSEPOLY
    current: List[np.ndarray] = []
    for v, c in zip(verts, codes):
        if c == Path.MOVETO:
            if current:
                arr = np.vstack(current)
                if arr.shape[0] >= 3:
                    polys.append(arr)
                current = []
            current.append(v[:2])
        elif c in (Path.LINETO, Path.CURVE3, Path.CURVE4):
            current.append(v[:2])
        elif c == Path.CLOSEPOLY:
            if current:
                arr = np.vstack(current)
                if arr.shape[0] >= 3:
                    polys.append(arr)
                current = []
    if current:
        arr = np.vstack(current)
        if arr.shape[0] >= 3:
            polys.append(arr)
    return polys


# -----------------------
# Mesh generation helpers
# -----------------------

def extrude_polygons_to_meshes(polygons: List[np.ndarray], height: float = 1.0, cap_uv_scale: float = 1.0):
    """Extrude a list of 2D polygon rings to prism meshes via forge3d.geometry."""
    from .. import geometry as fgeo
    meshes = []
    for poly in polygons:
        try:
            mesh = fgeo.extrude_polygon(poly.astype(np.float32), float(height), cap_uv_scale=float(cap_uv_scale))
            meshes.append(mesh)
        except Exception:
            continue
    return meshes


def thicken_lines_to_meshes(lines: List[np.ndarray], width_world: float = 1.0, join_style: str = 'miter', miter_limit: float = 4.0):
    """Thicken polylines into ribbon meshes using forge3d.geometry.generate_thick_polyline."""
    from .. import geometry as fgeo
    meshes = []
    for path in lines:
        try:
            mesh = fgeo.generate_thick_polyline(path.astype(np.float32), float(width_world), join_style=str(join_style), miter_limit=float(miter_limit))
            meshes.append(mesh)
        except Exception:
            continue
    return meshes


def line_width_world_from_pixels(pixel_width: float, z: float, fov_y_deg: float, height_px: int) -> float:
    """Approximate world width for a desired pixel width at depth z given camera params."""
    import math
    fov = math.radians(float(fov_y_deg))
    return float(pixel_width) * (2.0 * float(z) * math.tan(0.5 * fov) / max(1, int(height_px)))
