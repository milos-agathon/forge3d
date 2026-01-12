# examples/_m_3d_preview.py
# Shared 3D preview utilities for Workflow M examples.
# - Renders forge3d.geometry.MeshBuffers via Matplotlib 3D and saves deterministic PNG
# - Falls back gracefully if Matplotlib is unavailable

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from forge3d.helpers.offscreen import save_png_deterministic

# Optional Matplotlib import
try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # side-effect import
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

try:
    from forge3d.adapters import mpl_rasterize_figure
except Exception:
    mpl_rasterize_figure = None  # type: ignore

try:
    import forge3d.geometry as fgeo
    # P0.3/M2: Sun ephemeris - calculate realistic sun position from location and time
    from forge3d import sun_position, sun_position_utc, SunPosition
except Exception:
    fgeo = None  # type: ignore
    sun_position = None  # type: ignore
    sun_position_utc = None  # type: ignore
    SunPosition = None  # type: ignore


def _figure_inches(width: int, height: int, dpi: int) -> tuple[float, float]:
    return (float(width) / float(dpi), float(height) / float(dpi))


def _set_equal_3d(ax, pts: np.ndarray) -> None:
    # Set equal aspect for 3D axes based on point cloud bounds
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ranges = maxs - mins
    max_range = float(np.max(ranges)) if np.all(np.isfinite(ranges)) else 1.0
    if max_range <= 0:
        max_range = 1.0
    center = (maxs + mins) * 0.5
    for ctr, set_lim in zip(center, (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        set_lim(float(ctr - 0.5 * max_range), float(ctr + 0.5 * max_range))


def render_meshes_preview(
    meshes: List["fgeo.MeshBuffers"],
    out_png: str | Path,
    *,
    width: int = 800,
    height: int = 600,
    dpi: int = 150,
    elev: float = 30.0,
    azim: float = 35.0,
    color: tuple[float, float, float] = (0.9, 0.6, 0.2),
    alpha: float = 1.0,
) -> Optional[str]:
    """Render a list of MeshBuffers to a 3D preview PNG.

    Returns the path to the written PNG, or None if preview was skipped.
    """
    if not _HAS_MPL or mpl_rasterize_figure is None:
        print("3D preview skipped: Matplotlib adapter unavailable")
        return None

    # Collect all points for aspect
    all_pts = []
    for m in meshes:
        try:
            all_pts.append(np.asarray(m.positions, dtype=np.float32))
        except Exception:
            pass
    if not all_pts:
        print("3D preview skipped: no mesh positions available")
        return None
    pts = np.concatenate(all_pts, axis=0)

    fig_w, fig_h = _figure_inches(width, height, dpi)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.set_axis_off()

    # Plot each mesh via plot_trisurf; interpret Y as up-axis
    for mesh in meshes:
        try:
            pos = np.asarray(mesh.positions, dtype=np.float32)
            idx = np.asarray(mesh.indices, dtype=np.uint32).reshape(-1, 3)
            if pos.ndim != 2 or pos.shape[1] != 3 or idx.ndim != 2 or idx.shape[1] != 3:
                continue
            # Map to axes: X->X, Z->Y, Y->Z so that +Y is vertical in our preview
            X = pos[:, 0]
            Y = pos[:, 2]
            Z = pos[:, 1]
            ax.plot_trisurf(X, Y, Z, triangles=idx, linewidth=0.2, antialiased=True,
                            color=color, alpha=alpha, shade=True)
        except Exception:
            continue

    try:
        _set_equal_3d(ax, pts[:, [0, 2, 1]])  # match axis mapping above
    except Exception:
        pass

    # Rasterize deterministically using our adapter to ensure consistent PNG bytes
    try:
        rgba = mpl_rasterize_figure(fig, dpi=dpi, facecolor='white')
        out_path = Path(out_png)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_png_deterministic(str(out_path), rgba)
        plt.close(fig)
        return str(out_path)
    except Exception:
        # Fall back to vanilla savefig if adapter fails
        try:
            out_path = Path(out_png)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out_path), dpi=dpi, facecolor='white')
            plt.close(fig)
            return str(out_path)
        except Exception:
            plt.close(fig)
            return None


def make_displaced_plane_from_heightmap(
    heightmap: np.ndarray,
    *,
    scale: float = 10.0,
    resolution: tuple[int, int] = (128, 128),
) -> Optional["fgeo.MeshBuffers"]:
    """Create a plane mesh and displace it by a heightmap using UV sampling.

    Requires forge3d.geometry native bindings.
    """
    if fgeo is None:
        print("Displacement skipped: forge3d.geometry unavailable")
        return None
    try:
        # Generate a plane with UVs; sample using UV space for heightmap
        plane = fgeo.primitive_mesh('plane', resolution=resolution)
        hm = np.asarray(heightmap, dtype=np.float32)
        mesh = fgeo.displace_heightmap(plane, hm, scale=float(scale), uv_space=True)
        return mesh
    except Exception as exc:
        print(f"Displacement failed: {exc}")
        return None
