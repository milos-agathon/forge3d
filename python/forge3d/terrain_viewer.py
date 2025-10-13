"""Interactive 3D terrain viewer utilities."""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


def heightmap_to_mesh(
    heightmap: np.ndarray,
    spacing: Tuple[float, float] = (1.0, 1.0),
    vertical_scale: float = 1.0,
    subsample: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert heightmap to triangle mesh for 3D viewing.
    
    Args:
        heightmap: 2D array of elevation values (H, W)
        spacing: (sx, sy) pixel spacing in world units
        vertical_scale: Scale factor for elevation (exaggeration)
        subsample: Sample every Nth pixel (1=full res, 2=half res, etc.)
        
    Returns:
        vertices: (N, 3) float32 array of vertex positions
        uvs: (N, 2) float32 array of texture coordinates
        indices: (M, 3) uint32 array of triangle indices
    """
    h, w = heightmap.shape
    sx, sy = spacing
    
    # Subsample for performance
    if subsample > 1:
        heightmap = heightmap[::subsample, ::subsample]
        sx *= subsample
        sy *= subsample
        h, w = heightmap.shape
    
    # Generate vertex grid
    # Center the terrain at origin
    x = np.arange(w, dtype=np.float32) * sx - (w * sx * 0.5)
    y = np.arange(h, dtype=np.float32) * sy - (h * sy * 0.5)
    xv, yv = np.meshgrid(x, y)
    
    # Generate UV coordinates (0 to 1)
    u = np.linspace(0, 1, w, dtype=np.float32)
    v = np.linspace(0, 1, h, dtype=np.float32)
    uv, vv = np.meshgrid(u, v)
    uvs = np.stack([uv, vv], axis=-1).reshape(-1, 2)
    
    # Stack to (H, W, 3) then flatten to (N, 3)
    # Use Y as up axis: [X, Y, Z] -> [X, elevation, Z]
    vertices = np.stack([
        xv,
        heightmap.astype(np.float32) * vertical_scale,
        yv,
    ], axis=-1).reshape(-1, 3)
    
    # Generate triangle indices for grid
    indices = []
    for row in range(h - 1):
        for col in range(w - 1):
            # Two triangles per quad
            # v0---v1
            # |   / |
            # |  /  |
            # | /   |
            # v2---v3
            v0 = row * w + col
            v1 = row * w + col + 1
            v2 = (row + 1) * w + col
            v3 = (row + 1) * w + col + 1
            
            # Triangle 1: v0, v2, v1
            indices.append([v0, v2, v1])
            # Triangle 2: v1, v2, v3
            indices.append([v1, v2, v3])
    
    indices = np.array(indices, dtype=np.uint32)
    
    return vertices, uvs, indices


def open_terrain_viewer_3d(
    heightmap: np.ndarray,
    *,
    texture_rgba: Optional[np.ndarray] = None,
    spacing: Tuple[float, float] = (1.0, 1.0),
    vertical_scale: float = 1.0,
    subsample: int = 1,
    width: int = 1280,
    height: int = 720,
    title: str = "forge3d 3D Terrain Viewer",
) -> None:
    """Open interactive 3D terrain viewer with camera controls.
    
    Args:
        heightmap: 2D elevation array (H, W)
        texture_rgba: Optional (H, W, 4) RGBA texture to apply to terrain
        spacing: (sx, sy) pixel spacing in world units
        vertical_scale: Vertical exaggeration factor
        subsample: Subsample factor (1=full, 2=half, 4=quarter, etc.)
        width: Viewer window width
        height: Viewer window height
        title: Window title
        
    Controls:
        Tab: Toggle between Orbit and FPS camera modes
        Orbit mode:
            - Drag: Rotate camera
            - Scroll: Zoom in/out
        FPS mode:
            - WASD: Move forward/left/backward/right
            - Q/E: Move down/up
            - Mouse: Look around (hold left button)
            - Shift: Move faster
        Esc: Exit
    """
    try:
        from . import _forge3d as _native
    except ImportError as e:
        raise RuntimeError(f"Native module not available: {e}") from e
    
    print(f"[Terrain3D] Converting {heightmap.shape} heightmap to mesh (subsample={subsample})...")
    vertices, uvs, indices = heightmap_to_mesh(heightmap, spacing, vertical_scale, subsample)
    
    # Subsample texture to match mesh if provided
    if texture_rgba is not None and subsample > 1:
        texture_rgba = texture_rgba[::subsample, ::subsample]
    
    print(f"[Terrain3D] Mesh: {len(vertices)} vertices, {len(indices)} triangles")
    
    # Calculate mesh bounds
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    bounds_center = (bounds_min + bounds_max) * 0.5
    bounds_size = bounds_max - bounds_min
    
    print(f"[Terrain3D] Vertex bounds: X[{bounds_min[0]:.1f}, {bounds_max[0]:.1f}], "
          f"Y[{bounds_min[1]:.1f}, {bounds_max[1]:.1f}], "
          f"Z[{bounds_min[2]:.1f}, {bounds_max[2]:.1f}]")
    print(f"[Terrain3D] Bounds center: [{bounds_center[0]:.1f}, {bounds_center[1]:.1f}, {bounds_center[2]:.1f}]")
    print(f"[Terrain3D] Bounds size: [{bounds_size[0]:.1f}, {bounds_size[1]:.1f}, {bounds_size[2]:.1f}]")
    
    # Calculate camera distance to fit entire terrain in view
    # Use the maximum dimension to ensure everything fits
    max_dim = max(bounds_size[0], bounds_size[2])  # X and Z dimensions
    fov_rad = np.radians(60.0)
    camera_distance = (max_dim * 0.5) / np.tan(fov_rad * 0.5) * 1.5  # 1.5x for margin
    
    # Position camera above and behind the terrain center, looking down at it
    camera_offset_y = bounds_size[1] * 0.8  # Slightly above the terrain
    camera_offset_z = camera_distance * 0.6  # Behind the terrain
    
    camera_eye = np.array([
        bounds_center[0],
        bounds_center[1] + camera_offset_y,
        bounds_center[2] + camera_offset_z,
    ], dtype=np.float32)
    
    camera_target = bounds_center.astype(np.float32)
    
    print(f"[Terrain3D] Camera eye: [{camera_eye[0]:.1f}, {camera_eye[1]:.1f}, {camera_eye[2]:.1f}]")
    print(f"[Terrain3D] Camera target: [{camera_target[0]:.1f}, {camera_target[1]:.1f}, {camera_target[2]:.1f}]")
    print(f"[Terrain3D] Camera distance: {camera_distance:.1f}")
    
    # Convert to proper numpy arrays with correct dtypes and C-contiguous layout
    vertices_2d = np.ascontiguousarray(vertices, dtype=np.float32)
    uvs_2d = np.ascontiguousarray(uvs, dtype=np.float32)
    indices_2d = np.ascontiguousarray(indices, dtype=np.uint32)
    
    if not hasattr(_native, "open_mesh_viewer"):
        raise RuntimeError("Mesh viewer not available in this build")
    
    # Prepare texture if provided
    texture_data = None
    tex_width, tex_height = 0, 0
    if texture_rgba is not None:
        if texture_rgba.ndim == 3 and texture_rgba.shape[2] == 4:
            texture_data = np.ascontiguousarray(texture_rgba, dtype=np.uint8)
            tex_height, tex_width = texture_rgba.shape[:2]
            print(f"[Terrain3D] Using texture: {tex_width}x{tex_height} RGBA")
        else:
            print(f"[Terrain3D] Warning: texture_rgba must be (H,W,4), got {texture_rgba.shape}")
    
    _native.open_mesh_viewer(
        vertices_2d,
        indices_2d,
        uvs=uvs_2d,
        texture_rgba=texture_data,
        texture_width=tex_width,
        texture_height=tex_height,
        camera_eye=camera_eye,
        camera_target=camera_target,
        width=width,
        height=height,
        title=title,
        vsync=True,
        fov_deg=60.0,
        znear=0.1,
        zfar=100000.0,
    )
