# Curves & Tubes (F17)

Generate ribbon (flat strip) and tube meshes along a 3D path.
These helpers build consistent frames along the path and produce meshes with UVs and normals.

## API (Python)

- `forge3d.geometry.generate_ribbon(path: np.ndarray, width_start: float, width_end: float, *, join_style: str = "miter", miter_limit: float = 4.0, join_styles: Optional[np.ndarray] = None) -> MeshBuffers`
  - Builds a two-sided strip with width tapering from `width_start` to `width_end`.
  - `join_style` controls corner behavior: `"miter"`, `"bevel"`, or `"round"`.
  - `miter_limit` clamps miter length at sharp angles.
  - `join_styles` (optional) is a per-vertex uint8 array overriding `join_style` (encoding: `0=miter`, `1=bevel`, `2=round`).
  - UVs use path length as U, and [0,1] across the ribbon as V.

- `forge3d.geometry.generate_tube(path: np.ndarray, radius_start: float, radius_end: float, *, radial_segments: int = 16, cap_ends: bool = True) -> MeshBuffers`
  - Builds a tube with circular cross-section along the path, with tapered radius.
  - `radial_segments` controls circle tessellation; `cap_ends` adds end caps.

### Thick Polyline (F3)

- `forge3d.geometry.generate_thick_polyline(path: np.ndarray, width_world: float, *, depth_offset: float = 0.0, join_style: str = "miter", miter_limit: float = 4.0) -> MeshBuffers`
  - Generates a ribbon-like mesh along a 3D polyline with constant world-space width.
  - `depth_offset` can be used to push geometry slightly in Z to avoid z-fighting when overlaid.
  - For constant pixel width at camera distance `z`: `width_world ≈ pixel_width * (2*z*tan(fov_y/2) / image_height)`

## Example

```python
import numpy as np
from forge3d.geometry import generate_ribbon, generate_tube

# Helix path
turns, steps_per_turn = 3, 64
T = np.linspace(0.0, turns * 2.0 * np.pi, turns * steps_per_turn, dtype=np.float32)
path = np.stack([np.cos(T), np.sin(T), 0.25 * T / (2.0 * np.pi)], axis=1)

join_styles = np.array([0, 1, 2, 0], dtype=np.uint8)  # 0=miter, 1=bevel, 2=round per vertex
ribbon = generate_ribbon(path, width_start=0.2, width_end=0.05, join_style="miter", miter_limit=4.0, join_styles=join_styles)
tube = generate_tube(path, radius_start=0.15, radius_end=0.05, radial_segments=24, cap_ends=True)
print(ribbon.vertex_count, ribbon.triangle_count)
print(tube.vertex_count, tube.triangle_count)
```

### Example: Thick Polyline

```python
import math, numpy as np
from forge3d.geometry import generate_thick_polyline

def pixel_to_world_width(pixel_width: float, z: float, fov_y_deg: float, image_height_px: int) -> float:
    f = math.tan(math.radians(fov_y_deg) * 0.5)
    return pixel_width * (2.0 * z * f / image_height_px)

path = np.array([[0,0,0],[2,0,0],[2,1,0]], dtype=np.float32)
width = pixel_to_world_width(3.0, z=4.0, fov_y_deg=45.0, image_height_px=1080)
poly = generate_thick_polyline(path, width_world=width, depth_offset=0.001, join_style="round")
```
