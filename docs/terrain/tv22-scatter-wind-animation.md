# TV22 — Scatter Wind Animation

GPU-driven wind deformation for terrain scatter batches. Vegetation sways on the GPU with per-batch controls, distance-aware fade, and deterministic replay.

## Quick Start

```python
from forge3d import terrain_scatter as ts

wind = ts.ScatterWindSettings(
    enabled=True,
    amplitude=2.0,       # max bend displacement (contract-space units)
    speed=0.8,           # sway frequency scalar
    rigidity=0.3,        # 0 = fully flexible, 1 = rigid
    gust_strength=0.5,   # additive gust amplitude
    gust_frequency=0.3,  # gust temporal frequency
)

batch = ts.TerrainScatterBatch(
    levels=[ts.TerrainScatterLevel(mesh=tree_mesh)],
    transforms=transforms,
    wind=wind,
)

ts.apply_to_renderer(renderer, [batch])
frame = renderer.render_terrain_pbr_pom(
    material_set, ibl, params, heightmap,
    time_seconds=0.5,  # explicit clock for deterministic replay
)
```

## ScatterWindSettings Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `False` | Master switch. `False` = strict no-op. |
| `direction_deg` | float | `0.0` | Wind azimuth: 0 = +X, 90 = +Z in contract XZ. |
| `speed` | float | `1.0` | Sway frequency scalar (>= 0). |
| `amplitude` | float | `0.0` | Max bend displacement in contract-space units (>= 0). `0` = no-op. |
| `rigidity` | float | `0.5` | Resistance to bending [0, 1]. `1.0` = no sway. |
| `bend_start` | float | `0.0` | Normalized mesh height [0, 1] where bending begins. |
| `bend_extent` | float | `1.0` | Normalized height range above `bend_start` that bends (> 0). |
| `gust_strength` | float | `0.0` | Additive gust amplitude in contract-space units (>= 0). |
| `gust_frequency` | float | `0.3` | Gust temporal frequency scalar (>= 0). |
| `fade_start` | float | `0.0` | Distance where fade begins (>= 0). |
| `fade_end` | float | `0.0` | Distance where wind reaches zero (>= 0). |

**No-fade rule:** Wind is un-faded when `fade_end <= fade_start`.

## time_seconds Parameter

The `time_seconds` parameter is an explicit, caller-owned clock:

- **Offscreen:** Pass `time_seconds=<float>` to `render_terrain_pbr_pom()` or `render_with_aov()`. Default is `0.0`.
- **Viewer:** Time is accumulated automatically from the frame loop's `dt`.
- **Determinism:** Same `time_seconds` + same wind settings = same deformation. No internal random state.

## Mesh Convention

Wind bend weighting uses mesh-local `position.y` normalized by `mesh_height_max`. Meshes should have **Y=0 at the base** and Y increasing upward. Meshes that violate this (e.g., `primitive_mesh("cone")` with y in [-0.5, 0.5]) should be shifted before use:

```python
from forge3d.geometry import MeshBuffers

mesh = f3d.geometry.primitive_mesh("cone", radial_segments=10)
positions = mesh.positions.copy()
positions[:, 1] -= positions[:, 1].min()  # shift base to Y=0
mesh = MeshBuffers(positions=positions, normals=mesh.normals, uvs=mesh.uvs, indices=mesh.indices)
```

A warning is logged at runtime if `mesh_y_min` deviates from zero by more than 5% of the mesh Y extent.

## Per-Batch Controls

Each `TerrainScatterBatch` has its own `ScatterWindSettings`. Different batches can have different wind parameters. For example, tall trees can have high amplitude with low rigidity, while bushes have low amplitude with high rigidity:

```python
trees = ts.TerrainScatterBatch(
    levels=tree_levels, transforms=tree_transforms,
    wind=ts.ScatterWindSettings(enabled=True, amplitude=3.0, rigidity=0.2),
)
bushes = ts.TerrainScatterBatch(
    levels=bush_levels, transforms=bush_transforms,
    wind=ts.ScatterWindSettings(enabled=True, amplitude=0.5, rigidity=0.7),
)
ts.apply_to_renderer(renderer, [trees, bushes])
```

## Distance Fade

When `fade_end > fade_start`, wind displacement smoothly fades to zero between those distances (in approximate render-space units). Instances beyond `fade_end` have zero wind displacement. This prevents shimmer and saves GPU budget for distant vegetation.

## Backward Compatibility

- `wind` defaults to `ScatterWindSettings()` (disabled). Existing code that does not set `wind` is unaffected.
- `time_seconds` defaults to `0.0`. Existing render calls without this parameter are unaffected.
- The shader early-exits (zero cost) when wind is disabled or amplitude is zero.

## Accepted Limitations

1. **Spatial phase is render-path dependent.** The viewer and offscreen paths produce different per-vertex phase offsets for the same scene.
2. **Fade distances are approximate.** `fade_start/end * instance_scale` uses a uniform scale factor; the offscreen path's `render_from_contract` is anisotropic.
3. **Per-instance scale affects displacement.** Larger instances (higher `scale_range`) sway proportionally more. This is physically realistic for vegetation.
4. **Translation/basis frame mismatch.** TV22 operates in mesh-local frame to work around a pre-existing issue in scatter instance packing.

## Example

See `examples/terrain_tv22_scatter_wind_demo.py` for a complete example rendering wind-animated vegetation on Mount Fuji.
