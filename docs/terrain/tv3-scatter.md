# TV3: Terrain Scatter and Population

Epic TV3 is implemented as a terrain-native scatter workflow layered on top of the existing instanced-mesh renderer.

## What shipped

- New Python module: `forge3d.terrain_scatter`
- Deterministic placement generators:
  - `seeded_random_transforms(...)`
  - `grid_jitter_transforms(...)`
  - `mask_density_transforms(...)`
- Terrain-aware sampling through `TerrainScatterSource`, including:
  - terrain-width contract coordinates
  - scaled-height placement
  - slope sampling
  - normalized elevation helpers
- Placement filters through `TerrainScatterFilters` for slope and elevation bands.
- Shared batch contract for both offscreen terrain rendering and the interactive viewer.
- Renderer/viewer hooks:
  - `apply_to_renderer(renderer, batches)`
  - `apply_to_viewer(viewer, batches)`
- Runtime observability:
  - `TerrainRenderer.get_scatter_stats()`
  - `TerrainRenderer.get_scatter_memory_report()`

## Public API

```python
import forge3d as f3d
from forge3d import terrain_scatter as ts

source = ts.TerrainScatterSource(heightmap, z_scale=1.4)
transforms = ts.grid_jitter_transforms(
    source,
    spacing=18.0,
    seed=13,
    jitter=0.6,
    scale_range=(8.0, 14.0),
    filters=ts.TerrainScatterFilters(
        max_slope_deg=35.0,
        min_elevation=source.min_height,
        max_elevation=source.min_height + 120.0,
    ),
)

batch = ts.TerrainScatterBatch(
    name="trees",
    color=(0.20, 0.42, 0.18, 1.0),
    max_draw_distance=2200.0,
    transforms=transforms,
    levels=[
        ts.TerrainScatterLevel(
            mesh=f3d.geometry.primitive_mesh("cone", radial_segments=12),
            max_distance=900.0,
        ),
        ts.TerrainScatterLevel(
            mesh=f3d.geometry.primitive_mesh("box"),
            max_distance=1600.0,
        ),
    ],
)

ts.apply_to_renderer(renderer, [batch])
stats = renderer.get_scatter_stats()
memory = renderer.get_scatter_memory_report()
```

## Workflow notes

- Placement is deterministic for a fixed source, seed, and filter set.
- LOD thresholds must be strictly increasing, and only the final LOD may omit `max_distance`.
- `viewer_orbit_radius(...)` returns a terrain-width-scaled orbit radius that matches the scatter coordinate contract.
- The viewer IPC path uses the same logical batch/LOD structure as the offscreen renderer.

## Example and tests

- Example: `python examples/terrain_tv3_scatter_demo.py`
- Unit and integration tests: `tests/test_terrain_scatter.py`
- Viewer IPC tests: `tests/test_viewer_ipc.py`

The bundled TV3 demo renders the same scatter batches both ways:

- offscreen through `TerrainRenderer`
- interactive through `open_viewer_async(...)`
