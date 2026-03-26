# Terrain Material Virtual Texturing

Terrain material virtual texturing is shipped for the terrain renderer's material path. The current shipped scope focuses on paged albedo-family streaming for large terrains while keeping the public Python contract stable for later family expansion.

## What shipped

- Public terrain VT config surface: `VTLayerFamily` and `TerrainVTSettings` in `python/forge3d/terrain_params.py`
- Terrain renderer VT source API: `TerrainRenderer.register_material_vt_source()`, `clear_material_vt_sources()`, and `get_material_vt_stats()`
- Native decode and validation path: `src/terrain/render_params/decode_vt.rs` and `src/terrain/render_params/native_vt.rs`
- Terrain VT runtime: `src/terrain/renderer/virtual_texture.rs`
- Terrain shader integration and shader-side feedback writes: `src/shaders/terrain_pbr_pom.wgsl`
- Bind-group wiring and render-path integration across beauty, AOV, and offline terrain renders
- Example/demo coverage: `examples/terrain_tv20_virtual_texturing_demo.py`
- Regression coverage: `tests/test_tv20_virtual_texturing.py`

## Public API

```python
import forge3d as f3d
from forge3d.terrain_params import TerrainVTSettings, VTLayerFamily, make_terrain_params_config

config = make_terrain_params_config(
    size_px=(1280, 720),
    terrain_span=2400.0,
    z_scale=1.6,
    albedo_mode="material",
    colormap_strength=0.0,
    camera_mode="mesh",
)
config.vt = TerrainVTSettings(
    enabled=True,
    atlas_size=2048,
    residency_budget_mb=64.0,
    max_mip_levels=6,
    layers=[VTLayerFamily(family="albedo", virtual_size_px=(4096, 4096))],
)

renderer.clear_material_vt_sources()
renderer.register_material_vt_source(
    0,
    "albedo",
    rgba8_source,
    (4096, 4096),
    [0.5, 0.5, 0.5, 1.0],
)

params = f3d.TerrainRenderParams(config)
frame = renderer.render_terrain_pbr_pom(
    material_set=material_set,
    env_maps=ibl,
    params=params,
    heightmap=heightmap,
)
stats = renderer.get_material_vt_stats()
```

## Workflow Notes

- `TerrainVTSettings(enabled=False)` and `vt=None` preserve the pre-VT terrain material path.
- The shipped native runtime currently pages only the `albedo` family.
- `normal` and `mask` are still accepted in the Python contract and registration API so callers can keep a stable forward-compatible schema.
- Registering non-`albedo` sources logs a warning and stores the payload, but the native terrain runtime does not decode or sample those families yet.
- Feedback-driven residency uses the terrain shader's storage-buffer writes plus CPU readback through `FeedbackBuffer`. The old unused generic feedback compute shader path was removed.
- Source registration builds the RGBA mip chain on the CPU with box-filter downsampling at registration time.
- Changing the registered source set invalidates the current runtime instance and rebuilds residency state on the next render. That keeps page tables and atlas contents coherent, but it also drops the previous resident-tile cache.
- `get_material_vt_stats()` reports resident page count, total page count, cache budget, hit/miss counts, miss rate, streamed tiles, evictions, upload timings, resident memory, active source count, and feedback request count.

## Current Boundary

- Terrain VT is material-path paging, not a general material-graph system.
- The shader path is wired for paged terrain albedo sampling plus fallback colors.
- The Python contract is already shaped for future normal/mask paging, but that native decode/sampling work is intentionally out of the shipped v1 scope.

## Example And Tests

- Example: `python examples/terrain_tv20_virtual_texturing_demo.py`
- Regression tests: `tests/test_tv20_virtual_texturing.py`
- Design reference: `docs/superpowers/specs/2026-03-23-tv20-terrain-material-virtual-texturing-design.md`
