# DEM-resolution offline terrain mesh

Opt into DEM-driven mesh density with `camera_mode="mesh:zup:dem"` (or
`"mesh:dem"` for the legacy orbit camera). The `dem` suffix composes with
the existing `zup` and `north` options. Existing camera modes keep their
current geometry and output.

```python
from forge3d import TerrainRenderParams
from forge3d.terrain_params import make_terrain_params_config

params = TerrainRenderParams(make_terrain_params_config(
    size_px=(1080, 1080), render_scale=1.0, msaa_samples=1,
    z_scale=1.0, exposure=1.0, domain=(0.0, 2000.0),
    albedo_mode="colormap", colormap_strength=1.0,
    camera_mode="mesh:zup:dem",
    terrain_span=41600.0,
    cam_radius=30000.0, cam_theta_deg=62.0, clip=(600.0, 221600.0),
))
# Pass params and the full float32 DEM to TerrainRenderer.render_with_aov,
# TerrainRenderer.render_terrain_pbr_pom, or render_offline as usual.
```

For a DEM of shape `(height, width)`, the procedural mesh has
`W = max(width, 2)` by `H = max(height, 2)` vertices and
`2 * (W - 1) * (H - 1)` triangles. Each axis visits its source samples,
including on rectangular DEMs. A singleton axis repeats its sole sample
to supply the two vertices needed for a quad. The terrain still uses one
span for both world axes. This is a sampling-density control, not a new geospatial
extent or camera convention. Heights retain the existing texture filtering
and height-curve behavior.

The selected density is used for beauty, AOV, offline accumulation, planar
water reflections and terrain shadow casters. Default mesh rendering stays at
512 vertices per side, with its existing 1024-vertex shadow-caster grid.
In DEM mode, a source axis below 1024 samples therefore uses fewer shadow
vertices along that axis than the default; shadows can become coarser.
Shading normals still come from the full height texture.

For cached beauty renders, opting into DEM density invalidates the shadow
pass as well as the beauty pass. Existing modes retain their shadow cache
keys. AOV and offline cache support are unchanged.

A 3000-by-3000 DEM therefore uses 17,988,002 triangles and approximately
13.87 m grid spacing over a 41.6 km domain, versus 522,242 triangles and
81.41 m with the default grid. The opt-in mode performs more GPU vertex work
and draws the full grid on every frame.
No vertex/index arrays are built on the CPU. Draw counts that exceed the
native `u32` draw range are rejected instead of wrapping.

The earlier upload stall was separate from mesh density: unused CPU coarse
ambient occlusion scanned up to 288 neighbours per height texel. Current
renderers skip it when neither `ao_weight` nor raw-AO debug mode consumes it.
Explicit coarse-AO consumers retain that work. See the
[cost acceptance evidence](../terrain-native-cost-acceptance-evidence.md).
