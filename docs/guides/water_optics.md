# Caller-driven terrain water

`TerrainRenderer` accepts water optics through the existing native
`TerrainRenderParams` wrapper. Omitted `WaterSettings.optics` preserves legacy
masked-water shading. `WaterOpticsSettings()` selects physical shading;
`WaterOpticsSettings(mode="procedural")` explicitly selects legacy shading.
An unset water mask leaves the land path unchanged.

```python
from forge3d import WaterSettings, WaterOpticsSettings
from forge3d.terrain_params import make_terrain_params_config

config = make_terrain_params_config(
    size_px=(512, 512), render_scale=1.0, msaa_samples=1,
    terrain_span=1000.0, z_scale=1.0, exposure=1.0, domain=(0.0, 1.0),
    camera_mode="mesh:zup",
    water=WaterSettings(optics=WaterOpticsSettings(
        surface_height=simulated_surface,  # same (rows, columns) as heightmap
        color=(0.24, 0.13, 0.055),         # linear RGB sediment colour
        roughness=0.02,
    )),
)
# Pass forge3d.TerrainRenderParams(config), heightmap and water_mask to
# render_terrain_pbr_pom, render_with_aov, or the offline accumulation API.
```

`surface_height` is the absolute water elevation in heightmap units, before
`z_scale`; it is not a depth or a shore-distance ratio. It displaces masked grid
vertices. Normals are derived from this grid in Z-up world coordinates. Optional
`surface_normals` has shape `(rows, columns, 3)` and supplies world normals after
`z_scale`. Either input may be supplied independently. With neither, physical
shading uses the existing mesh geometry and its geometric normal. Both grids use
the heightmap's row orientation; the renderer does not flip them. Values must be
finite, grids must match the heightmap, and normals must have positive Z, as a
heightfield normal does, including after float32 normalization. Inputs that lose
positive Z at GPU precision are rejected. This prevents zero normals when
neighbouring texels are interpolated. The Python
settings snapshot arrays into compact immutable float32 buffers, so later
simulation changes require a new settings object/native wrapper. `asdict()`
exports ordinary nested lists, preserving the JSON representation. Direct and
AOV renders upload the current snapshot each time; offline accumulation uploads
it at setup and reuses it for that accumulation. There is no native surface cache.

`color=None` keeps the caller's material or overlay colour. An explicit colour
overrides the overlay on water only. Colours use finite linear RGB values in
`[0, 1]`. `roughness` is perceptual roughness in `[0, 1]`, with no terrain roughness
floor. A supplied `MaterialLayerSettings.roughness_path` takes precedence through
the existing material-map mask. Land keeps its existing roughness floor.

Physical water uses dielectric Fresnel reflection (IOR 1.33), the environment
sky, and a ray traversal through the original terrain grid. Terrain hits supply
material/overlay colour; misses sample the sky. The sun uses a GGX glint from the
same surface normal. Terrain ray visibility, CSM visibility and an enabled sun
visibility map suppress that glint. The physical path does not use the legacy
fixed blue colours, procedural waves or shore-distance depth tint. The existing
planar-reflection settings and procedural path remain available.

The sediment/body colour receives sky irradiance and shadowed sunlight. It is
dark when both light sources are off. Reflected terrain uses those light sources
as well, rather than a fixed ambient brightness.
For direct `TerrainRenderer` calls, sky lighting comes from the supplied
`env_maps` IBL object's intensity; set that intensity to zero to disable it.

The physical geometry path currently requires a mesh camera. Screen, clipmap and
globe cameras reject physical masked-water input explicitly. Terrain reflection
uses the procedural mesh grid and local material maps; it is not a general scene
reflection for meshes, buildings or off-grid objects. Roughness filters the sky
and reflected material textures; terrain silhouette reflection uses a single
surface ray. This work supplies optics, not FLUVIUS lake geometry or the DILUVIUM
flood solver.

Mesh geometry retains the renderer's fixed 512-by-512 vertex grid; finer supplied
grids improve shading normals but cannot add silhouette detail beyond that mesh.
Physical mask, height and encoded normal data share the existing mask texture,
so optics adds no sampled-texture binding (the assembled terrain shader remains
at the baseline count of 30). Independent renders on other backends remain absent.

Run `python examples/terrain_water_optics.py` from the rebuilt development
environment for physical-GPU measurements and images under
`artifacts/water-optics`. Existing goldens, signed certificates and determinism
hashes are not regenerated. Physical water is outside the existing shader proof's
zero-water-mask input domain; GPU execution is separate evidence from that proof.
The runtime observer records physical and masked water as `excluded` from that
proof domain, without aborting rendering. It still rejects failed safety checks;
observing a frame is not a claim that physical optics has a static proof.

`python -m pytest tests/test_terrain_water_optics.py` checks the input and JSON
contracts and runs the analytic, lighting, shadow-control and proof-exclusion
regressions when the existing terrain GPU capability gate is available. These
tests also run in the NVIDIA/Vulkan visual acceptance lane, whose existing
zero-skip check requires the GPU tests to execute. Terrain goldens on that lane
use `FORGE3D_TERRAIN_GOLDEN_VARIANT=nvidia-vulkan`.
