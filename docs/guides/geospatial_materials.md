# One recipe across map surfaces

C01 remains partial. Terrain, raster, vector and building materials are the
owner's required scope; the availability limits below do not waive it. Combined
physical surfaces and exact classes use native same-frame visibility. C01 remains
partial pending the remaining qualification and merge evidence.
See [current evidence](c01_material_validation.md).

`GeospatialMaterialRecipe` extends the same layer compositor, metre grid,
typed mask and palette contract as [terrain recipes](terrain_material_recipes.md).
It binds authoring decisions to the existing terrain, raster, vector and building
implementations. It does not introduce another renderer or silently emulate an
unsupported channel.

| Surface | Binding | Declared output interpretation |
| --- | --- | --- |
| Terrain | `recipe.bind(scene_recipe, grid, ...)` | Physically lit layers and composed normal/roughness; protected classes painted as exact sRGB on visible terrain |
| Raster | `recipe.prepare_raster(values, grid, units=...)`, `bind_raster(...)` | Typed value rules, optional categorical overlay, ground-scaled albedo and physical normal/roughness; optional measured elevation for relief |
| Provided vectors | `recipe.bind_vector(layer, anchors_m, ...)` | Exact protected paint, or physically lit scalar paint draped into terrain albedo before lighting |
| Provided building features | `recipe.bind_buildings(layer, anchors_m, ...)` | Lit scalar albedo or exact protected swatches using same-frame per-batch identity |
| Irregular samples | `recipe.evaluate_samples(...)` | Explicit projected-metre sample positions, optional ENU normals, fields and class IDs; an authoring result, not a render certificate |

All routes use the same `MaterialLayer`, `MaterialMask`, texture channel
definitions, override mechanism, serialization and categorical legend source.
Terrain's existing class name remains compatible. Recipes can be loaded with
`GeospatialMaterialRecipe.load()` and saved without rewriting relative asset
references.

```python
import forge3d as f3d

cover = f3d.CategoricalRasterStyle(
    {1: (80, 140, 50, 255), 2: (20, 90, 220, 255)},
    {1: "Forest", 2: "Water"},
)
material = f3d.GeospatialMaterialRecipe(
    (f3d.MaterialLayer("surface", (170, 125, 80)),), categorical=cover,
)
# Anchors, fields and classes must correspond to provided features in order.
# Geometry itself stays owned by the existing layer/preparation API.
vectors = material.bind_vector(vector_layer, vector_anchors_m, units="m", classes=vector_classes)
buildings = material.bind_buildings(building_layer, building_anchors_m,
    units="m", classes=building_classes, presentation="physical")
composed = material.attach(scene_recipe, vectors, buildings)
f3d.MapScene(recipe=composed).render("map.png")
```

Anchors and their CRS must describe the same prepared source as the geometry.
Vector layers supply `crs`; building layers use `metadata["crs"]`. The bindings
preserve feature IDs, geometry and unrelated properties. They put colour decisions
in the reserved `forge3d:material` and `forge3d:material:color` properties, and
retain the recipe and true palette legend in layer metadata. Apply the same
recipe again after replacing inputs or changing layer parameters. `attach()`
replaces layers with matching IDs, preserves other layers, and deduplicates the
shared publication legend while retaining separate meanings for each binding.

For raster rules, the supplied values are the `value` field and their units are
mandatory. Every other raster field/class array must share its shape. Relief
requires a separate unexaggerated elevation grid; generic raster values are never
treated as measured terrain height. Raster binding also requires the scene's
prepared terrain grid, shape and projected CRS to agree.
When that raster binding uses an in-memory DEM, it adds an explicit material
geometry snapshot to the scene metadata so bundle replay retains measured
geometry. Existing bare terrain bundles keep their descriptor-only contract.

`evaluate_samples()` accepts Nx2 or Nx3 positions in projected metres. Z, when
provided, is measured `elevation`. `normals_enu` means east/north/up **data normals**,
not sRGB normal-map pixels. It derives surface slope/aspect only when those normals
are supplied. Typed external fields cannot overwrite provided geometry fields.
An inverted range does not turn a missing observation into a matching sample.

The current vector and building bindings are **per-feature scalar albedo**.
They reject repeating surface textures instead of sampling one texel and calling
it a textured mesh. Terrain/raster texture sampling remains ground-planar with
declared repeat metres and phase. Milos's 2026-10-15 decision defers continuous
wall/roof textures and vector/building normal/roughness maps to roadmap ticket
`MAP-MESH-MATERIAL-CHANNELS`. Those channels remain rejected rather than being
represented by a single sampled texel. Both colour presentations are implemented
for all four families; this follow-on is a recorded scope decision.

Physical vector paint uses projected terrain-grid cell centres (`all_touched=False`).
Polygons own cells whose centres fall inside; lines/points buffer their geometry
by the declared ground width/radius. `width_world` is in metres. Legacy pixel
widths are interpreted in authoring-grid cells for this albedo path, independent
of output resolution. Paint uses straight-alpha source-over in linear sRGB;
native terrain albedo filtering blends its display edges before lighting.
Protected vectors retain exact, opaque pixel-centre paint after lighting.

Vector protected paint requires opaque styles. Building albedo is decoded from
declared sRGB bytes to linear factors before native lighting; its displayed pixels
change with sun/shadow. Its legend describes source albedo and the publication
caption states that difference. Protected buildings co-write an R32Uint batch ID
with the winning beauty fragment. Only selected IDs receive their exact display
swatches after lighting; lit neighbours retain their sun response. Buildings,
ridges and terrain share the same depth test. Authored C01 building meshes are
converted from their construction Y-up coordinates to native world Z-up, keeping
their metre heights and ground contact independent of DEM resolution.

CRS-tagged building footprints use the prepared terrain's grid or bounds, sharing
the vector projection frame. Their supplied heights remain metres, scaled by
the declared terrain exaggeration. They no longer stretch their own combined
footprint bounds to fill the terrain or scale height with DEM resolution.
`building_coordinate_space="terrain_crs"` records this route. Untagged legacy
screen-space buildings retain their existing coordinate interpretation.

Each attached legend entry carries its own `binding` and `interpretation`.
Physical entries in every family say that lit pixels differ; protected entries
describe exact swatches. Mixed legends retain both meanings,
including when class IDs and swatch colours coincide.

Protected terrain/raster recipes keep surface lighting enabled. Their class
swatches are composited after lighting using the actual frame's 32-bit terrain
and scene depth, including native scatter buildings. Native camera unprojection
recovers the visible terrain UV; pixel-centre cell membership uses `floor(uv *
grid_size)` with the last cell owning the outer edge. Classes never blend across
a boundary. Polygon fills require visible terrain; elevated strokes test scene
depth and can remain visible against the sky. Both use the native scene depth.
Strict depth testing makes terrain win exact terrain/building depth ties.
Mesh and clipmap scenes use Less with depth writes on both ordinary and AOV
routes. Fullscreen terrain retains its overlay policy on both routes, because
its foreground quad does not represent physical scene depth.
No product colour uses a shader debug mode.

Fullscreen C01 paint uses the native quad's pixel-centre UVs (north at the top),
with native building IDs excluding overlaid building pixels. It does not use
the perspective-camera inverse. A protected polygon without an explicit outline
owns visible terrain pixel centres, including its holes. Explicit outlines and
strokes use a half-coverage ownership rule and write the complete swatch;
antialiased edge colours are never mixed into protected output.

`relief_strength` remains independent of class membership. In protected mode,
nonzero relief is declared cartographic shading combined with physical lighting;
`last_render_metadata["material_shading"]` records that interpretation. Physical
presentation continues to reject relief, avoiding an implicit second shading
operation. A bound terrain material can carry aligned C01 material/categorical
rasters in declared order. Their valid samples compose into its surface colour;
protected bindings retain independent exact paint. For each physical channel,
the last aligned recipe that supplies it takes precedence at valid cells.
Raster nodata leaves the earlier channel or native default in place. Normal
and roughness maps must have the same final coverage when both are present,
because the native material pass shares one mask. An unequal-coverage stack,
a mismatched grid, or a competing legacy material map raises an error.

Protected/vector composition also supports offline accumulation, HDR/EXR and
denoise. It captures an unjittered native reference with the identical camera,
geometry and material state. With denoised C01 buildings this reference precedes
accumulation so it also selects valid guides; otherwise it follows resolve.
Exact paint is applied after
denoise and tonemapping. EXR protected pixels store the float32 linear-sRGB
equivalents of the declared display swatches; untouched physical pixels retain
their scene radiance. Metadata records this mixed interpretation. Output below
the native 64-pixel minimum shares the beauty frame's existing nearest resize
for depth and terrain UV, so the same retained native samples own visibility.

Water retains its physical effect on unprotected pixels. The shared one-shot
path can export scene-linear HDR after cloud and screen-space effects and can
apply protected terrain or raster paint using depth and surface ownership from
the same draw. Map furniture is applied after the paint. HDR and protected
paint require TAA to be disabled. Accumulation, denoising, public AOVs, provenance, PNG16 and
protected building or native overlay combinations with protected paint still
raise `MapSceneNativeUnavailable` on this path.

Offline terrain normal/albedo guides do not describe scatter buildings. C01
therefore
uses the native reference's terrain ownership to exclude buildings and sky from
terrain-guided denoising: their accumulated colour passes through unchanged.
This avoids filtering a building using the ground normal underneath it. The
metadata records this guide coverage; it does not claim building guide AOVs.

Authored materials use actual rendered geometric normals as their physical
base, then compose the declared tangent normals. Legacy terrain retains its
stylized Sobel normal treatment. A controlled 20-degree north/east tilt changes
visible pixels by six display codes toward/away from the corresponding sun,
matching real tilted geometry; this qualifies that fixture, not every asset.

For authored C01 materials, native triplanar albedo and its surface identity use
the displaced geometric normal. Shading-normal detail does not change texture
projection ownership. Legacy recipes retain their existing projection policy.
The qualification separates the CPU world-phase bake from native periodic
triplanar sampling; it does not claim that phase variation runs in every native
projection. Building wall/roof textures remain in the approved follow-on ticket.

The one attached legend keeps separate entries and captions for each binding,
even when their swatches match. Captions wrap within the output. Set its optional
`origin_px=[x, y]` to move the panel away from important map features; coordinates
are rounded to pixels and the complete panel must fit inside the output.

Local physical proof uses an RTX 3070/Vulkan. The vector proof uses the existing
precise Python compositor over a native GPU terrain frame, explicitly reported
as `python_precise_raster` for protected paint and `native_terrain_uv_albedo` for
physical paint; buildings use `terrain_scatter_instanced_mesh`.
This does not establish other adapters/backends, photographic material quality,
or matched competitive superiority.

The shared workflow example accepts `--vectors` and `--buildings` prepared
FeatureCollections, `--feature-crs`, and `--class-field`. Supply typed property
bindings with `--feature-field NAME PROPERTY UNITS`. Its scalar anchors are the
first supplied geographic vertices, stated explicitly rather than inferred
centroids. `--feature-recipe` selects an explicit scalar preset in the same
language when the terrain recipe uses unsupported feature textures. The saved
evidence includes the feature sources, preset and binding interpretations.
Use `--vector-presentation physical|protected`, `--vector-width-world METRES` and
`--building-presentation physical|protected` to choose their colour behaviour.

Material preparation is reused within a render/shared sequence context. Source
array values, masks, fields, geometry grid and texture bytes participate in the
key; caller mutations cannot alter retained results. Retained arrays use the
existing native host-allocation ledger and are released at context exit. A
denied retention allocation leaves ordinary uncached preparation available.
This reuses CPU material work; every uncached beauty request still draws a fresh
native frame. No interactive-viewer or preview-speed guarantee follows from it.


For the exercised adapter, real-data demonstration, reference comparison and
remaining support limits, see [C01 validation](c01_material_validation.md).
