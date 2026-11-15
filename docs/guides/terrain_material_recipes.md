# Terrain material recipes

`TerrainMaterialRecipe` reuses an appearance on another prepared extent without
editing WGSL. It composes albedo layers, geographic/data masks, cartographic
relief and an optional categorical palette. `MapScene` still owns native geometry,
camera rendering, frame validation and bundle replay.

For the shared terrain/raster/vector/building authoring interface, see
[Geospatial materials](geospatial_materials.md).

```python
import rasterio  # load the real package before adding repository python/ to sys.path
import forge3d as f3d

grid = f3d.MaterialGrid(
    crs="EPSG:32632", origin=(500000, 4600000), spacing=(2, 2),
    units="m", elevation_units="m",
)
cover = f3d.CategoricalRasterStyle(
    palette={1: (80, 140, 50, 255), 2: (20, 90, 220, 255)},
    labels={1: "Forest", 2: "Water"}, nodata=0,
)
rock = f3d.MaterialMask(
    "all", children=(
        f3d.MaterialMask("range", field="slope", units="degrees", lower=45, upper=90),
        f3d.MaterialMask("range", field="elevation", units="m", lower=1000, upper=3000),
    ),
)
material = f3d.TerrainMaterialRecipe(
    layers=(
        f3d.TerrainMaterialLayer("ground", (170, 125, 80)),
        f3d.TerrainMaterialLayer("rock", (160, 160, 165), mask=rock),
    ),
    categorical=cover, presentation="protected", relief_strength=0.6,
)
material.save("material.json")
# heights and class_ids are already aligned arrays supplied by the mapmaker.
prepared = material.prepare(heights, grid, classes=class_ids)
prepared.save_raster("material.png")
bound = material.bind(scene_recipe, grid, classes=class_ids)
scene = f3d.MapScene(recipe=bound)
scene.render("map.png")
scene.save_bundle("map.forge3d")
```

The elevation and slope thresholds above are authoring choices in this example,
not observations of rock or snow. Measured class meaning remains in the supplied
categorical raster and palette. `land_cover()` provides an approachable preset
over this same recipe; `with_layer()` changes a named layer without changing the
original recipe.

The example's optional MP4 uses `libx264rgb` at CRF 0 to preserve palette bytes.
It requires an FFmpeg build with that encoder and a player supporting H.264
High 4:4:4 RGB. PNG frames remain the canonical output. Lossy YUV 4:2:0 video
can alter swatches through chroma subsampling and is not an exact-color proof.

The workflow CLI's camera elevation is above the geographic horizon. It uses
the existing `mesh:zup:north` camera and converts that elevation to the native
polar angle from +Z. Its evidence records both angles and east/north/up axes;
use those values when matching another tool's camera.
Azimuth follows the native camera: zero is east and positive angles turn toward
north. It is not a clockwise compass bearing.

## Ground scale and masks

The grid origin is the north-west **sample centre**, not a pixel-edge affine
origin. X increases east; rows increase south. Horizontal and elevation units
must be metres and the CRS must be projected in metres. Geographic coordinates,
feet, rotated rasters and differently aligned categorical inputs need preparation
before this route. `bind()` checks supplied terrain bounds against the grid.
The existing native 3D mesh has a square horizontal footprint. This binding
rejects non-square ground extents in 3D to avoid silently stretching repeat
metres. Array preparation and the flat raster output retain rectangular grids.
Prepare a square extent for the qualified native camera route.

Masks support `range`, `classes`, `all`, `any` and `invert`. Built-in fields are
unexaggerated `elevation` in metres, `slope` in degrees, and compass downslope
`aspect` in degrees. Flat terrain has no aspect. An aspect range whose lower
endpoint exceeds its upper endpoint wraps north. Additional fields need explicit
units and the same shape; masks never normalize values to the current crop.
Ranges can omit either bound to express an open-ended measured threshold;
nonfinite observations still never match.
Categorical nodata is excluded from the observed class domain, including inverse
class masks. Class arrays must be real numeric data; object, string and complex
arrays are rejected rather than silently losing IDs at the native boundary.

`ramp` uses two increasing bounds in the field's units and the standard
[`smoothstep` interpolation](https://registry.khronos.org/OpenGL-Refpages/gl4/html/smoothstep.xhtml).
It yields zero below the lower bound, one above the upper bound, and smooth
weights between. `all` intersects weights by their minimum, `any` by their
maximum, and `invert` subtracts from one within the observed domain. Binary
masks retain their boolean behavior. Use these weights for rock/soil/snow
material transitions; categorical class masks and protected colors remain exact.
The thresholds are persisted geographic/data parameters, independent of camera
or tile. No random seed or additional native texture fetch is introduced.

Slope and relief use source elevation, independently of camera exaggeration.
Use `derivative_halo` with one actual neighbouring elevation sample around each
edge for adjacent crops. This is the central-difference stencil's support, not a
tile-size limit. The halo's centre must exactly match the supplied elevations.
Without a halo, outer edges use NumPy's one-sided derivative. Missing neighbours
leave derived slope/aspect undefined and cartographic relief neutral; they are
not inferred measurements.

`relief_mode="single"` keeps the existing directional hillshade. The optional
`relief_mode="multidirectional"` weights four hillshades by
`sin(aspect - light_azimuth)**2`, following the Mark/USGS method documented in
[GDAL's implementation](https://github.com/OSGeo/gdal/blob/master/apps/gdaldem_lib.cpp).
The default group illuminates from 225, 270, 315 and 360 degrees; changing
`relief_azimuth` rotates the group. Flat areas use equal weights. This extends
the existing hillshade rather than duplicating the terrain renderer.

This is **cartographic shading**, applied in linear light before the protected
categorical overlay. It changes no class memberships or legend swatches, and
physical presentation rejects baked relief to prevent double shading. It retains
Forge3D's central differences and 0.18 ambient term, without Mark's smoothed
aspect; it does not claim pixel equivalence with GDAL's Horn-derivative default.
[GDAL's documented multidirectional option](https://gdal.org/en/stable/programs/gdaldem.html#hillshade)
and rayshader's documented
[Lambert shading](https://www.rayshader.com/reference/lamb_shade.html) are reference
routes. Appearance comparisons must disclose those derivative, ambient and
normalization differences; neither mode is a declared visual winner.

```python
texture = f3d.MaterialTexture(
    "rock-albedo.png", repeat_m=(16, 16),
    color_space="srgb", channel="albedo", anchor=(500000, 4600000),
)
material = material.with_layer("ground", texture=texture)
```

`repeat_m` sets source texture scale from a fixed projected anchor. Grid sampling
integrates the source texels over each ground footprint in linear light and
premultiplied alpha, then returns straight alpha. New textures persist
`variation_seed=0`: smooth world-cell random phases remove the observed 32 m
lattice. Opaque albedo uses variance-normalized phase blending; transparent
coverage keeps its area average. Adjacent extents and halos use the same world
coordinates. Irregular point samples use bilinear sampling unless `footprint_m`
is supplied. `variation_seed=None` retains the periodic baseline; old serialized
textures without the seed retain that setting and their canonical identity.

Albedo images are RGB/RGBA sRGB. Normal and roughness images are linear data.
OpenGL normal export means +X east, +G north, +Z outward. The native handoff
flips G because the terrain shader's bitangent points south; exported sidecars
state both conventions. A controlled 20-degree north/east tilt produces a visible
six-code display response with the same direction as real tilted geometry. This
qualifies that fixture; asset-specific appearance still needs visual review. See
[current evidence](c01_material_validation.md).

Recipe files preserve relative texture references. Loading resolves them against
the recipe's directory; binding freezes resolved paths into the scene recipe.
Saving a loaded recipe in another directory adjusts relative references to keep
pointing at the original assets; it does not copy them.
On different Windows drives the saved reference is absolute, since a relative
path cannot name another drive.
Scene bundles retain external asset references under the existing bundle
contract. They do not pack those textures or convert materials to another
renderer. In-place texture edits participate in raster cache identity.

## Physical light and thematic meaning

`presentation="protected"` keeps native physical lighting on surface layers,
including their normal/roughness channels. Optional prepared relief is declared
cartographic shading combined with that lighting, recorded in render metadata.
After lighting, categorical swatches are painted only onto visible terrain using
the same native frame's terrain and scene depth. Buildings, ridges and sky
therefore occlude paint. Native camera unprojection supplies pixel-centre UV;
class cells use `floor(uv * grid_size)` and ClampToEdge, without blending.
Terrain wins exact terrain/building depth ties. The product uses debug mode zero.
The legend uses the palette's exact labels and RGBA values, rather than colours
inferred from a shaded image. Nodata in the categorical layer leaves the physical
surface visible; it does not create a new category.

`presentation="physical"` treats the prepared colours as surface albedo and lets
the existing renderer light them. The legend describes that source albedo;
shadowed output pixels differ. Cartographic relief is rejected in this mode to
avoid double shading. Class membership does not change with lighting.

The protected path supports display PNG/RGBA, native camera sequences, offline
accumulation, denoising and HDR/EXR. Offline paint uses an unjittered same-camera
native visibility reference. Denoised building scenes capture it before
accumulation to exclude non-terrain pixels from terrain-guided filtering.
Display swatches are applied after
tonemapping/denoise; EXR stores their float32 linear-sRGB equivalents while
unpainted physical pixels retain scene radiance. Small outputs share the native
beauty's nearest resize for depth/UV. With water, protected swatches stay exact
and unprotected pixels retain the physical effect. With cloud shadows and
screen-space effects, shared one-shot RGBA8 beauty supports scene-linear HDR
and same-draw terrain visibility. This supports protected terrain and raster
paint with map furniture. TAA must be disabled for either export. Accumulation,
denoising, public AOVs, provenance, PNG16 and protected paint combined with
native label, vector or point overlays or protected building identity remain
unavailable on this route.
Native material presentation requires opaque coverage of all
valid terrain samples, using a ground layer where needed. Other raster colour
sources cannot silently overwrite or reorder this presentation. A legend that
does not fit the output raises an actionable error rather than truncating classes.
Physical raster preparation and the exported PNG/JSON sidecar are CPU operations;
they are not represented as GPU frames.

## A still and camera sequence from another extent

`examples/terrain_material_recipe.py` takes a saved recipe, a DEM and an optional
already aligned categorical GeoTIFF. It writes the prepared raster and meaning
sidecar, native still, replay bundle, camera frames, and an optional MP4, together
with source/grid/camera/adapter/timing metadata. Frame count, output size and
camera settings are explicit workload choices:

```text
python examples/terrain_material_recipe.py --dem dem.tif --classes cover.tif \
  --recipe material.json --source-credit "Source and edition" --output result \
  --elevation-units m --vertical-datum "Source vertical reference" \
  --size 1280 720 --frames 48 --fps 24 --azimuth 140 160 --elevation 55 \
  --fov 45 --camera-distance 1.2 --movie
```

Use `--window COL ROW WIDTH HEIGHT` to substitute the extent. The example reads
actual neighbouring samples for derivatives. Movie encoding uses the existing
external FFmpeg executable; no renderer comparison tools are runtime dependencies.

Supported authoring coverage here is north-up terrain albedo, typed masks,
Lambert relief, categorical meaning and the declared raster/native display routes.
Continuous building/vector surface textures, stochastic/hex tiling, new relief algorithms, Unreal
material graphs and cross-backend visual equivalence require separate scoped
implementation and evidence. This document does not assert those deliveries or
a competitive advantage over Blender/rayshader.


Physical and protected recipes can add `normal=MaterialTexture(..., color_space="linear",
channel="normal", normal_convention="opengl")`, scalar `roughness`, or a
`roughness_texture` with `channel="roughness", color_space="linear"`. Albedo
still uses sRGB RGB/RGBA. Normal images are opaque RGB/RGBA; roughness images
are grayscale `L`. These data channels use the same projected repeat metres,
anchor, masks and opacity as colour. Normal directions blend then normalize;
roughness blends linearly. When roughness is composed, each layer supplies it,
so the recipe never invents a material value for an omitted layer.

Normal RGB encodes tangent/bitangent/outward XYZ, with the OpenGL positive-Y
convention. The existing native terrain shader builds its tangent from its
geometric normal and a reference axis; this is its terrain TBN frame, not a
promise of arbitrary imported mesh UV tangents. Geometric slope masks continue
to use the original DEM, independently of shading normals. Maps are prepared on
the declared DEM/value grid: details below that sampling footprint need a finer
prepared grid. This route adds no per-map WGSL.

`bind()` and `bind_raster()` send the composed maps to the existing native
terrain material inputs. In an aligned stack, each raster overrides only its
declared channels and only at valid cells, in layer order. The native material
pass requires equal final coverage for normal and roughness when both are
present. Referenced asset bytes participate in frame cache identity, including
in-place edits. Protected classes remain exact while other
pixels consume the surface channels under lighting. Feature paint and
building scatter reject these channels because their current binding does not
consume them. Milos explicitly deferred these channels and world-scale building
wall/roof textures to follow-on `MAP-MESH-MATERIAL-CHANNELS` on 2026-10-15;
all four families support physical and protected colour in C01.
`evaluate_samples()` returns normal/roughness authoring arrays;
`prepare_raster()` returns its colour result, while the bound style also prepares
surface data for native rendering.

`PreparedTerrainMaterial.save_raster()` exports any composed maps next to the
colour PNG as `.normal.png` and `.roughness.png`, with linear-channel and normal
convention metadata in the sidecar. Recipe/scene bundle round trips retain all
channel definitions; external texture references require those assets to remain
accessible. Exported rasters preserve appearance at their declared grid scale;
they do not export a procedural Blender/Unreal shader graph.


`GeospatialMaterialRecipe.rock_snow(ground, rock, snow, ...)` provides a second
preset beside `land_cover()`. Supply the actual layer assets and explicit
`rock_slope_degrees`, `snow_elevation_m` and `snow_slope_degrees` transition pairs.
It intersects existing layer masks with those measured-unit rules and retains
normal/roughness channels. Snow thresholds express art direction unless an
observational input establishes snow cover. The preset uses the ordinary recipe
compositor, overrides and serialization; it does not select regional thresholds.


For the exercised adapter, real-data demonstration, reference comparison and
remaining support limits, see [C01 validation](c01_material_validation.md).
