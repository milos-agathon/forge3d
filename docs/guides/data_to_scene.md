# Reusable local data preparation (C05)

For executed Rotterdam and Grindelwald/Eiger workflows, real-data recipes,
camera exports and the QGIS comparison, see [the C05 demonstrations](c05_real_data.md).

Declare a `PreparationRecipe`, execute it with `prepare_scene`, and save the
result with `PreparedScene.save`. Reuse the saved snapshot without its original
sources, or load its `recipe.json` to prepare a new source edition or region.
The output includes values, validity masks, source hashes, metadata, coordinate
operations, coverage and independent checkpoint residuals. This is separate from
rendering: a preparation report does not certify a GPU frame or an editor import.

The route uses the existing optional `forge3d[raster]` dependencies. With the
source checkout, import the real `rasterio` before adding `python/` to `sys.path`;
that directory also contains a test stub. `examples/prepare_scene.py` does this.

## Supported input contract

| Meaning | Local source | Operation |
| --- | --- | --- |
| Elevation | One-band GeoTIFF or `.npy` with a declared grid | Convert declared height units to metres, apply the supported vertical operation, then geographically warp to the exact target grid |
| Continuous measurement | One-band GeoTIFF or declared `.npy` grid | Float64 warp, bilinear by default; nearest/cubic are explicit alternatives |
| Categorical | Integer one-band GeoTIFF or declared `.npy` grid | Nearest only, preserving class IDs and a separate validity mask |
| Count | Integer-valued nonnegative GeoTIFF or declared `.npy` grid | Preserve native cells and precision; reject analysis resampling, including nearest |
| Imagery | Grayscale/RGB/RGBA GeoTIFF or band-first declared `.npy` grid | Geographic warp; uint8 imagery remains byte imagery, with separate validity |
| Vector | GeoJSON FeatureCollection plus declared CRS | Transform 2D points, lines, polygons, multi-geometries and collections; preserve IDs/properties |
| Observation | GeoJSON point features plus CRS and measurement units | Transform positions; retain values, dates and disclosure properties without aggregation or interpolation |

The source and target grids use a six-number **GDAL corner geotransform**:
`[origin_x, pixel_x, row_x, origin_y, column_y, pixel_y]`. A sample at column
`c`, row `r` is at transform `(c + 0.5, r + 0.5)`. Matching dimensions do not
establish alignment. CRS declarations that contradict a file are rejected.

Elevation requires a source and target vertical reference. Keeping the same
declared datum is supported, including NAP, LN02 and EGM2008. The existing native
EGM96 model supports `orthometric_egm96` ↔ `ellipsoidal` conversions at source
sample centres before warping. Other datum conversions refuse with a correction
message; externally prepared inputs must retain their operation provenance.
Units are `m`, `ft` (0.3048 m) or `us_survey_ft` (1200/3937 m). A height label
does not perform a conversion. Horizontal units come from the CRS.

GeoTIFF masks/nodata, nonfinite samples, a finite explicit `nodata`, and optional
boolean `.npy` disclosure masks all contribute to validity. True means usable.
No zero sentinel is invented, so category zero and zero population survive.
Scale/offset metadata is decoded for continuous data; explicit nodata is tested
in the raw stored sample domain first. Encoded discrete inputs
must be decoded and verified externally. Measurements retain float64 precision.
Warped categorical IDs must be exactly representable by the float64 warp scratch
path; large IDs can be mapped to a documented smaller lookup table. Native count
values never pass through that scratch representation.

Counts remain on their original grid, including CRS and per-cell meaning. Only
their coverage mask and eventual **display image** are sampled for the target
view. The report distinguishes this from analysis resampling. Conservative
aggregation, catalogue/network acquisition, NetCDF/HDF5, generic scientific
resampling and 3D vector height transformations are outside this local route.
Unknown routes fail instead of selecting a replacement provider or algorithm.

## Fuse detailed terrain with regional context

`fuse_terrain` takes local single-band elevation sources in priority order,
highest first. Declare each input as `bare_earth` or `DSM` and supply height
units and a vertical reference. GeoTIFF georeferencing and masks are read by the
same path as `prepare_scene`. For a NumPy grid, supply the affine, CRS and
dimensions in `PreparationSource.grid`. GDAL reports corner transforms for both
PixelIsArea and PixelIsPoint files. Sample positions are always computed from
the reported corners at `(column + 0.5, row + 0.5)`.

The target must be projected and axis-aligned. Non-square cells and projected
linear units such as survey feet are supported. `transition_width_m` is in
metres in the projected plane. Source grids may be rotated or georeferenced in
a different horizontal CRS. Choose a north-up target in metres when using the
existing `PreparedScene.to_scene` and `PreparedScene.viewer_assets` routes.

```python
import forge3d as f3d

target = f3d.PreparationGrid("EPSG:2056", 2000, 2000, (2560000, 1, 0, 1142000, 0, -1))
fine = f3d.PreparationSource("fine", "lidar.tif", "elevation", units="m", vertical_reference="LN02")
regional = f3d.PreparationSource("regional", "regional.tif", "elevation",
                                 units="m", vertical_reference="LN02")
fusion = f3d.fuse_terrain(
    (fine, regional), target, vertical_reference="LN02",
    surface_kinds={"fine": "bare_earth", "regional": "bare_earth"},
    transition_width_m=100,
)
directory = fusion.save("fused-terrain")
recipe = f3d.PreparationRecipe.load(directory / "recipe.json")
prepared = f3d.prepare_scene(recipe)
prepared.save("prepared-scene")
```

The example assumes two local GeoTIFFs with the declared reference and coverage.
Fusion checks vertical reference and surface tags at the dataset and band levels.
To blend bare earth and surface models deliberately, set
`allow_mixed_surfaces=True`. The result carries the `mixed` surface tag. For
inputs without a common verified datum, set `vertical_reference="unresolved"`
and `allow_unresolved_vertical_reference=True`. Fusion converts heights to
metres, retains the source declarations in the receipt, and labels the result
`unresolved`. Both options still require the input declarations to agree with
the file tags.
Fusion intersects the available validity masks and excludes samples matching
either file or explicit nodata before decoding scale and offset. Only finite
measurements contribute.

Choose a grid and width for your data. Where both a higher-priority DEM and the
lower-priority composite are valid, measure the Euclidean distance `d` from
the higher-priority sample centre to the nearest exposed edge of its union of
valid warped cells. Internal holes have exposed edges. A target crop retains
exterior validity wherever an edge could affect the transition.
For width `w`, use `t = clamp(d / w, 0, 1)` and higher-priority weight `t²(3 − 2t)`. This
weight has zero slope at both endpoints. The raster samples that continuous
weight at cell centres. Where just one surface is valid, it supplies the
height directly. Cells with no valid source have a false mask and NaN height.

The declared target centre determines which source pixel supplies coverage.
Across CRSs, fusion transforms that centre with PyProj from `forge3d[raster]`.
Centres outside the source or its projection domain, or in a masked source
pixel, are invalid. Nearest sampling copies the decoded value of the containing
pixel. Whole-cell windows at matching resolution copy values and validity
directly.

Bilinear and cubic interpolation use fixed GDAL windows on the target lattice
and exact coordinate transforms when reprojecting. If a window loses a sample
whose declared centre has coverage, fusion retries it on the original parent
affine with the same interpolator and an exact transform. An unresolved sample
raises `fusion_interpolation_failed` with the source and parent cell.
Parallel same-CRS grids use the source-to-target pixel scale for their filters.
For crops, derive the target as
`whole_grid.window(column, row, width, height)`. It carries the parent origin
and integer offsets into saved recipes and fusion receipts. Shared cells then
read the same resampled values and exterior context. Independently calculated
crop corners can differ from the parent lattice due to floating-point
translation.
For a standalone reference through `prepare_scene`, set the elevation source's
`resampling` to `fusion_nearest`, `fusion_bilinear`, or `fusion_cubic` in the
recipe. Use the method and target lattice from fusion. These options use
fusion's input validation, mask policy, datum conversion, and resampler. They
give the uncorrected height from the source before weighting or calibration.
For a surface model, declare `metadata={"surface_kind": "DSM"}` on the source.
Use an `unresolved` recipe reference for a units-only comparison.
The usual `nearest`, `bilinear`, and `cubic` preparation methods derive their
warp from the requested extent.

`fusion.height` and `fusion.weights` use Float64. Weight planes follow
`fusion.source_ids` and retain each source's effective contribution through
all the blends. `fusion.source_index` gives the index of the largest
contribution and uses `-1` for voids. Ties follow input priority.
`fusion.dominant_source_id` gives the corresponding source ID or `None`.

Each source reports all-overlap statistics against the accumulated
lower-priority surface. For a bias estimate, pass a boolean mask of known
stable, low-slope target cells in `bias_estimation_masks`, keyed by source ID.
The `bias_estimate` entry reports the eligible sample count, signed median of
higher minus lower heights, median absolute deviation, and selection checksum.
When no mask is supplied, the estimate is null. A supplied mask with no
eligible overlap has zero samples and null median and spread.

An optional `max_bias_slope` in metres of rise per metre of run filters the
selection on both surfaces. A sample then needs four valid cardinal neighbours.
For each axis the larger absolute one-cell slope is used, and the magnitude of
the two axis slopes must meet the limit. Use `calibration_masks` for the same
estimator with correction applied. Fusion subtracts the selected median from
the source's valid heights and records the applied offset. An unresolved result
continues to carry its unresolved datum label after any local bias correction.

For each blend stage, `overlap_statistics[source_id]["blend_steps"]` counts
horizontal and vertical neighbour pairs with valid output, at least one cell
in the overlap, and a fractional or changing source weight. Along with
`pair_count`, it records the absolute maxima `max_total_step_m`,
`max_blend_step_m`, and `max_blend_step_bound_m` in metres. The maxima can come from
different pairs. Fusion accounts for the terrain's own slope by evaluating the
overlap endpoint at the neighbouring cell's weight. It checks each remaining
step against the local source height difference times the weight change, with a
Float64 roundoff allowance. Magnitudes beyond the Float64 range use JSON
integers. A stage with no comparable pairs has no step record.

`TerrainFusion.save` publishes a new directory with these files.

| File | Contents |
| --- | --- |
| `terrain.tif` | Float64 heights in metres, internal validity mask, target georeference, surface tag, and vertical-reference tag |
| `weights.npy` | Float64 contribution planes in recorded priority order |
| `source_index.npy` | Int64 dominant-source index on the same grid, with `-1` for voids |
| `fusion.json` | Versioned receipt with input and auxiliary hashes, registration, conversions, warp settings, overlap statistics, and output checksums |
| `recipe.json` | Standard single-elevation preparation recipe pointing to the fused GeoTIFF |

`TerrainFusion.load(directory)` verifies the receipt against the checksum in
`recipe.json`, then verifies the raster and attribution checksums. It reopens the
original fusion without its DEMs. Use the regular preparation
route for layers, display and rendering. For a fused snapshot with voids, the
usual explicit `terrain_void_fill` applies only when building render assets.
Fusion works on one whole target grid and keeps one weight plane per DEM.

## Declare once, inspect, reuse

Grids use horizontal two-dimensional coordinate systems. Compound/vertical,
geocentric and three-dimensional geographic CRSs require explicit external
preparation; a third axis is never discarded implicitly. Recipe and manifest
JSON serialization rejects nonfinite numbers and canonicalizes numeric negative
zero while preserving literal strings and raw analysis arrays.

```python
import rasterio  # before a checkout's python/ path is added
import forge3d as f3d

grid = f3d.PreparationGrid(
    crs="EPSG:28992", width=100, height=100,
    geotransform=(90000, 100, 0, 440000, 0, -100),
)
recipe = f3d.PreparationRecipe(
    target_grid=grid,
    vertical_reference="NAP",
    sources=(
        f3d.PreparationSource(
            "ahn-dtm", "ahn-dtm.tif", "elevation", units="m",
            vertical_reference="NAP",
            metadata={"edition": "record the actual edition", "attribution": "record the source"},
        ),
        f3d.PreparationSource(
            "population", "population.tif", "count", units="people/cell",
            mask_path="population-disclosure-valid.npy",
            metadata={"edition": "record the actual corrected CBS edition"},
        ),
    ),
    display_styles={
        "population": {"range": [0, 1000], "colors": [[255, 245, 220, 0], [160, 30, 10, 255]]},
    },
)
recipe.save("rotterdam-preparation.json")
prepared = f3d.prepare_scene(recipe)
prepared.save("prepared-rotterdam")  # new folder; existing snapshots are preserved
prepared.coverage_preview("prepared-rotterdam/coverage.html")
reused = f3d.PreparedScene.load("prepared-rotterdam")
```

These paths/edition descriptions are an authoring example, not an acquired AHN
or CBS fixture. Replace them with recorded local inputs. Metadata such as source
dates, licence/attribution, observation time, disclosure meaning and prior
transformations travels through prepared assets and scene metadata.

Add independent checkpoints to the recipe when geographic accuracy matters:

```python
checkpoints = ({
    "source_id": "ahn-dtm",
    "source_pixel": [column, row],  # alternatively source_xy: [x, y]
    "expected_target_xy": [survey_easting, survey_northing],
    "tolerance": source_accuracy_in_target_units,
},)
```

The tolerance belongs to the source/control precision. A translated source
geotransform fails a pixel checkpoint even when array shapes still match. A
coverage footprint or an attractive render alone cannot prove registration.

To render, supply the existing scene camera/lighting/output controls:

```python
scene = reused.to_scene(
    lighting=lighting, output=output, camera=camera,
    terrain_void_fill=declared_fill_metres,  # needed only if terrain has voids
)
scene.render("map.png")
scene.save_bundle("map.bundle")
```

`to_scene()` writes each thematic display as a GeoTIFF on its prepared grid.
MapScene drapes georeferenced raster overlays in terrain UV space, so they stay
registered to the terrain in oblique views, and it projects inline vectors
through the terrain camera in 3D views. For an interactive camera, use prepared
viewer assets and the existing terrain viewer:

Malformed recipe JSON, missing/unknown recipe fields and malformed checkpoint
fields produce a structured `invalid_recipe` diagnostic rather than a traceback.
Recipe loading also validates the display-style keys and source/checkpoint
combination. Use `source_xy` for vector/observation checkpoints; `source_pixel`
requires a raster. Unknown keys such as `sacle` or `colour` are rejected instead
of silently selecting another style or failing later during rendering.

Recipes must contain finite JSON metadata. `NaN`, infinities and non-JSON
metadata fail before preparation outputs are created. Snapshot save writes into
a temporary sibling directory and publishes by rename only after all files are
complete. Failures clean up the staging directory; a corrected retry can use the
same target. An existing destination is protected by an actionable
`output_exists` diagnostic. The CLI validates camera JSON before saving as well.

`to_scene()` keeps the saved snapshot unchanged. Display GeoTIFFs go to the
adjacent `<snapshot-name>.scene-assets/` directory by default. Supply
`assets_dir="my-scene-assets"` for another persistent location outside the
snapshot; retain it while the resulting scene references those rasters.

Measurements can use a reusable nonlinear display without enumerating observed
values: `{"range": [0, 2010], "scale": "log1p", "colors": [[246,226,178,255],
[146,30,56,255]]}`. `linear` remains the default. The fixed endpoints saturate
display colours outside the range; analysis values and masks remain unchanged.
For `log1p`, valid values and endpoints must be greater than -1. Keep distinct
absence/disclosure states in a separate categorical source instead of treating
every masked measurement as the same state.

```python
assets = reused.viewer_assets("prepared-viewer", terrain_void_fill=declared_fill_metres)
with f3d.open_viewer_async(terrain_path=assets["terrain_path"]) as viewer:
    # Draped raster overlays use the viewer's PBR terrain path.
    viewer.send_ipc({"cmd": "set_terrain_pbr", "enabled": True})
    viewer.set_z_scale(1.0)  # physical metre heights on the metric target grid
    for overlay in assets["overlays"]:
        viewer.load_overlay(overlay["source_id"], overlay["path"],
                            extent=overlay["extent"], preserve_colors=True)
    # Apply the map's camera, lighting and height-scale settings with ViewerHandle.
    viewer.set_orbit_camera(phi_deg=25, theta_deg=45, radius=30000)
    viewer.snapshot("terrain-map.png")
```

Prepared render assets require a north-up projected grid with horizontal metre
units, matching the metre heights. Other source grids can warp into that target.
Raster styles are the same
saved mappings; counts are sampled only as display colors. Vector/observation
display requires a fixed `{"color": [r,g,b,a]}` style and uses the shared target
grid for rasterization. Original feature coordinates/properties remain in the
prepared snapshot. `viewer-assets.json` preserves grid, provenance and hashes.
Viewer snapshots follow the existing documented exclusion from offscreen
certificate scope. Choose camera radius and height scale using the viewer's
declared world-unit convention; exporting assets does not certify a camera view.

Raster display mappings use a fixed `range` and two RGBA `colors`, or `classes`
mapping every class ID to an RGBA color. A missing style refuses; it does not
normalize a measurement against each frame's data. The saved recipe retains the
styles, while the prepared values/masks stay unchanged. Byte imagery can display
directly. Arbitrary scientific multi-band imagery needs an explicitly prepared
byte composite for this display route. Existing `MapScene.render` and
`render_flythrough` keep their native frame/failure contracts. Prepared terrain
analysis values remain float64; the existing GPU terrain upload uses float32.

Every failure is a `PreparationError` with a structured `diagnostic` containing
its code, source identity and remediation. Missing CRS/height/units, no coverage,
unsupported datum operations, wrong mask dimensions and conflicting grids can be
fixed through the recipe or source metadata. No internal code edit is needed.

Run the command-line route with:

```text
python examples/prepare_scene.py recipe.json new-prepared-directory
```

`--help` lists optional native still and Landscape settings. For a revision,
change the source path/edition or target grid in the JSON and select a new output
directory. Snapshot asset hashes detect changed/corrupt prepared files on load;
original source hashes and operation metadata survive scene/bundle reuse.
The CLI checks both the snapshot and optional `--landscape` destination before
preparing inputs. An existing destination returns `output_exists` without
writing a snapshot; select a new directory and retry.

## Optional Unreal Landscape package

`export_landscape` accepts an exact, north-up projected terrain grid and an
aligned integer categorical layer. Select a `LandscapeLayout` explicitly; the
required dimensions are `(components * quads_per_component) + 1`. Each component
contains one or four sections, with 7/15/31/63/127/255 quads per section. The rules
and 16-bit height scale follow the [Epic technical guide](https://dev.epicgames.com/documentation/en-us/unreal-engine/landscape-technical-guide-in-unreal-engine).

```python
package = f3d.export_landscape(
    reused, "landscape-package",
    layout=f3d.LandscapeLayout(63, 1, 4, 4),  # requires a 253 x 253 prepared grid
    origin=(origin_easting, origin_northing, origin_height_m),
    categorical_source="landcover",
    material_layers={0: "Water", 1: "Forest", 2: "Rock"},
    void_fill_m=declared_fill_metres,
)
```

The example requires its own aligned landcover source and matching target grid;
the population-only recipe above is not that input. The package contains a
16-bit grayscale height PNG, binary 8-bit named material masks, height/category
validity images, `landscape.json` and step-by-step `IMPORT.md`. It refuses invalid
dimensions, rotated/sheared grids, unknown geographic scales and incomplete
class mappings. No crop, pad, resample or exaggeration happens at export. Source
pixel centres become vertices, so the footprint is the distance between first
and last sample centres, not the full pixel-area extent. One image supplies all
component-edge samples; no separate tile seam is introduced.

For nonconstant heights the step is `(max_height - min_height) / 65535`, with
half a step as the quantization bound. The package records the float64 arithmetic
bound and observed reconstruction error separately. Flat inputs use the engine's
default Z scale and one constant sample value. Heights below sea level and filled
voids are included in the declared encoding range.

The opt-in geographic sidecar declares the projected CRS, vertical datum, metre
height units, geographic anchor, centimetre scale, east/south/up axes and forward
and inverse transforms. The anchor uses easting/northing in source CRS units and
height in the declared datum. Actor positioning must account for component
SectionBase offsets; actor pivot and first vertex are not assumed identical.

Manual editor import is the initial route. Follow the package's exact settings
and the [Epic import guide](https://dev.epicgames.com/documentation/en-us/unreal-engine/importing-and-exporting-landscape-heightmaps-in-unreal-engine),
assign matching material layers, inspect the layer masks, and test edits on a
separate edit layer. Validity images retain missing-data meaning; they do not
automatically create holes. World Partition, tiled landscapes and automatic
Landscape creation remain separate work. The documentation target is Unreal
5.8; no editor build is qualified by package creation.

`verify_landscape_import(package, measured_positions_cm, editor_version=...,
measurement_tolerance_m=...)` compares actual measured editor vertices against
the package. Supply a complete `(height, width, 3)` array in image row/column
order and a measurement-derived tolerance. This detects shifted/scaled/flipped
imports and Z errors. It proves geometry only; material assignment, editability,
silhouette/banding/cracks, class edges, material scale and moving-camera LOD
inspection still require editor evidence. No competitive advantage is claimed
without a matched workload in the comparison tool.

The local C05 acceptance on 2026-10-19 exercised this manual route in Unreal
5.8.3 (`5.8.3-58210709+++UE5+Release-5.8`), D3D12 SM5 on an RTX 3070.
Grindelwald initial and revised imports retained every encoded height and named
class-mask sample. Native component transforms matched the package coordinates;
sculpt and paint on a separate artistic layer were undone to byte-identical
measured data. The evidence, saved project, complete camera flights and copied
delivery inputs were recorded in a local, git-ignored acceptance directory
(`artifacts/c05-unreal-acceptance-20261019/`); they are not part of the repository.
That is validation of those recorded inputs and settings, not automatic
qualification of another package or Editor version.

For this observed Editor route, create non-weight-blended Layer Info assets for
the exact material names before import. Apply the recorded actor transform after
checking the component SectionBase values. Rename the imported edit layer
`Measured_DEM`, lock it, then create an unlocked standard `Artistic_Validation`
layer for edits. Keep the original package immutable and save the Unreal project
separately. Switch back to Selection mode before changing levels: changing an
empty level while Landscape mode retained an edit-layer selection triggered an
Editor assertion during this validation.

Native `Landscape.render_heightmap` / `render_weightmaps` readback in this build
uses `RTF_RGBA8` and explicit pixel-area bounds half a grid spacing outside the
first and last native vertices. Automatic bounds resample between vertices at
equal output dimensions. The height readback was qualified against a separate
native UI uint16 PNG export before use for editing checks. Allow native lighting
to settle before recording a flight; the revised acceptance waited for two
consecutive identical captures at its fixed starting pose. These are validation
details, not a claim of automated Landscape creation or cross-backend pixel
identity.
