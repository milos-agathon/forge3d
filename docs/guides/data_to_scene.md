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
