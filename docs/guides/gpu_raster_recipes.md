# GPU raster recipes

`RasterRecipe` takes two single-band float32 GeoTIFFs and a uint8 validity mask
on one C05 `PreparationGrid`. The mask contains 1 for analysis and 0 for
invalid coverage. Supply decoded values in the same units. GeoTIFF scale and
offset tags must be 1 and 0. Set `RasterRecipeSource.observation` to a C04
`ObservationRecord` to retain acquisition time and product metadata.

```python
from forge3d import RasterRecipe, RasterRecipeSource
from forge3d.preparation import PreparationGrid

grid = PreparationGrid("EPSG:32635", 1800, 1800, (570000, 10, 0, 3998000, 0, -10))
recipe = RasterRecipe(
    RasterRecipeSource.from_file("august-ndvi.tif", "August NDVI"),
    RasterRecipeSource.from_file("july-ndvi.tif", "July NDVI"),
    RasterRecipeSource.from_file("valid.tif", "Joint clear-land mask"),
    grid, [-0.5, 0.0, 0.5],
    [(150, 47, 36, 255), (230, 151, 87, 255), (96, 171, 146, 255), (27, 97, 101, 255)],
    quantity="Normalized difference of NDVI",
)
```

Pass the recipe to `scene.render("map.png", raster_recipe=recipe)`. The scene
uses the existing D02 native terrain colour path and must have the same
registered square ground extent. The terrain's affine and dimensions provide
the ground span for both annotation compilation and drawing. Redundant
resolution metadata, if supplied, must agree with that span. Use a one-shot PNG terrain scene with the
thematic slot free. The recipe is serialized independently of the scene
bundle, using `to_json()` and `RasterRecipe.from_json()`; the JSON references
local files and pins their SHA-256 digests. The path-free recipe identity also
includes grid, operator, closure, palette, and source observation records.
The recipe's render route currently excludes scene caching, separately signed
certificates, provenance emission, denoising and the shared screen-effects
pipeline. Ordinary MapScene operations continue to use their existing gates.

Call `set_bins()` to revise thresholds within the palette, or `replace_inputs()`
to select another registered pair and mask. Preview and export call the same
render method. `legend()` returns the D02 swatches with the actual intervals and
acquisition times. `sample(row, column)` reads back 8 bytes, and `counts()` reads
one uint32 counter per class plus invalid coverage. Use `export_cells()` when
the values and class raster themselves are the final output, and `close()`
to release GPU allocations. Loading a serialized recipe verifies source
content at the next upload. A physical GPU and matching native build are
required.

## Numeric contract

The operator is `(a-b)/(a+b)`. Each operation rounds to binary32 using
round-to-nearest, ties to even. Export retains signed zero. For example, equal
negative operands yield negative zero, and the classifier treats either
zero sign as the same value. The supported operands are ±0 and finite
magnitudes from 2⁻¹⁰³ to half the largest finite binary32 value, inclusive.
Below the minimum, operands that nearly cancel can produce subnormal sums or
differences that WGSL allows an implementation to flush. The upper bound
keeps both pre-division operations finite. Values outside the range, declared
nodata, mask 0, nonfinite inputs, a zero denominator, and quotient overflow are
invalid. Invalid values export as NaN with class 0. Thresholds and finite nodata
must be exactly representable in f32. To author a decimal whose nearest
binary32 value is intended, pass `float(np.float32(decimal))` explicitly.

Classes start at 1. Default intervals are lower-inclusive with unbounded
tails, matching `gis.classify_raster(right=False)`. With `right=True`, an exact
edge belongs to the class below it. The shader divides binary32 significands
using integer arithmetic because WGSL permits floating-point division error
large enough to change a class at an exact boundary.

## Rhodes workload

Source IDs, product XML, raw asset URLs and SHA-256 digests are under
`examples/rhodes_raster_sources.json`. The example selects 18 July,
7 August and 17 August 2023 Sentinel-2 L2A products on tile 35SNV, baseline
05.09. It uses B08 (NIR) and B04 (red) for each date's NDVI. The raw SAFE
products declare −1000 DN and a quantification value of 10,000. The pinned
Earth Search COGs mark the BOA offset applied, so preparation multiplies by
0.0001 without applying the offset again. DN 0 is nodata. SCL 4 and 5 qualify
as clear land. SCL is sampled nearest from 20 m, and the 10 m bands are
windowed on their own grid. The DEM uses bilinear reprojection to that same
grid and retains EGM2008 orthometric metres.

The compiled quantity is the normalized difference of two NDVI observations,
not the ordinary subtraction of NDVI. It describes spectral change and carries
no inference of fire severity. The inset is a north-up reference for the same
grid, bins and mask.

Run from the repository root using matching native bindings, real rasterio
and Pillow. The preparation command accepts `--fetch` to acquire the pinned
public assets.

```powershell
$env:WGPU_BACKEND = 'vulkan'
python examples/rhodes_raster_prepare.py artifacts/gpu-raster-recipes --fetch
python examples/rhodes_raster_recipe.py artifacts/gpu-raster-recipes gpu
python examples/rhodes_raster_recipe.py artifacts/gpu-raster-recipes publication
python examples/rhodes_raster_recipe.py artifacts/gpu-raster-recipes publication-revised
python examples/rhodes_raster_motion.py artifacts/gpu-raster-recipes
```

The same render driver accepts `cpu` with the original exact-main wheel to
measure NumPy arithmetic, native CPU classification and D02 rendering.
Use QGIS's OSGeo environment as described in
[the C05 comparison](c05_real_data.md#qgis-comparison). In that interpreter,
run `python examples/rhodes_raster_qgis.py artifacts/gpu-raster-recipes`.
The QGIS route writes editable projects and 1024 × 768 plan renders of both
observation pairs and threshold revisions. Stage the two QGIS 3D cases with
`python examples/rhodes_raster_qgis_stage.py artifacts/gpu-raster-recipes`
after the GPU and QGIS raster runs. The local 3D capture reuses the C05
harness with the Rhodes camera heading, viewport, and RGBA mask adaptations
retained in `artifacts/gpu-raster-recipes/qgis_capture.py`. Its launcher,
exit receipts and inputs are retained beside it. Compose and check the
matching plates and movies with:

```powershell
python examples/rhodes_raster_publication.py artifacts/gpu-raster-recipes --camera-directory matched-publication
```

## Local validation, 14 November 2026

Base `f80dba3d613af33290689e49b9de18b167e49988` (verified remote main).
Windows, CPython 3.13.14, RTX 3070, driver 610.60, wgpu 0.19.4 Vulkan.
The table reports wall time in isolated CPU and GPU recipe runs using the
same 1800 × 1800 inputs, 1024 × 768 native frame, palette and six edits.
Timing includes PNG export. The first CPU baseline was captured before the
implementation, then the same script was run on its retained exact-main
wheel with file load included on both routes.

| Metric | CPU analysis + D02 | Resident WGSL + D02 |
| --- | ---: | ---: |
| First result including PNG | 9.28 s | 2.72 s |
| Four warm threshold edits, median | 4.95 s | 0.42 s |
| Replace observation and mask | 5.11 s | 0.60 s |
| Initial analysis/colour upload payload | 12,960,000 bytes | 29,160,000 input bytes + 88 parameter bytes |
| Each threshold edit | 12,960,000 upload bytes | 64 upload bytes |
| Per-frame beauty readback | 3,145,728 bytes | 3,145,728 bytes |
| Peak process working set | 485.2 MB | 367.0 MB |
| Peak Forge3D tracked allocation total | 235.5 MB | 290.6 MB |

The CPU upload call enqueued the first colour texture in 28 ms. The GPU
upload and completed normalization took 15 ms, then classification took
1.8 ms. The 88 initial parameter bytes include 24 bytes of normalization
settings and 64 bytes for classification. Across two input uploads and six
classifications, the cumulative parameter transfer was 432 bytes. These scopes differ in what they wait for. The tracked peak includes
3.15 MB of host-visible allocation in both runs, and does not measure driver
allocations. Each GPU analysis inspection read 8 bytes for one sample and
24 for six counts. Neither preview path read back analysis arrays. The
explicit values/classes GeoTIFF export did read back 25.92 MB on demand on
the GPU route, and the CPU route already had the results locally. The CPU
reference, write and reopen took 1.54 s and 1.17 s, respectively. The resident
readback, write and reopen took 0.82 s and 0.86 s, before the independent
reference comparison.

All six Forge3D PNGs and class histograms matched the CPU baseline. Each
original and revised publication exported 12 native frames and 3.24 million
matching raster values and IDs. With the fixed recipe, each sequence executed
one normalization and one classification dispatch, with zero analysis
readback during rendering. Native albedo captures showed all five classes
with identical class colours throughout each sequence. The smallest
class occupied at least 92 pixels in every original frame and 311 pixels in
every revised frame. There were 116 original and 83 revised registered
mask-edge pairs that remained valid across all camera positions. The
diagnostic draws matched the finished beauty frames pixel for pixel.
Both sequences had a minimum 52 px margin for visible thematic terrain.

QGIS 3.44.0 / GDAL 3.11.0 saved editable projects, rendered the six matching
plan revisions and exported both 12-frame native OpenGL sequences on the
same RTX 3070. The 3D run reused the C05 native capture harness with a
1024 × 768 viewport, the same DEM, the QGIS-calculated classes and geographic
target, 21 km radius, 58° field of view, and heading equal to Forge3D azimuth
plus 90°. QGIS retained the base DEM under invalid cells and used its own
terrain lighting. Both QGIS sessions exited successfully. The local
calculator/plan runs took 2.62–3.13 s each, and native camera capture took
3.83 s and 3.59 s. QGIS process wall time including startup, capture and
shutdown was 64.26 s and 22.11 s. The large variation limits timing
comparisons. One cell, row 894 / column 1430, landed
on QGIS's −0.5 edge after its double-precision calculation was stored as
float32. Forge3D's per-operation binary32 rule gave −0.5000000596046448,
which belongs in the lower bin. Both renderers produced matching 1340 × 870 plates and 12-frame movies
using the same title, observation dates, inset size, and class swatches.
The matched receipt records preparation, scripted authoring, revision,
startup, native frame export, composition, and encoding costs. The final
runs involved no manual repair. Earlier setup work included correcting
relative QGIS project paths and calibrating its heading. One-off script
development and troubleshooting time were not timed.

The affected Python selection passed 394 tests with freshly built bindings
on CPython 3.13.14, `FORGE3D_NO_BOOTSTRAP=1`,
`FORGE3D_RUN_GPU_ANAMNESIS=1` and `WGPU_BACKEND=vulkan`. It included
world annotation registration without redundant resolution, signed-zero
bit patterns, upload accounting, both bin closures, source rejection, and
resource release. `cargo forge3d-clippy` passed. Both WGSL entry points ran
on the physical adapter. Hosted CI, additional Forge3D backends, forced
device loss and OOM, and real-time movie playback were unrun.
