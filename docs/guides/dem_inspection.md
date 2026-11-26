# Inspect a DEM before rendering

`forge3d.dem_audit.audit` reads original elevations and validity. It returns an inspection sheet and JSON-compatible findings. Open the TIFF directly to include its band mask, nodata, scale, offset, transform and CRS. A loaded display raster may already have been filled or normalized.

```python
from forge3d import dem_audit

report = dem_audit.audit("terrain.tif", band=1)
report.save("terrain-inspection")     # terrain-inspection.json and .png
for finding in report.findings:
    print(finding)
```

For arrays, pass an Affine or six GDAL-order coefficients, plus a boolean `valid_mask` or a `nodata` value if applicable. Masked arrays and nonfinite values are supported. Without a transform, coordinates and area use pixel units. The audit uses Rasterio for connected-region polygons and file input, and Pillow for the inspection sheet.

Read the sheet alongside the JSON. Each panel shows the same raster extent. A point's `cell` is zero-based column, row and `xy` is its world-space centre. Rectangular `bounds` exclude the right and bottom edge. Stripe and quantization findings retain the full sampling extent in `analysis_bounds`. Their `support_runs` locate the evidence on the source grid as half-open `[row, first_column, end_column]` triples. The sheet tints those cells. Review the support before passing `{"runs": finding["support_runs"]}` as a repair region. Both classes keep `repair_region` null until a region is explicitly selected. Polygon `rings` use pixel edges, with the exterior first and any holes after it. The report's GDAL transform maps those edges into its CRS. Areas are in squared CRS units. A geographic CRS gives square degrees. Heights use the input band's elevation units.

## Detector defaults and resolution

The deterministic 96 × 96 fixtures in `tests/_extrema_fixture.py` contain a smooth sloping analytic surface, seeded fractal relief, and one damaged copy of that relief. The damaged copy combines quarter-unit rounding, a row and column hatch, a tile step, two isolated excursions, a hole and a constant patch. The two terrain controls yield **zero findings**. The following measurements guided selection inside the clean/defective separation. They describe these fixtures and do not establish accuracy on other terrain.

| Quantity | Evidence behind the default |
| --- | --- |
| Spike and pit | Opposing-neighbor pairs estimate the local surface. At a raster corner, the three inward samples define a one-sided plane. Divide the signed residual by median absolute neighbor departure, floored by typical spread across the terrain. With coherent hatch removed, the largest clean score was 0.551 and the weaker injected excursion scored 121.4. The default multiple of 5 leaves room for less extreme excursions. The signed height is always reported. |
| Directional periodicity | The strongest repeating clean bin carried 0.782 of the non-DC power of the median cross-line difference profile. The weaker injected direction carried 0.896. Their midpoint rounded to two decimals gives 0.84. Period resolution is one Fourier bin, with at least two cycles in the analysis extent. For each detected frequency, individual transects are checked in two-period windows. After local slope removal, the measured harmonic must explain at least the same power fraction in each reported window. Incomplete or weak windows may be absent from the support. If the peak is visible only after pooling, the detector subdivides transect groups and reports the smallest tested groups that retain the peak. The `support_basis` field identifies which measurement supports the location. Variable-phase or broadband striping can escape detection. |
| Seam | The strongest clean slope-corrected jump was 3.80 times the typical cross-line excess. The injected line's median was 57.8 times that scale. The default multiple is 12. Nine contiguous cells distinguish the injected line from the eight-cell edges of the deliberate flat patch. The signed step removes the adjacent slopes. Local peak selection compares these slope-corrected jumps for both step signs. |
| Quantization | Occupied histogram levels estimate the step, using repeated levels when available. The best candidate on either clean control explained 0.022% of heights. The defective quarter-unit lattice explained 99.30%. A 90% default permits isolated exceptions. The report gives the fraction on the lattice and marks finite cells within the reported tolerance. A naturally coincident height can appear in the support. |
| Flat fill | Neither clean control contained adjacent equal cells. Both the 4 × 4 GeoTIFF fixture and the 8 × 8 mixed-fixture patch contain exactly constant connected cells. The smaller measured area supplies the default of 16 cells. |
| Void | Every four-connected component of invalid cells is reported, including single cells and components touching the raster edge. Area is cell count times the absolute affine determinant. |

Invalid detector settings raise `dem_audit.AuditParameterError`. Its `diagnostic` provides the error code, parameter name and remediation. Accepted NumPy numeric settings are normalized to JSON numbers.

In the 96 × 96 localization fixtures, moving a four-cell-period stripe from the first 64 columns to the last 64 moved its entire 6,144-cell support. Moving quarter-unit rounding between the first and last 88 columns moved all 8,448 aligned cells. Transposing each input gave the corresponding row locations. Applying the confirmed runs preserved every pristine cell in these fixtures.

A single global spectrum or height histogram describes the supplied extent. Use georeferenced native-resolution windows to inspect a local patch. Window boundaries can divide a feature and change its score. Findings are candidates for visual inspection. A flat lake or cliff can produce a legitimate response.

## Declare a repair

Each public repair takes a region on the source grid and returns **both** a new masked raster and its change record. Supply one of `{"runs": [[row, first_column, end_column], ...]}`, `{"cells": [[column, row], ...]}`, `{"bounds": [left, top, right, bottom]}`, or `{"rings": finding["rings"]}`. The original stays intact. Nearest fill uses SciPy's Euclidean nearest donor. Spike removal uses opposing-neighbor interpolation. The notch filter fits and subtracts specified sinusoidal components with a separate trend term. Dequantization moves floating-point elevations towards their interpolated surface by at most half the declared rounding step.

To replay a decision, include the same arguments in the elevation source's ordered `repairs` list. For example, after confirming a void and a spike in a report:

```python
from dataclasses import replace
from forge3d.preparation import PreparationRecipe, prepare_scene

void = next(f for f in report.findings if f["class"] == "void")
spike = next(f for f in report.findings if f["class"] == "spike")
recipe = PreparationRecipe.load("terrain-recipe.json")
sources = [
    replace(source, repairs=[
        {"operation": "fill_missing", "method": "nearest", "region": {"rings": void["rings"]}},
        {"operation": "remove_spikes", "region": {"cells": [spike["cell"]]}},
    ]) if source.kind == "elevation" else source
    for source in recipe.sources
]
declared = replace(recipe, sources=sources)
declared.save("reviewed-recipe.json")
prepared = prepare_scene(PreparationRecipe.load("reviewed-recipe.json"))
```

For a confirmed stripe, pass `axis` (`row` or `column`) and `frequencies` in cycles per source cell to `notch_filter`. For a confirmed lattice, pass its height `step` to `dequantize`. Give either operation a separately confirmed region. All repairs run on decoded source elevations before unit conversion and warping. Direct and recipe repairs preserve the decoded storage dtype. Integer spike replacements round to the nearest representable integer, with ties to even. The interpolation workspace uses binary64. Declared repairs reject integer source heights outside its consecutive integer range [−2⁵³, 2⁵³] before conversion. Integer changes have exact differences, totals and maxima. A whole-number mean stays an integer. For a fractional mean the report also gives its exact numerator (`total_change`) and denominator (`measured_changes`). Dequantization requires floating-point heights. Unselected data and explicit mask bits are preserved. A separate disclosure mask remains authoritative. An empty repair list retains the existing recipe encoding and GDAL band-mask precedence. An elevation source with declared repairs also excludes its file nodata before scale and offset decoding. Other layers keep their existing validity rules.

Supported NumPy scalars and coordinate or frequency arrays are converted to ordinary JSON numbers and lists before recording, hashing and execution. Invalid repair parameters raise `PreparationError` with an `invalid_repair` diagnostic. Preparation takes a private copy of the ordered repair declarations before execution. Layer metadata contains the SHA-256 of that copy, and the saved recipe uses the same declarations even if the caller subsequently edits the original region or frequency list. The operations list records parameters, affected cells as half-open `[row, first_column, end_column]` runs, and mean and maximum absolute elevation differences over changed, previously valid cells. For newly filled holes there is no observed pre-fill height. Both changes are JSON null, alongside the newly valid count and the minimum, maximum and mean assigned heights. A step that changes zero cells records zero change.

Milos approved one compatibility exception. The existing public `forge3d.io.fill_nodata` helper continues to return a bare array. Use `dem_audit.fill_missing` when the fill needs a change record.

`load_dem` retains its default nearest fill for a declared nodata value. Its returned `fill_record` includes the request, effective method and number actually filled, including the legacy mean fallback when SciPy is unavailable.


## Production inspection, 26 November 2026

Milos confirmed Lagos GEM-15 bare-ground band 3 and Tracy Arm merged 5 m topobathymetry as the two trial inputs. Both audits used original measurements. Coordinates below use the source CRS. All counts are candidates for inspection.

| Source | Native coverage | Findings |
| --- | --- | --- |
| Lagos GEM-15, bare-ground band 3 of the raw film extract, EPSG:32631 | 3,097 × 2,029 cells, 4,497,021 valid | 2,318 voids, 190 spikes, 655 pits and 40 seams. The largest void covers 984,034 cells or 221.41 km² and touches the edge. The production minimum of −245.11 m lies within a depression containing several neighboring samples below −230 m. |
| Tracy Arm merged 5 m topobathymetry, EPSG:26908 | 11,898 × 13,284 cells, all 158,053,032 valid | 162,692 seam segments, 53,366 flat regions, 435 spikes and 5,541 pits. The longest seam is 747 cells at northing 6,410,360 m with a +68.45 m step. The default spectral test did not flag the documented 15–25 m hatch. |

The Lagos audit ran over the entire extract in one pass. Tracy Arm used 28 independent native-resolution windows, each up to the size of the Lagos extract. All source cells were exercised. Window edges can divide regions or interrupt lines, and natural cliffs and terraces can generate candidates. Neither report classified quantization at defaults.

To identify the inputs, the Lagos fetch in the owner's `examples/gem15_lagos/gem15_lagos_film.py` uses Google GEM-15 v1 at the documented Lagos bounds. The raw three-band extract had SHA-256 `51f103028c2202037f85bc1576f051f4aad55a74d5e5de832d79366bcf2f57bc`. The Tracy Arm film cites Shugar et al., Zenodo record 18306727. Its file `TracyArmMerged_new_water_shoreline_adj.tif` had SHA-256 `48c5d286959bfe29565858daf1264e01e7d51c306b9e1eca78f8615296343325`.
