# Monterey Bay measured-section acceptance

Implemented and exercised locally on branch `codex/map-measured-sections`
from `5de25854a75f52977367df94787aee5a4ad1bf68`. GitHub's `refs/heads/main`
and local `origin/main` both resolved to this revision on 14 November 2026.
The validation used a release wheel assembled in the isolated worktree
from those native sources and the measured-section Python changes.
The final local wheel SHA-256 is
`288a4cbd7cbe21fccd29322e410c69654bb7dabde798c9e69bdfad2e1df1e6f2`.
The source and evidence artifacts are local to this worktree.

## Source

NOAA NCEI ETOPO 2022 **v1, 15 arc-second surface** tile `N45W135`.
The subset starts at column 2952, row 1904, and is 312 × 208 cells. It was
copied without interpolation using rasterio 1.5.0 and GDAL 3.12.1. The
pixel-edge affine is `(1/240, 0, -122.7, 0, -1/240, 37.06666666666666)`
with bounds approximately 122.7° W, 36.2° N, 121.4° W, 37.066667° N.
In the demonstration, cells span approximately 373 m east–west and
462 m north–south. The raster supports regional terrain and bathymetry
explanation. Use the ETOPO model's sign boundary as approximate land/water
context. Survey or tidal-shoreline use requires other data.
[NOAA product and citation](https://www.ncei.noaa.gov/products/etopo-global-relief-model)
· [NOAA user guide, sections 2 and 5](https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/docs/1.2%20ETOPO%202022%20User%20Guide.pdf)

| Layer | Upstream SHA-256 | Local subset SHA-256 |
| --- | --- | --- |
| Surface | `4ff26df2ad67546a8c6d94e54c31f7080c80f87f1bf16f2f07e64624ec373432` | `0006ef25d8e5dbdd457279af86360462b30f6436170d36de89828910b71ac017` |
| Surface source ID | `996f9d17a36fd9482dca09e75c6e77ccbf78580d2765f830a7f683f73975d0b2` | `5c4bc1ce76d2012199cba307b0abd8c787537e532d765e57ee0585d3b93fa6e5` |
| Geoid | `3c434a66cfb169bc6241db494634298d8e17ef202275752d5e7cb75c825d4470` | `d9ce7c47e0e72bc30c8759033ea96e012bb45a824231a862c4d95f969593916f` |

The area-registered elevation TIFF declares EPSG:9518, horizontal WGS 84
EPSG:4326 and EGM2008 height EPSG:3855. It stores signed Float32 metres and
declares nodata `-99999`. The area-registered source-ID TIFF declares
EPSG:4326 and nodata `0`. Under NOAA's source-ID definitions, the subset
contains 411 cells from NOAA Regional DEMs (ID 4), 39,290 from GMRT v4.0
(5), one from Copernicus DEM 30m (9) and 25,194 from FABDEM (10). The
geoid companion is point-registered in EPSG:4326 with no TIFF nodata tag,
and ranges from about −37.89 to −32.05 m over the subset. Its provenance
tag identifies the EGM2008 transformation grid derived from
`egm08_25.gtx`. It was retained for provenance, with no vertical offsets
applied. The exact URLs, TIFF tags, grid geometry, cell payload hashes
and counts are in the run's `subset/source.json`.
[NOAA source IDs and geoid conventions](https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/docs/1.2%20ETOPO%202022%20User%20Guide.pdf)

## Original and revision

A starts at `(-121.74, 36.38)`. One recipe field changed, the offshore
endpoint `end`, from `(-122.37, 36.65)` to `(-122.39, 36.75)`. The
declared maximum spacing stays 375 m. The same change was used by both
programs. A fresh invocation per recipe regenerated the CSV, map trace,
profile, and all linked coordinates. There were zero manual changes to
the trace, chart, or annotations. Recipes and tables for both cases are
retained. Each tool required one coordinate-pair field edit and one
workflow run. The follow-up below measures the authoring and invocation
time for this revision in both scripted routes.

| Case | Length | Samples | Actual interval | A height | B distance / height | C height |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 63.89194 km | 172 | 373.637 m | +744.65 m | 32.133 km / −948.00 m | −1,915.15 m |
| Revised | 71.21325 km | 191 | 374.807 m | +744.65 m | 35.607 km / −941.67 m | −2,024.88 m |

QGIS 3.44.0 independently used `QgsDistanceArea` with the EPSG:7030 WGS 84
ellipsoid and the QGIS raster provider. The matched rule selects the
containing cell with the documented binary64 affine and samples its
centre. Direct provider sampling at the original endpoint, exactly on
the documented row edge, returned −1,820.8683 m from the adjoining row.
The shared edge rule selected row 100, column 79 and −1,915.1512 m.
The requested endpoint is identical in both tables, and the direct
provider result is retained in the QGIS CSV and comparison receipt.

All 363 matched samples agree exactly in elevation and validity. The
maximum differences across both runs were 1.46e−11 m in cumulative
distance and 8.53e−14° in coordinates. The tolerances are the existing
MENSURA GeodTest contracts (1e−8 m for length, 1e−9° for position).
The direct position tolerance is far smaller than a 15-arcsecond cell.
Containing-cell sampling requires exact Float32 height agreement for
the selected cell. These tolerances measure agreement and arithmetic.
Source model accuracy is a separate matter. Analytic controls use the
WGS 84 equator formula and a known raster with negative and zero
elevation, nodata, nonfinite values, reversed direction, endpoints
and raster edges. Both Monterey cases are fully valid. The analytic
plate verifies the visibly broken profile and the missing-value table.
The QGIS provider separately passed the analytic grid in both directions
and five exact-edge controls, including a nonfinite cell. Both rendered
analytic plates and the QGIS results are in the local artifact root.

## Registration and readability

The independent verifier checked each of the 584,064 native map pixels
per case against the corresponding raw ETOPO class, using the 3 × 3
pixel footprint of each source cell. All matched. QGIS agreed at all
64,896 source-cell centres and used an adjacent displayed cell color
within one output pixel at the differing edges. The QGIS and Forge3D
classified frames differ at 10,313 pixels per case. The verifier
checked both rendered traces at each sample position and both sets of
A/B/C markers at independently projected map and profile coordinates,
using one output pixel for drawing quantization.

Inspected both 1500 × 1130 plates from both tools at full image size.
In each, the red letter and dot identify the same sample in the two
views. A lies on land, B lies on the submarine slope, and C moves
offshore on the revision. The profile uses 0–80 km and −4,000 to +2,000 m
with EGM2008 named. Its exact display ratio is 1.69697× and the displayed
ratio is 1.7×. Missing samples have an explicit gray key. The filled
coast band is derived from the signed source values.

## Executed comparison

Windows, AMD Ryzen 9 5900X, about 48 GiB RAM, NVIDIA RTX 3070 with
eight GiB VRAM, driver 610.60. Forge3D ran with CPython 3.13.14 and Vulkan.
QGIS used its extracted 3.44.0 Solothurn runtime, GDAL 3.11.0, and
Qt's offscreen 2D painter. Both used the same subset, bins and output
dimensions. Each result includes load/sampling and render/export
measurements with peak process working set. The peak includes the
Python process and loaded libraries. VRAM was not measured. The QGIS
setup phase includes app initialization, font and raster provider.
Forge3D sampling includes opening and validating the GeoTIFF. The
Forge3D render phase includes native frame generation, composed labels,
hashes and export. QGIS's render phase includes producing its classified
display layer, map, Qt plate, and PNGs.

| Tool and case | QGIS setup (s) | Sample (s) | Render and export (s) | Peak process (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Forge3D original | — | 0.204 | 3.381 | 295.2 |
| Forge3D revised | — | 0.184 | 3.273 | 295.5 |
| QGIS original | 1.793 | 0.037 | 0.148 | 169.5 |
| QGIS revised | 1.428 | 0.037 | 0.132 | 169.0 |

Setup used a pinned-subset preparation command, a Python recipe adapter
for Forge3D, and a separate PyQGIS script for the comparison. Both were
script-authored in the same session. QGIS automation setup encountered an
offscreen font failure, corrected by loading the repository's Latin
font, and a raster edge convention difference as described above.
The viewport registration probe at QGIS source size shifted some
colours. The comparison uses its native output-size viewport and retains
the measured display differences. These are single local observations
for different rendering stacks.

### Authoring effort and operator experience

The review found that the original receipt lacked operator experience,
timed authoring effort and a preference assessment. In a fresh follow-up
trial, the same Codex coding agent set up both scripted routes in empty
output folders. The operator had previously developed and run both
routes, including the Forge3D recipe and test fixture, and the PyQGIS
sampling, map and Qt export script. This is an experienced repeat
operator. The unit of work was starting a publication from the prepared
subset with the existing workflow code and runtimes.

For Forge3D, setup involved writing the original JSON, a PowerShell
launcher and a small pytest adapter invoking the installed CLI with the
existing Pro fixture. For QGIS, it involved writing the same original
recipe and a PowerShell launcher invoking the existing standalone
PyQGIS workload. Each setup finished with one successful original
export. Each revision edited only the endpoint field and reran the
launcher. The timed follow-up succeeded on the first invocation for each
case. There were no manual trace, chart or annotation edits.

| Scripted route | Original authoring/setup (wall seconds) | Setup through original export (wall seconds) | Endpoint edit (wall seconds) | Edit through revised export (wall seconds) |
| --- | ---: | ---: | ---: | ---: |
| Forge3D | 25.23 | 30.59 | 0.11 | 4.89 |
| PyQGIS | 20.51 | 28.09 | 0.11 | 4.13 |

Authoring/setup time runs from starting the trial to having the recipe,
adapter where applicable, and launcher ready. It includes the agent's
authoring latency. End-to-end time includes shell and process startup and
export. Both endpoint edits were scripted substitutions. Original
implementation and troubleshooting time was not measured retrospectively.
The values reflect reuse of the existing scripts and prepared raster on
this machine. Human GUI setup, first-time onboarding and sustained
performance remain unmeasured. All 18 checked outputs across the four
follow-up runs matched the qualified originals byte-for-byte.

### Visual preference

The same operator compared both original and both revised plates at
1500 × 1130, with tool identities known. Both correctly showed the
source terrain and let the reader match A, B and C across views. For this
plate, the operator preferred QGIS's smoother diagonal trace on both
maps and Forge3D's stronger zero-height profile line and placement of
the missing-data key beside the top of the elevation legend. Neither
version had a decisive overall preference. This single non-blind
assessment is separate from the numerical and readability checks.
QGIS's interactive elevation-profile panel and motion comparisons remain
unrun.

Trial scripts, original and revised recipes, timestamped operations and
the machine-readable effort receipt are under
`artifacts/c08-measured-sections/authoring-review/`. The trial completes
the comparative evidence for the scripted terrain still.

## Validation and remaining scope

Commands are in [the measured-section guide](measured_sections.md). Tests
ran against the isolated installed wheel with `FORGE3D_NO_BOOTSTRAP=1`
and `FORGE3D_TEST_INSTALLED_WHEEL=1`. Both original and revised recipes
ran in fresh processes with the existing Pro fixture. The direct demo CLI
in the unlicensed environment reported the Pro license requirement.

- `pytest tests/test_measured_sections.py tests/test_geodesic_karney.py tests/test_terrain_scatter.py tests/test_map_plate_layout.py`
  ran 102 passes with the existing interactive viewer scatter case skipped.
- `pytest tests/test_publication.py tests/test_publication_gpu.py tests/test_value_faithful_raster.py tests/test_render_certificate_contract.py`
  ran 103 passes. The existing XML zero-skip gate passed.
- `pytest tests/test_measured_sections.py tests/test_render_certificate_contract.py`
  ran 29 passes after registering the new entry point's certificate and
  cache controls. The existing XML zero-skip gate passed. This includes
  an analytic native render that reuses only the unchanged base map
  after an endpoint change, regenerating the linked outputs.
- Physical test runs used `FORGE3D_RUN_GPU_ANAMNESIS=1`,
  `WGPU_BACKEND=vulkan`, and `WGPU_BACKENDS=vulkan`. The first sampling
  test run exposed an incorrect exception type for an invalid CRS
  declaration. The corrected run passed. The new entry point's initial
  certificate contract check identified the missing cache keyword.
  The final contract check passed with both controls.
- A fresh preparation with the pinned TIFFs reproduced all three subset
  TIFFs and the source receipt byte-for-byte. Chromium 148 loaded the
  roadmap and all four C08 deep links without page errors.
- Full CI, other operating systems/GPU backends, a new signed
  demonstration certificate, and the opt-in interactive viewer scatter
  test were unrun. Broader C08 cursor and flyover motion, sequence
  occlusion, and sections tied to an actual source volume still need
  their own acceptance. Uncertainty propagation remains under DUBIUM.

Local artifact root is `artifacts/c08-measured-sections`. It retains the
four PNG/CSV runs, the three NOAA subsets, the original and revised
recipes, `comparison.json`, `qgis-analytic.json`, `roadmap-browser.json`,
local logs and JUnit XML. Goldens, existing
certificates and the original working tree were preserved.
