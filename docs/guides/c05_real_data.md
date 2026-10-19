# C05 real-data demonstrations

The examples prepare Rotterdam ground heights and published population counts,
and Grindelwald/Eiger heights and measured land cover. They save executable
recipes, geographic checks, native analysis arrays, reusable viewer assets,
stills and complete camera sequences. The Swiss scenes also produce Landscape
packages. The manual Unreal Editor import of those packages is described in the
"Optional Unreal Landscape package" section of the [data-to-scene guide](data_to_scene.md).

The commands below write sources, results and the comparison report into a
local working directory, `artifacts/c05-real-data`, which git ignores. Large
source files, runtime tools and rendered products stay outside version control.

## Recorded sources and meaning

| Scene | Inputs | Coordinates and heights | Initial / revised target |
| --- | --- | --- | --- |
| Rotterdam | AHN4 DTM, official 10 m WCS subset; corrected CBS 2024 v2 | RD / EPSG:28992; NAP metres | 1200 × 1200 / 1100 × 1100, 10 m |
| Grindelwald/Eiger | 91 swissALTI3D 2025 2 m tiles; swissTLM3D 2026-02 land-cover footprints | LV95 / EPSG:2056; LN02 metres | 1009 × 1009 / 883 × 883, 8 m |

Rotterdam's initial bounds are E 88000–100000, N 430000–442000. The revision
insets every edge by 500 m. Swiss bounds are E 2640000–2648072,
N 1156928–1165000; the revision insets every edge by 504 m. Both revisions keep
the same source editions, datum and display domains. The Swiss dimensions obey
Landscape's documented 63-quads-per-component rule: 16 × 16 components initially,
14 × 14 after revision. Export does not add padding or resample again.

The AHN service's 10 m subset comes from the 0.5 m Squared-IDW ground product.
Its saved DescribeCoverage declares nearest-neighbour as the default, which
applies because the request omitted interpolation. The saved capabilities and
[AHN4 documentation](https://www.ahn.nl/ahn-4) identify the 2020–2022 acquisition
period; the local flight date is unresolved. The WCS request, metadata responses
and byte hashes are preserved. A swissALTI3D edition or
STAC publication date is not asserted to be its survey date. Swiss TLM retains
all original feature dates, source years and identifiers. Polygon Z is discarded
only for the explicitly declared 2D footprint rasterization; LN02 terrain heights
are neither relabelled nor converted. Unsupported NAP/LN02 conversions refuse.

CBS values remain on their original 100 m cells in both revisions. The subset has
6,197 published features, 6,129 valid count cells, 68 disclosed cells and 8,203
absent cells. The sum of the published, rounded values is 646,840; this is **not**
a complete regional population estimate. Values are rounded to five. Code
`-99997` is retained for present suppressed/unknown values. An absent feature has
no valid count; its masked storage payload is not a zero population estimate.
Original cell IDs, negative flags, presence and disclosure masks are retained.
The categorical `population_state` source travels through saved preparation,
viewer assets and the QGIS project: 0 = absent feature, 1 = present suppressed or
unknown value, 2 = published rounded count. Gray shows absence and purple shows
the present negative-code state. Published counts use the population colours.

The population display uses a reusable `scale: log1p` range [0, 2010], identical
in both regions and tools. New count values do not require code or class-lookup
edits. Colours saturate at the declared endpoints for out-of-range values;
analysis counts remain unchanged. Blue hatching means no measured terrain height;
the explicit 0 m render fill is not a measured water level.
That height hatch takes visual priority; the retained state layer still exposes
absence versus suppression in these cells.

Swiss TLM does not classify the entire landscape. Unmapped ground remains
unknown; no grass, settlement or other class is invented. The initial raster has
653,128 unambiguous class cells, 352,784 unknown cells and 12,169 cells with
overlapping different classes. Ambiguous cells are masked, and the original
polygons remain available. Every valid class centre is independently checked
against its measured polygon using Shapely.

## Reproduce from local files

These are the recorded Windows commands. Use the normal Forge3D environment with real rasterio, pyogrio/GeoPandas, Shapely,
Pillow and psutil available. These examples do not add production dependencies.
Import real rasterio before the repository shim to avoid the test stub. Set
`FORGE3D_NO_BOOTSTRAP=1` when validating the existing local binding.

```powershell
$env:FORGE3D_NO_BOOTSTRAP='1'
python examples/c05_sources.py artifacts/c05-real-data/sources
python examples/c05_real_data.py stage artifacts/c05-real-data --cbs-zip artifacts/c05-real-data/sources/cbs_vk100_2024_v2.zip
python examples/c05_real_data.py prepare artifacts/c05-real-data --case rotterdam
python examples/c05_real_data.py prepare artifacts/c05-real-data --case rotterdam --revision
python examples/c05_real_data.py prepare artifacts/c05-real-data --case grindelwald
python examples/c05_real_data.py prepare artifacts/c05-real-data --case grindelwald --revision
python examples/c05_real_data.py controls artifacts/c05-real-data
python examples/c05_render.py artifacts/c05-real-data/rotterdam/initial
python examples/c05_render.py artifacts/c05-real-data/rotterdam/revision
python examples/c05_render.py artifacts/c05-real-data/grindelwald/initial
python examples/c05_render.py artifacts/c05-real-data/grindelwald/revision
python examples/c05_report.py registration artifacts/c05-real-data
python examples/c05_report.py reexport artifacts/c05-real-data
```

Use a fresh output directory for staging/preparation. Reuse existing recorded
sources through their manifests; a changed pinned file fails verification.
Interrupted downloads have `.partial` names and are never treated as complete.
Swiss archive members are range-extracted with length and CRC checks. No
credentials, provider guessing or implicit fallback source is used.

Each prepared directory contains `recipe.json`, `prepared/`, `coverage.html`,
`checks.json`, `viewer-assets/`, `display.png` and `display.tif`. Edit the recipe's
target grid to author another region, then execute it with
`python examples/prepare_scene.py <recipe.json> <new-directory>`; application
internals need no change. `PreparedScene.load()` reuses saved data without
reopening original sources. The controls inject missing height metadata, a
one-source-pixel translation and an unsupported height conversion; they require
specific actionable diagnostics, not generic failures.

`forge3d-render/` contains 241 decoded 1280 × 720 PNGs, the eight-second camera
path (including its endpoint), a lossless animated WebP, a contact sheet and
resource/timing evidence. The path follows the existing camera animation demo
at the viewer API's default 30 fps. It uses physical height scale 1 and fixed
display mappings. PBR must be enabled for the viewer's draped overlays. Native
viewer snapshots retain their documented exclusion from offscreen CENSOR
certificates. Executable and binding hashes identify the actual runtime used;
these runs do not assert a fresh native build or cross-platform qualification.

The parent preparation binding and installed viewer binding are different local
binaries: workspace package 1.34.0 and installed viewer distribution 1.44.0.
Fresh frame runs record the child's loaded module paths/hashes and distribution
versions during each run, together with executed Python code hashes. The older
`runtime-identity.json` is a separate diagnostic probe, not frame-run identity.
QGIS's observed OpenGL share context
and Forge3D's Vulkan adapter both identify the RTX 3070 and driver 610.60; this is
local runtime evidence, not a portability certificate or matched shader result.

## QGIS comparison

The recorded run used an isolated extraction of QGIS 3.44.0 inside the working
directory, with its MSI source and hash recorded beside it. A normal QGIS 3.44
installation can supply the same OSGeo environment. Run the following in a
dedicated **Command Prompt** from the repository root. Set `QGIS_ROOT` to your
QGIS 3.44 folder (the one containing `bin\o4w_env.bat`):

```bat
call "%QGIS_ROOT%\bin\o4w_env.bat"
set QGIS_PREFIX_PATH=%OSGEO4W_ROOT:\=/%/apps/qgis
set GDAL_PAM_ENABLED=NO
set PATH=%OSGEO4W_ROOT%\apps\qgis\bin;%PATH%
set PYTHONPATH=%OSGEO4W_ROOT%\apps\qgis\python;%OSGEO4W_ROOT%\apps\qgis\python\plugins
set QGIS_AUTH_DB_DIR_PATH=%CD%/artifacts/c05-real-data/qgis-profile/standalone-auth
set QT_QPA_PLATFORM=offscreen
"%OSGEO4W_ROOT%\bin\python.exe" examples/c05_qgis.py artifacts/c05-real-data/rotterdam/initial
"%OSGEO4W_ROOT%\bin\python.exe" examples/c05_qgis.py artifacts/c05-real-data/rotterdam/revision
"%OSGEO4W_ROOT%\bin\python.exe" examples/c05_qgis.py artifacts/c05-real-data/grindelwald/initial
"%OSGEO4W_ROOT%\bin\python.exe" examples/c05_qgis.py artifacts/c05-real-data/grindelwald/revision
set QT_QPA_PLATFORM=
```

Run `examples/c05_qgis.py <prepared-directory>` with QGIS 3.44's Python and its
OSGeo environment. Use an isolated profile and authentication directory. This
separately implemented pipeline consumes the same shared raw subsets and recipe. It does
not import Forge3D to create its data. It saves an editable `qgis/scene.qgz`, the
native count layer, aligned terrain/classes and a registered plan view. Only
after producing its outputs does it compare them with the Forge3D snapshot.
Both paths use GDAL's warp family, and some extents are no-op grids or integer
crops. Exact agreement is not independent geographic ground truth. The report
separately compares BRT river footprints with AHN voids across the supplied
review's ±30 m shift sweep at half-pixel steps, and compares the Eiger DEM peak
with the [SAC's metre-rounded 3967 m summit height](https://www.sac-cas.ch/en/huts-and-tours/sac-route-portal/eiger-555/).
The report records every score and cardinal-shift margin, both for whole-extent
IoU and agreement in the re-review's ±60 m shoreline corridor. Zero beats a
one-pixel-west shift by only about 0.000132 IoU / 0.000530 corridor agreement.
That small margin supports approximate one-10-m-pixel registration; it cannot
rule out a westward half-pixel offset. No fitted shift is applied. The Swiss
horizontal check compares the ALTI3D maximum (2643451, 1158641) with the
independent official swissNAMES3D Eiger point (2643436.3, 1158637.9), a measured
15.023 m separation. Named-summit coordinates and a DEM maximum have different
definitions, and the landmark accuracy is unspecified: no subpixel or survey
accuracy is inferred. The raw API response, feature ID, CRS, URL and hash are
pinned by `c05_sources.py --swiss-reference-only` (also included in full source
acquisition). Height agreement with the SAC reference is a separate check. The original
source-affine checkpoint is only a regression/shift control.

Both QGIS scripts disable GDAL PAM before opening datasets, so comparison runs
cannot add `.aux.xml` files to the shared source identities. The earlier sidecar
and results are preserved in `history-before-review-fixes/`; fresh preparations
use clean staged inputs and are checked again after QGIS runs.

For native 3D captures, set `C05_SCENE_DIR` to an absolute prepared directory and
run QGIS with `--code` pointing to a small bootstrap that adds this repository's
`examples` directory to `sys.path` and imports `c05_qgis_3d`. The script uses
QGIS's native **Save as Image** action with the recorded camera samples. Its
viewport embedding follows the pinned QGIS 3.44 layout. Keep `QT_SCALE_FACTOR=1`
and `QT_AUTO_SCREEN_SCALE_FACTOR=0`. `C05_QGIS_PROBE=1` requests only one diagnostic
frame; this is not full-sequence evidence. The normal route requires all frames.
The capture waits for the named `Qgs3DMapScene.Ready` state and zero pending jobs;
it does not guess the enum's numeric value. After both tools finish all four
sequences, run the report into a fresh, separate output directory:

```powershell
python examples/c05_report.py report artifacts/c05-real-data --output artifacts/c05-real-data/report-next
python examples/c05_report.py presentation artifacts/c05-real-data --output artifacts/c05-real-data/report-next
```

This verifies every animation frame and writes the local HTML comparison into
`report-next/`. The second command is an idempotent presentation replay. The
recorded captures are read-only inputs: re-encoded QGIS movies, contact sheets
and checks go into the new report tree. A report output must be new, and its
exact producer source and SHA-256 are saved beside `producer.json`. To replay a
changed report script without touching recorded captures, run it on a copy of
the working directory and compare its numeric and image results with the
previous report.
No physical frame or QGIS regeneration is implied by this report replay.
No optional PyQt3D package is required. `C05_QGIS_OUTPUT_NAME` can select a fresh
output subdirectory while preserving a failed run.

QGIS runs the capture through a three-line bootstrap passed to `--code`. Save it
as `artifacts/c05-real-data/qgis_capture_bootstrap.py`, replacing `<repo>` with
the absolute path of this repository:

```python
import sys
sys.path.insert(0, "<repo>/examples")
import c05_qgis_3d
```

Before first GUI startup, the isolated profile's
`qgis-profile/profiles/c05-render/QGIS/QGIS3.ini` disables unused password-manager
integration with `generate-random-password-for-keychain=false` and
`using-generated-random-password=false` in `[authentication]`, and
`use_password_helper=false` in both `[authentication]` and `[auth]`. No account
or authentication configuration is needed for these local scenes.

Continue in the dedicated Command Prompt, after the environment setup above:

```bat
set C05_SCENE_DIR=%CD%/artifacts/c05-real-data/grindelwald/initial
set C05_QGIS_PROBE=0
set C05_QGIS_OUTPUT_NAME=qgis-3d
set QT_SCALE_FACTOR=1
set QT_AUTO_SCREEN_SCALE_FACTOR=0
set QGIS_AUTH_DB_DIR_PATH=%CD%/artifacts/c05-real-data/qgis-profile/c05-render-auth
set QT_PLUGIN_PATH=%OSGEO4W_ROOT%\apps\qgis\qtplugins;%OSGEO4W_ROOT%\apps\Qt5\plugins
start "" /wait "%OSGEO4W_ROOT%\bin\qgis-bin.exe" --nologo --noversioncheck --noplugins --profiles-path %CD%/artifacts/c05-real-data/qgis-profile --profile c05-render --code %CD%/artifacts/c05-real-data/qgis_capture_bootstrap.py
echo Native QGIS exit: %ERRORLEVEL%
```

Repeat with the other three prepared directories. Preserve the native exit
status as well as the full-sequence evidence. Each capture output directory must
be new; the script does not overwrite an earlier run.

Compare correctness, readability, visual preference and sequence stability
separately. Heights and masks matching numerically do not establish a visual
winner. Lighting, tone mapping and renderer LOD differ; record those differences.
The local timings are descriptive runs on a shared workstation, not isolated
benchmarks. Render-wall time is not a GPU timestamp. Per-process Windows kernel
memory peaks and their sums are labelled separately; a sum of peaks is not a
simultaneous peak. Acquisition and common staging are shared costs. Human setup
time and operator expertise were not measured by these agent-authored scripts.

## Landscape packages and editor evidence

The Swiss `landscape/IMPORT.md` contains the complete dimensions, physical scales,
anchor, axes, material-layer names and manual import procedure. `landscape.json`
records reconstruction, all hashes, validity and the distinction between source
pixel centres and destination vertices. Pillow independently decodes every
exported height and class mask. Initial maximum error is 0.0237206811 m against
its derived 0.0237206843 m bound; revised maximum error is 0.0232718899 m against
0.0232719677 m. These are encoding errors, not survey accuracy claims.

Package creation and a Forge3D/QGIS flight do not certify an Unreal import. Both
Swiss packages were then imported manually into Unreal Editor 5.8.3 and checked
for heights, masks, editing and full camera flights; see the
"Optional Unreal Landscape package" section of the [data-to-scene guide](data_to_scene.md) for
the route and its limits. That validates those recorded packages and settings,
not another package or Editor version.

Sources: [AHN dataroom](https://www.ahn.nl/dataroom),
[corrected CBS archive index](https://geodata.cbs.nl/files/Vierkant_statistiek/2024/),
[CBS grid semantics](https://www.cbs.nl/nl-nl/longread/diversen/2025/statistische-gegevens-per-vierkant-en-postcode-2022-2023-2024/2-technische-gegevens),
[swissALTI3D](https://www.swisstopo.admin.ch/en/height-model-swissalti3d),
[swissTLM3D](https://www.swisstopo.admin.ch/en/landscape-model-swisstlm3d),
[BRT water footprints](https://api.pdok.nl/kadaster/brt-top10nl/ogc/v1/collections/waterdeel_vlak),
[QGIS 3.44 API](https://qgis.org/pyqgis/3.44/), and
[Epic Landscape guide](https://dev.epicgames.com/documentation/en-us/unreal-engine/landscape-technical-guide-in-unreal-engine).
