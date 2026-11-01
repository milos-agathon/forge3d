# Linked plan-view publications (C06 first delivery)

`PublicationRecipe` builds native `MapScene` frames and composes them with
`MapPlate`. One Python recipe produces an extent-driven PNG series with a
classification-derived legend, actual screen-view footprint locator, source
and observation-year captions, and qualified geographic scale. It accepts no
pre-rendered map image. Map and locator pixels are the native unlit albedo, so
every displayed colour is a legend colour.

This local implementation starts at main
`37159353f618f76e6020f6221b2a475e7969013a`. The older root checkout
`7c81644ad7d7180dd51d59d93b2180878eec5131` and its unrelated edits were preserved.
Implementation and local validation used a dedicated worktree.
The evidence below establishes the local first delivery; merge and release are
separate integration steps.

## Use

Requires the matching Forge3D Python source, its native extension, existing
MapPlate Pro access, Pillow and pyproj. The CBS preparation example also uses
real rasterio, pyogrio and their existing geospatial dependencies. No new
production dependency is added. Import real rasterio before the example's
repository import shim, which otherwise finds the partial test stub.

```python
from forge3d import PublicationSource, PublicationExtent, PublicationRecipe, BBox

source = PublicationSource(
    "population-2024-utm31.tif", "CBS population grid", "2024",
    "2024 v2 / corrected archive", "https://geodata.cbs.nl/files/Vierkant_statistiek/2024/",
    "People / source 100 m cell", "No published count (absent or suppressed)",
)
recipe = PublicationRecipe(
    source=source,
    breaks=[50, 150, 300, 600],
    palette=[(255,243,204,255), (249,200,118,255), (234,139,76,255),
             (193,71,61,255), (115,31,58,255)],
    locator_bounds=BBox(580000, 5730000, 650000, 5820000, "EPSG:32631"),
)
pages = [PublicationExtent("city", "City", BBox(600000, 5750000, 616000, 5766000, "EPSG:32631"))]
receipts = recipe.render_series(pages, "publication", cache="native-publication-cache")
```

The coordinates and filenames above illustrate the API. Use the executable
CBS example below for the actual prepared extents and source files. All page
and locator extents must fit the source and follow its exact cell edges. Edge
comparison allows only float64 representational rounding: four ULPs of the
largest coordinate, in cells. A larger offset is rejected as misaligned.

## Dependency and rendering contract

Every call rereads the GeoTIFF and hashes its bytes, registration and declared
provenance. `gis.classify_raster` owns classification and interval closure.
The recipe creates `CategoricalRasterStyle` labels from that actual class
table, then uses its `RasterStyleResult.legend`, palette and units. File nodata,
masked cells and non-finite values share the declared missing-data key; valid
zero values remain classified values. The missing key is always displayed.

Each page's JSON receipt exposes identities and dependencies:

| Element | Inputs that affect it |
| --- | --- |
| Classification | Actual source bytes, breaks, bin closure, palette, units, missing rules |
| Main map | Classification, exact extent, output viewport, native scene |
| Legend | Class boundaries/closure, colours, units and missing-data key |
| Caption | Source name, declared observation year (checked against the embedded tag when present), edition, URI and view qualification |
| Locator | Its own native map and the main viewport's geographic footprint |
| Scale | Extent, CRS and actual displayed map width |

Changing extent leaves the legend unchanged when its semantics remain the
same. Changing year does the same when the classification is intentionally
shared. The underlying map and locator data still change. Replace the entire
`PublicationSource` with the corresponding year's actual file; a year field
does not download, select or manufacture another dataset.
When the GeoTIFF has an embedded `observation_year` tag, it must match the
declared year. Without that tag, verifying the observation year remains the
caller's responsibility. Non-identity band scale/offset metadata is rejected:
decode and verify such values before publishing. External GDAL metadata
sidecars (`.aux.xml`) are unsupported; inputs must be self-contained GeoTIFFs.

Every map passes through `MapScene.render`, including repeat calls. The lit
beauty frame is shaded and ACES tone-mapped (measured: every class at about
0.46× its sRGB palette value), so it cannot carry legend colours. The recipe
therefore sets the lighting setting `unlit_albedo`. With it, the native terrain
path encodes its albedo AOV to opaque sRGB RGBA8 and returns that as the frame
itself. The setting only supports one-shot flat `screen` views without effects,
and a missing AOV raises rather than falling back to the beauty frame.

Because the albedo is the frame, it passes through `MapScene`'s ordinary keyed
and hash-verified cache. Supplying `cache` reuses a frame only when the whole
scene identity is unchanged, including raster content hashes and the lighting
setting, so lit and unlit frames never share an entry. Without `cache`, maps
render fresh. There is no separate publication image cache. Fresh or reused,
every drawn colour must lie within one 8-bit level of a legend colour (the
half-float store's rounding), or the page fails. Receipts record the measured
deviation.

Palette and missing colours must be fully opaque. A non-opaque key raises,
because the opaque frame could not show it.

Neither an existing final PNG nor a filename/mtime is evidence of a current
frame. Receipts retain backend/cache metadata. A native failure propagates as
`MapSceneNativeUnavailable`; previous files are never returned as a successful
new result. Pages are composed in temporary storage before replacement. A
series is not a transaction: if a later page fails, earlier completed pages
remain, and each receipt identifies its own inputs. Concurrent source edits
during one call are unsupported.

## Qualified geography and output

Only flat, north-up `screen` views are supported. That native mode maps the
full cropped raster UV rectangle to its viewport. The locator polygon is
therefore the actual viewport footprint. Perspective/mesh/clipmap cameras are
rejected; no bounding box is substituted for their camera footprints.

Page IDs are compared with `casefold()` and must be distinct even on a
case-insensitive filesystem. Legend colours, including the missing-data key,
must differ by more than the renderer check's one-level RGBA8 tolerance in at
least one channel; duplicate or indistinguishable keys are rejected.

Input must be a single-band, north-up GeoTIFF in a projected metre CRS supported
by the native CRS helper. Geographic, foot-based, rotated/skewed and unsupported
CRSs fail explicitly. The scale preflights that helper and pyproj, avoiding
`ScaleBar`'s legacy approximate fallback. Scale is WGS84 ground distance along
the horizontal centre line, computed for the actual displayed map width. It
is labelled as varying across the map; it is not a perspective or globally
constant ground scale. Grid north is identified separately from true north.

The default publication is **1600 × 1200 pixels**, inspected at 100% in a PNG
viewer. MapPlate centres the geographic aspect inside its map region; the
recipe rejects any layout in which MapPlate would resample the map, because the
scale would no longer apply. Pixel rounding of width/height is recorded in the
receipt. The scale bar is drawn 20 px inside the bottom-left of the *drawn* map
rectangle (`map_rect`), not the wider region, and is rejected if it cannot fit
on the map. Text that cannot fit is rejected. Output is existing display RGBA8
PNG, with raster text. The `dpi` field inherited from MapPlate does not
establish physical print size or a colour profile. No editable vector text,
PDF/press-colour or new-format qualification is claimed. Scale-bar lengths are
quantized to whole output pixels; receipts include the drawn pixel span, its
rectangle and the actual label from the existing `ScaleBar` helper.
Sub-kilometre scale labels use metres. Before publication, the label's ground
distance must agree with the measured drawn bar to within one output pixel of
ground distance, or the recipe raises without replacing that page.

The bundled default font covers ASCII. Unsupported characters, including
superscripts, currency signs and accented place names, raise `TextShapingError`;
there is no fallback font. Use supported text (for example, `people/km2`) with
this first delivery.

## Real CBS demonstration

With this source installed and existing dependencies available:

```powershell
python examples/c06_population_publication.py artifacts/c06-publication `
  --cbs-2024 path/to/2026-cbs_vk100_2024_v2.zip `
  --cbs-2023 path/to/2026-cbs_vk100_2023_vol.zip
```

Without archive arguments the example downloads those exact names from the
official CBS year directories. It retains archive/member hashes, original RD
subsets, prepared UTM rasters, provenance and every page receipt. It independently
changes breaks, Amsterdam's extent, and observation year; it also repeats
unchanged inputs. It asserts which dependency IDs change and which stay equal,
that unchanged inputs hit the cache and give byte-identical pages, and that each
revision misses. Rotterdam remains unchanged in the
extent-only revision. Independently of the recipe, it checks each sampled
cell's colour against a NumPy classification of the GeoTIFF and checks the
scale bar's placement and ground distance. Its locator-coordinate assertion
reuses the recipe's affine formula and is a consistency check, not an
independent test of the drawn outline.

Sources accessed 29 October 2026:

- [CBS 2024 corrected v2 archive directory](https://geodata.cbs.nl/files/Vierkant_statistiek/2024/).
- [CBS 2023 complete archive directory](https://geodata.cbs.nl/files/Vierkant_statistiek/2023/).
- [CBS technical grid definition](https://www.cbs.nl/nl-nl/longread/diversen/2025/statistische-gegevens-per-vierkant-en-postcode-2022-2023-2024/2-technische-gegevens).

The source field is `aantal_inwoners`, placed by `crs28992res100m` cell IDs
and checked against the independent polygon bounds. CBS publishes a 100 m cell
only with at least 5 inhabitants or 5 dwellings, and codes `-99997` as
"0 t/m 4 / geheim / niet aanwezig", to be read as identical to unpublished
cells ([CBS figure description](https://www.cbs.nl/nl-nl/longread/diversen/2025/statistische-gegevens-per-vierkant-en-postcode-2022-2023-2024/4-beschrijving-cijfers)).
The example therefore keeps published rounded counts and combines absent cells
and `-99997` into one key labelled **"0-4 people or not published (CBS)"**. It
never shows them as a published count, and records the original negative codes
separately in provenance.

RD New is not supported by the native CRS helper. Preparation explicitly
transforms the data to UTM 31N (`EPSG:32631`), using 100 m destination cells and
nearest-neighbour sampling. Values still mean people per **original CBS cell**;
they are not destination-cell totals, an aggregation or a complete city total.
The two grids are rotated about 1.89° relative to each other (PROJ meridian
convergence: UTM 31N +1.50°, RD −0.39° at Amsterdam). Nearest-neighbour
sampling therefore never samples 67 (2024) / 68 (2023) valid source cells inside
the UTM envelope (4,070 / 4,060 people), and samples 59 cells twice. The UTM grid is the
envelope of RD bounds `(84000, 425000, 136000, 497000)`. To keep every grid
cell backed by loaded CBS data, preparation reads the RD window covering that
whole envelope (densified edges plus one cell): `(81500, 423100, 138600, 498900)`.
It fails if any destination cell samples outside it. File nodata therefore never
means "outside the loaded subset".

Rasterio's regular reproject path approximates transforms at 0.125 source
pixels; the independent calculation found 777 changed cell assignments in the
initial attempt. The example uses `WarpedVRT` with a tolerance derived from the
float64 coordinate spacing at the largest original RD coordinate, divided by
the 100 m source pixel: `5.820766091346741e-13` source pixels. Exact zero fails
in this installed GDAL path. Both editions must agree exactly with an independent
pyproj inverse transformation of every destination cell centre. This does not
claim sub-cell source accuracy. See [Rasterio VRT tolerance documentation](https://rasterio.readthedocs.io/en/stable/api/rasterio.vrt.html)
and the [Rasterio 1.4.3 implementation](https://github.com/rasterio/rasterio/blob/1.4.3/rasterio/_warp.pyx).

## Remaining C06 scope

The broader C06 capability remains partial. This delivery does not qualify
oblique locators, arbitrary scene layers/layouts, moving-frame furniture,
matching movie title/end frames, temporal authoring, Unreal handoff, a matched
GIS/commercial workload, comparative user effort or competitive superiority.
The roadmap ticket retains those wider acceptance gaps and records this local
PNG delivery separately. Local GPU evidence is not hosted, packaged-wheel,
cross-backend or cross-machine qualification.

## Observed local evidence (30 October 2026 correction run)

The 29 October pages under `D:/forge3d/artifacts/c06-publication` are
**superseded**. An independent review found three defects in them:

- Their maps were the lit beauty frame, so no displayed class colour matched its
  legend swatch. In CIELAB, 5 of 6 classes were nearest a different swatch.
- The scale bar was anchored to the map region, not the drawn map, so it hung
  off the map onto the margin (almost entirely, on the extent revision).
- 27,192 locator cells (6.8%) lay outside the loaded RD subset but were drawn
  with the CBS missing key.

The corrected run is retained under `D:/forge3d/artifacts/c06-publication-r3`:
`base/`, `breaks/`, `extent/`, `year-2023/`, `unchanged/`, `sources/` and
`evidence.json`. Base pages are Amsterdam and Rotterdam, 1600 × 1200 RGBA8 PNG.
The default breaks are `[50, 150, 300, 600]`; the independent revision uses
`[25, 100, 250, 500]`. The extent revision narrows Amsterdam by 1 km on each
east/west edge; Rotterdam stays unchanged. The year revision uses the separate
2023 complete archive, not relabelled 2024 values.

| Source | Archive SHA-256 | RD read-window evidence |
| --- | --- | --- |
| 2024 v2 | `7736161efaae51a57fe2b879e3a4df46404bb3016b1738db5db54d9ab23191bf` | 78,244 present cells; 77,611 published counts; 633 `-99997` values; 354,574 absent cells |
| 2023 vol | `8bd9c77fa1f158e0498879572f1be3fd15116b009438f67b599890c384a70f32` | 77,822 present cells; 77,222 published counts; 600 `-99997` values; 354,996 absent cells |

Both prepared grids are 544 × 738 cells:

- **Zero** disagreements with independently inverse-projected destination centres.
- **Zero** destination cells outside the loaded RD cells.
- Independent polygon bounds agree exactly with every selected original cell ID.

**Displayed colour.** Every main and locator frame in every revision has a
maximum legend deviation of **0** levels. The example also has an oracle that
is independent of the recipe. It reads each saved page PNG, classifies the
GeoTIFF with NumPy, and compares each cell-centre pixel with its expected
legend colour. That covered 25,325 cells per page (all cells not under the
scale bar), for base, break and year pages in both cities. Every class 0–5 was
present, and the maximum deviation was 0.

**Registration.** It was checked on an asymmetric 3 × 4 categorical grid: all
twelve cell positions and nearest palette identities match. The maximum
linear-albedo conversion error is `0.0016184584738064833`, against a half-float
spacing of `0.0009765625`; after sRGB encoding this rounds to the exact palette
bytes reported above.

**Scale.** Independent PROJ/WGS84 widths are 16,003.1077447592 m (Amsterdam)
and 16,004.361426404943 m (Rotterdam). The existing scale helper agrees to
below `7e-11` m in this calculation; that is numerical agreement, not source
accuracy. The 2 km bars occupy 116 pixels, corresponding to
1,996.086557410825 m and 1,996.2429306053477 m. The rounding errors,
3.913442589175 m and 3.757069394652 m, are each below one output pixel of
ground distance. On every page, the bar lies inside the drawn map rectangle:
`[150, 949, 290, 1010]` within `[130, 100, 1060, 1030]` on base pages, and
`[208, 949, 348, 1010]` within `[188, 100, 1002, 1030]` on the narrowed
Amsterdam extent. The locator-coordinate consistency check agrees exactly;
it reuses the recipe's formula and does not independently verify the outline.

**Cache.** Run from an initially empty cache. The table shows whether the main
and locator frames hit:

| Revision | Amsterdam main | Rotterdam main | Locator |
| --- | --- | --- | --- |
| base | miss | miss | miss |
| unchanged | hit | hit | hit |
| breaks | miss | miss | miss |
| extent (Amsterdam only) | miss | hit | hit |
| year 2023 | miss | miss | miss |

Unchanged pages are byte-identical to their base page. All ten pages are also
byte-identical to the `c06-publication-r2` pages, which were composed from the
albedo AOV outside the cache. The saving is small for these simple frames: about
32 ms on the unchanged Amsterdam main frame. Tracked GPU tests in
`tests/test_publication_gpu.py` cover a same-path edit preserving size and mtime
(miss, and the page changes), lit and unlit separation, and corrupted cache
storage (never published).

**Runtime.** Validation used CPython 3.13.14, an NVIDIA RTX 3070 with Vulkan /
driver 610.60, and the installed Forge3D 1.46.0 native extension (SHA-256
`aef9f2bfe7bc8d830778c23fcbbe7d835c9306d97fab5b4ae8f50147df33af4e`). The edited
Python source was overlaid on a copy of that runtime with its font resources.
The extension was **not rebuilt** or validated as a newly packaged wheel.
Shared Python changes include MapScene's `unlit_albedo` path, immutable
categorical-style snapshots in `raster_style.py`, package exports and scale-bar
label formatting.
No native code changed.

**Entitlement.** Ordinary page export fails at MapPlate's existing Pro gate
without an active Pro entitlement. Milos authorized the repository's existing
`pro_license` test fixture for acceptance evidence. These are real native GPU
frames composed under that test entitlement. No product license gate was
removed or weakened, and no production entitlement is claimed.

Commands run from the repository root, with
`FORGE3D_NO_BOOTSTRAP=1`, `WGPU_BACKEND=vulkan` and `PYTHONPATH` pointing to the
overlaid runtime:

```powershell
python -c "import sys,rasterio,forge3d,pytest; raise SystemExit(pytest.main(sys.argv[1:]+['-q','-rfEs','-W','error::RuntimeWarning']))" <tests>
# With FORGE3D_RUN_GPU_ANAMNESIS=1: 993 passed, 2 failed, 0 skipped, in about 98 minutes.
# Both failures also occur on an unmodified export of the base commit with the
# same extension: test_anamnesis_inertness::test_unqualified_native_profile_
# retains_real_device_fingerprint (requires FORGE3D_DETERMINISTIC=1) and
# test_anamnesis_portability::test_portable_store_hits_and_capability_mismatch_misses.
# <tests>: test_publication, test_publication_gpu, test_map_plate_layout,
# test_raster_styles, test_api_contracts, test_raster_styles_gpu,
# test_mapscene_typing, test_mapscene_geometric_integrity, test_mapscene_docs,
# test_p1_docs_support_matrix, test_label_api_docs_support,
# test_support_matrices_docs, test_scale_bar_native_metadata,
# test_scale_bar_projected_units, test_mapscene_native_pipeline,
# and every tests/test_*anamnesis*.py and tests/test_*cache*.py.

python <runner> D:/forge3d/artifacts/c06-publication-r3 --cbs-2024 <2024 v2 zip> --cbs-2023 <2023 vol zip>
# Full example under the Pro test fixture: preparation, revisions and all
# independent checks passed; evidence.json written.
```

The 29 October acceptance harness, `tests/artifacts/c06/test_cbs_acceptance.py`
(gitignored), was not run; the example now carries its assertions. Unit fixtures
use fake frames only to isolate dependency/error behaviour; they are not native
proof.

**Inspection (30 October 2026).** Seven distinct corrected R3 pages were
reopened in Krita 5.2.13 at 100% zoom, including their headers and footers:
Amsterdam and Rotterdam base, break and 2023 pages, plus Amsterdam's extent page.
No visible clipping, overlap or layout defects were found. The two unchanged
pages and Rotterdam's extent page are hash-identical copies of inspected base
pages, covering all ten exports. All ten pages' map colours were checked by the
cell oracle above or by their pixel hash.

The local work item's inspection receipt records file hashes, method and
limitations; that machine-local artifact is not shipped with the package.
This closes the corrected PNGs' target-application inspection gap. It is an
on-screen check, not a physical print or colour proof, and does not replace
independent locator-pixel testing: the example and unit-test locator check still
reuse the recipe's formula. The Pro test-fixture and unrun-check limits below
remain in force.

The roadmap's actual `validateAtlas`, `validateWork`, `validateResearch`,
`validateView`, `validateStatusUpdates` and `validateDelivery` startup functions
were executed with Node 24.4.0 against its embedded data. Coverage: 164 cards,
73 nodes, 87 work items and 12 research cards. The harness executes the original
script through its startup validation boundary; it does not replace validation
with a reimplementation. Browser UI loading was not exercised: local browser
access is disabled in Work settings. The frozen atlas remains unchanged. The
validators were re-run on 30 October after updating the MAP-LINKED-PUBLICATION
and SPEC-MAP-PUBLICATION records to this run, and pass (C06 `partial`, ticket
`open`).

The original PR head `583ff3b1` passed the routine hosted PR gate, including
Fast Contract and Build Docs. That Fast Contract run did not select the
publication regressions. The fast profile now explicitly includes
`test_publication.py`, `test_publication_ci.py` and
`test_scale_bar_projected_units.py`; its dependencies include pyproj. The
physical NVIDIA Vulkan lane now names `test_publication.py` and
`test_publication_gpu.py`, enables the GPU tests and enforces zero skips.
Selection does not prove execution: the new head's hosted results must be
checked separately.

Not run in the original local acceptance: a new wheel build, Rust/WASM lanes, goldens/certificate
refresh, other adapters/backends, motion/GIS comparisons or production-entitled
export. No native code, committed goldens, signed certificates or determinism
hashes were changed.

## P2 follow-up validation (1 November 2026)

The reviewed edge cases now reject indistinguishable legend keys,
case-colliding page IDs, non-identity TIFF scale/offset and contradictory
embedded observation years. Short bars use metre labels, and a measured
label/bar mismatch exceeding one ground pixel fails before page replacement.

With `FORGE3D_NO_BOOTSTRAP=1`, `FORGE3D_RUN_GPU_ANAMNESIS=1` and
`WGPU_BACKEND=vulkan`, the candidate Python source and the installed 1.46.0
native extension passed 63 publication/scale/CI-selection tests, including six
physical GPU cases, and 250 shared API, cache, composition and CI-contract
checks. Runtime warnings were treated as errors. Native 60 m and 300 m map
probes measure the actual exported bar against independent WGS84 calculations.
The canonical `cargo forge3d-clippy` gate also passed.

The complete CBS example was rerun from empty output/cache under the approved
Pro test fixture. Every assertion passed, and all ten PNGs are byte-identical
to R3, so the recorded final-size inspection still applies to those exact
bytes. The example emitted eight Rasterio/NumPy deprecation warnings; these
were not runtime warnings or assertion failures. No fresh wheel, other-backend
or production-entitled validation is implied. The 98-minute broad suite was
not rerun. The known locator-test, small-frame, NumPy-break, cache-receipt and
other broader C06 limits remain separate from these six P2 corrections.
