# Linked map publications (C06, partial)

`PublicationRecipe` builds native `MapScene` frames and composes them with
`MapPlate`. One Python recipe produces PNG series by extent, subject or declared
time, with a classification-derived legend, camera-footprint locator, source
and observation-year captions, and qualified geographic scale. It accepts no
pre-rendered map image. Map and locator pixels are the native unlit albedo, so
every displayed colour is a legend colour.

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

### Reusable layouts, subjects and title/end frames

`PublicationLayout(config, insets)` carries page geometry and linked
`PublicationInset(inset_id, title, bounds, rect)` declarations. An inset reads
the same current source and classification as the main map; it is not an
imported screenshot. `rect` gives absolute integer pixel edges. Reserve space
in the left margin: overlaps with the main map, right sidebar, title, caption
or another inset fail before rendering. Each inset gets its own native frame
at the displayed size, title, source/classification identities and render
receipt. Its scale is omitted. Source, encoding or inset revisions regenerate
the linked view. Assign the same layout to multiple recipes to reuse it.

`render_subjects(subjects, pages, directory)` uses that layout and those
extents for each `PublicationSubject(subject_id, title, source, breaks,
palette)`. Every subject declares its dataset and encoding; the method never
silently changes class breaks or units. Outputs live in distinct subject
directories. Case-colliding or unsafe subject IDs fail before rendering.

`render_bookends(first, last, directory, layout=frame_layout)` produces
`title.png` and `end.png` from a single source/style snapshot. The explicit
frame layout controls their final pixel dimensions, linked insets and furniture.
Changing the recipe regenerates both. This is a still-frame interface with no
binding to a flyover. Layouts that cannot
fit their contents raise instead of cutting labels or shrinking them silently.

`render_flyover(frames, directory, layout=frame_layout, base_seed=seed)` publishes
an explicit sequence of flat-terrain perspective cameras through CHRONOS.
Each `PublicationExtent` supplies a camera and a distinct frame ID; all frames
share one registered terrain extent. Camera authoring belongs to C07, and C06
does not interpolate cameras or observations. The native frame's pixel hash,
camera hash, scene hash and retained CHRONOS manifest bind each composed page
to its executed frame. The locator follows that camera's ground footprint.
The title and end PNGs are byte-for-byte copies of the first and last composed
frames, including their furniture. Perspective scale bars remain omitted.

The delivery requires a new or empty directory. Native frames and composed
pages are staged before installation; the completion manifest is installed
last. A failed render never publishes a completion manifest. This route emits
a PNG sequence, not an encoded movie. Run `examples/c06_flyover_publication.py`
with an executed oblique receipt and explicit frame count, duration, final
azimuth and seed to author cameras with C07's `CameraAnimation`. The example
regenerates the sequence after changing classification thresholds.
`examples/c06_qgis_publication.py --sequence` consumes the resulting
`publication-flyover.json` and renders each camera in independent QGIS layouts.
Machine timings do not measure human setup effort or expertise.

`render_times(track, presentation_times, pages, directory)` consumes
[`forge3d.observation_time`](observation_time.md), owned by C04. Each observation
payload contains `source: PublicationSource(...).description()`. C04 chooses
the record and produces its time/age caption; C06 takes the year, dataset and
provenance from that record. A held page visibly retains its age. A missing
selection produces a notice page with no map, source or invented observation.
`selection-table.json` retains all records, policy, selections and output
directories after successful rendering; individual receipts retain the exact
selection they display. Declared timezones remain visible and age is expressed
in days. A source year must match C04's observation year in its declared timezone.
Existing time folders or page files absent from the new selection table are
refused before writing: use a fresh output directory for a shorter series.
The table, missing-page PNGs and receipts are individually replaced atomically;
the entire series is not transactional. One layout and classification span the time series.

Run `python examples/c06_time_publication.py PREPARED_C06_DIRECTORY OUTPUT`
against an executed population example to use the real prepared 2023/2024 CBS
files. Their hashes must match the preparation receipts. The example uses
CBS's [1 January population reference date](https://www.cbs.nl/nl-nl/longread/diversen/2025/statistische-gegevens-per-vierkant-en-postcode-2022-2023-2024/4-beschrijving-cijfers)
(checked 1 November 2026). Midnight Netherlands time and annual validity are
explicit authoring conventions; they are not archive release times or a claim
that the January observation remains current throughout the year. This minimal
C04 dependency leaves `MAP-OBSERVATION-TIME` and both C04/C06 capabilities open.

### Output support by route

`publication_output_support()` returns these declarations. Each publication
receipt includes the declaration for its own route; it is not a visual or
production qualification certificate.

| Route | Format / resolution | Colour / losses | PDF | SVG | Editable text | Print colour |
| --- | --- | --- | --- | --- | --- | --- |
| PublicationRecipe extent/subject/time pages, bookends and flyover frames | PNG, configured pixel dimensions; maps rendered at displayed size | sRGB RGBA8, flattened text/maps, no embedded geographic registration | Unsupported | Unsupported | Unsupported | Unsupported |
| MapPlate.export_png | PNG, configured pixels; inputs can be resized nearest-neighbour | Caller-supplied RGBA8, no colour conversion; flattened text/maps | Unsupported | Unsupported | Unsupported | Unsupported |
| MapPlate.export_jpeg | JPEG, configured pixels; inputs can be resized nearest-neighbour | RGB8, alpha discarded, lossy colours/edges; flattened text/maps | Unsupported | Unsupported | Unsupported | Unsupported |

These declarations concern the named composition routes. They do not qualify
other exporters, physical prints, ICC conversion or press proofs. Existing
product entitlement checks still apply.

### Simple MapScene furniture

The simple legend accepts style `entries` with `label` and explicit RGBA8
`rgba` values, or existing `items` with `label` and `color`. It displays every
entry, measures the text and rejects overflow.
Legacy string `items`/`labels` remain text-only; their names never generate
invented swatches. Automatic scale bars require a screen plan view and an
explicit CRS and registered bounds. Bounds must transform to finite geographic
coordinates; the map centre must lie in the CRS's published area of use, when
one is defined. Edges may cross that area boundary, as national maps often do.
This check cannot distinguish degree-valued bounds incorrectly labelled as
Web Mercator from legitimate metre-valued coordinates near its origin; callers
must keep bounds, raster registration and CRS consistent. Automatic scale is
horizontal, along the map's centre line; it does not certify vertical scale.
An explicit measured `meters_per_pixel` bypasses geographic scale inference.
Geodesic scale calculation still requires a native-supported CRS transform.
For unsupported transforms, explicitly choosing `geodesic=False` measures
projected horizontal distance in the CRS's linear units; it is not geodesic
ground-distance qualification.
Perspective views must omit the bar or provide both an explicit
`meters_per_pixel` and a visible `qualification` describing where it applies.
The qualification must fit the bar image. A local perspective scale is the
caller's measured value, not an automatic geographic guarantee.

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
itself. The setting supports one-shot `screen` views and the narrowly qualified
flat-terrain perspective route described below, without effects. A missing AOV
raises rather than falling back to the beauty frame.

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

Flat, north-up `screen` views map the full cropped raster UV rectangle to the
viewport. `camera_mode="flat-perspective"` additionally accepts an explicit
`PublicationExtent.camera` and square ground extent. This is **flat-terrain
oblique only**, not general oblique support. The native gate verifies the actual
height samples are finite and constant. Offline rendering, effects, clipmap,
non-flat terrain and other camera modes remain rejected.

The perspective locator comes from ground-plane intersections of the actual
native camera projection. A corner ray missing the ground or any viewport
coverage outside terrain rejects the view; background never becomes a class or
missing-data value. No perspective scale bar is drawn, and the receipt explains
that ground distance per pixel varies across the view. Camera changes invalidate
the normal scene cache. All pixels, including class edges, retain the one-level
legend-colour tolerance. The GPU test independently reconstructs pinhole rays,
checks every main-map pixel and checks the drawn locator against geographic
coordinates; it does not reuse the publication footprint helper.

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

The broader C06 capability remains partial. Linked insets, subject/extent/time
series, flat-terrain oblique locators and matching title/end PNGs are implemented.
Time selection remains the limited C04 dependency, not full temporal authoring.
The flyover route binds title/end PNGs to executed CHRONOS endpoints; its
qualification requires retained sequence evidence and inspection of every new
output at final size, including all page edges. A fresh assessment after the
follow-up merge remains required.
Arbitrary terrain oblique views, arbitrary scene layers, comparative human effort
and competitive superiority remain unqualified. Unreal handoff and video encoding
are outside the C06 acceptance boundary.
Local evidence is not merged-main, hosted, cross-backend or cross-machine
qualification. Production licensing and the full merged-main gates must be
observed before closing the ticket.
