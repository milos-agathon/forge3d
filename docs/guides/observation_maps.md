# Fixed-camera observation maps

`forge3d.observation_map.compile_observation_map` takes one MapScene
`SceneRecipe`, an `ObservationTrack`, an explicitly authored sequence of
timezone-aware presentation instants and a fixed RGB domain. It returns
`CompiledObservationMap`. Use `frame(index)` to inspect a compiled MapScene,
`render(directory, base_seed=0)` to execute through CHRONOS, and
`selection_table()` to retrieve the source IDs, hashes, states, masks, event
windows, captions and interventions. Use the returned CHRONOS manifest's
`replay_observation_frame(index, output)` to reproduce and validate
the frame's pixel hash. A random run via `render(..., frames=[index])` retains
the original frame index, camera, seed and inputs.

Raster payloads use C06's `PublicationSource.description()` under `source`.
The file for this route is a prepared single-band or three-band RGB GeoTIFF
on the exact terrain grid. Write scale and offset into each band and embed the QA mask
in the TIFF. The compiler applies the same linear stretch to every acquisition
and uses immutable content-addressed styled TIFFs for native terrain draping.
Retain the snapshot directory and the static scene assets with the manifest.
If source provenance declares a SHA-256, update it explicitly when authoring
a legitimate acquisition replacement. A modified snapshot raises an error.
MapScene's existing cache eligibility applies. The observation furniture
causes an explicit cache bypass and fresh execution.

Raster validity is half-open. It never starts before acquisition. An active
overlap selects the latest acquisition and equal winning acquisition instants
raise an error. Outside validity, the track's policy either holds the latest
available acquisition and labels its age, or displays a missing caption.
`ObservationEvents` uses the intersection of each record's half-open validity
and `(presentation - lookback, presentation]` on acquisition time. Each event
record carries a GeoJSON `Feature` under `feature` and provenance under
`source`. All qualifying records enter the frame, including simultaneous
records. A zero count means zero reports in the declared window with coverage
unknown. Captions and legends are outside the mapped area.

## Reproduce the Dixie Fire example

Use Python with the matching Forge3D wheel, Rasterio, PyProj, NumPy and Pillow.
Put FFmpeg on PATH. The independent comparator uses QGIS 3.44.15's Python
launcher, its GDAL and NumPy, and Qt in offscreen mode.

The [frozen source list](dixie_observation_sources.json) specifies exact
windows, product IDs, unsigned URLs, hashes, QA and fixed spatial and visual
domains. Put the four raw STAC item JSON files and their five band files in
`<raw>/landsat/`, the two Copernicus tile item JSON files and GeoTIFFs in
`<raw>/dem/`, and the FIRMS CSV directly in `<raw>/`. Use the item IDs as
the JSON filenames and the original basenames for the raster files. The
Planetary Computer blob downloads require its public SAS signing service.
Preparation checks source bytes against the frozen list. Landsat MTL text
was unavailable via the tested asset URL. This example records acquisition
and scale/offset from the archived STAC items, and CRS and grid from the
actual original GeoTIFFs.

From the worktree root, set `$e` to the evidence directory that contains those
raw inputs. Set `$qgis` to the full path of QGIS's `python-qgis-ltr.bat`.
The following commands render from a clean output location.

```powershell
$python = ".venv/Scripts/python.exe"
$env:WGPU_BACKEND = "vulkan"
$env:DIXIE_FFMPEG = (Get-Command ffmpeg).Source
& $python examples/dixie_observation_map.py prepare $e "$e/prepared"
foreach ($variant in "original", "revised") {
    & $python examples/dixie_observation_map.py render "$e/prepared" "$e/final-$variant" --variant $variant
    & $qgis examples/dixie_observation_qgis.py "$e/prepared" "$e/final-$variant" "$e/final-qgis-$variant" $variant
}
```

The script writes full-size stills, a dated comparison plate, a lossless
hard-cut movie, the time-selection table, CHRONOS manifest and a native
execution receipt. The QGIS comparator independently selects source records
and all events, then checks their identities against the corresponding table.
It uses the same prepared RGB/QA inputs, fixed extent, style and output size
and writes its own matching stills and lossless movie.
The authored revision changes two presentation dates and replaces August 12
with August 28. Detections are individual FIRMS reports. The example makes
no fire-extent, fire-absence or smoke classification.

USGS documents the reflectance scale and offset and the QA flags. NASA
documents FIRMS archive coverage. [USGS reflectance products](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products)
[USGS QA bands](https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands)
[NASA FIRMS archive](https://firms.modaps.eosdis.nasa.gov/download/)
