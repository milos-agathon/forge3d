# Output And Integration

forge3d does not stop at a viewer window. This guide covers how the repo turns
scene state into files, notebooks, and downstream products.

## Snapshots And Frame Sequences

The simplest output path is a viewer snapshot:

```python
with f3d.open_viewer_async(terrain_path=f3d.fetch_dem("rainier")) as viewer:
    viewer.snapshot("rainier.png", width=1920, height=1080)
```

For repeated capture, pair the viewer with animation or camera-rig helpers and
write frames to disk. That is the basic pattern in:

- `camera_animation_demo.py`
- `terrain_camera_rigs_demo.py`
- `pnoa_river_showcase_video.py`

For lower-level helper functions, see the top-level re-exports from
`forge3d.helpers.offscreen` and `forge3d.helpers.frame_dump`.

## Audit source frames with ARGUS

```sh
python -m forge3d.film_audit shots/flythrough_manifest.json --out-dir audit --cut 73 --hold 73:96
python -m forge3d.film_audit shots/png --out-dir audit
```

```python
from forge3d.film_audit import AuditSettings, audit

report = audit("shots/png", out_dir="audit", cuts=[73], holds=[(73, 96)],
               settings=AuditSettings(mad_multiplier=1.0))
```

A PNG directory is ordered by the numbers at the ends of the filenames.
CHRONOS supplies frame numbers and relative PNG paths through its manifest.
Use `frames_dir` (CLI `--frames-dir`) to override the manifest's PNG directory
or to locate the PNGs for an in-memory `FlythroughManifest`. Frames must have a
common size. ARGUS measures decoded 8-bit sRGB and composites alpha over black
for visible differences. Exact repeats compare SHA-256 of the decoded RGBA.

A cut names the first frame of the new shot. An inclusive hold such as
`73:96` permits repeated frames from 74 through 96. A manifest JSON may also
declare top-level `"cuts": [73]` and `"holds": [[73, 96]]`, which ARGUS combines
with the supplied options. A hold still allows a lighting defect or pop to
be reported. Flash and pop checks use a frame on each side within the shot.
An isolated repeat or an inferred drop needs moving transitions on both sides.
ARGUS reports an explicit numbering gap at the first missing number when both
flanks establish the same stride, including during holds. In that case the
`at` crop is the first available frame after the gap, identified in
`context_roles`. A drop inferred from motion is reported at the first available
frame showing the jump.

The output directory contains `audit.json` and, for each finding, the
available before/at/after crops, a difference heatmap, and a contact sheet with
the measured time series below the image panels. The plot shows scores in green,
the cutoff in amber, and the reported frame in blue. Its horizontal axis uses
source frame numbers, and the JSON supplies the metric name, units, and values.
Each finding records the source frame number, class, score, cutoff, and
source-pixel region as `[left, top, right, bottom)`.
Disconnected pop regions on one frame get separate findings and evidence.
Single-frame shots retain their frame records and have no transition data.
Choose another output directory if `audit.json` is the input manifest.

The report includes decoded hashes, signed and absolute mean-luminance
differences for the full frame and each tile, the fraction of source pixels
exceeding the change level, and the temporal second difference. Shot spectra
give the single-sided amplitudes of linearly detrended tile luminance in cycles
per recorded frame, indexed by frequency, tile row, and tile column.
`analysis_tile_bounds` gives the tile edges in source pixels.

Motion registration, flash and pop comparisons, and fixed-line searches
use the corpus's 192-pixel reference long edge and tiles up to 12 analysis
pixels wide or tall. At that scale an 8-by-8 injected pop spanning four tiles
cleared the clean maximum. Area averaging can dilute smaller defects. A pulse
finding requires three three-frame cycles with a dominant period-3 tile
frequency. A fixed line requires a straight ridge persisting at the same
screen pixels through two transitions with moving surroundings. The CLI
returns 0 for a clean audit, 1 for findings, and 2 for input or usage errors.
Without `out_dir`, the Python call returns the report in memory.

### Default calibration

The corpus in `tests/_film_audit_corpus.py` generates 21-frame static, orbit,
faster orbit, zoom, daylight, authored hold and hard-cut controls. It also
creates independent injections for all seven classes. ARGUS uses
only the neighboring source images to estimate translation, bilinearly aligns
their overlapping pixels, and uses their disagreement as a control. Thresholds
are visible in the JSON and configurable with `AuditSettings`.

| Setting | Default | Calibration |
| --- | ---: | --- |
| Flash floor | 0.01 relative luminance | Clean peak 0.0, with 0.008898 for the white-end neutral sRGB code step and 0.031791 for the mildest injected flash. |
| Pop floor | 3.0 CIEDE2000 | The next whole unit above the clean peak of 2.668904. The smallest injected pop scored 3.541040. |
| Duplicate ratio | 0.12 | The geometric midpoint, rounded to two places, between 0.015248 for the near-repeat and 0.995131 for the smallest eligible clean transition ratio. Exact repeats scored zero. |
| Duplicate motion floor | 1.0 mean sRGB code | A transition on each side must exceed one decoded code on average to establish visible motion. |
| Changed pixel and stability level | 0.01 relative luminance | One neutral white-end sRGB code step measured 0.008898. The same level limits screen-space variation for a fixed ridge. |
| Lighting step floor | 0.02 relative luminance | Rounded geometric midpoint between the clean peak of 0.005300 and the lowest injected step at 0.117611. |
| Periodic pulse floor | 0.01 relative luminance | One white-end code step rounded up. Clean controls had no coherent pulse even with the floor disabled. The weaker injection scored 0.213220. |
| Inferred drop ratio | 1.5 | Midpoint between the nominal one-frame interval and a doubled interval. The clean peak was 1.004528 and the weakest inferred drop scored 1.784696. |
| Fixed-line contrast floor | 0.02 relative luminance | Rounded geometric midpoint between 0.001410 on a relaxed clean run and 0.256214 for the weaker injected line. |
| Outlier window and multiplier | ±3 frames, 1 MAD | Three frames per side give an isolated impulse's neighboring comparisons a minority of a full window. High findings must exceed their floor and the local median plus a median absolute deviation. Near-duplicates also check the lower median envelope. An explicit numbering gap uses the verified source stride. |

To measure PNG decode, comparison and JSON output on the same generated orbit
at 1080p and 4K, make the repository's `python` directory importable and run
`python tests/_film_audit_corpus.py --benchmark logs/argus/benchmark`. The timer
starts after the source PNGs have been written. Run
`python tests/_film_audit_corpus.py --calibrate logs/argus/calibration` to
reproduce the clean and injected measurements.

## Notebook Integration

`ViewerWidget` is the notebook-side integration layer. It keeps the viewer and
IPC model intact while giving you an embeddable front end inside Jupyter.

The notebooks in `examples/notebooks/` cover:

- `quickstart.ipynb`: basic terrain and viewer usage
- `terrain_explorer.ipynb`: notebook-driven exploration
- `map_plate.ipynb`: map-plate composition flow

## Pure-Python Composition On Top Of forge3d

Several examples use forge3d to render the base terrain image and then compose
the final deliverable outside the viewer:

- `belgium_bivariate_climate_map.py`
- `bosnia_terrain_landcover_viewer.py`
- `poland_population_spikes_height_shade.py`
- `pnoa_river_showcase.py`
- `pnoa_river_showcase_video.py`

That is a supported pattern. The viewer gives you the terrain image; PIL,
Matplotlib, NumPy, or other Python tools can handle annotations, poster layout,
or video assembly.

## Map Plates And Cartographic Furniture

The map-plate workflow is the main packaged-output layer on top of viewer
snapshots.

Core APIs:

- `MapPlate`
- `MapPlateConfig`
- `Legend` and `LegendConfig`
- `ScaleBar` and `ScaleBarConfig`
- `NorthArrow` and `NorthArrowConfig`

Use this workflow when you need titles, legends, scale bars, north arrows, and
panel layouts around a rendered map image.

See:

- `examples/notebooks/map_plate.ipynb`
- GIS tutorial 03
- gallery entry 10

## Vector Export

`forge3d.export` is the 2D vector-output path. It is separate from the raster
snapshot path and is intended for publication-style SVG/PDF generation.

Core APIs:

- `VectorScene`
- `generate_svg()`
- `export_svg()`
- `export_pdf()`
- `validate_svg()`

Use this path when the output needs to stay vector rather than being baked into
a PNG snapshot.

## Bundles And Repeatable Scenes

Bundles are the packaging and handoff format for saved scenes:

- `save_bundle()`
- `load_bundle()`
- `ViewerHandle.load_bundle()`

They matter when you need repeatable review scenes, saved presets, camera
bookmarks, or a stable asset package for another user or another machine.

See:

- `terrain_demo.py`
- Python tutorial 04

## CLI And Integration Entry Points

Beyond the viewer itself, the repo includes a few integration-oriented helpers:

- `terrain_demo.py` for the terrain-demo CLI wrapper
- `forge3d.terrain_pbr_pom` for scripted terrain rendering helpers
- `forge3d.viewer_ipc` for explicit command-based integration
- `forge3d.interactive` for example-side command loops

Use those when you are building your own wrapper or tool around forge3d rather
than following the shortest tutorial path.
