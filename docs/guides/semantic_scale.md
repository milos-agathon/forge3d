# Semantic scale for a terrain flyover

Author a `ScalePolicy` in `forge3d.semantic_scale` for a prepared MapScene. Supply one
network of source-identified edges, ordered connections that must survive,
named landmarks with reasons, a north-up locator extent, and explicit omissions.
Give each frame an existing `OrbitCamera`, a geographic focus, and any observed
unavailable-detail reasons. Build the camera path with the existing animation or
camera rig API.

`compile_scale(base_scene, policy, frames)` projects each route through MapScene
at the declared output size. It compares DEM-sampled source and GENERALIS
candidate polylines in both directions. Use a pixel criterion derived from
the output. The comparison measures sampled centreline displacement. A
required connection retains its original source endpoints in either
representation, and the locator shows the original path. A route that cannot
project remains in the locator and emits a diagnostic.

Required labels enter the existing MapScene placement and terrain-occlusion
pipeline at higher priority. Semantic names use one screen anchor per
geographic point so the placement cannot jump between offsets. The anchor
retains its source XY and sits one prepared mesh interval above the sampled
terrain. MapScene still applies depth occlusion and keepouts. An optional local name can declare a projected
ground span that must reach one label em. The resulting plan records placed,
rejected, and omitted names by source ID. A required name that does not place
emits a diagnostic and retains its geographic marker in the locator. CHRONOS
receives the frozen label placements and applies its existing timeline fades.
The semantic receipt retains the solver rationale alongside the placement decisions.
CHRONOS timeline events identify optional labels that would flash for one
interior frame. The compiler records their suppression before freezing the
timeline. The receipt distinguishes spatial placement from timeline admission
where the same pose borders different events. Required names and standalone
stills retain their placements.

Unavailability selects the existing overview candidate if it meets the pixel
criterion and suppresses optional local names. When an in-view edge exceeds the
criterion, the whole itinerary moves to the locator, which announces the
degradation. The frame records the measured error and affected source ID.
Only samples inside the viewport contribute to displacement, including where a
segment crosses the viewport boundary. Edges that cannot project have a locator
continuation note. The locator reserves a footer for either notice at every
pose, so its route and markers stay in place when the text changes. The receipt
records where foreground terrain occludes in-view
samples of a required route. The caller supplies readiness snapshots.
MapScene still owns terrain and texture resources. Keep the prepared terrain
and base assets fixed for the compiled sequence.

## Bernese Oberland to Grindelwald

The example uses the retained local Swisstopo input directory. It expects
`sources.json`, `features.json`, and `sources/` with the 100 original 2025
swissALTI3D 2 m tiles and the February 2026 swissTLM3D rail and place-name
shapefile components. The example checks each DEM tile against its recorded
SHA-256, matches selected UUIDs, XY, Z and names to the original shapefiles,
records component hashes, and checks one registered 20 m DEM sample against
the original heights in each tile. The input and locator use LV95 (EPSG:2056).
Heights are in metres, LN02 (EPSG:5728). Vector Z is verified at ingestion.
The route drapes on the prepared terrain.

On this base, GENERALIS accepts UTM 32N and returns an invalid-CRS error for LV95. The example
therefore supplies EPSG:32632 as a working CRS for simplification and retains
the exact original LV95 vertices by correspondence. One prepared sample (20 m)
sets the candidate simplification tolerance and half an output pixel sets the
displacement criterion. The route uses a constant three-pixel stroke. The
200 m neighbourhood witness and 20 px label em are authored style choices.
The example checks the named glyphs with LITTERA and bakes a local atlas from
Verdana. Pass `--font` with a different compatible font when needed.

The sequence has a descent, a hard azimuth cut, a one-second authored
missing-detail case and a reverse ascent. The missing-detail interval exercises
the semantic fallback. The separate fixed-pose height probe clears the
existing native height cache after it converges and renders both readiness
states through MapScene's geographic mesh. The mesh samples resident height
pages within the prepared DEM footprint and uses the prepared DEM when the
cache is empty. The probe retains both frames, the camera mode, and before
and after residency counters. The opening and closing camera poses fit the
prepared extent inside the clip volume. The alternate policy uses the
Kleine Scheidegg–Grund rail connection and a tighter locator extent, revises the
required labels, and changes the two camera keyframes. The preview shows its
scale transitions and locator fallback. The policy records the reasons for
these changes and uses the same compiler without per-frame authoring edits.

### Local reproduction on the recorded Windows host

Use the `codex/map-semantic-scale` worktree based on
`828df486b93094dcbf1d950106d04d6994b3af60`. The staged package combines this
worktree's Python source with the source-matching 1.47.1+828df486b930 native
binary. The base supplies Preparation, MapScene, GENERALIS, LITTERA, animation
and CHRONOS. The C07 artifact directory supplies the Swiss data and base-matching
binary. The example preflight rejects other engine revisions. The recorded
environment uses Python 3.13.14, rasterio 1.5.0, pyproj 3.7.2, PyShp 3.1.6 and
FFmpeg 8.0.1. PyShp and FFmpeg are local example dependencies.

Run each command in a fresh PowerShell process. Choose a fresh output root for
each repeat because the example preserves previous runs.

```powershell
Set-Location D:/forge3d/.worktrees/map-semantic-scale
$python = 'D:/forge3d/.venv/Scripts/python.exe'
$inputs = 'D:/forge3d/artifacts/c07-subject-story/input'
$stage = 'D:/forge3d/artifacts/c03-semantic-scale/runtime'
$env:PYTHONPATH = $stage
$env:FORGE3D_NO_BOOTSTRAP = '1'
$env:WGPU_BACKENDS = 'vulkan'
& $python -c "import rasterio, forge3d; from forge3d import semantic_scale; print(rasterio.__file__, forge3d.__file__, semantic_scale.__file__)"
$out = 'D:/forge3d/artifacts/c03-semantic-scale/reproduced'
& $python examples/c03_semantic_scale.py $inputs "$out/preview" --size 640 360 --fps 12 --descent-seconds 2
& $python examples/c03_semantic_scale.py $inputs "$out/final" --size 1280 720 --fps 24 --descent-seconds 3
& $python examples/c03_semantic_scale.py $inputs "$out/baseline" --size 1280 720 --fps 24 --descent-seconds 3 --baseline
& $python examples/c03_semantic_scale.py $inputs "$out/keypose-control" --size 1280 720 --fps 1 --descent-seconds 1 --baseline
& $python examples/c03_semantic_scale.py $inputs "$out/keypose-semantic" --size 1280 720 --fps 1 --descent-seconds 1
& $python examples/c03_semantic_scale.py $inputs "$out/keypose-semantic-first" --size 1280 720 --fps 1 --descent-seconds 1
& $python examples/c03_semantic_scale.py $inputs "$out/keypose-control-last" --size 1280 720 --fps 1 --descent-seconds 1 --baseline
& $python examples/c03_semantic_scale.py $inputs "$out/reuse" --size 640 360 --fps 12 --descent-seconds 2 --reuse
& $python examples/c03_semantic_scale.py $inputs "$out/revision" --size 1280 720 --fps 24 --descent-seconds 3 --reuse
& $python examples/c03_semantic_scale.py $inputs "$out/physical-height" --size 640 360 --fps 1 --descent-seconds 1 --height-residency-probe
```

The preview is a lower-resolution offline CHRONOS render. The final output uses
the same path with independent placement and representation decisions for its
larger viewport. Each rendered directory contains a movie and exact PNG
keyframes from its CHRONOS sequence. The baseline keeps full route detail and
admits all optional labels at all scales, with the same terrain, cameras,
cartographic styling, native backend and output size. Compare the four key-pose
`run.json` receipts in both execution orders. They cover the same six poses through the overview,
valley, cut, declared outage and return. The run receipt identifies the adapter and backend from the renderer process.
Time to first PNG includes renderer setup and first-frame work. The median PNG
interval measures elapsed time between file writes. CHRONOS wall time includes
frame-plan binding, rendering, and PNG/provenance output. Native tracked peak
and process working-set peak measure different scopes. These observations do
not establish GPU timings or a performance target.
