# SUTURA follow-up: geometric integrity for 3D MapScene

SUTURA (#190, shipped in 1.40.0 via #185) guarantees that `MapScene` draws
every layer for real or blocks with `MapSceneNativeUnavailable`. It did not
check that a layer is drawn *in the right place*, or that a recipe setting
reaches the renderer. In the 3D (`mesh:zup` / clipmap) camera modes this
let several defects through. This follow-up closes them.

## Findings (1.40.0, measured on RTX 3070 Vulkan)

| # | Defect | Evidence |
|---|---|---|
| F1 | `OrbitCamera.target` is never passed to the renderer; every 3D camera orbits the tile centre. | `_build_mapscene_terrain_params` omits `cam_target`. |
| F2 | 3D views are mirrored north-south. Rows map to +Y while east is +X and up is +Z (a left-handed frame). The default 2D (`screen`) mode is correct. | North-east peak renders bottom-right with a north-up `mesh:zup` camera; top-right in `screen` mode. |
| F3 | The compass sun azimuth (0 = N, clockwise) reaches the renderer unconverted; it reads it as a math angle from +X. An east sun lights terrain from the north. | `_sun_angles_from_direction` -> `decode_lighting.rs`. |
| F4 | `RasterOverlay` is resized to the whole output frame and alpha-blended after rendering, ignoring camera and terrain. Only correct for a top-down view matching the overlay footprint. The `mapscene_terrain_raster` golden locks this in. | `_composite_python_raster_overlays`. |
| F5 | In 3D modes, vector layers are placed by stretching their bounds to the frame (`_project_world_xy`); labels follow the camera only with pre-projected anchors. | `map_scene.py:2320`, `label_plan._project_anchor`. |
| F6 | Native sky/haze, terrain sun visibility, height AO, fog and tonemap exist but `MapScene` cannot reach them; `ibl.intensity` and a tonemap setting had no measurable effect. | Look-dev runs, 2026-09-30. |
| F7 | 3D mesh geometry is a fixed 512x512 grid; ~24 s per 1080p frame for a 3800^2 DEM. | Bryce flyover prototype. |
| F8 | **Terrain sun shading is in the wrong frame in every camera mode, including the default 2D mode.** The fragment shader builds its shading normal Y-up `(-dh/dx, s, -dh/dv)` = (east, up, south), but the sun direction (`decode_lighting.rs`) and view vectors arrive in the Z-up geometry frame (east, south, up) and are dotted unconverted. South-facing slopes are lit and north-facing slopes dark whatever the sun does; east/west response follows the sun's north-south component. Fixing it changes nearly every terrain golden and the shader hashes pinned by the signed recipe certificates. | Tilted-plane sweep, 2D mode, sun at 30 deg: faces_south 161 and faces_north 61 at every azimuth; a north sun lights east-facing slopes (181). |

## Local implementation review (2026-10-05)

Published main was verified at `9f14e648cbf9e7e2f071d5c47b2eecad18f3180c`.
Stage 1 (F1/F2/F3/F6/F8) is already on that base. The changes below are local,
unmerged work in `D:/forge3d/.worktrees/sutura-overlays-main`.

Stages 2-3 add native terrain-UV albedo draping, raster source-byte identity in
CHRONOS and pixel caches, and terrain-camera projection of vectors and automatic
world point, line and area labels. Lines/rings sample the DEM before projection.
Both vector compositors retain projected depth and reject occluded pixels;
polygon fills also use the visible terrain UV and original footprint/holes.
World-unit widths resolve through the perspective camera. Generated line-label
bounds include the atlas quads, and the text compositor consumes glyph rotations.
Label projection, candidates and visibility remain compile-time decisions;
bundle loading and rendering consume frozen plans. Serialized anchors bypass
new projection, and gappy DEMs use the renderer's filled heightmap.

The `globe` option follows the camera of MapScene's array-backed clipmap, which
is flat without a globe streaming runtime. This proves the existing MapScene
camera boundary, not geodetic COG globe-streaming support. Invalid world inputs
produce structured validation/compile diagnostics before rendering.

Native CPU depth compilation replaces Python per-cell triangle work. On the
same flat DEM / 256x256 viewport workload, a 512x512 grid went from 36.3917 s to
0.01948 s; a 3800x3800 grid took 1.5827 s. A separate asymmetric-grid comparison
has byte-identical depth arrays for mesh, clipmap and globe-option cameras.
These are CPU compilation measurements, not GPU timings. Label layers share
the depth image during validation.

Correction evidence is under `artifacts/sutura-corrections/` in the worktree.
The earlier `artifacts/sutura-overlays/` captures remain relevant for absent
overlay identity and the four original recipe round trips (SSIM 1.0 with
identical reports/manifests); the correction run checks affected contracts.

SUTURA remains **partial**. Stage 4/F7 mesh density and residency are excluded.
Changed terrain-raster pixels and the shader fingerprints in signed recipe
certificates require Milos approval. Protected references remain unchanged.
The existing vector/label golden also failed before this work. Physical Metal
is ABSENT; WASM and hosted CI are unrun. No narrower stage-3 scope is assumed.

## Stages

### Stage 1: settings honoured, frame correct, wrong layers blocked
1. Pass `camera.target` to the renderer; prove the CHRONOS camera hash covers it.
2. North-up 3D frame in `MapScene` only (the native z-up mode stays as is for
   ORBIS): flip the heightmap rows and every UV-aligned raster MapScene hands
   to the renderer when a 3D camera mode is active; map `camera.target` and
   the sun into that frame.
3. Convert the compass sun azimuth for 3D modes.
4. Public `lighting.settings` keys `sky`, `sun_visibility`, `height_ao`,
   `fog`, `tonemap` forwarded to the native settings, schema-validated; unknown
   keys raise.
5. In 3D modes, `RasterOverlay`, `VectorOverlay` and labels without a camera
   projection block with a structured diagnostic instead of being stretched
   across the frame (SUTURA rule: block, never draw wrong).
6. Tests (GPU lane, zero skips): asymmetric-DEM orientation, sun-azimuth
   lit-slope, target-moves-image, and a settings contract test (each declared
   setting changes the output or blocks).

### Stage 2: native terrain-UV raster drape
Albedo map sampled by terrain UV next to the existing material normal /
roughness / mask maps; `RasterOverlay` uses it in every camera mode; its
content hash enters the CHRONOS scene hash; byte-identical output when absent.
Re-approve `mapscene_terrain_raster{,.nvidia-vulkan,.metal}`.

### Stage 3: vectors and labels projected through the terrain camera in 3D.

### Stage 4: geometry density (configurable mesh grid or camera-following
clipmap) and keeping DEM/imagery resident across CHRONOS frames.

## Compatibility
The original stage-1 camera changes affect 3D camera modes (`mesh:zup`, clipmap): they
become north-up with a correct sun, and 3D scenes with screen-stretched layers
now raise instead of rendering wrong. The 2D `screen` mode is unchanged.

Stages 2-3 preserve the absent-overlay pixels and the existing screen vector/label
coordinate convention, including inline labels in 3D camera modes. Automatic
terrain-camera label placement explicitly opts in through
`LabelLayer.metadata={"coordinate_space": "world"}`; existing projected anchors
and geometry authority retain their meaning. Raster imagery is now lit as terrain albedo in every
camera mode, so raster-present pixels deliberately change and need golden
approval. No bundle/report schema, signed certificate, committed golden or pinned
hash is changed by this local implementation. The legacy screen height/material
sampling stays intact; the new drape maps the complete logical screen tile.
