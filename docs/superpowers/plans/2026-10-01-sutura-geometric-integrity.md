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

Stage 1 status: F1, F2, F5 (raster/vector blocks) and F6 are implemented with tests.
F3's 3D azimuth conversion (compass - 90) is correct for the geometry frame but
only half the fix: until F8 converts light and view vectors into the shading
frame, `test_sun_lights_the_flank_facing_it[0.0-north-south]` fails (the other
three azimuths pass because of F8's constant south bias). F8 needs an owner
decision because of the golden and certificate churn.

## Stages (each its own reviewable commit set; this PR delivers stage 1)

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
Stage 1 changes output only for 3D camera modes (`mesh:zup`, clipmap): they
become north-up with a correct sun, and 3D scenes with screen-stretched layers
now raise instead of rendering wrong. The 2D `screen` mode is unchanged.
