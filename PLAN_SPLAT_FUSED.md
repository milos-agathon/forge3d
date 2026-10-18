# SPLAT-FUSED "sensor + terrain" map — exploration plan

Branch: `feature/splat-fused-exploration` (created from `main`)

## Goal

Ship a runnable, self-contained example that renders **splats + LiDAR + terrain as one lit map**, with per-surface hit-kind, depth, transmittance and a cartographic sheet around it — i.e. a "sensor + terrain" visualization that only forge3d can produce (one ReSTIR integrator, multiplied transmittance `T = T_splat * T_lidar * T_terrain`, paged residency).

## Why this is unique

- One physically-lit integrator treats Gaussian splats, LiDAR/COPC surfels, and a terrain heightfield as a single scene; shadows transmit across representations.
- Deterministic, certificate-bearing renders from one Python script.
- The sheet itself carries the fusion evidence: per-layer hit counts, hit-kind AOV, and the transmittance split.

## Current state (verified on this branch)

- `splat_fusion_available() == True`, native `.so` includes `splat-fusion`.
- `load_gaussian_splats("tests/fixtures/splat_fusion/cloud.ply")` → 3,500 splats, SH degree 1.
- Test fixtures are coherent: `cloud.ply` + `dem_65x65.f32` + `swath.copc.laz`.
- Real DEM fetch works: `Mount_Fuji_30m.tif` (LFS CDN, cached in `~/.forge3d/datasets/`).
- `examples/splat_fused_map.py` exists (v3.2) and `tests/test_splat_fused_map_example.py` passes 3/3.
- `render_fused` accepts in-memory point arrays (`point_positions`/`point_colors`), so `--dem` runs show all three representations derived from the DEM.

## Phases

### Phase 0 — environment (done)
- Branch created: `feature/splat-fused-exploration`.
- Confirmed splat-fusion native path available and end-to-end render works with fixtures.
- Verified determinism: same seed → byte-identical PNG.

### Phase 1 — example script (`examples/splat_fused_map.py`) (done, v2)
A single CLI-driven script that renders the fused scene as a cartographic map sheet:
- Scene args: `--splats` (3DGS .ply), `--pointcloud` (COPC/LAZ), `--terrain` (GeoTIFF), `--dem` (fuji/swiss/rainier/luxembourg preset — fetched from the web on first run, cached in `~/.forge3d/datasets/`), `--crop-size`, `--splat-count/--splat-lift/--splat-offset/--splat-scale`, `--no-lidar`.
- Camera rig: `--view canonical|oblique|high|top|low` presets plus `--azimuth/--elevation/--distance/--target/--fov` orbit controls (raw `--origin/--look-at` override). Default = canonical fixture camera; default view for `--dem` is the full-bleed `top` preset (cover-fit: the largest distance whose projected ground quad still contains every frame corner, falling back to the frame fit).
- Render args: `--width/--height/--samples/--seed/--exposure/--sky-intensity/--aovs/--out`.
- Default run is offline: fixture ridge + 3,500 `cloud.ply` splats + 5,500-return COPC swath at the fixture's canonical camera.
- `--dem fuji` builds one coherent scene from a single real DEM: **y-up world centred on the origin**, elevation converted to the *same* world units as the grid (1 unit = 1 cell = 7 m), so slopes keep their real steepness — squashing z to 0..10 flattened Fuji to ~4% grades and path-traced relief turned to mush (the texture was in the DEM all along, `N·L` just could not see it). Hypsometric `albedo_map`, cartographic NW sun (az 215 / el 45, sky 0.6), and a splat **scan block derived from the DEM itself**: a circle on the summit flank of terrain-normal-oriented, lifted, semi-translucent violet splats (grid step ≈ 2 cells, sigma 0.9·step, opacity 0.45) — so every layer is geographically coherent and both representations take hits in roughly their area ratio.
- Output: annotated map sheet (title, legend with per-layer hit counts, SCENE/CAMERA/RENDER info blocks, scale bar in real metres, north arrow, footer), beauty PNG, `report.json`.
- Deterministic: byte-identical rerun verified for both modes (fixture and `--dem fuji`, sha256 a/b match on beauty + sheet).

### Phase 2 — real web data (done)
- Direct download from media.githubusercontent.com (no pooch needed) with local cache in `~/.forge3d/datasets/`.
- Tested with Mount Fuji DEM (2594×2594, cropped to 256×256 from centre).
- Mt St Helens COPC preset dropped: it loads via `las` but never produced a usable fused render here (IO failures / coordinate mismatch with any bundled terrain); `--pointcloud` remains for coherent user-supplied pairs.

### Phase 3 — visualization polish (done, v2)
- Full map-sheet composer: title + subtitle, legend with per-layer hit counts, SCENE/CAMERA/RENDER blocks (panel text wrapping), scale bar (nice 1/2/5×10ⁿ lengths with `:.3g` labels, real metres from the DEM geotransform or 1 unit = 1 m for the fixture), north arrow (projected world −z), footer, sidecar `report.json`.
- `--aovs` adds hit-kind / depth / sun-transmittance insets beside the beauty.
- Hypsometric palette tuned for contrast across the full elevation range (green → tan → rock → cool snow).

### Phase 3b — hero revision (done, v3)
- `--dem` default view is now `hero`: low cinematic oblique (az 35, el 10, fov 55) solved by `_hero_distance` so the frame's bottom ray lands inside the terrain while the summit keeps sky above it; aim lifted to 0.16 × relief so the summit sits ~20% below the top edge. `--view top` keeps the north-up map view.
- Full DEM extent by default (`--crop-size` default 0): true mountain proportions and a horizon-to-horizon landscape; `--crop-size N` still zooms to a central square.
- Realism defaults for the hero shot: warm low side light (az 125 / el 15, intensity 6, warm colour), bright Hosek-Wilkie sky (sky 2.5, turbidity 4, ground albedo 0.5 → the below-horizon band reads as distant haze instead of a dark slab); all overridable (`--sun-color`, `--sky-turbidity`, `--sky-ground-albedo`).
- Hypsometric greens darkened to forest tones; the derived scan block is a cleaner saturated violet (0.10 ground-tint bleed) with 6,000 finer splats in hero view (σ 2.7 vs 4.5).
- Block radius clamped to 130 cells so a full-extent DEM gets a survey-sized block, not one scaled to the mountain.
- The sheet's RENDER block dropped the wall-clock time row, so the "seed N reproduces this sheet byte-for-byte" footer is literally true (beauty and sheet sha256-verified identical across reruns).
- v3.1 (motion + cartography): plan views get a projected elevation-contour overlay (marching squares, index lines every fifth; annotation on the sheet copy only, beauty untouched; panel note drops instead of overlapping when readouts grow). `--animate N` renders a deterministic orbit fly-through GIF (sha256-verified identical across reruns). `--tile W H` exposed so large stills (1280×896 @128 spp) fit the 512 MiB budget via seamless tiling.

### Phase 3c — LiDAR from arrays + golden-hour sky (done, v3.2)
- Engine gap closed: the fused path ingested points only from files (COPC /
  page stores), so a DEM run could not show a LiDAR layer at all. New
  `PointArraySource` in `src/splat/stream.rs` mirrors `SplatCloudSource`
  (k-d `paginate`, identical `PageSource` contract, `PagePayload::Points`,
  linear-white default colour) over in-memory arrays; `render_fused` accepts
  `point_positions=(N,3) float32` world-space y-up and `point_colors=(N,3)`
  uint8 sRGB (the albedo-map convention). No new integrator behaviour: the
  arrays are paged through the same residency pool and unified
  `T_splat × T_lidar × T_terrain` as a streamed COPC cloud. Other
  `build_scene` callers (sequence, reference) pass `None`.
- Validation both sides: the Python wrapper rejects bad shapes, non-finite
  positions, colour/position pairing and non-uint8 colours *before* any GPU
  touch; the Rust boundary re-checks (defence in depth).
- Example: `_derive_lidar_swath` digitizes an airborne scan from the DEM
  grid itself — flight lines with sqrt(area/N) pulse spacing, 1.45× line
  spacing, deterministic jitter, slope-driven intensity colouring (sand →
  terracotta, the legend swatch family) and a pulse-sized sphelet radius
  (0.7 × along-spacing) so returns overlap into a scan strip. The strip
  underlaps the violet splat block, so all three representations share a
  region. Provenance is stated on the sheet ("derived scan swath - N pts, M
  flight lines"); `--no-lidar` drops it, `--pointcloud FILE` replaces it.
- Golden-hour hero default (probed at 640×448/32 spp, P1 chosen over a
  deeper el-5 variant): sun el 8, warm colour (1.0, 0.85, 0.62), turbidity
  5, ground albedo 0.35, sky intensity 3.0 — the Hosek sky grades cream →
  blue instead of a flat pale band, and the low sun reads the relief.
- Determinism re-verified end to end with the swath in scene: two full runs
  produce sha256-identical beauty *and* sheet.
- Gates: Rust unit test (`point_array_source_pages_an_in_memory_swath…`,
  run with `PYO3_PYTHON=.venv/bin/python cargo test --features
  default,splat-fusion point_array_source --lib` → 1 passed), new
  diagnostics + GPU byte-identity test in `tests/test_splat_api.py`, and
  `tests/test_splat_api.py + test_splat_fused_map_example.py +
  test_api_contracts.py` → 196 passed. `cargo check` clean (only
  pre-existing `fetch_update` deprecation warnings in untouched core files);
  `cargo forge3d-clippy` fails repo-wide on a newer local clippy
  (146 pre-existing `chunks_exact`/deprecation lints in untouched files, 0
  in the files changed here).

### Phase 3d — HDRI sky + aerial-perspective fog (done, v4)
- Engine gap closed (the "no HDRI" limitation below): `render_fused` and
  `render_fused_sequence` accept `env_map` (a `.hdr/.rgbe/.exr` path) +
  `env_rotation_deg`. `FusedRenderDesc.env_map: Option<Arc<FusedEnvMap>>`
  carries a pre-loaded equirect radiance; when set it replaces the
  Hosek-Wilkie bake in `render_fused_view` (the kernel already samples
  whatever equirect is uploaded, so **no shader change was needed** for the
  env swap itself). Loader `env_from_hdr_image` (`render_fused.rs`) reuses
  the `IBL.from_hdr` format dispatch, box-averages to ≤1024 texels wide,
  rotates about +y, normalises exposure to the **median** upper-hemisphere
  luminance (the sun disc is an outlier the mean would chase) and clamps at
  12× so the HDRI's own sun reads as a bright blob instead of an IBL
  firefly. Certificate now records `fused.env_map` (path or
  `hosek-wilkie`) alongside `fused.fog_*`.
- Real aerial perspective (the "ugly background" fix): the fused fog was
  shadow-ray-only, so distant surfaces ended at a hard silhouette. Added
  `fusion_media_along(origin, dir, dist)` in `unified_occlusion.wgsl`
  (the closed-form height-fog integral, refactored out of
  `fusion_media_transmittance`, which now calls it — bit-identical) and a
  `FUSED_TERRAIN_EDITS` anchor at the beauty composition so
  `L = L·T + env·(1−T)` with `T` the transmittance to the primary hit. At
  fog density 0 it is exactly the base line; verified bit-for-bit by the
  unchanged fixture path.
- Example: `--hdr auto|none|PATH` (auto = fetch the CC0 Poly Haven
  *Kloofendal 48d Partly Cloudy (pure sky)* 2k HDRI once into
  `~/.forge3d/datasets/hdri/`, md5-checked; `none`/fixture keep Hosek),
  `--hdr-rotation`, `--fog` (default 4e-4), `--fog-falloff` (default
  1.5e-3). Hero lighting moved golden-hour → **lit daytime** (sun el 42,
  near-neutral colour, sky 1.4). Fog is gated **off** on the `top` plan
  view: every ray there points down through the whole camera altitude, so
  the same density would only mute the hypsometric edges.
- Verified on the RTX 3070: hero A/B (`--hdr none --fog 0` vs default) —
  the analytic sky is a flat band with a hard horizon, the HDRI+ fog sky is
  a real partly-cloudy dome with ridges receding into haze; plan view stays
  crisp; two full hero runs sha256-identical (`d34c9750…`). Gates: 3
  `env_map` + 7 `fused_kernel` Rust tests, `test_splat_api.py` +
  `test_splat_fused_map_example.py` (32 passed), native-signature +
  validation checks.

### Phase 4 — regression gate (done)
- `tests/test_splat_fused_map_example.py`, 3 tests:
  - `test_example_module_loads`: module import + parser exists.
  - `test_parser_defaults`: default arg values (800×560, samples 64, seed 7, no dem/pointcloud/splats, `--view` default None).
  - `test_default_run_nonblank_and_deterministic`: runs the example twice at 8 spp, asserts non-blank beauty, byte-identical rerun hash, report structure, non-zero surface hits.
- 3/3 pass with GPU + Pillow (~8 s); on runners without PIL the full-run test skips cleanly while the parser tests still pass.

### Phase 5 — docs (done)
- New **"Map-sheet example"** section in `docs/splat-fused.md` linking `examples/splat_fused_map.py` with its three invocation modes and camera controls.
- `.gitignore` negation exceptions added (`!examples/splat_fused_map.py`, `!tests/test_splat_fused_map_example.py`) so both files are tracked, not local-only.

## Acceptance criteria

1. `python examples/splat_fused_map.py` runs offline with defaults and writes a PNG sheet. ✅
2. `python examples/splat_fused_map.py --dem fuji` uses real web-fetched terrain in a coherent scene. ✅
3. Hit-kind panel + legend show ≥2 surface kinds (terrain + splat). ✅ (fixture adds LiDAR + sky = all four)
4. Rerun with same seed produces byte-identical PNGs. ✅ (both modes, sha256 verified)
5. Tests pass on GPU lane; skip cleanly without GPU/PIL. ✅ (3/3)
6. Camera is controllable and reported: presets + orbit overrides, CAMERA block on the sheet. ✅

## Known limitations

- No public remote splat `.ply` exists; users supply their own 3DGS `.ply` via `--splats`.
- Mt St Helens COPC and Mount Fuji DEM are in different coordinate systems, so a
  real-file multi-layer run still needs user-supplied matching data. `--dem`
  runs instead derive every layer from the one DEM (heightfield, splat block,
  LiDAR swath) — geographically coherent, but the splat block and the swath
  are synthetic digitizations, not real captures.
- ~~The fused sky is the analytic Hosek-Wilkie model only: `render_fused` has no
  HDRI/env-map input~~ — **resolved in Phase 3d**: `render_fused` /
  `render_fused_sequence` take `env_map` + `env_rotation_deg` and the kernel
  samples the uploaded equirect. Hosek-Wilkie remains the offline default.
- Physical water (WaterOptics) + fused path not yet wired together.
- The DEM scan block is derived synthetic data (a demo of the fused pipeline), not a real 3DGS capture; pass `--splats` with a real capture to replace it.
