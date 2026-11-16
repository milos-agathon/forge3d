# Studio × Forge3D — implementable integration plan (ponytail edition)

Status: **plan**, not code. Phases 0–7 plus the HDRI sky are implemented;
everything below is future work. This revision replaces the earlier 8–18
roadmap: same coverage, far fewer moving parts, every increment sized to land
against the **current** `studio-tauri` with the files that already exist.

The rule applied throughout: use the ladder, stop at the first rung that holds.
Most Forge3D features are *settings on the render request we already send*, not
new subsystems. The plan only spends real design on the two things that cannot
be a payload field: vector layers and meshes.

---

## 0. Delivery status

| Area | State |
|---|---|
| Project model, commands, undo, schema v4 | done |
| Solid / Material / Rendered viewport, HDRI sky | done |
| Terrain stills, Rendered frames, animation export | done |
| Animation tab (timeline, keyframes, camera-path recording), render queue, asset integrity, samples | done |
| Sun from date/time/location (increment 9.0) | done |
| AOV + OIDN denoise final stills (increment 10.0) | done |
| Denoise runtime actually works (oidn+tbb in `.venv`, 1.4 commit order, TBB preload) | done |
| Denoise runs in linear space (sRGB decode → OIDN → encode) | done |
| GPU selection prefers discrete adapters + adapter name in the status bar | done |
| Viewport asks WebGL for the high-performance adapter | done |
| One light rig: scene sun exactly, no fill light, shared sky/environment (Solid/Material/Rendered) | done |
| Still uses the physically-based terrain model that matches the preview's BRDF | done |
| Studio HDR is a valid Radiance file (biased RGBE exponents, sky above ground) | done |
| Engine backend/adapter in the probe (first half of 8.0) | done |
| HDRI/sky axis fix (Z-up terrain → Y-up environment frame for the sky pass) | done |
| Properties sidebar (Render / Scene / Lighting / Object tabs, N, remembered sections) | done |
| Animation tab (Render / Scene / Lighting / Object / **Animation**) — the timeline moved out of the bottom strip into the sidebar | done |
| About Forge3D Studio (File menu): shell version, engine + adapter, device features, license, keyboard shortcuts | done |
| Fly camera (W A S D, Q E, drag-look, Shift to sprint) and camera-path recording onto the timeline | done |
| Captured camera keeps its own orientation (it used to be re-aimed at the world origin) | done |
| 5.5 shadows — filter (PCF/PCSS) and map size wired model → payload → engine | done (softness/intensity are inert in the terrain path; not exposed) |
| 5.2 water — level + slope threshold, mask derived with the engine's own `derive_water_mask` | done (optics deferred) |
| 5.4 bloom + vignette — service-side post-process on the finished frame | done (engine-native needs a pinned-hash refresh) |
| 5.3 clouds — the engine's cloud shadows exist, but only `render_interactive` consumes them | **blocked, measured** (the interactive path's shader contract is stale) |
| One lighting rig shared by preview and payload (fixture read by Rust and node) | done |
| 8.0b — granted device features in the probe/status bar | done |
| 5.1 sky & atmosphere — model, turbidity, ground albedo, exposure; AETHER gated on the shipped bake | done (backdrop-only by design) |
| 5.6 colormap — the engine's own catalogue through `Colormap1D`/`OverlayLayer` | done (needs `albedo_mode="mix"`; one LUT slot per frame) |
| 5.7 per-DEM vertical scale — 6% relief target, derived on import and DEM replacement | done |
| 5.8 terrain sun shadows — `SunVisibilitySettings` ray march, reach + steps | done (one hard ray; soft/resolution dials measured and skipped) |
| 5.9 despeckle — the land-cover gallery's 3×3 class rule on class overlays | done |
| Material layers (snow/rock/wetness) | **blocked, measured** (`vs_main` writes a constant `world_normal`; slope ≈ 0) |
| Tonemap operators/LUT, SSAO/SSGI/SSR, planar reflections | **inert on this path, measured 0** (see 5.8) |
| Ambient occlusion control | **removed** — 0 pixels changed, 2.0 s per frame (5.10) |
| Engine fidelity — Hosek–Wilkie evaluates green-dominant where the CPU radiance table is blue | open (measured; Preetham is the blue option) |
| 5.12 elevation legend — the engine's `Legend` in the frame's corner | done (needs a colormap) |
| 5.14 overlay texture follows the source (4096 texels) | done |
| 5.16 the elevation ramp in the Material preview (engine stops) | done |
| 5.17 map plate export (white page, depth cut-out, title, legend) | done |
| 5.18 the map look, in one undoable button | done |
| 5.19 plate furniture: scale bar and north arrow | done |
| 5.20 the categorical legend (class maps snap back to the palette) | done |
| 5.21 the interactive path's stale shader contract | repaired (measured; not a swap) |
| 5.22 vector layers — the rail network on the terrain | done (lines; labels next) |
| 5.23 vector labels — the engine's own text compositor | done (labels; furniture next) |
| 5.24 the cartography workspace — the map as a layer stack | done (the stack; the engine's MapScene next) |
| 5.25 the quality surface: the still on the engine's CONCORDIA path | done (the clouds are real; the GPU cartography next) |
| 5.26 the gallery's starter scenes, and the plate's furniture | done |
| 5.27 the engine's GPU cartography | done (the lines and labels in the terrain's pass) |
| Release audit as a CI gate (`tests/test_studio_surface_audit.py`, fast lane) | done |
| 5.15 vector overlays (rail network) | **blocked, measured**: the offscreen path has no vector geometry (0 changed samples) |
| Cartography: map plate | **blocked** (Pro-gated: `_check_pro_access`), and a class legend needs a class scheme |
| Vector layers (§7.1), meshes in Rendered (§7.2) | design increments, trigger-gated |

**The coverage review** — what Forge3D can do against what the Studio brings
of it, with the gaps and their costs — is `docs/feature-coverage.md`. It also
carries the stage and the phase plan (14–17).

The earlier roadmap's Phases 8–18 were a capability catalogue. They are mapped
to decisions in §7 and the appendix; most of them are now explicitly **not
built** until a trigger fires.

---

## 1. What already exists in studio-tauri (the map)

| Layer | File | What it already does |
|---|---|---|
| Model | `crates/studio-core/src/project.rs` | `RenderSettings`, asset kinds, validation, schema |
| Model | `crates/studio-core/src/command.rs` | `SetRenderSettings`, `AddAsset`, `Batch` — reuse these |
| Model | `crates/studio-core/src/workspace.rs` | sun/camera handoff, `sun_angles`, `camera_render_view` |
| Shell | `src-tauri/src/forge3d.rs` | engine probe (`EngineStatus`), terrain sampling |
| Shell | `src-tauri/src/render.rs` | `RenderRequest`, `with_settings`, `to_payload` |
| Shell | `src-tauri/src/studio_render.py` | one engine session, `TerrainRenderer`, IBL, sky |
| Shell | `src-tauri/src/commands.rs` | IPC, `render_frame`/`render_still`/`render_animation_frame` |
| Bridge | `src/bridge/types.ts` | `RenderSettings`, `EngineStatus`, command union |
| UI | `src/components/Inspector.tsx` | render settings + capability badges |
| UI | `src/components/StatusBar.tsx` | engine probe status |
| UI | `src/state/session.tsx` | render queue, playhead, actions |
| Tests | `studio-core/tests/*`, `render.rs` tests, `npm test` | cheap, no GPU for most |

## 2. The reusable recipe for a renderer setting

Every "look" feature follows this exact diff. No new command, no new IPC, no
new schema version (optional scalars with `serde(default)` are backward-safe,
as `environment` proved).

1. `project.rs`: add the field to `RenderSettings` with `#[serde(default…)]`;
   add its range check to `Project::validate`.
2. `types.ts`: add the field to `RenderSettings` and the default.
3. `render.rs`: copy it in `with_settings`, emit it in `to_payload`.
4. `studio_render.py`: map it to the engine dataclass in `render()`.
5. `Inspector.tsx`: one control (`NumberField`/`Toggle`/`RangeField`).
6. Tests: one schema-contract line, one payload assertion, one round-trip test.

Five places, one test each, no architecture. The
sections below list only what differs from this recipe.

---

## 3. Increment 8.0 — Engine facts in the existing probe (0.5 day)

**Trigger:** always; later gating depends on it.
**Reuse:** `forge3d.rs::probe_interpreter` + `EngineStatus`; extend its JSON.

- `PROBE_SCRIPT`: one extra line, `forge3d.capabilities()` plus
  `device_probe()` adapter/backend into the same JSON.
- `EngineStatus`: add `features: Vec<String>`, `backend: Option<String>`,
  `adapter: Option<String>`. TS type mirrors it.
- `StatusBar`: the existing title tooltip lists the adapter and granted
  features. No new panel.
- Do **not** add a capability framework, license UI or per-control registry.
  Gated controls appear only when a Pro-gated feature is actually built.

**Check:** a probe JSON fixture; status bar shows features on hover.

## 4. Increment 9.0 — Sun from date/time/location (0.5 day)

**Trigger:** always; highest value per line.
**Reuse:** the sun light object, `SetTransform`, `workspace::sun_angles`.

- Shell command `solar_position(latitude, longitude, datetime)` → Python
  `forge3d.sun_position(lat, lon, datetime)`. One command, no project fields.
- Core helper `sun_transform(azimuth_deg, elevation_deg) -> Transform` (the
  inverse of `sun_angles`), with a round-trip test.
- Inspector (light): four inputs (lat, lon, date-time) + **Apply** → compute →
  `setTransform` on the sun. The existing light transform persists the result;
  the date/lat/lon are not stored (add fields only if users ask for re-edit).
- **Check:** round-trip `sun_transform → sun_angles` within 0.5°.

## 5. Look increments — one setting group per commit

All use §2. Nothing new is introduced beyond the settings below.

### 5.1 Sky model and atmosphere (1 day) — **done**
- Shipped: `skyModel` (`hosek` | `preetham` | `aether`), `skyTurbidity`
  (1–10), `skyGroundAlbedo` (0–1), `skyExposure` (0–4) on `RenderSettings`
  (serde-default, v4-safe); payload `quality.sky*`; service maps to
  `SkySettings` and keeps the "environment ⇒ hdri" branch byte-identical
  (measured: the HDRI sample still hashes `848cea2e6fdcc450` at
  `render_scale=1`; since the viewport and the stills supersample at
  `render_scale=2` — the terrain's per-cell normals alias into sparkles at
  close camera ranges — the current anchor is `716555b1dd1a4cfd`, mean 97.8,
  and any change to `render_scale` moves it).
- The analytic branch is enabled again when no HDRI is set, so Rendered has a
  sky instead of the neutral surround.
- AETHER is offered only at the shipped bake's parameters (turbidity 2.0,
  ground albedo 0.3); any other pair fails loudly in the service.
- Measured: turbidity 2→8 moves the frame 11.6 (mean abs), sky exposure
  0.05→0.10 moves it 33.8; no-HDRI sky mean 163.
- **Open engine item:** Hosek–Wilkie reads green-dominant (75/88/85 at
  exposure 0.02) where the CPU radiance table is blue and Preetham renders
  blue; it is also ~7× brighter at the same exposure. The Studio exposes the
  model instead of hiding the difference.

### 5.2 Water (0.5 day) — **done**
- Settings: `water.enabled`, `water.level`, `water.slopeThreshold` (a
  `WaterSettings` group on `RenderSettings`).
- Service: the terrain path has **no water-level knob** — it shades water where
  a *mask* says so, and `params.water` is inert there (measured: 0 pixels
  changed). The service therefore derives the mask with the engine's own
  `forge3d.gis.derive_water_mask` (low ground below the level, flat enough to
  hold it) and passes it as the `water_mask` argument. Measured: a mask moves
  1,422–14,373 pixels on the Bryce DEM depending on the slope threshold; a
  zero mask moves 0, so the mask — not the plumbing — is what draws the water.
- The level is in the DEM's metres and is scaled with the heights into the
  renderer's units, so the control means the same thing on a 4 km amphitheatre
  and a 350 km country. Level 0 = auto (the engine's 15th percentile).
- Shipped since: the water surface (`WaterOpticsSettings(mode="physical",
  roughness=…)`) with its own height and normals, plus a roughness control —
  measured .35 → .95 changes 8,415 samples on a 480×300 frame, and the surface
  is what removed the per-cell glints (302 → 2 sparkle pixels in a 1180×330
  frame).
- Deferred: mask paths and planar reflections (`ReflectionSettings` measured 0
  changed samples on this path).
- Check: schema contract, payload assertion, four service tests, round-trip.

**Fog tint is conditional, and the UI says so.** The shader scatters the sky
colour whenever the sky pass is on — `select(fog_uniforms.fog_inscatter.rgb,
sky_color, sky_enabled)` — so the R/G/B tint is inert with the sky on
(measured 0 samples) and live with it off (30,127 samples on a 480×300 frame).
The tint fields now appear only when the sky background is off, with a hint
saying where the haze colour comes from otherwise; the density and reach dials
work in both states (reach 0.5 → 0.15 changed 92,457 samples).

### 5.3 Clouds (0.5 day) — **blocked, measured**
- Settings: `clouds.enabled`, `coverage`, `density`, `shadowStrength`.
- Service: `CloudSettings(enabled, shadows_enabled=…, coverage, density,
  shadow_strength, quality="medium")`.
- **A cloud shadow is not a post-process**, so the service-side route that
  shipped bloom and vignette cannot deliver it.
- The engine's cloud shadows are real — `ConcordiaFrame::configure_clouds`
  reads `coverage`, `density` and `shadow_strength` and drives
  `CloudShadowRenderer` — but they live in **`render_interactive`**, and
  `render_terrain_pbr_pom` takes no `clouds` argument at all. Measured:
  `CloudSettings(enabled, shadows_enabled, coverage=0.8, density=0.8,
  shadow_strength=0.6)` on the request changes **0 pixels**.
- `render_interactive` is itself **broken today**, for an unrelated reason:
  its shader assembly does `replace_once(source, "let shadow_z = h_curved *
  h_exag;", …)`, but the terrain shader now says `let shadow_z =
  det_barrier((h_curved - 0.5) * h_range) * h_exag;` — the TERRA-DETERMINATA
  rewrite changed the spelling. The path fails with `CONCORDIA shader contract:
  expected one "let shadow_z = h_curved * h_exag;", found 0` before it renders
  anything. The fix is in Rust (not a shader source, so no pinned-hash change),
  but there may be more stale contracts behind it.
- Adopting `render_interactive` for Studio would also move the render path
  (shared-depth HDR + the CONCORDIA surface), which changes the measured
  anchors — the HDRI sample hash `848cea2e6fdcc450` and the still/preview
  fidelity work.
- Trigger: someone needs clouds in a Studio render, and is willing to take the
  engine-side fix (repair the interactive path's contract, then adopt
  `render_interactive` with the anchors re-measured).

### 5.4 Post: bloom + lens vignette (0.5 day) — **done (service-side)**
- Settings: `bloom.enabled`, `threshold`, `softness`, `intensity`; `vignette`
  0–1. A `BloomSettings` group plus one scalar on `RenderSettings`.
- **Not the engine's path.** The engine's GPU bloom (`TerrainBloomProcessor`)
  and its vignette (in the viewer's terrain shader) live in render paths the
  Studio does not use: the terrain path Studio renders through has neither, and
  both were **measured to change 0 pixels** there. The engine-native wiring
  would mean an HDR target in the terrain path and a change to
  `terrain_pbr_pom.wgsl`, which is in `shader_sources::terrain_parts()` —
  that refreshes `PINNED_TERRAIN_SOURCE_HASH` (owner approval) and moves the
  determinism anchors.
- Shipped instead as a **service-side post-process** on the finished frame,
  next to the denoise: bloom is a bright-pass plus a separable box blur, the
  vignette a smooth radial falloff. Both are *display-referred* — the terrain
  shader has already tone-mapped the frame — so the bloom threshold is a
  display brightness (0.8), not the engine's HDR 1.5.
- Scope: the Rendered preview and the still show them; the Solid and Material
  previews do not, like AOVs and denoise. AOV files are written from the raw
  render, so they never carry the glow.
- Measured on the Bryce DEM: bloom off + vignette 0 vs the dry frame is
  **0 pixels changed** (byte-identical default); bloom on changes 760,908
  channel samples and lifts the brightest pixel 171 → 225; a 0.9 vignette takes
  the corner 159.7 → 16.7 and leaves the centre at 138.7.
- Defer DoF, motion blur, TAA: costly and rarely wanted.
- Check: schema contract, payload assertion, six service tests.

### 5.5 Shadows (0.5 day) — **partly done**
- Shipped: `shadowTechnique` (`pcf` | `pcss`) and `shadowResolution`
  (256–8192) on `RenderSettings`, through `to_payload` → `ShadowSettings`;
  verified on the GPU (PCSS and a 512² map change the frame).
- Not shipped: `shadowSoftness` and `ShadowSettings.intensity` are decoded but
  **inert** in `render_terrain_pbr_pom` (measured: softness 0.05 vs 8.0 and
  intensity 0 vs 1 produce identical pixels), so exposing them would be a
  silent no-op. The shader has to consume them first.
- Deferred: `cascades` (the terrain path keeps the engine's 3).

### 5.6 Colormap (0.5 day) — **done**
- Settings: `colormap` (engine name, `""` = none) and `colormapStrength`
  (0–1) on `RenderSettings`; payload `quality.colormap` /
  `quality.colormapStrength`.
- The engine ships the catalogue: `forge3d.available_colormaps()` is
  `forge3d:magma`, `forge3d:vik`, `forge3d:viridis`. `get_colormap` returns a
  `Colormap`; the terrain takes a `Colormap1D` built with `from_stops` over the
  elevation domain, wrapped in `OverlayLayer.from_colormap1d`. No engine code
  was added.
- **The colormap does nothing under `albedo_mode="material"`.** The shader's
  material branch assigns `material_albedo` last, so a colormap overlay
  measured **0** changed samples against the material set; `mix` measures
  162,541 and viridis vs magma 163,680 (Bryce, 640×400). The service
  therefore sends `albedo_mode="mix"` (material → ramp by strength) when a
  colormap is selected, and keeps `"material"` otherwise — a `mix` frame with
  no overlay would apply the engine's *fallback* LUT to the whole terrain.
- Strength 0 is off, not "a 0% ramp": a bound colormap at strength 0 still
  normalized the height the material layers sample by its domain, which moved
  162,905 samples (mean 4.4/255) against the same frame with no overlay. At 0
  the service binds no colormap, so strength 0 is byte-identical to the
  material set (measured: 0 changed samples).
- One LUT slot per frame: `extract_overlay_binding` binds the first overlay
  that has a LUT and stops, so a raster overlay and a colormap cannot both
  apply. The service **raises** instead of silently dropping one, and the
  Inspector disables the colormap while the terrain has a raster overlay
  (the selected value stays reachable so it can be cleared).
- Measured on the Swiss DEM at the gallery camera (922×795, zScale 1.0):
  material → viridis changes 2,189,640 samples (mean 34.1). The ramp renders,
  but at true scale the Alps are 1.1% relief — that is the vertical-scale
  default below, not a colormap defect.
- Check: schema contract + fixture, payload assertion, three service tests
  (built from the catalogue, strength 0 off, overlay conflict), and the
  byte-identical no-colormap still (`9f8f5c3c2f1ae4be` before and after).

### 5.7 A per-DEM vertical scale (0.5 day) — **done**
- `zScale` used to be `1.0` for every DEM. That is right for the bundled
  Bryce sample (454 m of relief across 4 km, 11.3%) and wrong for a
  country-size DEM: the Swiss Alps are 4,031 m across 352 km — 1.1% — so at
  true scale they render as a printed sheet.
- `scene::dem_default_z_scale` targets **6% relief**: `clamp(0.06 ×
  max_span / relief, 1.0, 50.0)`. It only ever exaggerates, so a DEM that
  already reads as terrain keeps its true scale: Bryce stays `1.0`, the Swiss
  becomes `5.246` (`0.06 × 352,427 / 4,031`). No footprint or no relief
  (a local DEM, a flat one) stays at `1.0` rather than guessing.
- Applied where a DEM is attached (`TerrainObject::with_dem_info`) and where
  one is replaced (`SetTerrainDem`, which re-derives unless the command names
  a scale; its undo carries the previous value, so undo restores `5.246`).
- The engine's range is the model's range: `TerrainRenderParams` raises
  `z_scale must be 0.1-50.0` instead of clamping, but `SetTerrainZScale`
  accepted any positive number (and the Inspector offered 0.05, then 20 as a
  maximum). Both now use `MIN_Z_SCALE`/`MAX_Z_SCALE` and the command rejects
  an out-of-range value with an error instead of producing a project the
  editor cannot validate.
- Measured on the Swiss DEM at the gallery camera (922×795, `render_scale`
  2): `1.0 → 5.246` changes 1,234,577 of 2,198,970 channel samples (mean
  10.16, frame mean 113.2 → 106.1). The relief is visible at 5.246 and the
  zScale-1/2.5/5.25/10 sheet in `/tmp/opencode/zs` shows why 5.25 was chosen
  over 10 (jagged) and 2.5 (still flat).
- Still open in the cartography tier: legend, despeckle, and the map plate.

### 5.8 Terrain sun shadows (0.5 day) — **done**
- Settings: `sunVisibility`, `sunVisibilityDistance` (metres, 100–20000) and
  `sunVisibilitySteps` (8–64) on `RenderSettings`; payload
  `quality.sunVisibility { enabled, distance, steps }`.
- The engine's `SunVisibilitySettings` ray-marches the heightfield toward the
  sun per pixel, so a ridge shadows the ground behind it — terrain
  self-shadowing the shadow map cannot resolve because it samples a different
  projection. Measured on the Bryce DEM (922×795, sun 18°): off → on changes
  395,389 of 2,198,970 channel samples (mean 3.47), and the still shows the
  valley shadows the shadow map leaves flat.
- Two dials, both measured: `distance` 2000 → 8000 m changes 38,304 samples
  (a short reach only catches nearby ridges), `steps` 32 → 16 changes 14,734
  (fewer steps miss a ridge line sooner).
- Not shipped: `mode="soft"` with 8 jittered samples moved 490 samples and
  `softness` 0.1 vs 0.5 moved 491 — a sub-pixel edge — so the service sends one
  hard ray. `resolution_scale` 0.5 moved 231 samples against 1.0, so the pass
  runs full-resolution.
- Not final-gated: one ray march per pixel is cheap enough for the Rendered
  viewport, so the shadows are visible while authoring.
- Measured inert on this path, all 0 samples: `ScreenSpaceSettings` SSAO,
  SSGI and SSR; `ReflectionSettings` (planar water reflections); every
  `TonemapSettings` operator (`reinhard`, `reinhard_extended`, `aces`,
  `uncharted2`, `exposure`) and white balance. The **existing** ambient
  occlusion control was in the same class and is deleted (see 5.10). The material-layer system is
  **blocked**: `vs_main` writes a constant `world_normal` (`(0,0,1)`), so the
  layer slope is ~0 — a rock layer at 0.001° changes 160,785 samples but at 1°
  changes **0**, even from a grazing camera (98,488 → 0). Altitude-driven snow
  would draw, but on cliffs too, so it is not shipped.
- Check: schema contract + fixture, payload assertion, one service test.

### 5.9 Despeckle a class overlay (0.25 day) — **done**
- `TerrainOverlay.despeckle` (serde-default `false`) → `overlay.despeckle` in the
  render request. The service runs the gallery land-cover recipe's own 3×3
  rule on a **single-band class raster** before paletting it: a class holding
  at most 2 cells of a 3×3 block (the cell included) is a speck, and it takes
  the block's majority when that majority holds at least 4 cells. Two passes.
  Class 0 (nodata) is never moved, so the raster's mask survives. An RGB
  raster is untouched — a drape is imagery, not classes.
- Measured on a synthetic 600² class raster (trees/crops/snow plus 6%
  salt-and-pepper), Swiss DEM at 640×400: off → on changes **55,383** of
  768,000 channel samples (mean 2.17), and the still goes from a speckled
  drape to clean class fields.
- The Inspector shows the toggle with the overlay; the hint says what it does
  and that an RGB raster ignores it.
- Check: schema/default test (phase4), payload assertion, service test on the
  rule (a lone cell absorbed, a 3×3 block kept, a clean raster unchanged).

### 5.10 Ambient occlusion — **removed (dead, and expensive)**
- The `ambientOcclusion` checkbox (labelled "final, ~5 s") was shipped when the
  still used the engine's *stylized* shading model. Studio's still now uses
  `lambert_physical`, and that branch applies `shadow_visibility` and the sun
  visibility but **no AO at all** (`terrain_pbr_shadows_...`: "Do not apply the
  stylized shadow or AO floors").
- Measured on the Bryce DEM (640×400, `render_scale` 2): off → on changes **0**
  of 768,000 channel samples, twice in a row (the two runs are byte-identical),
  while the frame goes from 22–23 ms to 1,987–2,017 ms. An expensive no-op.
- Deleted end to end: the field on `RenderSettings`, the request/payload keys,
  the service's `ao_weight`/`heightAo`, the Inspector section, and the wire
  fixture. Loading an existing project that still carries the key is fine —
  serde ignores the unknown field — and it renders exactly what it rendered
  before, two seconds faster.

### 5.11 The release audit, run after 5.10 — **green, one range fixed**
- **Every `RenderSettings` field reaches the payload.** 26 fields; each is read
  in `with_settings`/`to_payload` or, for `width`/`height`, is the request's
  frame size.
- **Payload and service agree in both directions.** Every key the service
  reads out of `quality` is a key `to_payload` sends, and every key
  `to_payload` sends is read by the service (the nested blocks included:
  `water`, `fog`, `bloom`, `sunVisibility`, `aov`). Checked by extracting both
  sets from the source, not by eye.
- **UI ranges sit inside the renderer's range.** One mismatch found and fixed:
  Width/Height offered 16384 while the render service caps a frame at 4096 px
  per side (the viewport already caps there too), so a larger still was being
  silently shrunk. The rest check out: vertical scale 0.1–50 (model and
  command too), sun reach 100–20000 m, ray steps 16–64 (engine `< 64`), sky
  turbidity 1–10, sky exposure 0–4, shadow map 512–4096 (engine `<= 8192`),
  colormap strength 0–1, overlay opacity 0–1, fog density `>= 0`.
- **The wiring is proven end to end**, not only in unit tests: an ignored GPU
  test (`the_new_settings_reach_the_pixels`) renders one frame twice (identical
  bytes), then with a colormap and with sun visibility (different bytes), so a
  key-name drift between the Rust payload and the service fails a test.
- **Preview supersampling is now on every preview.** The payload used to ask
  for `render_scale` 2 only up to 4 Mpx of *requested* frame; the viewport
  capped the request at 4096 per side, so a 4× viewport (3688×3180) landed at
  scale 1 — the exact configuration that shows the terrain's per-cell aliasing.
  The threshold is now the internal target (8 Mpx requested → 16 Mpx at 2×) and
  the viewport caps at 2048, which the engine renders at 4096 and blits down:
  a 4× viewport frame went from 11.7 Mpx of IPC at scale 1 to 3.6 Mpx at
  scale 2, while the engine still draws 4096 wide.
- **No client-side re-implementation.** Supersampling is `render_scale`; the
  class palette is `thematic.apply_palette`; the colormap is
  `Colormap1D`/`OverlayLayer.from_colormap1d`; the camera is
  `camera_render_view`; water, fog, shadows, sun visibility, sky and IBL are
  the engine's settings. Shell-side code is limited to what the engine has no
  consumer for on this path, and each case says why in code: bloom/vignette
  (needs an HDR target and a pinned-shader change), the water blob filter and
  the class despeckle (raster rules, not renderer mechanisms).

### 5.12 Cartography has a home: the Map tab — **done**
- Relief, the elevation ramp (with preview strips for `viridis`/`magma`/`vik`
  and the DEM's min/max metres), the legend and the draped imagery moved out of
  Object → Terrain into a **Map** tab of its own. Nothing about them changed
  except where they live: from the Object tab they were unfindable, which is
  what "where is the cartography" meant.
- The Relief section shows the DEM's **relief as a percentage of its
  footprint** (the number the rule is about: Bryce 11.3%, the Swiss 1.1%) and
  warns under 3% that 1.0 renders a plate.
- `Command::FitTerrainRelief` re-derives the DEM's default vertical scale in
  the core (`scene::dem_default_z_scale`) — one undoable command, no copy of
  the 0.06 rule in the shell. A project saved before the default (zScale 1.0)
  adopts it in one click; undo returns to 1.0.

### 5.13 Elevation legend (0.25 day) — **done**
- `RenderSettings.legend` (serde-default `false`) → `quality.legend`. When a
  colormap is selected, the service composes the engine's own
  `forge3d.legend.Legend` (gradient bar, five ticks in the DEM's metres, title)
  into the frame's bottom-right corner with an alpha blend: the same object the
  engine's cartography uses, drawn with its packaged text.
- The legend scales with the frame (a viewport preview gets a compact bar
  without the title, a 4K still the full card) and shrinks until it fits; a
  frame too small for a readable legend gets none rather than a smudge.
- Measured on the Swiss DEM (922×795, colormaps on): legend off → on changes
  138,501 of 2,198,970 channel samples, **all of them inside the legend patch**
  (68,844 there, 0 elsewhere); a 480×300 preview changes 31,179 of 432,000.
- The toggle lives beside the colormap strength, and only appears with a
  colormap — the legend explains a ramp, so without one there is nothing to
  label.
- **Not shipped: the map plate.** `forge3d.map_plate.MapPlate` (title, legend,
  scale bar, insets, north arrow) calls `_check_pro_access("Map plate
  composition")`, so it raises `LicenseError` without a Pro licence — and a
  categorical legend still needs a class scheme the project does not store.
- Check: schema contract + fixture, payload assertion, service test (the
  compositor leaves the top-left untouched and the shape unchanged).

### 5.14 The drape keeps the raster's own resolution (0.25 day) — **done**
- The overlay texture was read at the **frame's** size, so a 10 m land-cover
  map became 390 m blocks on a 900 px frame and coarser on a small one. The
  texture now follows the source, capped on its long side at the engine's
  4096 overlay ceiling (`OVERLAY_TEXTURE_LONG_SIDE`), so a close-up keeps the
  class pattern and a whole-country view is as crisp as the source allows.
- Measured on the Swiss land cover (10055×22069 → 1866×4096 texels): the 900 px
  frame's drape went from the averaged blocks to the raster's own class
  boundaries; a source under 4096 keeps its pixels untouched.
- **Cost, measured:** a national 10 m raster is ~2.6 GB of float32 across its
  bands. Reading it whole took **19.4 s per frame** because the drape was
  rebuilt every request. GDAL now reads the strips with `ALL_CPUS` (17.2 s →
  3.5 s for the three masked bands) and the decoded RGBA is cached by file
  identity (`overlay_key`: path, mtime, size, despeckle; the opacity is applied
  per request), so the first frame with a new overlay costs 8.3 s including the
  engine's warm-up and every later frame 280 ms — with the opacity slider
  reusing the same pixels.
- Check: service tests on the sizing rule and on the decode-once cache, 32
  service tests green.

### 5.15 Vector overlays (the rail-network gallery example) — **blocked, measured**
- The gallery's Luxembourg rail image is a **viewer** snapshot: the lines reach
  the renderer through `add_vector_overlay` IPC and are drawn by
  `src/viewer/terrain/vector_overlay.rs` (geometry, drape, depth test, halo).
- Studio's stills render through `TerrainRenderParams` → the offscreen terrain
  pass, which has **no vector geometry at all**: `VectorOverlaySettings` only
  carries `depth_test`/halo/contour flags, and a `VectorOverlayConfig` passed
  through the params is ignored. Measured on the Luxembourg DEM (640×480):
  depth/halo settings **0** changed samples, a `VectorOverlayConfig` triangle
  with `drape=True` also **0**.
- Two honest routes, both real work: (a) add an overlay pass to the offscreen
  terrain renderer (vertex/index buffers, drape from the height texture, depth
  test, halo — the viewer's implementation is the reference), or (b) render
  vector-bearing projects through the viewer/`MapScene` compositor, which is
  §7.1 and moves the anchors. Until one lands, Studio does not offer a vector
  layer control: it would draw nothing.
- Data is ready: `assets/gpkg/luxembourg_rail.gpkg` (2031 line features,
  EPSG:4326, same CRS as `assets/tif/luxembourg_dem.tif`), readable with the
  stdlib (`sqlite3` + GeoPackage WKB) if the engine side lands first.

### 5.16 The cartography in the Material preview (0.5 day) — **done**
- Material mode is the fast, lit viewport; it showed the draped raster but not
  the **elevation ramp**, so choosing a colormap changed only the Rendered
  frame — the map was invisible while authoring it.
- The ramp now paints the preview mesh: the shell fetches the engine's own
  stops (`colormapStops` → `forge3d.get_colormap(name).rgba`, 32 of the 256
  texels, linear RGB) and writes them as vertex colours, mixed with the clay
  by the colormap strength exactly as the renderer's `mix` albedo mode does.
  Solid stays clay; Material shows the drape when one is attached, otherwise
  the ramp.
- **The stops are the engine's, not a hand copy.** The first attempt used five
  hand-written hex stops; measured against the engine's LUT they were off by
  up to **74/255** (viridis), 84 (magma) and 189 (vik) — a preview that lies
  about the map. The new bridge command removes the copy: the panel strip and
  the mesh both read `forge3d.get_colormap`, and a node test pins the
  interpolation, the clamp, the strength mix and the display encoding.
- Check: `colormap.test.ts` in `npm test`, `tsc`, the vite build.

### 5.17 Map plate export (0.5 day) — **done**
- The gallery's Swiss image is not a viewer feature: its white page, title,
  categorical legend, caption and cut-out come from the example's own
  `compose_snapshot` (PIL). Only `forge3d.MapPlate` is Pro-gated, so a plate is
  a legitimate shell-side composite — and it is the single thing that made the
  gallery look like a printed map instead of a terrain floating in a sky.
- `RenderSettings.plate` + `plateTitle` → `quality.plate` (finals only, and
  asking for one asks for the depth AOV). The service cuts the frame out with
  the **renderer's depth buffer** (`aov.depth`, background pixels carry zero
  depth — never a colour key, so a dark valley survives), crops to that
  silhouette, lays it on white under the title, an elevation legend when a
  colormap is set, and a caption, and writes `<still>_plate.png`.
- The page follows the gallery's layout: title centred on top, legend top-left,
  subject scaled into the middle, caption along the bottom; the type is the
  engine's packaged Noto Sans Latin, not PIL's bitmap default.
- Measured (Swiss DEM, 1200², whole country at 5.246, hillshade on, land cover
  draped): the plate is a 968×951 page with the country cut out at its land
  cover border — the frame itself is a full rectangle, so the cut-out is what
  turns it into a map. Without a depth mask the service returns the frame
  unchanged rather than inventing a silhouette.
- Not yet on the plate: the **categorical** legend (the class palette is known
  but the names live in the gallery recipe), and a scale bar/north arrow.
- Check: payload test (final-only, depth AOV, preview off), service test (crop
  to the mask, white page, subject pixels preserved, no mask = no plate).

### 5.18 The map look, in one button (0.5 day) — **done**
- Cartography setup was four places (fit relief, a camera, a sun, a hillshade)
  and the numbers were in the gallery's recipe, not the app. `ApplyMapLook` does
  all of it in one undoable command: the DEM's own relief, the map camera
  (from the south, 31° off vertical — north up — framed on the whole footprint),
  the gallery's
  north-west sun (315°, 24°) and the ray-marched hillshade (4 km reach, 48
  steps). The Inspector's Map tab has the button; the viewport switches to the
  camera so the framing is what you see.
- The camera needed the *inverse* of `compute_render_view`, and the renderer
  keeps a fixed ten-unit orbit radius, so the eye has to stand at that radius
  and the **field of view** is what decides how much fits (61.93° for a
  country-sized DEM; the project's own field of view is kept when it is
  wider). `workspace::map_camera_transform` returns both, and a round-trip test
  pins it: the transform's orbit reads back as the requested azimuth and polar
  angle, and the still's visible width covers the terrain's long side.
- Measured on the Swiss DEM: the country fills the frame the way the gallery's
  does (1200×900, 5.246, hillshade on, land cover draped), instead of a
  rectangle floating in the sky.
- Check: core tests for the round trip, the sun direction and the command
  (relief 5.246, hillshade on, camera moved and widened, sun moved, undo
  restores all three), and the TS wire shape.

### 5.19 Map furniture: scale bar and north arrow (0.25 day) — **done**
- The plate gets the two pieces of furniture a map needs and that need no
  guesses: a **scale bar** (the ground scale at the frame's centre, snapped to
  a round 1/2/5 length — "50 km" on the Swiss plate) and a **north arrow**.
- Both are drawn only where they are true: the scale bar needs a near-vertical
  view (an oblique frame has no single ground scale). The north arrow is the
  engine's own and turns with the map (the engine's `north` camera mode reads
  the azimuth from east towards north, so the north-up view stands at phi
  270°, and the arrow's rotation is clockwise from north on the page).
- The scale comes from the camera the frame was rendered with: the visible
  width at the target (2 · radius · tan(fov/2), unscaled to metres) over the
  frame's width.
- A unit test covers the round-length rule: the first version of the loop
  multiplied by 1.0 and never terminated — it hung the render, and the test
  now pins it.

### 5.20 The categorical legend (0.5 day) — **done**
- The plate's legend is the *classification* legend when the drape is a class
  map: the palette's swatches and the class names that palette implies (Water,
  Trees, Flooded vegetation, Crops, Built area, Bare ground, Snow, Rangeland),
  listed for the classes the map actually shows.
- A land-cover export is usually an **image of the classes**, not class IDs —
  the bundled `switzerland_land_cover.tif` is 3-band. The service now snaps such
  a raster back to the palette (`snap_to_classes`), the same rule the gallery's
  `snap_overlay_to_legend` uses, so the drape wears the palette's exact
  colours and the plate can name the classes.
- The snap only applies to a **class map**: at least 80% of the valid pixels
  have to sit close to one of the *named* classes. An orthophoto is not a class
  map and is left alone — measured: the Bryce NAIP raster snaps to nothing, the
  Swiss land cover snaps to classes 1–8 (plus the grey spare).
- Check: service tests for the snap (palette colours snap, a grey image does
  not, nodata stays nodata) and for the legend's shape; 36 service tests green.

### 5.21 The interactive path's stale shader contract — **repaired (measured)**
- `render_interactive` (the shared-depth HDR pipeline behind clouds, vector
  batches, labels and points) was dead on arrival: CONCORDIA assembles the
  terrain shader by replacing five known fragments, and the TERRA-DETERMINATA
  rewrite changed one of them. The contract named `let shadow_z = h_curved *
  h_exag;`; the shader now says `let shadow_z = det_barrier((h_curved - 0.5) *
  h_range) * h_exag;`, so the count was 0 and the assembly failed.
- The repair names the *current* spelling and keeps CONCORDIA's own
  domain-centred, range-scaled shadow height. Measured after the repair (the
  extension rebuilt locally with `--features extension-module`):
  `render_interactive` renders a 320×240 frame where it previously raised.
- **But it is not a drop-in.** The interactive path is a different pipeline: on
  the same scene it differs from Studio's current path in **221,837 of
  230,400** channel samples (mean 33.18). Adopting it would move every
  measured anchor and change every picture. And clouds still draw nothing
  through it (0 changed samples) with the settings alone.
- So the contract fix unblocks a *capability*, not a swap: the interactive path
  is now available for measurement (and for MapScene, which already calls it),
  but Studio's stills stay on the mesh path until the difference is understood.

### 5.22 Vector layers — the rail network draws (1 day) — **done**
- The engine already had every piece, scattered: `TerrainRenderParams.
  project_terrain_points` projects terrain UV + height through the exact mesh
  camera, `forge3d.vector_render_oit_py` rasterizes anti-aliased lines, and
  MapScene composites the result. The tier is those three composed, not a new
  renderer: the service reads the file, maps the lines to the terrain's UV,
  samples the height along them, projects, rasterizes and composites.
- **The data is read without a third-party package**: a GeoPackage is a SQLite
  database with GeoPackage binary geometry, so `sqlite3` plus a WKB parser is
  enough (the studio's environment has no geopandas, fiona or pyproj —
  verified). GeoJSON is plain JSON. The CRSs must agree; a mismatch fails
  loudly instead of landing in the wrong place. The Luxembourg rail file
  (2,035 features, 780 km of track, EPSG:4326) matches its DEM, so nothing is
  reprojected.
- **The lines land on the ground**, at the frame's own scale. Measured on the
  Luxembourg DEM (900×700): radius 97,500 m → 15,496 changed samples; 60,000 m
  → 24,397 (the earlier figure with the OIT bug unfixed); 40,000 m → 30,448.
  The plate shows the network with the "Vector layer" legend entry, the north
  arrow and the scale bar.
- **An engine bug fell out of the measurement**: the OIT bridge packed one
  instance per *segment* but drew one per *polyline*, so a densified path lost
  all but its first few segments (1,610 samples instead of 24,397). Fixed in
  `src/py_functions/vector/render.rs`; the shell also hands over one polyline
  per segment, which is MapScene's own payload convention.
- **The width is a symbol width in pixels, not metres — measured, not
  assumed.** The first cut was the feature's real width in the DEM's metres,
  converted with the frame's ground scale. It is inert: at 900 px across a
  65 km DEM one pixel spans 62 m, and a rail is a few metres wide, so 40 m and
  10 m rails render **identically — 0 changed samples, mean difference 0.0**
  (the conversion's one-pixel floor was doing all the work). A cartographic
  symbol width is the honest unit: **1–20 px, default 2**, because the
  rasterizer's own floor is one pixel — 0.5 px and 1 px render identically
  (0 changed samples, mean 0.0), so the control starts at 1 instead of
  offering an inert half. Measured at 900×700 / 60 km: width 2 → 6 px changes
  40,875 samples (mean 1.197); 1 → 2 px changes 21,103 (mean 0.327); orange →
  blue changes 32,952 (mean 2.17); no layer → layer changes 32,928 (mean
  1.14). The superseded metres field and the ground-scale helper are gone
  rather than kept beside the pixel width.
- The layer is a terrain field (`TerrainVectorLayer`: asset, RGBA colour,
  symbol width in pixels), one undoable command (`SetTerrainVector`), a payload
  key (`vector`) and a Map-tab section.
- **Still lacking**: furniture, polygons and a legend for the *vector*
  attributes; the layer is lines plus labels, one layer, drawn over the
  terrain rather than occluded by its ridges.
- Check: the reader (a real GeoPackage), the vector legend, the stacked plate
  legends, the command's undo and its pixel-width range rejections, the
  payload and wire shape, and an end-to-end engine test that the layer reaches
  the pixels (`the_vector_layer_reaches_the_pixels`).

### 5.23 Vector labels — the engine's own text compositor (1 day) — **done**
- **The labels are the engine's, not the shell's.** The engine already had the
  whole cartography: `LabelPlan.compile` places the labels, resolves their
  collisions, shapes their glyphs against the packaged MSDF atlas, and the
  native text compositor draws them with the halo and the rasterizer. The
  studio supplies the features and the anchors its own camera projected — and
  nothing else. Laying out glyphs in the shell would have been a second
  implementation, so the engine's compositor was made public instead:
  `forge3d.composite_label_layers(base, layers, plans)` draws a compiled plan
  over an image the caller owns, with the same glyph stream `MapScene.render`
  uses (the private path now calls it, so the two cannot drift).
- **The anchors come from the engine's camera.** A named point becomes the
  terrain's UV, the height is sampled there, and
  `TerrainRenderParams.project_terrain_points` projects it through the exact
  camera the frame was rendered with, so a label sits on its feature. An
  anchor over the sky — where the engine's terrain depth says there is no
  ground — is dropped.
- **Measured** (Mount Fuji 30 m DEM, 900×700, 30 km radius, `name.en`): 28
  named features → 28 anchors → the engine's plan accepts **10** (13 rejected
  `priority_lost`, 5 `collision`). Labels off → on: **30,261** changed samples
  (mean 1.367). Label colour dark → blue: **20,114** (mean 0.375). Label size
  14 → 24 px: **78,312** (mean 2.722). Label field `name.en` → `name` (the
  file's Japanese names): **30,261** — the labels vanish, because the engine
  rejects 37 of them as `missing_glyph` and 2 as `shaping_failed`: the
  packaged atlas is Basic Latin. A field the file does not have: **30,261**
  (no points, no labels).
- **A layer that cannot draw says so.** When the labels are asked for and the
  engine accepts none, the service reports it (`the vector labels drew
  nothing: 39 features named, none accepted (missing_glyph 37, shaping_failed
  2)`), and the shell's reader now logs the warning events the protocol
  already had — a skipped overlay or water layer was being dropped on the
  floor before.
- **Labels are now occluded by the terrain's ridges** (the studio's own
  depth test, the same rule the lines use): an anchor is dropped where the
  ground's depth is in front of it, with the local depth gradient as the
  bias so an on-surface anchor does not flicker. The sky test is unchanged
  and the annotation placements use the same rule. Still lacking: the
  engine's solver runs with a zero margin, so dense labels may touch, and
  the packaged atlas is Latin only — a CJK place name needs a baked atlas.
- Check: the reader (the named points, a missing field), the engine's plan
  (the accepted label, its bounds, a rejected name), the model's label
  defaults and refusals, the payload, and an end-to-end engine test that the
  labels reach the pixels (`the_vector_labels_reach_the_pixels`).

### 5.24 The cartography workspace — the map as a layer stack (2 days) — **done**
- **The map's decisions had three homes**: the terrain object's draped raster
  and its vector layer, and the render settings' colormap and plate. A map is
  not three things; it is a stack. The studio now has one: `Project::cartography`,
  an ordered list of layers (relief → elevation ramp → imagery → vector layers
  → label layers → plate), each with a kind, a name, an eye and its own style.
  The v4→v5 migration moves the old fields into layers, so a reopened map
  looks the way it was left.
- **The Studio has a Cartography workspace**: a Photoshop-like layer panel on
  the left (the eye, a swatch of what the layer draws, the name, the kind, and
  the draw order), the map in the middle, and the selected layer's properties
  on the right. The TopBar switches between the scene and the cartography.
- **The renderer draws the stack back to front**, and the stack is the only
  thing the render request carries: `cartography` replaces the request's
  `overlay`, `vector`, `colormap`, `legend`, `plate` and `plateTitle` keys.
  Two layers of the same kind now stack (two vector layers, two label layers),
  and a hidden layer is simply not drawn.
- **Measured** (Mount Fuji 30 m DEM, 900×700, 12 km radius): the relief alone
  → the relief plus the elevation ramp: **1,680,769** changed samples (mean
  42.772); the ramp's eye off → back to the relief (1,680,769); the full stack
  (relief + ramp + vector + labels): **1,681,037** (mean 44.062); the label
  layer's eye off: **55,547** (mean 2.403).
- **The engine's one terrain-overlay slot is now the stack's business**: the
  elevation ramp and the draped imagery both need it, so the uppermost visible
  one wins and the service *reports* the other ("the elevation ramp is hidden
  under the imagery: the engine composites one terrain overlay per frame")
  instead of dropping it silently.
- **Still open**: the layer stack is the studio's own model, not MapScene's
  recipe — the shapes now match, so the convergence is a mapping rather than a
  rewrite, but it is not done. Furniture, polygons and a legend for the
  vector's attributes are still the rest of the tier.
- Check: the model's own tests (add, remove, move, edit, undo, the ranges),
  the migration (a v4 document becomes a stack), the payload (the stack, with
  every layer's own settings), the wire shapes, and the end-to-end engine
  tests that the ramp, the vector layer and the labels reach the pixels.

### 5.25 The quality surface: the still on the engine's CONCORDIA path (2 days) — **done**
- **The still rendered on the plain path** (`render_terrain_pbr_pom`), so the
  engine's whole screen-space tier — SSAO, SSGI, SSR, the cloud shadows, the
  volumetrics, the atmospheres — was unreachable from the studio, and the
  clouds measured **0 changed samples**. The still now renders through the
  engine's quality surface (`render_interactive`), and the frame's reply says
  which path made it.
- **The quality path is equal-resolution**: the engine refuses to supersample
  on it (`CONCORDIA equal-resolution TAA requires render_scale=1; TAAU is
  unavailable`), so the temporal anti-aliasing does the work the plain path's
  2× supersampling did. A still averages four frames; the measured edge
  energy is **2.496** against the plain 2× still's **2.582** — the quality
  path is the smoother of the two. The cost is 0.45 s against 0.30 s for the
  900×700 frame.
- **Two engine gaps had to be closed first**, and both are the engine's:
  - the clouds multiplied the *stylized* lighting path's shadow only, so the
    physical model (OD-9c — the one the studio renders with) never saw them:
    **0 changed samples**. The CONCORDIA specialization now rewrites the
    physical path's visibility too: the clouds change **368,996 samples**
    (mean 0.498, max 16) on the studio's own Mount Fuji frame;
  - the cloud field's world scale was the terrain's span, so the same clouds
    looked different at every scene size and vanished on a country-sized DEM
    (the flat-plane probe: 446,523 changed samples at a 256 m span, 12,607 at
    an 18.7 km span). The pattern is now an absolute 2 km period.
- **The AOV/denoise finals stay on the plain path**: the engine's AOV renderer
  has no quality surface, so a still that exports AOVs cannot carry the
  screen-space effects. The path is reported in the frame, so the studio says
  which one made the picture.
- **Still open**: the studio's cartography is composited in numpy on top of the
  quality surface (the OIT rasterizer + the label compositor); the engine's
  own GPU cartography (`render_interactive`'s label/vector/furniture batches)
  would draw them in the same pass, with the terrain's occlusion — that is the
  next increment, and it wants the MapScene convergence under it.
- Check: the engine's source contracts (the physical path's cloud visibility,
  the absolute world scale), the engine's GPU measurement (the clouds reach the
  physical model), the studio's payload and its e2e test (`the_clouds_reach_the_pixels`),
  and the service's own clouds settings.

### 5.26 The gallery's starter scenes, and the plate's furniture (2 days) — **done**
- **The New Project dialog now offers the gallery's recipes**, not just the
  bare DEMs: **Swiss land-cover** (the national DEM with the land-cover
  classification draped on it, the class legend, the title, the caption and the
  north arrow — the gallery's 03), **Luxembourg rail network** (the DEM with
  the 2,035 rail features draped in red — the gallery's 04), and **Mount Fuji
  labels** (the cone with the climbing stations named — the gallery's 02).
  Each recipe is data: the files it draws, the layers it builds, the camera
  direction and the sun. The whole recipe is **one undo step**, and a file the
  checkout does not have is skipped rather than faked.
- **The map's own settings live in the Cartography workspace.** The Map tab is
  the cartography's properties (the layer stack's), so it is hidden in the scene
  workspace and pinned in the cartography one.
- **The furniture belongs to the plate layer.** The north arrow, the scale bar
  and the legend were drawn unconditionally (the arrow only where the view is
  north-up, the bar only where the ground scale is known); they are now the
  plate layer's own switches, and the plate has a caption of its own. Measured
  on the Swiss plate: the north arrow **663 changed samples**, the scale bar
  **2,292**, the legend turns the page from 750×683 into 900×915 (the legend
  column), the whole furniture **2,955**. The land cover itself: **1,176,000**
  changed samples (mean 28.6). (The arrow's own rotation and the map look's
  azimuth turned out to be wrong for the engine's `north` camera mode — see
  5.28.)
- The plate's caption is drawn with the packaged Latin atlas, which has no
  middot: the recipes use a plain dash.
- Check: the recipes' own tests (the layers, the camera, the sun; the label
  layer naming the vector layer), the plate's furniture switches in the service,
  and the three scenes rendered end to end.

### 5.27 The engine's GPU cartography (2 days) — **done**
- **The map's cartography was composited in numpy over the frame**, so the
  ground never occluded it: a label floated over the mountain in front of it,
  and the lines were drawn on top of the ridges that hid them. The engine
  already draws both in the terrain's own pass — `render_interactive` takes
  the label and vector batches and the shared cartographic depth — so the
  studio hands its map over instead of compositing it itself.
- **The engine gained one public entry for it**: `forge3d.label_batches(layers,
  plans, width, height)` is the public twin of the compositor, the glyph
  stream `render_interactive` takes (the public surface is 383 names now). The
  studio's vector layers become the engine's vector batches: the OIT coverage
  plus the terrain's own depth raster (the lines are draped on it).
- **Measured** (Mount Fuji 30 m DEM, 900×700): the labels drawn by the engine
  in the terrain's pass change **69,770 samples** (mean 3.205) against the
  plain terrain; the engine's label batches carry 230 glyph rects for the
  scene's 16 accepted labels.
- **The quality surface is temporally filtered**, and the still is its
  converged frame: the temporal filter folds the jittered samples into its
  history, so the edge energy falls **3.063 → 2.558 → 2.295** over three
  frames against the plain path's 2.582 — the quality path is the smoother of
  the two, and the frame the studio keeps is the converged one. The frame's
  reply says so (`temporal`), and the byte-stability claim moved to the plain
  path (the AOV/denoise finals), which still is.
- **Still open**: the lines' own depth. The engine's own vector path computes
  the per-pixel depth of the projected geometry (so a line behind a ridge is
  hidden by it); the studio's coverage comes from the OIT rasterizer, which
  returns no depth, so the studio's lines are depth-tested against the
  terrain's depth — the terrain's own — and are not occluded by a ridge in
  front of them. Closing it is either an OIT bridge that also returns the
  depth or the MapScene convergence below. **Closed by 5.28.**
- Check: the engine's public batch builder, the studio's cartography
  preparation (the batches' shape, the depth, the glyph rects), the e2e tests
  (the labels and the clouds reach the pixels), and the determinism claim
  moved to the path that keeps it.

### 5.28 The lines' own depth, the engine's furniture, the graticule (2 days) — **done**
- **The lines now carry their own depth.** The engine's vector path computes
  the per-pixel depth of the projected geometry; the studio's coverage came
  from the OIT rasterizer, which returns no depth, so a line drew *through*
  the ridge in front of it. The engine's rule is now public:
  `forge3d.vector.projected_depth(coverage, paths, width, height)` returns the
  nearest segment's interpolated depth and the sub-pixel offset from it. The
  engine's own recipe path (`_map_scene_native_inputs`) calls it too, so there
  is **one** rule and not two.
- **The occlusion is the engine's own test**: a line is hidden where its depth
  is behind the ground's depth at that pixel, with the bias the engine uses
  (the terrain's depth gradient across the pixel times the line's sub-pixel
  offset), so a line that lies *on* the ground does not flicker against it.
  The CONCORDIA overlay shader still reads a constant depth (the authored
  depth is uploaded but unused), so the studio applies the engine's Python
  rule to the coverage it drew — once, so the plain path's composite and the
  engine's GPU path draw the same line.
- **Measured** (Mount Fuji 30 m DEM, 900×700, the quality surface): a line
  across the cone changes **12,822 samples** (mean 0.518) against the plain
  terrain, and **2,147 of the line's 4,400 covered pixels are behind the
  ridge** — the line stops at the ground.
- **The plate's furniture is the engine's**: `forge3d.scale_bar.ScaleBar`
  (the graduated, geodetic bar and its label) and
  `forge3d.north_arrow.NorthArrow` (the arrow geometry and its "N"), not the
  studio's rectangle and triangle. The arrow turns with the map: the engine's
  `north` camera mode reads the azimuth from east towards north (measured with
  the engine's own `project_terrain_points`: at phi 0/90/180/270 the north edge
  lands right/down/left/up), so the north-up view stands at phi = 270 and the
  arrow's rotation is `(phiDeg + 90) % 360`. The studio's own map look used to
  stand at phi = 90 — the map upside down with a south-up plate — and now
  stands at 270.
  **Measured** (the Swiss plate, 900×700, phi 270): the arrow and the bar
  change **4,908 samples** (mean 0.190).
- **The graticule is a map layer**, drawn by the engine:
  `forge3d.graticule.generate_graticule` makes the meridians and the parallels
  of the map's own bounds (in the DEM's CRS; a projected DEM's bounds are
  transformed with the engine's own `transform_bounds`) and says what each
  line is worth ("8 degE"). The lines are draped on the terrain and rasterized
  by the engine's OIT bridge, like every other line; the annotations are
  placed by the engine's label plan, on their own line, where it leaves the
  frame (the bottom edge for a meridian, the left edge for a parallel).
- **Measured** (the Swiss map, 900×700): the graticule at 1° changes
  **23,329 samples** (mean 0.657) against the plain terrain; the annotations
  alone change **3,834** (mean 0.147). Every control of the layer moves
  pixels: the interval (1° against 2°) **16,245**, the width (1 px against
  3 px) **26,262**, the colour **14,156**, the annotations **3,834**.
- The graticule's interval range (0.05°–30°) is inside the engine's (any
  positive interval); the studio refuses what the engine would refuse.
- Check: the engine's public depth rule (and the recipe path sharing it),
  the occlusion test, the engine's furniture, the graticule's geometry and
  its annotation anchors, and the studio-core layer's own ranges. Two bugs
  found while verifying, both fixed in this increment: the engine's polygon
  vector overlay on terrain referenced a deleted name (`distance` → the
  offsets' squared length), and the studio's map look stood at phi = 90 —
  the map upside down for the engine's `north` camera mode; it now stands at
  270.

### 5.29 MapScene — the first mapping: the still (0.5 day) — **done, step 4 remains**

- **The still always renders through the engine's `MapScene` recipe.** The hand-
  built `make_terrain_params_config` path is deleted; the studio composes a
  `TerrainSource` from the heights it already scaled into the view budget, an
  `OrbitCamera` from the request, a `LightingPreset` carrying the sun, the
  IBL's path/intensity and the studio's sky/fog/water/sun-visibility settings
  (passed as `LightingPreset.settings` mappings, not a second parameter list),
  and the recipe's own `OutputSpec`, then calls `scene.render()`. The engine's
  own `_shared_terrain_render_context` warms the temporal history exactly as
  the old default did (three frames), so the recipe's frame equals the old
  default byte-for-byte. AOVs and denoise keep the one-shot `render_with_aov`
  route, exactly as the old AOV/denoise finals did. The still-sidecar names
  `forge3d.mapscene.render`; `RenderSettings::render_via_mapscene` and the
  Inspector's MapScene toggle are gone (the still is the recipe, always).
- **Step 3 landed with the drapes carried on the recipe's overlay channel.**
  The recipe's settings channel is JSON-serializable, so native
  `OverlayLayer`s cannot ride inside `LightingPreset.settings`; and the
  recipe's native raster drape (`RasterOverlay`) reserves an estimated
  ~4.0 GiB of host-side temporaries for a DEM this size, which the engine's
  allocation budget refuses. The studio bakes the drapes into straight-alpha
  RGBA PNGs instead — the elevation ramp from the engine's own LUT at the
  DEM's resolution (the same 64-stop sampling the old in-shader overlay
  used), the imagery from the same cached RGBA the overlay used to decode —
  and hands the recipe `{"overlays": [{"path": png, "strength": …}]}`, which
  the engine re-creates through its own `OverlayLayer.from_raster_rgba`.
  One overlay per frame: imagery and ramp still compete for the slot, the
  stack's business unchanged.
- **The recipe produces a different image on the first attempt — fixed in
  step 2.** Measured on the Swiss DEM (900×700, phi 270): the recipe's alpine
  output was bright, low contrast, green-tinted, because the IBL's `.hdr` was
  being wired at the wrong `renderer_config` location and the recipe's IBL
  silently fell back to its neutral built-in sky. The correction was wiring
  the same path at the top level `settings["ibl"]["path"]`.
- **The plate composite works on the recipe path too.** The recipe's
  `OutputSpec(aovs=("depth",))` produces a native depth frame on the engine's
  `render_with_aov` path, and `MapScene.last_aov_frame()` hands it to the
  caller. The studio's plate path (`AovFrame.has_depth()` /
  `AovFrame.depth()`) consumes it unchanged; a still with a plate layer now
  composites the plate transparently in the recipe instead of dropping it with
  a warning. The reply carries `plate: true` and `plateSkippedFor: null` when
  the plate is composited, `plateSkippedFor: <name>` when the recipe failed to
  produce a depth frame (a warning is also printed; the plate file is not
  written).
- **Still AOVs work when requested:** `aov={albedo, normal, depth}` with
  `mapsceneStill` writes `<stem>_albedo.png` / `<stem>_normal.png` /
  `<stem>_depth.png` through the recipe's native capture — the same files the
  plain path writes from its capture. Everyone pays out of the same native
  flow; nothing crosses the Python boundary twice.
- **Step 2 — the recipe matches the default still byte-for-byte (done).** The
  recipe now receives everything the hand-built path passed it, through the
  recipe's own settings channel (no second code path):
  the IBL's .hdr path is wired at the TOP level of the lighting settings
  (`"ibl": {"path": ...}` — `_native_ibl_path` only reads the top-level key;
  before the fix the recipe silently fell back to its built-in neutral sky,
  which was the entire systematic colour-temperature difference), plus the
  studio's terrain span, math-light azimuth/elevation, the `clip` pair, the
  no-op camera token `mesh:zup:north:dem`, the `overlays: []`, and the PCF
  shadow block (`map_size 4096, cascades 1, moment_bias 5e-05, max_distance
  30000, slope_scale_bias 1e-04, normal_bias 1e-05`) forwarded through the
  renderer_config's raw shadows dict. The engine grew three small recipes in
  `_build_mapscene_terrain_params` to honour them: the `terrain_span` metadata
  override (the span the studio scaled the heights to — the recipe's
  diagonal heuristic taking `max(cols*sx, rows*sy)` disagreed by 4025 m on the
  Fuji DEM), the `light_azimuth_deg`/`light_elevation_deg` settings override
  (the recipe's `atan2`-decode cannot even reach the studio's 315° azimuth —
  the light would wrap below the horizon), and the `clip` sequence override.
  The recipe also warms its TAA history exactly as the default does — three
  `scene.render()` calls inside the engine's shared native render context
  (one renderer, cumulative temporal history), keeping the converged frame —
  and when AOVs or denoise are requested the recipe skips TAA entirely (the
  engine forbids the combination) and takes the same one-shot
  `render_with_aov` path the default falls back to.
- **Measured (Mount Fuji, Swiss DEM, 900×700): both byte-identical to the
  default still** (Fuji phi 68/theta 45, mean abs diff 0.0, 0/630 000 samples
  differ; Swiss phi 270/theta 10, same). The still-AOV PNGs are byte-identical
  as well (albedo/normal/depth md5s match), and the AOV/denoise path takes the
  same one-shot `render_with_aov` route as the default — no 3-frame warm-up,
  no TAA, exactly the default's behaviour in that case. The recipe costs ~3.2 s
  for a converged still vs ~0.6 s for the default (it renders the same passes
  three times plus the PNG write); the one-shot AOV captures are ~1.7 s.
- **Step 4 — the engine's own typed `RasterOverlay` layers take over the drape
  (done).** The imagery drape is a typed `f3d.RasterOverlay` layer handed to
  `MapScene(layers=…)` — the recipe's own UV drape composites it, from the
  PNG the studio already keeps in cache (its rgba is cached by file identity).
  The elevation ramp rides the engine's own colormap overlay: the recipe
  builds a `Colormap1D` from the project's LUT, and the terrain shader's
  `mix` blend combines it with the material albedo; the studio passes the
  LUT name through the recipe's own settings channel (`"colormap"`,
  `"albedo_mode": "mix"`, `"colormap_strength"`), with no second
  implementation in the shell. Both share
  the single colormap/raster slot the shader provides, so with an imagery
  visible the ramp is hidden exactly as before — the uppermost visible
  source wins, a warning is printed, and the stack keeps its rule.
- **The terrain raster drape now runs under the host budget.** The native
  drape path previously reserved an estimated ~4.0 GiB of host temporaries
  for the 3200×7294 Swiss DEM (6 full-float grids at once), which the
  allocation budget rightly refused. It now composites in row bands, so the
  peak host allocation is one full-width RGBA8 canvas plus a few dozen MiB
  of banded buffers. The Swiss land-cover drape renders end to end.
- **The baked-PNG channel is deleted.** The studio no longer rasterizes the
  ramp or the imagery into PNGs on its own; both go through the recipe's
  own overlay builds now. Verified end to end: the opaque footprint of
  the imagery is unchanged against the step-3 frame, the AOV finals are
  byte-identical to the pre-switch capture, the baseline still is
  byte-identical against the committed default, the ramp keeps the project's
  LUT with its legend, and a plate composites on the same path.
- **Phase 16 — the preview matches the still — begins here.** The first slice
  landed: the **terrain editor's Terrain section** now has a `Hillshade`
  checkbox that turns a live slope-angle relief bake
  (`bakeHillshade(preview)`) on and off the 3D viewport's terrain material
  (a raster multiplier consulted by `applyTerrainShading`), so the author
  sees relief without waiting for a still. The flag lives on
  `ViewportState` (serde-default on, restored by the schema contract
  unchanged), set via `setViewport(…)` — a single undoable command, no
  new field on the project root. The still renders with the engine's own
  shading regardless of the toggle (the engine decides its own specular
  response; the hillshade is a preview aid, not a hidden print change).
- **The second slice landed: the map's vector layers, their labels and the
  graticule, in the viewport** — the same layers the still rasterizes, draped
  on the same preview terrain, read through the still's own reader. A new
  `vector_geometry` IPC (a `-c` script that imports `studio_render.py` for
  `read_vector_features`/`graticule_geometry`, one implementation — never a
  second GeoPackage parser) emits feature lines and labelled points
  normalised to the preview's 0..1 lattice in the DEM's bounds; the DEM's
  bounds come from a rasterio header read, never a full raster pass (measured:
  Luxembourg rail, 2,035 lines ≈ 150k vertices — 1.3 s cold → 0.36 s with the
  header read; ≈ 990 KB of JSON over IPC per layer load, cached per stack
  edit). The lines draw as one `LineSegments2` per layer (fat lines at the
  layer's `widthPx`, draped at the mesh's relief with a clearance lift) and
  hide behind the terrain where the ground stands in front (`depthTest`), and
  are clipped to the DEM's footprint so out-of-frame stretches never float in
  space. The
  labels are HTML divs ride the camera projection, hidden behind the terrain
  by a raycast on camera move, dropped over the sky (behind-camera/outside
  view tests), with the engine's own halo rule inverted for readability;
  graticule annotations pin to the point that leaves the frame — lowest for a
  meridian, leftmost for a parallel, the same rule the still's anchors use —
  and take the engine's own 12 px grid size. Label/graticule anchors are
  sampled on the mesh surface like every other drape.
- **The plate's crop frame landed** — while a visible Plate layer exists, the
  viewport draws a dashed letterbox at the still's render aspect (project
  render settings) centred on the canvas, labelled `plate · W × H` — the
  frame the still will compose into. It is an aspect indicator, not the
  page: the plate compositor still cuts the subject silhouette onto the page
  in Python at render time, and no three.js code mimics it.
- **The third slice landed: the Material preview matches the still's terrain
  light and shadow rig.** The terrain's Material material is now a
  `MeshPhysicalMaterial` with `specularIntensity = 0` — the engine's
  `lambert_physical` terrain is diffuse only (albedo/π · nDotL · PCF visibility
  + env irradiance), and so is the preview's. The project's shadow settings
  now drive the preview's shadow rig instead of a hardcoded 1024 / PCF:
  `shadowResolution` sets the map size, `shadowTechnique` maps to `pcf`
  → `PCFShadowMap` / `pcss` → `PCFSoftShadowMap` (three has no blocker
  search; the softer PCF variant stands in for PCSS, documented),
  `slope_scale_bias` 1e-04 maps to a −1e-04 depth bias, and a world-space
  normal bias (0.05 at viewer scale) replaces the metre-scale 1e-05. The
  engine's ray-marched sun visibility (a ridge shadowing the ground behind
  it) has no three.js equivalent and stays a documented gap — the preview
  shows the PCF term only. Also open question closed: the preview's
  Material tooltip no longer claims "soft shadows"; the stale
  `render_scale`-supersampling comment and the Inspector's hillshade toggle
  (preview aid, not in the still) are fixed.
- **Phase 16 is done.** The still and the preview agree on the stack, the
  exposure, the tone-map, the lighting rig, the shadow rig and the map's
  overlay; the one unmatched term is the engine's ray-marched sun visibility,
  documented in `feature-coverage.md`.


### 5.30 The Phase 14 remainder: polygons, and labels occluded by ridges (2 days) — **done**

The cartographic tier's last two open items. **Labels behind ridges are
hidden**: the studio's own label anchors apply the lines' rule — an anchor
falls where the ground's depth is in front of it (`hidden_by_ground`-style,
with the local depth gradient as the bias), so a point on a back-facing
slope no longer draws its label on the ridge. The sky test is unchanged.

**Polygons are in as their own layer kind.** `CartographyLayerKind::Polygon`
(draped on the terrain through the engine's own polygon rasterizer,
`forge3d.vector_render_polygons_fill_py`, at the frame's size): the rings come
through the still's reader (GeoPackage binary or GeoJSON, with holes), the
same CRS contract as the vector layers (an EPSG mismatch fails the frame
with both codes named), the same sky and ridge rules (`forge3d.vector.
projected_depth` per-pixel depth from the boundary rings), and the same
numpy composite rule as the lines — drawn below the lines, above the
imagery/ramp. Measured on the Fuji DEM at 512² against the same frame
without the layer: **11,100 of 262,144 samples changed (4.23%), mean
|Δ| ≈ 110, hole open**. In the viewport they are a draped triangle mesh built from
the same rings (`polygonMesh.ts`, earcut via THREE.ShapeUtils, windings
normalised). The layer shows up in the stack, the Inspector and the
`Add layer` menu like the rest. Still open in the tier: a legend for the
vector attributes.

### 5.31 The vector-attribute legend (0.5 day) — **done**

A vector layer can now colour its features by one attribute
(`legendField`, e.g. `usage`): categories are ranked by frequency
(desc, name asc), mapped to the Okabe–Ito colour-blind-safe palette
(the engine ships no categorical LUT), with the tail beyond eight folding
into a grey "other". Lines draw per category through the engine's OIT
bridge — one pass per class — and the plate's legend prints one swatch per
class instead of one swatch per file. The viewport draws the same classes
with the same palette (the geometry IPC re-exports it), so authoring and
print match. Measured on the Luxembourg DEM at 512² with the rail network
drawn by `usage` (main/industrial/branch/tourism/test/siding + an
uncategorised bucket): **3,959 samples change vs the one-colour frame
(mean |Δ| 171.5)**; the plate composite carries the six swatches
(`plate: true`, no skip reason).

### 5.32 The preview's own sun-visibility bake (0.5 day) — **done**

Phase 16's last unmatched term: a ridge shadowing the ground behind it. The
still ray-marches every fragment; the preview cannot do that cheaply, so the
viewport bakes the same march once per (DEM, zScale, sun) grid as a
`DataTexture` and multiplies it into the terrain studio material's
direct-light irradiance via `onBeforeCompile` — the exact
`direct_radiance × terrain_visibility` term, no new UI. Opt-in via the
project's `render.sunVisibility` toggle (off by default, like the still's
bow of Occam), reuses `sunVisibilitySteps`, rebuilds on sun scrub via a
rounded key. Baked at grid resolution (64–4096 points per pass, ~100–200 ms
on the Swiss), so a sun scrub costs a few rebuilds per gesture, each a few
hundred ms. Node-tested on synthetic walls and spikes (sunlit side lit,
shadowed side dark, high sun lifting short shadows). Known limit: it is a
march of the **grid**, not per-fragment, so a sub-triangle ridge still
shows as a smooth transition rather than a hard edge; the toggle deletes
it entirely when off (no texture, no program patch).

### 5.33 The renderer presets as named Studio recipes (0.5 day) — **done**

The engine's five `forge3d.presets` lighting packages are one-click recipes
under Render → **Look presets** — `outdoor_sun`, `rainier_showcase`,
`rainier_relief`, `studio_pbr`, `toon_viz`. Each recipe emits one
`Batch` of commands (`SetRenderSettings` ± `SetLight`/`SetSunAngles` ±
`SetTerrainZScale`), so the whole look lands or reverts atomically in one
undo step; camera, DEM, layers and selection are untouched. Sun azimuth
and elevation use the engine's azimuth convention (east → south,
`atan2(z, x)`); `SetSunAngles` round-trips through the engine's own
`sun_angles` in both directions, unit-tested. The still honours the
project's zScale now — the render session keys on `(dem, zScale)` and the
service multiplies heights by it at construction — so the relief recipes
are real in the Rendered path, not only in the viewport. Honest deviations,
named in each recipe's tooltip: engine-frame sun direction vectors are
not ported; toon_viz's toon BRDF and hard shadows have no terrain-pipeline
equivalent (closest-honest PCF at the preset's map size instead); rainier's
camera moves and cascade/light-size/ambient-occlusion are not ported
(cascades and light size are not available on the terrain pipe). Measured
deltas on Mount Fuji at 512² against the starter look (|Δ|>8, changed
samples): **outdoor_sun 159,556** (exposure 1.0), **rainier_showcase
206,585** (sun 135°/25°, 4.0), **rainier_relief 66,613** (zScale 1.5),
**studio_pbr 257,284** (sun 6.0, IBL off, flat backdrop — the preset's atmosphere-off mapped to the sky toggle; 102,960 of those samples are the sky alone), **toon_viz 159,585** (IBL 0) — pinned by
the recipe conformance tests in `tests/test_studio_render_service.py`.

### 5.34 The stranger-installable deb, with honest engine discovery (0.5 day) — **done**

The deb never requires the engine at install time: the postinst names the
remedy in one breath (`pip install --user forge3d`, or set
`FORGE3D_STUDIO_PYTHON`) — no pip, no sudo, no engine bundled, no bare
python fallback. The shell discovers and *validates* its interpreter in a
fixed order: `FORGE3D_STUDIO_PYTHON` (import-checked, never trusted
blind) → ancestors' `.venv/bin/python` (repo checkouts) → `PATH python3` → `PATH python`,
each probe overriding none of the others. `--engine-check` prints every
candidate tried, one line each with its result, the verdict, and exits 0/1
— stranger- and CI-verifiable, starts no GUI. Verified on this machine in
four scenarios: env var set (exit 0), repo-cwd `.venv` fallback (exit 0),
a fresh `uv` venv with `pip install forge3d` (1.49.0 → exit 0), a bare
`PATH python3`/`PATH python` without forge3d (exit 1 with the remedy printed). The
desktop entry validates with `desktop-file-validate`; the deb's
`Depends` are `libgbm1, mesa-vulkan-drivers` merged with the bundler's
webkit/gtk defaults; the `.desktop` `Comment` no longer duplicates
`Name`. CI gains a `deb-audit` job (PR + main): builds the deb, `sudo
dpkg -i`, pre-installing the declared `Depends` so a wrong dependency
fails strictly, validates the desktop entry, and runs `--engine-check`
in both interpreter scenarios — **the gate is the probe, not a GPU
render**; the hosted runner has no terrain-safe GPU. On this machine a
real `dpkg -i` was executed inside a user-namespace stranger root (Debian
`dpkg_1.23.12`; declared Depends seeded in `var/lib/dpkg/status`; the
maintainer script ran for real; `ii` recorded; the installed bundle probed
the fresh `uv` venv's forge3d 1.49.0 from a repo-less cwd, exit 0). `rpm`/`appimage`
remain built but unaudited, documented as such in the README. When the
shell runs inside a repo checkout, the probe imports the checkout's
`python/forge3d` with the caller's `PYTHONPATH` kept as the tail
(unit-tested); in the installed bundle there is no checkout above
`/usr/bin`, so a stranger probes the interpreter alone.

## 6. Increment 10.0 — High-quality stills: AOV + denoise (1 day) — **done**

**Trigger:** a user asks for AOVs or cleaner stills.
**Shipped:** `stillAov { albedo, normal, depth }` and `stillDenoise` on
`RenderSettings` (serde-default, v4-safe); final-gated in `to_payload`
(`preview` pays neither); `render_with_aov` writes `<name>_*_albedo/normal/
depth.png` next to the still; OIDN runs on the beauty with the AOV guidance
from the same frame and **fails loudly** when the runtime is missing
(`pip install oidn` / `pip install pyoidn`); two checkboxes in the render
section (no panel); sidecar names the actual renderer and flags. Previews
and viewport frames skip both entirely. AOV files are written for sequence
frames too (they are final renders).
**Check (done):** schema contract + wire shape + round-trip tests, payload
assertions (final vs preview), `npm test` all green; GPU-verified albedo/
normal/depth AOVs written correctly and the denoise-missing failure path.

## 7. The two real design increments

### 7.1 Vector layers (3–5 days) — trigger: a user brings GeoJSON
This is the first feature the current `TerrainRenderParams` path cannot do, so
it is the only place worth adding a second path.

**Minimal model (schema v5 + migration):**
```
MapSettings { layers: Vec<VectorLayer> }
VectorLayer { id, name, asset, visible, style: { color, widthPx, opacity } }
```
- One new asset kind (`geojson`), registered through the existing `AddAsset`.
- One command `SetMapLayers { layers }` (whole list, like `SetChannelKeys`):
  add/edit/reorder/delete is one undo step; no per-property command zoo.
- New service command `render_map` that builds a `SceneRecipe` with the terrain
  plus `VectorOverlay` layers and renders via `MapScene`. Used only when the
  project has layers; terrain-only projects keep the current path unchanged.
- UI: a Layers list under the scene tree; import GeoJSON; flat style fields.
- Defer: labels, furniture, Mapbox style import, style expressions, PDF/SVG,
  plate layouts, alignment/reprojection. Each is a later trigger.
- Check: v4→v5 migration, layer round-trip/undo, one MapScene render smoke
  (GPU-opt-in).

### 7.2 Meshes in Rendered (5–10 days) — trigger: users need glTF in Rendered
**Lazy path only:** `io.import_gltf` + `geometry` + `instance_mesh_gpu_render`,
composited over the terrain frame with the same camera. No path tracer, no
BVH product path, no hybrid preview.

**Measured 2026-11-16 — the plan's premise needs an engine change first.**
`instance_mesh_gpu_render` has no camera parameter and its Rust entry point
hard-codes the eye at (3, 2, 3) looking at the origin
(`src/render/instancing.rs:204`), so a mesh cannot share the terrain's
orbit camera; the renderer supports `set_view_proj`, the Python entry point
just never exposes it. The engine's in-recipe alternative for a glTF is a
*texture* composite in a screen rectangle (a landmark card), which needs the
model's albedo as a separate PNG while the importer exposes only texture
slots. Details, arithmetic and the two other native-layer findings:
[`native-layers-findings.md`](native-layers-findings.md).
- Reuse the existing glTF asset and node selection.
- Service command `render_meshes` or an extra pass inside `render_map`.
- UI: none beyond what exists (objects already exist; Rendered just starts
  including them).
- Defer: LOD chain, tangents, PT, SDF, AOV per object.
- Check: a cube import renders over terrain at the viewport camera.

## 8. Explicitly not built (with the trigger that would change that)

The table is a *trigger* list, not a wish list: each row names the condition
that would justify building the thing. The **state today** column is the
audit trail — it says which triggers have since been met (and the increment
that answered them) and which are still unmet. Reviewed 2026-11-16.

| Feature | Why not now | Trigger | State today |
|---|---|---|---|
| MapScene as the default path | current path already renders; rewrite risk | vector layers land | **met** — Phase 15 made MapScene the render path for every still and frame |
| Labels / furniture / style import | big model + style spec surface | vectors shipped, users ask for text/legend | **partly met** — labels, furniture and the graticule shipped (Phase 14/16); *style* import (Mapbox/style expressions) is still open |
| Point clouds / 3D Tiles / buildings | engine marks them underdeveloped/native-required | engine exposes an offline render | **unmet** — the engine has them natively; Studio cannot author them yet. Next big increment (§12) |
| VT / scatter / smoke volumes | subsystems, not settings | a large-scene user asks | unmet |
| Camera rigs / CHRONOS / ANAMNESIS | current per-frame export works; caching is an optimization | sequences get slow | unmet (the per-frame export is the Studio path) |
| Path tracing / splat / inverse / globe | different product tier | explicit request | unmet |
| Bundles / certificates / provenance | no interchange/provenance need yet | sharing/verification requested | unmet |
| COG remote streaming | local COGs are GeoTIFFs and already import via rasterio | a URL-based DEM is requested | **partly met** — probe + whole-file materialization into a hash-keyed cache shipped (Phase 17.2.4); the tile-laddered progressive read is the open part (§12) |
| Licensing UI | no Pro-gated feature is exposed yet | first gated feature | unmet — the engine's `MapPlate` is Pro-gated (`_check_pro_access`), so plate export needs a licensing decision before it is an increment |
| Detached viewer / Jupyter widgets | Studio viewport is the product path | diagnostics request | unmet |
| Datasets / presets / material library | sample projects already exist | onboarding gap measured | **met** — the dataset registry gallery (Phase 17.2.1/2.2) and the five renderer-preset recipes (Phase 17b) shipped; the *material* library is still open |
| Precision/dd harnesses | internal determinism tooling | a precision bug is reported | unmet |

## 9. Schema policy (keeps the lazy path honest)

- Optional scalar settings: no version bump; `#[serde(default)]` (established
  by `environment`, `environmentBackgroundIntensity`).
- New collections (`MapSettings.layers`): **v5 + `migrate_v4_to_v5`**; older
  builds must reject, not silently drop.
- Engine recipes are always derived; no engine struct in the schema.
- Bundle import/export stays a conversion, never the project format.

## 10. Testing that fits the work

- Rust: one schema-contract line per field, one round-trip test per group.
- Shell: one `to_payload` assertion per new request field.
- Python: one model-validation test per new engine dataclass.
- Browser: one manual check per UI group; no e2e framework.
- GPU: only when the pixel is the product (e.g. HDRI sky). Look settings do not
  need golden images; their payload is the contract.

## 11. Build order and effort — what shipped, what is next

The original table listed ten increments; **eight of them shipped** (Phases 5–16
plus the AOV stills). What remains is the heavy pair and the streaming work.
Reviewed 2026-11-16.

| # | Increment | Days | State |
|---|---|---|---|
| 1 | 8.0 engine facts in probe (backend, adapter, granted-features snapshot) | 0.5 | **done** |
| 2 | 9.0 sun from date/time/location | 0.5 | **done** |
| 3 | 5.1 sky model + atmosphere | 1 | **done** |
| 4 | 5.2 water | 0.5 | **done** |
| 5 | 5.3 clouds | 0.5 | **done** |
| 6 | 5.4 post (bloom/vignette) | 0.5 | **done** |
| 7 | 5.5 shadows | 0.5 | **done** |
| 8 | 10.0 AOV/denoise stills | 1 | **done** |
| 9 | 7.1 vector layers | 3–5 | **done** — Phase 14/15 (lines, labels, furniture, graticule, polygons, legend) |
| 10 | 7.2 meshes in Rendered | 5–10 | **open** — §12, the visible gap: glTF shows in Solid/Material and vanishes in Rendered |
| 11 | COG tile-laddered streaming | 1–2 | **open** — §12; the probe and the download shipped, the progressive read did not |
| 12 | MapScene native layers (buildings / 3D Tiles / point clouds) | big | **open** — §12; one layer kind per increment |

## 12. Next increments (2026-11-16)

The list the user agreed to work through, in order. Each is one increment per
commit, gates green, docs landing with the work, and every shipped control
measured in pixels (changed-sample counts in the commit message).

1. **Docs truth pass** — this table, §8's state column and §11: the plan docs
   described shipped work as pending. **done** (2026-11-16).
2. **rpm/appimage audit** — **done** (2026-11-16). Built and inspected all
   three bundles. Findings: the `rpm` declared *fewer* runtime dependencies
   than the `deb` (the bundler auto-detects only gtk/webkit, so
   `libgbm1`/`mesa-vulkan-drivers` were missing) and carried **no**
   maintainer script — it now declares the same four dependencies and runs
   the same postinst, so a Fedora stranger gets the same remedy message. The
   `appimage` bundles its own GTK/WebKit stack (113 MB) and, having no
   install step, cannot print the remedy — that lives in the README and in
   `--engine-check`. All three carry the same binary
   (`/usr/bin/forge3d-studio-shell`) and a `desktop-file-validate`-clean
   entry. The README now states what each bundle declares instead of
   shrugging at "unaudited".
3. **Open-by-name for import / imagery / vector** — **done** (2026-11-16).
   One new IPC (`dataset_path`) resolves a registry name to a local path
   through the engine's own fetch (hash-verified, cached), and the three
   import sites share it: the DEM (Inspector → "Engine DEMs…"), the imagery
   overlay and the vector layer (the Cartography panel's "+ Add layer" menu).
   The picker lists the registry by kind — `dem`, `vector`, `overlay` — so a
   point cloud can never be offered where a DEM belongs (node test). Pixel
   evidence: the registry resolves to the same file the file-picker would
   give (Fuji 512²: **262,144 of 262,144 samples identical**, delta 0), and
   the import itself moves the frame against the starter scene (**157,949**
   changed samples for Fuji, **261,340** for the Swiss). Two registry DEMs
   differ from each other (230,445), so the picker really picks terrain.
4. **Meshes in Rendered** (§7.2) — **engine-facing; measured 2026-11-16**
   ([`native-layers-findings.md`](native-layers-findings.md)). The mesh
   renderer works, but its camera is hard-coded at (3, 2, 3) looking at the
   origin, so it cannot be composited with the terrain's camera; the engine's
   in-recipe route for a glTF is a texture card, not a model in the scene.
   Split once the engine exposes a camera: (a) the still composites imported
   meshes through the engine's own instanced path, measured; (b) the Rendered
   viewport shows them.
5. **COG tile-laddered streaming** (`phase-17-plan.md` §2.4) — **studio-side
   only; measured 2026-11-16** ([`native-layers-findings.md`](native-layers-findings.md)).
   The engine's reader is already tile-based (`read_tile(x, y, lod)`), so
   nothing is missing engine-side; the work is Studio's asset model (a remote
   DEM asset kind), a tile cache, and a preview heightfield that refines with
   the camera — the probe and the whole-file materialization shipped, the
   progressive read did not.
6. **MapScene native layers** (§8) — **engine-facing; measured 2026-11-16**
   ([`native-layers-findings.md`](native-layers-findings.md)). The engine's
   recipe owns buildings, point clouds and 3D Tiles, and the buildings route
   works at city scale (the engine's own suite is green here) — but over a
   country-scale DEM it changes **0 of 262,144 samples**, because the
   building height is a fraction of a ~45 m terrain extent. Point clouds need
   a DEM in the cloud's CRS (the repository's only COPC has none). One layer
   kind per increment, each mapping onto the recipe's own layer type rather
   than a Studio-side re-implementation — but only after the engine's
   anchoring is real-world.

Held for a decision, not for lack of work: the **map plate** (the engine's
`MapPlate` calls `_check_pro_access`, so a plate export needs a licensing
answer before it is an increment), and the **PR that runs `deb-audit`** on a
real runner (the job is written and YAML-valid; only the merge is the user's).
---

## Appendix A — module coverage decisions

Every public module from the earlier audit, now with a decision instead of a
promise. `build` = in the increments above; `trigger` = §8; `tool` = test/bench
surface; `out` = deliberately outside Studio.

| Module | Purpose | Decision |
|---|---|---|
| `terrain_params` | renderer settings catalog | build (5.x) |
| `offline`, `denoise`, `denoise_oidn` | offline quality/denoise | done (10.0) |
| `sky`, `atmosphere`, `geo`, `astro` | sky/sun/moon | build (5.1, 9.0) |
| `map_scene`, `vector`, `vector_line` | vectors/layers | trigger (7.1) |
| `style`, `style_expressions`, `raster_style`, `thematic`, `colors`, `materials`, `textures`, `presets` | styling/material library | trigger |
| `label_plan`, `text`, `text_atlas`, `legend`, `scale_bar`, `north_arrow`, `graticule` | cartography | trigger |
| `export`, `map_plate`, `publication` | print/vector export | trigger |
| `geometry`, `mesh`, `io`, `path_tracing` | meshes/PT | trigger (7.2) |
| `cog`, `gis`, `alignment`, `crs`, `preparation`, `codec` | geo data | trigger (local COG already works) |
| `datasets` | sample datasets | out (samples exist) |
| `pointcloud`, `tiles3d`, `buildings` | data layers | trigger (engine incomplete) |
| `terrain` (VT), `terrain_scatter`, `smoke`, `media` | surface systems | trigger |
| `camera_rigs`, `animation`, `chronos`, `anamnesis` | rigs/caching | trigger |
| `bundle`, `certificate`, `provenance`, `recipe_manifest`, `determinism`, `verify`, `precision`, `_license` | evidence/licensing | trigger (except engine facts 8.0) |
| `render_pass` | pass composition | trigger |
| `splat`, `inverse`, `sdf`, `lighting`, `guiding`, `landscape`, `viewer*`, `interactive`, `widgets` | lab/viewer | out |
| `config`, `diagnostics`, `mem`, `_gpu`, `_memory`, `_native`, `_degradation` | runtime plumbing | build only where 8.0 needs it |
| `bench`, `smoke` (harness), `terrain_demo`, `terrain_pbr_pom`, `_map_scene_*`, `_viewer_*`, `_png`, `_canonical_json`, `_deltae`, `_ed25519`, `_pointcloud_octree`, `_render_pass_bundle`, `_detail_normals`, `_cache_policy`, `_validate` | internal/tooling | tool |
| `_anamnesis_*`, `_anamnesis_params`, `_anamnesis_scene` | cache internals | trigger with 16 |

## Appendix B — Rust surface decisions

`terrain*`/`render*`/`scene`/`passes` → build only as settings reach the
payload; `vector`/`style`/`labels`/`export` → trigger with 7.1; `geometry`/
`mesh`/`accel`/`import`/`loaders` → trigger with 7.2; `gis`/`geo`/`converters`/
`codec`/`tiles3d`/`pointcloud`/`smoke`/`media`/`splat`/`sdf`/`viewer`/`astro`/
`lighting`/`license`/`verify` → trigger or out per Appendix A. Cargo features
(`copc_laz`, `cog_streaming`, `proj`, `enable-globe`, `enable-inverse-pt`,
`atmosphere-bake`, `splat-fusion`, `weighted-oit`) are reported by 8.0 and
never assumed.

## Appendix C — earlier phase mapping

| Earlier phase | Now |
|---|---|
| 8 capabilities/licensing | 8.0 (facts only; licensing deferred) |
| 9 MapScene recipes | not now; 7.1 uses MapScene only for vectors |
| 10 look | 5.1–5.5 |
| 11 cartography | trigger (7.1 then labels/furniture) |
| 12 geo/time | 9.0 sun; COG local already works; rest trigger |
| 13 data layers | trigger |
| 14 VT/scatter/volumes | trigger |
| 15 meshes/PT | 7.2 (raster only); PT out |
| 16 rigs/CHRONOS/ANAMNESIS | trigger |
| 17 bundles/evidence | trigger |
| 18 lab | out |
