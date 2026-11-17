# Phases 18 onward — the next Studio implementation plan (2026-11-16)

This is the plan after the Studio surface was rebased onto upstream `main`
(PR #310, all checks green, including the hosted `deb audit`). It replaces the
"next increments" list in `forge3d-integration-roadmap.md` §12 as the working
document; that section now points here.

Rules carried forward (non-negotiable): **reuse the engine's machinery, never a
second implementation**; MapScene stays the render path; every shipped control
moves ≥1 pixel under measurement or says honestly why it cannot; one increment
per commit with gates green; docs land with the work; the engine changes only
when a failing test makes the fix unavoidable.

## Where we are

| Done | Evidence |
|---|---|
| Renderer presets as five named Studio recipes | payload node tests; Fuji 512² deltas 159,556 / 206,585 / 66,613 / 257,284 / 159,585 changed samples |
| Stranger-installable deb, rpm parity, honest probe | hosted `deb audit` on PR #310: `dpkg -i`, postinst remedy, bare PATH exit 1, fresh venv forge3d 1.49.0 exit 0, desktop entry clean |
| Open-by-name dataset imports (DEM/imagery/vector) | registry file gives the same frame (262,144/262,144 samples identical) and moves the frame vs starter (157,949 Fuji, 261,340 Swiss) |
| COG ladder, rung 1 (overview preview) | 34,256 bytes = 6.0% of the file, 17 range requests, no full-body GET |
| Three deep items measured, not guessed | `native-layers-findings.md`: mesh renderer's camera is hard-coded at (3,2,3); buildings render 0/262,144 samples over a 24 km DEM; point cloud needs a matching DEM |

## What upstream 1.49.0 added that Studio can adopt

From `CHANGELOG.md` on `origin/main` (59 commits since the fork's old `main`;
the PR branch already contains them):

- **Viewer terrain grid cap** (`FORGE3D_VIEWER_MAX_TERRAIN_GRID`, #299), a
  terrain-aware near plane and cascade fixes shared by onscreen/offscreen
  effects — directly relevant to Studio's Solid viewport and Rendered stills.
- **Subject-led terrain flyovers** with feature-ID binding, clearance/framing
  checks and a Sequencer/Movie Render Queue handoff (#300) — an engine
  capability Studio's timeline could drive instead of reimplementing.
- **`ValueFaithfulRasterBinding`** (#302): one declaration for statistical
  relief, fixed colours, coverage and a top-down inset — the engine's own
  mapping of exactly the raster layers Studio currently composes itself.
- **Offline semantic-scale workflow** (#301): routes, landmarks, a north-up
  locator, frozen scale decisions — a close match to the map plate's furniture.
- 1.48.0 also added **`PublicationRecipe`** with plan-view PNG publications,
  legends derived from classification, and linked insets.

Each is a *capability negotiation* item: probe what the installed engine
actually exposes, then map it, with the deviation named when it cannot be
expressed.

---

## Phase 18 — finish the COG ladder (Studio-side, no engine change)

**Why:** the remote DEM still materializes the whole file at open; the preview
reads a coarse overview through the engine's range reader (rung 1 landed), but
the viewport and the still have no progressive path. The engine's reader is
already tile-based (`open_cog` → `read_tile(x, y, lod)`, `ifd_info(level)`), so
nothing is missing engine-side.

**Increments:**

1. **Remote DEM asset kind.** A project can reference a URL-backed DEM
   (`kind: "remote-dem"`, hash-keyed cache path, CRS/bounds from the probe)
   without downloading the file at open. Migration + round-trip tests; the
   asset's integrity check uses the engine's probe, not a filesystem stat.
2. **Camera-refining heightfield.** The Solid viewport samples the DEM through
   a tile ladder: the overview for the far view, finer tiles as the camera
   closes; a nearest-first top-up loop with backpressure like the engine's
   `ClipmapStreamer`. Measured: range requests and bytes per camera distance,
   plus the preview's pixels at each rung against the local-file render.
3. **Streaming still.** The final render reads tiles through the engine's
   reader at the requested resolution; the whole-file materialization becomes
   an explicit user action ("download for offline"), not the default. Byte
   comparison against the current materialized-file still on Fuji/Swiss.

**Gate:** every rung's pixels equal the whole-file render (or the difference is
quoted and justified); the cache respects its byte budget; `npm test`, fast
lane, service tests, fmt/clippy green.

**Status: rungs 1 and 2 shipped** (`ef9df55a`, this increment). Rung 1 read
the coarsest overview through the engine's range reader for the dialog.
Rung 2 adds a camera-driven resolution ladder in the viewport: three rungs
(96/160/256) chosen from orbit distance over the DEM's measured span
(`src/state/terrainLadder.ts`, node-tested), remote DEM previews routed through
`remoteCogPreview` at that resolution, and a debounced mesh rebuild when the
camera crosses a rung. The engine reader's own level selection makes the
choice real: a 64-px request selects overview level 4, a 256-px request level 3
(`test_the_remote_preview_ladder_selects_finer_overviews`). Byte counts do NOT
distinguish these rungs — the range reader fetches block-aligned ~128 KiB
windows, measured at 132,911 B for both — so the test pins the level and the
returned heightfield size instead, and the earlier overview test pins the
fraction-of-file claim.

**18.3 landed as well.** A `url:` project now opens from the engine's probe
plus one coarse overview — no whole-file download — and the still assembles
the DEM from the engine's COG tiles through `open_cog`/`read_tile`, using the
engine's own GIS helpers for the metric span. `url_fetch` and its script were
deleted (dead after streaming replaced materialization). The range-server test
runs the render in a child process, as the shell does, and verifies that the
still fetches under half of a 2048² COG and produces a non-flat mapscene frame.
The engine's native HTTP reader cannot be driven from the pytest process, so
the test matches the shipped process shape rather than working around it.

Phase 18 is now functionally complete for open → preview → still; 18.1's
separate `RemoteDem` asset kind is still open as a schema improvement (a URL
currently lives in `Asset.path` and is detected by prefix).

---

## Phase 19 — glTF models in the Rendered frame (engine + Studio)

**Why:** a user imports a GLB, sees it in Solid/Material, and it vanishes in
Rendered — a visible silent no-op. The measured blocker is engine-side:
`geometry.instance_mesh_gpu_render` works but hard-codes the camera at
`(3, 2, 3)` looking at the origin (`src/render/instancing.rs:204`); the renderer
supports `set_view_proj`, the Python entry point just never exposes it.

**Increments:**

1. **Engine: expose the camera.** Add view/projection (and a depth target) to
   `instance_mesh_gpu_render` with a default preserving today's behaviour;
   failing test first: render the same mesh from two cameras and assert
   different pixels. No certificate/schema drift without a new ledger entry.
2. **Studio: the still includes imported meshes.** The service collects the
   project's visible mesh objects with their transforms, renders them through
   the camera-aware engine path with the same camera as the terrain, and alpha
   composites over the MapScene frame. Measured: changed samples at 512² with a
   known cube over the Fuji DEM, plus an ablation gate (remove the mesh call and
   watch the test fail).
3. **Studio: the Rendered viewport shows them** — same path, preview quality;
   the Solid viewport keeps its three.js scene. One honest limitation note for
   anything the engine path cannot express (per-object AOVs, shadows on meshes).

**Gate:** the ablation is real; pixels move; no parallel renderer; the engine's
feature inventory and contract tests stay green.

**Status: 19.1 and 19.2 landed (2026-11-17).** 19.1 added the orbit camera to
`instance_mesh_gpu_render`; scoping 19.2 showed the terrain frame is in metres,
so the camera contract also gained `near`/`far` (the renderer's defaults were
0.1/100 — a 24 km scene was entirely beyond the far plane). The regression test
first: `tests/test_mesh_gpu_camera.py` (two eye positions, different pixels).
Both pass after the change; the f32 ledger stayed green because the original
`as f32` statement was preserved byte-for-byte.

19.2 completes the path: `Transform::row_major_matrix` and
`workspace::mesh_render_matrix` in the core are the single viewport→render
conversion `(vx, vy, vz) -> (vx*s, -vz*s, vy*s - half_relief)` with a unit test;
the shell hands visible imported meshes to the service with render-space
matrices; the service draws them through the engine's instanced path with the
still's own camera and clip range, and composites non-black pixels over the
MapScene frame. `tests/test_studio_mesh_composite.py` pins it end to end: a
quad centred on the camera aim lands at the frame centre (131, 130 in 256²),
the ablation is byte-identical, and a node-scoped mesh is reported instead of
silently rendered. Known limit: the instanced renderer clears to opaque black,
so a black-painted model cannot be distinguished from the clear colour; that
is recorded in the service docstring, not hidden.

Verified: service+mesh+camera 73/73, fast lane 760/760, studio-core 77,
shell 25, fmt/clippy(0)/tsc/npm.

**19.3 lands with 19.2 — no second renderer was needed.** The Rendered
viewport already calls the same `render_frame` command as a still (with a view
override and preview quality), and the viewport re-renders on every scene
revision, camera change and gizmo release. Wiring meshes into the shared render
payload therefore makes them appear in the Rendered viewport as well as the
still, with no new UI control. Verified by the off-aim test in
`tests/test_studio_mesh_composite.py`: with a viewport-style orbit
(phi 135, theta 35, fov 50) a mesh moved 3 viewport units east projects
 measurably off-centre, while the ablation frame is identical. Phase 19 is
 therefore complete; no separate viewport mesh path exists or is planned.

## Phase 20 — MapScene native layers (engine + Studio)

**Why:** the engine's recipe owns `BuildingLayer`, `PointCloudLayer` and
`Tiles3DLayer`; Studio's map stack cannot author them. Measured: the buildings
route works at city scale (engine suite 7/7) but changes 0/262,144 samples over
a 24 km DEM because heights are a fraction of a ~45 m terrain extent; the
native path refuses a GeoJSON building layer with screen-space effects on
because its `source.path` is read as a glTF path.

**Increments:**

1. **Engine: real-world building anchoring.** Height in metres, anchored to the
   DEM's bounds rather than the layer's own bbox; the failing test is a
   country-scale DEM with a city footprint that must move ≥1 pixel. Fix
   `_building_gltf_path` so a GeoJSON source is not treated as a glTF; test the
   layer with TAA enabled (the still's configuration).
2. **Studio: a buildings layer kind.** Import GeoJSON/CityJSON footprints from
   the registry's `mount-fuji-buildings` and `sample-buildings`; the service
   hands the engine its own `BuildingLayer`; the Cartography panel gets an eye,
   a source and an honest "engine-rendered; Rendered only" note (the Solid
   viewport cannot draw it).
3. **Point clouds and 3D Tiles.** The registry ships `mt-st-helens` (COPC) and
   the engine projects `PointCloudLayer` through the recipe camera — but the
   cloud needs a DEM in the same CRS. First increment: a matching St Helens DEM
   (or a reprojection path), then one layer kind, then the tile layer.

**Gate:** each layer moves pixels on the real DEM; the engine's validation
reports "supported" for what the studio actually renders (no validate/render
contradiction); no Studio-side geometry reimplementation.

**Status: engine anchors landed; Studio layer kind still open.** measured and scoped (`native-layers-findings.md` §3); not started.

---

## Phase 21 — publication and plate convergence (upstream 1.48/1.49)

**Why:** upstream now has `PublicationRecipe`, subject-led flyovers, semantic
scale and value-faithful rasters. Studio has its own plate and scale-bar code;
the roadmap forbids a second implementation of engine capabilities.

**Increments:**

1. **Capability probe for publication.** Report which upstream publication
   APIs the installed engine exposes; map Studio's plate/furniture/legend onto
   `PublicationRecipe` where possible; name the gaps.
2. **Adopt `ValueFaithfulRasterBinding`** for Studio's raster layers — one
   declaration replacing the separate statistical-relief, palette, coverage
   and inset code paths. Pixel comparison against today's plate.
3. **Map plate printing** — gated on the licensing decision: the engine's
   `MapPlate` calls `_check_pro_access` and raises without a Pro licence.

**Status:** blocked on the licensing answer for item 3; items 1–2 can start.

---

## Phase 22 — viewer/terrain parity with upstream fixes

**Why:** 1.49.0 added the grid cap, a terrain-aware near plane and cascade
fixes; Studio's viewport has its own camera and near/far handling. Honest
convergence means negotiating those capabilities, not copying the code.

**Increment:** probe the engine's viewer capabilities, expose the grid cap as a
capability-tagged control (only when the installed engine grants it), and
compare the viewport's Rendered frame against the still under the same camera.
No new UI unless the pixel comparison shows a difference the user must choose.

**Status:** not started.

---

## Ordering and first next commit

1. Phase 18.1 — the remote DEM asset kind (no engine change, self-contained,
   testable offline with the existing range-server fixtures).
2. Phase 19.1 — the engine camera exposure (unlocks the most visible gap).
3. Phase 20.1 — buildings anchoring (unlocks all three native layers).
4. Phases 21/22 — capability negotiation with upstream's newer surface.

Phases 19 and 20 need engine changes; each must be driven by a failing test and
keep the engine's contract, determinism and ledger gates green. Phase 18 does
not, and is the one to start while the others are specified in detail.
