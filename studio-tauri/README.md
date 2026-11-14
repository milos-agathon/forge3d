# Forge3D Studio — Tauri shell

The desktop shell for the renewed **Forge3D Studio** product described in
[`../FORGE3D_STUDIO_IMPLEMENTATION_PLAN.md`](../FORGE3D_STUDIO_IMPLEMENTATION_PLAN.md):
a Tauri app around a canonical project model, with one project serving both the
Studio workspace and (from Phase 3) the Forge3D workspace.

**Phases 1–6 are implemented**: the desktop shell, project schema and migration,
open/save/autosave/recovery, command-based undo/redo, a Three.js viewport with
Solid / Material / Rendered modes, orbit/pan/zoom and a camera view,
Blender-style move/rotate/scale tools, an orientation gizmo, add/duplicate/delete,
glTF/GLB import, the Forge3D workspace adapter (per-parameter capability table,
engine probe, camera/sun sync), and the animation timeline with keyframes and
PNG sequence export.

> The pre-existing `../studio/` tree is the earlier terrain-focused eframe/egui
> Studio. It is a separate application and a separate Cargo workspace. This
> directory is the Tauri product surface; the two do not share a build.

## Status against the plan

| Deliverable | Where | State |
|---|---|---|
| **Phase 1** — Tauri shell, React UI, Rust project service | `src-tauri/`, `src/`, `crates/studio-core/` | done |
| Versioned schema v2 + v1 migration | `project.rs`, `persist.rs` | done |
| Open / save / autosave / recovery | `persist.rs`, `autosave.rs`, `session.rs` | done |
| Command model + undo/redo (14 commands) | `command.rs`, `editor.rs` | done |
| **Phase 2** — Three.js solid viewport | `src/viewport/Viewport3D.tsx` | done |
| Orbit / pan / zoom | `OrbitControls` in `Viewport3D.tsx` | done |
| Object selection, tree ↔ viewport sync | `Viewport3D.tsx`, `SceneTree.tsx` | done |
| Move / rotate / scale tools (gizmos) | `TransformControls` + `ViewportToolbar.tsx` | done |
| Orientation gizmo (top right, clickable) | `src/viewport/AxisGizmo.tsx` | done |
| Add / delete / duplicate | `command.rs`, `session.tsx` | done |
| glTF / GLB import | `ImportGltf` command + `assets.ts` | done (see limitation) |
| Look through the active camera | `Viewport3D.tsx` (`0`, toolbar) | done |
| **Phase 3** — workspace switching | removed (see below) | dropped deliberately |
| Capability report per object | `crates/studio-core/src/workspace.rs` | done, tested |
| Engine discovery | `src-tauri/src/forge3d.rs` | done, shown in the status bar |
| Separate viewer window | — | removed: rendering lives in the viewport |
| Camera + sun transfer | `RenderRequest::from_handoff` | done |
| Diagnostics + idempotent handoff | `forge3d.rs`, `workspace.rs` | done |
| **Phase 4** — terrain-first default scene | `Project::starter`, `TerrainSeed` | done |
| DEM heightfield preview in Studio | `terrain_preview`, `buildTerrainMesh` | done |
| Embedded renderer service (viewport frames + stills) | `studio_render.py`, `render.rs` | done |
| **Rendered viewport mode (embedded)** | `studio_render.py`, `render.rs` | done |
| Render size control (viewport / DEM native / project) | `Viewport3D.tsx` render bar | done |
| Render still to disk (+ JSON sidecar), render settings UI | `render_still`, `RenderSettingsEditor` | done |
| **Phase 10** — AOV export + OIDN denoise for final stills | `render_still`, `RenderRequest`, `studio_render.py` | done |
| **Phase 5** — capability-tagged material/light/camera parameters | `workspace.rs` table → `capability_table` → Inspector badges | done |
| Material inspector (terrain albedo + engine summary, mesh base colour/texture) | `Inspector.tsx` | done |
| Texture asset management (attach, replace, remove; non-empty invariant) | `session.tsx`, `commands.rs` | done |
| Environment/IBL controls (sky background, environment intensity) | `RenderSettings`, `RenderSettingsEditor` | done |
| HDRI environment for Material + Rendered (`read_hdr`, PMREM, `IBL.from_hdr`) | `RenderSettings.environment`, `Viewport3D.tsx`, `studio_render.py` | done |
| Basic lights & camera properties (direction/intensity/colour, fov/clip) | `Inspector.tsx`, `render.rs` | done |
| Asset relinking and dependency counts | `Inspector.tsx`, `command.rs` | done |
| Material-preview albedo drape (raster → PNG via engine python) | `raster_preview`, `Viewport3D.tsx` | done |
| glTF node selection (`SetGltfNode`, runtime-derived node names, Inspector picker) | `assets.ts`, `command.rs`, `Inspector.tsx` | done |
| **Phase 6** — timeline, keyframes, interpolation | `crates/studio-core/src/timeline.rs`, `src/components/Timeline.tsx` | done |
| Playhead / scrub / playback (one clock in `playback.ts`) | `setPlayhead` → `Session::set_playhead` | done |
| Per-channel keys (`SetChannelKeys`), 9 transform channels + camera FOV | `command.rs`, `timeline.rs` | done |
| Offline PNG sequence export (`renderAnimationFrame` per frame) | `src/state/session.tsx`, `render.rs` | done |
| Camera animation reaches Forge3D (`camera_render_view` → `with_view`) | `workspace.rs`, `commands.rs` | done |
| Sun from date/place via the engine ephemeris (`sun_rotation`) | `commands.rs`, `workspace.rs`, light Inspector | done |
| **Phase 7** — crash recovery, atomic save, recovery store | `autosave.rs`, `session.rs`, `RecoveryBanner.tsx` | done |
| Asset integrity (existence + size on open/relink, Inspector badges) | `integrity.rs`, `check_assets`, `Inspector.tsx` | done |
| Large-scene performance guards (2k objects / 500 tracks, 10k stress) | `crates/studio-core/tests/performance.rs` | done |
| Offline render queue + cancellation (stills and sequences) | `src/state/renderQueue.ts`, `RenderQueue.tsx` | done |
| Packaging (deb/rpm/appimage config, release bundle job) | `tauri.conf.json`, `.github/workflows/studio-tauri.yml` | done (Linux) |
| Sample projects + docs | `samples/`, `examples/make_samples.rs`, this file | done |
| Workspace-switch regression suite | — | dropped with Phase 3 |

Exit criteria — *edits change the canonical model, reloading preserves them* and
*no viewport-only copies* — are covered by `crates/studio-core/tests/phase1.rs`
and `phase2.rs`; the viewport only reads model state and writes through commands.

## Architecture

```text
studio-tauri/
├── crates/studio-core/        canonical model — no Tauri, no GPU, no Forge3D
│   ├── project.rs             Project document v2: format tag, assets, validate
│   ├── scene.rs               Scene graph: objects, transforms, mesh sources
│   ├── command.rs             Every mutation as a reversible Command
│   ├── editor.rs              Session: revision, dirty flag, undo/redo stacks
│   ├── persist.rs             Version gate, v1→v2→…→v4 migrations, atomic save
│   ├── autosave.rs            Recovery envelopes + pruning
│   ├── session.rs             One open project: open, save, autosave, recovery
│   ├── timeline.rs            Keyframes, interpolation, the one evaluate()
│   └── workspace.rs           Capability report, camera/sun + render-view handoff
├── src-tauri/                 Tauri v2 shell
│   ├── src/commands.rs        IPC surface (mesh files, Forge3D adapter, animation)
│   ├── src/forge3d.rs         Engine probe, viewer process, diagnostics
│   ├── src/render.rs          Embedded renderer service + render request
│   ├── src/studio_render.py   Embedded Python bridge (relays viewer IPC)
│   ├── src/session.rs         Mutex around the core session
│   └── src/lib.rs             App wiring + background autosave thread
└── src/                       React + TypeScript UI
    ├── bridge/                Tauri IPC (+ a "start the shell" page outside it)
    ├── components/            Top bar, tree, inspector, toolbar, timeline
    ├── state/session.tsx      Session actions + active tool/space
    ├── state/playback.ts      Tested timeline playback clock (no React)
    ├── viewport/
    │   ├── Viewport3D.tsx     Three.js scene, modes, picking, gizmos, assets
    │   ├── viewMath.ts        Viewport → renderer camera mapping (+ test)
    │   ├── AxisGizmo.tsx      SVG orientation widget (no second WebGL context)
    │   └── assets.ts          glTF parsing and per-path scene cache
    └── styles.css
```

### Decisions that matter

- **The model is its own crate.** `studio-core` compiles and tests with
  `cargo test` on a machine with no webkit, no display, and no GPU, so the exit
  criteria can be verified in CI. The Tauri shell only translates IPC to it.
- **Commands carry new values, return inverses.** `Command::apply` reads the
  previous value from the model and returns the command that reverses the edit.
  Undo is not a second implementation of editing; it applies the stored inverse.
- **Redo is canonicalised.** After applying a command, the editor derives the
  redo command by applying the inverse to a copy of the new state. Without that
  step, redoing `duplicateObject` or `createPrimitive` would mint a fresh UUID
  and "redo" would not describe the object undo removed. `tests/phase2.rs` pins
  this for both.
- **IDs are minted in Rust.** `createPrimitive`, `duplicateObject` and
  `importGltf` generate their object (and asset) IDs inside the model, so the UI
  cannot invent identities and two windows cannot collide.
- **Schema v2 is a real migration.** v1 meshes carried `kind.primitive`; v2
  nests it under `kind.source` (which may also be an imported glTF asset) and
  adds the asset registry. `persist::migrate` rewrites v1 files on load, one
  version at a time, and a test saves a migrated file back as v2.
- **Asset paths are stored relative** to the project file and resolved back to
  absolute on load, so a project directory stays portable. Files outside the
  project directory stay absolute.
- **Imported glTF geometry is cached per path and cloned per object.** Two
  objects referencing the same GLB share geometry; only geometry the app itself
  created is disposed on rebuild.
- **The viewport is a projection.** `Viewport3D` reads `SessionState` and writes
  only through commands; gizmo drags preview locally and commit one
  `setTransform` on release, so a drag is a single undo step.
- **Euler convention is pinned.** Studio applies X, then Y, then Z
  (`Rz·Ry·Rx`); three.js is told `"ZYX"`. Both `applyTransform` and the tree
  order comment say so, because getting it wrong looks like "the gizmo rotates
  the wrong way" rather than an error.
- **Autosave never touches the project file.** Snapshots go to the app-data
  recovery directory; successful saves prune the snapshots for that path.
- **The wire format is locked.** `tests/schema_contract.rs` asserts the exact
  key shape that `src/bridge/types.ts` mirrors by hand.

## Building and running

### Linux prerequisites (Fedora)

The window uses the system WebKitGTK. Development headers are required — the
runtime package alone is not enough:

```bash
sudo dnf install -y webkit2gtk4.1-devel gtk3-devel libsoup3-devel \
  javascriptcoregtk4.1-devel librsvg2-devel openssl-devel \
  libappindicator-gtk3-devel patchelf
```

On Debian/Ubuntu the equivalents are `libwebkit2gtk-4.1-dev`,
`libgtk-3-dev`, `libsoup-3.0-dev`, `libjavascriptcoregtk-4.1-dev`,
`librsvg2-dev`, `libssl-dev`, `libayatana-appindicator3-dev`, `patchelf`.

### Commands

```bash
npm install          # once
npm run tauri:dev    # the real desktop app (Rust + webview)
npm run tauri:build  # desktop bundles (deb/rpm/appimage)
npm run dev          # UI only; the browser shows a "start the shell" notice
npm run build        # typecheck + production bundle of the UI
```

## Installing the deb (the stranger's path)

The CI `bundle` job on `main` ships a `deb` artifact. Locally:

```bash
npm run tauri:build -- --bundles deb
sudo dpkg -i "src-tauri/target/release/bundle/deb/Forge3D Studio_0.1.0_amd64.deb"
python3 -m pip install --user forge3d   # the engine — PyPI, 1.48.0 (verified)
```

Only the `deb` is built in CI and audited (`deb-audit` job: builds the deb,
installs it with `dpkg -i`, validates the desktop entry, then runs
`--engine-check` in both interpreter scenarios — the gate is the probe, not a
GPU render, because the runner has no terrain-safe GPU). The `rpm` and
`appimage` targets remain unverified: build them with
`npm run tauri:build -- --bundles rpm` / `--bundles appimage` and treat them
as unvetted until they get the same audit.

The package deliberately *never* requires the engine at install time; the
postinst only names the remedy. Verify what the shell will see:

Real output, run from a repo checkout with `FORGE3D_STUDIO_PYTHON` set:

```bash
forge3d-studio-shell --engine-check
# candidate               interpreter  result
# FORGE3D_STUDIO_PYTHON   /path/to/venv/bin/python  ok (forge3d 1.48.0)
# repo .venv              /path/to/repo/.venv/bin/python  ok (forge3d 1.47.1)
# PATH python3            python3  fail: import forge3d failed: No module named 'forge3d'
# PATH python             python  fail: could not run python: No such file or directory (os error 2)
# engine: found via FORGE3D_STUDIO_PYTHON (forge3d 1.48.0)
```

The `repo .venv` line appears only when the shell is launched with a repo
checkout as an ancestor (the `.venv` next to the project). On a stranger's
machine only `FORGE3D_STUDIO_PYTHON` and the `PATH python3` / `PATH python` candidates can match — that is
the intended honest behaviour, not a bug.

`--engine-check` exits 0 on success, 1 on failure (with
the exact remedies it tried). Discovery order: `FORGE3D_STUDIO_PYTHON`
(validated, never trusted blind) → ancestors' `.venv` → `PATH python3` → `PATH python`.
No auto-install, no `sudo pip`, no bare-python fallback. One nuance: when the
shell runs inside a repo checkout, the probe imports the checkout's
`python/forge3d` (a dev convenience), with your own `PYTHONPATH` kept as the
tail; the installed bundle has no checkout above it, so a stranger probes
the interpreter alone. The probe is also what `dpkg -i` expects you to run
next — the postinst names it.

Troubleshooting:

| `--engine-check` says | Fix |
|---|---|
| `fail: import forge3d failed: No module named 'forge3d'` | `pip install --user forge3d` |
| `fail: import forge3d failed: No module named 'numpy'` | `pip install --user forge3d` (PyPI wheel brings numpy) |
| `fail: could not run python3: No such file or directory` | `apt-get install python3` |
| `fail: <path> did not answer the engine probe within 20s` | interpreter hangs on import — try a plain venv; the probe times out by design |

If the app opens with **"the engine is unavailable"** in the status bar, run
`forge3d-studio-shell --engine-check` — the shell is telling the truth about
what it sees; the remedy is one `pip install` away.

### Tests

```bash
cargo test -p studio-core      # 76 tests: model, commands, migration, workspace
cargo test -p forge3d-studio-shell --lib   # 23 shell tests: engine probe, candidates
npx tsc --noEmit               # UI types
```

## Viewport controls

| Input | Action |
|---|---|
| Left drag | Orbit |
| Middle drag | Pan |
| Scroll | Zoom |
| Left click | Select (shift/ctrl to add or remove) |
| `V` / `G` / `R` / `S` | Select / Move / Rotate / Scale tool |
| `Shift + D` | Duplicate selection |
| `X`, `Del` | Delete selection |
| `Esc` | Clear selection |
| `0` or the camera button | Toggle camera view (look through the active camera) |
| `T` or the Timeline button | Show/hide the animation timeline (hidden by default) |
| Click the corner gizmo | Snap the view to an axis |

The tool strip also holds **Global/Local** axis orientation and the **Add** menu
(cube, sphere, plane, DEM terrain from a GeoTIFF — which also works from an
empty scene — and import glTF/GLB).

## Solid, Material and Rendered modes

The Studio viewport has three modes, toggled in the viewport toolbar:

- **Solid** — a Blender-style clay view: neutral shading under the scene's sun,
  with **no albedo, textures or material colours**. Fast orbit/pan/zoom, gizmos,
  and instant selection. This is the editing view.
- **Material** — a lightweight material preview, not an engine render: the
  three.js scene lit by the project's HDRI (pre-filtered with PMREM) or, when
  none is set, a built-in studio environment, plus soft sun shadows. It shows
  the albedo (terrain drape, mesh base colour and texture, glTF materials) and
  uses the HDRI as its background, so it reads as a look-dev view. The camera
  orbits freely here exactly like Solid.
- **Rendered** — the Forge3D terrain pipeline, rendered offscreen and displayed
  in the viewport. Same engine, same shaders, same PBR/POM/shadows/sky settings
  as the golden-tested renders; no second window. It lights with the project
  HDRI and, when the project sky is on, draws it as the background as well
  (`IBL.from_hdr` for the light, a sky-model sample of the same map for the
  backdrop).

Implementation: `src-tauri/src/render.rs` keeps one Python process alive
(`studio_render.py`) — session, renderer, and a generated neutral sky HDR are
built once, because shader compilation and IBL prefiltering dominate start-up.
Each frame is a JSON request (camera, sun, size) answered with a base64 PNG.
Frames are requested on demand and coalesced: entering the mode, orbiting
(throttled), and every scene edit (sun, exposure, overlay — debounced) trigger at
most one queued frame. The frame's **Size** control in the render bar picks the
resolution — 1×/2×/4× the viewport, the DEM's own pixels (`DEM (1024 × 1024)`),
or the project's output size. The top-bar **Render** button opens a save dialog
defaulting to `<project>/renders/<name>-<timestamp>.png` with a JSON sidecar
describing every setting that produced it.

Mode changes only re-shade the built content (Solid swaps in a clay material and
back); they never rebuild geometry or re-fetch textures.

### Environment (HDRI)

`Lighting → Environment` adds a Radiance `.hdr` file to
the project as an asset and makes it the scene's image-based light. It is one
setting used by two consumers:

- **Material** reads the bytes through `read_hdr`, decodes them with three's
  `HDRLoader`, pre-filters with PMREM, and uses the result for `scene.environment`
  — with the equirect source as the background. A replaced map is freed once the
  new one is bound; the built-in studio sky is the fallback.
- **Rendered** sends the *path* with each frame; `studio_render.py` hands it to
  `IBL.from_hdr` and caches the prefiltered map by file identity (path, mtime,
  size), so orbiting costs nothing and replacing the file takes effect.

`Environment / IBL intensity` scales the image-based light in both views (0
turns it off; the built-in studio fallback is additionally dimmed so it reads as
fill, the project HDRI is used as authored). **Background intensity** is the
backdrop's own brightness, 0..1: it scales the HDRI sky in the Material preview
(`scene.backgroundIntensity`) and 0 swaps it for the neutral studio backdrop.
Removing the HDRI returns both views to the studio sky. The asset cannot be
deleted while it is the environment — the Assets panel reports it as in use.

When an HDRI is set and **Sky background** is on, Rendered draws it as the sky
too: the terrain renderer samples the same environment map as its background
(`SkySettings(model="hdri")`), with **Background intensity** as the sky
exposure. Without an HDRI the engine's analytic Hosek-Wilkie sky is used. The
sky pass and the environment cubemaps share one frame (Y-up, +Z south), so the
background, the IBL lighting and the Material preview's orientation all agree.
Z-up terrain directions are permuted into that frame for the sky pass only; the
terrain shading and its Z-up hash are untouched.

## Properties sidebar

The right side is a Blender-style properties panel: a vertical tab rail and a
stack of collapsible sections, toggled with **N** (the workspace gives the
viewport the extra width when it is hidden).

| Tab | What lives there |
|---|---|
| **Render** | output size, exposure, shadows (filter + map size), quality (final AO), still extras (AOV, denoise) |
| **Scene** | project and scene facts, the asset list with relinking |
| **Lighting** | the sun (type, intensity, linear colour, pitch/yaw/roll, solar placement), the sky (model, turbidity, ground albedo, exposure) and the environment (IBL intensity, HDRI map, backdrop intensity) |
| **Object** | the selected object: identity, transform, and its kind-specific editors |

Sections remember whether they were open, and the panel remembers its tab,
between runs (`localStorage`) — a settings panel whose sections forget their
state is one you re-open by hand every time. Settings stay one command per
edit (`setRenderSettings` with the whole object), so undo is one step per
change, not one per keystroke.

### Sky vs environment

Two different things, deliberately kept apart:

- **Environment** is what *lights* the scene: the project's HDRI, or the
  built-in studio gradient. Its intensity is `iblIntensity` and the preview
  and Rendered use the same map.
- **Sky** is the backdrop Rendered draws behind the terrain: an analytic model
  (Hosek–Wilkie, Preetham, or AETHER from the shipped LUT) with its own
  turbidity, ground albedo and exposure. `skyExposure` scales only the sky
  pass, so changing it cannot move the terrain's shading.

The preview cannot draw the analytic sky (it would be a second sky
implementation in three.js), so Solid and Material keep their neutral or HDRI
backdrop. Setting a project HDRI makes it the sky as well, and the Sky
controls become a readout pointing at Environment.

Measured while landing this: Hosek–Wilkie reads green-grey in the engine's sky
pass at every exposure (RGB ≈ 75/88/85 at exposure 0.02, where the CPU radiance
table says blue), and roughly 7× brighter than Preetham at the same exposure;
Preetham gives the expected blue. The controls are real either way — model,
turbidity and exposure all move pixels — but Preetham is the bluer fit until
the Hosek channel evaluation is fixed engine-side.

The tabs split a single long scroll into the four questions you actually ask:
what comes out (Render), what the project is (Scene), what lights it
(Lighting), and what is selected (Object). Light *colour and intensity* live
in the Lighting tab even when the sun is selected, so the rig is edited in one
place; the Object tab keeps the sun's generic transform.

## Final stills and sequence frames (Phase 10)

`Render → Stills` has two final-quality toggles that viewport
previews never pay for (they apply to Still and to the frames of an
animation sequence):

- **AOVs next to the still (albedo · normal · depth)** renders with the
  engine's AOV path and writes `<name>_albedo.png`, `<name>_normal.png`,
  `<name>_depth.png` next to the still — the engine's own naming, so
  texture-synthesis tools pick them up without configuration. The beauty PNG
  is written exactly as before.
- **Denoise the still with OIDN** runs Open Image Denoise on the beauty
  (guided by the albedo/normal AOVs from the same frame). The engine's frame is
  display-referred sRGB while OIDN and its guidance images are linear, so the
  service decodes to linear, denoises, and re-encodes — denoising the sRGB
  values in place fed the filter the wrong space. This needs an OIDN runtime
  next to the engine (`pip install -e .[denoise]`, which installs `oidn` and
  `tbb`); when it is missing the frame **fails loudly** instead of silently
  coming back un-denoised. OIDN 1.4 requires the device to be committed before
  filters are created — handled in `denoise_oidn.py`.

The sidecar `<name>.json` is the payload the renderer received — width,
height, camera, sun, exposure, IBL and terrain settings, all from one
producer — plus the provenance only the shell knows: which path was used
(`forge3d.render_with_aov` vs `render_terrain_pbr_pom`), the AOV flags, the
denoise flag, the frame time, the project revision and the handoff key. Depth is the engine's linear AOV normalised to
[near, far]; normals are signed world-space normals in the engine's
published environment frame — both are meant as inputs to other tools, not
as images to look at.

## The default scene is a DEM (Phase 4)

`File → New project` opens a Bryce Canyon scene, not a cube: the shell resolves
`assets/tif/Bryce_Canyon_1024.tif` (or `FORGE3D_STUDIO_DEM`), samples it for the
viewport, and stores its pixel size, georeferenced footprint, and elevation range
on the terrain object. The Bryce starter also ships its NAIP orthophoto
(`assets/tif/Bryce_Canyon_naip.tif`) draped as the terrain's albedo (overlay,
0.82 opacity, source colours preserved) whenever the file exists — that is the
"Bryce material". Studio draws a 96×96 heightfield preview; Forge3D receives
the original GeoTIFF.

Rendered frames use the camera you are looking through. The viewport maps its
camera onto the renderer's frame (east = +X, north = −Z in the viewport; +X/+Y/+Z
east/north/up in the renderer, terrain centred on the origin) and passes
`{phiDeg, thetaDeg, radius, target, fovDeg}` with every render request, so a
Rendered frame is framed like the viewport instead of a canned shot. Pressing
`End` captures the view into the active camera object, which is what camera view
(`0`) shows. When no viewport camera is available (e.g. the shell renders without
one), the fallback frames the terrain at `radius = 1.5·max(span)` aimed at the
mesh centre.

Two things were needed to make the viewer agree with Studio, both of which look
like "nothing happened" if you get them wrong:

1. **The viewer needs the interpreter's `libpython`.** uv/pyenv interpreters keep
   it outside the virtualenv, so the bridge adds `sysconfig` `LIBDIR` to
   `LD_LIBRARY_PATH` before spawning the viewer. Without it the viewer exits
   immediately with "cannot open shared object file".
2. **Georeferenced DEMs live in metres, far from the origin.** The viewer's world
   is `origin = (mapX, -mapY)` with `span = (|a|·w, |e|·h)` and it frames terrain
   at `radius = 1.5·max(span)` around `(span.x/2, relief/2, span.z/2)`. Bryce
   Canyon is 4000 × 3800 m at ≈ 4 165 774 m north, so a pixel-space guess aims at
   empty space and the render is blank. `TerrainObject::viewer_framing` reproduces
   that formula exactly and is pinned by a test.

## Sun from date and place

Use **Lighting → Sun → Sun from
date & place**: latitude, longitude, and a local date/time, then **Set sun**.
The shell asks the engine's SIDERA ephemeris (`forge3d.sun_position`) for the
direction, converts it into the light's rotation (`workspace::sun_rotation`,
one tested axis conversion from MapScene's east/up/north frame), and applies a
single `SetTransform`. Both viewport modes and the renderer read that rotation,
so one undo restores the previous sun. The date and place are inputs to the
calculation, not project state: apply again to change them. A result below the
horizon is applied and reported (`daytime: false`), never clamped.

## Engine discovery

Studio probes a Python interpreter for `import forge3d` (repo `.venv` first, then
`python3`, `FORGE3D_STUDIO_PYTHON` to override) and reports the result in the
status bar — `forge3d 1.47.1 · Vulkan · RTX 5060 Laptop GPU …` when the
renderer can run (the GPU is named in the bar), `engine unavailable` with the
reason when it cannot. The tooltip carries the full backend and adapter, which
is the GPU actually rendering. The engine selects its adapter by ranking
discrete > integrated (hybrid laptops never silently bind the iGPU for a
renderer), the viewport asks WebGL for the high-performance adapter, and the
Tauri web process is the same GPU — `nvidia-smi` shows both the shell and
`WebKitWebProcess` resident on the discrete adapter.

## One light rig for Solid, Material and Rendered

The three modes differ in the *surface* they show, never in how the scene is
lit. That is enforced on both sides of the renderer:

- The preview (three.js) uses exactly the scene's sun — direction, colour in
  linear radiance, and intensity, no fudge factor — plus one environment
  image and the project's exposure. There is no fill light and no hemisphere
  fill while an environment is bound: the engine's terrain shader has neither,
  so adding them was what made the previews brighter than the still.
- The environment is the same image in both places. Without a project HDRI the
  viewport prefilters the *same* gradient the render service writes into its
  studio HDR (`write_sky_hdr`), so Material's orbit and the Rendered frame are
  lit by one sky instead of a three.js studio room vs. an engine sky.
- The still uses the engine's `lambert_physical` terrain model, which evaluates
  `albedo/pi * sun*intensity * n.l + albedo * irradiance` — the same equation
  three.js applies to `MeshStandardMaterial`. The alternative ("stylized")
  model answers the same sun with a hardcoded, style-matched response and lit
  the still roughly four times darker at the same exposure.

**The extension must be built from this checkout.** A stale
`python/forge3d/_forge3d.abi3.so` is what produces errors like
`sky.model must be preetham, hosek-wilkie, approximate, or aether; got "hdri"`:
the Python source accepts a setting the native binary predates. Rebuild with
`maturin develop --release` (or copy `target/release/libforge3d.so` over the
`.so`) and restart the shell. There is no separate viewer window any more: opening
one was a Phase 3 detour, and it has been removed along with its workspace tab.
What survives from it in `workspace.rs` is the handoff itself (terrain, orbit
camera, sun angles, environment) and the per-parameter capability table the
Inspector reads; the object-by-object report that only the removed viewer
consumed is gone.

## Known limitations

- **glTF import reads the file whole** and is capped at 32 MB (streamed asset
  loading is not implemented). Self-contained `.glb` works; a `.gltf` that
  references external `.bin`/texture files fails with an explicit message,
  because the webview cannot read neighbouring files.
- **Imported meshes are transform-editable, relinkable, and node-selectable.**
  Solid draws them as clay; the Material preview draws their glTF materials.
  Meshes do not reach Rendered at all yet (see Primitives below).
- **No shadows or PBR environment** in the Solid viewport; it is a modelling
  view, not the Forge3D render. The Material preview is where HDRI lighting and
  soft shadows live (Phase 4).
- **The engine probe needs only an importable `forge3d`.** Rendering runs
  in-process (`studio_render.py`), so no viewer binary is required; the probe
  still looks for `interactive_viewer` and reports what it finds as optional
  diagnostics in the status-bar tooltip. `FORGE3D_STUDIO_PYTHON` overrides the
  interpreter choice.
- **Primitives still do not reach Forge3D.** The Phase 4 default scene is terrain
  precisely because that path is complete; exporting primitive meshes as glTF is
  the remaining piece of Rendered mode.
- **Object animation (Phases 6) is evaluated in the model and exported per frame,
  but the engine cannot draw those objects**, so only camera animation shows up
  in Rendered frames.
- **Rendered shows one still frame at a time.** Playback animates the Solid and
  Material viewports live; Rendered re-renders on edits, orbit/zoom, and camera
  changes (throttled), not per playback frame. Use **Export frames…** for the
  sequence. Entering Rendered drops the previous frame, so a stale view never
  covers the scene while the new one renders.
- **Rendered re-renders on orbit/zoom and edits.** The previous frame stays on
  screen while the new one renders (with an "Updating…" chip); only Entering
  Rendered drops the frame outright. The viewport Size control replaces the old
  preview scale selector and caps at 4096 px on the longest side *without*
  changing the aspect.
- **The Material preview and Rendered share the HDRI frame** (Y-up, +Z south).
  The terrain renderer forwards Z-up terrain directions in that frame for the
  sky pass, and the sun swizzle maps east/north/up → east/up/south, so the sky
  is upright for both `mesh:zup` and `mesh:zup:north` modes. The terrain
  shading is unchanged (its Z-up hash stays).
- **Some webviews cannot filter half-float textures** (WebKit/ANGLE), which
  three rejects with `INVALID_OPERATION: … sampler type`. The Material preview
  probes once per change and degrades visibly instead: hard shadows → no IBL →
  no shadows, each with a note in the viewport. The HDRI backdrop survives as an
  8-bit copy; only the image-based lighting is dropped.
- **HDRI environments are Radiance `.hdr` only.** Both consumers read RGBE
  (three's `HDRLoader` and the engine's `IBL.from_hdr`); `.exr` is rejected with
  a specific message rather than silently mis-lighting. The viewport reads the
  file whole over IPC (64 MB cap) and keeps one decoded map at a time, so a very
  large HDRI costs a decode (~a second) when first selected or replaced.
- **Asset integrity checks size, not content.** A file replaced in place with a
  different same-sized file passes; this is a relink aid, not tamper detection.
- **The offline queue lives in the UI process.** If the app quits during a
  sequence, the frames already written stay on disk but the job does not resume
  on restart; queue it again.
- **Sample GeoTIFF references need Git LFS.** Without `git lfs pull` the Bryce
  sample loads with its assets flagged missing/changed (the HDRI ships normally).
- **Country-scale DEMs render through a scale budget.** The studio renderer
  scales any DEM into a 30 000-unit world, so wide maps frame correctly
  (verified end-to-end with Switzerland, 353 km — its relief is only ~1 % of
  the span at Vertical scale 1, so raise the slider to read the Alps). The
  older ±65 536 view-coordinate clamp no longer bites in this path
  (`experiments/swiss-land-cover.py` documents the engine-level case).
- **Raster overlays are the terrain's albedo.** The engine has no other terrain
  material override; a draped raster (the Bryce NAIP orthophoto) is how the
  surface gets imagery — the Material preview and Rendered drape it (Material
  decodes the raster through the shell's `raster_preview`), while Solid stays
  neutral clay. The inspector shows it under the terrain's **Material**
  section — `Add albedo raster…` / `Replace…` / `Remove` — alongside what the
  engine actually uses (`MaterialSet.terrain_default()`, four triplanar layers,
  no texture maps shipped in this checkout, so the default is flat colours until
  an albedo is attached or maps appear in `assets/materials/`).

## Timeline (Phase 6)

The animation timeline is a drawer under the viewport, hidden by default. Open
it with the **Timeline** button in the top bar or `T`; **Hide** (or `T` again)
closes it. While closed, the viewport gets the whole centre column. The model lives in
`crates/studio-core/src/timeline.rs` and is persisted with the document (schema
v4; older builds reject a v4 file rather than silently dropping its tracks).

- One `evaluate(project, frame)` samples every keyed channel into the project's
  object transforms. **Playback and frame export both call it** — the session's
  playhead through `Session::set_playhead`, the export through a project clone in
  `render_animation_frame` — so what you scrub is what renders.
- Moving the playhead is **view state**: it does not bump the revision, dirty the
  project, or push undo, so scrubbing never floods the undo stack or autosave.
- Channels are the nine transform scalars plus a camera's FOV, with linear,
  constant, and smooth (smoothstep) interpolation. Terrain cannot be animated
  (its transform is georeferenced); its track is refused with a reason.
- **Key selected** keys the selected object's current values at the playhead;
  **Clear keys** removes the track; **Export frames…** writes one PNG per frame
  into a chosen folder, with live progress and a working Stop.
- Playback is a fixed-rate clock (`src/state/playback.ts`, unit-tested). It
  advances at most one frame per animation tick, so a slow shell runs playback
  slow instead of queueing IPC calls.
- A camera animation reaches Forge3D through `workspace::camera_render_view`:
  the exported frame uses the evaluated camera's eye, rotation, distance, and
  FOV — matching a camera-view still — rather than the terrain's canned framing.
- **Only camera animation is Forge3D-native.** The engine draws terrain, so a
  mesh's animation is visible in the Studio viewport and evaluated for export,
  but it does not appear in Rendered frames (the same geometry-export gap as
  primitives). The UI does not claim otherwise.

## Asset integrity

Every referenced file is checked when the project opens and whenever an asset
is imported or relinked: `studio_core::integrity::asset_integrity` takes a
`stat` closure (the shell passes `std::fs::metadata`; tests pass a map), so the
rules stay pure. The Assets panel shows a `missing` badge (with the OS error) or
`changed` when the file's size differs from what the project recorded, plus a
summary line. A missing file never blocks editing — Relink… is the repair, and
the viewport/renderer still report the real read failure if you render anyway.

## Offline render queue

Stills (**Render**) and frame sequences (**Export frames…**) are FIFO jobs run
by one worker, so the engine never has two offline jobs fighting over it:

- **Progress** is per job (frames done / total) and shown in a queue strip under
  the viewport, with the sequence bar and a Cancel button.
- **Cancelling** a queued job drops it immediately; a running sequence stops
  between frames. A running still is a single frame already inside the
  renderer, so it cannot be interrupted without killing the engine (and any
  viewport preview with it) — it finishes and reports what happened.
- **Capture at enqueue**: a job records the render width/height when it is
  queued, so later settings edits cannot retarget work already started.
- **Closing the project** cancels queued jobs and stops the running sequence.
- The state machine lives in `src/state/renderQueue.ts` and is unit-tested
  (`npm test`); `session.tsx` only does the IPC. Viewport previews are a
  different scheduler: newest-wins, never queued.

## Performance guards

`crates/studio-core/tests/performance.rs` measures the interaction path rather
than micro-benchmarks: 2,000 objects with 500 animated tracks (3 channels × 40
keys) evaluating 240 frames, and a save/load round trip of the same scene. The
budgets are two orders of magnitude above the measured value (≈130 ms and
≈400 ms on the development machine) so they catch accidental quadratic work
without flaking on CI. A 10,000-object / 2,000-track stress case is `#[ignore]`d
and runs with `cargo test -p studio-core --test performance -- --ignored`.
`--nocapture` prints the measured numbers.

## Sample projects

`samples/` ships three projects plus a committed HDRI:

| Sample | Shows |
|---|---|
| `bryce-terrain.studio3d` | terrain-first starter, NAIP drape, HDRI environment |
| `cube-animation.studio3d` | keyed cube (position + rotation), 96 frames at 24 fps |
| `hdri-studio.studio3d` | a sphere lit by `media/studio-sky.hdr` (sun turned down) |

Asset paths are **relative to the sample file**, so the folder is portable. The
GeoTIFF references are Git LFS objects; a plain checkout without `git lfs pull`
still loads the project (integrity marks those two files changed/missing).
Regenerate with `cargo run -p studio-core --example make_samples` from
`studio-tauri/`; `tests/samples.rs` loads, validates, checks each sample's
feature content, and round-trips them through save/load.

## Packaging

`npm run package` runs `tauri build`. `src-tauri/tauri.conf.json` bundles
`deb`, `rpm`, and `appimage` for Linux with the icon set in
`src-tauri/icons/` (regenerate with `scripts/make_icons.py`). The release build
is gated in CI: `.github/workflows/studio-tauri.yml` runs fmt, clippy, `tsc`,
the node tests, both Rust suites on every pull request, and builds + uploads the
deb on pushes to `main`. Windows and macOS installers are not configured yet;
their bundle config needs a signing/notarisation decision.

## Roadmap: the rest of the engine

Studio currently draws a terrain frame through one narrow engine path. The
phased plan for bringing the rest of Forge3D in — scene recipes, look
development, cartography, geo data, meshes, bundles and evidence — is in
[`docs/forge3d-integration-roadmap.md`](docs/forge3d-integration-roadmap.md).
It lists the actual engine APIs behind each phase and the readiness of each
(including the layers the engine itself still marks underdeveloped).

## No browser fallback

`src/bridge/` speaks Tauri IPC when `__TAURI_INTERNALS__` exists; anywhere else
the app renders a "start the desktop shell" notice instead of mounting the
editor. There used to be an in-memory mock of the whole command layer so
`npm run dev` could be used for UI work, but it was a second, untested
implementation of `studio-core` (54 command cases, its own undo/redo and
timeline sampling) and it drifted silently. `npm run dev` now only serves the
UI bundle; visual checks happen in the Tauri window.

The logic that used to be mirrored there is covered where it lives: the timeline
by `crates/studio-core/tests/phase6.rs`, and the viewport clock and camera math
by the dependency-free `node` tests `npm test` runs.

## Not yet implemented (deliberately)

- Geometry export for primitives — Rendered still draws terrain only
- Shared material library — **Phase 5** (environment/IBL controls, capability badges, the Material-preview albedo drape, and glTF node selection shipped)
Relinking, the material inspector (terrain albedo, mesh base colour and
texture), the render settings UI, the Phase 6 timeline, and the Phase 7 queue,
integrity checks, and samples already shipped; there is no Forge3D workspace tab
(Phase 3 was dropped) — the top-bar **Render** button renders to a chosen path
with a JSON sidecar.

## Regenerating the app icon

```bash
python3 scripts/make_icons.py   # writes src-tauri/icons/*
```

## License and the Pro boundary

Studio lives inside the forge3d repository and carries the same licence as the
engine: **Apache-2.0 OR MIT** (`../LICENSE`, `../LICENSE-APACHE`; the Cargo
workspace sets `license = "Apache-2.0 OR MIT"` and this package repeats it).
Contributions to the shell land under that licence, like any other part of the
repository.

The engine's Pro-gated surface — `MapPlate` composition, SVG/PDF export, Mapbox
style loading, the building import pipelines and scene bundles
(`_check_pro_access`) — is **not used by Studio**: the shell renders through the
open-source terrain path (`TerrainRenderParams`, `Legend`, `thematic`, the
viewer IPC) and needs no licence key. That is also why the map plate is a
blocked feature in the roadmap: shipping it would make the shell require a key.
