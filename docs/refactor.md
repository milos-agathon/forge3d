# Viewer Mod.rs Refactoring Plan

## 1. Constraints and Rules Enforced

Based on AGENTS.md and `docs/codebase-refactor.md`, the following rules govern this refactoring:

- **≤300 LOC per file** (excluding license headers) — mandatory split threshold
- **Refactor safely in tiny steps** — keep a working baseline; verify frequently
- **No behavior changes** — structural moves only; any semantic change is a separate task
- **`cargo check` and `cargo test` must pass** after every atomic commit
- **No broken builds left behind** — rollback if verification fails
- **Preserve public API** — re-export in mod.rs to maintain Rust and Python compatibility
- **Respect Rust/Python boundary** — core GPU/rendering stays in Rust; Python orchestrates/tests
- **Single responsibility** — each file/struct has one dominant reason to change
- **Meaningful names** — use verbs for functions, nouns for data structures
- **Acyclic dependencies** — avoid circular imports by careful ordering
- **Comment why, not what** — preserve existing doc comments, don't add noise
- **RAII ownership** — GPU resources live with their owners; command encoding is separate
- **No speculative flexibility** — extract what exists, don't invent abstractions
- **Tests in Python** — Rust tests optional but Python test suite is the acceptance gate

---

## 2. Inventory of Current src/viewer/mod.rs

### 2.1 File Statistics
- **Total lines:** 8,668
- **Location:** `@/Users/mpopovic3/forge3d/src/viewer/mod.rs`

### 2.2 Top-Level Public Exports

| Symbol | Kind | Consumers |
|--------|------|-----------|
| [Viewer](mod.rs:114:0-320:1) | struct | Internal (viewer event loop) |
| `ViewerConfig` | struct (re-export) | [lib.rs](lib.rs:0:0-0:0), [cli/interactive_viewer.rs](interactive_viewer.rs:0:0-0:0), Python |
| [run_viewer](mod.rs:6949:0-8437:1) | fn | [lib.rs](lib.rs:0:0-0:0), [cli/interactive_viewer.rs](interactive_viewer.rs:0:0-0:0) |
| [run_viewer_with_ipc](mod.rs:8471:0-8666:1) | fn | [cli/interactive_viewer.rs](interactive_viewer.rs:0:0-0:0) |
| `set_initial_commands` | fn | [lib.rs](lib.rs:0:0-0:0), [cli/interactive_viewer.rs](interactive_viewer.rs:0:0-0:0) |
| `set_initial_terrain_config` | fn | [lib.rs](lib.rs:0:0-0:0) |
| `ViewerCmd` | enum (re-export) | IPC, event loop, CLI |
| `VizMode` | enum (re-export) | Viewer state |
| `GiVizMode` | enum (re-export) | Viewer state |
| `IpcServerConfig` | struct (from `ipc` submodule) | [cli/interactive_viewer.rs](interactive_viewer.rs:0:0-0:0) |

### 2.3 Major Responsibilities Mixed in mod.rs

| Responsibility | Line Range (approx) | LOC Est. |
|----------------|---------------------|----------|
| Imports & module declarations | 1-150 | 150 |
| Uniform/param structs (`SkyUniforms`, `FogCameraUniforms`, `VolumetricUniformsStd140`, etc.) | 150-350 | 200 |
| [Viewer::new](mod.rs:1144:4-3164:5) (device/surface/pipeline init) | 350-3165 | **2,800** |
| [Viewer::window](mod.rs:3166:4-3168:5), [resize](mod.rs:3194:4-3451:5), [update_lit_uniform](mod.rs:3170:4-3192:5) | 3167-3195 | 30 |
| [Viewer::resize](mod.rs:3194:4-3451:5) (texture recreation) | 3195-3452 | 260 |
| [Viewer::handle_input](mod.rs:3453:4-3539:5) (keyboard/mouse) | 3454-3540 | 90 |
| [Viewer::update](mod.rs:3541:4-3610:5) (camera, GI sync) | 3542-3612 | 70 |
| [Viewer::render](mod.rs:3612:4-5222:5) (main render loop) | 3613-5224 | **1,600** |
| P5 capture helpers ([dump_gbuffer_artifacts](mod.rs:5226:4-5450:5), `capture_*`) | 5226-5852 | 630 |
| [Viewer::handle_cmd](mod.rs:5856:4-6855:5) (command dispatch) | 5856-6948 | **1,100** |
| [upload_mesh](mod.rs:6857:4-6940:5), [load_albedo_texture](mod.rs:6942:4-6946:5) | 6858-6948 | 90 |
| [run_viewer](mod.rs:6949:0-8437:1) (event loop entry) | 6950-8438 | **1,500** |
| [run_viewer_with_ipc](mod.rs:8471:0-8666:1) | 8472-8668 | 200 |
| IPC queue/stats helpers | 8440-8470 | 30 |

### 2.4 Embedded WGSL Shaders

| Label | Location | Current Loading |
|-------|----------|-----------------|
| `viewer.gbuf.geom.shader` | ~L700 | inline `include_str!("../shaders/gbuffer/geom.wgsl")` |
| `viewer.comp.shader` | ~L850 | inline raw WGSL string |
| `viewer.fallback.shader` | ~L1050 | inline raw WGSL string |
| `viewer.lit.shader` | ~L1150 | `include_str!("../shaders/lit.wgsl")` |
| `viewer.gi.baseline.shader` | ~L1400 | inline raw WGSL string |
| `viewer.gi.split.shader` | ~L1600 | inline raw WGSL string |
| `viewer.sky.shader` | ~L2050 | `include_str!("../shaders/procedural_sky.wgsl")` |
| `viewer.fog.shader` | ~L2400 | `include_str!("../shaders/fog_volumetric.wgsl")` |
| `viewer.fog.upsample.shader` | ~L2890 | `include_str!("../shaders/fog_upsample.wgsl")` |
| `viewer.csm.depth.shader` | ~L2150 | inline raw WGSL string |

---

## 3. Upstream Callers and Downstream Dependencies

### 3.1 Upstream Callers (Rust)

| File | Symbols Used |
|------|--------------|
| `@/Users/mpopovic3/forge3d/src/lib.rs:3066` | [run_viewer](mod.rs:6949:0-8437:1), `set_initial_commands`, `ViewerConfig` |
| `@/Users/mpopovic3/forge3d/src/lib.rs:3163` | [run_viewer](mod.rs:6949:0-8437:1), `set_initial_commands`, `set_initial_terrain_config`, `ViewerConfig` |
| `@/Users/mpopovic3/forge3d/src/cli/interactive_viewer.rs:4` | [run_viewer](mod.rs:6949:0-8437:1), [run_viewer_with_ipc](mod.rs:8471:0-8666:1), `set_initial_commands`, `ViewerConfig`, `ipc::IpcServerConfig` |

### 3.2 Downstream Dependencies (Existing Viewer Submodules)

| Submodule | Size (bytes) | Est. LOC | Status |
|-----------|--------------|----------|--------|
| [camera_controller.rs](camera_controller.rs:0:0-0:0) | 8,184 | ~280 | OK |
| [hud.rs](hud.rs:0:0-0:0) | 6,490 | ~220 | OK |
| [image_analysis.rs](image_analysis.rs:0:0-0:0) | 7,788 | ~260 | OK |
| [ipc.rs](ipc.rs:0:0-0:0) | 17,215 | ~580 | **Needs split** |
| [viewer_analysis.rs](viewer_analysis.rs:0:0-0:0) | 2,822 | ~95 | OK |
| [viewer_cmd_parse.rs](viewer_cmd_parse.rs:0:0-0:0) | 10,471 | ~350 | **Needs split** |
| [viewer_config.rs](viewer_config.rs:0:0-0:0) | 2,510 | ~85 | OK |
| [viewer_constants.rs](viewer_constants.rs:0:0-0:0) | 932 | ~30 | OK |
| [viewer_enums.rs](viewer_enums.rs:0:0-0:0) | 5,998 | ~200 | OK |
| [viewer_image_utils.rs](viewer_image_utils.rs:0:0-0:0) | 5,741 | ~190 | OK |
| [viewer_p5.rs](viewer_p5.rs:0:0-0:0) | 18,839 | ~630 | **Needs split** |
| [viewer_p5_ao.rs](viewer_p5_ao.rs:0:0-0:0) | 14,415 | ~480 | **Needs split** |
| [viewer_p5_cornell.rs](viewer_p5_cornell.rs:0:0-0:0) | 8,058 | ~270 | OK |
| [viewer_p5_gi.rs](viewer_p5_gi.rs:0:0-0:0) | 19,175 | ~640 | **Needs split** |
| [viewer_p5_ssr.rs](viewer_p5_ssr.rs:0:0-0:0) | 14,516 | ~490 | **Needs split** |
| [viewer_render_helpers.rs](viewer_render_helpers.rs:0:0-0:0) | 4,543 | ~150 | OK |
| [viewer_ssr_scene.rs](viewer_ssr_scene.rs:0:0-0:0) | 4,132 | ~140 | OK |
| [viewer_struct.rs](viewer_struct.rs:0:0-0:0) | 9,483 | ~320 | **Needs split** |
| [viewer_terrain.rs](viewer_terrain.rs:0:0-0:0) | 25,143 | ~840 | **Needs split** |
| [viewer_types.rs](viewer_types.rs:0:0-0:0) | 5,191 | ~175 | OK |

### 3.3 Python/CLI Entrypoints

1. **Python `open_3d_viewer()`** → `lib.rs:open_3d_viewer()` → [run_viewer()](mod.rs:6949:0-8437:1)
2. **Python `open_terrain_viewer()`** → `lib.rs:open_terrain_viewer()` → [run_viewer()](mod.rs:6949:0-8437:1)
3. **CLI `cargo run --bin interactive_viewer`** → `cli/interactive_viewer.rs:main()` → [run_viewer()](mod.rs:6949:0-8437:1) or [run_viewer_with_ipc()](mod.rs:8471:0-8666:1)

---

## 4. Target Module Tree (≤300 LOC per file)

```
src/viewer/
├── mod.rs                      (~250 LOC) - orchestrator, re-exports
├── camera_controller.rs        (existing, ~280 LOC) - OK
├── hud.rs                      (existing, ~220 LOC) - OK
├── ipc.rs → ipc/mod.rs         (split into submodules)
│   ├── mod.rs                  (~100 LOC) - re-exports
│   ├── server.rs               (~250 LOC) - TCP server
│   └── protocol.rs             (~230 LOC) - message parsing
├── image_analysis.rs           (existing, ~260 LOC) - OK
├── viewer_analysis.rs          (existing, ~95 LOC) - OK
├── viewer_cmd_parse.rs → cmd/  (split)
│   ├── mod.rs                  (~100 LOC) - re-exports
│   ├── gi_commands.rs          (~150 LOC) - GI/SSAO/SSR parsing
│   └── scene_commands.rs       (~150 LOC) - scene/viz/fog parsing
├── viewer_config.rs            (existing, ~85 LOC) - OK
├── viewer_constants.rs         (existing, ~30 LOC) - OK
├── viewer_enums.rs             (existing, ~200 LOC) - OK
├── viewer_image_utils.rs       (existing, ~190 LOC) - OK
├── viewer_render_helpers.rs    (existing, ~150 LOC) - OK
├── viewer_ssr_scene.rs         (existing, ~140 LOC) - OK
├── viewer_types.rs             (existing, ~175 LOC) - OK
│
├── state/                      (NEW - Viewer struct decomposition)
│   ├── mod.rs                  (~80 LOC) - re-exports
│   ├── gpu_state.rs            (~280 LOC) - device, queue, adapter, surface
│   ├── render_state.rs         (~280 LOC) - pipelines, bind group layouts
│   ├── gi_state.rs             (~250 LOC) - GI manager, weights, timing
│   ├── fog_state.rs            (~280 LOC) - fog textures, pipelines, params
│   ├── sky_state.rs            (~200 LOC) - sky pipeline, params
│   └── scene_state.rs          (~200 LOC) - mesh buffers, transforms
│
├── init/                       (NEW - Viewer::new decomposition)
│   ├── mod.rs                  (~100 LOC) - orchestrates init sequence
│   ├── device_init.rs          (~250 LOC) - instance, adapter, device, surface
│   ├── gbuffer_init.rs         (~280 LOC) - GBuffer pipelines, textures
│   ├── lit_init.rs             (~250 LOC) - lit compute pipeline
│   ├── gi_init.rs              (~280 LOC) - GI baseline/split pipelines
│   ├── sky_init.rs             (~280 LOC) - sky compute pipeline
│   ├── fog_init.rs             (~300 LOC) - fog raymarch/froxel pipelines
│   └── csm_init.rs             (~200 LOC) - CSM shadow pipeline
│
├── render/                     (NEW - render loop decomposition)
│   ├── mod.rs                  (~100 LOC) - orchestrates render sequence
│   ├── sky_pass.rs             (~200 LOC) - sky compute dispatch
│   ├── geometry_pass.rs        (~280 LOC) - GBuffer geometry pass
│   ├── fog_pass.rs             (~300 LOC) - volumetric fog dispatch
│   ├── gi_pass.rs              (~200 LOC) - HZB build, GI execute
│   ├── lit_pass.rs             (~200 LOC) - lit compute
│   ├── composite_pass.rs       (~280 LOC) - final composite + HUD
│   └── snapshot.rs             (~200 LOC) - snapshot capture
│
├── input/                      (NEW - input handling)
│   ├── mod.rs                  (~80 LOC) - re-exports
│   ├── keyboard.rs             (~150 LOC) - key handling
│   └── mouse.rs                (~150 LOC) - mouse/scroll handling
│
├── event_loop/                 (NEW - run_viewer decomposition)
│   ├── mod.rs                  (~100 LOC) - re-exports run_viewer, run_viewer_with_ipc
│   ├── run_viewer.rs           (~300 LOC) - main event loop
│   ├── run_viewer_ipc.rs       (~300 LOC) - IPC event loop
│   └── cmd_parsing.rs          (~300 LOC) - stdin command parsing (from run_viewer)
│
├── cmd/                        (NEW - handle_cmd decomposition)
│   ├── mod.rs                  (~100 LOC) - dispatch to handlers
│   ├── gi_handlers.rs          (~280 LOC) - GI/SSAO/SSR/SSGI commands
│   ├── scene_handlers.rs       (~250 LOC) - viz/obj/gltf/transform commands
│   ├── sky_fog_handlers.rs     (~200 LOC) - sky/fog commands
│   ├── ibl_handlers.rs         (~200 LOC) - IBL commands
│   └── capture_handlers.rs     (~150 LOC) - P5 capture commands
│
├── p5/                         (CONSOLIDATE existing viewer_p5_*.rs)
│   ├── mod.rs                  (~100 LOC) - re-exports
│   ├── gbuffer_dump.rs         (~250 LOC) - dump_gbuffer_artifacts
│   ├── cornell.rs              (~270 LOC) - P5.1 Cornell captures (existing viewer_p5_cornell.rs)
│   ├── ao_grid.rs              (~280 LOC) - P5.1 AO grid (from viewer_p5_ao.rs)
│   ├── ao_sweep.rs             (~200 LOC) - P5.1 AO sweep (from viewer_p5_ao.rs)
│   ├── ssgi.rs                 (~300 LOC) - P5.2 SSGI (from viewer_p5_gi.rs split)
│   ├── ssgi_temporal.rs        (~280 LOC) - P5.2 SSGI temporal (from viewer_p5_gi.rs split)
│   ├── ssr_glossy.rs           (~280 LOC) - P5.3 SSR glossy (from viewer_p5_ssr.rs split)
│   ├── ssr_thickness.rs        (~210 LOC) - P5.3 SSR thickness (from viewer_p5_ssr.rs split)
│   └── gi_stack.rs             (~250 LOC) - P5.4 GI stack (from viewer_p5.rs)
│
├── terrain/                    (SPLIT existing viewer_terrain.rs)
│   ├── mod.rs                  (~100 LOC) - re-exports
│   ├── scene.rs                (~280 LOC) - ViewerTerrainScene struct
│   ├── render.rs               (~280 LOC) - terrain render pass
│   └── camera.rs               (~280 LOC) - terrain camera handling
│
└── uniforms.rs                 (NEW - uniform struct definitions)
    (~250 LOC) - SkyUniforms, FogCameraUniforms, VolumetricUniformsStd140, etc.
```

**Total new files: ~45** | **All files ≤300 LOC**

---

## 5. Move Map (old → new)

| Old Location (mod.rs lines) | Content | New File |
|-----------------------------|---------|----------|
| 1-50 | Imports | [mod.rs](mod.rs:0:0-0:0) (keep minimal) |
| 51-150 | Module declarations | [mod.rs](mod.rs:0:0-0:0) |
| 150-350 | Uniform structs | `uniforms.rs` |
| 350-1200 | [Viewer::new](mod.rs:1144:4-3164:5) (device init) | `init/device_init.rs` |
| 1200-1600 | [Viewer::new](mod.rs:1144:4-3164:5) (GBuffer init) | `init/gbuffer_init.rs` |
| 1600-1800 | [Viewer::new](mod.rs:1144:4-3164:5) (lit init) | `init/lit_init.rs` |
| 1800-2100 | [Viewer::new](mod.rs:1144:4-3164:5) (GI init) | `init/gi_init.rs` |
| 2050-2450 | [Viewer::new](mod.rs:1144:4-3164:5) (sky init) | `init/sky_init.rs` |
| 2450-2970 | [Viewer::new](mod.rs:1144:4-3164:5) (fog init) | `init/fog_init.rs` |
| 2970-3165 | [Viewer::new](mod.rs:1144:4-3164:5) (CSM, HUD, final) | `init/csm_init.rs` + `init/mod.rs` |
| 3167-3195 | [window](mod.rs:3166:4-3168:5), [update_lit_uniform](mod.rs:3170:4-3192:5) | `state/render_state.rs` |
| 3195-3452 | [resize](mod.rs:3194:4-3451:5) | `state/gpu_state.rs` |
| 3454-3540 | [handle_input](mod.rs:3453:4-3539:5) | `input/mod.rs` |
| 3542-3612 | [update](mod.rs:3541:4-3610:5) | `render/mod.rs` |
| 3613-3730 | Sky render | `render/sky_pass.rs` |
| 3730-3965 | Geometry render | `render/geometry_pass.rs` |
| 3965-4430 | Fog render | `render/fog_pass.rs` |
| 4430-4510 | Pre-SSR lit + HZB/GI | `render/gi_pass.rs` |
| 4510-4785 | Lit compute + composite | `render/lit_pass.rs` + `render/composite_pass.rs` |
| 4785-5224 | HUD, fallback, present | `render/composite_pass.rs` |
| 5226-5450 | [dump_gbuffer_artifacts](mod.rs:5226:4-5450:5) | `p5/gbuffer_dump.rs` |
| 5450-5852 | P5.1 Cornell helpers | `p5/cornell.rs` |
| 5856-6250 | [handle_cmd](mod.rs:5856:4-6855:5) (SSAO/GI queries) | `cmd/gi_handlers.rs` |
| 6250-6500 | [handle_cmd](mod.rs:5856:4-6855:5) (sky/fog) | `cmd/sky_fog_handlers.rs` |
| 6500-6700 | [handle_cmd](mod.rs:5856:4-6855:5) (IBL) | `cmd/ibl_handlers.rs` |
| 6700-6860 | [handle_cmd](mod.rs:5856:4-6855:5) (transform, terrain) | `cmd/scene_handlers.rs` |
| 6858-6948 | [upload_mesh](mod.rs:6857:4-6940:5), [load_albedo_texture](mod.rs:6942:4-6946:5) | `state/scene_state.rs` |
| 6950-7500 | [run_viewer](mod.rs:6949:0-8437:1) (pending_cmds parsing) | `event_loop/cmd_parsing.rs` |
| 7500-8310 | [run_viewer](mod.rs:6949:0-8437:1) (stdin thread) | `event_loop/cmd_parsing.rs` |
| 8310-8438 | [run_viewer](mod.rs:6949:0-8437:1) (event loop) | `event_loop/run_viewer.rs` |
| 8440-8470 | IPC helpers | `ipc/mod.rs` |
| 8472-8668 | [run_viewer_with_ipc](mod.rs:8471:0-8666:1) | `event_loop/run_viewer_ipc.rs` |

### Cycle Avoidance Strategy

1. **Dependency direction:** [mod.rs](mod.rs:0:0-0:0) → `init/*` → `state/*` → `uniforms.rs`
2. **Render depends on state:** `render/*` → `state/*`
3. **Commands depend on state:** `cmd/*` → `state/*`
4. **No cycles:** `state/*` modules do not import from `init/*`, `render/*`, or `cmd/*`
5. **Shared types in `uniforms.rs` and [viewer_types.rs](viewer_types.rs:0:0-0:0)** — both are leaf modules

---

## 6. Step-by-Step Refactor Plan with Verification Commands

### Verification Commands (run after each step)
```bash
cargo fmt
cargo check --all-features
cargo test --all-features -- --test-threads=1
pytest tests/ -v --tb=short -x
```

### Phase 0: Baseline Snapshot
```bash
# Capture git state
git stash  # if needed
git checkout -b refactor/viewer-split
cargo check --all-features && cargo test --all-features -- --test-threads=1
pytest tests/ -v --tb=short
# Record passing test count
```

---

### Step 1: Extract `uniforms.rs` (uniform structs)

**Action:** Move `SkyUniforms`, `FogCameraUniforms`, `VolumetricUniformsStd140`, `FogUpsampleParamsStd140` from [mod.rs](mod.rs:0:0-0:0) to new `uniforms.rs`.

**Edits:**
1. Create `src/viewer/uniforms.rs` with structs
2. Add `mod uniforms;` + `pub use uniforms::*;` to [mod.rs](mod.rs:0:0-0:0)
3. Remove struct definitions from [mod.rs](mod.rs:0:0-0:0)

**Risks:** None — pure data structs with `#[repr(C)]`, no complex deps.

**Rollback:** `git checkout src/viewer/mod.rs src/viewer/uniforms.rs`

---

### Step 2: Extract `state/` module (Viewer sub-structs)

**Action:** Create state decomposition structs and move relevant Viewer fields.

**Substeps:**
- 2a: Create `state/mod.rs` with stubs
- 2b: Extract `GpuState` (device, queue, adapter, surface, config)
- 2c: Extract `RenderState` (pipelines, bind group layouts)
- 2d: Extract `GiState` (gi manager, weights, timing)
- 2e: Extract `FogState` (fog textures, pipelines, params)
- 2f: Extract `SkyState` (sky pipeline, params)
- 2g: Extract `SceneState` (mesh buffers, transforms)

**Risks:** Borrow checker issues if state is split across structs while Viewer holds all. Mitigation: Use composition pattern where Viewer holds sub-structs.

**Rollback:** `git checkout -- src/viewer/`

---

### Step 3: Extract `init/` module (Viewer::new decomposition)

**Action:** Split [Viewer::new](mod.rs:1144:4-3164:5) into init functions returning sub-state structs.

**Substeps:**
- 3a: `init/device_init.rs` — instance, adapter, device, surface
- 3b: `init/gbuffer_init.rs` — GBuffer pipeline setup
- 3c: `init/lit_init.rs` — lit compute pipeline
- 3d: `init/gi_init.rs` — GI baseline/split pipelines
- 3e: `init/sky_init.rs` — sky compute pipeline
- 3f: `init/fog_init.rs` — fog pipelines and textures
- 3g: `init/csm_init.rs` — CSM shadow pipeline
- 3h: `init/mod.rs` — orchestrator calling all init functions

**Risks:** Lifetime issues with winit Window/Surface. Mitigation: Keep Window Arc<> in GpuState, pass references to init functions.

**Rollback:** `git checkout -- src/viewer/`

---

### Step 4: Extract `input/` module

**Action:** Move [handle_input](mod.rs:3453:4-3539:5) to `input/mod.rs`, split keyboard/mouse.

**Edits:**
1. Create `input/mod.rs`, `input/keyboard.rs`, `input/mouse.rs`
2. Move keyboard handling to `keyboard.rs`
3. Move mouse/scroll handling to `mouse.rs`
4. Update [mod.rs](mod.rs:0:0-0:0) to call [input::handle_input](mod.rs:3453:4-3539:5)

**Risks:** Low — isolated functionality.

---

### Step 5: Extract [render/](mod.rs:3612:4-5222:5) module

**Action:** Decompose [render()](mod.rs:3612:4-5222:5) into pass modules.

**Substeps:**
- 5a: `render/sky_pass.rs` — sky compute dispatch
- 5b: `render/geometry_pass.rs` — GBuffer geometry pass
- 5c: `render/fog_pass.rs` — volumetric fog dispatch
- 5d: `render/gi_pass.rs` — HZB build, GI execute
- 5e: `render/lit_pass.rs` — lit compute
- 5f: `render/composite_pass.rs` — final composite + HUD
- 5g: `render/snapshot.rs` — snapshot capture
- 5h: `render/mod.rs` — orchestrates render sequence

**Risks:** Complex state threading through passes. Mitigation: Pass `&Viewer` or sub-state refs to each pass function.

---

### Step 6: Extract `cmd/` module (handle_cmd decomposition)

**Action:** Split [handle_cmd](mod.rs:5856:4-6855:5) match arms into handler modules.

**Substeps:**
- 6a: `cmd/gi_handlers.rs` — GI/SSAO/SSR/SSGI commands
- 6b: `cmd/scene_handlers.rs` — viz/obj/gltf/transform commands
- 6c: `cmd/sky_fog_handlers.rs` — sky/fog commands
- 6d: `cmd/ibl_handlers.rs` — IBL commands
- 6e: `cmd/capture_handlers.rs` — P5 capture commands
- 6f: `cmd/mod.rs` — dispatch to handlers

**Risks:** Large enum match fragmentation. Mitigation: Keep dispatch in `cmd/mod.rs`, call into handler modules.

---

### Step 7: Extract `event_loop/` module

**Action:** Move [run_viewer](mod.rs:6949:0-8437:1) and [run_viewer_with_ipc](mod.rs:8471:0-8666:1) to dedicated module.

**Substeps:**
- 7a: `event_loop/cmd_parsing.rs` — stdin command parsing
- 7b: `event_loop/run_viewer.rs` — main event loop
- 7c: `event_loop/run_viewer_ipc.rs` — IPC event loop
- 7d: `event_loop/mod.rs` — re-exports

**Risks:** Global statics (`INITIAL_CMDS`, `IPC_QUEUE`). Mitigation: Keep statics in [mod.rs](mod.rs:0:0-0:0) or dedicated `globals.rs`.

---

### Step 8: Split existing oversized submodules

**Action:** Split files exceeding 300 LOC.

| File | Split Into |
|------|-----------|
| [ipc.rs](ipc.rs:0:0-0:0) (~580 LOC) | `ipc/mod.rs`, `ipc/server.rs`, `ipc/protocol.rs` |
| [viewer_cmd_parse.rs](viewer_cmd_parse.rs:0:0-0:0) (~350 LOC) | Already handled by `cmd/` extraction |
| [viewer_p5.rs](viewer_p5.rs:0:0-0:0) (~630 LOC) | `p5/gi_stack.rs` + consolidate |
| [viewer_p5_ao.rs](viewer_p5_ao.rs:0:0-0:0) (~480 LOC) | `p5/ao_grid.rs`, `p5/ao_sweep.rs` |
| [viewer_p5_gi.rs](viewer_p5_gi.rs:0:0-0:0) (~640 LOC) | `p5/ssgi.rs`, `p5/ssgi_temporal.rs` |
| [viewer_p5_ssr.rs](viewer_p5_ssr.rs:0:0-0:0) (~490 LOC) | `p5/ssr_glossy.rs`, `p5/ssr_thickness.rs` |
| [viewer_struct.rs](viewer_struct.rs:0:0-0:0) (~320 LOC) | Merge into `state/` or split |
| [viewer_terrain.rs](viewer_terrain.rs:0:0-0:0) (~840 LOC) | `terrain/scene.rs`, `terrain/render.rs`, `terrain/camera.rs` |

---

### Step 9: Final Cleanup

**Action:**
1. Ensure [mod.rs](mod.rs:0:0-0:0) ≤300 LOC (only imports, re-exports, statics)
2. Run full verification suite
3. Update any broken doc comments
4. Squash/rebase commits

---

## 7. Shader Extraction Plan

| Current Label | Proposed File | Loading Method |
|---------------|---------------|----------------|
| `viewer.comp.shader` (inline) | `src/shaders/viewer/composite.wgsl` | `include_str!` |
| `viewer.fallback.shader` (inline) | `src/shaders/viewer/fallback.wgsl` | `include_str!` |
| `viewer.gi.baseline.shader` (inline) | `src/shaders/viewer/gi_baseline.wgsl` | `include_str!` |
| `viewer.gi.split.shader` (inline) | `src/shaders/viewer/gi_split.wgsl` | `include_str!` |
| `viewer.csm.depth.shader` (inline) | `src/shaders/viewer/csm_depth.wgsl` | `include_str!` |

**Existing external shaders (no change needed):**
- `src/shaders/gbuffer/geom.wgsl`
- `src/shaders/lit.wgsl`
- `src/shaders/procedural_sky.wgsl`
- `src/shaders/fog_volumetric.wgsl`
- `src/shaders/fog_upsample.wgsl`

**Validation:**
```bash
# Before extraction - hash inline shaders
sha256sum <(grep -Pzo '(?s)let.*shader.*=.*create_shader_module.*wgsl::ShaderSource::Wgsl\(.*?\.into\(\)\)' src/viewer/mod.rs)

# After extraction
sha256sum src/shaders/viewer/*.wgsl
# Hashes must match (byte-identical content)
```

---

## 8. Risks, Unknowns, and Stop-Conditions

### Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Borrow checker rejects state decomposition | High | Use `&mut self` methods that take sub-state by ref, or keep monolithic Viewer with impl methods in submodules |
| Winit surface lifetime tied to window | Medium | Keep `Arc<Window>` and surface in same `GpuState` struct |
| Feature-gated code (`#[cfg(feature = "extension-module")]`) | Medium | Maintain all feature gates during moves |
| Shader include paths change | Low | Use consistent `include_str!("../shaders/...")` relative paths |
| Python API breakage | High | Preserve all `pub use` re-exports; test with `pytest` after every step |
| GPU resource leaks during refactor | Medium | Run P5 captures before/after to verify identical output |

### Unknowns

1. **Exact LOC after split** — estimates may be off by 10-15%
2. **Hidden coupling** — some methods may have unexpected cross-dependencies
3. **Test coverage gaps** — some viewer paths may not be covered by `pytest tests/`

### Stop-Conditions (from AGENTS.md)

1. **STOP if `cargo check` fails** — fix before proceeding
2. **STOP if `pytest tests/` has new failures** — investigate before proceeding
3. **STOP if any file exceeds 300 LOC** — split further before proceeding
4. **STOP if public API changes** — add re-exports or revert
5. **STOP if shader content changes** — verify byte-identical before proceeding
6. **STOP if GPU memory usage increases** — profile before/after

### Final Acceptance Criteria

| Criterion | Verification |
|-----------|--------------|
| [src/viewer/mod.rs](mod.rs:0:0-0:0) ≤ 300 LOC | `wc -l src/viewer/mod.rs` |
| No viewer file exceeds 300 LOC | `find src/viewer -name '*.rs' -exec wc -l {} + | sort -n | tail -20` |
| `cargo check --all-features` passes | CI / local |
| `cargo test --all-features` passes | CI / local |
| `pytest tests/ -v` passes | CI / local |
| Viewer still runs interactively | `cargo run --bin interactive_viewer -- --width 640 --height 480` |
| P5 captures produce identical output | Hash comparison of `reports/p5/*.png` |

---

## Summary

The refactoring plan for [src/viewer/mod.rs](mod.rs:0:0-0:0) is complete. Key deliverables:

- **Current state:** 8,668 LOC monolithic file with mixed responsibilities
- **Target:** ~45 files across 9 subdirectories, all ≤300 LOC
- **Major decomposition areas:**
  - `state/` — Viewer sub-structs (GPU, render, GI, fog, sky, scene)
  - `init/` — [Viewer::new](mod.rs:1144:4-3164:5) decomposition (7 init modules)
  - [render/](mod.rs:3612:4-5222:5) — Render loop decomposition (7 pass modules)
  - `cmd/` — Command handling (5 handler modules)
  - `event_loop/` — [run_viewer](mod.rs:6949:0-8437:1) decomposition (3 modules)
  - `p5/` — Consolidated P5 captures (9 modules)
  - `terrain/` — Split from 840 LOC file (3 modules)

- **9 execution steps** with verification after each
- **5 inline WGSL shaders** to extract to `src/shaders/viewer/`
- **8 existing submodules** flagged for splitting (>300 LOC)

**Verification commands:**
```bash
cargo fmt && cargo check --all-features
cargo test --all-features -- --test-threads=1
pytest tests/ -v --tb=short -x
```

**Acceptance criteria:** All viewer files ≤300 LOC, all tests pass, viewer runs interactively, P5 captures identical.