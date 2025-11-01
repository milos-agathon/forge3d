Here’s what remains to fully implement p5.md and the concrete deliverables to close it. I audited the repo and mapped the spec to what’s already in place vs. what still needs work.

Findings (what’s already implemented)
- GBuffer
  - GBuffer allocation and formats are implemented in [src/core/gbuffer.rs](cci:7://file:///Users/mpopovic3/forge3d/src/core/gbuffer.rs:0:0-0:0) (depth R32F, normals RGBA16F, material RGBA8).
  - The viewer raster path fills GBuffer attachments from real geometry in [src/viewer/mod.rs](cci:7://file:///Users/mpopovic3/forge3d/src/viewer/mod.rs:0:0-0:0) (VS passes view-space normals/depth; FS writes to GBuffer).
  - WGSL utility for gbuffer is present in [src/shaders/gbuffer.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/gbuffer.wgsl:0:0-0:0) (not used by viewer’s inline shader, which is OK).

- SSAO/GTAO
  - SSAO compute and bilateral blur passes are implemented in [shaders/ssao.wgsl](cci:7://file:///Users/mpopovic3/forge3d/shaders/ssao.wgsl:0:0-0:0). GTAO is already toggled via `SsaoSettings.technique` and used in-shader.
  - Renderer scaffolding and settings plumbing exist in [src/core/screen_space_effects.rs::SsaoRenderer](cci:2://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:129:0-154:1).
  - Viewer toggles and runtime parameter updates are wired (`:gi ssao on|off`, `:ssao-radius`, `:ssao-intensity`, `:ssao-technique`) in [src/viewer/mod.rs](cci:7://file:///Users/mpopovic3/forge3d/src/viewer/mod.rs:0:0-0:0) and [examples/interactive_viewer.rs](cci:7://file:///Users/mpopovic3/forge3d/examples/interactive_viewer.rs:0:0-0:0).

- SSGI
  - Compute + temporal accumulation pipelines exist in [src/core/screen_space_effects.rs::SsgiRenderer](cci:2://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:699:0-725:1) referencing [shaders/ssgi.wgsl](cci:7://file:///Users/mpopovic3/forge3d/shaders/ssgi.wgsl:0:0-0:0).
  - Half-res toggling API and texture reallocation are present ([SsgiRenderer::set_half_res](cci:1://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:913:4-959:5), [ScreenSpaceEffectsManager::set_ssgi_half_res_with_queue](cci:1://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:690:4-695:5)).
  - Viewer runtime flags are available (`:ssgi-steps`, `:ssgi-radius`, `:ssgi-half on|off`, `:ssgi-temporal-alpha`), CLI pre-seeding is supported in [examples/interactive_viewer.rs](cci:7://file:///Users/mpopovic3/forge3d/examples/interactive_viewer.rs:0:0-0:0).

- SSR
  - Compute + temporal pipelines exist in [src/core/screen_space_effects.rs::SsrRenderer](cci:2://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:1015:0-1038:1) referencing [shaders/ssr.wgsl](cci:7://file:///Users/mpopovic3/forge3d/shaders/ssr.wgsl:0:0-0:0).
  - Thickness-based testing is implemented in WGSL. Temporal filter pass exists.
  - Viewer runtime flags (`:ssr-max-steps`, `:ssr-thickness`) and CLI pre-seeding exist.

- Composite and debug viz
  - The viewer can display material, normal, depth (properly remapped), or a GI debug output (`VizMode::{Material, Normal, Depth, Gi}`) from [ScreenSpaceEffectsManager::gi_debug_view()](cci:1://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:628:4-644:5) in [src/viewer/mod.rs](cci:7://file:///Users/mpopovic3/forge3d/src/viewer/mod.rs:0:0-0:0).

- CLI and tests
  - Interactive one-terminal flow with `:gi` and parameter commands is implemented in [src/viewer/mod.rs](cci:7://file:///Users/mpopovic3/forge3d/src/viewer/mod.rs:0:0-0:0) and pre-seeded via [examples/interactive_viewer.rs](cci:7://file:///Users/mpopovic3/forge3d/examples/interactive_viewer.rs:0:0-0:0).
  - Python CLI utilities exist ([python/forge3d/screen_space_gi.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/screen_space_gi.py:0:0-0:0)) with tests in [tests/test_p5_screen_space_effects.py](cci:7://file:///Users/mpopovic3/forge3d/tests/test_p5_screen_space_effects.py:0:0-0:0).
  - GPU-side struct layout tests for P5/P6 are present in [tests/test_shader_params_p5_p8.rs](cci:7://file:///Users/mpopovic3/forge3d/tests/test_shader_params_p5_p8.rs:0:0-0:0).

Remaining gaps and deliverables
1) SSGI half-res execution correctness
- Issue: [SsgiRenderer::execute()](cci:1://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:587:4-613:5) dispatches using full-res `self.width/height` but half-res targets are reallocated by [set_half_res()](cci:1://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:913:4-959:5), leading to resolution mismatch.
- Deliverables:
  - Track current target dimensions in [SsgiRenderer](cci:2://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:699:0-725:1) (e.g., `out_width`, `out_height`), update them in [new()](cci:1://file:///Users/mpopovic3/forge3d/src/core/gbuffer.rs:60:4-130:5) and [set_half_res()](cci:1://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:913:4-959:5), and use them for dispatch.
  - If half-res mode is ON, ensure upsample path or present the half-res target explicitly in GI viz. Optionally add a simple upsample compute for presentation.

2) Temporal accumulation finalization (SSGI and SSR)
- Issue: After temporal pass, the “filtered” output is not copied back to the “history” texture for the next frame.
- Deliverables:
  - In both [SsgiRenderer::execute()](cci:1://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:587:4-613:5) and [SsrRenderer::execute()](cci:1://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:587:4-613:5), add a copy from the filtered target to the history target at the end of the frame.
  - Ensure appropriate usages and barriers (history textures already have `COPY_DST`, filtered targets are readable as `TEXTURE_BINDING`).

3) Camera space correctness for world directions
- Issue: Shaders use `camera.view_matrix` to transform directions back to world in [shaders/ssgi.wgsl](cci:7://file:///Users/mpopovic3/forge3d/shaders/ssgi.wgsl:0:0-0:0) and [shaders/ssr.wgsl](cci:7://file:///Users/mpopovic3/forge3d/shaders/ssr.wgsl:0:0-0:0). This is incorrect for direction vectors; the inverse view is required.
- Deliverables:
  - Extend [CameraParams](cci:2://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:102:0-108:1) in [src/core/screen_space_effects.rs](cci:7://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:0:0-0:0) to include `inv_view_matrix`.
  - Update [src/viewer/mod.rs](cci:7://file:///Users/mpopovic3/forge3d/src/viewer/mod.rs:0:0-0:0) camera upload to compute and write `inv_view_matrix`.
  - Update [shaders/ssgi.wgsl](cci:7://file:///Users/mpopovic3/forge3d/shaders/ssgi.wgsl:0:0-0:0) and [shaders/ssr.wgsl](cci:7://file:///Users/mpopovic3/forge3d/shaders/ssr.wgsl:0:0-0:0) to use `inv_view_matrix` for direction-to-world transforms.

4) IBL integration for SSGI/SSR
- Issue: SSGI/SSR currently use a placeholder 1x1x6 cube for env fallback.
- Deliverables:
  - Implement `Viewer::load_ibl(path: &str)` in [src/viewer/mod.rs](cci:7://file:///Users/mpopovic3/forge3d/src/viewer/mod.rs:0:0-0:0) using `crate::core::ibl::IBLRenderer` (already imported).
  - Store env cubemap view and sampler in viewer (`ibl_env_view`, `ibl_sampler`) and call [ScreenSpaceEffectsManager::set_env_for_all()](cci:1://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:670:4-677:5) to propagate to SSGI/SSR.
  - The runtime already parses `:ibl <hdr.exr|hdr>` and CLI `--ibl <path>` is pre-seeded in [examples/interactive_viewer.rs](cci:7://file:///Users/mpopovic3/forge3d/examples/interactive_viewer.rs:0:0-0:0).

5) SSAO composite into color (optional but recommended)
- Issue: SSAO compute/blur is implemented, but the composite pipeline (`cs_ssao_composite`) is not used in viewer; Material viz shows raw albedo, and GI viz shows a debug texture.
- Deliverables:
  - Add an SSAO composite step in viewer rendering when SSAO is enabled: multiply `material_view` by blurred AO to produce a composited color texture and display it in material viz mode.
  - Alternatively, keep current GI viz for debugging and add a toggle to switch material-with-AO composite.

6) Viewer command parser completeness
- Issue: `ViewerCmd::SetVizDepthMax(f32)` exists and is bound in composite shader, but there’s no `:viz-depth-max` handler in the input thread.
- Deliverable:
  - Add parsing for `:viz-depth-max <float>` to send `ViewerCmd::SetVizDepthMax`.

7) Acceptance assets and golden images
- Deliverables:
  - Add minimal assets and snapshots to validate acceptance:
    - AO creases: a mesh with tight corners or a Cornell-like scene.
    - SSGI bounce: colored walls and diffuse surfaces with an environment map.
    - SSR: a glossy plane reflecting sky/bright objects.
  - Extend [scripts/generate_golden_images.py](cci:7://file:///Users/mpopovic3/forge3d/scripts/generate_golden_images.py:0:0-0:0) and [tests/golden_images.rs](cci:7://file:///Users/mpopovic3/forge3d/tests/golden_images.rs:0:0-0:0) to produce and validate golden images.

8) Documentation updates
- Deliverables:
  - Update [p5.md](cci:7://file:///Users/mpopovic3/forge3d/p5.md:0:0-0:0) with setup & usage:
    - How to launch the viewer and toggle effects (`:gi`, `:viz`, parameter commands).
    - Parameter cheatsheet for [ssao](cci:1://file:///Users/mpopovic3/forge3d/src/lighting/py_bindings.rs:533:4-545:5), `ssgi`, `ssr` (radius/steps/intensity/thickness/temporal).
    - Debug viz modes and `:viz-depth-max`.
  - Cross-link in [docs/rendering_options.rst](cci:7://file:///Users/mpopovic3/forge3d/docs/rendering_options.rst:0:0-0:0) to point at GI options and the viewer commands.

9) Optional quality/perf presets
- Deliverables:
  - Provide ready-made presets (e.g., “performance” with SSGI half-res+fewer steps; “quality” with more steps and temporal) and document them.
  - Map via CLI flags in [examples/interactive_viewer.rs](cci:7://file:///Users/mpopovic3/forge3d/examples/interactive_viewer.rs:0:0-0:0) or a small preset command helper.

Concrete file-by-file changes
- src/core/screen_space_effects.rs
  - Add `inv_view_matrix` to [CameraParams](cci:2://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:102:0-108:1) and propagate in [update_camera](cci:1://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:419:4-422:5).
  - SSGI: track current output width/height; fix dispatch in [execute()](cci:1://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:587:4-613:5).
  - SSGI/SSR: after temporal pass, copy filtered → history each frame.
  - SSAO: optionally expose a `composite()` method to multiply AO into color if you choose to integrate at this level.

- shaders/ssgi.wgsl and shaders/ssr.wgsl
  - Replace uses of `view_matrix` for direction transforms with `inv_view_matrix`.

- src/viewer/mod.rs
  - Implement `load_ibl()` to initialize `IBLRenderer`, keep env view/sampler, and call [set_env_for_all()](cci:1://file:///Users/mpopovic3/forge3d/src/core/screen_space_effects.rs:670:4-677:5).
  - Add parser for `:viz-depth-max`.
  - Optional: after SSAO, run AO composite and display that in `VizMode::Material`.

- examples/interactive_viewer.rs
  - Already seeds CLI flags for GI and parameters. If you add AO composite or presets, document and wire corresponding flags.

- tests and assets
  - Add small acceptance assets and golden image comparisons.
  - Optionally add a couple of unit/integration tests verifying that viewer parameter commands update GI settings (e.g., exercised via a headless harness).

Deliverables checklist (to close P5)
- Backend/WGSL
  - [ ] SSGI half-res dispatch fixed + optional upsample for display.
  - [ ] SSGI/SSR temporal: copy filtered → history per frame.
  - [ ] Camera `inv_view_matrix` added; SSGI/SSR WGSL fixed.
  - [ ] IBL integration: `:ibl` loads HDR and is bound to SSGI/SSR.
  - [ ] Optional: SSAO composite into material color.

- CLI/Viewer
  - [ ] `:viz-depth-max` runtime command parser added.
  - [ ] All listed GI tuning flags confirmed working end-to-end.

- Acceptance and docs
  - [ ] Assets + snapshot tests for AO creases, SSGI bounce, SSR reflections.
  - [ ] p5.md updated with usage and troubleshooting tips.
  - [ ] Cross-links in docs to GI options; optional presets documented.

I also created a P5 TODO board to track these items. If you want, I can start by implementing the small, high-impact fixes first (SSGI dispatch + temporal history copy + inv_view_matrix + load_ibl), then move to AO composite and golden images.

python3 scripts/generate_golden_images.py --overwrite --obj assets/cornell_box.obj --ibl assets/snow_field_4k.hdr

python3 scripts/generate_golden_images.py --overwrite --obj assets/cornell_box.obj --ibl assets/snow_field_4k.hdr --filter blinnphong_msm_nogi

cargo run --release --example interactive_viewer -- --obj assets/cornell_box.obj --viz normal --snapshot tests/golden/manual.png