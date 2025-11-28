I’ll first load and skim `AGENTS.md` to align with your repo norms, then inspect `examples/terrain_demo.py` and trace what it drives on the Rust/shader side. After that I’ll synthesize a very concrete, high‑level spec aimed at a strong “ChatGPT 5.1” agent: explicit goals, constraints, milestones, and deliverables focused on making the PBR + IBL + triplanar + POM + water stack *visibly* work, not just technically exist.

I’ll set up a small TODO plan, then start by reading those files and finding the terrain backend.
I’ve read your orientation guide and the thin CLI wrapper; next I’ll inspect the actual `terrain_pbr_pom` implementation and its Rust/shader backend so the spec I write is grounded in the real pipeline, not generic PBR talk. Then I’ll produce a concrete spec aimed at a strong “5.1” agent: explicit phases (diagnostics, shader/plumbing fixes, water material, proof images), each with success criteria and test hooks.

Let me quickly open the backend files and update the task list, then I’ll draft the requirements.

I now understand how `terrain_demo` is wired: Python CLI → `forge3d.terrain_pbr_pom` shim → legacy `terrain_demo` helpers → Rust [TerrainRenderer](cci:2://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:59:0-61:1) + `MaterialSet` + `IBL` + [terrain_pbr_pom.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/terrain_pbr_pom.wgsl:0:0-0:0). The shader already has PBR, IBL, triplanar, POM, and optional shadows; the issue is that the current presets, camera, and debug outputs don’t *demonstrate* those features (and the water path is basically just a blue overlay).

Next I’ll (1) quickly inspect the Python terrain helpers to see how materials, IBL, and water detection are configured; then (2) define a very explicit multi-milestone spec for a high‑reasoning agent, including concrete renders (material balls, low‑angle close‑ups, water surface) and acceptance criteria for each feature: visible IBL specular/diffuse, material separation, triplanar stress tests, POM at grazing, and a proper water shader.


# Spec for “ChatGPT 5.1” – Terrain PBR/IBL/Triplanar/POM/Water

This spec assumes the agent runs inside the `forge3d` repo and follows the rules in [AGENTS.md](cci:7://file:///Users/mpopovic3/forge3d/AGENTS.md:0:0-0:0) (Rust‑first, tests/docs define behavior, cleanliness, small changes, no comment churn).

Goal:  
Make [examples/terrain_demo.py](cci:7://file:///Users/mpopovic3/forge3d/examples/terrain_demo.py:0:0-0:0) *visibly* demonstrate the full PBR stack:

- Physically‑sane **PBR + IBL** lighting.
- Clear **material separation** (rock/soil/grass/snow).
- Demonstrable **triplanar** mapping and **POM**.
- A convincing **water material** (Fresnel, reflection, roughness, shoreline depth).
- Reasonable **shadows/contact** (even if CSM integration remains a follow‑up).

All work must stay within the existing pipeline:

- Python: [python/forge3d/terrain_demo.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_demo.py:0:0-0:0), [terrain_params.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_params.py:0:0-0:0)
- Rust: [src/terrain_renderer.rs](cci:7://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:0:0-0:0), [src/terrain_render_params.rs](cci:7://file:///Users/mpopovic3/forge3d/src/terrain_render_params.rs:0:0-0:0), `src/material_set.rs`, `src/ibl_wrapper.rs`
- Shaders: [src/shaders/terrain_pbr_pom.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/terrain_pbr_pom.wgsl:0:0-0:0), `lighting.wgsl`, `lighting_ibl.wgsl`

Do **not** break existing tests or CLIs; extend surgically.

---

## Context (for 5.1)

- **Entry:** [examples/terrain_demo.py](cci:7://file:///Users/mpopovic3/forge3d/examples/terrain_demo.py:0:0-0:0) → `forge3d.terrain_pbr_pom` → [python/forge3d/terrain_demo.py::run](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_demo.py:606:0-918:12).
- **Config path:** [_build_params()](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_demo.py:264:0-339:42) → [make_terrain_params_config()](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_params.py:343:0-488:5) → [python/forge3d/terrain_params.py::TerrainRenderParams](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_params.py:222:0-317:39).
- **Native bridge:** [src/terrain_render_params.rs::TerrainRenderParams](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_params.py:222:0-317:39) (+ [decoded()](cci:1://file:///Users/mpopovic3/forge3d/src/terrain_render_params.rs:707:4-709:5)).
- **Renderer:** [src/terrain_renderer.rs::TerrainRenderer::render_terrain_pbr_pom](cci:1://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:455:4-490:5) → [TerrainScene::render_internal](cci:1://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:1221:4-1741:5).
- **Shader:** `src/shaders/terrain_pbr_pom.wgsl::vs_main/fs_main` with:
  - Triplanar, layer mixing (`layer_*` uniforms).
  - POM (`parallax_occlusion_mapping`, `pom_steps`, `triplanar_params.w`).
  - IBL via `lighting_ibl.wgsl::eval_ibl` and `u_ibl`.
  - Debug modes and colormap/overlay logic.
- **Materials:** `src/material_set.rs::MaterialSet::terrain_default` (rock/grass/soil/snow).
- **Water today:** purely CPU‑side blue tint in [python/forge3d/terrain_demo.py::_apply_water_tint](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_demo.py:446:0-482:14), no PBR/IBL.

---

## Milestone M0 – Non‑regression + Baseline Renders

**Objective:** Get a baseline of current behavior and ensure nothing regresses.

**Tasks**

- **T0.1** Run existing tests related to terrain and IBL:
  - `pytest -k "terrain_"` and `pytest -k "TerrainRenderParams"`.
  - `cargo test -p forge3d -- terrain_render_params` (or equivalent existing test names).
- **T0.2** Generate and save current demo outputs (no code change):
  - Top‑down default: `python examples/terrain_demo.py --output examples/out/terrain_baseline_topdown.png`.
  - Same view with `--albedo-mode material --colormap-strength 0.0` → `terrain_baseline_pbr_only.png`.
- **Deliverables**
  - Short note in code comment or docstring *only where appropriate* summarizing that `examples/out/*.png` are informal baselines (no test dependency).
- **Acceptance**
  - All tests still green.
  - `terrain_baseline_*` images committed or reproducible via documented commands.

---

## Milestone M1 – Make PBR + IBL Visibly Readable

**Objective:** Ensure the terrain clearly shows IBL and PBR behavior (diffuse+specular, view dependence, environment tint).

**Code boundaries**

- [python/forge3d/terrain_demo.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_demo.py:0:0-0:0) (parameter choices).
- [python/forge3d/terrain_params.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_params.py:0:0-0:0) ([make_terrain_params_config](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_params.py:343:0-488:5)).
- [src/terrain_render_params.rs](cci:7://file:///Users/mpopovic3/forge3d/src/terrain_render_params.rs:0:0-0:0) (no semantic change, but may need small plumbing if something is not wired).
- [src/shaders/terrain_pbr_pom.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/terrain_pbr_pom.wgsl:0:0-0:0) (only if strictly necessary).

**Tasks**

- **T1.1 – Audit parameter mapping**
  - Verify [TerrainRenderParams.decoded()](cci:1://file:///Users/mpopovic3/forge3d/src/terrain_render_params.rs:707:4-709:5) fields used in [TerrainScene::build_shading_uniforms()](cci:1://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:2005:4-2129:5) (you must inspect that function) correctly map:
    - `triplanar.scale/blend_sharpness/normal_strength` → `u_shading.triplanar_params.xyz`.
    - `pom.scale/min_steps/max_steps/refine_steps/...` → `u_shading.triplanar_params.w` and `pom_steps`.
    - `clamp.*` → `clamp0/1/2`.
    - `ibl.intensity` (Python) → `u_ibl.intensity` (already set in [render_internal](cci:1://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:1221:4-1741:5); verify no hard‑coded overrides).
- **T1.2 – Improve defaults for PBR clarity (no breaking change)**
  - Adjust **only** defaults in [make_terrain_params_config](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_params.py:343:0-488:5) and `MaterialSet::terrain_default` when they are clearly too flat:
    - Ensure at least one material is noticeably glossy (e.g. wet rock: roughness ~0.1–0.2).
    - Keep existing ranges but allow more contrast between rock/grass/snow roughness and maybe metallic.
  - Ensure `ibl.enabled` is honored; no forced disabling in Rust.
- **T1.3 – “Material/IBL A/B” renders**
  - Add a *documented* pair of CLI invocations (no new CLI flags required) that produce:
    - **PBR+IBL on:** default HDR, `gi` includes [ibl](cci:1://file:///Users/mpopovic3/forge3d/src/terrain_render_params.rs:654:4-657:5), `--albedo-mode material`, `--colormap-strength 0`, mid camera, resolution ≥ 1024².
    - **PBR+IBL off:** same but with `--gi ""` or equivalent so only direct light remains.
  - Update `docs/` (e.g. short subsection in an existing terrain or demo page) with those commands and embed or reference sample PNGs.
- **Deliverables**
  - Verified parameter mapping notes in code comments only where missing/necessary.
  - Two reproducible commands in docs plus their expected visual differences (IBL adds view‑dependent specular and sky tint).
- **Acceptance**
  - In the “IBL on” image you can clearly see:
    - View‑dependent highlights on steep rock.
    - Slight color shift matching HDR sky.
  - “IBL off” looks flatter and lacks those cues, under same exposure.

---

## Milestone M2 – Triplanar & POM Demonstration (Low‑Angle Close‑ups)

**Objective:** Produce renders where triplanar mapping eliminates stretching and POM/self‑occlusion are obviously at work.

**Code boundaries**

- [python/forge3d/terrain_demo.py::_build_params](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_demo.py:264:0-339:42) (camera & POM defaults for debug views).
- [python/forge3d/terrain_params.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_params.py:0:0-0:0) ([PomSettings](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_params.py:117:0-148:57), [TriplanarSettings](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_params.py:98:0-114:60) defaults).
- [src/terrain_render_params.rs](cci:7://file:///Users/mpopovic3/forge3d/src/terrain_render_params.rs:0:0-0:0), [src/terrain_renderer.rs](cci:7://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:0:0-0:0) (only if POM/triplanar uniforms are miswired).
- [src/shaders/terrain_pbr_pom.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/terrain_pbr_pom.wgsl:0:0-0:0) (POM/triplanar logic only if a bug is found).

**Tasks**

- **T2.1 – Confirm wiring**
  - Trace [PomSettings](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_params.py:117:0-148:57) → [TerrainRenderParams](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_params.py:222:0-317:39) (Python) → [TerrainRenderParams](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_params.py:222:0-317:39) (Rust) → [decoded().pom](cci:1://file:///Users/mpopovic3/forge3d/src/terrain_render_params.rs:707:4-709:5) → [build_shading_uniforms](cci:1://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:2005:4-2129:5) → `u_shading.triplanar_params.w` and `pom_steps`.
  - Fix any mismatches (e.g. wrong unit, wrong flag bits) *without changing public Python API*.
- **T2.2 – Add a “grazing debug view” preset (no new file)**
  - Extend [_build_params](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_demo.py:264:0-339:42) to interpret a debug mode encoded via existing arguments (e.g. a special `--preset terrain_pbr_debug` or a `--cam-theta` ≤ 15° plus small `--cam-radius`).
  - This preset should:
    - Place camera near the surface (grazing angle).
    - Use `albedo_mode="material"`, `colormap_strength=0.0` to focus on material/relief.
    - Increase `pom.scale` (e.g. 2–4× default) and steps to strongly exaggerate parallax for this debug case only.
- **T2.3 – Paired POM renders**
  - Document two commands and outputs (same camera and material set):
    - `... --preset terrain_pbr_debug --height-curve-strength 0.0` **with** POM enabled (default).
    - Same but with `PomSettings.enabled=False` (this can be toggled in [make_terrain_params_config](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_params.py:343:0-488:5) based on CLI or a new `--pom` boolean).
- **Deliverables**
  - Debug preset implemented entirely via existing config plumbing ([TerrainRenderParams](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_params.py:222:0-317:39)).
  - Two images (POM on/off) showing:
    - Apparent relief changing with view.
    - Self‑occlusion along ridges when POM is on.
- **Acceptance**
  - Visual inspection: at grazing angle, micro‑relief clearly shifts with view when POM is on, and becomes flat when off.
  - No regressions in default top‑down render quality.

---

## Milestone M3 – PBR Water Material Integrated with IBL

**Objective:** Replace the “flat blue fill” lake with an actual water material driven by IBL and Fresnel.

**Constraints**

- Respect existing water detection logic in [python/forge3d/terrain_demo.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_demo.py:0:0-0:0) (`_detect_dem_water_mask`, [_resize_mask_to_frame](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_demo.py:363:0-383:54), [_remove_border_connected](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_demo.py:386:0-443:14)).
- Keep memory allocation bounded; no large extra GPU textures beyond one binary/float mask and maybe a small normal map.
- Minimize API footprint changes; any new public flags must be clearly documented.

**Tasks**

- **T3.1 – Introduce a GPU water mask (optional input)**
  - Extend [TerrainRenderer::render_terrain_pbr_pom](cci:1://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:455:4-490:5) / [TerrainScene::render_internal](cci:1://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:1221:4-1741:5) to optionally accept a water mask texture:
    - Add an *optional* parameter in Python [TerrainRenderParams](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_params.py:222:0-317:39) or a side channel (e.g. a separate method on [TerrainRenderer](cci:2://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:59:0-61:1)) that uploads a 2D R8Unorm or R32Float mask.
    - In Rust, add **one** new binding slot to `terrain_pbr_pom` group(0) (e.g. `@binding(11)` texture + sampler) only if this does not break existing pipelines/tests; otherwise, derive water mask from height alone (fallback).
  - From Python, reuse the already computed `combined_mask` and upload it once per render.
- **T3.2 – Water shading branch in WGSL**
  - In `terrain_pbr_pom.wgsl::fs_main`, add a **local** branch:
    - Derive `is_water` from water mask sample (or from height mask if GPU mask optional).
    - For water pixels:
      - Override material parameters:
        - `albedo`: deep/wet color (e.g. dark teal).
        - `roughness`: lower and spatially modulated by a simple normal map or noise (wind).
        - `metallic`: 0, but `f0` set from IOR 1.33 (≈0.02).
      - Compute reflection using existing `eval_ibl` with strong Fresnel (view‑dependent).
      - Optionally darken shallow areas via simple depth term (height relative to `water_level` range).
  - Ensure non‑water pixels remain **bit‑identical** (no change to existing math paths).
- **T3.3 – CLI and docs**
  - Add controlled flags, e.g.:
    - `--water-material {none,overlay,pbr}` choosing between:
      - `none`: keep original terrain only.
      - `overlay`: current [_apply_water_tint](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_demo.py:446:0-482:14) 2D overlay (back‑compatible).
      - `pbr`: new GPU water material, overlay disabled.
  - Document behavior and example command lines in a short docs subsection.

- **Deliverables**
  - New optional water mask bindings and shader path, gated behind `water-material=pbr`.
  - At least one comparison triplet:
    - No water highlighting.
    - 2D overlay (current behavior).
    - PBR water (new), showing:
      - Fresnel reflection.
      - Environment reflection consistent with sky.
      - Slight shoreline color change.
- **Acceptance**
  - Non‑water areas match original images within a small numeric epsilon.
  - Water areas exhibit:
    - Stronger reflection at grazing angles than at nadir.
    - Color and brightness changing if HDR map is swapped.

---

## Milestone M4 – Shadows / Contact Grounding (Terrain‑Local)

**Objective:** Improve sense of depth via shadows or contact–like occlusion, without fully integrating the global CSM system if that’s too large.

**Tasks**

- **T4.1 – Evaluate current shadow path**
  - Inspect:
    - `TERRAIN_USE_SHADOWS` constant (currently `false`).
    - [TerrainScene::create_shadow_bind_group_layout](cci:1://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:1051:4-1111:5) and [create_noop_shadow](cci:1://file:///Users/mpopovic3/forge3d/src/terrain_renderer.rs:868:4-1049:5).
  - Confirm that no real CSM data is currently bound; only noop textures are used.
- **T4.2 – Minimal terrain‑only shadowing**
  - Short‑term: enhance occlusion term in `fs_main`:
    - Use height + slope + POM occlusion to build a more grounded “ambient shadow” factor (respecting `u_shading.clamp1` ranges).
    - Avoid full CSM integration in this milestone; treat it as improved AO.
- **T4.3 – Optional stretch: CSM integration**
  - If feasible without large refactor:
    - Replace `noop_shadow` with bindings from the existing shadow mapping system used by mesh PBR.
    - Set `TERRAIN_USE_SHADOWS = true` only after verifying bind group compatibility and tests.
- **Deliverables**
  - Tuned occlusion term with clear, documented mapping from [ClampSettings](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_params.py:200:0-219:62) to shader behavior.
  - Before/after images showing stronger grounding of ridges/valleys without over‑darkening.
- **Acceptance**
  - Visual: ridges and crater walls show more convincing shadowing; flat regions remain reasonably lit.
  - No new haloing/light leaks at DEM edges.

---

## Milestone M5 – Explicit “Proof” Renders & Minimal Tests

**Objective:** Bake in “no excuses” diagnostic outputs and minimal tests so future changes can’t silently regress the stack.

**Tasks**

- **T5.1 – Material ball / grid harness**
  - Reuse the existing PBR + IBL shader stack used by meshes (from `lighting.wgsl`, `lighting_ibl.wgsl`) and add a small example (either new `examples/terrain_ibl_probe.py` or an option in an existing viewer) that renders:
    - A grid of spheres/boxes with varying roughness and metallic.
    - Under the *same IBL path* as terrain (same `IBL` object and eval_ibl).
  - Purpose: verify env orientation, BRDF, and specular/diffuse balance.
- **T5.2 – Scripted “debug shots” for CI/manual checks**
  - Add a tiny Python helper (or documented commands) that renders:
    - PBR+IBL terrain mid‑view.
    - Grazing POM/triplanar close‑up.
    - PBR water view centered on a lake.
  - Save them under `examples/out/terrain_debug_*.png` with stable naming.
- **T5.3 – Lightweight tests**
  - Add tests reusing existing infrastructure (e.g. `tests/test_terrain_render_params_native.py`) to assert:
    - [TerrainRenderParams](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/terrain_params.py:222:0-317:39) → [decoded()](cci:1://file:///Users/mpopovic3/forge3d/src/terrain_render_params.rs:707:4-709:5) → `u_shading` packing is consistent (IBL intensity, POM steps, triplanar scale).
    - Water PBR mode can be toggled without panic and yields non‑NaN color values on a small synthetic heightmap + mask.
  - Do **not** add heavy golden‑image tests unless they match existing project patterns.

- **Deliverables**
  - New example or mode for material grid.
  - Documented debug commands and sample images.
  - 1–3 unit/medium tests guarding config→uniform wiring.

- **Acceptance**
  - Running the debug script produces all three diagnostic PNGs without manual tweaking.
  - Tests pass and fail meaningfully if uniforms or feature flags are miswired.