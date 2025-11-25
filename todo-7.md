## P6 Implementation Plan for ChatGPT 5.1 (High-Reasoning)
---

## Phase 0 – Pre‑flight Orientation (maps to Milestone 0)

- **[0.1] Re-read specs**
  - Read in full:
    - [AGENTS.md](cci:7://file:///Users/mpopovic3/forge3d/AGENTS.md:0:0-0:0)
    - [p6.md](cci:7://file:///Users/mpopovic3/forge3d/p6.md:0:0-0:0)
    - [todo-6.md](cci:7://file:///Users/mpopovic3/forge3d/todo-6.md:0:0-0:0)
    - [docs/api/atmospherics_p6.md](cci:7://file:///Users/mpopovic3/forge3d/docs/api/atmospherics_p6.md:0:0-0:0)
- **[0.2] Verify current wiring matches design skeleton**
  - Open:
    - WGSL: `src/shaders/sky.wgsl`, `src/shaders/volumetric.wgsl`, `src/shaders/fog_upsample.wgsl`.
    - Rust: `src/lighting/mod.rs`, `src/lighting/types.rs`, `src/lighting/shadow_map.rs`, `src/render/params.rs`, `src/core/mod.rs`, `src/viewer/mod.rs`.
    - Python: `python/forge3d/config.py`, `python/forge3d/lighting.py`.
  - Confirm that:
    - `cs_render_sky`, `cs_volumetric`, and fog upsample kernels exist and are wired in `Viewer::render` as described.
    - `SkyModel`, `VolumetricPhase`, `VolumetricMode`, `VolumetricParams`, `AtmosphereParams` are present and shaped as in the design note.
- **[0.3] Snapshot current tests**
  - Check for presence and status of:
    - `tests/test_media_fog.py`, `tests/test_media_hg.py`
    - `tests/test_p6_sky.rs`, `tests/test_p6_fog.rs`
  - Run:
    - `cargo test --workspace --all-features -- tests::p6`
    - `pytest tests/test_media_fog.py tests/test_media_hg.py -v`
  - Note which P6 tests already exist and pass.

*(If any of these files/tests diverge from the design skeleton, update [docs/api/atmospherics_p6.md](cci:7://file:///Users/mpopovic3/forge3d/docs/api/atmospherics_p6.md:0:0-0:0) first to match **reality**, then continue.)*

---

## Phase 1 – Sky Models & Integration (Milestone 1)

**Goal:** Confirm/complete physical sky implementation and config wiring.

- **[1.1] Confirm WGSL sky implementation**
  - In `src/shaders/sky.wgsl`:
    - Ensure functions equivalent to `sky_preetham(...)` and `sky_hosek_wilkie(...)` exist and implement the analytic models correctly.
    - Ensure the compute entry (`cs_render_sky`) uses:
      - Sun direction (world space) from uniforms.
      - `turbidity`, `ground_albedo`, intensity/exposure as described.
- **[1.2] Validate Rust integration**
  - In `src/viewer/mod.rs`:
    - Confirm `Viewer::render` step:
      - Uploads `SkyUniforms` and camera matrices.
      - Computes sun direction in world space from the “sun” light.
      - Dispatches `cs_render_sky` to `sky_output`.
  - In `src/lighting/types.rs` and `src/render/params.rs`:
    - Ensure `SkyModel` enum and `SkySettings` / `AtmosphereParams` are consistent with the design skeleton.
- **[1.3] Ensure config and CLI mapping**
  - In `python/forge3d/config.py`:
    - Confirm `_SKY_MODELS` supports `"hosek-wilkie"`, `"preetham"`, `"hdri"`.
    - Ensure `RendererConfig.atmosphere.sky` propagates cleanly to Rust `RendererConfig`.
  - If a CLI or example harness parses `--sky ...`:
    - Make sure it maps to `RendererConfig.atmosphere.sky` as in [todo-6.md](cci:7://file:///Users/mpopovic3/forge3d/todo-6.md:0:0-0:0).
- **[1.4] Implement/extend tests**
  - **Rust**: extend `tests/test_p6_sky.rs`:
    - Sweep sun elevation and assert monotonic chromaticity change (e.g., horizon `R/B` ratio) noon → sunset.
  - **Python**: add `tests/test_p6_sky_config.py`:
    - Round-trip `RendererConfig.atmosphere.sky` across Python↔Rust for Hosek–Wilkie / Preetham / HDRI.
- **[1.5] Run tests**
  - `cargo test --workspace --all-features -- test_p6_sky`
  - `pytest tests/test_p6_sky_config.py -v`

---

## Phase 2 – Volumetric Height Fog (Milestone 2)

**Goal:** Fully wire volumetric fog via `AtmosphereParams`/`VolumetricParams` and tests.

- **[2.1] Confirm WGSL volumetric kernel**
  - In `src/shaders/volumetric.wgsl`:
    - Ensure `cs_volumetric` (and froxel path if present) use:
      - Exponential height fog.
      - HG phase with `g` anisotropy.
      - Depth, sun direction, and (optionally) shadow map.
- **[2.2] Confirm Rust data structures**
  - In `src/lighting/types.rs`:
    - Verify `VolumetricSettings` matches fields in the design note (density, anisotropy, height falloff, max_steps, ranges, colors, temporal, shadows, jitter).
  - In `src/render/params.rs`:
    - Ensure `VolumetricParams` and `AtmosphereParams` match the doc.
- **[2.3] Python config mapping**
  - In `python/forge3d/config.py`:
    - Ensure `VolumetricParams` / `AtmosphereParams` dataclasses expose `density`, `phase`, `anisotropy` (`g`), `mode`, `max_steps`, etc.
    - Implement the override mapping shown in [docs/api/atmospherics_p6.md](cci:7://file:///Users/mpopovic3/forge3d/docs/api/atmospherics_p6.md:0:0-0:0) for:
      ```python
      overrides = {
          "atmosphere": {
              "volumetric": {
                  "density": 0.015,
                  "phase": "hg",
                  "g": 0.7,
                  "mode": "raymarch",
                  "max_steps": 48,
              },
          },
      }
      ```
- **[2.4] Viewer integration sanity**
  - In `src/viewer/mod.rs`:
    - Confirm fog is an optional pass after geometry/depth and before final composite.
    - Ensure fog uses `FogCameraUniforms` and `VolumetricUniformsStd140` consistent with config.
- **[2.5] Implement tests**
  - `tests/test_p6_fog_config.py`:
    - Assert mapping `RendererConfig.atmosphere.volumetric` → Rust `VolumetricSettings` → WGSL params (bounds/clamping/defaults).
  - New Python test using synthetic depth:
    - Render with volumetric off vs on.
    - Check depth–alpha correlation (far pixels more foggy).
- **[2.6] Run tests**
  - `cargo test --workspace --all-features -- test_p6_fog`
  - `pytest tests/test_p6_fog_config.py -v`

---

## Phase 3 – God-Rays & Temporal Reprojection (Milestone 3)

**Goal:** Ensure volumetric shadows and temporal reprojection meet spec and are well-tested.

- **[3.1] Verify shadow integration**
  - In `src/lighting/shadow_map.rs` and volumetric shader bindings:
    - Confirm `shadow_map`, `shadow_sampler`, `shadow_matrix` are bound at the groups/bindings described in the design note.
    - Ensure volumetric shader uses them to compute occluded vs unoccluded scattering.
- **[3.2] Temporal jitter & history**
  - In `src/viewer/mod.rs` + `volumetric.wgsl`:
    - Confirm per-frame jitter is applied to ray steps.
    - Confirm history buffer(s) are allocated, bound, and updated each frame.
- **[3.3] Implement tests**
  - Add `tests/test_p6_godrays.rs`:
    - Construct synthetic depth and shadow maps with a ridge occluding sun.
    - Assert higher scattering in the occluded band vs control.
  - Add `tests/test_p6_temporal.rs`:
    - Render multi-frame sequence with jitter.
    - Compare temporal variance with and without reprojection; reprojection must reduce variance while preserving mean.
- **[3.4] Run tests**
  - `cargo test --workspace --all-features -- test_p6_godrays test_p6_temporal`

---

## Phase 4 – End-to-End QA & Acceptance (Milestone 4)

**Goal:** Prove P6 meets [p6.md](cci:7://file:///Users/mpopovic3/forge3d/p6.md:0:0-0:0) acceptance criteria across the full viewer path.

- **[4.1] Implement `tests/test_p6_viewer_sky_fog.py`**
  - Launch headless viewer with a known terrain and controlled `RendererConfig`.
  - Sweep sun elevation (sunrise → noon) with volumetric fog and god-rays enabled.
  - Capture frames or AOVs.
  - Assert:
    - Sunrise/sunset color shift (sky pixel stats).
    - Beams visible when sun occluded by terrain (beam band intensity).
    - Temporal stability (frame-to-frame deltas under threshold).
- **[4.2] Performance & memory check**
  - Use appropriate tools/metrics (as per `docs/memory_budget.rst` and existing perf tests) to:
    - Measure frame time impact.
    - Ensure P6 textures respect the 512 MiB host-visible budget; prefer GPU-only allocations.
- **[4.3] Final CI run**
  - `cargo test --workspace --all-features -- --test-threads=1`
  - `pytest tests/ -v --tb=short`

---

## Phase 5 – Docs & Examples (Milestone 5)

**Goal:** Ensure discoverability and future maintainability.

- **[5.1] Docs**
  - Update:
    - `docs/index.rst` to link P6 doc.
    - Any lighting/atmosphere user docs to mention:
      - `--sky` usage,
      - `--volumetric` usage,
      - Interactions with GI/postfx.
- **[5.2] Examples**
  - Add/extend:
    - Python example:
      - Loads terrain.
      - Configures sky + volumetrics via `RendererConfig`.
      - Produces a short sunrise-to-noon sequence.
    - Optional Rust viewer example showing interactive sky+fog+god-rays.
  - Add light tests that run these examples at low resolution to keep them in sync.

---


Assess the current level of implementation of Phase 5 from todo-6.md. If the requirements are met, do nothing. If the requirements are not met, think very hard to turn these into an extremely surgically precise and strict prompt for ChatGPT 5.1 (high reasoning) to accomplish the missing requirements

you must read fully @AGENTS.md to get familiar with my codebase. Next, you must carefully read @todo-5.md as a whole. Then you must fully implement P5.8. Test after every change. Do not stop until you meet all the requirements for P5.8

you must read fully @AGENTS.md  to get familiar with my codebase. Next, you must carefully read @p6.md as a whole. Then you must design an extremely surgically precise and accurate and specific set of requirements for ChatGPT 5.1. (high reasoning)  with coherent milestones and clear deliverables