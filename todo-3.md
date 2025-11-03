# P3 — Shadow system (Hard, PCF, PCSS, VSM/EVSM/MSM, CSM)

Goal: Deliver a pluggable shadow system with a shadow map atlas and resource guardrails, supporting Hard/PCF/PCSS and moment-based VSM/EVSM/MSM, including Cascaded Shadow Maps (CSM) with stabilized splits. Integrate visibility into the BRDF lighting loop and expose CLI controls.

Note: Significant groundwork already exists:
- `src/shadows/manager.rs` implements a CSM-backed atlas controller, memory budgeting, technique params, and bind group layout creation.
- `src/shadows/csm.rs` provides uniforms, cascade update, and depth/moment atlas creation.
- `src/shaders/shadows.wgsl` contains CSM and PCF sampling scaffolding with bind group group(2) binding(0–2).
- CLI flags are already present in `examples/terrain_demo.py` (`--shadows`, `--shadow-map-res`, `--cascades`, PCSS knobs).
This plan finalizes technique implementations (PCSS, VSM/EVSM/MSM), stabilizes cascades, wires the manager into pipelines, adds tests/docs, and validates performance and determinism.

## Milestone 1 — Shadow manager, atlas, and budget guardrails

- P3-01: Enforce 256 MiB budget deterministically (High, 0.5 day)
  - Validate and finalize `ShadowManager::enforce_memory_budget()` against target budget (256 MiB), including EVSM/MSM moment costs.
  - Ensure downscaling logs are clear and single-step stable; don’t thrash resolutions frame-to-frame.
  - Exit criteria: Given technique, cascade count, and requested map size, the atlas resolution is clamped so total ≤256 MiB. Unit test covers edge cases.

- P3-02: Bind group layout alignment (High, 0.25 day)
  - Confirm the runtime layout from `ShadowManager::create_bind_group_layout()` matches `src/shaders/shadows.wgsl`:
    - group(2) bindings: (0) `CsmUniforms`, (1) `texture_depth_2d_array` shadow, (2) `sampler_comparison` shadow.
    - Moment path uses additional bindings: (3) `texture_2d_array<f32>`, (4) `sampler` (filtering) for VSM/EVSM/MSM.
  - Exit criteria: Pipelines using shadows compile; non-moment techniques do not require moment bindings; moment techniques bind fallback texture if needed.

- P3-03: Cascade stabilization and texel snapping (High, 1 day)
  - Implement view-projection snapping to world texel size to reduce shimmering when camera moves (`src/shadows/csm.rs`).
  - Keep 2–4 cascades (default 3). Maintain practical split scheme; add stabilization and optional cascade fade at boundaries.
  - Exit criteria: With camera orbit, cascade edges do not shimmer; debug visualization toggles work as expected.

Notes
- Keep Rust warnings and Clippy clean (see project guideline). Prefer small, well-documented methods.

## Milestone 2 — WGSL techniques (hard, PCF, PCSS, VSM/EVSM/MSM) and CSM integration

- P3-04: Finalize hard/PCF/Poisson PCF paths (High, 0.5 day)
  - `src/shaders/shadows.wgsl` already implements basic and PCF sampling with Poisson option. Audit biasing (depth + slope) and texel-size handling per cascade.
  - Exit criteria: Hard/PCF code produces expected hardness/softness, bias clamped to avoid acne/peter-panning.

- P3-05: Implement PCSS (High, 1 day)
  - Add blocker search and penumbra estimation, clamped by per-cascade texel size. Use `technique_params` in `CsmUniforms` (via manager) for `pcss_blocker_radius`, `pcss_filter_radius`, `light_size`.
  - Exit criteria: Visually larger penumbra for distant occluders; softening increases with light size; no catastrophic self-shadowing.

- P3-06: Implement VSM/EVSM/MSM (High, 1.5 days)
  - Add moment-writing pass (compute or render) to populate `evsm_maps` (Rgba32Float) or a 2-channel VSM variant.
  - Implement EVSM warp (pos/neg exponents) and MSM 4th-moment sampling; wire `moment_bias` and leak reduction heuristics.
  - Update WGSL and bind groups to read moments when `ShadowTechnique::{VSM,EVSM,MSM}` active.
  - Exit criteria: Variance methods compile and run; reduced aliasing/softer penumbrae; light-leak limited by bias/warp.

- P3-07: CSM selection, transform, and fade (Medium, 0.5 day)
  - Ensure `select_cascade`, world→light transform, and optional cross-fade at split boundaries are robust.
  - Exit criteria: Split selection stable; optional fade avoids visible transitions.

Notes
- Match `ShadowTechnique` values from `src/lighting/types.rs` to WGSL constants; expose `as_u32()` where needed.

## Milestone 3 — Pipeline integration and BRDF loop

- P3-08: Bind shadows in mesh PBR path (High, 0.5–1 day)
  - Integrate the shadow bind group (group(2)) into `src/pipeline/pbr.rs` or `src/render/pbr_pass.rs`/`src/core/pbr.rs` depending on your architecture.
  - Ensure per-frame uploads: `ShadowManager::update_cascades(...)` and `upload_uniforms(...)` are called with camera/light.
  - Exit criteria: Mesh PBR pass runs with shadow visibility applied.

- P3-09: Apply visibility in BRDF (Medium, 0.5 day)
  - In `src/shaders/pbr.wgsl`, multiply direct lighting by `visibility = calculate_shadow(...)` imported from `shadows.wgsl`.
  - Keep IBL/ambient unaffected. Preserve existing BRDF dispatch (from P2) and normal/AO/emissive paths.
  - Exit criteria: Hard/PCF/PCSS visibly affect direct lighting; IBL contribution unchanged.

- P3-10: Terrain path hook (Medium, 0.5 day)
  - Add optional shadow sampling in `src/shaders/terrain_pbr_pom.wgsl` direct light term, using world position/normal and view depth; gate behind a feature/flag to avoid regressions.
  - Exit criteria: Terrain demo can enable shadows with no POM/shading regressions; defaults remain unchanged.

## Milestone 4 — CLI, config, and Python bridge

- P3-11: Wire CLI flags fully (Medium, 0.25 day)
  - Confirm `examples/terrain_demo.py` flags map to config (already present: `--shadows`, `--shadow-map-res`, `--cascades`, PCSS/EVSM params).
  - Ensure `RendererConfig` → native `ShadowManagerConfig` mapping sets `technique`, `cascade_count`, resolution, PCSS radii, `light_size`, and `moment_bias`.
  - Exit criteria: Running the demo switches techniques and resolutions deterministically; invalid inputs produce clear errors.

- P3-12: Python debug utilities (Low, 0.25 day)
  - Add a small debug function to print cascade info, map size, memory bytes, technique flags (via PyO3 or CPU fallback state in `src/shadows/state.rs`).
  - Exit criteria: `--debug` path prints cascade near/far/texel_size and computed memory budget.

Notes
- Preserve the single-terminal interactive workflow; no changes required to viewer/snapshot behavior in P3.

## Milestone 5 — Tests, perf, and CI

- P3-13: Rust unit tests (High, 0.5 day)
  - Tests for: memory estimator; PCSS radius clamp (`clamp_pcss_radius`); cascade splits monotonic and within near/far; `ShadowTechnique::requires_moments()` logic; fallback moment texture present when moments unused.
  - Exit criteria: `cargo test` passes on CPU-only CI.

- P3-14: Perf validation (Medium, 0.5–1 day)
  - Add a small perf harness under `tests/perf/` or `bench/` to time a single shadowed frame at 1280×920 with a mid-range technique (e.g., PCF) and report frame time. Document target GPU.
  - Exit criteria: Documented perf at or under 16.6 ms on target hardware; CI logs include measured timing (informational when GPU absent).

- P3-15: Visual goldens (Medium, 0.5 day)
  - Record tiny goldens for `hard`, `pcf`, `pcss` with a simple test scene. Moment-based techniques optional due to variability.
  - Exit criteria: Goldens show progressive softening and reduced acne; tolerances set for cross-GPU variability; skipped gracefully without GPU.

## Milestone 6 — Docs and acceptance

- P3-16: User docs (Medium, 0.5–1 day)
  - Add `docs/user/shadows_overview.rst`: techniques overview, atlas budget math, CSM splits/stabilization, CLI flags, troubleshooting (acne vs peter-panning), and performance tips.
  - Exit criteria: `make html` builds; page linked from user docs index.

- P3-17: Acceptance documentation (Low, 0.25 day)
  - Document acceptance steps: visual PCSS softening vs PCF/hard, reduced acne, deterministic atlas allocation, and compile on win_amd64/linux_x86_64/macos_universal2.
  - Exit criteria: Steps reproducible via CLI one-liners; included in docs or `docs/P3_ACCEPTANCE.md`.

---

## Recommended implementation order

1) P3‑01/02 atlas + bindings  
2) P3‑03 cascade stabilization  
3) P3‑04/05/06 WGSL techniques  
4) P3‑08/09/10 pipeline/BRDF integration  
5) P3‑11/12 CLI + debug  
6) P3‑13/14/15 tests + perf + goldens  
7) P3‑16/17 docs + acceptance

Rationale
- Stabilized cascades reduce visual churn early.  
- Technique implementation next enables meaningful perf/visual validation.  
- Wiring into pipelines and CLI completes end-to-end usage; tests/docs finalize.

## Estimates summary (single engineer; add 30–40% for review/CI)

- P3‑01: 0.5 day  
- P3‑02: 0.25 day  
- P3‑03: 1.0 day  
- P3‑04: 0.5 day  
- P3‑05: 1.0 day  
- P3‑06: 1.5 days  
- P3‑07: 0.5 day  
- P3‑08: 0.5–1.0 day  
- P3‑09: 0.5 day  
- P3‑10: 0.5 day  
- P3‑11: 0.25 day  
- P3‑12: 0.25 day  
- P3‑13: 0.5 day  
- P3‑14: 0.5–1.0 day  
- P3‑15: 0.5 day  
- P3‑16: 0.5–1.0 day  
- P3‑17: 0.25 day

Base ship: ~6–9 days

## Risks and mitigations

- Light leaks with variance techniques  
  Mitigation: Proper `moment_bias`, EVSM warp exponents, clamp/min variance; doc trade-offs.

- Cascade shimmering and peter-panning  
  Mitigation: Texel snapping, slope-scaled bias tuning, fade across splits; unit tests for splits.

- Binding/layout drift across pipelines  
  Mitigation: Centralize constants, document group/bind ownership in shaders; compile tests.

- Perf variability across GPUs  
  Mitigation: Tiny goldens, perf harness, optional skipping in CI without GPU; document target device.

- Warning regressions  
  Mitigation: Address unused code/variables immediately; keep Clippy delta minimal.

## Deliverables per milestone

- M1: Budgeted atlas with clear bindings; deterministic downscale behavior; tests.  
- M2: Hard/PCF/PCSS/VSM/EVSM/MSM implemented in WGSL; CSM selection/fade stable.  
- M3: Shadows bound in mesh PBR; BRDF visibility applied; optional terrain hook.  
- M4: CLI flags fully effective; Python debug prints cascade/memory info.  
- M5: Unit tests + perf logs + optional goldens.  
- M6: Shadow system docs + acceptance steps.

## File references

- Shadow system (Rust)  
  - `src/shadows/manager.rs` (atlas, bind group, budget, params)  
  - `src/shadows/csm.rs` (uniforms, cascade updates, depth/moment textures)  
  - `src/shadows/state.rs` (CPU fallback + validation helpers)  
  - `src/lighting/types.rs` (`ShadowTechnique`, settings, helpers)
- WGSL  
  - `src/shaders/shadows.wgsl` (CSM, PCF; extend for PCSS/VSM/EVSM/MSM)  
  - `src/shaders/pbr.wgsl` (apply visibility in BRDF loop)  
  - `src/shaders/terrain_pbr_pom.wgsl` (optional direct-light hook)
- Pipelines  
  - `src/pipeline/pbr.rs`, `src/render/pbr_pass.rs`, `src/core/pbr.rs` (bind group wiring, updates)
- CLI & Python  
  - `examples/terrain_demo.py` (flags already present)  
  - `python/forge3d/config.py`, `python/forge3d/__init__.py` (mapping to native config)
- Tests/Perf/Docs  
  - `tests/` unit + optional goldens under `tests/golden/p3/`  
  - `bench/` or `tests/perf/` for timing harness  
  - `docs/user/shadows_overview.rst`, `docs/P3_ACCEPTANCE.md`

## Acceptance (from p3.md)

- Visual: PCSS shows softening vs PCF/hard; acne reduced (bias/slope-bias tuned).  
- Deterministic atlas allocation under budget; cascade count and map res reproducible.  
- Performance target documented (e.g., ≤16.6 ms @ 1280×920 on a mid‑range GPU).  
- Shaders compile across win_amd64, linux_x86_64, macos_universal2.
