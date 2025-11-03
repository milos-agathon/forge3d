# P2 — BRDF library + material routing (switchable shading models)

Goal: Implement a BRDF library in WGSL and a clean dispatch to choose the model per-material or via a global override. Route parameters from CPU→GPU consistently. Preserve current visuals as defaults while enabling model switching.

Note: Some dispatch scaffolding already exists (see `src/shaders/pbr.wgsl::ShadingParamsGPU` and simple `if`-based dispatch). This plan formalizes the library, extends supported models, centralizes common math, and wires per-material/global routing without regressing pipelines.

## Milestone 1 — WGSL BRDF module and common math

- P2-01: Create `src/shaders/brdf/` module (High, 1 day)
  - Add:
    - `common.wgsl`: shared helpers: Schlick Fresnel (incl. roughness variant), geometry terms (Smith GGX/Beckmann), NDFs (Trowbridge-Reitz GGX, Beckmann), VNDF sampling hooks, Oren–Nayar helpers.
    - Model files: `lambert.wgsl`, `phong.wgsl`, `oren_nayar.wgsl`, `cook_torrance.wgsl` (GGX + Beckmann), `disney_principled.wgsl`, `ashikhmin_shirley.wgsl`, `ward.wgsl`, `toon.wgsl`, `minnaert.wgsl`.
    - `dispatch.wgsl` exposing:
      - `struct ShadingParamsGPU { brdf_model: u32; metallic: f32; roughness: f32; sheen: f32; clearcoat: f32; subsurface: f32; anisotropy: f32; }`
      - `fn eval_brdf(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, base_color: vec3<f32>, params: ShadingParamsGPU) -> vec3<f32>` with a `switch` by `brdf_model` covering all models; default to Lambert.
  - Exit criteria: All new WGSL files compile (via existing pipelines or a minimal compile test), and `dispatch.wgsl` can be imported by `pbr.wgsl` and terrain shaders without symbol conflicts.

- P2-02: Choose safe bindings for `ShadingParamsGPU` (Medium, 0.25 day)
  - Mesh PBR (`src/shaders/pbr.wgsl`) already uses `@group(0) @binding(2)` for `shading: ShadingParamsGPU`.
  - Terrain shader (`src/shaders/terrain_pbr_pom.wgsl`) has its own `TerrainShadingUniforms` at `@group(0) @binding(5)`; keep as-is for terrain-specific knobs in P2. Optionally add a bridging struct or a subset mapping for BRDF evaluation (see P2-05).
  - Document per-pipeline binding indices to avoid collisions with lights (`src/shaders/lights.wgsl` uses group(0) bindings 3–5) and IBL.
  - Exit criteria: A short doc comment in each pipeline shader enumerates group/binding ownership; no binding collisions at runtime.

Notes
- Keep math stable and numerically robust; clamp roughness to [0.04, 1.0] as currently practiced to avoid singularities.

## Milestone 2 — Integrate BRDF dispatch into mesh PBR pipeline

- P2-03: Refactor `src/shaders/pbr.wgsl` to call `brdf/dispatch.wgsl` (High, 0.5–1 day)
  - Replace the current `if`-based branches with calls to `eval_brdf(...)` for direct lighting.
  - Ensure constants for `brdf_model` align with Rust/CPU side (see `src/lighting/types.rs::BrdfModel`).
  - Keep normal mapping, metallic/roughness maps, AO, emissive, IBL code paths intact.
  - Exit criteria: Visual parity for `CookTorranceGGX` vs current default; switching to `Lambert`/`Phong`/`Disney` yields visible, expected changes.

- P2-04: Add ornamental models with safe fallbacks (Medium, 0.5 day)
  - Implement non-PBR models (`toon`, `minnaert`) with clear documentation and clamps.
  - Default to Lambert if a model is unsupported in a given pass.
  - Exit criteria: Shader compiles and runs for all model IDs; `toon`/`minnaert` produce distinct lobes on test meshes.

## Milestone 3 — Terrain path compatibility (optional for P2)

- P2-05: Optional BRDF hook in terrain shader (Medium, 0.5 day)
  - Where terrain currently computes Cook-Torrance terms (`calculate_pbr_brdf`), optionally route the direct-light term through `eval_brdf(...)` when a feature gate is enabled.
  - Map `TerrainShadingUniforms` → `ShadingParamsGPU` subset (roughness, metallic; ignore unsupported knobs) for the call site.
  - Exit criteria: Flagged build maintains current terrain look by default; enabling the hook allows BRDF switching on terrain without breaking existing knobs.

Notes
- Terrain shading includes specialized triplanar/POM logic; do not regress that. The BRDF hook should affect only the microfacet/lobe computation.

## Milestone 4 — CPU/Backend routing and overrides

- P2-06: Route material shading params to GPU (High, 0.5 day)
  - Leverage `src/lighting/types.rs::MaterialShading` (already GPU-aligned) to populate `ShadingParamsGPU` for mesh PBR.
  - Add a small bridge in pipeline setup to write the uniform with `brdf`, `roughness`, `metallic`, etc.
  - Exit criteria: Changing material shading on CPU updates the uniform and results in expected BRDF selection.

- P2-07: Global override via `RendererConfig.brdf_override` (Medium, 0.5 day)
  - Implement precedence: `brdf_override` (if set) wins over per-material setting.
  - Wire Python `RendererConfig` (already present in `python/forge3d/config.py`) through PyO3 to set the override in the renderer.
  - Exit criteria: `Renderer(..., brdf="lambert")` or `Renderer.apply_preset(..., brdf=...)` selects the BRDF regardless of per-material.

Notes
- Respect the ongoing warning-cleanliness guideline; do not introduce dead code or unused bindings.

## Milestone 5 — Tests, goldens, and cross-platform compile

- P2-08: Shader compile smoke tests (High, 0.5 day)
  - Add a compile-time test (Rust) that creates the `pbr` pipeline and ensures WGSL includes import and binding layouts are valid on CI (no render required).
  - Exit criteria: CI passes on Linux/macOS/Windows builders.

- P2-09: Golden images demonstrating BRDF differences (High, 0.5–1 day)
  - Render a tiny scene (e.g., a UV-mapped sphere or simple mesh) at small resolution for 3 models: Lambert, CookTorranceGGX, Disney.
  - Save to `tests/golden/p2/` and add tolerances suitable for GPU variability.
  - Exit criteria: Goldens update only on intentional changes; visible lobe differences for the 3 models.

- P2-10: Python unit tests for override precedence (Medium, 0.5 day)
  - Tests exercising `RendererConfig.brdf_override` vs per-material settings via Python API; assert the uniform model index chosen.
  - Exit criteria: Passing tests on CPU-only CI (use stubs/mocks if native path not available).

## Milestone 6 — Docs and examples

- P2-11: BRDF reference docs (Medium, 0.5–1 day)
  - Add `docs/user/brdf_overview.rst` describing each model, expected use, and parameter knobs.
  - Include a small image panel (from goldens) showing model differences; link to terrain note about specialized shading.
  - Exit criteria: `make html` builds; page linked from user docs index.

- P2-12: Example script (Low, 0.25 day)
  - Add `examples/brdf_gallery.py` rendering a 3×N grid of BRDFs across a roughness sweep; small sizes to run quickly.
  - Exit criteria: Script runs and saves a mosaic; honors global override and per-material settings.

---

## Recommended implementation order

1) P2‑01/02 WGSL module + bindings  
2) P2‑03/04 integrate into mesh PBR  
3) P2‑05 (optional) terrain hook behind a flag  
4) P2‑06/07 CPU routing + overrides  
5) P2‑08/09/10 tests + goldens  
6) P2‑11/12 docs + examples

Rationale
- Establishing the module and dispatch API first enables pipeline integration.  
- Tests/goldens stabilize functionality across platforms.  
- Docs/examples improve discoverability and acceptance.

## Estimates summary (single engineer; add 30–40% for review/CI)

- P2‑01: 1 day  
- P2‑02: 0.25 day  
- P2‑03: 0.5–1 day  
- P2‑04: 0.5 day  
- P2‑05: 0.5 day  
- P2‑06: 0.5 day  
- P2‑07: 0.5 day  
- P2‑08: 0.5 day  
- P2‑09: 0.5–1 day  
- P2‑10: 0.5 day  
- P2‑11: 0.5–1 day  
- P2‑12: 0.25 day

Base ship: ~5–7 days

## Risks and mitigations

- Binding/layout conflicts across pipelines  
  Mitigation: Document group/bind choices per pipeline; centralize constants; add compile tests.

- Numerical instability and look drift  
  Mitigation: Clamp roughness; unit-test functions in `common.wgsl` where feasible; verify GGX parity with current visuals.

- GPU variability affecting goldens  
  Mitigation: Tiny images, exposure-agnostic comparisons, tolerance windows, optional CPU reference path if available.

- Warning regressions  
  Mitigation: Fix unused variables, remove dead code; run `cargo clippy` locally; keep CI clean.

## Deliverables per milestone

- M1: `src/shaders/brdf/` module with `common`, models, `dispatch`; compiles.  
- M2: Mesh PBR uses `eval_brdf`; model switching visible; parity for GGX.  
- M3: Optional terrain hook under a feature flag; no regression by default.  
- M4: CPU routing + global override wired; Python API exercises override.  
- M5: Shader compile tests + golden images + override tests.  
- M6: Docs and `examples/brdf_gallery.py`.

## File references

- WGSL BRDF module  
  - `src/shaders/brdf/common.wgsl` (new)  
  - `src/shaders/brdf/lambert.wgsl`, `phong.wgsl`, `oren_nayar.wgsl`, `cook_torrance.wgsl`, `disney_principled.wgsl`, `ashikhmin_shirley.wgsl`, `ward.wgsl`, `toon.wgsl`, `minnaert.wgsl` (new)  
  - `src/shaders/brdf/dispatch.wgsl` (new)
- Mesh PBR pipeline  
  - `src/shaders/pbr.wgsl` (refactor to call dispatch)
- Terrain shader (optional hook)  
  - `src/shaders/terrain_pbr_pom.wgsl` (map to `ShadingParamsGPU` subset under a feature flag)
- CPU/Backend  
  - `src/lighting/types.rs` (`MaterialShading`, `BrdfModel`)  
  - pipeline setup where `ShadingParamsGPU` uniform is written (renderer modules)
- Python config/API  
  - `python/forge3d/config.py` (`ShadingParams`, `RendererConfig.brdf_override`)  
  - `python/forge3d/__init__.py` (ensure override path is honored)
- Tests/Goldens  
  - `tests/golden/p2/` (new), shader compile tests under `tests/`  
- Docs/Examples  
  - `docs/user/brdf_overview.rst` (new)  
  - `examples/brdf_gallery.py` (new)

## Acceptance (from p2.md)

- Golden images: same scene rendered with Lambert vs GGX vs Disney shows distinct lobes.  
- Shaders compile across win_amd64, linux_x86_64, macos_universal2 (CI).  
- Global override and per-material routing both work; defaults preserve current look.
