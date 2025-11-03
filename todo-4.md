# P4 — IBL pipeline (diffuse irradiance + specular prefilter + BRDF LUT)

Goal: High-quality environment lighting without RT. This plan implements offline/first-frame precompute (compute shaders), runtime bindings, Python/CLI wiring, docs, and acceptance tests with cache reuse.

## Milestone 1 — Precompute (compute shaders) + Cache

- P4-01: Equirectangular → cubemap conversion (High, 0.5–1 day)
  - Ensure `src/core/ibl.rs` uses compute entry `cs_equirect_to_cubemap` from `src/shaders/ibl_equirect.wgsl` to generate a base cubemap at `IBLQuality::base_environment_size()`.
  - Validate `Rgba16Float` format and face write via `D2Array` view; check row padding logic.
  - Exit criteria: Generated cubemap has 6 faces, correct orientation, no seams, smoke tested with a small HDR.

- P4-02: Irradiance convolution (Lambertian) (High, 0.5–1 day)
  - Implement/verify `cs_irradiance_convolution` in `src/shaders/ibl_prefilter.wgsl` and call from `IBLRenderer::generate_irradiance_map()`.
  - Size from `IBLQuality::irradiance_size()`; store as `Rgba16Float` cube.
  - Exit criteria: Diffuse-only shaded scene brightens consistently; energy-conserving response under diffuse-only visualization.

- P4-03: GGX specular prefilter (mip chain) (High, 1 day)
  - Implement/verify `cs_specular_prefilter` in `src/shaders/ibl_prefilter.wgsl` with roughness→mip mapping and sample count.
  - Generate full mip chain per `IBLQuality::specular_mip_levels()`; `IBLRenderer::generate_specular_map()` loops mips.
  - Exit criteria: Visible specular lobes narrowing with low roughness; correct mip addressing; no NaNs.

- P4-04: BRDF 2D LUT (NdotV × roughness) (High, 0.5 day)
  - Implement/verify `cs_brdf_integration` in `src/shaders/ibl_brdf.wgsl` and generation in `IBLRenderer::generate_brdf_lut()`.
  - Store as `Rgba16Float` 2D texture of size `IBLQuality::brdf_size()`.
  - Exit criteria: LUT sampling returns stable values in [0,1]; unit tests cover edge UVs.

- P4-05: On-disk cache (.iblcache) keyed by HDR + resolution + GGX settings (High, 0.5–1 day)
  - Ensure `IBLRenderer::{configure_cache,write_cache,try_load_cache}` serialize metadata (hdr path, dims, quality, sizes, mips, brdf size, sha256) and binary payloads.
  - Verify row padding and payload sizes via `cubemap_data_len` checks.
  - Exit criteria: First run computes; second run loads cache and is measurably faster (>3× on local dev). File extension `.iblcache` respected.

Notes
- Keep memory usage aligned with `lighting/memory_budget.rs` and degrade quality if budget exceeded (already wired via `ibl_wrapper.rs`).

## Milestone 2 — WGSL bindings + runtime integration

- P4-06: Unified IBL bindings (High, 0.5 day)
  - Adopt the spec bindings repo-wide: `@group(2) @binding(0) envSpecular: texture_cube<f32>`, `(1) envIrradiance: texture_cube<f32>`, `(2) envSampler: sampler`, `(3) brdfLUT: texture_2d<f32>`.
  - Ensure `src/shaders/terrain_pbr_pom.wgsl` matches (it already aligns). Add shared IBL include helpers if needed.
  - Exit criteria: Terrain path compiles and samples IBL via the unified layout.

- P4-07: PBR pipeline (mesh path) migration to cubemaps (High, 1–1.5 days)
  - Update `src/shaders/pbr.wgsl` to use cubemap irradiance/prefilter resources per the unified layout (currently uses 2D textures).
  - Update `src/pipeline/pbr.rs` `PbrPipelineWithShadows` to create an IBL bind group with cube views and BRDF LUT 2D; remove 2D fallback for irradiance/prefilter (or keep as debug-only path behind a feature flag).
  - Exit criteria: Mesh PBR path renders with IBL parity to terrain path; no bind layout mismatch.

- P4-08: Split-sum helper `eval_ibl(...)` (Medium, 0.5 day)
  - Add a shared WGSL include (e.g., `src/shaders/lighting.wgsl`) implementing `eval_ibl(n, v, base_color, metallic, roughness)` using split-sum (irradiance + prefilter × LUT).
  - Use in both terrain and PBR shaders to keep logic consistent.
  - Exit criteria: Identical visual response between terrain and mesh for the same material.

## Milestone 3 — Python bridge + CLI wiring

- P4-09: Python IBL wrapper polish (Medium, 0.5 day)
  - Validate `src/ibl_wrapper.rs` PyO3 API: `IBL.from_hdr()`, `set_base_resolution`, `set_cache_dir`, auto-quality.
  - Ensure import-safety on CPU-only systems (defer GPU allocation until needed).
  - Exit criteria: `examples/terrain_demo.py` can control IBL resolution and cache without import failures.

- P4-10: Terrain demo CLI integration (done; verify) (Low, 0.25 day)
  - Confirm flags: `--gi ibl`, `--hdr`, `--ibl-res`, `--ibl-cache` in `examples/terrain_demo.py`.
  - Validate: `ibl.set_base_resolution(--ibl-res)`, `ibl.set_cache_dir(--ibl-cache)`; error out if `--ibl-res <= 0`.
  - Exit criteria: CLI one-liner from `p4.md` runs and writes output.

- P4-11: IBL gallery example options (Low, 0.25–0.5 day)
  - Extend `examples/ibl_gallery.py` to accept `--ibl-res` and `--ibl-cache` and pass to `IBL` if native path is used.
  - Exit criteria: Rotation sweep respects resolution and uses cache when provided.

- P4-12: Viewer commands for single-terminal workflow (Medium, 0.5–1 day)
  - Add interactive commands to `src/viewer/mod.rs` (and Python viewer bindings if applicable): `ibl on/off`, `ibl intensity <v>`, `ibl rotate <deg>`, `ibl load <hdr>`, `ibl cache <dir>`, `ibl res <n>`.
  - Preserve Automatic Snapshot mode integration per user preference.
  - Exit criteria: After `viewer` starts, commands can be issued at the same prompt to tweak IBL and snapshot.

## Milestone 4 — Tests and acceptance

- P4-13: Unit tests for IBL object and parameters (High, 0.5 day)
  - Extend `tests/test_ibl.py` with cache round-trip tests (create TMP cache dir; run twice; second run faster or cache hit reported).
  - Validate quality→texture size mapping (irradiance/specular mips, brdf size) from `IBLQuality`.
  - Exit criteria: Tests pass on CI without GPU flakiness.

- P4-14: Integration tests (Medium, 0.5–1 day)
  - Add `tests/test_p4_ibl_cache.py` to assert `.iblcache` created, metadata parsed, and payload sizes match expectations; warm vs cold timing heuristic (skip if no GPU timer available).
  - Add a rendering integration that toggles IBL on/off and asserts image arrays differ; keep sizes tiny to avoid runtime blowups.
  - Exit criteria: CI green; determinism within tolerance.

- P4-15: Energy-conservation sanity checks (Medium, 0.5 day)
  - Validate that average brightness increases with IBL intensity and that metallic vs dielectric trends match expectations.
  - Use scenes similar to `tests/test_b15_ibl_polish.py` but scoped to P4 acceptance.
  - Exit criteria: Stable monotonicity checks pass.

## Milestone 5 — Docs and examples

- P4-16: User docs for IBL (High, 0.5–1 day)
  - Update `docs/environment_mapping.md` (or add `docs/user/ibl_overview.rst`): IBL concepts, CLI usage, cache behavior, quality levels, and troubleshooting.
  - Include WGSL binding layout reference and a minimal code snippet.
  - Exit criteria: `make html` succeeds; docs linked from main index.

- P4-17: Example how-to updates (Low, 0.25 day)
  - Add a short “IBL quickstart” with `examples/terrain_demo.py` one-liner and gallery thumbnails.
  - Exit criteria: Rendered thumbnails included; paths stable.

## Milestone 6 — CI, lint, and warnings

- P4-18: CI wiring and skip conditions (Medium, 0.5 day)
  - Ensure tests that require GPU/adapter skip gracefully when unavailable.
  - Add cache tests that operate purely on serialized bytes without GPU compute where possible.
  - Exit criteria: Consistent CI runtimes; no spurious flakes.

- P4-19: Warnings/Clippy hygiene (Low, 0.25 day)
  - Keep tree warning-free when building with `cargo build --all-features`; reduce Clippy deltas if touched.
  - Exit criteria: No new warnings introduced by P4.

---

## Recommended implementation order

1) M1 precompute + cache (P4‑01..P4‑05)  
2) M2 bindings + runtime (P4‑06..P4‑08)  
3) M3 Python/CLI + viewer (P4‑09..P4‑12)  
4) M4 tests/acceptance (P4‑13..P4‑15)  
5) M5 docs (P4‑16..P4‑17) → M6 CI/lint (P4‑18..P4‑19)

Rationale
- Precompute and cache unblock all runtime and CLI work.  
- Bindings unification ensures both terrain and mesh paths look consistent.  
- Viewer commands align with single-terminal workflow preference and aid acceptance.  
- Tests and docs finalize the feature and stabilize CI.

## Estimates summary (single engineer; add 30–40% overlap for review/CI)

- P4‑01: 0.5–1 day  
- P4‑02: 0.5–1 day  
- P4‑03: 1 day  
- P4‑04: 0.5 day  
- P4‑05: 0.5–1 day  
- P4‑06: 0.5 day  
- P4‑07: 1–1.5 days  
- P4‑08: 0.5 day  
- P4‑09: 0.5 day  
- P4‑10: 0.25 day  
- P4‑11: 0.25–0.5 day  
- P4‑12: 0.5–1 day  
- P4‑13: 0.5 day  
- P4‑14: 0.5–1 day  
- P4‑15: 0.5 day  
- P4‑16: 0.5–1 day  
- P4‑17: 0.25 day  
- P4‑18: 0.5 day  
- P4‑19: 0.25 day  

Base ship: ~6–9 days

## Risks and mitigations

- Binding layout mismatch between terrain and mesh paths  
  Mitigation: Unify WGSL layouts (P4‑06) and add compile-time asserts where viable.

- 2D vs cubemap migration in PBR path  
  Mitigation: Migrate `pbr.wgsl` and `PbrPipelineWithShadows` (P4‑07) with a fallback flag.

- GPU memory budget/time on low-end adapters  
  Mitigation: Auto-quality downgrades in `ibl_wrapper.rs` memory budget path; keep `IBLQuality` scalable.

- Cache portability across platforms and shader changes  
  Mitigation: Include versioning in `.iblcache` metadata; invalidate on mismatch; include sha256.

- CI flakiness without GPU  
  Mitigation: Skip heavy tests; test cache serialization logic CPU-side; keep image-based tests tiny.

## Deliverables per milestone

- M1: Cubemap generation, irradiance, specular, BRDF LUT; `.iblcache` read/write round-trip.  
- M2: Unified WGSL bindings; PBR path migrated to cubemaps; shared `eval_ibl`.  
- M3: Python wrapper polished; `terrain_demo.py` verified; `ibl_gallery.py` flags; viewer IBL commands.  
- M4: Unit + integration tests including cache round-trip and energy sanity checks.  
- M5: Docs updated with CLI and binding layout; thumbnails.  
- M6: CI & lint green; no new warnings.

## Acceptance (from p4.md)

- Compare with/without IBL; energy-conserving response; cached precompute time decreases on second run.  
- CLI one-liner works: `python examples/terrain_demo.py --gi ibl --hdr assets/sky_4k.hdr --ibl-res 256 --ibl-cache .cache/ibl` (or repo asset).  
- Viewer can toggle IBL at single terminal prompt; Automatic Snapshot preserved.

## File references

- Precompute & cache: `src/core/ibl.rs`, `src/lighting/ibl_cache.rs`, `src/ibl_wrapper.rs`, `src/shaders/ibl_equirect.wgsl`, `src/shaders/ibl_prefilter.wgsl`, `src/shaders/ibl_brdf.wgsl`  
- Terrain shader path: `src/shaders/terrain_pbr_pom.wgsl`, `src/terrain_renderer.rs`  
- Mesh PBR path: `src/shaders/pbr.wgsl`, `src/pipeline/pbr.rs`  
- Python & examples: `examples/terrain_demo.py`, `examples/ibl_gallery.py`  
- Docs: `docs/environment_mapping.md` (or `docs/user/ibl_overview.rst`)  
- Tests: `tests/test_ibl.py`, `tests/test_b15_ibl_polish.py`, add `tests/test_p4_ibl_cache.py`