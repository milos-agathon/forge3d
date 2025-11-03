# P0 — Config & CLI plumbing for Lighting/BRDF/Shadow/GI/Atmosphere (no rendering changes yet)

Goal: Introduce and finalize typed config, CLI flags, and Python API to select lighting types, BRDFs, shadows, GI, and atmospherics—wired through to the renderer but behind feature gates. No shader or visual changes; this is configuration, validation, and plumbing only.

Note: Much of this surface already exists in this repo. The tasks below focus on verifying coverage vs p0.md, closing deltas, documenting, and adding tests and acceptance so the configuration is stable and discoverable.

## Milestone 1 — Rust configuration surface (enums, structs, validation)

- P0-01: Verify enum coverage and string parsing (High, 0.5 day)
  - Ensure `src/render/params.rs` enums match p0.md and cover:
    - `LightType { Directional, Point, Spot, AreaRect, AreaDisk, AreaSphere, Environment }`
    - `BrdfModel { Lambert, Phong, BlinnPhong, OrenNayar, CookTorranceGGX, CookTorranceBeckmann, DisneyPrincipled, AshikhminShirley, Ward, Toon, Minnaert, Subsurface, Hair }`
    - `ShadowTechnique { Hard, PCF, PCSS, VSM, EVSM, MSM, CSM }`
    - `GiMode { None, IBL, IrradianceProbes, DDGI, VoxelConeTracing, SSAO, GTAO, SSGI, SSR }`
    - `SkyModel { HosekWilkie, Preetham, HDRI }`
    - `VolumetricPhase { Isotropic, HenyeyGreenstein }`
  - Confirm `serde` names and `FromStr` normalization align with kebab-case/aliases.
  - Exit criteria: Unit test asserts parse-from-string round-trips for all entries and canonical names match expectations.

- P0-02: Structs and defaults + validation completeness (High, 0.5 day)
  - Confirm presence and defaults in `src/render/params.rs` for:
    - `LightingParams` (lights list, exposure)
    - `ShadingParams` (BRDF default, toggles)
    - `ShadowParams` (enabled, technique, map size/cascades, PCSS params, memory budget)
    - `GiParams` (modes, AO strength)
    - `AtmosphereParams` (enabled, sky, HDR path, optional volumetrics)
    - `RendererConfig` aggregate
  - Ensure `RendererConfig::validate()` covers:
    - Directional lights require `direction`; positional lights require `position`.
    - `environment` lights require `hdr_path` unless `atmosphere.hdr_path` provided.
    - Shadow map size power-of-two, memory budget cap, cascades in [1,4], technique-specific invariants (PCSS radii/light size positive; moments require positive bias; CSM requires `cascades >= 2`).
    - Atmosphere `sky=hdri` requires HDR path from either atmosphere or an environment light.
    - Volumetric density non-negative, HG anisotropy within [-0.999, 0.999].
    - GI `ibl` requires env light or atmosphere HDR.
  - Exit criteria: Tests in `#[cfg(test)]` pass and cover all invariants; add any missing assertions.

- P0-03: Config threading stubs behind feature gates (Medium, 0.5 day)
  - Thread `RendererConfig` through native entry points where appropriate (constructor/plumbing only) behind feature gates; no shader or pipeline behavior changes.
  - Suggested anchor points: `src/lib.rs` (PyO3 exports), `src/renderer/mod.rs`, and/​or `src/terrain_renderer.rs` for future use.
  - Provide a no-op getter for debugging in native layer if feasible (mirrors Python `Renderer.get_config()`), guarded by feature flags.
  - Exit criteria: Project builds with `cargo build --all-features` and introduces no warnings; no runtime visual change.

Notes
- Keep to the project’s warning-free goal; don’t regress rustc or clippy warnings.
- Do not change shader behavior in this phase.

## Milestone 2 — Python API and CLI wiring (validation, overrides, ergonomics)

- P0-04: Finalize `python/forge3d/config.py` normalization (Medium, 0.5 day)
  - Confirm mappings exist and align with Rust: `_LIGHT_TYPES`, `_BRDF_MODELS`, `_SHADOW_TECHNIQUES`, `_GI_MODES`, `_SKY_MODELS`, `_PHASE_FUNCTIONS`.
  - Validate `RendererConfig.from_mapping()` and `load_renderer_config()` override merging (`lighting`/`shading`/`shadows`/`gi`/`atmosphere`, plus flat keys via `_build_override_mapping`).
  - Exit criteria: Calling `load_renderer_config()` with nested and flat overrides yields validated, normalized config matching Rust schemas.

- P0-05: Python `Renderer` accepts config and exposes `get_config()` (Low, 0.25 day — verify)
  - Confirm `python/forge3d/__init__.py::Renderer` already accepts `config` and recognized kwargs, normalizes via `_split_renderer_overrides()`, and exposes `get_config()`.
  - Add/adjust docstrings if needed.
  - Exit criteria: `Renderer(width, height, brdf="toon", shadows="pcf", gi=["ibl"])` returns expected values via `get_config()`.

- P0-06: Terrain demo CLI flags (Medium, 0.5 day — verify)
  - Ensure `examples/terrain_demo.py` exposes flags:
    - `--light` (repeatable key=value spec: `type=directional,dir=...,...`)
    - `--brdf`, `--shadows`, `--shadow-map-res`, `--cascades`
    - `--gi`, `--hdr`, `--sky`
    - `--volumetric 'density=...,phase=...,g=...'`
    - `--preset` (applies preset first, then overrides)
  - Wire through `_build_renderer_config(args)` to `load_renderer_config()`; no visual behavior change beyond exposure/hdr selection being honored by example.
  - Exit criteria: One-liner acceptance works and `print(renderer_config.to_dict())` reflects flags.

Notes
- Maintain CPU-only import safety (no hard native imports at module import time).
- The interactive single-terminal viewer is out of scope for P0 (no change); ensure no regression to Automatic Snapshot behavior per user preference.

## Milestone 3 — Docs and tests

- P0-07: User docs — Rendering options overview (High, 0.5 day)
  - Add Sphinx page (e.g., `docs/user/rendering_options_overview.rst`) with:
    - Tables of config fields, defaults, and valid values.
    - CLI examples showing `--light`, `--brdf`, `--shadows`, `--gi`, `--sky`, `--volumetric`, and `--preset` usage.
    - Python examples using `forge3d.Renderer(..., config=..., brdf=..., shadows=..., gi=...)` and `.get_config()`.
  - Exit criteria: `make html` builds; page linked from the user docs index.

- P0-08: Unit tests — parsing, normalization, round-trips (High, 0.5–1 day)
  - Python tests (new): `tests/test_renderer_config.py`
    - Nested vs flat override merges in `load_renderer_config()`.
    - Normalization of enums and values (e.g., `ggx` → `cooktorrance-ggx`, `env` → `hdri` or environment light).
    - Validation error messages for common mistakes (non-POT shadow map, invalid cascades, missing HDR for IBL sky).
  - Rust tests: extend `src/render/params.rs` #[cfg(test)] if any enum/default/validation gaps remain.
  - Exit criteria: Tests pass locally and on CI; coverage includes typical user errors.

- P0-09: CLI integration smoke test (Medium, 0.5 day)
  - Add tiny smoke test (or extend `tests/test_terrain_demo.py`) to construct `_build_renderer_config()` with a few flags and assert selected fields (no rendering performed).
  - Exit criteria: Test passes in CPU-only CI.

## Milestone 4 — Acceptance, CI, and hygiene

- P0-10: Acceptance one-liner (High, 0.25 day)
  - Demonstrate the pipeline and verify config plumbing without visual change:
    - `python examples/terrain_demo.py --preset outdoor_sun --brdf cooktorrance-ggx --shadows pcf --cascades 2 --hdr assets/snow_field_4k.hdr --size 256 256 --output examples/out/p0_demo.png`
  - Confirm printed/serialized config reflects flags and preset merge; resulting image is produced (content is not part of P0 acceptance).

- P0-11: CI and lint/warnings (Low, 0.25 day)
  - Ensure tests that require a GPU are skipped; P0 tests are CPU-only.
  - Keep tree warning-free when building with `cargo build --all-features`; avoid introducing new Clippy warnings.
  - Exit criteria: CI green; no new warnings.

---

## Recommended implementation order

1) M1 enums/structs/validation verification (P0‑01..P0‑02) → optional config threading stub (P0‑03)
2) M2 Python API/CLI verification (P0‑04..P0‑06)
3) M3 docs + tests (P0‑07..P0‑09)
4) M4 acceptance + CI hygiene (P0‑10..P0‑11)

Rationale
- Confirming/closing Rust/Python config gaps unblocks proper docs/tests.
- CLI and Python API verification ensures the same normalization is available to end-users.
- Acceptance validates the “no rendering changes” constraint while proving end-to-end plumbing.

## Estimates summary (single engineer; add 30–40% for review/CI)

- P0‑01: 0.5 day  
- P0‑02: 0.5 day  
- P0‑03: 0.5 day  
- P0‑04: 0.5 day  
- P0‑05: 0.25 day  
- P0‑06: 0.5 day  
- P0‑07: 0.5 day  
- P0‑08: 0.5–1 day  
- P0‑09: 0.5 day  
- P0‑10: 0.25 day  
- P0‑11: 0.25 day

Base ship: ~4–6 days

## Risks and mitigations

- Config drift between Rust and Python schemas  
  Mitigation: Keep enums and defaults mirrored; add explicit unit tests for normalization and validation parity.

- CLI ambiguity for environment HDR source  
  Mitigation: Keep precedence documented (atmosphere.hdr_path wins; otherwise environment light hdr_path; otherwise `--hdr`). Add test.

- Unintended visual changes via examples  
  Mitigation: Ensure P0 restricts itself to configuration and validation; no shader or pipeline behavior changes.

- CI flakes requiring GPU  
  Mitigation: Tests only cover config normalization and CLI wiring; skip heavy rendering.

## Deliverables per milestone

- M1: Enum coverage validated; defaults and `RendererConfig::validate()` complete; Rust tests pass.  
- M2: Python config normalization + `Renderer.get_config()` verified; CLI flags wired; example acceptance command works.  
- M3: Sphinx docs page for rendering options; Python + Rust tests covering normalization/validation.  
- M4: Acceptance one-liner documented; CI + lint green; no new warnings.

## File references

- Rust config: `src/render/params.rs`, `src/render/mod.rs`, `src/lib.rs`, `src/renderer.rs`, `src/terrain_renderer.rs`  
- Python config and API: `python/forge3d/config.py`, `python/forge3d/__init__.py`  
- Example CLI: `examples/terrain_demo.py`  
- Docs: `docs/user/rendering_options_overview.rst` (new)  
- Tests: `tests/test_renderer_config.py` (new), existing `tests/test_terrain_demo.py` (extend)
