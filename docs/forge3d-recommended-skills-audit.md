# forge3d Recommended Skills Audit

- **Date:** 2026-07-02
- **Task source:** `docs/fable-5-forge3d-skill-creation-prompt.md`
- **Deliverable:** 6 repo-local skills under `.codex/skills/`, selected from 11 candidates after direct repo inspection.

## Inspected Areas

| Area | What was inspected |
|---|---|
| `src/` | `lib.rs` pymodule entry; `src/py_module/` registration hub (`classes.rs` ~50 `add_class`, `functions/*.rs` ~189 registrations); `src/py_functions/`, `src/py_types/`; render orchestration (`src/terrain/renderer/{core,draw,offline,pipeline_cache}.rs`, `src/core/postfx/chain.rs`); memory subsystems (`src/render/memory_budget.rs`, `src/core/{memory_tracker,staging_rings,virtual_texture,tile_cache}`); `Cargo.toml` features vs `pyproject.toml` maturin features |
| `src/shaders/` | ~126 WGSL files; `include_str!` ownership; bind-group layout co-location; terrain shader assembly (`preprocess_terrain_shader`/`strip_includes`); load-time `@group` rewrites; postfx/offline chains; tonemap fragmentation |
| `python/forge3d/` | `_native.py` loader; `__init__.py` native injection tuple + `__all__`; `.pyi` stubs; per-module fallback policy (native-only `gis.py`/`vector.py`, pyproj-fallback `crs.py`, degradable `map_scene`/`_gpu`/`mem`); `map_scene.py` + `_map_scene_*` split; `recipe_manifest.py` |
| `examples/` | `_import_shim.py`; output/cache conventions; `--snapshot` flag vocabulary; canonical-example + monkeypatch-variant pattern; top-level/subdir duplication |
| `tests/` (validation patterns only) | `test_api_contracts.py` `EXPECTED_*` locks and floors; golden lane (`test_terrain_visual_goldens.py`, `test_terrain_tv10_goldens.py`, `_ssim.py`); example paired tests (stubbed `forge3d`, CPU helpers only); `test_recipe_manifest.py` |
| `docs/`, `specs/`, root | `docs/carto-engine/` (GIS constraints, gap ledger, crosswalk, g-00x plans); `docs/gallery/` + `scripts/{regenerate_gallery,gen_gallery_images,compare_images,terrain_validation}.py`; `docs/superpowers/{plans,specs}`; `specs/001-diagnostics-support-matrices/`; `AGENTS.md`; `CONTRIBUTING.md`; `.github/workflows/ci.yml`; `codex-review/` bundle precedent; `.gitignore` PNG whitelist |

Worktree state at audit time: 2 modified files (`docs/carto-engine/golden-map-recipe-capability-audit.md`, `logs/.9a9f…-audit.json`) and 2 untracked prompt docs — all unrelated to this task and left untouched.

## Skill Selection Rationale

Ranking followed the prompt's leverage criteria (repeat frequency x mistake cost x checklist-preventability x doc-coverage gap):

1. **The PyO3 surface is the highest-frequency, highest-cost failure surface.** Four of six `AGENTS.md` sections (P0.2, P0.3, P0.4, P2.3) are registration/wrapper/feature-gate failures, and the surface spans five hand-synchronized locations (`classes.rs`/`functions/*.rs`, maturin features, `__init__.py` injection tuple + `__all__`, `.pyi`, `EXPECTED_*` lists). Failures are silent (code compiles, symbol invisible). → `pyo3-native-surface`.
2. **Shader/Rust layout drift has no static check.** Drift only surfaces at pipeline creation; the audit found a live mismatch (`pt_kernel.wgsl` group 4 binding 6 `rgba8unorm` vs `setup.rs` `R8Unorm`). Non-obvious machinery (`#include` is fake, load-time `@group` string rewrites, 7-group terrain contract) makes this checklist-preventable. → `wgsl-shader-pipeline`.
3. **Quality claims need measurable evidence.** The repo has a full evidence toolchain (golden lane with SSIM≥0.995 / mean-abs≤2.0 thresholds, `compare_images.py`, `terrain_validation.py`) that agents can use instead of subjective claims — plus a footgun (MapScene silently falls back to a non-native renderer). → `render-quality-validation` (merged from the rendering-quality-audit + visual-golden-validation candidates, which shared the same tools and evidence rules).
4. **GIS has written, violable invariants.** `rust-gis-implementation-plan.md` states hard rules (no Python backend, no CRS guessing, explicit resampling, separate raster/vector reprojection APIs) that generic agents routinely violate by reaching for rasterio/pyproj. → `gis-carto-contract`.
5. **Examples/gallery/manifests form a three-layer consistency problem** with two divergent gallery generators, committed images CI never regenerates, and a brand-new manifest layer whose last two commits were exactly alignment fixes. → `example-gallery-recipes`.
6. **The dev loop and review bundle are already practiced but undocumented as a skill.** Every g-00x plan mandates temp review bundles outside the repo; `codex-review/` shows the bundle shape; the prompt requires constraining autonomous execution to one bounded slice. → `slice-review-loop` (merged from the feature-slicing + review-bundle candidates, which are two halves of one loop).

Six skills is the prompt's ceiling; it was reached only by merging three candidate pairs — no skill was split to inflate the count, and each remaining skill encodes a distinct recurring workflow with distinct validation.

## External Workflow Pattern Finding (dzhng/skills-style patterns)

Treated as a pattern library only; no files copied.

| Pattern | Disposition | Where |
|---|---|---|
| Feature slicing (independently verifiable slices, API seams, typed contracts, reslicing) | **Adopted, customized** to forge3d seams: native registration + wrapper + contract-test as one seam; `.pyi`/dataclass/closed-vocabulary typed contracts; visual checkpoints via the golden lane | `slice-review-loop` |
| Renderer/WebGPU review discipline (shader contracts, bind groups, depth semantics, resource ownership, perf gates) | **Adopted, customized** to forge3d's actual machinery: `include_str!` ownership, fake `#include`, `@group` rewrites, `PostFxResourcePool` ownership, pipeline-creation-time-only validation | `wgsl-shader-pipeline` |
| Screenshot critique (full frame + zoomed crops, no subjective claims) | **Adopted, customized** with repo metrics: `compare_images.py` SSIM/MAE, golden thresholds, backend labeling (`last_render_backend`) | `render-quality-validation` |
| Refactor-clean discipline (prefer deletion, converge on existing abstractions, avoid sediment) | **Adopted, embedded** rather than made a standalone skill: anti-sediment boundaries in `wgsl-shader-pipeline` (duplicate AO precedent) and refactor discipline + 300-line guideline in `slice-review-loop` | both |
| Generic docs discipline | **Adopted narrowly** where it supports correctness (schema doc + `.pyi` + vocab updated together; `AGENTS.md` lessons appended per slice); rejected as a standalone generic skill | `example-gallery-recipes`, `slice-review-loop` |
| Autonomous "keep going until everything is done" loop | **Rejected unmodified.** Replaced with bounded-slice execution and four hard stop conditions (slice boundary, cross-contract change, out-of-slice golden failure, ambiguity) | `slice-review-loop` |

Gaps the external patterns could not cover (forge3d-specific content written from repo evidence): PyO3 registration/wrapper contracts, WGSL-as-implemented details, GIS/CRS/no-Python-backend rules, recipe-manifest provenance rules.

## Created Skills

| Skill | Trigger / use case | Why needed | Evidence from repo | Key validation commands |
|---|---|---|---|---|
| `pyo3-native-surface` | Adding/renaming/exposing native API; symbol compiles in Rust but missing in Python | Silent-failure surface across 5 hand-synced locations; 4 of 6 `AGENTS.md` sections are this failure class | `src/py_module/classes.rs` (~50 `add_class`), `functions/*.rs` (~189 regs); maturin features `[extension-module, weighted-oit, enable-tbn, enable-gpu-instancing, copc_laz, gis-remote]` narrower than Cargo/CI; `test_api_contracts.py` locks + floors; `AGENTS.md` P0.2–P0.4, P2.3 | `maturin develop`; `python -m pytest tests/test_api_contracts.py -q`; hasattr probe; `cargo forge3d-clippy` |
| `wgsl-shader-pipeline` | Editing WGSL, changing `@group/@binding`, pipeline-creation errors, adding passes | No static drift check — mismatches surface only at pipeline creation; non-obvious assembly machinery; a live format mismatch found during audit | `include_str!` ownership; `pipeline_cache.rs` fake-`#include` assembly + documented 7-group terrain contract; `.replace("@group(2)", "@group(3)")` in `src/pipeline/pbr/rendering.rs`; sole shader test `tests/verify_terrain_pbr_pom_shader.py`; duplicate `ao/` vs `ssao/` sediment | `python -m pytest tests/verify_terrain_pbr_pom_shader.py -q`; subsystem runtime test after `maturin develop`; `python scripts/terrain_ci_probe.py` |
| `render-quality-validation` | Any rendering-quality claim or audit; golden failures; golden regeneration | Converts subjective claims into metric evidence; regeneration and fallback-backend footguns | Golden lane gates (`FORGE3D_RUN_TERRAIN_GOLDENS`, `FORGE3D_UPDATE_TERRAIN_GOLDENS`, artifact dir), thresholds SSIM≥0.995 / mean-abs≤2.0 in `tests/test_terrain_visual_goldens.py`; `tests/_ssim.py`; `scripts/compare_images.py`; `.gitignore` PNG whitelist; `MapScene.last_render_backend` | golden lane pytest run with env gate; `python scripts/compare_images.py --ssim --diff --json`; `scripts/terrain_validation.py` |
| `gis-carto-contract` | GIS raster/vector/CRS/resampling/remote work; g-00x items | Written hard invariants that generic agents violate (Python backends, guessed CRS, implicit resampling) | Global constraints in `docs/carto-engine/rust-gis-implementation-plan.md`; `gis.py` `_require_native()`; `crs.py` sole fallback; `src/py_module/functions/gis.rs` (~78 regs); "G-002 Later" helper block in contract test; commit `3db7f46` (no native-tls) | `python -m pytest tests/test_api_contracts.py tests/test_crs_reproject.py tests/test_crs_auto.py tests/test_gis_alignment_windowing.py -q`; `cargo test gis` with CI features |
| `example-gallery-recipes` | Adding/modifying examples; regenerating gallery; editing recipe manifests | Three-layer alignment problem; two divergent gallery generators; CI never regenerates committed images; manifest source-tracking rules just landed | `examples/_import_shim.py`; germany→bosnia monkeypatch variant pattern; `scripts/regenerate_gallery.py` vs legacy `gen_gallery_images.py`; `recipe_manifest.py` closed vocabularies + byte-for-byte fixtures; commits `b7fab06`/`00b5803` | `python -m pytest tests/test_recipe_manifest.py -q`; paired example test; `git diff --stat docs/gallery/images` review |
| `slice-review-loop` | Starting a roadmap/plan item; multi-subsystem features; preparing external review | Encodes the practiced loop with stop conditions; prevents unsafe repo-wide autonomous runs; standardizes the bundle | Bundle mandate in every g-00x plan; `codex-review/` bundle shape (status/diff-stat/name-status/full patch/untracked/metadata/validation-logs); gitignored `codex-review/`; `CONTRIBUTING.md` PR rules; branch pattern `codex/<slug>` | focused slice tests + `tests/test_api_contracts.py` when surface moved; `cargo forge3d-clippy`; bundle contains real test logs; `git status --short` scope check |

## Rejected / Deferred Skill Candidates

| Candidate | Reason rejected or deferred |
|---|---|
| Studio-runtime skill | **Rejected — no evidence.** `RenderSpec`, "sandboxed worker", and a `RenderSpec -> MapScene` compiler exist only as conditional language in the skill-creation prompt itself. Repo-wide grep found no Studio subsystem (only a lighting preset named "Studio" in `presets.py` and Visual Studio build refs). The prompt gates this skill on evidence that is absent. |
| Forge3D architecture skill (standalone) | **Merged, not created separately.** Rust-core-ownership / thin-wrapper / anti-sediment rules are embedded where they are enforced: `pyo3-native-surface` (ownership + wrapper contract), `wgsl-shader-pipeline` (sediment boundary), `slice-review-loop` (refactor discipline, 300-line guideline). A standalone skill would duplicate all three. |
| Rendering-quality-audit (standalone) | **Merged** into `render-quality-validation` — same tools (`compare_images.py`, goldens, `terrain_validation.py`), same evidence rules; two skills would overlap heavily. |
| Review-bundle skill (standalone) | **Merged** into `slice-review-loop` — the bundle is one step of the loop; separating it invited a bundle without slice discipline or vice versa. |
| Performance/memory skill | **Deferred.** Subsystems are real (`memory_budget.rs`, `memory_tracker`, `virtual_texture`, `tile_cache`, `staging_rings`, 512 MiB budget in `mem.py`) but the audit found no recurring agent workflow or repeated failure pattern around them yet — no `AGENTS.md` lesson, no dedicated validation lane beyond `tests/test_memory_budget.py`. Create when profiling/streaming work becomes a recurring workstream. |
| Generic Rust coding / Python testing / documentation skills | **Rejected** per operating rules — no forge3d-specific procedure that differs from what a competent agent already does. |

## Maintenance Notes

**When to update each skill:**
- `pyo3-native-surface`: when registration moves out of `src/py_module/` (note: `AGENTS.md` P0.3 still says `src/lib.rs` from before the refactor — the skill reflects the current layout); when `[tool.maturin].features` changes; when `__init__.py` line anchors drift materially; when contract-test floors are intentionally rebased.
- `wgsl-shader-pipeline`: when the `pt_kernel` group-4/binding-6 format drift is fixed (remove the "known drift" note); if a standalone naga/WGSL validation harness is added (rewrite the Validation section around it); if the duplicate `ao/` vs `ssao/` set is consolidated; if `#include` handling becomes a real preprocessor.
- `render-quality-validation`: when golden thresholds or env-var names change; when goldens exist outside `tests/golden/terrain/`; when the `.gitignore` PNG whitelist changes.
- `gis-carto-contract`: when deferred `src/gis/` modules (`remote.rs`, `tiles.rs`, `osm.rs`, `terrarium.rs`) land; when gap ledger items G-001…G-011 close; if the no-Python-backend policy is ever relaxed (would require rewriting the skill's hard rules).
- `example-gallery-recipes`: when the legacy `gen_gallery_images.py` is deleted (drop the disambiguation); when the examples top-level/subdir duplication is resolved to one layout; when manifest `_SCHEMA_VERSION` bumps past "1"; when the deferred golden selection (`recipe_manifest_golden_not_selected`) is implemented.
- `slice-review-loop`: when a bundle-generation script is added to `scripts/` (replace the manual file list with the command); if the planning-doc landscape changes (a real roadmap file appears, or g-00x/TV/specs numbering is restructured).

**What would make a skill obsolete:**
- An automated registration/codegen layer (e.g. generated stubs + contract lists) would obsolete most of `pyo3-native-surface`.
- Reflection-based bind-group layout generation or a CI naga+layout-diff check would obsolete much of `wgsl-shader-pipeline`.
- A CI job that runs the golden lane and gallery regeneration automatically would shrink `render-quality-validation` and `example-gallery-recipes` to their claim-evidence rules.
- Moving GIS constraints into enforced code (lint/test that forbids reference-library imports, CRS-required types) would let `gis-carto-contract` shrink to a pointer.
- A `forge3d`-specific bundle CLI plus PR template would reduce `slice-review-loop` to slice-definition guidance.
