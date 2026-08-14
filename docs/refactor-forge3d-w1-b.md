# W1-B read-only audit: Python, PyO3, tests, examples, and recipes

## Outcome and provenance

W1-B completed against exact base and head
`f5db54f95d202681f95dad649162d18efdae8987` in worktree
`/private/tmp/forge3d-refactor-20260812`, branch
`codex/refactor-forge3d-20260812`.

The audit was read-only. It made no source, manifest, lockfile, test, example,
fixture, golden, certificate, generated-output, or user-work change. No build,
test, network operation, commit, push, or delegated task was performed. The
only pre-existing worktree change observed before this report was the untracked
`docs/refactor-forge3d.md` ledger.

## Exact owned manifest

The manifest authority was `git ls-files` with this exact inclusion predicate:

```text
path == "src/lib.rs"
OR path starts with "src/py_functions/"
OR path starts with "src/py_module/"
OR path starts with "src/py_types/"
OR path starts with "src/scene/py_api/"
OR path starts with "python/"
OR path starts with "tests/"
OR path starts with "examples/"
OR path starts with "bench/"
OR path starts with "benches/"
OR path starts with "data/"
OR path == "conftest.py"
OR path == "pytest.ini"
```

The resulting newline-delimited manifest contains exactly **1,272 tracked
paths** and has SHA-256 digest
`51efb1a88ad236ce8a0214d189408ccb041d83af061d83f2e9fcb86b6245bcdf`.

| Owned surface | Tracked paths |
|---|---:|
| `src/lib.rs` | 1 |
| `src/py_functions/**` | 37 |
| `src/py_module/**` | 20 |
| `src/py_types/**` | 10 |
| `src/scene/py_api/**` | 25 |
| `python/**` | 140 |
| `tests/**` | 988 |
| `examples/**` | 46 |
| `bench/**` | 1 |
| `benches/**` | 1 |
| `data/**` | 1 |
| `conftest.py` | 1 |
| `pytest.ini` | 1 |
| **Total** | **1,272** |

Extension classification was: 546 `.json`, 499 `.py`, 99 `.rs`, 63 `.png`,
18 `.pyi`, 9 `.toml`, 7 `.txt`, 6 `.npy`, 5 `.ttf`, 4 `.md`, 3 `.sha256`,
3 `.ipynb`, 3 `.dat`, and one each of `.wgsl`, `.typed`, `.tif`, `.pub`,
`.pdb`, `.ini`, and `.geojson`.

The small-file pass included one-line binary arrays, hashes, public keys,
package markers, and two-to-five-line module files. The large-file pass included
the Unicode corpora, `python/forge3d/map_scene.py`, smoke examples, root stubs,
and long test/example modules. File length was used only for navigation.

The torture JSON corpus and its `COVERAGE.json`/`MANIFEST.json`, Unicode and
shaping reference corpora, recipe fixtures, golden PNGs, certificate JSON and
public key, determinism hashes, packaged fonts/data, tracked PDB, GeoTIFF, and
binary arrays were classified as generated, fixture, corpus, package-data,
golden, certificate, or provenance surfaces. They are not ordinary cleanup.

## Commands and searches

All commands were non-mutating inspection commands:

- `git status --short --branch`, `git rev-parse HEAD`, and `git merge-base HEAD
  f5db54f95d202681f95dad649162d18efdae8987` established identity and worktree
  provenance.
- `git -C /Users/mpopovic3/forge3d show
  codex/refactor-runbook-refresh-20260811:docs/refactor-forge3d-sol-ultra-runbook.md`
  read the controlling prompt.
- `sed`/`nl` inspected root `AGENTS.md`, `.claude/rules/build-and-ci.md`,
  `.claude/rules/rust-core.md`, `Cargo.toml`, `pyproject.toml`, `pytest.ini`,
  `conftest.py`, and relevant CI workflows.
- `git ls-files` plus the exact predicate above produced the owned manifest,
  per-surface counts, extension inventory, small/large-file inventory, special
  artifact classification, and manifest digest.
- `find`, `wc`, and `rg` enumerated all PyO3 module/function/type and Scene API
  files and searched registrations, PyO3 signatures, `cfg`/features, exports,
  imports, dynamic loading, `getattr`, `__all__`, stubs, canonicalization,
  bundles, recipes, fixtures, goldens, certificates, TODO/FIXME/legacy/fallback
  diagnostics, and candidate private helpers.
- `sed`/`nl` inspected `src/lib.rs`, every central `src/py_module/**`
  registration file, PyO3 module topology, the Python native loader and root
  export surface, relevant stubs, SUTURA render/validation/label/bundle/recipe
  modules, AETHER wrapper/registration/tests, and exact `MaterialSet` and
  `SunPosition` Rust APIs.
- `git log --all --oneline -S...` checked the history of N03 and the previously
  proposed private-helper deletions.
- `rustc --version --verbose`, `cargo --version`, and the installed stable
  standard-library documentation established the local compiler/API state.
- `cargo tree -i once_cell --edges normal --locked` established direct and
  transitive normal dependency ownership without building the repository.
- Read-only `python3 -B` AST scripts with `PYTHONDONTWRITEBYTECODE=1` compared
  root runtime exports with stub definitions and import aliases without writing
  bytecode.

No test, compilation, maturin, golden-update, certificate-refresh, formatter,
or Clippy command was run because W1-B explicitly prohibited mutating tests and
build outputs.

## Contract-family evidence map

| Contract family | Evidence inspected | Preserved behavior and conclusion | Remaining proof boundary |
|---|---|---|---|
| Rust-PyO3-stub parity | `src/lib.rs`; `src/py_module/**`; `src/py_functions/**`; `src/py_types/**`; `src/scene/py_api/**`; `python/forge3d/_native.py`; root `__init__.py`/`.pyi`; `pyproject.toml`; API, signature, and installed-wheel tests | The crate root gates binding modules on `extension-module`; `_forge3d` registers typed exceptions and centralized function/class registrars. The Python loader caches the native module and exact import failure. Runtime exports use `_NATIVE_ONLY_EXPORTS` plus `hasattr`. Maturin maps `forge3d._forge3d`, ABI3 Python 3.10+, `release-lto`, and the explicit wheel features. W1B-01 is the one accepted parity gap. | Exact-base installed-wheel runtime/signature parity was not executed. |
| CENSOR | Central diagnostics, degradation, allocation, provenance and certificate registrations; Python certificate/diagnostic modules and tests; fast-lane membership | Capability/provenance truth, explicit degradation, allocation enforcement, canonical certificate/tamper behavior, render capture, and shader reporting remain independently represented. No cleanup claim was accepted. | GPU-backed and production-signing evidence was not run. |
| SUTURA | `map_scene.py`, `_map_scene_render.py`, `_map_scene_validation.py`, `_map_scene_labels.py`, `bundle.py`, `recipe_manifest.py`, `_sutura_recipes.py`, MapScene/bundle/recipe tests and examples | Tests lock removal of placeholder rendering, native-or-structured-diagnostic behavior, compiled-plan-only label culling, frozen cull decisions, stale-plan recompilation, canonical manifest bytes, v2-to-v3 compatibility, v3 compiled-plan persistence, and save/load re-render behavior. | `tests/test_mapscene_sutura_integrity.py` is in `tests/UNRUN.toml` through 2026-10-10 for its hardware requirement; live SSIM/pixel proof is `NOT_PROVEN`. |
| LIMES | `src/py_functions/vector/coverage*.rs`, vector registration, Python vector wrapper, `tests/test_vector_coverage.py` | Opt-in coverage/report fields and the canonical-content-keyed, single-entry compiled-scene cache remain local to the coverage subsystem. No genericization was accepted. | Runtime GPU coverage was not executed. |
| VT, visibility, terrain, TESSELLA | Native diagnostic/stat exports, terrain/VT/visibility tests, TESSELLA evidence tests, CI lane selection | Python-visible diagnostic seams and fail-closed evidence routing are mapped. Code/test existence was not treated as physical acceptance. | Exact-SHA adapter-bound NVIDIA/Vulkan physical acceptance is `NOT_PROVEN`. |
| SUBSTRATIA | Terrain golden variants, certificate paths, evidence-report tests, NVIDIA/Vulkan CI lanes | Golden selection, adapter identity, exact SHA, zero-skip verification, and fail-closed physical routing remain distinct from portable proof. | No physical NVIDIA/Vulkan run occurred; acceptance is `NOT_PROVEN`. |
| SIDERA | Astro registrar, Python astro exports/stubs, astronomy tests and deterministic-night artifacts/workflow | The public astronomy surface and 2000-2050 numerical-window evidence remain mapped, with backend-specific hashes classified as evidence rather than cleanup. | Backend-specific deterministic night evidence was not regenerated or run. |
| AETHER | Atmosphere registrar and types, root exports/stubs, `atmosphere-bake` wheel feature, atmosphere tests, `src/py_functions/path_tracing/aether_reference.rs`, independent Rust reference driver and shader | The Python wrapper imports only `HybridPathTracer` and `AetherSpectralReferenceDesc`; registration remains in `py_module/functions/rendering.rs`; the independent implementation remains under `src/path_tracing/hybrid_compute/aether_reference.rs`. Tests explicitly reject LUT or CPU injection into the stochastic oracle. N03 must not alter retryable initialization or merge the oracle into production. | Physical Metal closure and exact installed-wheel behavior were not run. |
| Determinism and certificates | `_canonical_json.py`, certificate modules/tests, certificate fixtures/public key, determinism hashes and workflow references | Canonical byte inputs, backend-specific hashes, signing/tamper semantics, and protected refresh boundaries remain separate. | No golden, hash, certificate, or signing refresh occurred. |
| Text and Unicode | Packaged font inventory and provenance, atlas data, Unicode/shaping corpora, relevant tests and packaging declarations | Generated/reference ownership, Unicode provenance, package data, shaping, and atlas contracts were classified. Size and age supplied no deletion evidence. | Corpora were not replayed behaviorally. |
| GIS and units | Python GIS wrapper and stub, PyO3 GIS registrations, GIS parity/behavior tests, torture descriptors and docs | CRS/raster/vector/unit behavior and native wrapper ownership remain mapped. `derive_water_mask` has direct consumers and tests, but its intended export/stub status is unresolved and therefore rejected as a change claim. | Public owner intent for `derive_water_mask` is `NOT_PROVEN`; GIS runtime suites were not run. |
| Examples, CLI, recipes, and bundles | All 46 tracked examples, import shims/dynamic-import tests, argparse/main seams, MapScene examples, recipe manifests, bundle tests, golden and certificate update guards | Examples remain real runnable paths; cross-example imports and dynamic import tests were accounted for. MapScene examples exercise bundle creation and canonical recipe manifests. Golden/certificate writes require explicit environment gates. Remembered Iberia example files are absent at this SHA, so no claim relies on them. | External downloads, optional dependencies, viewers, renders, and golden comparisons were not run. |
| Tests and CI as behavioral evidence | 988 tracked test paths, `pytest.ini`, root and test `conftest.py`, `tests/UNRUN.toml`, `scripts/ci_pytest_lane.py`, wheel workflow, CI matrix and physical lanes | Fast/full/slow/compat/physical ownership and installed-wheel path checks were mapped. Tests were evidence of explicit behavior, not automatic acceptance. | No test lane was executed during this read-only audit. |
| Fixtures, goldens, certificates, corpora, package data | All matching tracked owned paths and their manifests/consumers | These surfaces were classified and preserved; no ordinary cleanup was proposed. | Per-object replay was outside the read-only audit and remains `NOT_PROVEN`. |

## Finding register

| ID | priority basis | subsystem | path:symbol | claim | evidence | behavior/contract | smallest transformation | risk/dependencies | required proof | status | commit/PR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| W1B-01 | Public Rust-PyO3-stub parity is an explicit contract and precedes wrapper cleanup | Python typing / PyO3 | `python/forge3d/__init__.pyi`; `python/forge3d/__init__.py:_NATIVE_ONLY_EXPORTS,__all__`; `src/render/material_set/{core,py_api}.rs:MaterialSet`; `src/lighting/py_bindings/sun_position.rs:PySunPosition,sun_position,sun_position_utc` | `MaterialSet`, `SunPosition`, `sun_position`, and `sun_position_utc` are registered, runtime-exported, and listed in root `__all__`, but absent from the root stub | AST comparison found 97 `_NATIVE_ONLY_EXPORTS` and accounted for stub classes, functions, annotations, and import aliases. Eighteen native-accessible names lacked stub provision, but only these four are also explicitly in root `__all__`. API tests name all four; `MaterialSet` is already used as an unresolved forward annotation in `__init__.pyi`; exact Rust methods and signatures were inspected. | Preserve native registration and runtime identity, exact signatures/defaults/return types, class methods/properties, and existing `__all__`; improve only the installed static typing surface. | Add exact root-stub class declarations for `MaterialSet` and `SunPosition`, exact declarations for `sun_position` and `sun_position_utc`, and a contract test covering these public names and runtime signatures. Do not type the other fourteen merely accessible names without owner evidence. | PyO3 text-signature and Python type mapping, ABI3 installed-wheel behavior, and exact class factory/property definitions; no runtime implementation change. | Static AST/public-export parity test; installed-wheel import plus `inspect.signature` and property checks; configured stub/type parsing; focused `test_api_contracts` and `test_install_smoke`; fast lane at candidate head. | `VALIDATED` | `NOT_PROVEN` |

## N03 decision

N03 is **rejected for this refactor run**.

All ten direct-use Rust files and the Cargo declaration were confirmed. The
owned seam at `src/py_functions/path_tracing/aether_reference.rs:66` uses
`once_cell::sync::OnceCell<HybridPathTracer>` and at line 119 calls
`get_or_try_init(HybridPathTracer::new)`. This preserves retry after failed GPU
initialization while keeping the immutable, process-lifetime AETHER reference
pipeline graph associated with the global GPU device.

The local authoritative stable toolchain is `rustc 1.90.0`; its installed
standard-library documentation marks
`std::sync::OnceLock::get_or_try_init` as nightly-only under
`once_cell_try`. CI selects unpinned stable, and `Cargo.toml` declares no
`rust-version` or other authoritative MSRV. Replacing this call directly would
not compile on stable. Replacing it with a custom error-caching or locking
scheme would no longer be the proposed exact per-use standard-library
substitution and could change fallibility, retry, initialization, poisoning,
or concurrency behavior.

`cargo tree -i once_cell --edges normal --locked` also shows `once_cell`
remains transitively required through PyO3, wgpu, wgpu-core/wgpu-hal, and winit;
removing the direct dependency would not establish lockfile removal. The
smallest behavior-preserving N03 transformation is therefore not proven.

This decision does not weaken AETHER independence: the Python acceptance seam
only calls the independent reference tracer; registration is separate from the
production atmosphere functions; the independent driver and shader remain
outside the production LUT path; and tests explicitly prohibit substituting
the production LUT or CPU reference for the stochastic oracle.

## Rejected claims

- **Previously proposed private-helper deletion:** `_smith_g1`,
  `_ggx_distribution`, `_render_terrain_renderer_rgba`,
  `_native_building_mesh_for_layers`, `_lonlat_to_pixel`, and
  `_smoke_light_transmittance` have definition-only raw text-search results,
  but absence of tracked textual callers plus history is not proof against
  downstream or dynamic use; the same claim was already visible and rejected,
  so the authoritative fuse prevents resurrecting it.
- **`derive_water_mask` export/stub drift:** the function is directly imported,
  tested, used by MapScene, and described as public in older documentation, but
  the exact current `gis.__all__` and its parity test intentionally omit it;
  owner intent for changing that public surface is `NOT_PROVEN`.
- **Fourteen other native-accessible root names missing from the root stub:**
  their presence in `_NATIVE_ONLY_EXPORTS` makes them conditionally accessible,
  but they are absent from root `__all__`; accessibility alone does not prove
  that they belong to the typed root public contract.
- **Example `_render_with_optional_placeholder` names:** the private names are
  misleading in isolation, but their implementations explicitly catch and
  report structured SUTURA native-unavailability and do not synthesize
  placeholder pixels; a rename is not contract-necessary.
- **Long files:** length identifies navigation hotspots but does not prove mixed
  responsibility, duplication, or a safe extraction boundary.
- **1,482 TODO/FIXME/HACK/legacy/placeholder/stub/fallback text hits:** diagnostic
  volume is not semantic evidence and supplied no independently necessary
  behavior-preserving claim.
- **Tracked PDB, corpora, binary arrays, package fonts/data, fixtures, goldens,
  certificates, public keys, and backend hashes:** these are package,
  provenance, fixture, corpus, or acceptance artifacts, not ordinary cleanup.
- **Duplicate-looking CLI option groups:** their option ordering, help output,
  parser ownership, defaults, and parsed namespace compatibility were not
  proven identical across all consumers; no consolidation was accepted.
- **Crate-root wildcard/import concentration:** PyO3 macro resolution,
  registration, and ownership are behaviorally coupled; no safe reduction was
  proven.
- **N03:** stable standard-library API support and exact fallible-initialization
  semantics are insufficient, so the direct dependency replacement is not an
  accepted candidate.

## Remaining uncertainty and `NOT_PROVEN`

- The exact-base installed wheel was not built or imported. Source, packaging,
  workflow, and test evidence map the seam, but installed extension identity,
  runtime signatures, and ABI3 parity remain `NOT_PROVEN`.
- Full, slow, compatibility, GPU, viewer, golden, certificate, signing, Metal,
  NVIDIA/Vulkan, and other physical lanes were not run.
- SUTURA live pixel/SSIM reproduction remains `NOT_PROVEN` on this host; the
  hardware-bound integrity test is explicitly quarantined in `UNRUN.toml`.
- AETHER physical Metal closure, SUBSTRATIA/TESSELLA NVIDIA/Vulkan acceptance,
  backend-specific SIDERA determinism, and production certificate evidence are
  `NOT_PROVEN`.
- Binary corpora, fixtures, goldens, and certificates were classified through
  manifests, provenance, code consumers, tests, and CI routing; no per-object
  behavioral replay was performed.
- W1B-01 requires exact-head installed-wheel and focused contract proof after
  implementation before it can advance beyond `VALIDATED`.
- Public owner intent for `derive_water_mask` remains `NOT_PROVEN`.
- No accepted candidate other than W1B-01 was established by W1-B.
