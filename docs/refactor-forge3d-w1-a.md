# Forge3D refactor Phase 1 — W1-A Rust/GPU/build inspection

W1-A is complete read-only at exact base `f5db54f95d202681f95dad649162d18efdae8987`. Six behavior-preserving candidates were accepted. N03 and shader consolidation were rejected.

## Owned manifest

Exact derivation:

```sh
git ls-files |
awk 'BEGIN{n=0}
$0=="Cargo.toml" || $0=="Cargo.lock" || $0=="build.rs" ||
$0=="CMakeLists.txt" || $0 ~ /^cmake\// || $0 ~ /^\.cargo\// ||
($0 ~ /^src\// &&
 $0!="src/lib.rs" &&
 $0 !~ /^src\/py_functions\// &&
 $0 !~ /^src\/py_module\// &&
 $0 !~ /^src\/py_types\// &&
 $0 !~ /^src\/scene\/py_api\//) { print; n++ }
END { print "COUNT=" n > "/dev/stderr" }'
```

The exact manifest is the newline-delimited stdout of that command at the audited SHA:

- `src/**` tracked: 1,338
- Excluded: `src/lib.rs` 1; `src/py_functions/**` 37; `src/py_module/**` 20; `src/py_types/**` 10; `src/scene/py_api/**` 25
- Owned `src/**`: 1,245
- Build roots: `Cargo.toml` 1; `Cargo.lock` 1; `build.rs` 1; `CMakeLists.txt` 1; `cmake/**` 2; `.cargo/**` 1
- Exact owned total: **1,252**
- Exact manifest SHA-256: `3abf0c315b4bb7416935a54253ad9920ed588b08307be8439e215ab1f4697a33`

Inventory evidence: 1,090 Rust files, 142 WGSL files, 1,244 text files, 324,728 text lines, 104 files of 20 lines or fewer, and no exact duplicate Git blobs in the owned surface. These metrics were used only for coverage/navigation, not as evidence for findings.

## Commands and searches

- Verified SHA, branch, status, tracked paths, exclusions, manifest digest, extensions, text/binary classification, line counts, small files, largest files, and duplicate Git blobs.
- Read `/Users/mpopovic3/forge3d/AGENTS.md`, the controlling prompt with `git -C /Users/mpopovic3/forge3d show codex/refactor-runbook-refresh-20260811:docs/refactor-forge3d-sol-ultra-runbook.md`, applicable `.claude/rules`, Cargo manifests and lockfile, `build.rs`, CMake files, and CI workflows.
- Searched all owned Rust/WGSL and relevant tests/specs/CI for:
  - modules, re-exports, public APIs, features, and `cfg`
  - traits, implementations, `dyn`, ownership, lifetimes, and `Drop`
  - WGSL includes, generated registration, entry points, bindings, layouts, `Pod`, `repr(C)`, and `size_of`
  - CENSOR, LIMES, VT, visibility, TESSELLA, SIDERA, AETHER, and SUBSTRATIA
  - determinism, certificates, Unicode, GIS, CRS, and units
  - N01 build helper, N03 `once_cell`, Cargo dependency kinds, and CMake consumers
- Used `git log -S` and `git blame` for provenance of build-script and exact-duplicate functions.
- Ran a read-only exact-function-body inventory.
- Ran a temporary Rust compiler probe: on repository Rust `1.90.0`, `std::sync::OnceLock::get_or_try_init` fails with `E0658`, proving the proposed N03 replacement is not generally available. No repository file was created.
- No build, test, formatter, generator, network, commit, or repository write was performed during the inspection.

## Contract-family evidence map

| Contract family | Evidence inspected | Conclusion |
|---|---|---|
| Rust module/API/feature graph | Cargo features, `.cargo/config.toml`, CI feature matrices, 1,046 module declarations, public re-exports, trait and dynamic-dispatch seams including `HeightReader`, `OverlayReader`, and `VirtualTextureStore` | No feature or public API deletion proven. Textual reachability heuristics produced nested-`#[path]` false positives and were not treated as proof. |
| Rust-WGSL assembly | `src/shader_sources.rs`, `build.rs` recursive `include_str!`/`#include` scan, generated `registered_wgsl.rs` inclusion in `src/verify/mod.rs`, assembly validation tests | Assembly order, cfg-specific source sets, group remapping, and entry-point ownership are observable contracts. Shader consolidation rejected. |
| Layouts/bindings/resources | `repr(C)`, `Pod`/`Zeroable`, size/layout assertions, binding and entry-point tests, `TrackedBuffer`/`TrackedTexture`, raw-allocation gates | No layout, binding, or resource-owner merge accepted. |
| CENSOR | `core/capabilities.rs`, `degradation.rs`, `resource_tracker.rs`, `certificate.rs`, `shader_registry.rs`, validation policy, certificate/verifier/contract tests, and CI selectors | Negotiation, degradation, allocation, registry, and certificate behavior are coupled acceptance surfaces; preserved. |
| LIMES | `vector/coverage/{ingest,binning,raster,resolve,render,types}.rs`, coverage shaders, Python cache seam, coverage tests | Local binding helpers remain owned by their passes; cross-pass extraction rejected. |
| VT / visibility / TESSELLA | Terrain VT renderer/store trait, runtime and feedback ownership, family residency/page digests, visibility write/resolve assembly, VT/TESSELLA tests and physical lanes | Ownership and evidence contracts preserved; no structural merge accepted. |
| SIDERA | `src/astro/**`, star shader, night golden and determinism tests | Public time window, backend-bound rendering, and golden behavior preserved. |
| AETHER | Production bake/precomputed/runtime/spectral code and shaders; independent hybrid-compute reference oracle | Oracle independence is a contract; production/oracle consolidation rejected. |
| SUBSTRATIA | Terrain residency/page digests plus SUBSTRATIA tests/scripts and physical workflow | Static contract surface accounted for; exact hardware acceptance remains `NOT_PROVEN`. |
| Determinism/certificates | `core/certificate.rs`, `core/anamnesis/**`, `verify/**`, determinism WGSL include, matrix workflow, and tests | Provenance and deterministic evidence paths preserved. |
| Unicode | Generated Unicode tables, `UCD_VERSION`, `SOURCES.sha256`, shaping/bidi/GPOS/GSUB modules, and tests | Generated-source edits rejected. |
| GIS/units | Sealed typed units, CRS/epoch/height systems, GIS modules, affine/planetary datum/raster tests | Two exact GIS helper consolidations accepted with lexical and diagnostic behavior explicitly fixed. |
| Build/Cargo/CMake | Build provenance envs and consumers, dependency kinds, CMake wrapper/consumer references, CI authority | N01/N02 plus dead CMake locals accepted. CMake remains preservation-sensitive and is not CI-proven. |

## Accepted finding register

| ID | priority basis | subsystem | path:symbol | claim | evidence | behavior/contract | smallest transformation | risk/dependencies | required proof | status | commit/PR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N01 | One build-provenance truth before manifest cleanup | Build provenance | `build.rs:main::git_revision/full_sha` | The duplicated Git command/output/parsing ladder can use one helper accepting a full argument slice; `GITHUB_SHA` precedence remains separate. | Lines 51-76 repeat the ladder; provenance dates to `93622716`; outputs feed `FORGE3D_GIT_SHA` certificate and full-SHA anamnesis paths. | Preserve valid 40-hex `GITHUB_SHA` precedence, 12-character short SHA, local full-SHA fallback, rerun directives, and `unknown` on command/UTF-8/status failure. | Extract `git_revision(&[...])`; call with `["--short=12","HEAD"]` and `["HEAD"]`. Do not pass `"HEAD"` to the current short-SHA helper because that would execute `rev-parse HEAD HEAD`. | Tarballs/no Git, invalid environment SHA, invalid UTF-8, nonzero Git, provenance consumers. | Characterize valid/invalid/missing environment, Git success/failure/no-Git; compare emitted vars; run provenance/certificate tests and whole-diff review. | `VALIDATED` | `NOT_PROVEN` |
| N02 | Manifest dependency-kind truth | Cargo | `Cargo.toml:[dependencies]/[dev-dependencies] env_logger,sha2` | Duplicate dev declarations are redundant because identical normal dependencies are already available to dev targets. | Normal and dev entries are both `env_logger = "0.10"` and `sha2 = "0.10"`; `env_logger` has bench use, `sha2` runtime use, and build-time `sha2` remains separately required; lockfile has one resolution per version. | Preserve normal, dev, bench, and build availability, features, resolution, and lockfile. | Remove only the two duplicate `[dev-dependencies]` lines. | Cargo target/dependency-kind behavior and lock resolution. | Compare `cargo metadata`/dependency trees before and after; locked checks/tests and bench compilation; prove lockfile unchanged. | `VALIDATED` | `NOT_PROVEN` |
| A01 | Dead duplicated build truth | CMake | `CMakeLists.txt:RUST_LIB_EXT,CARGO_BUILD_TYPE` | Five local assignments are never read. | `RUST_LIB_EXT` is assigned on three platform branches and `CARGO_BUILD_TYPE` on two profile branches; exact searches find no `${...}` consumer, cache export, or parent-scope use. | Preserve configure messages, targets, installation, prefix/suffix, cargo flag, and target directory logic. | Delete only those five assignments; retain `RUST_LIB_PREFIX`, `PYTHON_EXT_SUFFIX`, `CARGO_BUILD_FLAG`, and `RUST_TARGET_DIR`. | Platform/configuration branches. | CMake configure and target-graph proof on applicable platforms, or authoritative static trace plus local configure; compare messages and targets. | `VALIDATED` | `NOT_PROVEN` |
| A02 | One exact geometry algorithm truth | Geometry | `src/geometry/{displacement.rs,subdivision.rs}:recompute_normals` | Two private functions have the same 38-line body and should share one geometry-private implementation. | Exact normalized bodies match; both entered in `7b08af80`; four displacement and one subdivision call sites were traced through public APIs and tests. | Preserve accumulation order, counts, degenerate fallback, floating-point ordering, and public displacement/subdivision output. | Move the unchanged body to one geometry-owned `pub(super)` helper and replace only the five calls. | Float ordering; extension-only displacement versus unconditional subdivision compilation. | Pre/post characterization for triangle, degenerate, and missing-normal inputs; focused Rust/Python geometry tests; format and lint. | `VALIDATED` | `NOT_PROVEN` |
| A03 | One exact topology predicate truth | Exact overlay | `src/geometry/overlay/{faces.rs,sweep.rs}:point_on_segment` | Identical private exact predicates should share one overlay-private implementation. | Bodies match exactly and share origin commit `8e96602e`; one face and two sweep call sites use identical orientation, sign-ordering, and inclusive bounds logic. | Preserve boundary classification, inclusivity, sweep intersection behavior, and floating-point ordering. | Place the unchanged helper in the overlay module or existing shared overlay file with `pub(super)` visibility. | EUCLIDEA topology and exact boundary behavior. | `test_boolean_overlay.py`, Rust overlay tests/fuzzing, ordering-source gate, format and lint. | `VALIDATED` | `NOT_PROVEN` |
| A04 | One in-memory raster metadata default truth | GIS | `src/gis/{domain.rs,rasterize.rs,thematic.rs}:synthetic_info` | Three exact private constructors should share one GIS-private implementation. | Exact bodies produce the same `RasterInfo`: memory driver, dimensions, dtype, band count, and nodata defaults; corresponding public GIS tests were traced. | Preserve casts, dtype strings, nodata vector, metadata fields, and Python-facing output. | Add one `#[cfg(feature="extension-module")] pub(crate)` GIS helper and replace the three bodies only. | Extension feature compilation and PyO3 diagnostics. | Domain/rasterize/thematic focused tests, native feature compile, exact dictionary characterization, format and lint. | `VALIDATED` | `NOT_PROVEN` |
| A05 | One CRS lexical/structural truth | GIS CRS I/O | `src/gis/raster_tags.rs:{looks_like_wkt,validate_wkt_literal}` and `src/gis/raster_write.rs:{looks_like_wkt,validate_wkt_structure}` | The token predicate and bracket structural validator are exact duplicates under different local names. | Bodies and error strings match; read and write paths use the same lexical contract. The write path's stricter WGS84/IAU semantic validation wraps this structural layer and is distinct. | Preserve trimming, case handling, accepted tokens, bracket checks, diagnostics, and stricter semantic validators. | Extract only the token predicate and structural validator into one GIS-private owner; leave WGS84/IAU semantic validation separate. | Public CRS behavior, round trips, and error text. | Focused WKT read/write, unbalanced-input, GIS raster, CRS-affine, and metadata tests; format and lint. | `VALIDATED` | `NOT_PROVEN` |

## Rejected claims

- **N03:** Rejected. Ten direct `once_cell` use files exist, including excluded `src/lib.rs` and `src/py_functions/path_tracing/aether_reference.rs`; `src/sdf/hybrid.rs` requires `get_or_try_init`, whose standard-library equivalent is unstable on Rust 1.90.0, and public viewer statics expose the concrete `once_cell` type. Partial conversion would retain the dependency and fragment conventions.
- Terrain/PBR shader-list consolidation rejected: source sets have different cfg gates, prefixes, CSM remapping, group ownership, and exact assembly order.
- Cross-subsystem WGPU descriptor-helper extraction rejected: matching syntax does not prove interchangeable bind-group ownership, stage visibility, or layout contracts.
- LIMES storage/uniform helper consolidation rejected: each pass owns distinct shader bindings; no behavior gap requires an abstraction.
- AETHER production/reference consolidation rejected: oracle independence is explicitly contractual.
- Generated Unicode, `Cargo.lock`, assets, goldens, and certificate edits rejected.
- Parallel/public AABB type consolidation rejected: API identity, layout, and subsystem ownership differ.
- `u32`/`u64` allocator helper merging rejected: a generic abstraction adds flexibility without a contract need.
- Cargo feature deletion rejected: every current feature has cfg, CI, packaging, or downstream evidence; absence counts alone are not proof.
- CMake version/path "cleanup" rejected: Cargo `1.34.0`, CMake project `1.9.0`, and ForgeConfig `1.2.0` mismatch is potentially observable corrective work outside this behavior-preserving phase.
- GIS `bounds_for_features` consolidation rejected: matching outer bodies feed different visitor implementations, so semantic equivalence is unproven.
- Apparent orphan modules such as `path_tracing/importance.rs`, `lighting/area_lights.rs`, `core/ground_plane/presets.rs`, and vector point-split files rejected: textual caller heuristics are insufficient and nested `#[path]` modules generated false positives.
- Long files, small files, TODOs, legacy/fallback labels, age, diagnostic counts, and search absence were rejected as standalone evidence.
- No exact duplicate blob deletion exists.

## Remaining uncertainty / `NOT_PROVEN`

- No compilation, tests, render, generator, or GPU execution was authorized; runtime equivalence of accepted transformations remains `NOT_PROVEN`.
- NVIDIA/Vulkan and Metal physical acceptance, plus SUBSTRATIA/TESSELLA, SIDERA night, and AETHER evidence lanes remain `NOT_PROVEN`.
- CMake cross-platform validity is `NOT_PROVEN` and no authoritative CI CMake lane was found; preservation is required.
- Compiler-resolved reachability of apparent orphan modules remains `NOT_PROVEN`.
- N01 no-Git/environment characterization and N02 before/after Cargo metadata/lock proof remain `NOT_PROVEN`.
- Generated WGSL registration was inspected but not regenerated.
- Full CENSOR acceptance and exact downstream compatibility remain `NOT_PROVEN`.
- No commits or PRs were created.

## Provenance

The W1-A inspection was read-only: it made no repository edits, ran no build or test that could mutate tracked files, created no commit or PR, used no network access, and delegated no work. This report file is the sole subsequent authorized repository edit.
