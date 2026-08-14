# W1-V independent coverage and claim-freeze review

## Review identity and boundary

- Exact base: `f5db54f95d202681f95dad649162d18efdae8987`.
- Worktree reviewed: `/private/tmp/forge3d-refactor-20260812`.
- Role: independent adversarial coverage and claim-freeze reviewer.
- Review inputs: root `AGENTS.md`, the controlling runbook at
  `codex/refactor-runbook-refresh-20260811:docs/refactor-forge3d-sol-ultra-runbook.md`,
  the current ledger, and the read-only W1-A/B/C coverage outputs.
- Provenance: W1-V performed read-only repository inspection and one compiler
  API probe. It made no source, manifest, test, fixture, documentation, ledger,
  generated-file, or asset edit; ran no repository tests or builds; used no
  network or autoreview; and made no commit or push. This file is the only
  write subsequently requested from W1-V.

## Manifest reconciliation

| Manifest | Paths | SHA256 |
|---|---:|---|
| A | 1,252 | `3abf0c315b4bb7416935a54253ad9920ed588b08307be8439e215ab1f4697a33` |
| B | 1,272 | `51efb1a88ad236ce8a0214d189408ccb041d83af061d83f2e9fcb86b6245bcdf` |
| C | 269 | `e685bff1ab3790153131bc7dc3c57baa9da3fe955378de2ed67552595b457781` |
| All tracked | 2,793 | not separately hashed |

Independent predicate reconstruction proves:

- `A union B union C` equals the complete `git ls-files` stream.
- `A intersect B`, `A intersect C`, and `B intersect C` are empty.
- `1,252 + 1,272 + 269 = 2,793`.
- `/private/tmp/w1c-owned-manifest.txt` exactly matched the derived C stream.

### Deterministic serialization and predicates

The digests were computed from newline-delimited `git ls-files` output in Git
index order. No additional `sort` was applied and no locale was explicitly set.
Each `awk` `print` emitted one path followed by `\n`, including a final trailing
newline. Those bytes were passed directly to `shasum -a 256`. The reviewed path
stream consists of ordinary ASCII/UTF-8 path text.

Manifest A:

```sh
git ls-files | awk '
function is_a(p) {
  return p=="Cargo.toml" ||
         p=="Cargo.lock" ||
         p=="build.rs" ||
         p=="CMakeLists.txt" ||
         p ~ /^cmake\// ||
         p ~ /^\.cargo\// ||
         (p ~ /^src\// &&
          p!="src/lib.rs" &&
          p !~ /^src\/(py_functions|py_module|py_types)\// &&
          p !~ /^src\/scene\/py_api\//)
}
is_a($0) { print }
' | shasum -a 256
```

Manifest B:

```sh
git ls-files | awk '
function is_b(p) {
  return p=="src/lib.rs" ||
         p=="conftest.py" ||
         p=="pytest.ini" ||
         p ~ /^src\/(py_functions|py_module|py_types)\// ||
         p ~ /^src\/scene\/py_api\// ||
         p ~ /^(python|tests|examples|bench|benches|data)\//
}
is_b($0) { print }
' | shasum -a 256
```

Manifest C is exactly the complement of A and B:

```sh
git ls-files | awk '
function is_a(p) {
  return p=="Cargo.toml" ||
         p=="Cargo.lock" ||
         p=="build.rs" ||
         p=="CMakeLists.txt" ||
         p ~ /^cmake\// ||
         p ~ /^\.cargo\// ||
         (p ~ /^src\// &&
          p!="src/lib.rs" &&
          p !~ /^src\/(py_functions|py_module|py_types)\// &&
          p !~ /^src\/scene\/py_api\//)
}
function is_b(p) {
  return p=="src/lib.rs" ||
         p=="conftest.py" ||
         p=="pytest.ini" ||
         p ~ /^src\/(py_functions|py_module|py_types)\// ||
         p ~ /^src\/scene\/py_api\// ||
         p ~ /^(python|tests|examples|bench|benches|data)\//
}
!is_a($0) && !is_b($0) { print }
' | shasum -a 256
```

The identical streams piped to `wc -l` produced the recorded counts.

## Approved claims and exact conditions

### N01 - build revision helper reuse

Approved only for replacing the duplicated full-SHA local Git
command/parsing ladder in `build.rs` with the existing revision-helper
mechanism. Preserve validated 40-hex `GITHUB_SHA` precedence, short-SHA
behavior, whitespace trimming, Git-failure fallback, and the exact `unknown`
result. Characterize environment present, invalid, and absent cases plus Git
success and failure before editing.

### N02 - duplicate Cargo dev dependencies

Approved only for removing the duplicate `[dev-dependencies]` declarations
`env_logger = "0.10"` and `sha2 = "0.10"`. Retain both normal dependencies and
the `[build-dependencies]` `sha2` declaration. Require locked metadata and
dependency-tree equivalence, an unchanged `Cargo.lock`, and affected locked
build/test proof.

### A01 - five dead CMake local assignments

Approved only for removing the five assignments proven unread: the three
platform branches assigning `RUST_LIB_EXT` and the two build-type branches
assigning `CARGO_BUILD_TYPE`. Do not remove externally observable CMake
variables or CMake integration. Prove configure output/cache and existing
targets are unchanged.

### A02 - exact `recompute_normals` duplication

Approved for consolidating the byte-identical bodies in
`src/geometry/displacement.rs` and `src/geometry/subdivision.rs` into one
geometry-private helper. Preserve iteration, accumulation, division,
normalization, arithmetic order, and zero-length fallback exactly. Characterize
both displacement and subdivision outputs before moving the implementation.

### A03 - exact overlay `point_on_segment` duplication

Approved for consolidating the byte-identical exact-predicate implementations
in `src/geometry/overlay/sweep.rs` and `src/geometry/overlay/faces.rs` into one
overlay-private helper. Preserve exact orientation and inclusive bound
comparisons. Run the full overlay test corpus.

### A04 - exact GIS `synthetic_info` triplication

Approved for consolidating the byte-identical implementations in
`src/gis/domain.rs`, `src/gis/rasterize.rs`, and `src/gis/thematic.rs` into one
extension-module-private GIS helper. Preserve driver, dimensions, band count,
dtype strings, nodata values, and every remaining `RasterInfo::new` default.
Exercise array-only input through all three public surfaces.

### A05 - exact WKT token and structure duplication

Approved only for consolidating common WKT token recognition and bracket
structure validation shared by `src/gis/raster_tags.rs` and
`src/gis/raster_write.rs`. Do not merge the writer's stricter WGS84 and IAU
semantic validation into metadata reading. Prove all four accepted token
families, case and leading-whitespace handling, balanced and unbalanced
brackets, planetary WKT, and the intentionally different reader/writer
semantics.

### W1B-01 - root stub parity

Approved only as a root-stub parity defect, not as a runtime `__all__` defect.
`python/forge3d/__init__.py` already includes `MaterialSet`, `SunPosition`,
`sun_position`, and `sun_position_utc` in runtime `__all__`. Add accurate class
and function declarations for those four symbols to
`python/forge3d/__init__.pyi`; do not edit runtime `__all__`. Prove static
imports and signatures plus the existing runtime API contracts.

### W1C-01 - stale example catalog and Fuji contradiction

Approved for reconciling `docs/examples/index.md`,
`docs/guides/data_and_scene_workflows.md`, and `docs/guides/feature_map.md` with
the tracked examples. The catalog claims exhaustive coverage while naming
nonexistent scripts and omitting tracked runnable examples. Correct Fuji
specifically: `examples/fuji_labels_demo.py` uses public
`MapScene`/`LabelLayer`, not `viewer_ipc` or `ViewerHandle`.

### W1C-02 - absent sdist manifest input

Approved for removing `include rust-toolchain.toml` from `MANIFEST.in` because
neither `rust-toolchain.toml` nor `rust-toolchain` is tracked. Prove that the
sdist builds and inspect its member list.

## Rejected claims

- `N03`: rejected. Rust 1.90 reports
  `std::sync::OnceLock::get_or_try_init` as unstable, while four live call paths
  require fallible initialization. `viewer::viewer_config` also exposes
  concrete public `once_cell::sync::OnceCell` statics, so whole-dependency
  replacement would change public Rust types.
- `C01`: descriptor/layout extraction lacks necessity beyond repetition and
  carries binding-layout risk.
- `C02`: a shared example-CLI abstraction is not required by the contract and
  expands ownership.
- `C03`: definition-only searches do not prove private Python helpers dead
  under dynamic access or downstream use.
- `C04`: shader textual repetition does not prove safe consolidation with
  preserved evaluation order and physical output.
- `C05`: repeated light-mutation ladders do not establish one
  borrow/error-preserving helper.
- `C06`: LIMES layout-helper consolidation is unnecessary and risks a protected
  shader/binding boundary.
- `C07`: broad native-idiom substitutions are style churn, not necessary
  behavior-preserving work.
- W1-B's fourteen non-`__all__` names: rejected because they are not
  package-public exports, so the root-stub parity contract does not apply.
- Dead-helper candidates: rejected absent dynamic, import, registration, and
  downstream reachability proof.
- CMake removal, shader/LIMES/AETHER consolidation, public API or public-type
  removal, CI-policy rewriting, assets, generated sources, history, goldens,
  certificates, fixtures, corpora, binaries, dependency upgrades, and
  correctness/performance redesign: rejected as outside scope or protected
  contract surfaces.
- Existing `R01` through `R13` remain rejected for their recorded contract
  reasons. The freeze should replace "incorrectly routed" as the sole rationale
  for `C01` through `C07` with the evidence-based reasons above.

## Original freeze blockers

These blockers prevented source implementation at W1-V and motivated the W1-F
ledger/classification follow-up:

1. `docs/refactor-forge3d.md` still contained pre-W1 `DISCOVERED` coverage rows
   and did not record the verified manifests, digests, decisions,
   transformations, or proof requirements.
2. The architecture map covered every contract family explicitly named by
   Phase 1 and was sufficient at family level.
3. Nonordinary classification was insufficient: the ledger had category-level
   prose but no exact path-to-classification evidence for all A/B/C paths. The
   three manifests needed to be persisted and every tracked path classified as
   ordinary source, generated, fixture/corpus, asset/binary,
   golden/certificate, historical, or policy/config.
4. Each W1 coverage report, or its path-bound evidence, needed to be persisted
   in the ledger. The supplied summaries alone did not prove every manifest
   entry was inspected.
5. `W1B-01` needed to be frozen with the corrected root-stub-parity wording;
   calling it a missing runtime `__all__` export would be false.

The smallest follow-up was ledger-only reconciliation and an independent
reread, with no source edit before approval.

### Expected exact path-classification evidence

W1-V used only the exact A/B/C ownership predicates above. It did not receive
or derive a complete per-path nonordinary classification map. The expected
deterministic evidence was one TSV row for every `git ls-files` path:

```text
path<TAB>wave<TAB>classification<TAB>evidence
```

- `wave` must be computed exclusively by the predicates above.
- `classification` must distinguish at least `ordinary-source`,
  `build-or-package-config`, `policy-or-CI`, `generated`, `fixture`, `corpus`,
  `asset`, `golden`, `certificate`, `binary`, and `historical`.
- Prefix rules may seed the classification, but their expanded path lists must
  be preserved in the TSV. Ambiguous exceptions require explicit path
  overrides instead of extension-only inference.
- Mandatory known overrides include
  `src/labels/unicode/generated.rs` as `generated`, `docs/superpowers/**` as
  `historical`, and `tests/golden/**` as `golden`; provenance and certificate
  paths require individual classification.
- Ignored output cannot be a member of A/B/C and must be recorded separately
  from `git status --ignored` or `git check-ignore` evidence.
- Proof should hash the TSV, assert exactly one row per tracked path, reject
  duplicate and missing paths, and compare its first column byte-for-byte with
  `git ls-files`.

## Disjoint implementation ownership

- `I1 / N01`: `build.rs`.
- `I2 / N02`: `Cargo.toml`; `Cargo.lock` is inspection-only unless exact proof
  exposes an unexpected change, which blocks rather than authorizes editing.
- `I3 / A01`: `CMakeLists.txt`.
- `I4 / A02`: `src/geometry/mod.rs`,
  `src/geometry/displacement.rs`, and `src/geometry/subdivision.rs`.
- `I5 / A03`: `src/geometry/overlay/mod.rs`,
  `src/geometry/overlay/sweep.rs`, and `src/geometry/overlay/faces.rs`.
- `I6 / A04+A05`: one GIS owner for `src/gis/mod.rs`, `src/gis/crs.rs`,
  `src/gis/domain.rs`, `src/gis/rasterize.rs`, `src/gis/thematic.rs`,
  `src/gis/raster_tags.rs`, and `src/gis/raster_write.rs`.
- `I7 / W1B-01`: `python/forge3d/__init__.pyi` and
  `tests/test_api_contracts.py`.
- `I8 / W1C-01`: `docs/examples/index.md`,
  `docs/guides/data_and_scene_workflows.md`,
  `docs/guides/feature_map.md`, and new `tests/test_example_catalog_docs.py`.
- `I9 / W1C-02`: `MANIFEST.in` and new `tests/test_sdist_manifest.py`.

The listed ownership is disjoint. Test ownership stays with the implementation
whose contract it proves; integration gates are orchestrator-owned and must not
be edited merely to make them pass.

## Exact proof required

Each owner must first run its claim-specific characterization or contract proof
on the pre-edit exact base, make only the approved transformation, and rerun the
same proof on the candidate head. Required claim-specific evidence is stated
under each approved claim above.

After every targeted proof passes, integration requires:

1. `cargo fmt --check` at the exact candidate head.
2. `cargo forge3d-clippy` at the exact candidate head.
3. `python3 -m maturin develop` from the candidate worktree and readback that
   the installed extension belongs to that exact source/head.
4. `FORGE3D_NO_BOOTSTRAP=1 python scripts/ci_pytest_lane.py --profile fast -v --tb=short`.
5. The applicable exact Rust checks, tests, doctests, CMake configure/target
   checks, static-typing checks, documentation-reference checks, and sdist
   inspection identified by I1-I9.
6. An exact-head complete-diff review for public API, type-stub, feature,
   documentation, packaging, lockfile, generated-file, fixture, golden,
   certificate, and user-work drift.
7. The live policy's per-change and whole-diff adversarial reviews, followed by
   ponytail review and Standards/Spec code review before any authorized commit,
   push, or PR.
8. Final ledger reconciliation to the exact final SHA, including claim-to-commit
   mapping, test identities/results/skips, residual risk, remote evidence, PR
   head/checks, and mergeability readback where authorized.

These approved transformations do not change WGSL or intended GPU behavior, so
they do not independently require physical GPU acceptance. Any unavailable
physical evidence must remain explicitly `NOT_PROVEN`; it cannot be reported as
passed or failed by inference.
