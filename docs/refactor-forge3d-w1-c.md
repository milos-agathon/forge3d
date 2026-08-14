# W1-C — docs, CI, packaging, policy, scripts, tools, specs, assets, and shaders

## Outcome and provenance

W1-C completed a read-only Phase 1 inspection at exact base and checkout head
`f5db54f95d202681f95dad649162d18efdae8987` in
`/private/tmp/forge3d-refactor-20260812`.

The work read repository policy, the controlling prompt, manifests, workflows,
documentation, scripts, tools, specs, assets, shader contracts, and their
current source/test consumers. It made no repository edit while inspecting,
ran no test or build, used no network, created no commit, and delegated no
work. This report is the only W1-C repository write. Before this report was
created, the worktree had only these unrelated, pre-existing untracked files:
`docs/refactor-forge3d.md`, `docs/refactor-forge3d-w1-a.md`, and
`docs/refactor-forge3d-w1-b.md`.

## Exact owned manifest

The owned manifest is the LF-terminated, bytewise `LC_ALL=C sort -u` union of:

```sh
git ls-files \
  '.claude/**' '.github/**' 'docs/**' 'scripts/**' 'tools/**' \
  'specs/**' 'assets/**' 'shaders/**' pyproject.toml MANIFEST.in

git ls-files \
  | awk 'index($0,"/")==0' \
  | rg -v '^(Cargo\.toml|Cargo\.lock|build\.rs|CMakeLists\.txt|conftest\.py|pytest\.ini)$'
```

The first predicate contributes 259 paths. The second contributes the 10
remaining tracked root policy, documentation, and licence files not owned by
W1-A/B: `.gitattributes`, `.gitignore`, `.pre-commit-config.yaml`, `AGENTS.md`,
`CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE`, `LICENSE-APACHE`, `README.md`, and
`SECURITY.md`. Their sorted union contains exactly **269 paths**.

The exact manifest distribution is:

| Surface | Paths |
| --- | ---: |
| `.claude` | 2 |
| `.github` | 11 |
| tracked root, including `pyproject.toml` and `MANIFEST.in` | 12 |
| `assets` | 50 |
| `docs` | 139 |
| `scripts` | 37 |
| `shaders` | 13 |
| `specs` | 1 |
| `tools` | 4 |
| **Total** | **269** |

SHA-256 of that exact sorted LF-terminated manifest is
`e685bff1ab3790153131bc7dc3c57baa9da3fe955378de2ed67552595b457781`.

The six excluded tracked root files are seam inputs owned by W1-A/B:
`Cargo.toml`, `Cargo.lock`, `build.rs`, `CMakeLists.txt`, `conftest.py`, and
`pytest.ini`. W1-C read them where they control packaging, CI, documentation,
or N02, but claims no semantic ownership of them.

## Commands and searches

The inspection used only non-mutating commands:

- Read `/Users/mpopovic3/forge3d/AGENTS.md`,
  `.claude/rules/build-and-ci.md`, `.claude/rules/rust-core.md`, and the
  controlling prompt with
  `git -C /Users/mpopovic3/forge3d show codex/refactor-runbook-refresh-20260811:docs/refactor-forge3d-sol-ultra-runbook.md`.
- Established identity and source state with `git status --short --branch`,
  `git rev-parse HEAD`, and
  `git merge-base HEAD f5db54f95d202681f95dad649162d18efdae8987`.
- Derived the owned surface and top-level distributions with `git ls-files`,
  `awk`, `rg -v`, `LC_ALL=C sort -u`, `wc -l`, `uniq -c`, and `shasum -a 256`.
- Read the root policies/configuration and the packaging seams in
  `pyproject.toml`, `MANIFEST.in`, `Cargo.toml`, `Cargo.lock`, `build.rs`,
  `CMakeLists.txt`, `conftest.py`, and `pytest.ini`.
- Read every `.github/workflows/*.yml`; indexed triggers, path filters,
  conditions, job dependencies, runner classes, exact Cargo/maturin/pytest/docs
  commands, artifacts, zero-skip gates, exact-head checks, and PR/acceptance
  summaries with `rg`, `sed`, and `nl`.
- Searched all owned surfaces and their current source/test/example consumers
  for CENSOR, SUTURA, LIMES, TESSELLA, SUBSTRATIA, SIDERA, AETHER, ANAMNESIS,
  LITTERA, Rust/PyO3/stub parity, WGSL/shader contracts, determinism,
  certificates, packaging, CMake, and N02 implications.
- Read the live CENSOR policy, docs root/configuration, examples catalog,
  feature/support/workflow guides, applicable specs and plans, asset manifests
  and provenance, and relevant evidence scripts and tools.
- Classified owned files by extension and size; used `git check-attr` for LFS
  attributes and `git check-ignore -v --no-index` for tracked-but-ignored
  historical/policy/generated surfaces.
- Computed owned-file SHA-256 values to identify exact duplicates, then treated
  those results only as navigation evidence. Parsed each owned Python file with
  `ast.parse` and inventoried functions/classes/main entry points without
  executing it.
- Searched TODO/FIXME/XXX/HACK/deprecated/obsolete/stale/legacy/temporary/
  placeholder/fallback markers and missing textual callers, while explicitly
  refusing to treat those signals, file size, age, or search absence as proof.
- Compared documentation example references to `git ls-files examples`, read
  `examples/fuji_labels_demo.py`, and traced it to
  `tests/test_mapscene_examples.py` and related documentation contract tests.
- Checked every `MANIFEST.in` root referent for tracked/existing state and
  searched current tests/docs/workflows for sdist, wheel, package-data, and
  manifest assertions.

No `pytest`, Cargo command, maturin command, Sphinx build, package build,
renderer, GPU lane, workflow dispatch, or live GitHub query was run.

## Contract-family evidence map

| Contract family | Current authority/evidence mapped | Boundary retained |
| --- | --- | --- |
| CENSOR | `docs/censor-validation-policy.md` maps to `.github/workflows/ci.yml`'s PR core, affected-integration, acceptance, and release split; current test owners include `tests/test_no_silent_degradation.py`, allocation/certificate tests, and workflow-policy tests. | Policy/artifact presence is not execution acceptance. PR Core is not Full Acceptance or physical proof. |
| SUTURA | Live `AGENTS.md`, `docs/guides/offline_3d_map_rendering.md`, and `docs/guides/data_and_scene_workflows.md` map to `python/forge3d/map_scene.py`, `tests/test_mapscene_sutura_integrity.py`, and `tests/test_mapscene_examples.py`. | Preserve compiled-plan ownership, canonical bundle/serialization, validation, label-culling ownership, structured diagnostic blocks, and zero-placeholder behavior. |
| Rust–PyO3–stub parity | `MANIFEST.in` includes Python and `.pyi` sources; `pyproject.toml` defines the extension and wheel features. These map to `src/py_module/**`, `python/forge3d/*.pyi`, `tests/test_api_contracts.py`, and feature-honesty checks in `tests/test_no_silent_degradation.py`. | File presence does not prove registration, installed-extension behavior, feature parity, or downstream compatibility. |
| Rust–WGSL/GPU | Thirteen `shaders/contracts/*.toml` files are embedded or loaded by `src/verify/mod.rs`, `src/verify/contract.rs`, runtime contract/render owners, and associated tests. `build.rs` writes the WGSL registry to `OUT_DIR`; it is not a competing tracked source. | Preserve layouts, stage limits, resource lifetimes, renderer-owned assembly, and entry points. Contract files are live, not cleanup candidates. |
| LIMES | CI feature/TESSELLA surfaces map to `src/vector/coverage/**`, `src/py_functions/vector/coverage*.rs`, Python vector exposure, and `tests/test_vector_coverage.py`. | Preserve opt-in/default behavior and compiled-scene cache seams; no refactor was justified from docs/config evidence alone. |
| Virtual textures, visibility, terrain, TESSELLA | `.github/workflows/ci.yml` has an explicit path selector and exact-head NVIDIA/Vulkan lane covering VT, HZB, visibility, source differentials, six named pytest families, zero-skip checking, certificate/report scripts, and an acceptance artifact. | Workflow existence is not physical success. Current exact-SHA NVIDIA/Vulkan acceptance is `NOT_PROVEN`. |
| Astronomy and atmosphere | `assets/astro/MANIFEST.toml`, `THIRD_PARTY_NOTICES.md`, embedded astronomy binaries, their source owners, and astronomy tests preserve SIDERA's declared 2000–2050 window. AETHER scripts, workflow routing, tests, and source owners are distinct; its production path and independent oracle remain separate. | Asset or test existence is not current numerical/backend acceptance. The AETHER oracle must not be deduplicated into production. |
| SUBSTRATIA | The NVIDIA acceptance job maps to `scripts/run_nvidia_visual_acceptance.py`, `scripts/substratia_evidence_report.py`, golden/certificate inputs, adapter probes, and evidence-report tests. | Acceptance remains exact-SHA, adapter-bound, golden-bound, and zero-skip; portable, mock, hosted, or Metal evidence cannot substitute. |
| Determinism and certificates | `determinism-matrix.yml`, determinant scripts/tools, committed goldens, certificate workflows, and their tests feed Full Acceptance separately from PR Core. | Hosted diagnostics, cross-adapter divergence, or Metal evidence must not be promoted to NVIDIA or release proof; exact-head results are non-transitive. |
| Text and Unicode | Font provenance/licences, package-data declarations, sdist declarations, label guides, LITTERA source owners, and typography/API tests form the current chain. | Generated/subset fonts and atlas assets are provenance-controlled, not ordinary cleanup; installed package contents still require package proof. |
| GIS, units, examples, recipes | Current GIS docs/specs and example/recipe guides map to GIS source/tests and tracked examples. | Coordinates, units, canonical inputs/outputs, and real runnable paths must remain truthful. W1C-01 captures an exact contradiction in that truth. |
| Assets, generated outputs, fixtures, corpora, goldens, certificates, binaries, ignored/history | LFS attributes, asset manifests, font/astro/geoid provenance, test consumers, package-data rules, CI fixture fan-out, and tracked ignore exceptions establish distinct roles. `docs/superpowers/**` and prompt/audit material are historical where live policy/code/tests/CI supersede them. | Size, identity, age, tracked-ignore status, and historical status are classifications, not deletion or deduplication proof. |

## Accepted finding register

| ID | priority basis | subsystem | path:symbol | claim | evidence | behavior/contract | smallest transformation | risk/dependencies | required proof | status | commit/PR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1C-01 | Documentation truth is necessary to preserve the supported runnable-path contract. | examples/docs | `docs/examples/index.md:1-76`; `docs/guides/data_and_scene_workflows.md:76-118`; `docs/guides/feature_map.md:8-31` | The claimed complete runnable-example catalog and two dependent guides contradict the exact tracked example/source truth. | The catalog says it covers every runnable example, but names numerous absent root examples/support file `_png.py`, omits many tracked runnable examples, and describes current `examples/fuji_labels_demo.py` as raw `viewer_ipc`; the source uses public `MapScene`/`LabelLayer`, while `tests/test_mapscene_examples.py:56-62` explicitly locks absence of raw IPC. | Preserve exact currently supported examples and honest public-interface ownership; do not imply absent examples, unsupported paths, or placeholder rendering. | Reconcile the catalog and dependent guide rows to `git ls-files examples` and current tested interfaces; use exact subdirectory paths; leave historical plans untouched. | Intended catalog membership for data-, network-, GPU-, and manually operated scripts requires owner judgment; feature-status wording must not be upgraded during reconciliation. | Static existence validation for every catalog referent; focused documentation contract tests including label API/P1 docs and the MapScene example structural test; Sphinx HTML build. | `VALIDATED` | `NOT_PROVEN` |
| W1C-02 | A source-distribution declaration must not claim a nonexistent required input. | packaging | `MANIFEST.in:10` (`include rust-toolchain.toml`) | The sdist manifest names a file absent from the exact tracked tree and worktree. | `test -e rust-toolchain.toml` is false; `git ls-files` has no such path; no inspected docs, tests, or workflows establish it as a required file; other root manifest referents exist and are tracked. | Preserve actual sdist/wheel contents and supported toolchain-selection behavior. | Remove only the stale include line. Restore a toolchain file instead only if W1-A proves an authoritative requirement; W1-C found none. | Maturin may ignore `MANIFEST.in` or only warn, so the active backend's observed archive behavior is required; package contents must remain correct cross-platform. | Build the sdist before and after, inspect complete archive inventories, establish the exact warning/content delta, and run installed-wheel/package smoke as applicable. | `VALIDATED` | `NOT_PROVEN` |
| N02 | Manifest cleanup is admissible only after feature/build ownership is proven. | Cargo manifest; W1-C docs/CI/packaging implication only | `Cargo.toml:[dev-dependencies]` declarations for `env_logger` and `sha2` | Duplicate dev-kind declarations appear removable while identical normal dependencies keep runtime/test/bench availability and the separate build dependency keeps build-script `sha2`. | No owned document, workflow, packaging rule, test, or bench command addresses dependency kind or relies on duplicate dev declarations. Current CI commands are ordinary Cargo/maturin metadata consumers. A `Cargo.toml` change appears in the M06, F3DZ, ANAMNESIS, and determinism path selectors, but those acceptance families run only for scheduled or explicitly selected manual scopes; routine PR gating is preflight plus Fast Contract. | Preserve dependency availability for library, binary, test, bench, and build targets; preserve feature and locked resolution behavior; do not infer physical acceptance from manifest cleanup. | W1-A may remove only the two duplicate dev declarations if its Cargo metadata and target analysis proves equivalence. W1-C requires no docs, workflow, packaging, script, or tool edit for N02. | Cargo target-kind semantics, build-dependency separation, lockfile stability, and conservative acceptance path filters. W1-A owns the manifest edit and validation. | W1-A Cargo metadata/tree comparison across applicable targets/features, locked build/test/bench proof, and exact `Cargo.lock` diff inspection; routine PR Fast Contract. No physical lane is required merely by W1-C, and no physical acceptance is implied. | `VALIDATED` for cross-surface implications; manifest transformation remains `NOT_PROVEN` | `NOT_PROVEN` |

## Rejected claims

- R-W1C-01 — Delete or deduplicate `assets/highres.png` and
  `docs/assets/highres.png`: rejected because byte identity alone does not prove
  identical package, source-asset, or documentation ownership.
- R-W1C-02 — Delete identical tutorial SVGs: rejected because identical
  placeholder/source artwork may occupy distinct stable page paths.
- R-W1C-03 — Delete large LFS TIFFs, other assets, or gallery binaries:
  rejected because size and age are not slop, and source/tests/docs/CI identify
  consumers or provenance.
- R-W1C-04 — Edit or regenerate history, corpora, goldens, certificates, fonts,
  astronomy data, or geoid assets: rejected absent an explicit behavior or
  provenance defect and its authoritative generation workflow.
- R-W1C-05 — Clean up or remove CMake: rejected because no behavior contract or
  supported-build evidence establishes it as dead; lack of documentation
  references is not proof.
- R-W1C-06 — Clean up shader contracts: rejected because every tracked
  contract is embedded or verifier-owned, and changing it is behavior-contract
  work rather than documentation slop.
- R-W1C-07 — Simplify `.gitignore` and its tracked exceptions: rejected because
  tracked ignored policy/evidence files are deliberate and ignore-rule
  complexity alone is not proof.
- R-W1C-08 — Simplify CI path filters or gates: rejected because tests lock
  scopes and cost controls, and exact-SHA/adapter-bound physical evidence must
  not be weakened.
- R-W1C-09 — Remove public interfaces or features based on owned-doc evidence:
  rejected because documentation cannot prove downstream reachability or
  installed-extension compatibility.
- R-W1C-10 — Remove locked features or dependencies beyond N02: rejected absent
  Cargo metadata, target, feature, and behavior proof.
- R-W1C-11 — Delete or rewrite `docs/superpowers/**`, prompts, audits, or specs:
  rejected because historical classification does not establish deletion
  safety; those files cannot overrule live policy but may preserve intent.
- R-W1C-12 — Modularize scripts or tools based on length, function count,
  TODOs, age, or absent textual callers: rejected because those signals do not
  prove behaviorally equivalent duplication, a bad seam, or dead reachability.

## Remaining uncertainty and `NOT_PROVEN`

- No package/sdist build, archive inspection, installed-wheel smoke, Sphinx
  build, documentation test, Cargo metadata/tree comparison, locked test/bench,
  hosted CI, GPU render, NVIDIA/Vulkan physical lane, golden comparison,
  certificate verification, or live branch-protection query was executed.
- W1C-01's exact contradictions are tree-proven. The complete intended catalog
  membership remains an owner decision for scripts requiring external data,
  network access, a GPU, or manual operation; reconciliation must not invent a
  support guarantee.
- The active maturin backend's handling of `MANIFEST.in`, including whether the
  missing `rust-toolchain.toml` produces a warning or archive difference, is
  `NOT_PROVEN` until an sdist is built and inspected.
- N02's actual manifest edit, target-kind equivalence, metadata, lockfile
  stability, and build/test/bench result are W1-A-owned and `NOT_PROVEN` here.
- Workflow inspection proves current repository policy and command routing,
  not that any hosted or physical run passed at this SHA. Every physical,
  adapter-specific, golden-bound, certificate-bound, and exact-run acceptance
  claim remains `NOT_PROVEN` unless backed by its own exact-SHA evidence.
- CMake removal, asset deduplication, generated/history cleanup, feature
  removal, shader consolidation, and public-interface cleanup remain rejected
  absent behavior-contract evidence; long files, TODOs, age, size, and textual
  search absence were never used as proof.

## W1-C mutation boundary

W1-C changed no source, policy, manifest, workflow, documentation claim, asset,
shader contract, script, tool, spec, test, generated artifact, golden,
certificate, or binary. It did not stage or commit anything. Creating this
report at `docs/refactor-forge3d-w1-c.md` is the sole authorized persistence
step and does not advance any finding beyond the statuses recorded above.
