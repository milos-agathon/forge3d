# GPT-5.6 Sol runbook (Ultra mode): behavior-preserving forge3d refactor

Paste the prompt below into Codex with **GPT-5.6 Sol** in **Ultra mode**. Start
it at the intended forge3d checkout. This is an execution prompt, not merely a
request for a plan.

---

`/goal` Exhaustively inspect forge3d and reduce every verified instance of code
or documentation slop through small, reviewable, behavior-preserving refactors.
Preserve every supported function, public API, shader contract, recipe, render
path, test, example, golden, certificate, build lane, and documented behavior.
Create and continuously maintain the durable plan and progress ledger at
`docs/refactor-forge3d.md`; do not stop after planning.

## Contract and decision rule

The contract is the requested exhaustive refactor plus the smallest evidence
that proves it without regressions. Apply the current applicable `AGENTS.md`
policy dynamically. Under its Minimum Sufficient Work (MSW) kernel, every audit
finding, review comment, plan step, and attractive cleanup is only a **claim**.
Act on a claim only when deleting it would leave this contract unmet or
unproven. Take the smallest reliable action that closes the gap, prove it, and
stop at the fixed point. Record rejected claims in one line; do not investigate
or fix them further. Apply authoritative fuses exactly as the live policy states.

Never invent a cap, threshold, quota, budget, timeout, retry count, round count,
file-size gate, line-count gate, acceptance count, or priority formula. A limit
is valid only when required by the requester, a technical/platform contract,
authoritative project policy, or measured evidence needed to prove this task.
Repository specs and tests may define exact technical thresholds, but only for
the claims they govern. Metrics below are coverage and triage evidence, never
targets, quality gates, or deletion instructions.

This is a refactor, not a redesign, migration, dependency upgrade, feature
project, performance rewrite, or style churn. Improve internal structure while
preserving observable behavior. Use transformations such as Extract Function,
Extract Module, Inline Function, Move Function, Rename, simplified conditionals,
consolidated duplicate truth, or proven-dead-code removal only when repository
evidence makes the transformation necessary and safe.

## Authority and worktree safety

Before any edit:

1. Locate and read every applicable `AGENTS.md` and override from repository root
   to the target file. Read `CLAUDE.md` only if present, plus relevant
   `.claude/rules/`, `CONTRIBUTING.md`, manifests, CI, specs, and policies.
2. Fetch/refresh refs only when the invoking user and environment authorize it.
   Record the intended base ref, exact base SHA, branch, source-worktree status,
   isolated-worktree path, toolchains, platform, adapter/backend, and unavailable
   evidence in `docs/refactor-forge3d.md`.
3. Preserve all user work. Inspect the source checkout, index, and untracked
   files at run time; never rely on a historical dirty-file list. Work in a clean
   isolated worktree on a `codex/` branch at the exact intended base. Never
   overwrite, stage, reformat, revert, or commit user-owned changes.
4. Use `git ls-files` as tracked-surface authority. Classify generated files,
   fixtures, corpora, assets, goldens, certificates, binaries, ignored outputs,
   and historical material before treating anything as cleanup.

The runbook authorizes only repository inspection, in-scope edits, and
non-destructive local proof that the invoking user and platform already allow.
It does not independently authorize pushes, PRs, publishing, dependency
installation/upgrades, certificate/key rotation, golden refresh, external
writes, or service changes. Follow the live `AGENTS.md` orchestration, review,
`/simplify`, `/code-review`, commit, push, and mergeable-PR workflow when the
invocation authorizes those actions; otherwise stop at the first missing
authority and report the ready-to-execute action. Never simulate an unavailable
review or evidence lane.

## Provenance and freshness

This runbook reconstructs the complete purpose and controls of an untracked
2026-07-16 predecessor from its Codex rollout, then refreshes them against
`origin/main` at `de97cedd1da91ebcb234aa2edd729ea4778a8222` on 2026-08-11.
The following snapshot is SHA-bound and **non-authoritative if the run starts at
any other SHA**. Re-measure it rather than carrying these facts forward.

| Surface at `de97cedd` | Measured size |
|---|---:|
| Rust | 1,189 files / 300,661 lines |
| Python plus stubs | 555 files / 202,037 lines |
| WGSL | 143 files / 32,486 lines |
| `src/` | 1,338 tracked paths |
| `tests/` | 986 tracked paths |
| `python/` | 140 tracked paths |
| `docs/` | 139 tracked paths |
| `examples/` | 46 tracked paths |

### Exact-SHA, run-bound evidence

These results are non-transitive snapshots and require a live recheck at every
later SHA. They do not apply to this runbook's subsequent documentation change.

| Run | Exact evidence at `de97cedd` |
|---|---|
| [CI 31468556939](https://github.com/milos-agathon/forge3d/actions/runs/31468556939) | Push run completed successfully at 2026-08-11 07:25 UTC; CI Preflight, Fast Contract, and PR Core Success passed. The same run skipped the exhaustive Rust, Python, wheel, golden, and physical acceptance families plus Full Acceptance Summary, so full and physical acceptance at this SHA is `NOT_PROVEN`, neither failed nor passed. |
| [Docs 31468556665](https://github.com/milos-agathon/forge3d/actions/runs/31468556665) | Run completed successfully at 2026-08-11 07:21 UTC, including Build Docs and Deploy Docs. |

As of 2026-08-11, a live branch-protection snapshot shows `main` requires the
app-bound `PR Core Success` context (`app_id: 15368`) with `strict=false`;
recheck before relying on it, and do not treat it as Full Acceptance or physical
proof.

A SHA-1 content scan at this SHA found **no byte-identical Rust, Python, stub, or
WGSL files**. Do not claim duplication was found: semantic or partial repetition
still requires inspection and proof.

Measured navigation signals at `de97cedd` (not proven slop):

- `examples/california_cigar_smoke_demo.py`: 11,031 lines;
  `NumpyPhysicalSmokeDomain` 453, `hybrid_fire_sources_rgba` 395, and
  `render_video` 547 lines.
- `python/forge3d/map_scene.py`: 6,205 lines; `MapScene` 1,480, `validate` 618,
  and `_render_impl` 229 lines.
- `python/forge3d/terrain_params.py`: 2,558 lines;
  `make_terrain_params_config` 231 lines.
- `src/shaders/terrain_pbr_pom.wgsl`: 5,203 lines.
- `src/gis/vector.rs`: 2,932 lines and 140 function declarations.
- `src/terrain/renderer/virtual_texture.rs`: 2,973 lines and 90 function
  declarations; `TerrainMaterialVT` impls begin near lines 389 and 2,836, and
  `TerrainMaterialVTRuntime` near 1,258.
- `src/verify/ir/engine.rs`: 2,516 lines and 68 function declarations.
- `src/core/atmosphere/bake.rs`: 2,330 lines.
- `src/shader_sources.rs::terrain()` and `::pbr()` repeat a BRDF/lighting
  assembly sequence. This is only a consolidation candidate; shared ownership,
  preprocessing, entry-point contracts, and output equivalence must be proven.

## Non-ordinary cleanup boundaries

- `src/labels/unicode/generated.rs` is generated: do not edit it by hand.
- `build.rs` writes `registered_wgsl.rs` to `OUT_DIR`; generated output is not a
  competing tracked implementation.
- `Cargo.lock`, reference corpora, assets, goldens, certificates, binaries, and
  ignored outputs are not ordinary source cleanup. Change them only when an
  explicit contract requires it and the proper generation/provenance workflow is
  proven.
- Large tracked assets include Switzerland land cover (about 229 MB) and Bryce
  Canyon (about 119 MB). Size alone is not slop or deletion evidence.
- `Cargo.toml` explicitly quarantines historical Clippy categories. Do not erase
  or broaden that policy merely to produce a smaller configuration.
- Treat `docs/superpowers/` as historical wherever live policy, code, tests, or
  CI supersede it. It may explain intent but cannot overrule current truth.

## Phase 0 — establish current truth

Map the current build, packaging, runtime, and public-contract topology before
editing: the Rust crate/feature graph; PyO3 registrations, wrappers, exports, and
stubs; Rust/WGSL structs, bindings, shader assembly, and entry points; Python and
CLI APIs; tests, examples, recipes, docs, CI; assets, bundles, serialization,
goldens, and certificates.

Run the cheapest trustworthy baseline checks relevant to the first claims.
Record exact command, environment, identity, result, skip count, and material
output. Separate pre-existing failures from regressions and do not absorb
unrelated failures into this refactor. Measure structural signals and call/reach
graphs where useful, with methodology and false-positive limitations.

## Phase 1 — exhaustive claim sweep

Inspect every tracked product-relevant area, including `src/`, `python/`, type
stubs, `tests/`, `examples/`, `docs/`, `scripts/`, `tools/`, benches, build and
packaging files, workflows, specs, fixtures, assets, and other tracked top-level
surfaces. Cover small files and cross-language seams, not just hotspots.

Search for falsifiable evidence of behaviorally equivalent duplication,
copy/paste drift, mixed responsibilities, confusing state flow, dead reachable
structure, stale fallbacks/probes, wrapper/native/feature drift, unnecessary
indirection, silent degradation, unsafe resource ownership, broad error
handling, misleading defaults, stale or contradictory documentation, weak or
hidden tests, and duplicated build truth. A text-search non-caller, long file,
old code, TODO, unfamiliar abstraction, or reduced line count is not proof.
Account for downstream/public use, PyO3 registration, imports, traits/dynamic
dispatch, cfg/features, reflection, build scripts, shader entry points, CI,
packaging, tests, examples, docs, and history before accepting a claim.

The sweep must explicitly map these live contract families and their relevant
code/spec/test evidence without claiming acceptance merely because artifacts
exist:

- **CENSOR:** authoritative execution truth is
  `docs/censor-validation-policy.md`; preserve capability/provenance truth,
  tracked allocation and budget behavior, explicit degradation, certificate and
  tamper semantics, shader-use reporting, and honest ABSENT/CRASH outcomes.
- **SUTURA:** `MapScene` must not regain placeholder rendering; preserve
  compiled-plan ownership, canonical serialization/bundle behavior, validation,
  label-culling ownership, and structured diagnostic blocks.
- **Rust–PyO3–stub parity:** verify native signatures, module/class ownership,
  registration, exports, typing, feature gates, and installed-extension behavior.
- **Rust–WGSL/GPU contracts:** verify alignment and array stride, binding/pipeline
  layouts, stage limits, resource lifetimes, and renderer-owned shader assembly.
- **LIMES:** preserve its opt-in, default-preserving vector coverage and
  compiled-scene cache boundaries in `src/vector/coverage/*` and
  `src/py_functions/vector/coverage_cache.rs`.
- **Virtual textures, visibility, and terrain / TESSELLA:** preserve the
  single-source VT store, residency/streaming and picking seams; physical proof
  is fail-closed.
- **Astronomy and atmosphere:** preserve SIDERA's public numerical window
  2000–2050 and backend-specific deterministic night evidence. Preserve AETHER's
  production LUT/provenance path under `src/core/atmosphere`; the independent
  acceptance oracle in
  `src/path_tracing/hybrid_compute/aether_reference.rs` must remain independent
  and must not be deduplicated into production.
- **SUBSTRATIA:** acceptance requires exact-SHA, adapter-bound, golden-bound,
  zero-skip physical evidence; portable or mock proof cannot substitute.
- **Determinism/certificates, text/Unicode, GIS/units, and examples/recipes:**
  preserve canonical inputs/outputs, provenance, Unicode generation/behavior,
  coordinate and unit semantics, and real runnable paths.

## Required ledger: `docs/refactor-forge3d.md`

Create the ledger before implementation and keep it accurate in every refactor
checkpoint and commit. It must contain:

1. Mission, behavior-preservation rule, exact scope, non-goals, authority, and
   completion conditions.
2. Baseline identity and environment, including exact SHA, worktree/user-work
   state, toolchains, backend/adapter, limitations, and stale-snapshot warning.
3. Architecture and contract map covering every family above.
4. Baseline and final validation matrices with exact commands and results.
5. Metrics with methodology, SHA, caveats, and explicit non-gate status.
6. A tracked-surface coverage matrix: area, paths, evidence inspected, result,
   accepted/rejected claim IDs, and remaining uncertainty.
7. A finding register with this schema:

   `ID | priority basis | subsystem | path:symbol | claim | evidence | behavior/contract | smallest transformation | risk/dependencies | required proof | status | commit/PR`

   Derive priority only from contract necessity, dependency order, evidence, and
   regression risk; never from a fabricated score or threshold.
8. Dependency-ordered implementation waves. For each: claim IDs, purpose,
   files/symbols, prerequisite evidence, preserved behavior, exact
   transformation, targeted proof, live path if required, rollback boundary,
   and applicable authoritative gates.
9. One-line decision records for rejected claims; reasons and owner decisions for
   deferred, blocked, or unproven claims.
10. An append-only checkpoint log: timestamp, IDs, files, before/after facts,
    tests/live runs and identities, review findings, commit SHA, residual risk,
    and next necessary claim.
11. Final proof, exact head/PR identity when applicable, and residual-risk report.

Use these statuses exactly: `DISCOVERED`, `VALIDATED`, `PLANNED`, `IN_PROGRESS`,
`LOCALLY_PROVEN`, `REMOTE_PROVEN`, `DEFERRED`, `REJECTED`, `BLOCKED`, and
`NOT_PROVEN`. No accepted claim may advance beyond its evidence.

## Phase 2 — execute and prove checkpoints

For each necessary claim:

1. Record the behavior, evidence, exact transformation, smallest sufficient
   proof, and rollback boundary.
2. If behavior is not locked, first add and run a characterization or contract
   test against pre-refactor code. Test behavior, not a preferred implementation
   shape, except where an authoritative architecture rule is the contract.
3. Make the smallest complete behavior-preserving transformation. Keep
   mechanical moves separate from semantic changes when the proof requires it.
4. Run targeted format/static checks, focused contract/behavior tests, and any
   affected real example, recipe, CLI, installed-wheel, render, or golden lane.
5. Inspect the complete diff for API, feature, docs, fixture, golden,
   certificate, generated-file, and user-work changes. Perform the live policy's
   mandated adversarial review cycle; fix necessary findings and repeat the
   affected proof until approved.
6. Update the ledger with observed results and mark the checkpoint commit-ready.
   Stage and commit only at the point allowed by the live mandatory workflow;
   include only owned, self-contained, reviewed, proven changes and map the
   resulting commit to its claim IDs. If commit authority is absent, stop first.
7. Re-evaluate remaining claims under MSW. Do not reopen proven work without new
   evidence and do not create in a later round a claim already visible earlier.

When Plan mode is active, follow the live policy's planner and three blind
peer-review loops before execution and save the approved plan where that policy
requires. When Plan mode is inactive, begin at its execution stage. After each
task use the mandated reviewer loop; after whole-change approval invoke
`/simplify` and `/code-review`, then follow the live policy's exact authorized
commit, push, and PR sequence. If a PR is authorized, make it mergeable. Resolve
the current model routing from live policy; this runbook does not freeze a stale
route.

Within that routing, give parallel read-heavy work disjoint ownership and have
the lead integrate its evidence and decisions.

## Validation ladder

Discover commands and features again from the exact candidate SHA. At
`de97cedd`, the **Fast Contract** portion of **PR Core Success** in
`.github/workflows/ci.yml` is:

```bash
cargo fmt --check
cargo forge3d-clippy
maturin develop
FORGE3D_NO_BOOTSTRAP=1 python scripts/ci_pytest_lane.py --profile fast -v --tb=short
```

Full PR Core Success additionally depends on the base-owned preflight. Neither
Fast Contract nor PR Core Success is physical acceptance. Depending on the
accepted claim, the next evidence may include:

- exact `cargo check`, Rust test, and doctest feature commands copied live from
  CI at the candidate SHA;
- `cargo forge3d-clippy-acceptance`;
- `FORGE3D_NO_BOOTSTRAP=1 python scripts/ci_pytest_lane.py --profile full -v --tb=short`;
- `FORGE3D_NO_BOOTSTRAP=1 python scripts/ci_pytest_lane.py --profile full --slow-lane -v --tb=short`;
- `python -m sphinx -b html docs docs/_build/html`;
- affected installed-wheel tests, examples, recipes, renders, goldens,
  certificates, platform matrices, or physical GPU lanes.

Run the narrowest proof sufficient for each checkpoint and all applicable
authoritative evidence before final completion. `CONTRIBUTING.md`'s raw
`pytest tests/` and `cargo test --all-features` are not current PR-core
authority. A skipped, ABSENT, unavailable, mocked, wrong-SHA, wrong-adapter, or
wrong-backend lane is `NOT_PROVEN`, never a pass. Local Metal or portable CI does
not prove NVIDIA/Vulkan or another physical lane.

## Completion and report

Halt when either an applicable authoritative fuse requires halt with open items
reported, or every completion condition below is true:

- the coverage matrix contains evidence for every tracked product-relevant area;
- each discovered claim has an evidence-backed terminal status, with no vague
  `DISCOVERED`, `PLANNED`, or `IN_PROGRESS` items;
- every necessary in-scope transformation is proven at the final exact head;
- public behavior and all affected contract families are preserved by the
  applicable validation ladder;
- unavailable remote/physical evidence is explicitly `NOT_PROVEN` and no feature
  is called accepted on code/docs existence alone;
- `docs/refactor-forge3d.md` matches the exact final SHA and records commits/PR,
  proof, rejected claims, blockers, deferrals, uncertainties, and residual risk;
- the original user work is untouched; and
- every live-policy review, `/simplify`, `/code-review`, and authorized PR
  requirement is satisfied, or the run halts at a clearly identified missing
  authority/evidence boundary.

Report only the outcome against the contract, exact branch/SHA/PR identity,
claim-to-commit map, before/after structural facts, validation table, live paths
actually run, remaining `NOT_PROVEN` evidence, residual risk, and one line per
rejected claim worth the user's attention. Reduced LOC and phrases such as
“looks good,” “cleaner,” or “should work” are not proof.
