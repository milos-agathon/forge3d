# Forge3D C: to D: migration plan

**Date:** 2026-08-03

This is a staged, reversible relocation of Forge3D and its owned state. The
source of truth is the target checkout and the repository's executable gates;
no application-code or manifest change is expected because runtime paths are
already relative.

## Scope and baseline

Move to `D:`:

- the repository, all worktrees, Git metadata, Git LFS objects, Rust toolchains
  and caches, Python installations owned for this project, source and
  wheel-validation virtual environments, datasets, and generated outputs;
- build, pip, dataset, COG, IBL, anamnesis, and temporary caches.

Leave Windows-managed GPU drivers, Visual Studio/SDK components, and optional
system Git/Vulkan in place unless the owner separately authorizes a reinstall;
record each as **retained system dependency** or **reinstalled on D:**.

Observed baseline (2026-08-03):

- `D:` NTFS has approximately 704.6 GiB free; `C:` has approximately 4 GiB.
- The repository occupies approximately 74.1 GB; `.git/lfs` is approximately
  49.2 GB with 415 cached objects.
- The initial persistent audit saw 20 paths on `C:` plus 2 paths on `D:` that
  still attach to the `C:` common Git directory. Temporary audit/PR worktrees
  can change the live count; Phase 0 must regenerate the authoritative ledger.
- `core.hooksPath` contains an absolute `C:` path.
- `main` is dirty and has suspected sensitive untracked material. Record and
  disposition it generically; do not enumerate, read, hash, copy, log, or stash
  sensitive files.

## Target layout

Use this layout (create directories before any install):

```text
D:\forge3d\main                 primary checkout
D:\forge3d\main.git             new bare transport repository
D:\forge3d\worktrees            recreated linked worktrees
D:\forge3d\deps\cargo           CARGO_HOME
D:\forge3d\deps\rustup          RUSTUP_HOME
D:\forge3d\deps\python         project-owned Python/tool files
D:\forge3d\deps\venv           source and wheel-validation venvs
D:\forge3d\cache\pip            PIP_CACHE_DIR
D:\forge3d\cache\datasets      dataset cache
D:\forge3d\cache\tmp           TEMP/TMP and tool scratch
D:\forge3d\artifacts            bundles, manifests, receipts, wheels, builds
```

## Phase 0 — quiesce, inventory, and custody

1. Stop editors, terminals running Forge3D, Python/Rust/GPU jobs, file
   watchers, sync clients, and CI jobs. Confirm no process has an open handle
   under the source checkout or its caches.
2. Regenerate the authoritative migration ledger from a fresh enumeration of
   every live worktree. For every entry record the old path, intended D path,
   branch or detached HEAD OID, Git dir/common-dir, upstream, dirty-state
   class, ignored-state class, disposition, owner, and receipt identifier.
   Preserve every branch/tag/ref, stash, custom ref, and temporary ref,
   including refs reachable only from detached HEADs.
3. With the owner, disposition every staged, unstaged, untracked, and ignored
   item: commit, export to user custody, carry forward, or explicitly discard.
   Automation must never read, hash, copy, log, or stash a sensitive item.
   Give the owner a custody receipt for anything excluded from automation; a
   missing decision stops the migration.
4. Before creating the bundle, create a named
   `refs/migration/worktrees/<safe-name>` ref for every detached worktree HEAD.
   Record each ref/OID in the ledger and verify those refs are present in the
   bundle and later in the imported D repository; `--all` alone does not make
   an otherwise unreferenced detached commit reachable. Then create a verified
   all-ref Git bundle and an owner-reviewed, path-redacted LFS manifest. Record
   source object/ref OIDs, LFS OIDs and sizes, the 415-object inventory, and
   checksums in `D:\forge3d\artifacts`. Keep the original source read-only after
   capture.

## Phase 1 — independent Git/LFS migration

1. Initialize `D:\forge3d\main.git` as a new bare repository and import the
   bundle. Verify every `refs/migration/worktrees/<safe-name>` ref from the
   ledger exists in the imported repository. Check out `D:\forge3d\main`
   through a transport URL with `--no-local`; do not move the existing common
   `.git` directory or repair the old primary worktree in place.
2. Prove exact ref and OID parity, no alternates, and a clean full `git fsck`.
   Export any retained reflog evidence separately in the migration ledger.
   Rehearse independence with a second D-only clone, remove its source
   dependency, and verify it still resolves the expected refs and commits.
3. Configure and confirm that the new D bare repository and primary clone use
   D-local LFS storage (an explicit path such as `D:\forge3d\cache\lfs`).
   Record `git lfs env` for both; any `C:` path in `LfsStorageDir` or
   `LocalMediaDir` is a stop condition before fsck or cutover. For every ref,
   compare each LFS OID and byte size; checksum and copy or fetch all 415
   cached objects into that D storage, run `git lfs fsck`, and materialize the
   11 files required by the recorded HEADs. A missing object or size mismatch
   is a stop condition.
4. Keep `D:\forge3d\main` as the primary worktree and recreate every approved
   linked worktree under `D:\forge3d\worktrees`. Retire a linked entry only
   with a ledger entry and owner approval. Re-attach detached worktrees to
   their recorded OIDs before changing branches.
5. Set D-local hooks, `safe.directory`, editor/workspace paths, and any project
   Git configuration. Verify no Git config, worktree administrative file,
   hook, or editor setting still points at `C:`. Keep the old tree untouched as
   the rollback source.

## Phase 2 — D-owned toolchains and state

1. Before installing anything, set `PATH` (D-owned bins first),
   `CARGO_HOME=D:\forge3d\deps\cargo`,
   `RUSTUP_HOME=D:\forge3d\deps\rustup`,
   `PIP_CACHE_DIR=D:\forge3d\cache\pip`, and `TEMP`/`TMP` to
   `D:\forge3d\cache\tmp`. Persist only after the proof in Phase 4.
2. Install a clean Rust stable MSVC toolchain and Python 3.13. Create separate
   D-only source and wheel-validation venvs under `D:\forge3d\deps\venv`.
   Bootstrap a temporary pip constraints file from the clean source venv,
   record the resolver output and hashes, and discard the constraints file
   after the migration. Do not change a committed lockfile or dependency
   declaration.
3. Install the existing repository, test, and docs requirements plus the
   already-required Make, CMake, and Ninja tools. Keep Cargo targets and other
   build outputs per worktree on D; do not share stale C: targets.
4. Rebuild every generated output in the ledger on D, including native
   extensions, documentation, shader-derived products, and test artifacts.
   Redirect dataset, COG, IBL, anamnesis, and temporary caches to the target
   directories and record their source/version/checksum receipts.
5. In a VS Developer PowerShell, verify the MSVC compiler, SDK, RTX adapter,
   and Vulkan path. Explicitly classify drivers, VS/SDK, and optional system
   Git/Vulkan as retained system dependencies or separately reinstalled; never
   delete them as part of this migration.

## Phase 3 — proof on D only

1. In a fresh D shell, prove every checkout, ref, LFS object, tool, cache,
   generated output, and environment variable resolves on D. Use path/ref/LFS
   checks and receipts; do not use C as a fallback.
2. Run the exact authoritative gates in [Commands](#commands). Add shader
   reachability, an adapter/capability probe, and focused GPU and golden tests.
   Keep golden-update switches off; do not place secrets in logs or artifacts.
3. Build an exact release wheel into D, then install it into the fresh
   wheel-validation venv with `scripts/install_compatible_wheel.py`. Set
   `FORGE3D_NO_BOOTSTRAP=1` and `FORGE3D_TEST_INSTALLED_WHEEL=1`; run install,
   license, smoke, and canonical tests against the installed wheel, not the
   source tree. Save command lines, tool versions, paths, checksums, and exit
   statuses as evidence receipts under `D:\forge3d\artifacts`.
4. Mark CI-only lanes and any unrun local lanes explicitly. Expected
   environment-specific lanes that may remain unrun locally include
   informational Windows/macOS interactive-viewer lanes, the multi-OS/Python
   wheel matrix, `test-python-slow`, `test-terminus-fuzz`, Linux
   ARM/container, Linux system-PROJ, doctests or `cargo doc` if unrun,
   protected signing/golden refresh, NVIDIA M-06/F3DZ/anamnesis, and GitHub
   artifact aggregation. Do not claim local proof for an unexecuted lane or
   weaken any gate.

## Phase 4 — cutover and rollback rehearsal

1. Open a fresh shell and workspace; update user/project environment variables,
   IDE folders, task runners, and shortcuts to D paths. Work exclusively from
   `D:\forge3d\main` and `D:\forge3d\worktrees`.
2. Perform a normal edit, native rebuild, test, docs build, LFS checkout, and a
   representative GPU render from D. Confirm receipts and no C path leakage.
3. Retain the C tree read-only until the fresh rebuild and normal workflow are
   successful and the rollback rehearsal passes. Rollback means stop writes,
   restore the prior environment/workspace pointers, and reopen the preserved C
   checkout; verify refs, worktrees, LFS files, and a smoke test.
4. Destructive cleanup of C data is a separate, explicit user approval. Never
   delete Windows-managed drivers, VS/SDK components, or other system deps.

## Definition of done and rollback

Done means every Phase 0 ledger entry is closed; all approved state,
refs/stashes, Git/LFS data, 11 HEAD files, tools, caches, datasets, and
generated outputs have D receipts; exact ref/OID/LFS parity, fsck, no-alternate,
shader/GPU/golden, installed-wheel, smoke, license, API, Rust, Python, and docs
gates are green; every approved linked worktree is recreated or explicitly
retired; every unrun CI-only lane is named; and a fresh-shell normal workflow
plus rollback rehearsal succeeds. Until then, C remains read-only and the
migration is not complete.

Rollback uses the preserved C checkout and source bundle: stop D writes,
restore C environment/workspace paths, validate the original refs and LFS
objects, and resume only after the owner signs the rollback receipt. Do not
delete or overwrite either copy while investigating a failed proof.

## Commands

Run from `D:\forge3d\main` in the D source venv, with the Phase 2 environment
set. Current [AGENTS.md](../AGENTS.md) and CI are authoritative at execution;
these are the reviewed local migration gates. Reconcile any command or feature
drift against them before running.

```powershell
python -m pip install -U pip maturin pytest
maturin develop
python -m pytest tests/test_api_contracts.py -v --tb=short
python scripts/ci_pytest_lane.py -v --tb=short
cargo fmt --check
cargo forge3d-clippy
cargo test --workspace --features default,async_readback,copc_laz,cog_streaming,gis-remote,geos-topology,weighted-oit,wsI_bigbuf,wsI_double_buf,enable-pbr,enable-tbn,enable-normal-mapping,enable-hdr-offscreen,enable-renderer-config,enable-staging-rings,shader-contract-asserts -- --test-threads=1 --skip gpu_extrusion --skip brdf_tile
make -C docs html
```

For installed-wheel proof, use the exact D output directory and fresh venv:

```powershell
maturin build --release --out D:\forge3d\artifacts\wheels
python scripts/install_compatible_wheel.py D:\forge3d\artifacts\wheels
$env:FORGE3D_NO_BOOTSTRAP = '1'
$env:FORGE3D_TEST_INSTALLED_WHEEL = '1'
python -m pytest tests/test_license.py -v --tb=short
python scripts/ci_pytest_lane.py -v --tb=short
python -m pytest tests/test_install_smoke.py -v --tb=short
```

Read-only shader and adapter probes:

```powershell
python -m pytest tests/test_shader_reachability.py tests/test_shader_proofs.py -v --tb=short
python scripts/terrain_ci_probe.py --mode terrain
```

Cross-check command intent and paths against [CI](../.github/workflows/ci.yml),
the [pytest lane](../scripts/ci_pytest_lane.py), the
[wheel installer](../scripts/install_compatible_wheel.py),
[pyproject.toml](../pyproject.toml), [architecture](start/architecture.md),
and the [feature map](guides/feature_map.md) before execution.

## Non-goals

- No application-code, manifest, lockfile, or committed dependency change.
- No custom migration tool, new dependency, speculative framework, or silent
  placeholder/fallback.
- No automated handling or disclosure of sensitive files.
- No system-driver/SDK deletion or unrequested reinstall.
- No golden, determinism hash, certificate, or signed artifact update merely to
  make a gate pass.

## Supplemental context

The [harness-engineering repository](https://github.com/lopopolo/harness-engineering)
is supplemental context only. Target-local truth governs; its useful framing is
limited to a reversible candidate, native proof, cutover, and rollback sequence.
Do not copy its policy into this plan.
