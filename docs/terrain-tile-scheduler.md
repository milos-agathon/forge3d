# Shared terrain tile-request scheduler

Architecture review item 32 is implemented by the private
`terrain::page_table::scheduler` module. `AsyncTileLoader` and
`AsyncOverlayLoader` retain their existing constructors, reader traits,
request/cancel methods, drain methods, statistics and counter tuple layouts.
Their concrete adapters produce covered height tiles and RGBA tiles; the
scheduler owns admission, channels, dispatch, workers, completion selection,
acknowledgement and publication. The browser height-loader entry point uses
the same bounded request state.

The two established policies remain distinct inside this shared owner:

| Behavior | Height | Overlay |
| --- | --- | --- |
| Worker count | Existing normalization to 1 through `MAX_LOADER_WORKERS` (16) | Existing normalization to at least 1; no new ceiling |
| Dispatch | Workers consume the shared FIFO request channel | Dispatcher assigns requests round-robin to worker queues |
| Capacity | Retained until complete, error or cancelled terminal acknowledgement | Logical pending slots released by cancellation or LOD replacement |
| Coalescing | Reject requests conflicting with the preferred pending LOD | Reject conflicts or replace pending ancestors/descendants after successful enqueue |
| Drain limit | Counts all consumed terminal events | Counts delivered, uncancelled tiles |
| Explicit cancellation counter | Counts newly cancelled tickets | Explicit `cancel` does not increment it; every accepted fine-policy request retains its existing increment, even without a replaced ancestor |
| Drop | Nonblocking; workers exit after disconnection, including failed completion sends | Nonblocking; dispatcher and workers finish queued reads and exit after disconnection |

Height errors remain terminal errors, without fabricated tiles. Cancellation
and result selection retain their atomic state transition, and browser
generation tickets, payload validation, cancellation and error acknowledgement
remain unchanged. Reader-returned dimensions, height coverage, RGBA bytes and
host-memory accounting are preserved by the concrete adapters. No dependencies,
public interfaces, serialized formats or concurrency budgets were added.

The adapter tests exercise duplicate requests, saturation, zero configuration,
both LOD policies in both request orders, completion limits, cancellation,
parallel workers, overlay round-robin ordering and shutdown with queued reads.
The existing height error, cancellation and acknowledgement tests are retained;
their internal atomic transition test now lives beside the shared bounded state.
An additional dispatch-saturation test proves enqueue failure does not cancel
the requests a replacement would supersede. Tests reuse the existing height
test polling deadline and interval.

Worktree base: `origin/main` at
`a5e37efd17ed9c5a2cb0a94c7314876d0aa93c30`, the source revision inspected by the
review. The primary checkout's older branch and unrelated changes are preserved.

## Changed files

- `src/terrain/page_table/scheduler.rs`: shared channel, worker, dispatcher,
  cancellation and completion owner.
- `src/terrain/page_table/scheduler/bounded.rs`: height's existing native/browser
  ticket lifecycle, now independent of payload type.
- `src/terrain/page_table/scheduler/coalescing.rs`: overlay's existing replacement
  and logical-capacity policy.
- `src/terrain/page_table/height_loader.rs` and `overlay_loader.rs`: concrete
  payload adapters using the shared scheduler.
- `src/terrain/page_table/mod.rs` and `src/wasm_page_table.rs`: native and browser
  module registration.
- `src/terrain/page_table/loader_tests.rs`: shared adapter characterization and
  the existing polling helper.
- This document records the contract, preserved policies and execution evidence.
- The original, untracked `D:/forge3d/docs/architecture-review-forge3d-50.html`
  receives only item 32's completion status, source links and evidence note.

## Executed validation

Windows x86_64 MSVC; Rust tests use real OS worker threads with controlled
height and RGBA readers. The browser check builds the optimized
`wasm32-unknown-unknown` artifact and executes it in Node through the existing
ORBIS registry/residency fixture.

| Check | Result |
| --- | --- |
| `cargo test --lib terrain::page_table --features default,cog_streaming,shader-contract-asserts` | 19 passed, 0 failed, 0 ignored; 1,536 unrelated tests filtered out |
| `python -m pytest tests/test_orbis_wasm_contract.py -m wasm -v --tb=short --junitxml=terrain-scheduler-wasm.xml` | 1 passed; real optimized WASM artifact executed |
| `python scripts/assert_junit_zero_skips.py terrain-scheduler-wasm.xml` | 1 test, 0 failures, 0 errors, 0 skips |
| `cargo forge3d-clippy` | Passed the repository's all-target lint alias with `extension-module,default,cog_streaming,shader-contract-asserts` and `-D warnings` |
| `cargo fmt --check` | Passed |
| `git diff --check` | Passed |
| Adapter method-name comparison against the base and ownership inspection | Existing methods retained; neither adapter owns channels, workers, pending sets or counters |

Rust tests and lint explicitly set `CARGO_TARGET_DIR` to this worktree's `target`.
The WASM test ran with `FORGE3D_NO_BOOTSTRAP=1` and a separate `target-wasm`
output directory; its build products were subsequently moved under
`target/wasm-contract` to keep generated products out of the source diff.
The passing native run follows the corrected counter assertion. The only
subsequent Rust edit moved an unchanged test module below its implementation
for lint compliance; the final lint gate compiled that placement. Native runtime
and WASM evidence were reused because scheduling logic and assertions did not
change. The native feature lane emitted warnings in unchanged modules; lint
also reported filesystem cache hard-link warnings and copied cache files.
Neither check reported a scheduler source warning in the final run.

Full logs and the zero-skip JUnit report are retained in
`C:/Users/milos/.codex/visualizations/2026/10/14/01a13a3c-d101-7660-bd4a-da0dc892454e/`:
`terrain-scheduler-tests.log`, `terrain-scheduler-wasm.log`,
`terrain-scheduler-clippy.log`, `terrain-scheduler-format.log` and
`terrain-scheduler-wasm.xml`.

Physical GPU renders, native-extension Python runtime tests, the broader CI
matrix and hosted runs were not executed: item 32 changes the request scheduling
boundary, and its runtime proof uses both existing Rust adapters and the browser
facade. No native Python binary was reused as runtime evidence. No production
dependency, test gate, golden, certificate or determinism hash was changed.
The published PR contains the scheduler, adapters, native/browser registration,
tests and these implementation notes. Its hosted checks are reported on the PR;
the validation above records the executed local evidence.

## Reusable setup observations

The review's inspected revision is newer than the primary checkout; the fresh
worktree therefore uses the review's `origin/main` revision. During the initial implementation, an app-managed
checkout at `C:/Users/milos/.codex/worktrees/terrain-tile-scheduler/forge3d` had not yet
completed; it contains none of this implementation. The verified implementation
worktree is `D:/forge3d/.worktrees/terrain-tile-scheduler`.

The ordinary Git checkout stalled on unrelated LFS assets. Completing the fresh
checkout with `GIT_LFS_SKIP_SMUDGE=1` was sufficient for these fixtures. Shell
calls do not retain an earlier call's task-specific environment settings, and the
host supplies a shared build directory, so each Cargo validation command needs
its own explicit `CARGO_TARGET_DIR`. Suggested addition to
`.codex/usage-saving/validation.md` (not applied): document those two setup steps
for fresh worktree tasks that require no LFS assets. No agent contract or skill
was changed.
