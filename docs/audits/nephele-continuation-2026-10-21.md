# Bounded continuation transport — 2026-10-21

**Local candidate; physical acceptance not yet measured.** The candidate is now
based on owner-approved `eb312522`, with the corrected graph assertions and
owner-approved G4b relative-error policy described at the end of this note.
Milos approved the
bounded transport investigation and the exact CPU-only fast-lane selection
with “Yes to both.” No subagents or local GPU tests were used.

The preceding NVIDIA capture at `d5507249` failed: B1 94.17%, G4a 67.31% (corrected CIELAB diagnostic).
The accepted reference response remains a first-order estimate, and the
existing G4b SSIM rule remains unchanged.

## Measured sampling error

`make_terrain_params_config` defaults to clip `(0.1, 6000.0)`; the capture does
not override it. The old continuation march uses `reach / media.grid.z`, so
an unblocked ray with 64 depth slices starts sampling at 46.875 world units.
That spacing follows the camera far plane rather than the physical cloud.
`froxel_coord` also clamps arbitrary continuation points to camera-field cells.

A CPU reconstruction used all 622 active froxel centres, the captured camera,
tracked blue-noise ranks, the shader's HG direction sampling, and the unmodified
historical bilinear terrain/trilinear density helpers. It found:

- 467 continuation rays did not hit terrain.
- 84.80% of those rays place their first old sample beyond the medium exit.
- 37.37% of bounded samples with actual medium density lie outside the visible
  camera field.
- Mean unblocked density integral: 6.16762 with the old camera lookup and
  8.38666 with a world-density query over the actual medium interval.

These are float64 CPU diagnostics over all active froxels, including froxels
behind the primary surface. They are not GPU measurements or proof that the
candidate passes a visual gate. The [JSON](nephele-continuation-2026-10-21.json)
records the settings, output, and candidate source hashes.

## Smallest correction prepared

The froxel shader intersects continuation segments with the actual medium box
and places the existing `media.grid.z` midpoint samples within that interval.
The CPU comparison above uses the same existing 64 samples on both sides.
This keeps the existing continuation work-count contract; it adds no iteration
budget or cap. It samples extinction from the full world-density field and gathered sun
radiance from the full world-sun field. Outside the physical box, both are zero.
The existing filtered environment cache is converted to a source per density
and applied at the actual sample density. Its camera-direction approximation
remains; this change does not establish a complete angular transport solution.
The unbounded homogeneous path retains its existing integration scheme.

Rust binds the two existing world textures to injection. The frame graph declares
those reads. Its current `SampledTexture` classification does not express
storage-write usage (`src/core/framegraph_impl/barriers.rs`); wgpu derives actual
texture usage transitions from the encoded passes. NVIDIA execution still has
to prove that the new bindings and execution are valid.
Milos approved the exact corresponding graph-test update in this release chat
on 2026-10-21 ("Approved"); the approved assertions are applied below.
No new scattering orders, fitted gains, ray-length
caps, iteration caps, reference edits, fixture edits, masks, thresholds,
goldens, or terrain source pins have been added. The graph assertions below are
the only test changes in this correction.

## Validation so far

- Naga parsing and validation passed for all five froxel shader entry points.
- `cargo fmt --check` passed after the final graph-read declaration and approved
  graph-test update.
- f32 inventory is unchanged: 1931 sites,
  `3d7e2b08e7e456a558dbed3817c47effd05804eac820df704a3b84f0cf504dee`.
  All 32 existing f32 contract tests passed in 53.38 seconds with
  `FORGE3D_NO_BOOTSTRAP=1 FORGE3D_TEST_INSTALLED_WHEEL=1`. These source-inventory
  checks do not claim execution of a rebuilt native extension.
- The existing Rust shader/Naga unit test passed (1 test) with
  `extension-module,default,cog_streaming,shader-contract-asserts,splat-fusion`.
  Of the three existing media graph tests, two passed and the approved barrier
  assertion failed: `barriers_before("nephele.media.inject")` does not include
  `world_sun`. The failure is recorded in
  `.release-evidence/2026-10-21/continuation-graph-tests.log`; it needs diagnosis
  before this candidate can be committed as validated.
- The prior implementation's installed-wheel CPU fast lane passed: **781
  passed, zero skipped, two deselected**, in 1105.35 seconds, with
  `FORGE3D_NO_BOOTSTRAP=1 FORGE3D_TEST_INSTALLED_WHEEL=1`. The deselections
  were the profile's existing slow shader-construction test and the exact
  owner-approved GPU-device memory case. The complete lane remains in CI.
  This result covers the clean `d5507249` wheel, not this uncommitted correction.
- The earlier shader-unit and queued Clippy processes ended without completion
  evidence. Both were rerun: the shader-unit check passed and
  `cargo forge3d-clippy-acceptance` passed (5m 07s, including the build-lock wait).
  All Cargo checks used
  `CARGO_TARGET_DIR=D:/forge3d/.worktrees/pr172-sync-main/target`.
- `git diff --check` passed for the candidate source and audit Markdown; all
  three normalized source hashes match the audit JSON.

The next necessary proof is CPU validation of this candidate, followed by a
new exact-head NVIDIA capture. Full acceptance and release remain blocked by
the failed physical measurements and unresolved G4b replacement policy.

## Exact graph-test update (owner-approved and applied)

In `media_aov_injection_reuses_forward_shadow_transition_and_order_stays_strict`,
replace the old assertion that injection has no barriers with:

```rust
let inject_barriers = g.plan.barriers_before("nephele.media.inject");
assert!(inject_barriers.iter().any(|barrier| barrier.resource == media.world_sun));
assert!(inject_barriers.iter().all(|barrier| barrier.resource != g.handles.shadow));
```

After the existing `let inject = ...` declaration, add:

```rust
assert!(inject.reads.contains(&media.world_medium));
assert!(inject.reads.contains(&media.world_sun));
```

All existing ordering, shadow-transition, resource-use, and rejected-out-of-order
execution checks stay intact. The new assertions cover the new sampled reads
and retain the intended absence of a repeated shadow transition. The existing
graph test rejects the new barrier expectation in the freshly built CPU test
binary; no pass is claimed.
Approval source: Milos's explicit "Approved" response to this exact proposal in
the release chat on 2026-10-21.

The failure is an incorrect proposed assertion: the graph's sampled-texture
usage classification cannot produce that barrier. A corrected proposal is
pending with Milos: retain the original empty-barrier assertion and add only
the two explicit world-resource read assertions. Changing the framegraph's
resource-type model would expand this transport correction into a separate
subsystem; no such change has been made.

## Concurrent branch change

The remote branch advanced independently from `a8a96c47` to `eb312522` while
this candidate was prepared. That commit replaces the acceptance scripts and
removes the 320-spp controls. This conflicts with the release task's explicit
instruction to preserve those controls. Milos's choice of acceptance baseline
is pending before integration or push; no concurrent changes were overwritten.

## Resolved owner decisions

On 2026-10-21 Milos explicitly approved both decisions in this release chat:
"continue on top of eb312522 and restore the original empty-barrier assertion
while keeping both new resource-read checks. Proceed with validation and
publishing." This supersedes the pending decisions above. The original
empty-barrier assertion has been restored; both world-resource read checks
remain. Integration will preserve the approved acceptance refactor, including
its removal of the old scripts and 320-spp controls.

The previous proposed barrier assertion was incorrect because this framegraph
does not model storage writes for `SampledTexture`. A future documentation
note at the framegraph boundary should explain that limitation before tests
assert explicit barriers for sampled-texture resources. This is a proposed
documentation improvement, not an applied agent-contract change.

## Approved G4b replacement

Milos explicitly selected "Approve +/-20% relative error" in response to the
numeric policy proposal on 2026-10-21. The policy and matching pass/rejection
tests are authorized. `scripts/nephele_acceptance.py::gate4` now compares the
signed mean linear luminance response over the frozen god-ray ROI with the
accepted first-order reference response `0.03731673831968643`:

```
relative_error = abs(observed_response - reference_response) / reference_response
pass = isfinite(relative_error) and relative_error <= 0.20
```

The original numerical inverse sRGB + ACES mapping is reused, including its
existing sample domain and resolution. This does not establish a new
path-traced ablation reference. Policy provenance identifies the accepted
measurement's Git revision, path, raw-blob SHA256 and ROI-mask SHA256. SSIM
remains a diagnostic. G4a and the medium-ablation check are unchanged.

The synthetic unit ablation now represents the accepted response. Additional
cases reject no response, reversed response, and responses on both sides of
the approved band. The physical suite still executes the same six named
cases, including the unchanged `gate4_terrain_coupling` verdict assertion.

The first post-refactor Python run passed two producer tests, but 11 cases
failed setup because pytest could not create its sandbox temporary directory.
The affected cases are being rerun with a fresh explicit temporary directory
inside this worktree. No test or gate was skipped or weakened.

The owner-approved policy test suite passed all 13 tests in 13.45 seconds
with an explicit task-owned pytest temporary directory. Re-evaluating the
actual d5507249 capture confirms the accepted-reference response is
0.018602208951602937 and the relative error is 50.1505%, so the new policy
correctly rejects it. Earlier candidate DeltaE percentages were corrected
after fixing the diagnostic helper's RGB-versus-Lab input mismatch; production
gates and thresholds were already using Lab correctly.

The post-refactor source inventory also passed all 32 existing f32 tests in
64.09 seconds. These source-only policy/inventory checks used the previously
built d5507249 wheel and do not claim native execution of this candidate.

The post-refactor sandboxed graph/Clippy attempt did not establish a result.
The graph build reported incremental-cache hard-link failure and copying; the
Python attempt had also established a sandbox temporary-directory failure.
Only the two validation process trees whose shell log signatures matched this
task were stopped. The same CPU graph tests and acceptance Clippy are being
rerun outside that sandbox with the same isolated CARGO_TARGET_DIR. No source
or test was changed to bypass the environment problem.

The corrected graph suite passed all three tests, zero failed or ignored,
in the host retry (24m 17s build). Features were
`extension-module,default,cog_streaming,shader-contract-asserts,splat-fusion`;
the approved read assertions and original empty-barrier assertion all passed.
Acceptance Clippy is now running after that successful graph check.

Acceptance Clippy passed in 2m 13s on the integrated candidate. The retained
shader/call-graph unit test also passed in the freshly built test executable
(1 passed, zero failed or ignored, 0.13 seconds). CPU validation now supports
committing/pushing this candidate for exact-head NVIDIA measurement. Final
head-wheel fast/verify and full acceptance still remain; no physical pass or
release is claimed.
