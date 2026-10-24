# NEPHELE cloud-light candidate — 2026-10-21

Historical raw evidence removed by the trim remains available at `a8a96c47`. Use `git show a8a96c47:docs/audits/nephele-cloudlight-candidate-2026-10-21.md` to retrieve this summary and follow its evidence paths at that revision.

**NVIDIA physical acceptance FAILED; not release ready.** Implementation
`d5507249a2327d183edf0bb3ff11dcb83a773333` was committed, pushed, and physically
captured. Full acceptance, merge, tags, and publication have not been performed.

Milos's message “All approved” approved the preceding requests to push the
measurement audit and expand the physical-lighting diagnosis. Audit commit
`38a30db0f0ff829225a28064253862ad6db3db4e` was pushed to PR #172's branch
`codex/20-nephele`. The implementation below is based on that commit. Milos's second message
“All approved” approved the exact pin, shader-contract, f32-ledger, and existing
test changes listed below, plus committing/pushing the candidate and dispatching
the NVIDIA NEPHELE lane after CPU validation. Those protected updates have now
been applied. No subagents were used.

## Candidate behavior

- Store a phase-independent sun source separately from the camera-direction
  single-scatter source. During multiple scattering, evaluate the gathered
  sunlight for the continuation ray's direction. Preserve the existing spatial
  filter when separating and recombining the two source components.
- Upload a field covering the complete bounded medium, using the existing live
  depth resolution (64 by default) on each world-space axis. Canonical medium
  evaluation supplies physical density and full-domain sun transmittance.
  Production GPU terrain traversal supplies sun visibility.
- Compute sunlight scattered by the medium onto each visible terrain hit with
  the selected 192-direction Fibonacci quadrature. The march spacing follows
  the canonical integrator's half-cell rule; ray length comes from the actual
  medium and terrain intersections. No gain is fitted to the reference frame.
- Store surface irradiance, then apply each pixel's actual material albedo
  divided by pi before exposure and camera-medium transport. Offscreen and
  viewer paths use the field. The terrain-occlusion ablation also controls the
  new medium light-path visibility.
- Track the added textures and uniform buffer. The offscreen frame graph has
  an explicit cloud-light pass before forward shading. Cloud-light and froxel
  passes have separate uniform buffers, so queued writes cannot overwrite an
  earlier pass's inputs before submission.

The existing unbounded homogeneous transport path is retained. The new surface
sun term is evaluated for bounded heterogeneous media. The legacy diagnostic
`terrain_trace_queries` still counts the froxel sun/continuation queries; it
does not count the new cloud-light kernels. Total GPU cost was not measured.
The physical capture below failed acceptance. Adding the saved terrain term
alone had predicted only 62.17% G4a; the combined candidate measured 67.31%.

Implementation is in `src/terrain/renderer/media.rs`,
`src/shaders/nephele_froxel.wgsl`,
`src/shaders/nephele_realtime_terrain_trace.wgsl`, and
`src/shaders/terrain_pbr_pom.wgsl`, with the matching frame-graph, binding, and
viewer integration. The [candidate manifest](nephele-cloudlight-candidate-2026-10-21.json)
records SHA-256 hashes of the Git blob bytes and approved pin/ledger values.

## Owner approval and applied integration updates

Milos approved this exact package on 2026-10-21 with the second “All approved”
reply. Items 1–5 were completed, including the approved NVIDIA dispatch:

1. Update `PINNED_TERRAIN_SOURCE_HASH` in `src/verify/ir/engine.rs` from
   `0xffa4448a20d6e60a` to **`0x895bb1d3845497e1`**. The candidate includes six
   rounding barriers inserted by the repository determinism rewriter.
2. Add the new 2D irradiance texture to
   `shaders/contracts/terrain_pbr_pom.toml`, with finite binary16 range
   `0..65504` (the storage format's maximum finite value) and its actual width
   and height. Retain the existing proof inputs, invariants, and inactive-media
   proof domain. This does not certify active media or physical render quality.
3. Record the two new integer-grid-to-f32 casts in `upload_world_medium` in
   `tests/data/world_coord_f32_ledger.json` / `nephele_main_sync`, and update the
   inventory constants in `tests/test_world_coord_f32_gate.py`: **1929 → 1931**,
   digest **`3d7e2b08e7e456a558dbed3817c47effd05804eac820df704a3b84f0cf504dee`**.
   Both casts normalize integer sample indices; neither narrows an absolute
   world coordinate. Preserve the recorded main-side and NEPHELE-side base
   inventories when extending the merged result.
4. Update the adapter source test's exact terrain-query expression count
   **2 → 5**; bind the added texture at binding 4 in the existing GPU layout
   test; add cloud-light-before-forward assertions to the existing graph-order
   tests and execute the new prepass before the shadow pass in their manual
   execution sequence; update the AOV graph's exact debug assertion **8 → 9 passes**.
   Keep all existing assertions and pass/fail requirements.
5. After those changes and CPU validation, commit/push this candidate to
   `codex/20-nephele` and dispatch the already requested NVIDIA NEPHELE lane.

G4b additionally needs an owner-specified **numeric percentage** for
`abs(delta_realtime - delta_reference) / abs(delta_reference)`, using mean
linear luminance change in the existing ROI. Milos explicitly accepted the proposed
first-order reference on 2026-10-21 (“Accept proposed reference”):
0.03731673831968643. It is an estimate, not a regenerated path-traced ablation.
No numeric tolerance has been supplied, and none is invented.
The gate, policy, and Python gate tests remain unchanged.

## Observed CPU evidence

All local Cargo commands use
`CARGO_TARGET_DIR=D:/forge3d/.worktrees/pr172-sync-main/target`.
A dev-profile abi3 Windows wheel was built from a clean checkout of the exact
implementation commit, with a separate worktree-specific target directory.
It installed and imported as 1.46.0 in an isolated Python 3.13.14 environment.
The native build output confirms the embedded full source revision.
With `FORGE3D_NO_BOOTSTRAP=1 FORGE3D_TEST_INSTALLED_WHEEL=1`, focused
`test_world_coord_f32_gate.py` and `test_pipeline_validation_gate.py` checks
passed: **33 passed, zero skipped**. This is not the entire local fast lane.
The complete hosted Fast Contract and PR Core Success passed on the same head.
The fast lane includes a GPU-device memory test; a CPU-only local selection
needs the owner's requested exception, with the complete lane retained in CI.

- `cargo fmt --check`: passed.
- `cargo forge3d-clippy-acceptance`: passed on the final local candidate before
  the proposed pin/ledger/test changes, including the AOV order correction,
  using the worktree-specific target dir. The alias also passed after the
  approved pin/contract/test updates (`cloudlight-clippy-approved.log`).
- Naga parsing and validation: passed for the fully assembled terrain, hybrid
  kernel plus terrain/cloud-light adapter, viewer shader, and froxel shader.
  The terrain assembly was validated again after the six rounding edits.
- `FORGE3D_DET_REWRITE=1 cargo test --lib --features
  extension-module,default,cog_streaming,shader-contract-asserts,splat-fusion
  det_instrument_rewrite -- --nocapture`: one test passed, six edits applied.
- The initial direction-correction candidate's existing shader test passed
  with the renderer-enabled feature set (one test executed). That earlier
  result is not reused as proof of the later cloud-light additions.
- An earlier attempt selected zero tests with the non-extension CI feature
  set and an incorrect full test name. It supplies no shader-validation proof.
- Existing CPU graph tests: all three passed after applying the approved
  prepass assertions and manual execution sequence. The test still rejects
  skipping the forward pass. The AOV caller executes cloud light between
  prepare and shadow, matching the graph planner.
- Existing shader source/call-graph test: one test passed after the approved
  exact terrain-query count update (2 → 5).
- The f32 inventory contains exactly 1931 sites with the approved digest;
  both recorded historical branch inventories restore exactly.
- The earlier full `verify::` CPU run used CI features plus `extension-module`
  and finished with 103 passed, 5 failed, and 1 already ignored. The failures
  included the old pin and terrain proof tests. That failed run is not release
  evidence. The fresh run with the approved pin and contract passed:
  **108 passed, 0 failed, 1 existing ignored test**, in 685.85 seconds. The
  ignored test is `verify::ir::tests::inspect_deterministic_kernel_ir_shape`.
  Features matched the CI portable matrix plus `extension-module`; no GPU
  tests were selected by the `verify::` filter.

Full logs are local under `.release-evidence/2026-10-21/`, including
`cloudlight-clippy-final-candidate.log`, `cloudlight-naga.log`,
`cloudlight-clippy-after-order-fix.log`, `cloudlight-naga-after-rewrite.log`,
`cloudlight-graph-tests.log`, and `cloudlight-det-rewrite.log`.
The scratch artifacts remain unstaged.

The frozen reference sources, fixture inputs, masks, reference sample counts,
goldens, thresholds, dependencies, and release versions were not edited.
Pre-existing deleted assets in this worktree were preserved and remain unstaged.

Reusable steering discovered: Windows' default text encoding cannot read every
WGSL source, so diagnostic assembly tools must explicitly use UTF-8. The f32
inventory's lexical scanner is sensitive to an added explicit Rust lifetime in
a closure argument; using the equivalent inferred slice type kept the actual
cast delta to the two grid-index conversions. A standalone Naga diagnostic on
Windows needed a normal Rust worker thread instead of the smaller platform main
thread stack. These are proposals for future tool/documentation guidance; no
agent contract, skill, or inventory scanner was modified.

## NVIDIA capture result

Run [39545456967](https://github.com/milos-agathon/forge3d/actions/runs/39545456967)
used NVIDIA GeForce RTX 3070 / Vulkan / driver 610.60. Producers and both
independent captures completed. The shared physical-case setup rejected Gate 3;
all six cases therefore errored in setup, with zero skips. No physical gate-case
pass or Full Acceptance Summary pass is claimed.

| Measurement | Earlier capture | Candidate | Required |
| --- | ---: | ---: | ---: |
| B1 sky fraction with DeltaE2000 < 2.5 | 95.5871% | 94.1686% | >= 95% |
| Godray SSIM against reference | 0.982692 | 0.989971 | > 0.95 |
| G4a shadowed-terrain fraction with DeltaE2000 < 2 | 59.6660% | 67.3089% | >= 95% |
| G4b realtime ablation SSIM | 0.989950 | 0.987554 | < 0.80 (unchanged gate) |
| Realtime ablation / accepted first-order reference response | 45.8457% | 49.8495% | Owner tolerance unset |

The candidate's mean linear luminance response is 0.018602208951602937, versus
accepted reference estimate 0.03731673831968643 (relative error
0.5015049602609736). These G4 values are diagnostics calculated from the raw
capture; the production verifier stopped at Gate 3. All 34 shared frozen input
files, optical-depth output, and transmittance output are byte-identical to the
earlier artifact. The mean shadowed-terrain Lab error is
`[0.538553, -0.128885, -1.131776]`.

The [measurement tool](nephele-cloudlight-capture-2026-10-21.py) reuses the
already accepted first-order response rather than repeating that CPU proof.
Its [output JSON](nephele-cloudlight-capture-2026-10-21.json) includes the exact
input hashes. Reproduce with the two downloaded artifacts:

```powershell
python docs/audits/nephele-cloudlight-capture-2026-10-21.py --artifact-dir .release-evidence/2026-10-21/ci-39545456967 --baseline-artifact-dir .release-evidence/2026-10-21/ci-39517423981 --output docs/audits/nephele-cloudlight-capture-2026-10-21.json
```

The phase sign agrees with the historical first-order tool. The sky camera
in-scattering mean RGB fell from `[0.282403, 0.258618, 0.227771]` to
`[0.278813, 0.255469, 0.225203]`; the reference is
`[0.298821, 0.270100, 0.233472]`. The direction correction is consequently not
a sufficient transport fix. No gain or ROI-specific correction was added.

Two remaining structural questions are visible in the existing path:
`froxel_coord` clamps arbitrary world points to camera-field cells, including
continuation rays, and `encode_inject` dispatches a single multiple-scatter pass
that gathers the single-scatter field. Their contribution to the remaining
error has not been isolated. Replacing that continuation transport or adding
higher scattering orders would materially expand beyond this approved
candidate. A bounded investigation was proposed to Milos; no further transport
implementation, invented iteration cap, or new pin/test/gate change was made.
The separate G4b tolerance and local fast-lane GPU-device-test decisions remain
pending. Full CI was not started on this known failing physical candidate.

## Further owner decisions

Milos answered “Yes to both” after the explicit questions about (1) investigating
continuation rays outside the camera froxel field and missing higher scattering
orders, and (2) omitting only
`test_600mib_host_visible_raises_named_budget_error` from the local CPU fast lane
while retaining the complete lane in CI. Both are approved. This authorizes the
bounded transport investigation and that exact local test selection; it does
not establish a new G4b relative-error percentage. The existing SSIM criterion
is preserved while the transport investigation proceeds.

## Diagnostic correction

The first capture diagnostic passed uint8 RGB directly to the CIELAB-only
`delta_e_2000` helper. It reported incorrect candidate B1/G4a percentages.
The tool now converts sRGB to Lab before evaluating DeltaE2000, matching the
production gate. The corrected candidate values are B1 94.1686% and G4a
67.3089%; both still fail 95%. The baseline values, SSIM and linear ablation
response are unaffected. The capture JSON has been regenerated. Its accepted
reference now comes from the owner-approved policy because the approved
refactor removed the old audit measurement from the working tree; the policy
records that historical measurement by immutable Git revision and blob hash.
