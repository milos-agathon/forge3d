# NEPHELE cloud-light candidate — 2026-10-21

**Local candidate; physical acceptance NOT_PROVEN.** No new GPU render, B1/G4
measurement, head-wheel fast lane, full CI, merge, tag, or publication is claimed.

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
does not count the new cloud-light kernels. Actual GPU cost and visual quality
remain unmeasured. In particular, adding the saved terrain term alone predicted
only 62.17% G4a; the multiple-scatter direction correction needs physical proof.

Implementation is in `src/terrain/renderer/media.rs`,
`src/shaders/nephele_froxel.wgsl`,
`src/shaders/nephele_realtime_terrain_trace.wgsl`, and
`src/shaders/terrain_pbr_pom.wgsl`, with the matching frame-graph, binding, and
viewer integration. The [candidate manifest](nephele-cloudlight-candidate-2026-10-21.json)
records SHA-256 hashes of the Git blob bytes and approved pin/ledger values.

## Owner approval and applied integration updates

Milos approved this exact package on 2026-10-21 with the second “All approved”
reply. Items 1–4 are applied; item 5 follows the required CPU validation:

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
The head-wheel build and Python fast lane follow this implementation commit.
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
