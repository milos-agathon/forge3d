# NEPHELE certificate repair and release preparation — 2026-10-24

This preparation starts at `024c7803d17814c447637208cb4535112c9ac166` on PR #172. Fetch confirmed that `origin/main` (`d6b9e556586cceddb9c08f16ffc35f7d71fb4d88`) is already an ancestor; no additional merge was needed. Merge, tagging and publishing remain unauthorized at this stage.

The real-DEM certificates at the base recorded nine `shader_provenance_missing` degradations on the first frame and three on the unchanged frame. NEPHELE registered modules as `nephele.media.froxel.shader`, `nephele.media.terrain_trace.shader` and `nephele.media.tonemap.shader`, but recorded pipeline labels as their identities. The execution calls now record the corresponding module labels, including the legacy terrain trace. All production NEPHELE modules already use `create_labeled_shader_module`; no shader source or new registration is needed. The unused per-dispatch pipeline-label string is removed.

The new CPU Rust test compares recorded labels with module-registration labels in the production sources. It normalizes line endings before finding the test-module boundary, and also passed when compiled from actual CRLF source, matching a Windows checkout. The real-DEM producer now raises an error if either timed frame has any certificate degradation. A CPU check confirms that it accepts an empty degradation list and rejects a nonempty list for both frame states. This is stronger evidence collection, not a change to an acceptance gate or threshold.

## Owner decision and release notes

Milos's exact words: “Accept and document it: merge with the slowdown stated in the CHANGELOG. I'll then write the remaining merge-prep instructions: the certificate fix, CHANGELOG, roadmap, full CI, and a stop before merge.” The subsequent detailed instructions authorize this preparation and require another stop before merge.

CHANGELOG dates 1.46.0 to 2026-10-24 and removes the release-pending block. It describes the directional stored-f16 fixed point without an order cap, packed-bit checks, compiled AOV pass order, external media cache identities and shader provenance. It records the approved 2026-10-22 open-surface G4b target `0.02187741965233074` ±20%.

The accepted observations remain explicit: original 1500×1500 Gore Range DEM, 1920×1080 viewport, RTX 3070/Vulkan, 76.151 s first completed frame and 328.481 ms unchanged frame, including output readback. The full directional solve reruns when the camera changes and on every frame with a live rendered sky. These are two observations, not averages, percentiles or a speedup claim. The NEPHELE roadmap node was updated locally; main coverage remains none until integration. No other main-checkout item was edited.

## CPU evidence at preparation

- New label regression: one test passed with `extension-module,default,cog_streaming,shader-contract-asserts,splat-fusion`.
- Existing shader registry: seven CPU tests passed; media presentation: three CPU tests passed with the same features.
- `cargo fmt --check` and `cargo forge3d-clippy-acceptance` passed. Exact durations and input environment are in the accompanying JSON.
- All 356 shader assemblies pass Naga; zero unresolved sources and compute comparison-sampling calls. Terrain/hybrid pins match; the trace assembly remains `0xcf62804a9054a070` (SHA-256 `e381395ec6f91cacbbad9ff193aece9c6c698fb1d8bbbdbeebd6d6bbfd7c7025`).
- The f32 inventory remains 1,936 sites / `8f9f378ce8e27a98e9b99961073ae70128584485778adf1dc2c597c464c0cee7`. Pins, ledgers, committed certificates/goldens, tests, fixture, policy, workflow and gate functions are unchanged. No protected-value approval batch is required.

The exact-head wheel build, installed-wheel CPU suites, `scope=nephele` and then `scope=full` are the next validation steps; this committed preparation snapshot makes no claim that they have already passed. Their official CI artifacts and the final report supply the result. The accompanying CPU verifier requires all six gates, physical 6/0/0, fast at least 661/0/0, environment 10/0/0, exact revision/adapter, clean timed certificates and converged cached DEM transport. Run it with `--artifact`, `--head` and `--output`; it never executes GPU work.

Historical 2026-10-23/24 notes, JSON, patches and reproducers are archived with this change. The proposed memory-test exception is explicitly marked rejected and never applied; current CI runs the named budget case. Earlier audit outcomes retain their original revision and failure status. `.release-evidence/` remains uncommitted.

## Limits and reusable steering

No local GPU execution occurred. D3D12, Metal and WASM are ABSENT. The reference sampler RNG-dimension overlap is unchanged. Timing and fixed-point convergence still require the physical lane; module-label bookkeeping does not predict a transport change, so the measured base image metrics are expected to remain identical.

Proposed guidance, not applied: use worktree-owned temporary directories for Windows CPU tools, and pass Naga shader paths relative to the helper's working directory to avoid the Windows command-length contract. The existing validation wrapper does exactly this while preserving all 356 inputs and all analyzer checks. Source scanners can misread Rust lifetime/character syntax; verify the exact site inventory and test-module exclusion before proposing a ledger change. During preparation, sandbox access, an overlong command and test-only syntax errors were corrected; none counts as successful validation. No agent instruction or skill was changed.
