# NEPHELE packed-state confirmation — 2026-10-23

The exact patch approved by Milos with **“Approved”** was committed as `95b66c50566e38006aa2e7ac5ecb2065e9253731`, fast-forward pushed to PR #172, and exercised by [run 40023229060](https://github.com/milos-agathon/forge3d/actions/runs/40023229060). The run failed after convergence because the AOV renderer calls passes in a different order from its compiled directional graph. No further production edits or CI dispatches followed.

## Result

On NVIDIA GeForce RTX 3070, Vulkan, driver 610.60, DiscreteGpu, software fallback false, the loop terminated at order **22**, elapsed **1,380 ms**, final progress **[0, 931111685, 956246357, 0]**. All waits returned in 56–64 ms; all receives returned within the logged millisecond. No order cap was introduced. This is loop timing, not completed-frame or DEM timing. The nonzero floating residual fields are not the approved packed-bit stopping predicate.

Before this fix, the trace continued past 3,283 orders in 181.419 seconds and was cancelled ([run 39990155173](https://github.com/milos-agathon/forge3d/actions/runs/39990155173)). The subsequent stored-bit proof is in [run 39998974173](https://github.com/milos-agathon/forge3d/actions/runs/39998974173).

The installed-wheel fast JUnit reports **661 passed, 0 failed, 0 errors, 0 skipped**. `test_600mib_host_visible_raises_named_budget_error` passed in 0.001 s. `tests/test_no_silent_degradation.py` and CI configuration are unchanged by this commit.

The trace then failed with:

`renderer pass order mismatch: expected "nephele.media.inject", got "terrain.forward_aov"`

Directional mode makes forward read the radiance provider (`render_graph.rs:448`); injection also reads it (`:455`). The compiled plan may therefore put injection first. AOV execution still calls forward at `aov.rs:1217` and injection at `:1267`. The strict check rejects that order. These files are unchanged from the parent commit. Existing directional graph tests inspect resource declarations; existing AOV order tests use directional=False. All four existing terrain/media graph tests pass.

Proposed, **not applied**: align AOV host execution with the compiled directional pass order, preserving the strict checks and accurate resource declarations.

## Image metrics

No image was produced. The forecast remains numerically untested; there is no measured image contradiction to explain or refit. GPU convergence took 22 orders versus the CPU model’s 19; neither is a fixed runtime limit.

| Metric | Last physical image (68cc04e9) | Qualified CPU forecast | New physical image |
|---|---:|---:|---|
| B1 (>=0.95) | 0.712766, fail | 1.000000 | ABSENT |
| SSIM (>=0.95) | 0.991600, pass | 0.992349 | ABSENT |
| G4a (>=0.95) | 0.710983, fail | 0.996146 | ABSENT |
| Medium changed fraction (>=0.10) | 0.321566, pass | 0.379974 | ABSENT |
| G4b relative error (<=0.20) | 0.102739, pass | 0.007617 | ABSENT |

The prior physical image is from [run 39580065706](https://github.com/milos-agathon/forge3d/actions/runs/39580065706). Gate 1/2 CPU rescoring of the new artifact passes: heterogeneous relative error 0.000202857, no majorant violations; energy residual 0.000316700. All 356 static Naga assemblies validate, with zero unresolved expressions and compute comparison-sampling calls. Complete Gate 5 also needs an uncaptured termination slice. Gate 6 and the six-case physical JUnit are ABSENT.

## Changes and approval

Only two production files changed: `nephele_realtime_terrain_trace.wgsl` packs source/surface values once and compares the stored u32 words; `media.rs` changes the two source textures to Rg32Uint. Each texture remains 8 bytes per texel. The exact approved patch SHA-256 is `166062a22b8a30c8e08998a8fa4d4d963d03b4d51be7a49e892a7bb49f516863`.

Pins, contracts, 1,936-site f32 inventory/ledger, tests/assertions, thresholds, goldens and committed certificates are unchanged. Required protected-path diffs are empty. The source worktree is clean. This note, JSON and CPU verifier are saved locally as ignored audit evidence, not added to the two-file commit.

## Checks

- `cargo fmt --check`: pass, 4.390 s.
- `cargo forge3d-clippy-acceptance`: pass, 303.212 s; exact alias/feature list recorded in JSON. Required commit-hook lint: pass, 170.583 s.
- Correct shared-media shader test: 1 passed, 1.840 s. Core frame graph: 4 passed, 2.750 s. Terrain/media graph: 4 passed, 0.123 s. Features: extension-module,default,cog_streaming,shader-contract-asserts,splat-fusion. The initial wrong module filter ran zero tests and is rejected as proof.
- World-coordinate suite: 37 passed, 130.710 s. Its native camera case was separately rechecked against the verified new wheel: 1 passed, 6.177 s. Inventory and ledger unchanged.
- Worktree-target wheel: build passed in 695.254 s; embeds exact HEAD. Installed to an explicit worktree directory and imported native SHA-256 matches the wheel. A log-printing encoding failure after the successful build and a sandbox-temp install failure were corrected without rebuilding or changing dependencies.
- `test_nephele_acceptance.py`, `test_nephele_fixture_contracts.py`, `test_nephele_public_api.py`: 30 passed, 13.190 s, exact new wheel. FORGE3D_NO_BOOTSTRAP=1, FORGE3D_TEST_INSTALLED_WHEEL=1; PYTHONPATH points to the isolated installed wheel. Fresh pytest basetemps used.
- NVIDIA installed fast: 661/0/0/0, with FORGE3D_ALLOW_HOSTED_WINDOWS_TERRAIN=1, FORGE3D_NO_BOOTSTRAP=1, FORGE3D_TEST_INSTALLED_WHEEL=1, Vulkan backend. Hosted Fast Contract and CI Windows wheel jobs pass.
- CPU verifier reopens official artifact 12531439533 (41,587,262 bytes), checks exact head/adapter/JUnit/trace and CPU Gate 1/2 evidence. JSON records hashes and all commands, durations, features and environment values.

Both GPU environment cases, six physical cases, complete Gate 5/6, 1500² DEM timing and full CI were not executed after the trace failed. D3D12, Metal and WASM are ABSENT. No merge, dry run, tag or publication occurred. Release remains blocked by physical green and the realistic-DEM measurement.

Residual risk: the new image and disabled ablation arm remain unmeasured, as do real-DEM memory/convergence/timeouts and latency. The known reference RNG-dimension overlap remains unchanged. No image gains, fixture branches, caps, test deselections or weaker checks were introduced.

Proposed steering changes, **not applied**: record the actual shared-media/graph test module paths; classify the world-coordinate suite’s one native case; use UTF-8 plus explicit writable temp/install directories in Windows evidence scripts. The analysis venv inherits global packages, so use the JSON’s explicit PYTHONPATH for this installed wheel.

Reproduce from the release worktree (CPU only):

`analysis-venv/Scripts/python.exe docs/audits/nephele-packed-state-2026-10-23.py --artifact .release-evidence/2026-10-23/ci-40023229060 --output <local-report.json>`

Use the actual interpreter at `.release-evidence/2026-10-22/analysis-venv/Scripts/python.exe`. No GPU objects are constructed by this verifier.
