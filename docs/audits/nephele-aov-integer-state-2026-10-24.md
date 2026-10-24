The two requested fixes are committed together as `bbda8eeb0f77f386126a15edd1268b74f48f537b`, fast-forward pushed to PR #172, and exercised by [NEPHELE run 40031822131](https://github.com/milos-agathon/forge3d/actions/runs/40031822131). Both fixes work. The run finishes with **5 physical tests passed, 1 failed, 0 errors, 0 skipped** because Gate 6 compares a volatile wall-clock field. Release is not ready; the 1500-square DEM measurement is absent.

Milos authorized: **"Two fixes, one commit, then dispatch scope=nephele with the trace on."** No pin, contract, ledger, golden, certificate or existing assertion value changed, so the conditional approval stop did not trigger. The source tree and index are clean. This note, JSON and CPU reproducer are ignored local audit evidence, not an additional commit.

Changed production files and reasons:

- `src/terrain/renderer/aov.rs`: the variable forward/media segment follows `ordered_passes()`. Existing depth/background, forward/media and media/resolve encoder splits remain. The shadow guard carries a transition observed at an earlier injection pass to the later forward reader; a missing transition still errors. No duplicate transition is required for a later read.
- `src/terrain/renderer/render_graph.rs`: correct the two stale provider/order comments; add `directional_aov_execution_follows_the_compiled_plan`. It checks directional injection-before-forward, exact full-plan execution, and preservation of the shadow guard. Existing assertions are unchanged.
- `src/shaders/nephele_realtime_terrain_trace.wgsl`: extract RGB f16 fields with integer masks/shifts; normalize negative zero for checks; reject fields above 0x7bff and unsigned decreases. Compute the maximum difference exactly in integer units of 2^-24 using low/high u32 words, then encode the existing f32 diagnostic bits with nearest-even rounding. No float pack/unpack can bypass those checks. Packed-word equality remains the stopping rule, with no cap. Transport, source packing/stores, quadrature and resources are unchanged.

The physical runner is NVIDIA GeForce RTX 3070, Vulkan, driver 610.60, DiscreteGpu, software fallback false. Checkout and both capture diagnostics name the exact commit. The trace terminates at **order 22**, **1,380 ms**, **progress [0,0,0,0]**. Both independent captures report orders=22, converged=true, maximum_source_change=0.0 and maximum_surface_change=0.0. Loop duration is not completed-frame or real-DEM timing. Before these fixes the loop stopped at 22 with nonzero false residuals and then hit the AOV order error; that trace is [run 40023229060](https://github.com/milos-agathon/forge3d/actions/runs/40023229060).

The NVIDIA fast JUnit is **661 passed, 0 failed, 0 errors, 0 skipped**. The 600 MiB named-budget-error test passed in 0.001 s. Its source and `tests/test_no_silent_degradation.py` are unchanged. Zero-skip environment: FORGE3D_ALLOW_HOSTED_WINDOWS_TERRAIN=1, FORGE3D_NO_BOOTSTRAP=1, FORGE3D_TEST_INSTALLED_WHEEL=1, WGPU_BACKEND=vulkan, WGPU_BACKENDS=vulkan. The direct capture additionally has FORGE3D_NEPHELE_TRACE=1.

Both GPU environment cases pass: `test_camera_visible_uniform_environment_readback` (4.878 s) and `test_shadow_scattering_component_readback` (9.799 s). The shader-contract/environment step has **10 passed, 0 failed, 0 skipped**.

| Physical image criterion | Last image, 68cc04e9 | Qualified forecast | Measured bbda8eeb | Passing margin |
|---|---:|---:|---:|---:|
| B1 >=0.95 | 0.712766, fail | 1.000000 | 1.000000 | +0.050000 |
| SSIM >0.95 | 0.991600 | 0.992349 | 0.992584 | +0.042584 |
| G4a >=0.95 | 0.710983, fail | 0.996146 | 0.997431 | +0.047431 |
| Medium fraction >=0.10 | 0.321566 | 0.379974 | 0.387035 | +0.287035 |
| G4b relative error <=0.20 | 0.102739 | 0.007617 | 0.008058 | +0.191942 |

All five image criteria pass. G4b signed response is 0.022053710105061672 against target 0.02187741965233074, with the unchanged open-surface policy. Measured-minus-forecast error is 0.000000 B1, +0.000235 SSIM, +0.001285 G4a, +0.007060 medium fraction and +0.000441 G4b relative error. The prior image is from [run 39580065706](https://github.com/milos-agathon/forge3d/actions/runs/39580065706). Neither 95b66c50 nor the earlier directional heads completed a physical image, so their image scores were absent.

All six gates, from the unchanged gate functions:

| Gate | Current evidence | Result |
|---|---|---|
| 1 | Heterogeneous relative error 0.0002028565; 0 majorant violations; RR relative difference 0.0001074220 | PASS |
| 2 | Energy residual 0.000316700; SE 0.0002072154; 10,000,000 outcomes | PASS |
| 3 | B1 1.000000; shaft SSIM 0.992584 | PASS |
| 4 | G4a 0.997431; medium 0.387035; G4b relative error 0.008058 | PASS |
| 5 | 356 Naga-valid assemblies; 0 compute comparison-sampling calls; ridgeline fraction 1.000000 | PASS |
| 6 | Frame hashes identical; host-visible peak 2,242,684 bytes <536,870,912; diagnostic wall times differ | FAIL |

Gate 1/2 producer metrics and the 356/zero static shader counts agree with the previous attempt; full Gate 5/6 were previously absent. Full metric objects, standard errors, hashes and before/forecast/after values are in JSON.

Gate 6's only failure is `diagnostics identical across runs`. Recursive comparison finds exactly one differing field:

| Field | Run A | Run B |
|---|---:|---:|
| directional_transport.iteration_wall_milliseconds | 1360.0408 | 1348.4775 |

The field is f64 at `media.rs:357`, populated from actual elapsed wall time at `media.rs:2151`; it is unchanged by this commit. Both primary frame hashes are `de2e1b8020ea809eddcdf37e803b2da8412c1db7d6bb5d3950fdba578ea1f3bc`, from independent processes 43816 and 44628. All other diagnostic fields match. The image forecast is supported; this failure is a separate deterministic-diagnostic contract issue, not a measured light-transport contradiction. No refit, weaker comparison, new cap or redispatch followed.

Proposed, **not applied**: keep real elapsed-time telemetry outside deterministic execution diagnostics. This requires a separate diagnostic-contract decision and exact approval before changing the contract. Do not zero or quantize measurements, and do not edit the frozen Gate 6 comparison.

The **1500x1500 DEM frame time is ABSENT**. CI skipped its benchmark step after Gate 6 failed. No local GPU work or substitute timing was run. The 1,380 ms figure measures only the 64x64 fixture's traced solver loop, not a frame or DEM benchmark. Full CI, release preparation, merge, dry run, tag and publishing were not performed. D3D12, Metal and WASM are ABSENT. The known reference sampler RNG-dimension overlap remains unchanged.

CPU verification actually run:

- Formatting and acceptance Clippy pass; required commit-hook lint passes. Commands and measured durations are in JSON.
- Five terrain/media graph tests and one shared-media shader test pass with extension-module,default,cog_streaming,shader-contract-asserts,splat-fusion. The new directional case and its negative shadow-guard check executed.
- Exact source f32 gate passes: 1,936 sites, digest 8f9f378ce8e27a98e9b99961073ae70128584485778adf1dc2c597c464c0cee7. Existing assertion lines are unchanged.
- All 356 real shader assemblies validate through the existing analyzer; zero unresolved constructions and compute comparison-sampling calls. Terrain/hybrid pins still match. Only the unconfigured trace assembly hash changes, 0x3e9a8b6bee37c914 -> 0xcf62804a9054a070.
- Integer diagnostic model: 126,976 finite-field difference checks match IEEE f32 exactly; all 65,536 field-validity checks, all finite adjacent decreases and signed-zero cases pass.
- Local wheel builds with CARGO_TARGET_DIR in the release worktree, embeds exact HEAD, and imports with a native hash matching the wheel. All 67 world-coordinate/NEPHELE acceptance/fixture/public-API CPU tests pass on it. Runtime flags are FORGE3D_NO_BOOTSTRAP=1 and FORGE3D_TEST_INSTALLED_WHEEL=1, with PYTHONPATH pointing to the isolated installed wheel. Fresh basetemp used. No dependency changes or local GPU calls.
- Required fixture/acceptance/physical-test/deltaE/SSIM and gate4-policy diffs are empty. Pins, ledgers, committed certificates, goldens, CI configuration and existing test assertions are unchanged. Hosted Fast Contract and Windows wheel jobs are green.

The official artifact is 12536620801, `nephele-physical-gpu-evidence-40031822131-1`, 42,013,486 bytes. It is downloaded under `.release-evidence/2026-10-24/ci-40031822131/`. The CPU reproducer checks raw head/adapter/JUnit/convergence data, diffs both diagnostics, and reruns all six unchanged gates. Its successful exit verifies the recorded failure; it does not turn CI green.

Proposed steering changes, not applied: distinguish volatile telemetry from G6's deterministic execution contract; use explicit UTF-8 and worktree-owned temporary/install paths in Windows evidence scripts. No AGENTS.md or skill changes were made.

CPU-only reproduction from the release worktree:

`.release-evidence/2026-10-22/analysis-venv/Scripts/python.exe docs/audits/nephele-aov-integer-state-2026-10-24.py --output <local-report.json>`
