# Native terrain upload cost acceptance evidence — 2026-10-12

Default terrain renders computed CPU coarse ambient occlusion that their shader never read. The fix skips it unless `ao_weight > 0` or the resolved colour debug mode is 28. Beauty, AOV and offline accumulation share this decision and release a previous coarse-AO texture when it becomes unused. Explicit consumers retain the original arithmetic.

The owner-approved base is recorded `origin/main` **a500ec0a**, Forge3D 1.43.0. Validation used the isolated `terrain-native-cost-origin` worktree. The owner subsequently authorized publishing the nine cost-stage files on `codex/terrain-native-cost` through a commit, push and draft PR. No subagents were used. Mesh density, public defaults, signatures, registration, exports, stubs, API contract lists, goldens, certificates, determinism reference hashes and gates are unchanged.

## F1: original wheel and film path

Both the public **1.42.0 Windows abi3 wheel** and the **1.43 base plus an environment-gated timing patch** were executed on the physical **RTX 3070 / Vulkan / NVIDIA 610.60**, Windows, Python 3.13.14 and NumPy 2.5.1. Baseline measurements set `FORGE3D_PROFILE_COARSE_AO=1`; the saved patch adds a timer and gated diagnostic print without changing AO arithmetic. The wheel was installed only into a separate validation venv. A fresh repository adapter probe passed. Input hashes, parameters, module paths, native binary hashes and timings are retained locally.

The controlled workload uses identical 1080×1080 `mesh:zup` settings across versions and before/after: 41,600 m span, MSAA 1, shadows/height AO/POM/detail off, default coarse-AO weight zero, fixed material/HDR, seed-0 random heights or a Gaussian hill. Fixture generation and readback are excluded from native-call timings. Pipelines are primed outside timing.

**Build-overlap caveat (*) for both tables:** the baseline release build ran 12:10:08–12:26:53 UTC. The accepted 1.42 synthetic run (12:25:34–12:27:52 UTC) overlapped its final 80 seconds; the 1.42 Tracy Arm run (12:22:13–12:24:55 UTC) overlapped entirely. The superseded synthetic attempt also overlapped. CPU contention was not controlled for these 1.42 columns, so differences from the 1.43 baseline must not be interpreted as version effects. Acceptance rests on the matched 1.43 before/after runs, the separate 1.42 range probe with no recorded concurrent build and byte comparisons. Build overlap does not invalidate frame-byte comparisons.

| Fixture | Size | 1.42 wheel wall s (*) | 1.43 baseline wall s | Coarse-AO span s | Fixed wall s | Fixed process CPU ms | Fixed GPU passes ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 1490² | 5.528 | 4.915 | 4.568 | 0.0921 | 15.62 | 57.99 |
| random | 2080² | 10.943 | 9.401 | 9.000 | 0.1215 | 62.50 | 57.72 |
| random | 3000² | 23.855 | 17.988 | 17.610 | 0.2271 | 156.25 | 58.25 |
| smooth | 1490² | 3.959 | 3.440 | 3.308 | 0.0397 | 31.25 | 8.77 |
| smooth | 2080² | 7.841 | 6.619 | 6.462 | 0.0605 | 46.88 | 8.75 |
| smooth | 3000² | 16.861 | 13.386 | 13.166 | 0.1054 | 93.75 | 8.84 |

The existing Tracy Arm `render_core.py` was also executed with the original local DEM and RGBA arrays, **PCF shadows, height AO, bilinearly upsampled overlays and MSAA 4**. Camera target is UTM (613200, 6413800, 300), radius 9000 m, bearing 200°, theta 62°, FOV 32°. The harness redirects its temporary map directory into the evidence directory and leaves source inputs unchanged.

1490² uses `far_dem_40m.npy` with resampled base imagery; 2080² strides the full 4160² DEM/imagery by two; 3000² bilinearly resamples them; 4160² uses the original full arrays. Negative elevations are clamped to zero as in the example. These are native renderer calls through the actual wrapper, not a replay of the film compositor or simulation.

| DEM size | 1.42 native s (*) | 1.43 baseline native s | Coarse-AO span s | Fixed native s | Fixed process CPU ms | Fixed GPU passes ms | Fixed whole wrapper s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1490² | 4.360 | 4.476 | 4.309 | 0.0385 | 31.25 | 13.15 | 0.813 |
| 2080² | 7.514 | 8.568 | 8.348 | 0.0735 | 46.88 | 14.88 | 1.818 |
| 3000² | 16.115 | 16.525 | 16.190 | 0.1430 | 78.12 | 14.07 | 4.416 |
| 4160² | 31.238 | 27.282 | 24.172 | 0.2926 | 203.12 | 27.19 | 7.807 |

Native instrumentation identifies `render_internal_with_aov` → `upload_height_inputs` → **`TerrainScene::compute_coarse_ao_from_heightmap`** in [ao.rs](../src/terrain/renderer/resources/ao.rs). It surrounds allocation plus the CPU kernel, before texture creation/upload. Every texel scans up to 288 radius-8 neighbours and computes square roots and arctangents for higher neighbours. The AO kernel's source is identical at `v1.42.0` and the approved base. Its recorded spans are wall time inside CPU-only work; aggregate process CPU is measured separately. The exact instrumentation patch is retained and removed from the final code.

The October 11 scratch inputs, binary hash and execution environment were not retained here, so their exact 6/13.5/27.8 s figures cannot be reconstructed. The table's 23.855 s 1.42 result was affected by build overlap and is not a matched version comparison with the 17.988 s 1.43 baseline.

A controlled 3000² height-range experiment on the public 1.42 wheel changes only random-height magnitude: 0–2000 takes **22.813 s** (22.531 CPU s), while the same samples scaled to 0–1 take **30.336 s** (30.203 CPU s). GPU pass time decreases from 202.32 to 112.41 ms. This reproduces the reported roughly 30-second CPU stall and shows that height-value range can account for several seconds of discrepancy. Inference from the unchanged kernel: argument-range-sensitive arctangent work explains this content dependence; the historical script's actual range remains unknown. The corrected binary takes **0.3476 s** and **0.0928 s** respectively, with identical input hashes/settings in both pairs. Raw results and the frozen probe are retained with the main evidence.

## CPU and GPU accounting

| Call | Baseline coarse-AO / wall | Fixed aggregate process CPU / wall | Fixed timestamped GPU passes / wall |
|---|---:|---:|---:|
| synthetic random 1490² | 92.9% | 17.0% | 63.0% |
| synthetic random 2080² | 95.7% | 51.4% | 47.5% |
| synthetic random 3000² | 97.9% | 68.8% | 25.6% |
| synthetic smooth 1490² | 96.2% | 78.7% | 22.1% |
| synthetic smooth 2080² | 97.6% | 77.5% | 14.5% |
| synthetic smooth 3000² | 98.4% | 88.9% | 8.4% |
| tracy  1490² | 96.3% | 81.2% | 34.2% |
| tracy  2080² | 97.4% | 63.8% | 20.2% |
| tracy  3000² | 98.0% | 54.6% | 9.8% |
| tracy  4160² | 88.6% | 69.4% | 9.3% |

CPU work and GPU passes can overlap; these columns are **not an exclusive partition**. Windows process CPU is quantized to 15.625 ms. GPU timestamp sums cover reported passes, not every transfer, driver operation or synchronization interval. Individual matched calls are reported, not a latency distribution. Upload copying/hashing and the wrapper's overlay upsampling remain; explicit coarse-AO consumers retain their original CPU cost. No GPU clock or shader improvement is claimed. No unused coarse-AO allocation appears in the corrected large workloads.

## F2 and F3: regression and output proof

- The refresh regression now uses a displaced, non-flat valley after the first hill and a camera radius appropriate to the 2.8-unit domain. It requires visible non-white AO pixels, checks reused versus fresh renderer bytes and rejects unchanged/flat results; separate height AO is explicitly disabled so it cannot substitute for coarse AO. Consumer allocation and parameter-over-environment precedence are exercised across beauty, AOV and offline rendering.
- **144 synthetic arrays** match shape, dtype and bytes before/after, including **60 offline accumulation arrays**: beauty, float HDR, depth, normal and albedo for 16 seeded samples across three camera modes and all tested AO/debug consumers.
- **8 Tracy Arm beauty/depth arrays** match before/after. All 144 synthetic/offline and eight film-path baseline arrays also match the public 1.42 wheel. Inputs and supported render settings match; 1.43 adds the unused `materials.albedo_path=None` field to the film configuration. Every other field compares exactly. Before/after configurations on the approved 1.43 base are strictly identical without schema normalization.
- **14 golden candidate arrays** match before/after with every original golden assertion retained. Coverage includes terrain visual, TV10, SUBSTRATIA mesh grazing, terrain-raster recipe, **AETHER active sky** and **ORBIS Rainier ground**. The Rainier DEM was restored from a local copy only after matching its LFS SHA-256. NVIDIA/Vulkan golden files were verified as committed.
- The final run passed **13 golden tests**, **18 strengthened coarse-AO cases**, and the two named existing tests `test_single_sample_offline_matches_manual_session_baseline` and `test_offline_is_deterministic_for_same_seed`: **33 selected tests, no skips**. Logs use `-rA` to retain test IDs.

## F4: reproducible provenance

The new paired runs supersede the earlier thinly documented measurements. [The benchmark](../tools/benchmark_terrain_upload.py) and [the Tracy Arm harness](../tools/benchmark_tracy_arm_upload.py) were frozen before the accepted runs. Exact script snapshots, source patches, hashes, commands, relevant environment, start/end timestamps and exit codes are retained under `artifacts/terrain-native-cost/review/`. `verified-summary.json` checks frame bytes, input/settings identity, binary provenance and script hashes. Old logs and unsuccessful attempts remain available; no missing historical binary hash is invented.

Native binary SHA-256 values:

- baseline: `f3a06806a69a08d1409df2fa40db638663ef128fc72779ad52d9374afca86c48`
- fixed: `318ac093741c64f5c64b2a39f83aa8dcbb7fd2422aca3922933514254367defb`
- wheel142: `3fd9b3cf351f2a878c6f70f92334cd5de50dd8df2456ab121a588e4c1df400e4`

Both new `maturin develop --release --offline` builds passed, using the release features in unchanged `pyproject.toml` (also captured in `release-features.json`). `cargo forge3d-clippy` and `cargo fmt --check` passed with explicit worktree `CARGO_TARGET_DIR`. Clippy used the repository alias and its existing `-D warnings` feature lane. `git diff --check` passed. Profiling code is absent from the final Rust diff, which exactly matches the saved cost-fix patch.

Clippy emitted filesystem warnings about falling back from hard links to copies in its incremental cache; it exited successfully. Build-generated tracked debug symbols were restored to their committed LFS pointer through Git. The final patch contains source, tests, tools and documentation; the rebuilt native extension hash was verified afterward.

Golden runs set `WGPU_BACKEND=vulkan`, `FORGE3D_NO_BOOTSTRAP=1`, `FORGE3D_RUN_TERRAIN_GOLDENS=1`, terrain/recipe/SUBSTRATIA variants `nvidia-vulkan`, `FORGE3D_RUN_ORBIS_GPU=1`, a valid `FORGE3D_EXPECTED_ADAPTER_PROBE`, `FORGE3D_AETHER_PHYSICAL_BACKEND=vulkan` and `FORGE3D_AETHER_PHYSICAL_REQUIRED=1`. Regression runs use Vulkan and no bootstrap. `CI` and `GITHUB_ACTIONS` are unset. Exact values, writable `TEMP`/`TMP`, pytest basetemp and executable paths are in run metadata. No update or signing modes were used.

An initial wheel run overlapped a metadata-only edit and was superseded by the frozen rerun. The Tracy warm-up initially had an invalid zero-width height range and was corrected before measured calls. The first baseline golden run passed 12 tests but blocked SUBSTRATIA because the evidence wrapper overwrote probe JSON with run metadata; distinct filenames fixed the envelope, and only that failed test was rerun successfully. The strengthened refresh assertion also exposed the original test's 1200-unit default camera radius: it saw only background. An explicit 5-unit camera fixes framing and all 18 affected cases were rerun; the two unchanged, named offline passes are reused from the preceding run. Final golden coverage passed together. These were harness/fixture failures; no production check was weakened.

## F5 and F6: environment and placement

Earlier maturin logs show system pip was selected because the old venv lacked its own pip. This failed to isolate package installation and should have been disclosed. The film's exact pre-task wheel state cannot be reconstructed. On 2026-10-12, in this task's owner reply to the question about which Forge3D version Tracy Arm should use, Milos answered: **“Keep the currently installed 1.43 wheel.”** New builds use venv-local pip and Python; the corrected extension is installed in the worktree venv.

The system native SHA-256 was `855d478aae75a91f8d32c1bf477b899bfb28c477b15a1eaa28f6bf5a11db4ed3` at the recorded before/after checks during this correction, whose final recorded validation run ended at 13:21:58 UTC (15:21:58 local). This is a claim about that window only. The later read-only reassessment reported a different system binary, `41f3de48…`, installed at 16:05 local from a self-hosted runner build after that window. This follow-up does not change the system installation.

This evidence report was moved from the Sphinx guides directory beside the other acceptance reports. Repository paths are relative. Raw evidence is local and ignored by Git; its presence does not mean it is committed or published.

## Scope and limits

Production changes remain the five renderer files `upload.rs`, `draw/setup/context.rs`, `draw/mod.rs`, `aov.rs` and `offline.rs`. The test, two benchmark tools and this report supply proof. The ignored local roadmap item receives the review evidence and remains open for its geometry decision.

Mesh geometry stays at **512×512 vertices** (~81.4 m spacing over 41.6 km). No density control or default was added. Density-specific silhouettes and memory tests therefore do not apply and were not run. Native diagnostic captures report 1024-byte host-visible peaks in these paired calls, before array readback; tracker categories do not cover ordinary CPU vectors or process RSS. Existing SUBSTRATIA render-memory assertions also passed within their 512 MiB budget.

No broader Cargo/CI matrix, WASM, Metal/DX12 or other physical adapter was exercised during validation. No cross-backend portability claim is made. The validation stage performed no commits, pushes, PR creation, releases, signing or hosted writes; the owner's later publication request covers the cost-stage commit, push and draft PR.

Reusable steering proposals, not applied to agent contracts: verify pip resolves inside the venv before maturin; set the worktree `CARGO_TARGET_DIR` for Git commands whose hooks invoke Cargo, too; freeze/save benchmark scripts at run start; keep command metadata distinct from probe products; assert visible terrain with explicit camera scale in pixel regressions; check an asset's type before text reads; use writable worktree temporary directories when sandbox fixture cleanup fails. The initial publication commit attempt was interrupted when its Clippy hook inherited the global target directory, then rerun with an explicit worktree target. Promoted project skill directories were absent, so the supplied root and original checkout's nested Rust/WGSL contracts were followed directly.
