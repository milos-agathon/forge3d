# NEPHELE deterministic physical terrain remediation

Status: **shader proof prerequisites passed; physical acceptance NOT_PROVEN**. Worktree: `D:/forge3d/.worktrees/nephele-172-completion`, branch `codex/20-nephele`. This report distinguishes dirty-tree diagnostic evidence from the required clean H1 campaign.

## Current B1–B9 observations

| Check | Executed evidence | Result |
|---|---|---|
| B1 vacuum terrain | Fresh rebuilt-wheel capture against independently reproduced, registered 320-spp control: 1,546/1,558 pixels (99.2297818%) have ΔE2000 < 2; mean 0.33803308, p95 0.86612297, max 6.53942811. | PRECHECK_PASS under the unchanged ≥95% rule |
| B2 environment radiance | Earlier physical provider readback: every RGB texel is 0.25; diffuse irradiance is f32(π × 0.25). Unit regression re-executed in the 25 media tests. | Provider/units pre-check passed |
| B3 froxel XY | The 64×64 viewport uses 8×8 visible cells plus the one-cell border, giving 10×10×64 allocated cells. Grid coverage, depth transforms and edge-coordinate tests executed. | Structural pre-check passed; physical fidelity still fails G3 |
| B4 phase direction | Both HG evaluation/sampling tests passed. The shader uses dot(camera-to-cell incident direction, direction-to-sun), equivalent to the incoming/outgoing photon-angle cosine. | Direction/model pre-check passed |
| B5 multiple scattering | Actual GPU diagnostics: one single and one multiple dispatch; mean luminances 0.1519101150 and 0.0467867403. Analytic Beer-integral and accounting regressions passed. | Executed approximation; reference fidelity NOT_PROVEN |
| B6 terrain-to-sun and environment | Actual bounded nested-midpoint sun integration: 36,723 steps, max segment 68.22770691, measured coarse/fine max RGB difference 0.008301794529. Independent 640-spp zero-sun reference completed in 4,467.33 s; candidate has only 34/1,557 pixels (2.1836866%) below ΔE 2, mean 3.80689448, max 9.41999081. | Sun-segment tests passed; environment comparison FAILED |
| B7 visibility source | Production per-cell sun terrain trace and terminal-surface manual CSM remain active. Shader assembly/manual-shadow tests passed. The fixture leaves the separate heightfield SunVisibility pass disabled, using its unity fallback. | Source/pipeline pre-check passed |
| B8 terrain occlusion | Per-cell phase/sun trace assembly and resource contracts passed. Independent occlusion-disabled capture gives godray ROI SSIM 0.9381720465, failing the unchanged strict <0.80 criterion. | Structural pre-check passed; physical ablation FAILED |
| B9 direct-light lookup | Froxel XY/log-depth lookup and assembled same-medium transmittance tests passed. Physical terrain applies the RGB medium transmittance to direct sunlight. | Source/pipeline pre-check passed |

[B1 report and arrays](nephele-b1-deterministic-2026-09-30/b1-report.json) contain the actual camera, adapter and diagnostics. [B2 readback](nephele-b2-2026-09-30.json) remains historical provider evidence.

The fresh vacuum frame, medium-enabled frame, in-scatter, transmittance and cloud-shadow arrays are all byte-identical to the earlier proposed-pin capture. Thus the deterministic rewrite removed proof alarms without changing these observed frames. The old 40-spp B1 failure remains preserved in [its historical capture](nephele-physical-terrain-2026-09-30/b1-report.json); it is not the current B1 result.

## Plan publication and independent control reproduction

The historical `c7804827…` plan was not externally published before rendering. A command-line `--plan-sha256` value and its registration echo do not establish publication; the earlier report's assertion of a pre-render chat publication was unsupported.

A new [fixed plan](nephele-vacuum-control-plan-published-spp320-2026-09-30.json), SHA-256 `fdaed43c61ad5c06c327682909462ffb90ce8567b31a3ff79deecadf9047bb1b`, was committed and pushed in [1cf04e4704a76c4627aacfe51aba335e8b178f35](https://github.com/milos-agathon/forge3d/commit/1cf04e4704a76c4627aacfe51aba335e8b178f35) before its generator was invoked. A remote `ls-remote` observation was saved before rendering.

The generator verified the retained reference wheel and the actual imported native SHA-256 `7661b8f81f6a5253d035b8d2e6241ccf73e8b1c001c4db6981b5c65a3012607f` before GPU work. It executed 1,310,720 samples at the planned 320 spp, fixed seed and camera, in 757.669539 seconds. [New registration](nephele-vacuum-control-published-spp320-2026-09-30/vacuum-control-spp320.json) SHA-256: `e44c0f3417c6fe88eea707831587dc0b6d5e1a7e23ad47ee2a3e135b97f56080`.

The independently rendered array reproduced SHA-256 `7e9912a7370597b2395b22edca0b0dfff16a407314bff0962e7a78c474aed0d6` exactly. The frozen 640-spp acceptance reference, masks, original 20/40-spp controls, thresholds and goldens remain unchanged.

## Shader fix and pin conditions

`terrain_reference_geometric_normal` now uses guarded cell counts, deterministic division, interpolation and normalization. The physical lighting branch uses deterministic dot/division and barriers between radiance products and sums. The reciprocal guard uses the minimum normal f32, matching the deterministic normalization helper; it does not introduce a scene-tuning threshold.

The assembled terrain source pin is `0xe3539a322a87b8cc`. Its conditional authorization is satisfied by the observed source, proof, ablation, shader-proof and unchanged NVIDIA golden checks below. The old four Rust and two Python failures were genuine alarms in the newly added shader code; they were not failures of the pin procedure.

| Check | Observed result | Log in [committed validation records](nephele-remediation-2026-09-30/) |
|---|---|---|
| Rust verifier/source pins/proofs/ablations | 108 passed, 0 failed, 1 existing inspection-only ignored | `remediation-proof-tests.log` |
| Rust realtime-media contracts | 25 passed, 0 failed/ignored; expected names confirmed | `remediation-media-tests.log` |
| Rust HG phase | 2 passed, 0 failed/ignored | `remediation-phase-tests.log` |
| Rust runtime input contracts | 4 passed, 0 failed/ignored | `remediation-runtime-tests.log` |
| Python shader proofs + nine existing terrain goldens + physical-albedo regression | 18 passed, 0 failed/skipped | `remediation-shader-goldens.log` |
| API and degradation contracts | 180 passed, 0 failed/skipped | `remediation-api-tests.log` |
| Control and diagnostic provenance checks | 17 passed, 0 failed/skipped | `remediation-control-tests.log` |
| Full three-file NEPHELE suite on rebuilt wheel | 151 passed, 0 failed/skipped; 1,257.45 s | `remediation-nephele-suite.log` |
| Formatting / canonical clippy | Both exit 0 | `remediation-format.log`, `remediation-clippy.log` |
| Source whitespace | Passed; archived raw CRLF evidence is preserved byte-for-byte and produces whitespace notices in an unfiltered diff check | `remediation-source-diff-check.log` |

The historical `approved-shader-goldens.log` contains **11 failures**, consisting of two shader-proof failures and all nine goldens. Its golden selector was missing, so NVIDIA output was compared to the generic/Metal fallback. The independent NVIDIA run and the current 18-test run explicitly use `FORGE3D_TERRAIN_GOLDEN_VARIANT=nvidia-vulkan`; no golden file changed. The historical 143-test result used an earlier wheel and is not final-source evidence.

Current wheel SHA-256: `6df33167c94c7b01bb8fd7abcf04a60eb212d8b67eb91b378274749236c21997`. Installed native SHA-256: `e4cfbd5123be2255ab411d93906f56d44678188a4a89358bba51b2cb48a89586`. It reports source revision `1cf04e47…`, but was built from dirty source. The new B1 report correctly records `source_matches_repository_head: false`; a matching embedded HEAD alone does not prove a clean-source build. Historical captures are preserved rather than rewriting their hash-bound metadata.

GPU evidence uses a literal NVIDIA GeForce RTX 3070 / Vulkan / NVIDIA 610.60; adapter qualification succeeded with `software_fallback: false`. Python checks set `FORGE3D_NO_BOOTSTRAP=1` and `FORGE3D_TEST_INSTALLED_WHEEL=1`; GPU checks also set `WGPU_BACKEND=vulkan` and `WGPU_BACKENDS=vulkan`. Golden checks additionally set `FORGE3D_RUN_TERRAIN_GOLDENS=1` and the explicit NVIDIA variant. Golden-update variables were not enabled. Rust tests and clippy used `extension-module,default,cog_streaming,shader-contract-asserts` and the isolated worktree `target` directory.

## Hosted core CI correction

The first source commit, `1c6500154efff602c1be0b691492adf67a947bae`, failed hosted Fast Contract because the computed SSIM was `0.9942275395295544`, while the registered Windows value was `0.9942275395295402`. The verifier already checked recomputation against registration with its existing `1e-12` tolerances, then returned the computed value to a caller that expected the exact registration. It now returns the validated registered `metrics` and separately exposes `recomputed_metrics`. Acceptance still uses the recomputed values and unchanged thresholds; no test was weakened. The new measured-roundoff regression and tracked-fixture test passed (2); all 24 provenance/convergence tampering cases passed; the full six-gate report recomputation test passed (1).

The initial plan-only CI failure was a shallow-history lookup (`fatal: not a tree object`); the full-history workflow change resolved it. Follow-up [core CI 36783277414](https://github.com/milos-agathon/forge3d/actions/runs/36783277414) is green on `ddbdcc086b401fe1842a9725f32c5e493d5c65cf`: Fast Contract reports 588 passed, 28 intentional lane skips, 1 deselected; PR Core Success passed. This is not a zero-skip physical acceptance result.

## Remaining physical evidence

Recomputing the fresh medium-enabled arrays against the unchanged approved reference gives sky/cloud ΔE pass fraction **0.3955870764**, godray ROI SSIM **0.8134602396**, and cloud-shadowed terrain pass fraction **0.4656390495**. These fail the existing G3/G4 criteria. Medium removal changes **0.3414634146** of terrain pixels by ΔE > 5, passing that separate ablation condition. These frames also match the earlier capture byte-for-byte, establishing that the physical fidelity failures precede this deterministic rewrite.

B6 isolates environment illumination by setting sun intensity to zero on both retained-reference and candidate paths. Its sample count and seed come from the frozen reference provenance (640 spp); it never replaces that reference. The comparison uses the committed cloud-shadow mask and reports the unchanged ΔE < 2 / ≥95% diagnostic criterion. B8 separately captures the terrain-occlusion-disabled scene with the same physical model, camera and native runtime. Both diagnostics failed; [B6 reference](nephele-b6-environment-reference-2026-09-30/report.json), [B6 candidate](nephele-b6-environment-candidate-2026-09-30/report.json), and [B8 ablation](nephele-b8-occlusion-2026-09-30/report.json) retain the raw arrays and their hashes. They do not justify changing the frozen fixture, thresholds or transport model.

First H1: **FAILED before rendering or the six-case suite**, in [36785284320](https://github.com/milos-agathon/forge3d/actions/runs/36785284320) on clean source `ddbdcc086b401fe1842a9725f32c5e493d5c65cf`. Exact-head wheel installation and literal NVIDIA Vulkan adapter qualification passed. The majorant producer accumulated NumPy booleans into an `int64`, which JSON could not serialize. Its increment now converts the boolean to a plain integer, preserving the count and comparison. The strengthened existing producer test reproduced this error, then passed; the producer and transported-input group passed all 11 tests. No threshold or check was weakened. [Raw first-run artifact](https://github.com/milos-agathon/forge3d/actions/runs/36785284320/artifacts/11130632018) and [failure record](nephele-remediation-2026-09-30/h1-ddbdcc08-result.json) are retained. A follow-up clean H1 will run after this correction is committed and core CI passes. H2, merge, release and package publication are outside this request and have not been performed. The earlier Vulkan committed-determinism hash mismatch remains historical and unchanged; it was also observed on the baseline wheel. No determinism golden/hash was updated.

## Scope and reusable steering

The existing task-owned Python/native physical mode, material unit fixes, stubs, input validation, tests and full-history workflow changes are retained and completed alongside the deterministic shader fix. The maintained B1 tool and its tests will be committed together. New B6/B8 tools save real physical diagnostic arrays; they do not synthesize acceptance. Unrelated changes in the primary checkout were preserved. Work used the parent only, with no subagents.

A first media test filter selected `terrain::renderer::media::tests` and executed zero tests; it is not evidence. Inspection located the canonical `terrain::realtime_media::tests` module, and all 25 expected tests then executed. Proposed documentation clarification only: record that module route and the exact golden-selector variable in local validation guidance. No agent contract or skill was edited.

Hash-bound diagnostic files and generator scripts now preserve their bytes across platform checkouts through scoped Git attributes. This prevents checkout line endings from invalidating recorded raw hashes.

A Windows report edit encountered the default cp1252 encoding error. The report was restored from its committed copy and rewritten as UTF-8. Proposed documentation clarification only: require explicit UTF-8 for Unicode audit-file reads and writes.

## Current-main integration

[Follow-up H1 36789042480](https://github.com/milos-agathon/forge3d/actions/runs/36789042480) was blocked before any producer or render: protected preflight could not merge the candidate with current policy base `2c5fb1614d3719fc7eaab9efb903f02a9f24e60c`. The two conflicts were the source conversion inventory ledger and its test. Both original NEPHELE and CHRONOS review records, counts, digests, additions and removals are preserved unchanged. Reversing either peer transition from the merged source reconstructs every site of the other approved parent; there are zero unreviewed sites. The combined source inventory has 1,881 entries and SHA-256 `45ca418834228c2e25d26dfded6e8d86f1cda1ba67b1eb0079cf32f6df33b192`. All 19 world-coordinate gates passed; formatting passed. Render determinism hashes and golden files were unchanged. The native API and seeded accumulation updates arriving from main require fresh exact-head wheel validation; the earlier diagnostic wheel remains historical. H1 will be re-executed on the integrated source.
