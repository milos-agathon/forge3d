# NEPHELE bounded remediation round — 2026-10-01

> **Round 2 update (same day):** the sky-attenuation change described below was reverted, and B1 reproduces 1,546/1,558. A validated first-order split identified the best-defined missing term, but G4's terrain-occlusion-ablation SSIM (0.958, limit 0.80) and G3 (sky-only, 91.69%) already fail independently of it, so the fix was chosen but not implemented and NEPHELE stays NOT_PROVEN (owner sign-off recorded below). The hosted-Linux segfault came from a wgpu-core 0.19 wait defect and is fixed. See [Round 2](#round-2--revert-first-order-split-and-hosted-linux-crash). The sections before Round 2 record round 1 as it happened.

**Round complete; physical acceptance NOT_PROVEN. NEPHELE remains unreleased and targets 1.41.0.** G3 and G4 still fail after the two lighting changes and the conditional shadow-mask diagnosis. The authorized stop rule has been applied: no further quality changes, fixture edits, H1 dispatch or H2 sequencing in this round.

The clean source snapshot is `48d6a052d3b05ef2db16ab3d33fe8028e07d3819`. The final release-profile 1.41.0 wheel SHA-256 is `3870d78f55ea11da2ac05e702d657cb27fb1cbb8e53e74e3e7056929f0e49e1b`; its actually imported native SHA-256 is `b3ceff207d1eb3f001d7156a237fe2b61bc56ee7f39bff7b5eceda4286b53f84`. Capture A and B used separate processes, this clean source, the same fixed fixture, and the independently retained reference wheel. Observed adapter: Windows X64, NVIDIA RTX 3070, Vulkan, NVIDIA 610.60.

## Implemented changes

- Both ordinary terrain rendering and the acceptance AOV path clear a uniform authored HDR environment to its linear boundary radiance. The common presentation pass applies exposure and ACES/sRGB once. The no-medium camera-background readback is exactly `[165, 165, 165]` for linear 0.25 and fixture exposure 1.
- Physical terrain environment lighting now multiplies by a deterministic cosine-weighted upward sky-transmission factor. Four equal-area disk midpoint nodes have `r²=1/2`, `phi=(2k+1)π/4`, directions `(±1/2, 1/√2, ±1/2)` and weights 1/4. Each ray reuses the existing canonical bounded-medium integration used by sunlight; it integrates the complete density bounds rather than marching a clamped camera-frustum field. Its tracked RGB volume is bound at group 4/binding 4 and declared as a forward-pass input. The analytic colored-slab/vacuum regression passes. No RNG or fitted constants were introduced.
- The historical reference-source check remains. A new live check compares all 17 retained reference source inputs to `6a4ae50a4c3c01b390b1222468632e88d1db6ca0`. The only reviewed exception is `src/terrain/mod.rs`, and its exact accepted SHA-256 is `154b5c88a739e3331615976a1497621bbfd6fa79afec215543356b98daf51698`. A further edit to that file is rejected too. The new live-check regressions pass.
- The private acceptance capture has an optional `capture_scatter_components=False` argument. When enabled, it reads the GPU's existing integrated multiple-scatter luminance channel. Its aggregate agrees with the independently returned native diagnostic. `scripts/nephele_shadow_inscatter_diagnostic.py` captures the unchanged scene, an isotropic ablation and a terrain-occlusion ablation on the actual cloud-shadow mask.
- Cargo, lockfile package metadata, Python metadata and package version target 1.41.0. The changelog places NEPHELE under planned/unreleased 1.41 work and requires G3/G4 before delivery.

The terrain pin is now `0x8bed35a0ffd0b5e1`. Its pre-approved conditions all passed: Rust proof/ablation/source pins, Python `test_shader_proofs`, the nine unchanged NVIDIA goldens, and B1 ≥95% against the unchanged registered 320-spp control. A new shader proof entry exercises sky transmission with media enabled. Qualification used a dirty source tree and is not presented as clean physical acceptance evidence; the final measurements below use the rebuilt clean source.

## Local visual results

| Metric | Background fix only | Final sky attenuation | Required result |
|---|---:|---:|---:|
| G3 sky/cloud pixels below ΔE 2.5 | 91.6863672% | 91.6863672% | ≥95% |
| G3 godray ROI SSIM | 0.9627141234 | 0.9056682493 | >0.95 |
| G4 shadowed terrain pixels below ΔE 2 | 46.5639049% | 6.4868337% | ≥95% |
| Medium-removal changed fraction | 34.1463415% | 78.3055199% | ≥10% |
| B8/G4 terrain-occlusion ablation SSIM | 0.9578603137 | 0.9566974834 | <0.80 |

The original H1 sky/cloud fraction was 39.5587076%; its godray SSIM was 0.8134602396. Correcting the background therefore helped substantially, but the final coupled frame still fails both visual gates. [Saved visual metrics](nephele-bounded-round-2026-10-01/final-local/visual-metrics.json) and [comparison image](nephele-bounded-round-2026-10-01/round1-visual-comparison.png) preserve the measurements and visible cell pattern.

## Current B1–B9 observations

| Check | Final executed evidence | Result |
|---|---|---|
| B1 vacuum terrain | 1,546/1,558 pixels below ΔE 2 (99.2297818%); mean 0.33803308, p95 0.86612297, maximum 6.53942811 against the registered 320-spp control. | PASS under the unchanged ≥95% rule |
| B2 boundary/environment units | Actual camera-background readback `[165,165,165]`; uniform linear radiance 0.25, exposure 1, shared presentation. Both render routes use the same clear helper. | Readback PASS |
| B3 froxel representation | 64×64 viewport, 8×8-pixel visible tiles, one-cell border, 10×10×64 allocated cells; grid/depth/edge regressions passed in the renderer tests. Cell patterns remain visible. | Structural checks PASS; fidelity NOT_PROVEN |
| B4 phase direction | Existing camera-to-cell/direction-to-sun HG convention retained. Isotropic ablation lowers shadow-mask in-scatter luminance from 0.09829501 to 0.07449612 (−24.2%). | Measured phase sensitivity; no phase-sign change justified |
| B5 multiple scattering | Shadow-mask single luminance 0.07213668 and multiple luminance 0.02615833; total 0.09829501 versus reference 0.17567929. Multiple is 26.6% of the real-time total. One single and one multiple GPU dispatch executed. | Only 55.95% of reference in-scatter; fidelity NOT_PROVEN |
| B6 environment under cloud | Zero-sun candidate/reference: 160/1,557 pixels below ΔE 2 (10.2761721%); mean 6.84888369, maximum 12.74670025. The unchanged independent reference is 640 spp. | FAILED; candidate is too dark |
| B7 visibility source | Canonical sun segment, per-cell sun/phase terrain trace and terminal-surface manual CSM remain active; 1,244 terrain queries observed. Renderer/source contracts passed. | Source/pipeline checks PASS |
| B8 terrain occlusion | Godray SSIM 0.9566974834. Disabling occlusion increases shadow-mask in-scatter by 0.00906621 (9.2%). | Physical ablation FAILED |
| B9 direct-light lookup | The existing guarded XY/log-depth coordinate calculation is shared with sky transmission; direct sunlight continues to use the same canonical RGB medium transmittance. Renderer tests, proofs and B1 passed. | Source/pipeline checks PASS |

[B1 arrays/report](nephele-bounded-round-2026-10-01/b1/b1-report.json), [B6 arrays/report](nephele-bounded-round-2026-10-01/b6/report.json), [B8 report](nephele-bounded-round-2026-10-01/b8/report.json), and [shadow-component arrays/report](nephele-bounded-round-2026-10-01/shadow/report.json) bind the actual outputs. B6's mean encoded RGB changed from roughly `[129,128,126]` before attenuation to `[104,102,101]`, while the reference is `[121,118,115]`. Adding terrain sky occlusion would further darken the already-too-dark candidate, so it was not added.

The phase ablation makes the deficit worse; removing terrain occlusion recovers only a small part of it. Frozen reference arrays contain total in-scatter, not a per-order split, so the remaining deficit cannot be uniquely assigned to missing scattering orders. The coarse froxel pattern and weak B8 signal are consistent with the OD-10 resolution concern, but do not prove resolution is the only cause. No coefficients, sample weights or thresholds were fitted to this fixture.

## Other local gates and validation

G1, G2, G5 and G6 were independently recomputed with the production verification functions and unchanged rules. These are standalone measurements, not a six-gate acceptance certificate. [Independent gate record](nephele-bounded-round-2026-10-01/final-local/independent-gates.json):

- G1 passed: heterogeneous relative error 0.00020285654; Russian-roulette relative difference 0.00010742202; the one-million-probe majorant check has zero violations.
- G2 passed: relative energy residual 0.0003167.
- G5 passed: all 303 assemblies validated, tracked exhaustive analysis matches, and ridgeline agreement within one slice is 100%.
- G6 passed: independent frame hashes are identical (`70e8cb16…`), with 913,248 froxel device bytes, 153,600 staging bytes, 171,008 peak host-visible bytes and 262,144 readback bytes. The sky volume adds 51,200 device bytes and 51,200 staging bytes for this fixture.

| Check | Observed result |
|---|---|
| Rust proof, ablation and source pins | 108 passed; one existing inspection-only test ignored |
| Rust renderer contracts, including new binding and slab/vacuum checks | 69 passed, zero failed/ignored |
| Final clean wheel: shader proofs + nine NVIDIA goldens + physical-albedo regression | 18 passed, zero skips, 86.97 s |
| Final clean wheel: background and scattering-component readbacks | 2 passed, zero skips, 5.60 s |
| Final clean wheel: API contracts | 165 passed, zero skips, 5.28 s |
| Final clean wheel: original three-file NEPHELE suite plus three live-source cases | 155 passed, zero skips, 1,178.44 s |
| Canonical clippy and formatting | Exit 0 |
| Exact six-case physical suite | Six shared-setup errors at G3, zero skips |
| Full physical verifier | Rejected the failed JUnit record; physical acceptance NOT_PROVEN |

The zero-skip Python checks used `FORGE3D_NO_BOOTSTRAP=1`, `FORGE3D_TEST_INSTALLED_WHEEL=1`, `WGPU_BACKEND=vulkan`, `WGPU_BACKENDS=vulkan`, and the worktree-specific `CARGO_TARGET_DIR`. Golden checks additionally used `FORGE3D_RUN_TERRAIN_GOLDENS=1` and `FORGE3D_TERRAIN_GOLDEN_VARIANT=nvidia-vulkan`; the physical lane used `RUNNER_ARCH=X64`, `RUNNER_OS=Windows`, and `FORGE3D_NEPHELE_LANE=windows-nvidia-vulkan`. No golden-update flag was set.

[Byte-preserved validation logs](nephele-bounded-round-2026-10-01/validation/) and [archive manifest](nephele-bounded-round-2026-10-01/manifest.json) identify the scope. The archive is a subset: generated native binaries and large statistical raw payloads remain in the local `.tmp/nephele-completion/round1-final-local` campaign. This committed subset cannot serve as a complete acceptance certificate. All captured arrays referenced above are retained. The published 320-spp plan/control, frozen 640-spp reference, fixture, masks, thresholds, golden files, hybrid pin and determinism-helper pin are unchanged.

## Hosted CI integration follow-up

The initial broader core run [36816534089](https://github.com/milos-agathon/forge3d/actions/runs/36816534089), at audit commit `7fdbf972105dfbab6d21b5478682d490f02de5f3`, passed its wheel builds and compatibility checks but failed the Linux full suite: three failed, 3,802 passed, 316 skipped and 55 deselected. The failures were the stale atmosphere resource count, a background readback tied to the Windows-only control registration path, and a crashed diagnostic subprocess.

The atmosphere test now requires all five exact group-4 resources, including sunlight and sky transmission. The background test reads the same no-medium frame directly through the diagnostic, retaining its full pixel assertion; the immutable B1 producer and its control validation are unchanged. The diagnostic explicitly selects the installed wheel and Vulkan, and its test subprocess enables fault reporting. No tests were skipped or weakened. The combined atmosphere/readback tests passed locally: 31 passed. All 15 default diagnostic GPU arrays remain byte-identical to the archived measurements. These are test-harness corrections; rendering/native source and the terrain pin are unchanged from `48d6a052`. The archive records the historical measurement producer at that source; the corrected producer is a later revision. Hosted results for the corrected harness are reported separately after push.

The subsequent broader run [36819717059](https://github.com/milos-agathon/forge3d/actions/runs/36819717059), at `57cfdfc3`, passed 3,804 tests with 316 skips and 55 deselections; one background subprocess still crashed inside the dual media/no-medium capture. The atmosphere contract and separate scattering-component test passed. That background capture also requested the optional additional multiple-scatter readback, which its pixel assertion does not need. The background mode now requests only its required color readbacks, while the separate scattering test still exercises the actual component readback. Both local tests passed again (2 passed, zero skips, 4.92 s). The combined extra-component/no-medium native path on hosted software Vulkan remains unproven; its root cause was not established or changed after the OD-10 stop. No physical portability claim is made for it.

The final broader run [36822968213](https://github.com/milos-agathon/forge3d/actions/runs/36822968213), at `41979211d3db364a85386cfeeafc67ae86f7e2de`, still failed that native dual-frame capture with the optional component request disabled: 1 failed, 3,804 passed, 316 skipped, 55 deselected, 1,263.39 s. The fault occurs inside `_capture_nephele_acceptance`, before Python receives its arrays. This rules out the additional component request as a necessary cause. The crashed capture emitted no verified adapter identity; references to software Vulkan above describe the hosted lane assumption, not verified capture metadata. Its native root cause remains unresolved. The background assertion remains active and failing on hosted Linux; it has not been skipped, weakened or converted into a pass.

Required PR core [36822973299](https://github.com/milos-agathon/forge3d/actions/runs/36822973299) and the separate hosted runner configurations passed on the same revision. All three wheel builds and compatibility lanes passed. **Broader core CI is not green.** No further native/rendering investigation was undertaken: G3/G4 already trigger the owner's bounded-round stop rule. Reopening the Linux native capture problem requires a separately authorized extension of this round; routing or skipping the failing test also requires owner approval under `AGENTS.md`. The final audit-only push does not change these native/test inputs; its required PR result is reported separately.

## Release and stop decision

[v1.40.0](https://github.com/milos-agathon/forge3d/releases/tag/v1.40.0) was already published at 2026-09-30 23:11:23 UTC, resolving to main commit `2c5fb1614d3719fc7eaab9efb903f02a9f24e60c`. Main's core run [36788022640](https://github.com/milos-agathon/forge3d/actions/runs/36788022640) succeeded. The red scheduled full run [36694460997](https://github.com/milos-agathon/forge3d/actions/runs/36694460997) ran the older `6937017c4345d13939da5251410872a5a2829327`: Rust determinism instrumentation failed with 378 fixable IR violations, and the NVIDIA ORBIS evidence rejected dirty downloaded DEM assets despite its test groups passing. A full-acceptance green result on the released revision was not observed. This round did not create or replace a release/tag.

OD-10 is recorded as **NOT_PROVEN / NEPHELE unreleased**. A larger viewport with the same crop would require an explicitly authorized new fixture revision, regenerated reference and approval. That authority was not supplied, and no fixture change was made. H1 and H2 were not run because the local G3/G4 prerequisite failed. Cross-backend physical portability is ABSENT; D3D/Metal/WASM physical render lanes were not exercised. Branch core CI is reported separately after push and cannot convert these failed physical gates into delivery.

Execution and final review were parent-only; no subagents were used. The reusable workflow gap is that strict capture rejects untracked audit outputs, while newly authored scripts match repository ignore rules. A small note in the local validation guide about staging output under `.tmp` until capture completes, force-adding intended new scripts, using explicit UTF-8 reads on Windows, and invoking the installed `maturin` CLI would prevent the observed false starts. No agent contract, skill or registry was changed.

## Round 2 — revert, first-order split and hosted-Linux crash

Round 2 had one diagnosis budget: decide one physically defined fix, with no fitting. If G3/G4 still failed after it, NEPHELE stays NOT_PROVEN. The fixture, masks, thresholds, frozen 640-spp reference, registered 320-spp control and goldens are unchanged.

### Sky-attenuation revert

Commit `7945d1b0` restores the terrain shader, its contract, the group-4 layout/binding, the render-graph input, the media sky volume and the verifier pin (`0xe3539a322a87b8cc`) to `48d6a052^`. The terrain shader is byte-identical to `9e5cbb96`. The atmosphere test again asserts five `@group(4)` occurrences (one ownership comment plus four resources). The optional scattering-component readback, its diagnostic and the background fix are kept.

| Check (release wheel `b870961b…`, RTX 3070, Vulkan, NVIDIA 610.60) | Result |
|---|---|
| B1 against the registered 320-spp control | 1,546/1,558 (99.2297818%); mean 0.33803308, p95 0.86612297, max 6.53942811. Identical to round 1. |
| Rust `verify::` + `terrain::renderer::` (proof, ablation, source pins, renderer contracts) | 177 passed, 1 ignored (existing inspection-only test) |
| `tests/test_shader_proofs.py` | 8 passed |
| Nine NVIDIA goldens, albedo regression, both readbacks, atmosphere contract | 41 passed |
| NEPHELE suite (evidence report, fixture contracts, public API) | 155 passed |

None of these runs skipped anything. They used the round-1 environment variables plus `FORGE3D_TERRAIN_GOLDEN_VARIANT=nvidia-vulkan` for the goldens. No golden-update flag was set.

**Unlogged:** the wheel SHA `b870961b…` and the counts 177/1, 8, 41 and 155 in this table come from local runs whose output was not committed. They cannot be checked from the repository. The B1 values match the committed round-1 evidence, but the round-2 rerun that reproduced them was not logged either.

### First-order split

[`round2/nephele_first_order.py`](nephele-bounded-round-2026-10-01/round2/nephele_first_order.py) is a deterministic quadrature over the frozen fixture: 48 camera steps and 48 sphere directions. It was validated against the reference AOVs:

- terrain-hit mask: 2/4,096 mismatches;
- camera transmittance: mean |ΔT| 0.006;
- cloud-shadow transmittance: mean |Δ| 0.007.

[`first_order_compare.py`](nephele-bounded-round-2026-10-01/round2/first_order_compare.py) recovers linear radiance by inverting the fixture's sRGB8 + ACES presentation; the round trip of the captured frame is exact. The real-time terms are the medium-only arrays of the reverted state. The table gives mean luminance ([comparison.json](nephele-bounded-round-2026-10-01/round2/first-order/comparison.json)).

| Term | Cloud-shadowed terrain (1,557 px) | Sky/cloud (2,538 px) |
|---|---:|---:|
| Reference beauty | 0.2642 | 0.3607 |
| Real-time beauty | 0.2458 | 0.3482 |
| First-order total | 0.1536 | 0.2441 |
| **Higher-order remainder** (reference − first order) | **0.1106** | **0.1167** |
| Camera transmittance, first order / real time | 0.7064 / 0.7012 | 0.3485 / 0.3528 |
| Single scatter, first order (sun + sky) | 0.0464 (0.0309 + 0.0155) | 0.1569 (0.1068 + 0.0502) |
| Single scatter, real time | 0.0721 | 0.2009 |
| Multiple scatter, real time | 0.0262 | 0.0594 |
| Surface radiance, first order (sun + attenuated sky) | 0.1538 (0.0648 + 0.0891) | — |
| Surface radiance, real time | 0.2147 | — |

Reading the split:

1. **Real-time single scatter versus first order.** These are not like for like. In `cs_nephele_inject_single`, the real-time "single" term takes one phase sample per froxel. When that sample hits terrain, it adds lit-terrain radiance (`terminal_radiance`), which is a surface-to-medium second-order path. The surplus of 0.026 on terrain and 0.044 on sky is therefore not evidence of a single-scatter bias.
2. **Real-time surface versus first order.** The surplus of 0.061 is entirely unattenuated sky irradiance: the real-time sky term is about 0.150, against an attenuated first-order 0.089. In this medium, with single-scattering albedo about 0.85, out-scattered sky light is mostly replaced by in-scattered light. That is why attenuation alone (round 1) went to G4 6.5%.
3. **Remainder.** The remainder is 0.111 on terrain and 0.117 on sky. It holds multiple scattering, interreflection, and light scattered from the sunlit medium onto the terrain. One of these terms was computed by [`surface_cloudlight.py`](nephele-bounded-round-2026-10-01/round2/surface_cloudlight.py): sun single-scattered by the medium onto the terrain hit point, with 192 directions. After camera transmittance it gives 0.0180, which matches the real-time terrain luminance deficit of 0.0184. It is the best-defined missing physical term.

**Decision: fix chosen, not implemented.** The chosen fix is the medium-to-terrain sun term above. The contract reads "decide one physically defined fix… if G3/G4 still failed after it", that is, apply and then measure. This round departs from that wording: it did not implement or render the fix, because two gates already fail independently of it:

- **G4 ablation:** G4 also requires the terrain-occlusion-ablation SSIM to be below 0.80. It measures 0.9578603137 (`terrain_occlusion_ablation_ssim` in [background-local/visual-metrics.json](nephele-bounded-round-2026-10-01/background-local/visual-metrics.json)), which is the state the revert restores. The candidate only adds sun light scattered by the medium onto the terrain surface. That it leaves the ablation SSIM unchanged is reasoned, not measured. A move from 0.958 to below 0.80 is very unlikely, but it was not tested.
- **G3:** G3 scores sky/cloud pixels only, and the candidate changes terrain pixels only, so G3 stays at 91.69% (below its 95%).

[`oracle_bounds.py`](nephele-bounded-round-2026-10-01/round2/oracle_bounds.py) scores candidates with the production ΔE and the unchanged masks ([oracle-bounds.json](nephele-bounded-round-2026-10-01/round2/first-order/oracle-bounds.json)):

| Candidate | G4 shadow ΔE<2 (≥95%) | G3 sky ΔE<2.5 (≥95%) |
|---|---:|---:|
| Real time, reverted state | 46.56% | 91.69% |
| Reference noise floor (320 vs 640 spp) | 99.61% | 100% |
| Real time + the medium-to-terrain sun term (offline prediction, not rendered) | 60.18% | 91.69% |
| Mean-matched per-channel gain over all terrain (estimate, not a maximum) | 71.36% | — |
| Mean-matched per-channel gain over all sky (estimate, not a maximum) | — | 94.37% |
| Mean-matched per-channel gain per 8×8 froxel tile (estimate, not a maximum) | 83.69% | 97.20% |

The 60.18% is a prediction: the term is added offline to the captured frame (`rt + T*S2`), not rendered. The mean-matched rows pick gains from the reference itself, so they are not physical fixes. They are also **not upper bounds**. Each gain is the per-channel ratio of means over the whole terrain or sky mask (or the tile intersected with it). It is not the gain that maximises the ΔE pass fraction, and G4 scores only the cloud-shadowed subset. A gain chosen on the scored mask, or one chosen to maximise the pass fraction, could score higher. These rows therefore do not show that no uniform or per-tile correction could reach 95%. The stop decision does not rest on them; it rests on the ablation SSIM and on G3.

Implementing the candidate would cost a new terrain pin, goldens and a B1 rerun with no prospect of changing the outcome. The stop rule therefore applies: **NEPHELE remains NOT_PROVEN / unreleased; OD-10 still needs fresh evidence.** The sub-tile G4 error is consistent with the froxel-resolution concern, but this does not prove that resolution is the only cause. H1 was not run.

**Owner sign-off (2026-10-01):** Milos approved recording the fix as chosen but not implemented, for the reasons above, on condition that the reasons cite the ablation SSIM and G3 rather than the mean-matched rows, that 60.18% is labelled a prediction, and that the unchanged ablation SSIM is labelled an inference.

### Hosted-Linux segfault: root cause and fix

The CI-identical environment was a WSL Ubuntu 24.04 with `mesa-vulkan-drivers 25.2.8-0ubuntu0.24.04.3` (llvmpipe, LLVM 20.1.2), the exact CI package, and a cold shader cache. In it, the hosted CI wheel crashed in both `include_no_medium=False` and `True` captures. The reverted-source Linux build `3f208f88…` crashed the same way.

Evidence, in [round2/linux-crash/](nephele-bounded-round-2026-10-01/round2/linux-crash/):

- **gdb with matching Mesa debug symbols:** SIGSEGV in lavapipe's queue thread at `lvp_execute.c:4956`, while iterating the command list of a submitted command buffer.
- **Valgrind:** an invalid read in JIT vertex code, at an address that is "not stack'd, malloc'd or free'd".
- **Khronos validation:** 15× `VUID-vkResetCommandPool-commandPool-00040` (pool reset while a command buffer is pending) and 3× `VUID-vkDestroyBuffer-buffer-00922` (buffer destroyed while in use).

Root cause: wgpu-core 0.19.4 `Device::maintain` (`device/resource.rs:322-338`) handles `Maintain::Wait` like this:

1. It waits at most `CLEANUP_WAIT_MS` (5,000 ms).
2. It discards the timed-out result.
3. It sets `last_done_index` to the waited index.
4. `triage_submissions` then resets those command pools and frees resources.

On lavapipe with a cold LLVM shader cache, a single submission takes more than 5 s. The queue thread then executes freed memory. This explains why the crash depends on the shader cache and only appears on hosted CI. Native drivers keep their own references, which also hides the defect there.

Fix (no dependency change):

- `crate::core::gpu::wait_for_device_idle` repeats `Maintain::Poll` until the queue is empty. Poll reads the real fence value, so no unfinished submission is ever retired.
- On the GL backend it keeps `Maintain::Wait` (round-2 review F3). wgpu-hal 0.19.5 GLES creates the submission fence without flushing (`gles/queue.rs`, `fence_sync` after the command list) and `Poll` only reads its status (`gles/mod.rs` `Fence::get_latest`), so an unflushed fence may never signal and a poll loop could spin forever. Only the `Wait` path flushes (`client_wait_sync(SYNC_FLUSH_COMMANDS_BIT)`, `gles/device.rs:1460`). GL hands commands to the driver at submit time and defers deletion of objects still in use, so the 5 s early retirement is harmless there.
- The backend is read from the device's own id: wgpu-core 0.19.4 stores it in the top 3 bits (`id.rs` `BACKEND_SHIFT`, value 4 = GL), and native `Device::global_id().inner()` is that raw id. `Id::inner` is a hidden testing API, so a unit test pins it (`core::gpu::backend_request_tests::wait_for_device_idle_classifies_gl_and_returns_after_real_work`). The test creates a device on every local adapter and keeps them all alive at once, checks each classification against the adapter backend, and waits for a real 1 MiB copy on each.
- The first version (`d8f90eb1`) used `Device::as_hal::<Gles>` instead (round-2 re-review). wgpu-core's `as_hal` lookup (`storage.rs` `try_get`) indexes the GL registry by slot and does not check the backend, so a live GL device in the same slot made a Vulkan device look like GL, or would panic on the epoch check. With all devices alive, that version failed the new test on the RTX 3070: the Vulkan device was classified as GL ([as-hal-check-before.log](nephele-bounded-round-2026-10-01/round2/gl-wait/as-hal-check-before.log)). The id-based check passes on Vulkan, DX12 (RTX 3070 and Microsoft Basic Render Driver) and GL ([id-backend-check-after.log](nephele-bounded-round-2026-10-01/round2/gl-wait/id-backend-check-after.log)).
- The non-GL loop still has no time limit: a submission that never completes and never reports device loss would spin forever. Device loss on Vulkan, DX12 or Metal ends in wgpu's fatal-error panic, not a hang. Any timeout is an owner decision; none was added.
- 113 statement-form `poll(Maintain::Wait)` calls in the native crate use it.
- Eight calls stay in three frozen reference-transport sources: `media_reference.rs`, `render_terrain.rs` and `terrain_heightfield.rs` under `src/path_tracing/hybrid_compute/`. The live-source provenance check rejects any edit to those files, and none of the eight runs during the real-time capture. They keep the latent defect for offline reference/hybrid renders on slow devices.
- Three `Maintain::Wait` calls inside `resource_tracker.rs` unit tests stay, because that file is also compiled for wasm.
- The 16 `WaitForSubmissionIndex` calls, in vector/fence-tracker code outside this capture path, keep the same latent 5 s defect. They are unchanged and unproven.
- Four `Maintain::Wait` calls in `bench/upload_policies/policies.rs` (lines 193, 229, 240, 245) stay. That file is the shipped `policies` `[[bin]]` benchmark target (`Cargo.toml`, `required-features = ["async_readback"]`), outside the native library, so it keeps the same latent 5 s defect on slow devices.

Results with the fixed Linux build `cb5c33cc…` on the same CI-identical Mesa and cold cache:

- `media_only` and dual captures: pass.
- Khronos validation: 0 VUIDs.
- `tests/test_nephele_environment_defects.py`: both tests pass (2 passed, 112.6 s), with the background pixel assertion unchanged and active.

Results with the fixed Windows wheel `1e95fe05…`:

- B1 frame bytes identical to the pre-fix capture.
- Rust `verify::`/`terrain::renderer::`/`core::gpu`: 183 passed, 1 ignored.
- Goldens, readbacks, atmosphere, API contracts and source contracts: 266 passed. The 7 skips are 6 ORBIS lanes gated on `FORGE3D_RUN_ORBIS_GPU`, plus one shader-proof case skipped by process ordering.
- `test_shader_proofs` alone: 8 passed.

The local Linux interpreter was Python 3.12.3; CI uses 3.11 with the same abi3 wheel.

**Unlogged:** the wheel SHAs `3f208f88…`, `cb5c33cc…` and `1e95fe05…`, and the counts 183/1, 266 with 7 skips, and 8 in the Windows list, come from local runs whose output was not committed. They cannot be checked from the repository. The committed [`pytest-fix-env-defects.log`](nephele-bounded-round-2026-10-01/round2/linux-crash/pytest-fix-env-defects.log) records the 2 passed in 112.60 s but does not name the wheel or commit it ran on. The committed gdb, valgrind and validation-layer logs back the crash evidence above. The post-fix validation log shows only `OK media_only`; it has no positive sign that the layer loaded, though it came from the same script and environment as the pre-fix log, which did report errors.

All of these local wheel checks were built before the GL branch above was added. Every wait now first reads the device id to classify the backend; on Vulkan, DX12 and Metal it then runs the same poll loop as before. The local wheel and Python checks were not repeated after this change; hosted results for the final head are reported separately.

The first push (`5b6a875d`) also edited the three frozen reference sources. In broader run [36845903178](https://github.com/milos-agathon/forge3d/actions/runs/36845903178), both `test_nephele_environment_defects` tests passed on hosted Linux. The three live-reference-source provenance checks failed: 3 failed, 3,802 passed, 316 skipped, 55 deselected. The follow-up commit restores those three files byte-for-byte. With that source, the NEPHELE suite passes locally (155 passed, zero skips; unlogged). The Linux and Windows wheel checks above were built before this restore. The restored files are outside the real-time capture path. Hosted results for the final head are reported separately.

