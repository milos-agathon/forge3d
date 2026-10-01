# NEPHELE bounded remediation round — 2026-10-01

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

## Release and stop decision

[v1.40.0](https://github.com/milos-agathon/forge3d/releases/tag/v1.40.0) was already published at 2026-09-30 23:11:23 UTC, resolving to main commit `2c5fb1614d3719fc7eaab9efb903f02a9f24e60c`. Main's core run [36788022640](https://github.com/milos-agathon/forge3d/actions/runs/36788022640) succeeded. The red scheduled full run [36694460997](https://github.com/milos-agathon/forge3d/actions/runs/36694460997) ran the older `6937017c4345d13939da5251410872a5a2829327`: Rust determinism instrumentation failed with 378 fixable IR violations, and the NVIDIA ORBIS evidence rejected dirty downloaded DEM assets despite its test groups passing. A full-acceptance green result on the released revision was not observed. This round did not create or replace a release/tag.

OD-10 is recorded as **NOT_PROVEN / NEPHELE unreleased**. A larger viewport with the same crop would require an explicitly authorized new fixture revision, regenerated reference and approval. That authority was not supplied, and no fixture change was made. H1 and H2 were not run because the local G3/G4 prerequisite failed. Cross-backend physical portability is ABSENT; D3D/Metal/WASM physical render lanes were not exercised. Branch core CI is reported separately after push and cannot convert these failed physical gates into delivery.

Execution and final review were parent-only; no subagents were used. The reusable workflow gap is that strict capture rejects untracked audit outputs, while newly authored scripts match repository ignore rules. A small note in the local validation guide about staging output under `.tmp` until capture completes, force-adding intended new scripts, using explicit UTF-8 reads on Windows, and invoking the installed `maturin` CLI would prevent the observed false starts. No agent contract, skill or registry was changed.
