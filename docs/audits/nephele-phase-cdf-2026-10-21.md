# Phase sampling and physical capture — 2026-10-21

**CPU-validated sampler correction; physical acceptance remains failed.**
No subagents, local GPU tests, reference edits, mask changes, golden changes,
pin changes, test changes, or additional gate-policy changes were used.

## Exact-head capture and local proof

NVIDIA run [39566600606](https://github.com/milos-agathon/forge3d/actions/runs/39566600606)
used `ac7390968bcba2b393ad4fbea58716f6dc1acd59`, RTX 3070, Vulkan, driver
610.60. Two independent captures completed. Gates 1, 2, 5 and 6 passed;
Gates 3 and 4 failed: **4 passed, 2 failed, zero skipped**.

| Metric | Measured | Required |
| --- | ---: | ---: |
| B1 sky/cloud DeltaE2000 pass fraction | 0.7884160756501182 | >= 0.95 |
| God-ray reference SSIM | 0.9916078255428148 | > 0.95 |
| G4a shadowed-terrain DeltaE2000 pass fraction | 0.7597944765574823 | >= 0.95 |
| G4b signed mean linear luminance response | 0.020363695197357472 | reference +/-20% |
| G4b relative error | 0.45430131050294364 | <= 0.20 |

All 14 remaining canonical fixture/mask NPY files match the preceding capture
byte-for-byte. G4b uses the reference and tolerance explicitly approved by
Milos; neither is changed here.

Local checks on the clean `ac739096` wheel passed: 654 Python fast cases,
zero skips, with only the exact owner-approved GPU-device memory case
deselected; CI-feature Rust verification passed 108 cases with one existing
ignored IR inspection test. The installed native binary contains the full
candidate SHA. Both Python environment flags were `1`:
`FORGE3D_NO_BOOTSTRAP` and `FORGE3D_TEST_INSTALLED_WHEEL`. Full CI and release
were not started on this failing physical candidate.

## Joint CDF defect

Both real-time ray generation and froxel injection used a shifted neighbouring
rank for their second phase-sampling coordinate. The existing blue-noise tile
is designed to spatially separate ranks; its diagonal neighbour is strongly
anti-correlated with the original rank. It is unsuitable as the second CDF
dimension of this phase sample.

Across the complete existing 64-rank tile:

| Quantity | Shifted neighbour | Rank bit reversal |
| --- | ---: | ---: |
| Occupied joint 8x8 CDF cells | 8 | 64 |
| Empty joint CDF cells | 56 | 0 |
| Maximum samples in one cell | 8 | 1 |
| Coordinate correlation | -0.937546 | 0.140659 |
| Mean tangent component (expected zero) | -0.0478135 | -0.00483744 |
| Mean bitangent component (expected zero) | -0.135399 | 0.00191579 |

The mean axial component remains 0.4500735 for the fixture's `g=0.45`.
The replacement keeps the first rank, reverses its 32 bits, divides by the
u32 range `2^32`, and adds the existing half-rank-bin offset. For these 64
ranks, this is the six-bit radical inverse and fills every joint CDF cell
once. All coordinate arithmetic is exactly representable in f32. The rank
asset and sample count are unchanged; no gain or constant was fitted.

`phase_direction` and `nephele_phase_direction` are updated together so the
precomputed terrain hit and the consuming continuation ray still agree.
This fixes a sampling defect; it does not establish complete physical
transport or prove a native visual pass.

The [CPU reproducer](nephele-phase-cdf-2026-10-21.py) and
[JSON](nephele-phase-cdf-2026-10-21.json) record the data and candidate source
hashes. Reproduce after downloading the run's evidence artifact:

```powershell
python docs/audits/nephele-phase-cdf-2026-10-21.py --artifact-dir .release-evidence/2026-10-21/ci-39566600606 --output .release-evidence/2026-10-21/phase-cdf-recheck.json
```

## Validation and remaining physical gap

The existing production shader analyzer resolved and Naga-validated all
**304** shader assemblies, with zero compute comparison-sampling calls and
zero stale disabled-shadow comments. Acceptance Clippy passed in 2m 22s.
The five froxel entry points also passed standalone Naga validation. Rust
source, its f32 conversion inventory and the pinned terrain shader are
unchanged; their existing proof is retained for those identical inputs.
Corrected-sampler GPU execution is recorded below; it did not pass physical
acceptance.

A separate CPU reconstruction reproduces the current capture's regional
in-scatter to about 0.001 mean absolute RGB error. A scalar higher-order
fixed-point forecast alone still misses G4a and G4b; it is not a full angular
transport proof and has not been adopted as a renderer fix.

An isolated first-order CPU calculation attributes 0.0154541 of the accepted
ablation response to sunlight. The current coarse-froxel model produces
0.0145223, about 94% of that sunlight response. The environment contribution
is the larger deficit: 0.0218626 in the reference estimate versus about
0.00431369 in the coarse model before subtracting terrain-reflected return.
The remaining bounded investigation therefore measures environment angular
visibility and valid-air source reconstruction. It changes no acceptance
threshold or frozen input.

## Corrected-sampler NVIDIA result and owner scope decision

Run [39580065706](https://github.com/milos-agathon/forge3d/actions/runs/39580065706)
executed `68cc04e9ab87de3d90e92e76a8690e016e5d759b` on NVIDIA GeForce RTX
3070, Vulkan, NVIDIA 610.60. Checkout and both capture diagnostics identify
that exact SHA. Fast Contract and the Windows wheel passed. The physical
suite recorded **4 passed, 2 failed, zero skipped**: Gates 1, 2, 5 and 6
passed; Gates 3 and 4 failed. Full Acceptance Summary failed.

| Metric | Measured | Required |
| --- | ---: | ---: |
| B1 sky/cloud DeltaE2000 pass fraction | 0.7127659574468085 | >= 0.95 |
| God-ray reference SSIM | 0.9916001251541653 | > 0.95 |
| G4a shadowed-terrain DeltaE2000 pass fraction | 0.7109826589595376 | >= 0.95 |
| Medium-ablation changed fraction | 0.3215661103979461 | >= 0.10 |
| G4b signed mean linear luminance response | 0.019629758417801648 | reference +/-20% |
| G4b relative error | 0.4739690738875221 | <= 0.20 |
| G4b SSIM diagnostic | 0.9894866614716061 | diagnostic only |

All 14 canonical fixture/mask NPY files are byte-identical to `ac739096`.
The approved policy and acceptance implementation are also unchanged. The
sampler correction fixes the joint-CDF defect but does not fix the visual
acceptance failures; no native pass is inferred from its CPU properties.

The [measurement JSON](nephele-phase-cdf-nvidia-2026-10-21.json) records the
test outcomes, exact metrics, fixture hashes and raw artifact hashes. Its
[CPU-only reproducer](nephele-phase-cdf-nvidia-2026-10-21.py) reads the
downloaded artifact and the existing gate functions; it performs no render.

Milos answered **"Keep the current bounded scope"** on 2026-10-21 in response
to the coupled environment/multiple-scattering proposal. That expansion is
not authorized and is not started. No further renderer, test, gate, pin,
golden or reference changes follow from these measurements.

Release remains blocked by the observed physical failures. Full-scope CI
has not been dispatched on this failing candidate, and this task has not
merged the PR, pushed either release tag or published to PyPI. The new
candidate's local installed-wheel fast lane and full Rust verification were
not repeated after the shader-only change; the preceding exact-head proof
above applies to `ac739096`, while the shader checks and NVIDIA execution
above cover the sampling correction. No local GPU tests ran.
