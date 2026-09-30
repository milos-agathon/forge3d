# NEPHELE reference convergence measurements, 2026-09-30

These are reference-versus-reference noise measurements, **not G3/G4 physical
acceptance**. The fixture remains UNRESOLVED until its tracked exact-prefix
convergence record passes every unchanged criterion.

## Measured baseline

Source: `5b5be1d766ce717cd59dc4e5d452b0b005c52726`, algorithm
`integrated-hybrid-terrain-ratio-delta-tracking-surface-env-mis-v2`.
Windows x64, NVIDIA GeForce RTX 3070, Vulkan, NVIDIA driver 610.60.
Both `WGPU_BACKEND=vulkan` and `WGPU_BACKENDS=vulkan` were set, with
`FORGE3D_NO_BOOTSTRAP=1` and `FORGE3D_TEST_INSTALLED_WHEEL=1`.
The retained local wheel's SHA-256 is
`72f05f82a23990f9bab54a1407ecd96f10e8a56c8d969e2551df4f8bbbd638ca`;
its native member SHA-256 is
`22caf702bc6d9625842141f7c534f4ab35179273c52a75d6d781ebe4bbfd49b7`.

Command, from a clean checkout of that revision:

```powershell
python -m scripts.generate_media_fixture --samples-per-pixel N --native-wheel dist-nephele/forge3d-1.39.0-cp310-abi3-win_amd64.whl --workers 1
```

The full 64 by 64 fixture, seed, camera, color pipeline, and scene inputs were
unchanged. Each pixel used the exact sample prefix `[0, N)`.

| Samples per pixel | Wall seconds | Seconds per spp |
| --- | ---: | ---: |
| 10 | 64.3841 | 6.4384 |
| 20 | 132.8264 | 6.6413 |
| 40 | 247.4329 | 6.1858 |
| 80 | 462.4669 | 5.7808 |

At 10 spp, 24 workers took 67.1683 seconds. Every saved numeric array and mask
was exactly identical to the one-worker result. More workers did not improve
this measured workload; subsequent generations used one worker.

| Prefix comparison | Sky/cloud fraction, dE < 2.5 | Shaft ROI SSIM | Shadow terrain fraction, dE < 2 | Changed shadow-mask pixels |
| --- | ---: | ---: | ---: | ---: |
| 10 to 20 | 0.019701 | 0.768067 | 0.008387 | 6 |
| 20 to 40 | 0.077620 | 0.851463 | 0.038635 | 3 |
| 40 to 80 | 0.198976 | 0.923762 | 0.072026 | 2 |

Every row fails the unchanged 0.95 / >0.95 / 0.95 requirements. All other
masks were identical. Mask identity is also required for approval; increasing
the sample count does not authorize substituting or editing masks.

In the 10-to-20 comparison, failing shadow pixels had luminance-only dE RMS
3.52095 and chroma-only dE RMS 14.00324. The analogous sky/cloud values were
2.78819 and 13.22893. These diagnostic decompositions hold the other Lab
components fixed; they are not additive CIEDE2000 components or new gates.
They show substantial color noise and motivate the plan's weighted RGB
reference proposal. The production single-channel tracking functions used by
G1/G2 remain unchanged.

## Owner runtime decision

Milos set a **12-hour wall-clock budget per reference generation** on this
Windows NVIDIA machine, and required the 20-to-40 measurement for the batching
decision. Individual GPU terrain queries must remain if the projection is at
most 12 hours; a greater projection requires batched queries with demonstrated
parity before generation.

At 20-to-40, the 95th-percentile shadow-terrain error was dE 20.2488298.
Assuming sampling noise shrinks as the inverse square root of the sample
count gives `40 * (20.2488298 / 2.0)^2 = 4100.15` spp. The corresponding
sky/cloud projection is `40 * (16.4555666 / 2.5)^2 = 1733.03` spp.
The governing shadow estimate rounds up to **5120 spp** in the prescribed
doubling sequence. At `247.4328747 / 40 = 6.1858219` seconds per spp, one such
generation projects to **8.7976 hours**, below the owner's budget.

**Decision: retain individual GPU terrain queries; do not adopt batching.**
This is an extrapolation, not proof of convergence or an authorized relaxation.
The weighted RGB reference will be measured separately and use a new algorithm
identifier; its outputs cannot be mixed with these baseline prefixes.

The local raw prefixes and provenance are retained under
`D:/forge3d/cache/nephele172-5b5be1d7-prefixes/spp{10,20,40,80}`. They are
diagnostic records, not committed acceptance fixtures or SHA-bound G1-G6
evidence. The final acceptance artifacts must be tracked separately.

## Weighted RGB reference

Source: `6a4ae50a4c3c01b390b1222468632e88d1db6ca0`, algorithm
`integrated-hybrid-terrain-spectral-ratio-delta-tracking-surface-env-mis-v3`.
The scene inputs, sample identities, adapter, environment variables, and
individual GPU terrain-query path are unchanged. RGB channels share a path,
with real/null-event likelihood weights preserving each channel's transport.

| Samples per pixel | Wall seconds | Seconds per spp |
| --- | ---: | ---: |
| 10 | 65.7829 | 6.5783 |
| 20 | 117.7482 | 5.8874 |
| 40 | 260.8402 | 6.5210 |
| 80 | 544.8317 | 6.8104 |
| 160 | 1121.0426 | 7.0065 |
| 320 | 1786.6192 | 5.5832 |
| 640 | 4063.4773 | 6.3492 |

| Prefix comparison | Sky/cloud fraction, dE < 2.5 | Shaft ROI SSIM | Shadow terrain fraction, dE < 2 | Changed shadow-mask pixels |
| --- | ---: | ---: | ---: | ---: |
| 10 to 20 | 0.847912 | 0.935726 | 0.587742 | 6 |
| 20 to 40 | 0.947991 | 0.949899 | 0.704443 | 3 |
| 40 to 80 | 0.988968 | 0.971742 | 0.848875 | 2 |
| 80 to 160 | 0.997636 | 0.981909 | 0.931877 | 1 |
| 160 to 320 | 1.000000 | 0.990388 | 0.982659 | 1 |
| 320 to 640 | 1.000000 | 0.994228 | 0.996146 | 0 |

The weighted 20-to-40 shadow p95 is dE 4.18913111. The same noise projection
gives `40 * (4.18913111 / 2)^2 = 175.488` spp; rounding up in the exact-doubling
sequence gives 320 spp. At 6.521004 seconds per spp, this projects to 34.78
minutes per generation. The owner's no-batching decision therefore remains
applicable. These estimates do not replace observed convergence.

At 320 spp every numeric convergence criterion passed, but one shadow-mask
pixel changed. The 640-spp doubling was therefore required to establish mask
identity as well.

The authoritative recorder approved the **640-spp reference** against its exact
320-spp prefix. Every mask and terrain classification is identical. All three
numeric criteria pass without changing any threshold, scene input, or mask
rule. The final generation took 67.7246 minutes, below the owner's 12-hour
budget. The fixture, manifest, 320-spp prefix, and generating wheel are retained
together. This proves reference convergence; it is not candidate G1-G6
physical acceptance.

The weighted reference wheel is `forge3d-1.40.0-cp310-abi3-win_amd64.whl`,
SHA-256 `e1e1f8444ff0300bf63ec24963b70a1578871647be2a3a9753e7b3a26dd7928e`.
Its `forge3d/_forge3d.pyd` member has SHA-256
`7661b8f81f6a5253d035b8d2e6241ccf73e8b1c001c4db6981b5c65a3012607f`.

## T4/B1: owner decision OD-9 required

The medium-disabled real-time terrain was compared with a vacuum reference
(sigma_a = sigma_s = zero) on the frozen terrain mask, on the same Windows
RTX 3070 Vulkan adapter and source `6a4ae50a4c3c01b390b1222468632e88d1db6ca0`.
Camera contracts matched exactly; all tracked scene-input hashes were checked.
This is the prescribed vacuum diagnostic, not a G3/G4 candidate observation.

The vacuum reference used the existing 10, 20, 40 spp doubling sequence.
20-to-40 convergence passed: 96.7908% of 1,558 terrain pixels have dE < 2,
mean 0.644416, p95 1.916096. The 40-spp generation took 94.5785 seconds.

| Medium-disabled vs vacuum reference | Measured |
| --- | ---: |
| Terrain pixels | 1,558 |
| Fraction with dE < 2 | 0.172657253 |
| Required fraction | 0.95 |
| Mean dE | 8.787831 |
| p95 dE | 19.033360 |
| Maximum dE | 31.024163 |

This fails B1 and triggers OD-9. Work is stopped pending the owner's choice:
revise fixture material inputs before any candidate G3/G4 comparison, or extend
the reference BRDF to the shipped terrain shading; either requires regeneration.
No threshold or mask has been relaxed. These measurements establish a parity
gap, not its detailed cause. G1-G6 acceptance, merge and release remain pending.

Raw diagnostic arrays, source, source/adapter metadata, report and SHA-256
inventory are retained in `docs/audits/nephele-b1-2026-09-30/`. The diagnostic
source preserves its original local paths; it is an audit artifact, not a new
supported CLI. `physical_acceptance` is false in its report.
