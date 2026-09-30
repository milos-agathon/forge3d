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
