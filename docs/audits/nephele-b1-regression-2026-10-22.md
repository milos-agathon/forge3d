# NEPHELE B1 regression — 2026-10-22

**Release incomplete; a transport blocker has not been established.** This is CPU analysis of four existing NVIDIA Vulkan
captures, not a new render. The runtime, reference, fixture, masks, viewport,
policy, presentation inverse and tests are unchanged. Work is isolated at
`D:/forge3d/.worktrees/nephele-146-release`, starting at
`ca4f5bd3dbea63dfc2e9bab7e888c3202a3b1e17`. The unrelated main checkout and
pre-existing worktree were only read.

The [JSON](nephele-b1-regression-2026-10-22.json) records raw input hashes,
per-region errors, per-pixel threshold crossings, capture identities and all
five current image criteria. The [CPU reproducer](nephele-release-analysis-2026-10-22.py)
imports the existing metric/presentation helpers without importing forge3d or
creating a GPU session. All four requested artifacts were downloaded to
`.release-evidence/2026-10-22/ci-<run>/`. Every analyzed file matches the
previously downloaded input byte for byte (36, 36, 19 and 19 files respectively).
An interrupted transfer prompted a duplicate download of the first artifact;
that duplicate reported an existing file. Complete input/hash verification
establishes that the original download had supplied every analyzed file.

## Encoding and path classification

Beauty is uint8 sRGB after the fixture's ACES fitted transform, exposure 1.
The unchanged numerical presentation inverse followed by that transform
round-trips every captured beauty byte. Runtime AOV RGB channels are linear
in-scatter, dimensionless transmittance and optical depth; see
`src/shaders/nephele_froxel.wgsl`, integration/composite entry points.
Mean absolute `T - exp(-tau)` is approximately 0.000245 per channel, consistent
with the AOV's half-float storage. Sky beauty reconstructed as
`in_scatter + T * 0.25` has mean absolute inverse-display RGB error of
0.0012–0.0017, with beauty quantization and AOV rounding retained.

The reference AOVs use the same units, but have different estimator semantics.
`src/path_tracing/hybrid_compute/media_reference.rs` adds the **entire path's
radiance** to reference in-scatter whenever `collision_count > 0`. This also
counts paths that first hit the primary terrain, then scatter. The runtime
AOV integrates source along the camera segment before the primary terrain.
They are not interchangeable terrain decompositions. Reference optical depth
is `-log(mean Monte Carlo T)`, rather than a deterministic extinction integral.
Its independently estimated sky composite has mean absolute RGB residual
0.00375–0.00387. Neither residual is silently treated as an encoding gain or
fitted correction.

## Measured regression

All rows are rescored using the **current unchanged** policy and canonical
640-spp reference. Each linked run's checkout and both capture diagnostics
identify its corresponding head. All identify NVIDIA GeForce RTX 3070,
Vulkan, NVIDIA 610.60.

| Run / head | B1 fraction | Shaft SSIM | G4a fraction | Medium changed fraction | G4b relative error |
|---|---:|---:|---:|---:|---:|
| [39517423981](https://github.com/milos-agathon/forge3d/actions/runs/39517423981), `d38758e1` | 0.955871 | 0.982692 | 0.596660 | 0.386393 | 0.541543 |
| [39545456967](https://github.com/milos-agathon/forge3d/actions/runs/39545456967), `d5507249` | 0.941686 | 0.989971 | 0.673089 | 0.379974 | 0.501505 |
| [39566600606](https://github.com/milos-agathon/forge3d/actions/runs/39566600606), `ac739096` | 0.788416 | 0.991608 | 0.759794 | 0.310655 | 0.454301 |
| [39580065706](https://github.com/milos-agathon/forge3d/actions/runs/39580065706), `68cc04e9` | 0.712766 | 0.991600 | 0.710983 | 0.321566 | 0.473969 |
| Required | >= 0.95 | > 0.95 | >= 0.95 | >= 0.10 | <= 0.20 |

Historical JUnit results are recorded separately from this rescoring. In
particular, the oldest artifact's six historical passes are **not** proof of
six passes under today's policy. The latest physical suite has four passes,
two failures and zero skips.

Camera transmittance and optical-depth arrays are byte-identical across all
four runs. The changing component is camera in-scatter, not extinction or the
sky background. On the 2,538 sky/cloud pixels:

| Change | Newly failing / newly passing | Net additional failures | Mean in-scatter change, linear RGB |
|---|---:|---:|---|
| `d38758e1` → `d5507249` | 60 / 24 | 36 | (-0.003590, -0.003150, -0.002568) |
| `d5507249` → `ac739096` | 443 / 54 | 389 | (-0.013781, -0.011421, -0.010311) |
| `ac739096` → `68cc04e9` | 308 / 116 | 192 | (-0.010736, -0.009235, -0.007360) |

`d5507249` changes both surface cloudlight and the directional sun source used
by multiple scattering. A primary surface-lighting term cannot explain this
sky-only loss: the in-scatter AOV itself falls, while camera T is identical.
The affected shader boundary is `continuation_source` in
`src/shaders/nephele_froxel.wgsl`. Replacing only that capture's in-scatter
with the preceding capture restores B1 to 0.953901; the small difference from
0.955871 retains display/AOV rounding.

`ac739096` bounds continuation to the medium and switches continuation density
and sun queries to world fields. The mean sky in-scatter falls again; restoring
the preceding in-scatter restores B1 to 0.941292. `68cc04e9` changes only the
phase-coordinate pairing at this transport boundary (and its matching terrain
trace). Restoring the preceding in-scatter gives 0.783294 rather than the
observed 0.712766. These capture comparisons identify where the regression
entered; they do not justify reverting physically correct changes or retaining
accidental extra energy from older approximations.

Per-pixel DeltaE2000 maps are retained as
`.release-evidence/2026-10-22/analysis/delta-e2000-<run>.npy`.
The next decision and all candidate failures are recorded in the
[residual study](nephele-residual-study-2026-10-22.md).

## Reproduction and checks

```powershell
python docs/audits/nephele-release-analysis-2026-10-22.py `
  --artifacts .release-evidence/2026-10-22 `
  --models .release-evidence/2026-10-22/continuation-study `
  --first-order .release-evidence/2026-10-22/first-order `
  --output .release-evidence/2026-10-22/independent-b1-recheck
```

The optional model inputs are the preserved historical CPU arrays and source
programs, copied without changing their bytes from the previous worktree;
their hashes are in the residual JSON. Omitting both model options reproduces
the capture/B1 analysis from artifacts and canonical repository inputs alone.
The output directory must be new. The corrected reproducer reads the archived
cloudlight Git blob in memory and never rewrites an evidence input. JSON hash
keys now use actual repository-relative POSIX paths, without `models/` aliases.
Python 3.13 and NumPy 2.5.1 were used. No native-wheel environment flags were
needed: neither analysis executes native code. All 14 canonical fixture/mask
NPY files match `ac739096` byte for byte. The acceptance script, physical test,
gate4 policy, DeltaE and SSIM helpers have an empty diff from `ac739096`.

The measured frame is `68cc04e9`, not the uncaptured worktree HEAD `ca4f5bd3`.
The intervening merge includes renderer cache/water changes; source inspection
does not establish physical equivalence. The independent review reproduced
the capture scores and hashes. Its objections concern the subsequent residual
models and completion claims, corrected in the linked residual study.
