# DIFFERENTIA item 03 review — 2026-09-20

## Assessment

**Full.** The admitted-event gradient implementation closes the local
source-ray and numerical-bound proof gap, and the required exact hosted NVIDIA
Vulkan acceptance completed successfully against clean candidate
`b4f3dd3afab442615c2e15c97febb81cd34a7311`.

## Implemented and observed locally

- AEQUITAS CIEDE2000 scoring is packaged in production Python. A freshly
  rebuilt `forge3d-1.35.0-cp310-abi3-win_amd64.whl` was installed into an
  isolated Python 3.13 environment. On a physical NVIDIA GeForce RTX 3070
  (Vulkan, driver 610.60, discrete GPU, no software fallback), the exact
  inverse solver test pair passed 5/5 with JUnit failures=0, errors=0, and
  skips=0. Recovery measured median ΔE2000 2.621349898823963, sun error
  0.42385607882760007 degrees, turbidity relative error
  0.0730443000793457, intensity relative error 0.1656758387883504, and
  peak host-visible memory 147456 bytes (the enforced limit is 536870912).
  The target/recovered beauty and ground-truth/recovered albedo PNGs were
  inspected. Local evidence is under
  `D:/forge3d/artifacts/build/differentia-followup/acceptance-source-proof-final/`.
- The mapped-spectral score uses the complete nonlinear pixel loss and
  counts fresh and spatial categorical draws once. The owner-approved
  replacement removed the unsound per-realization strict-improvement claim.
- `geometry_cert.rs` uses outward-rounded intervals to enumerate smooth
  grazing, bilinear crease/perimeter, tmin cutoff, and below-surface cell
  entry events. It checks both receiver charts, first-admissible-hit order,
  camera density and signed boundary velocity. It rejects unresolved
  intervals rather than emitting a record. Independent scoped static review
  found no concrete geometry error in those paths.
- Each accepted edge audit now captures the actual shader source snapshot:
  frame/sample seed, terrain dimensions and flags, light/environment inputs,
  albedo bilinear taps and weights, environment lookup, sampled spectrum,
  and the independently drawn IBL random pair. The host interval-replays the
  keyed xorshift draw, cosine direction, normal offset, source texture and
  atmosphere evaluation before checking terrain first-hit and visibility.
  A ray, source value, cell, map texel, or branch that cannot be enclosed on
  one certified path rejects the event. Test-only corruptions of source IBL
  direction, albedo, environment value, and map lookup reject even when the
  reported branch, loss, and gradient are left unchanged.
- The numerical certificate outwardly encloses f32 event upload weights,
  source radiance, complete nonlinear loss, atomic accumulation, replay
  order, host chunk addition, and final scalar addition. It now rejects a
  division outside WGSL's documented 2.5-ULP divisor domain rather than
  assigning an unsupported tolerance; a finite high-exposure Reinhard
  regression proves that rejection. In the final source suite, observed
  maximum enclosures were IBL direction 0.0020496845245363553, source
  radiance 0.000020086765289306644, nonlinear loss 0.000024408102035522464,
  and atomic rounding no larger than `[0.00003051757812500001,
  0.00012207031250000003, 0.00012207031250000003]`. These are observed
  bounds, not newly added acceptance thresholds.
- The inverse edge shader now consumes certified records and evaluates the
  complete nonlinear loss jump with the selected reservoir history and
  original sample replay. The former adjacent-AOV entry was removed. The
  primal any-hit hierarchy was corrected to retain below-surface cells
  until the leaf's strict interval check.
- Certified events retain their receiver cell. Projection rejects ideal or
  uploaded receivers touching its boundary; the shader also rejects actual
  receiver grid coordinates outside that cell. A regression covers an ideal
  interior coordinate that rounds onto the cell boundary in f32. A separate
  scoped review found no concrete failure in this guard.
- Independent GPU coverage includes smooth/crease, perimeter, cutoff/entry,
  reservoir selections, and a nonuniform environment-map source path. The
  final feature-enabled inverse suite passed 9/9, and the geometry
  certificate suite passed 5/5, including f32 texel-boundary and
  high-exposure division rejection. The full affected feature-matrix Clippy
  gate passed with `-D warnings`. Independent one-sided forward GPU oracles
  continue to cover the admitted terrain boundary classes and selected
  reservoir channels; finite differences are corroborating evidence, not the
  source/branch proof.

## Hosted acceptance evidence

The full [hosted workflow 35486972634](https://github.com/milos-agathon/forge3d/actions/runs/35486972634)
completed successfully for the exact clean candidate
`b4f3dd3afab442615c2e15c97febb81cd34a7311`; both `Visual Goldens (NVIDIA
Vulkan)` and `Full Acceptance Summary` succeeded. Its `visual-gpu-evidence`
artifact was downloaded and inspected:

- `inverse/junit.xml` reported 5 tests with zero failures, errors, and skips.
- `adapter-probe.json` recorded a physical NVIDIA GeForce RTX 3070 on Vulkan
  (vendor 4318, device 9348, `DiscreteGpu`, NVIDIA driver 610.60) with
  `software_fallback: false`.
- Recovery measured median ΔE2000 `2.6210417695592048` (< 4), sun error
  `0.42404854307296574` degrees (< 2), and turbidity relative error
  `0.07304167747497559` (< 0.15). Peak host-visible memory was 147456 bytes,
  below the enforced 536870912-byte (512 MiB) limit.
- The hosted target/recovered beauty and ground-truth/recovered albedo PNGs
  were visually inspected. The 96×96 beauty pair and 16×16 albedo pair agree
  with the recovery evidence; supplemental image comparisons measured beauty
  SSIM 0.951509 and albedo SSIM 0.861913.

Every hosted term in `differentia-gradient-contract.md` is now supported by
observed artifact evidence. The ignored local `docs/moonshot-roadmap.html`
tracker is marked **Full** alongside this review.
