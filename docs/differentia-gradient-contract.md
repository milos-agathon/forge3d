# DIFFERENTIA terrain gradient contract

This records the estimator target for roadmap item 03. It is an implementation
contract, not evidence that the current GPU gradient satisfies it.

## Objective and admitted inputs

For a fixed scene and parameters θ, let `Y[r,p](θ)` be the linear radiance
at pixel `p` from replicate `r` of the shared forward terrain renderer,
including its finite frame and sample count. With `R` independent idealized
replicates and the renderer's display-space per-pixel loss `ℓ`, the target is

`J_R(θ) = E[ (1/P) Σ_p ℓ((1/R) Σ_r Y[r,p](θ), target[p]) ]`.

The gradient claim applies only to flat-earth, terrain-only inverse scenes
whose heightfield is the current bilinear DEM, with positive sun elevation,
finite nonnegative light/surface values, and regular boundary events. A
candidate event is regular only if its root, first admissible hit, event
branch, chart Jacobian, and sampling density can be certified. Coincident
roots, zero Jacobians, unresolved first-hit ordering, and loss or proposal
kinks must be diagnosed and rejected. This is a mathematical continuous
random-draw contract; WGSL f32 error requires separate measurement. Any new
rejection at the public API boundary needs the owner's compatibility approval.

## Required derivative terms

1. Replay the exact shared forward chain and differentiate smooth radiance
   contributions with visibility and discrete selections held fixed. Include
   the derivative of the inverse proposal weight. Apply the chain-rule factor
   `1/R` to each replicate's contribution to the mean image.
2. For each parameter-dependent categorical draw, add its log-probability
   derivative once, multiplied by the **complete** realized image loss. In
   the current mapped spectral specialization, each pixel's reservoir chain
   and loss are local, so that pixel's nonlinear loss divided by `P` is an
   equivalent reward with cross-pixel zero-mean terms removed.
   Fresh terrain spectral candidates count even when later discarded; an
   eligible spatial spectral redraw counts once. The temporal spectral merge
   has parameter-independent selection weights in this specialization.
   A clipped score requires an unbiased tail correction; an unclipped score
   needs no such correction.
3. For each moving cast-shadow boundary, sample a point on the actual
   boundary with known nonzero density. Use the exact loss difference across
   that event, including the affected sample's `1/R` contribution to the
   mean image, rather than the realized image adjoint times a radiance jump.
   Multiply by boundary normal velocity and arclength/proposal Jacobian.

The visibility sampler must cover smooth grazing contacts, bilinear-cell
creases and DEM perimeter, and the shadow ray's `tmin` cutoff. The last case
can occur at a transverse contact inside a blocker cell. Enumerate both
receiver-coordinate charts, filter to the primal's first admissible hit, and
honor its normal-offset ray origin. Primary visibility is detached only
because geometry and camera are fixed by this inverse contract.

## Evidence required before claiming completion

- A completeness argument for the admitted event set and sampling density,
  including diagnosed rejection paths.
- Independent numerical gradient checks on accepted terrain cases with
  smooth grazing, crease/perimeter, cutoff, and reservoir selections; report
  error and GPU conditions. Selected finite differences alone are not a
  completeness proof.
- Installed-wheel AEQUITAS scoring and the zero-skip hosted NVIDIA Vulkan
  acceptance gate, with the three recovery metrics, memory result, and
  inspection images recorded before roadmap status changes.
