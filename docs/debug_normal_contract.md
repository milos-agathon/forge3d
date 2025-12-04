# Debug Normal Contract

This contract describes the canonical packing used by proofpack normal outputs and how to interpret them for analysis.

- **Encoding**: a unit normal `n` in `[-1,1]` is encoded per channel as `encoded = (n * 0.5 + 0.5) * 255`, rounded to `uint8`. Decoding reverses the transform: `decoded = encoded / 255.0 * 2.0 - 1.0`, followed by normalization.
- **Channel order**: `R = x`, `G = y`, `B = z`. No channel swizzling or sign flips are applied.
- **Alpha usage**: alpha is a validity mask. `255` means the normal is valid and must be included in metrics; values `<255` mark pixels that should be excluded from angular-error calculations.
- **Masking rules**: analyses must intersect the validity masks of reference and test images. Pixels with alpha != 255 in either image are ignored. If a dedicated validity mask PNG is provided, it uses the same convention (255 = valid).
- **Current proofpack emitters**: the synthetic proofpack harness emits fully valid normals (alpha = 255 everywhere). If future modes introduce invalid pixels, they must still follow the alpha semantics above so downstream metrics remain correct.

Any new debug-normal emitter must follow this exact encoding so that comparisons between ddx/ddy normals and Sobel ground truth remain numerically consistent.
