// docs/api/denoise.md
// A-trous edge-aware denoiser API and usage.
// Documents inputs/outputs, parameters, and limitations for Workstream A5.
// RELEVANT FILES:python/forge3d/denoise.py,tests/test_a5_denoise.py,README.md

# Denoiser (A‑trous)

The Python API provides a simple, edge‑aware A‑trous wavelet denoiser.

It uses albedo, normal, and depth as guides to preserve edges while reducing noise.

## API

```python
from forge3d.denoise import atrous_denoise

denoised = atrous_denoise(
    color,                    # float32 (H, W, 3)
    albedo=albedo,            # float32 (H, W, 3)
    normal=normal,            # float32 (H, W, 3), unit-length
    depth=depth,              # float32 (H, W)
    iterations=3,             # a-trous passes
    sigma_color=0.10,
    sigma_albedo=0.20,
    sigma_normal=0.30,
    sigma_depth=0.50,
)
```

## Notes

- Inputs should be linear‑space float32.

- Normals should be normalized; depth in linear units.

- The filter is deterministic and dependency‑free (NumPy only).

- For large images, prefer smaller iteration counts to stay within budget.

