# docs/api/firefly_clamp.md
# Firefly Clamp (A17)
# Provides a luminance-based clamp to suppress bright outliers with minimal bias.
# RELEVANT FILES:python/forge3d/path_tracing.py,tests/test_a17_firefly_clamp.py,python/forge3d/path_tracing.pyi

## Summary

Adds an optional luminance clamp to the Python path tracer stub to reduce fireflies by scaling pixel colors when their luminance exceeds a threshold.

## Usage

```
from forge3d import PathTracer

pt = PathTracer(256, 256, seed=123)
img = pt.render_rgba(256, 256, frames=3, luminance_clamp=0.6)
```

- `luminance_clamp` is in normalized units [0, 1]. 0 disables.
- Alias `firefly_clamp` is accepted.
- Scaling uses Rec. 709 luminance and preserves color ratios to minimize bias.

## Acceptance

The test `tests/test_a17_firefly_clamp.py` asserts ≥10× reduction in high-luminance outliers with <10% shift in mean luminance.
