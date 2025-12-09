// docs/api/guiding.md
// User-facing notes for A13 guiding utilities (Python implementation).
// This exists to document the minimal API and current limitations.
// RELEVANT FILES:python/forge3d/guiding.py,README.md

# Path Guiding (A13)

This release introduces minimal scaffolding for spatial/directional guiding.

## Python API

```python
from forge3d import OnlineGuidingGrid

g = OnlineGuidingGrid(width, height, bins_per_cell=8)
g.update(x, y, bin, weight)  # accumulate samples
pdf = g.pdf(x, y)            # returns normalized distribution (bins_per_cell,)
```

The `OnlineGuidingGrid` keeps a per-cell histogram over `bins_per_cell` azimuthal directions.

## Limitations

- Pure Python implementation (no GPU acceleration).
- No SD-tree or on-GPU updates.
- Intended for experiments/tests.

