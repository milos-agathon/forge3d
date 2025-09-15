// docs/api/guiding.md
// User-facing notes for A13 guiding utilities (Python/Rust scaffolding).
// This exists to document the minimal API and current limitations.
// RELEVANT FILES:python/forge3d/guiding.py,src/path_tracing/guiding.rs,src/shaders/pt_guiding.wgsl,README.md

# Path Guiding (A13)

This release introduces minimal scaffolding for spatial/directional guiding.

Python provides `forge3d.OnlineGuidingGrid(width, height, bins_per_cell=8)`, which keeps a per-cell histogram over `bins_per_cell` azimuthal directions and supports `update(x,y,bin,weight)` and `pdf(x,y)` returning a normalized distribution.

Rust includes `src/path_tracing/guiding.rs` with a matching grid and simple stochastic updates. WGSL buffer layouts are stubbed in `src/shaders/pt_guiding.wgsl` for future kernel integration.

Limitations:

- No SD-tree or on-GPU updates yet.
- Not yet wired into the path tracing kernels; intended for experiments/tests.

