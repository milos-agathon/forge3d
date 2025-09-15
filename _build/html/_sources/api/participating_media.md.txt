<!-- docs/api/participating_media.md
     Minimal API docs for participating media helpers
     Adds CPU utilities for A11 acceptance (media sampling; sun/env scatter)
     RELEVANT FILES:python/forge3d/lighting.py,src/shaders/lighting_media.wgsl,tests/test_media_*.py -->

# Participating Media Helpers (A11)

The `forge3d.lighting` module provides minimal CPU utilities to aid testing and prototyping of participating media:

- `hg_phase(cos_theta, g)` — Henyey–Greenstein phase function normalized over the sphere.
- `sample_hg(u1, u2, g)` — Samples directions from the HG phase in local space; returns `(dir, pdf)`.
- `height_fog_factor(depth, density=0.02)` — Homogeneous medium fog factor along a view ray.
- `single_scatter_estimate(depth, sun_intensity=1.0, density=0.02, g=0.0)` — Tiny single-scatter proxy useful for tests.

These are CPU-only reference implementations designed for deterministic tests and do not replace a full volumetric integrator.

