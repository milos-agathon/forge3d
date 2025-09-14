.. docs/user/path_tracing.rst
.. High-level user documentation for Workstream A (Path Tracing) skeleton API.
.. This exists to document the initial surface and roadmap while kernels land.
.. RELEVANT FILES:python/forge3d/path_tracing.py,python/forge3d/__init__.py,tests/test_path_tracing_api.py,docs/index.rst

Path Tracing (Workstream A)
===========================

Status: A1 complete (CPU fallback with spheres + triangles, HDR accumulation, tile scheduler); WGSL kernel scaffold present; GPU wiring pending.

Overview
--------

The path tracing module introduces an offline, high‑quality compute rendering pipeline.

This initial version provides a minimal, importable API plus a CPU fallback renderer implementing A1 features: deterministic RNG, spheres and triangles intersectors with Lambertian shading, HDR accumulation, and a tiled traversal for cache locality.

API
---

.. code-block:: python

    import forge3d.path_tracing as pt

    tracer = pt.PathTracer(256, 256, max_bounces=4, seed=1234)
    img = tracer.render_rgba(spp=1)
    # img: np.ndarray (H, W, 4), dtype=uint8

Or use the factory:

.. code-block:: python

    tracer = pt.create_path_tracer(512, 512)

Roadmap
-------

Planned milestones align with A1–A25 in ``roadmap2.csv``:

- A1: CPU fallback implemented (triangles + spheres, HDR buffers, tiled); WGSL kernel scaffold added.
- A3/A7: CPU/GPU BVH build and traversal; watertight intersections.
- A2/A4/A6/A8/A10/A11: BSDFs, sampling (NEE/MIS), media.
- A5/A14/A16/A17: Denoising, AOVs/EXR IO, variance control, clamping.

Until these land, the skeleton allows docs/tests/CI to evolve without breaking users.
