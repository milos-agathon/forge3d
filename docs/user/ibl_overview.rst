Image-Based Lighting (IBL) Overview
===================================

This page describes the IBL pipeline and the Milestone 5 tone-mapping validation
workflow. The GPU is responsible for producing linear HDR frames from the precomputed
M4 resources (irradiance/specular prefilter/BRDF LUT) plus a directional light
(M1). The CPU applies tone mapping and assembles the comparison grid.

Milestone 5 — Tone mapping validation
-------------------------------------

The goal is to validate tone-mapping correctness and monotonicity on top
of the GPU-generated linear HDR frames.

.. code-block:: rst

  * GPU supples the linear HDR frames (from M4 resources + M1 dir light).
  * CPU must apply tone mapping and build the 3x5 grid.

  What it proves Tone-map stage is correct, monotone, and does not introduce clipping beyond configured tolerance.

  Backend expectation • GPU supplies the linear HDR frames (from M4 resources + M1 dir light). • CPU must apply tone mapping and build the 3×5 grid.

  CLI

  python examples/m5_generate.py --hdr assets/snow_field_4k.hdr --outdir reports/m5

  Outputs • m5_tonemap_compare.png (columns: Linear, Reinhard, ACES × roughness) • m5_meta.json

  Acceptance checks • E1: Linear column equals the pre-tonemap baseline pixel-for-pixel. • E2: Clipped channel ratio for Reinhard/ACES ≤ 0.01%. • E3: Reinhard and ACES curves monotone over [0, 32].

  Meta must include

  { "milestone": "M5", "backend": "gpu", // frames came from GPU; tone-map on CPU is implied "tone_curves": ["linear", "reinhard", "aces"], "exposure": 1.0, "clip_fraction": {"reinhard": 0.00005, "aces": 0.00012}, "accept": {"E1": true, "E2": true, "E3": true} }

CLI usage
---------

Run the generator (uses a deterministic synthetic HDR if the asset is unavailable)::

  python examples/m5_generate.py --hdr assets/snow_field_4k.hdr --outdir reports/m5

Outputs
-------

- **m5_tonemap_compare.png**
- **m5_meta.json**

Acceptance checks
-----------------

- **E1**: Linear column equals the pre-tonemap baseline pixel-for-pixel.
- **E2**: Clipped channel ratio for Reinhard/ACES ≤ 0.01%.
- **E3**: Reinhard and ACES curves are monotone over [0, 32].

Metadata example
----------------

.. code-block:: json

  {
    "milestone": "M5",
    "backend": "gpu",
    "tone_curves": ["linear", "reinhard", "aces"],
    "exposure": 1.0,
    "clip_fraction": {"reinhard": 0.00005, "aces": 0.00012},
    "accept": {"E1": true, "E2": true, "E3": true}
  }

Notes
-----

- The production path is GPU-first for frame generation (WGSL compute and render
  passes). The tone-map stage and image composition happen on the CPU.
- For headless CI or CPU-only environments, the generator can still be run
  backend-agnostically. The acceptance thresholds and artifact layout remain
  the same.
