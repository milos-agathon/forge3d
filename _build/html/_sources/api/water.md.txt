# docs/api/water.md
# Water dielectric helpers: Fresnel and Beer–Lambert.
# This file exists to document A6 PBR utilities with brief usage and ranges.
# RELEVANT FILES:python/forge3d/water.py,README.md,tests/test_water_bsdf.py

## Overview

Water shading blends reflection and transmission.

We provide small helpers for Fresnel and Beer–Lambert to build plausible results.

## API

- `fresnel_schlick(cos_theta, ior)` returns reflectance in [0,1].

- `beer_lambert_transmittance(absorption, distance)` returns per‑channel transmittance.

- `water_shade(normal, view_dir, light_dir, base_color, ior=1.33, absorption=(0,0.05,0.1), roughness=0.02, thickness=1.0)` computes a simple water color.

## Notes

This is a lightweight offline preview helper.

For production path tracing, implement full microfacet specular and proper refraction paths.


