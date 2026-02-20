<!-- docs/api/reflections.md -->
<!-- API reference for Scene reflection controls. -->
<!-- RELEVANT FILES: python/forge3d/__init__.pyi, src/lib.rs, src/scene/mod.rs -->

# Reflections API

## Overview

Scene-level controls for planar and screen-space reflections.
All methods are instance methods on `Scene`.

## API

### Enable / Disable

- `scene.enable_reflections(quality=None)` -- Enable reflections with optional quality preset (`"low"`, `"medium"`, `"high"`, `"ultra"`).
- `scene.disable_reflections()` -- Disable reflections.

### Configuration

- `scene.set_reflection_plane(normal, point, size)` -- Define the reflection plane geometry.
  - `normal`: `(float, float, float)` -- Plane normal direction.
  - `point`: `(float, float, float)` -- A point on the plane.
  - `size`: `(float, float, float)` -- Plane extents.

- `scene.set_reflection_intensity(intensity)` -- Set reflection strength (`0.0`--`1.0`).

- `scene.set_reflection_fresnel_power(power)` -- Control Fresnel falloff exponent.

- `scene.set_reflection_distance_fade(start, end)` -- Fade reflections between `start` and `end` distances.

- `scene.set_reflection_debug_mode(mode)` -- Toggle debug visualisation (integer mode selector).

## Notes

Reflections are off by default. Call `enable_reflections()` before rendering to activate.
