<!-- docs/api/cloud_shadows.md -->
<!-- API reference for Scene cloud shadow controls. -->
<!-- RELEVANT FILES: python/forge3d/__init__.pyi, src/lib.rs, src/scene/mod.rs -->

# Cloud Shadows API

## Overview

Scene-level controls for procedural cloud shadow overlays.
All methods are instance methods on `Scene`.

## API

### Enable / Disable

- `scene.enable_cloud_shadows(quality=None)` -- Enable cloud shadows with optional quality preset (`"low"`, `"medium"`, `"high"`, `"ultra"`).
- `scene.disable_cloud_shadows()` -- Disable cloud shadows.

### Configuration

- `scene.set_cloud_shadow_intensity(intensity)` -- Set shadow darkening strength (`0.0`--`1.0`).

- `scene.set_cloud_shadow_softness(softness)` -- Control edge softness of cloud shadow boundaries.

## Notes

Cloud shadows are off by default. Call `enable_cloud_shadows()` before rendering to activate.
