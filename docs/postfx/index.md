---
title: Post-Processing (PostFX)
---

# Post-Processing Effect Chain

Forge3D provides a configurable post-processing chain that can be controlled from Python. Effects can be enabled, disabled, listed, and parameterized at runtime.

## Features
- Chain management with priorities and temporal support
- Effects: tonemap, bloom, blur, sharpen, FXAA, temporal AA (catalog may expand)
- Parameter validation with ranges and sensible defaults

## Python API

```python
import forge3d.postfx as postfx

# Enable postfx chain (no-op if already enabled)
postfx.set_chain_enabled(True)

# Enable effects with parameters
postfx.enable("bloom", threshold=1.0, strength=0.6)
postfx.enable("tonemap", exposure=1.1, gamma=2.2)

# Adjust a parameter
postfx.set_parameter("bloom", "strength", 0.8)

# Inspect
print(postfx.list())
print(postfx.list_available())
print(postfx.get_effect_info("bloom"))

# Disable
postfx.disable("bloom")
```

## Notes
- Effects and parameters are validated; out-of-range values are clamped with a warning.
- The effect list is ordered by priority (higher runs later).
- Integration with GPU timing exists via `postfx.get_timing_stats()` if GPU timing is enabled.

## Demo
See `examples/postfx_chain_demo.py` for a runnable demo that enables a small chain and prints stats.

