# Offscreen Rendering

This section covers headless offscreen rendering and Jupyter notebook integration.

## Overview

forge3d supports headless offscreen rendering for batch processing, testing, and Jupyter notebook visualization without requiring a display server or windowed environment.

## Quick Start

```python
import forge3d as f3d

# Render offscreen to RGBA array
rgba = f3d.render_offscreen_rgba(width=800, height=600, seed=42, frames=1)

# Save deterministic PNG
f3d.save_png_deterministic("output.png", rgba)

# Display in Jupyter notebook
from forge3d.helpers.ipython_display import display_rgba
display_rgba(rgba, title="My Render")
```

## Features

- **Deterministic PNG Export** — Stable byte-for-byte output for regression testing
- **Jupyter Integration** — IPython.display helpers for inline notebook visualization
- **Headless Operation** — No display server or GPU window required
- **NumPy Integration** — Zero-copy RGBA arrays compatible with scientific Python stack

## API Reference

See:
- [python/forge3d/helpers/offscreen.py](../../python/forge3d/helpers/offscreen.py)
- [python/forge3d/helpers/ipython_display.py](../../python/forge3d/helpers/ipython_display.py)
- [examples/viewer_offscreen_demo.py](../../examples/viewer_offscreen_demo.py)
