# Interactive Viewer

This section covers the interactive windowed viewer functionality for real-time scene exploration.

```{toctree}
:maxdepth: 2

interactive_viewer
```

## Overview

The forge3d interactive viewer provides real-time windowed rendering with camera controls for exploring 3D scenes interactively.

## Quick Start

```python
import forge3d as f3d

with f3d.open_viewer_async(terrain_path=f3d.mini_dem_path()) as viewer:
    viewer.set_orbit_camera(phi_deg=35, theta_deg=55, radius=1.8)
    viewer.snapshot("viewer.png")
```

## Topics

- **[Interactive Viewer](interactive_viewer.md)** — Windowed application with orbit/FPS camera controls
