from __future__ import annotations

"""High-level helpers for the TerrainRenderer PBR+POM pipeline.

This module is the Python-side companion to the native
``TerrainRenderer.render_terrain_pbr_pom(...)`` API.  It exposes the same
helpers that previously lived in :mod:`forge3d.terrain_demo`, but under a
name that matches the underlying Rust/WGPU terrain pipeline.

``forge3d.terrain_demo`` remains as a thin compatibility shim; new code
should prefer importing :mod:`forge3d.terrain_pbr_pom` instead.
"""

from .terrain_demo import (  # type: ignore[F401]
    DEFAULT_DEM,
    DEFAULT_HDR,
    DEFAULT_OUTPUT,
    DEFAULT_SIZE,
    DEFAULT_DOMAIN,
    DEFAULT_CAM_RADIUS,
    DEFAULT_CAM_PHI,
    DEFAULT_CAM_THETA,
    DEFAULT_CAM_FOV,
    DEFAULT_CAMERA_MODE,
    render_sunrise_to_noon_sequence,
    run,
    _build_renderer_config,
)

# Canonical aliases using the native function name
render_terrain_pbr_pom_sequence = render_sunrise_to_noon_sequence
render_terrain_pbr_pom = run

__all__ = [
    "DEFAULT_DEM",
    "DEFAULT_HDR",
    "DEFAULT_OUTPUT",
    "DEFAULT_SIZE",
    "DEFAULT_DOMAIN",
    "DEFAULT_CAM_RADIUS",
    "DEFAULT_CAM_PHI",
    "DEFAULT_CAM_THETA",
    "DEFAULT_CAM_FOV",
    "DEFAULT_CAMERA_MODE",
    "render_sunrise_to_noon_sequence",
    "run",
    "_build_renderer_config",
    "render_terrain_pbr_pom_sequence",
    "render_terrain_pbr_pom",
]
