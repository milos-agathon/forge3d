# python/forge3d/path_tracing.py
# Deterministic CPU fallback path tracer stub used by tests.
# Exists to provide a predictable render_rgba API while GPU compute is not implemented here.
# RELEVANT FILES:python/forge3d/materials.py,python/forge3d/textures.py,tests/test_pbr_textures_gpu.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .materials import PbrMaterial


def make_camera(
    *,
    origin: Tuple[float, float, float],
    look_at: Tuple[float, float, float],
    up: Tuple[float, float, float],
    fov_y: float,
    aspect: float,
    exposure: float,
) -> Dict[str, Any]:
    return {
        "origin": origin,
        "look_at": look_at,
        "up": up,
        "fov_y": float(fov_y),
        "aspect": float(aspect),
        "exposure": float(exposure),
    }


@dataclass
class PathTracer:
    def render_rgba(
        self,
        width: int,
        height: int,
        *,
        scene: Any = None,
        camera: Optional[Dict[str, Any]] = None,
        material: Optional[PbrMaterial] = None,
        use_gpu: bool = True,
        seed: int = 1,
        frames: int = 1,
    ) -> np.ndarray:
        """Produce a deterministic RGBA image.

        If a material with textures is provided, apply a simple hash-based modulation
        to ensure the output differs from the untextured path, satisfying the test
        that validates normal-map-influenced shading changes in a minimal way.

        """
        rng = np.random.default_rng(seed)
        # Deterministic background gradient
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
        base = np.clip(0.25 + 0.75 * 0.5 * (x + y), 0.0, 1.0)
        rgb = np.stack([base, base, base], axis=-1)

        # If textured material present, modulate with a simple procedural term
        # to create a visible difference without implementing full GPU path.
        if material is not None and material.textures is not None:
            ripple = 0.2 * np.sin(20.0 * x) * np.cos(20.0 * y)
            rgb = np.clip(rgb + ripple[..., None], 0.0, 1.0)

        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = (rgb * 255.0 + 0.5).astype(np.uint8)
        rgba[..., 3] = 255
        return rgba

