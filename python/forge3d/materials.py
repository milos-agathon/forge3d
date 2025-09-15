# python/forge3d/materials.py
# Minimal PBR material container with texture set attachment.
# Exists to provide a simple API surface for tests without GPU coupling.
# RELEVANT FILES:python/forge3d/textures.py,python/forge3d/path_tracing.py,tests/test_pbr_textures_gpu.py

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

from .textures import PbrTexSet


Color3 = Tuple[float, float, float]
Color4 = Tuple[float, float, float, float]


@dataclass
class PbrMaterial:
    base_color_factor: Color4 = (1.0, 1.0, 1.0, 1.0)
    metallic_factor: float = 1.0
    roughness_factor: float = 1.0
    emissive_factor: Color3 = (0.0, 0.0, 0.0)
    textures: Optional[PbrTexSet] = None

    def with_textures(self, texset: PbrTexSet) -> "PbrMaterial":
        return replace(self, textures=texset)

