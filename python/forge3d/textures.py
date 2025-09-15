# python/forge3d/textures.py
# Minimal texture utilities for building PBR texture sets from numpy arrays.
# Exists to satisfy tests for channel mapping and material texturing without GPU.
# RELEVANT FILES:python/forge3d/materials.py,python/forge3d/path_tracing.py,tests/test_gltf_mr_channels.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, "np.typing.NDArray[np.uint8]"]


@dataclass
class Tex:
    path: Optional[Path]
    data: Optional[np.ndarray]
    srgb: bool


@dataclass
class PbrTexSet:
    base_color: Optional[Tex] = None
    metallic_roughness: Optional[Tex] = None
    normal: Optional[Tex] = None
    emissive: Optional[Tex] = None


def _ensure_rgba8(arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError("texture must be a numpy array")

    if arr.dtype != np.uint8:
        raise TypeError("texture dtype must be uint8")

    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("texture must be (H,W,3|4)")

    if arr.flags.c_contiguous is False:
        arr = np.ascontiguousarray(arr)

    if arr.shape[2] == 3:
        h, w, _ = arr.shape
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = arr
        rgba[..., 3] = 255
        return rgba
    return arr


def load_texture(path_or_array: Union[str, Path, ArrayLike], srgb: bool) -> Tex:
    """Load a texture from a numpy array or a file path.

    For MVP tests we only support numpy arrays to avoid extra deps.

    """
    if isinstance(path_or_array, (str, Path)):
        # Keep simple: do not load from disk to avoid adding dependencies now.
        # Represent as path-only Tex; downstream code should handle None data.
        return Tex(path=Path(path_or_array), data=None, srgb=srgb)

    arr = _ensure_rgba8(np.asarray(path_or_array))
    return Tex(path=None, data=arr, srgb=srgb)


def build_pbr_textures(
    base_color: Optional[ArrayLike] = None,
    metallic_roughness: Optional[ArrayLike] = None,
    normal: Optional[ArrayLike] = None,
    emissive: Optional[ArrayLike] = None,
    *,
    srgb_defaults: bool = True,
) -> PbrTexSet:
    """Construct a PBR texture set with sensible sRGB defaults.

    - base_color and emissive are sRGB by default.
    - metallic_roughness and normal are linear by default.

    """
    bc = load_texture(base_color, srgb=True) if base_color is not None else None
    mr = (
        load_texture(metallic_roughness, srgb=False)
        if metallic_roughness is not None
        else None
    )
    nm = load_texture(normal, srgb=False) if normal is not None else None
    em = load_texture(emissive, srgb=True) if emissive is not None else None
    return PbrTexSet(base_color=bc, metallic_roughness=mr, normal=nm, emissive=em)


def gltf_mr_channels(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (roughness, metallic) channels from a glTF MR texture.

    Convention: G = roughness, B = metallic.

    """
    rgba = _ensure_rgba8(arr)
    rough = rgba[..., 1]
    metal = rgba[..., 2]
    return rough, metal

