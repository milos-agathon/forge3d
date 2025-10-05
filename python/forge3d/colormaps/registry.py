from __future__ import annotations
from typing import Dict, Callable
from .core import Colormap
import numpy as np

_REGISTRY: Dict[str, Callable[[], Colormap]] = {}

def register(name: str, factory: Callable[[], Colormap]) -> None:
    key = name.lower()
    if key in _REGISTRY:
        raise ValueError(f"Colormap already registered: {name}")
    _REGISTRY[key] = factory

def get(name: str) -> Colormap:
    key = name.lower()
    if key in _REGISTRY:
        return _REGISTRY[key]()
    # try provider:name form
    if ":" in key:
        provider, _, cname = key.partition(":")
        from .providers import load_provider
        return load_provider(provider)(cname)
    raise KeyError(f"Unknown colormap: {name} (available={list(_REGISTRY.keys())})")

def available() -> list[str]:
    return sorted(_REGISTRY.keys())

def to_linear_rgba_u8(cm: Colormap) -> bytes:
    # Convert float32 0..1 to u8 for texture upload (premultiplied alpha not required)
    arr = (np.clip(cm.rgba, 0.0, 1.0) * 255.0 + 0.5).astype("uint8")
    return arr.tobytes()
