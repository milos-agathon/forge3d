# python/forge3d/path_tracing.pyi
# Type stubs for path tracing API skeleton (Workstream A).
# This exists to keep the package typed and IDE-friendly.
# RELEVANT FILES:python/forge3d/path_tracing.py,python/forge3d/__init__.pyi,tests/test_path_tracing_api.py

from __future__ import annotations
from typing import Tuple, Iterable, Dict, Mapping, Callable, Any, Optional
import numpy as np

class PathTracer:
    def __init__(self, width: int, height: int, *, max_bounces: int = ..., seed: int = ..., tile: int | None = ...) -> None: ...
    @property
    def size(self) -> Tuple[int, int]: ...
    def render_rgba(self, *, spp: int = ...) -> np.ndarray: ...
    def add_sphere(self, center: tuple[float, float, float], radius: float, material_or_color) -> None: ...
    def add_triangle(self, v0: tuple[float, float, float], v1: tuple[float, float, float], v2: tuple[float, float, float], material_or_color) -> None: ...
    def render_progressive(
        self,
        *,
        callback: Optional[Callable[[Dict[str, Any]], Optional[bool]]] = ..., 
        tile_size: int | None = ..., 
        min_updates_per_sec: float = ..., 
        time_source: Callable[[], float] = ..., 
        spp: int = ..., 
    ) -> np.ndarray: ...

def create_path_tracer(width: int, height: int, *, max_bounces: int = ..., seed: int = ...) -> PathTracer: ...
def _fresnel_schlick(cos_theta: np.ndarray, F0: np.ndarray) -> np.ndarray: ...

# AOV API (A14)
def render_aovs(
    width: int,
    height: int,
    scene,
    camera: dict | None = ..., 
    *,
    aovs: Iterable[str] = ..., 
    seed: int = ..., 
    frames: int = ..., 
    use_gpu: bool = ..., 
) -> Dict[str, np.ndarray]: ...

def save_aovs(
    aovs_map: Mapping[str, np.ndarray], 
    basename: str, 
    *, 
    output_dir: str | None = ..., 
) -> Dict[str, str]: ...

def iter_tiles(width: int, height: int, tile: int) -> Iterable[Tuple[int, int, int, int]]: ...
