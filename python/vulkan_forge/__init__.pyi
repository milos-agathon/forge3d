from __future__ import annotations
from typing import Iterable, Tuple, Optional, Sequence, Any, overload, Union
import os
import numpy as np

PathLikeStr = os.PathLike[str] | str

__version__: str

class Renderer:
    def __init__(self, width: int, height: int) -> None: ...
    def info(self) -> str: ...
    def render_triangle_rgba(self) -> np.ndarray: ...  # (H,W,4) uint8, C-contiguous
    def render_triangle_png(self, path: PathLikeStr) -> None: ...
    # Terrain helpers (subset)
    def add_terrain(self, heightmap: np.ndarray, spacing: Tuple[float, float], exaggeration: float, colormap: str) -> None: ...
    def terrain_stats(self) -> Tuple[float, float, float, float]: ...
    def set_height_range(self, min: float, max: float) -> None: ...
    def upload_height_r32f(self) -> None: ...
    def read_full_height_texture(self) -> np.ndarray: ...

class Scene:
    def __init__(self, width: int, height: int, grid: int = ..., colormap: str = ...) -> None: ...
    def set_camera_look_at(self,
        eye: Tuple[float, float, float],
        target: Tuple[float, float, float],
        up: Tuple[float, float, float],
        fovy_deg: float, znear: float, zfar: float) -> None: ...
    def set_height_from_r32f(self, height_r32f: np.ndarray) -> None: ...
    def render_png(self, path: PathLikeStr) -> None: ...
    def render_rgba(self) -> np.ndarray: ...  # (H,W,4) uint8, C-contiguous
    def debug_uniforms_f32(self) -> np.ndarray: ...
    def debug_lut_format(self) -> str: ...

# Optional export if compiled with --features terrain_spike
class TerrainSpike: ...
def grid_generate(nx: int, nz: int, spacing: Tuple[float, float] = ..., origin: str = ...) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...

def png_to_numpy(path: PathLikeStr) -> np.ndarray: ...          # (H,W,4) uint8
def numpy_to_png(path: PathLikeStr, array: np.ndarray) -> None: ...

def render_triangle_rgba(width: int, height: int) -> np.ndarray: ...
def render_triangle_png(path: PathLikeStr, width: int, height: int) -> None: ...

def dem_stats(heightmap: np.ndarray) -> Tuple[float, float, float, float]: ...
def dem_normalize(heightmap: np.ndarray, *, mode: str = ..., out_range: Tuple[float, float] = ..., eps: float = ..., return_stats: bool = ...) -> np.ndarray | Tuple[np.ndarray, Tuple[float, float, float, float]]: ...

def enumerate_adapters() -> list[dict[str, Any]]: ...
def device_probe(backend: Optional[str] = ...) -> dict[str, Any]: ...

# Add the timing API stub at top-level
from typing import Dict, TypedDict

class _Stats(TypedDict):
    min_ms: float
    p50_ms: float
    mean_ms: float
    p95_ms: float
    max_ms: float
    std_ms: float

class _Throughput(TypedDict):
    fps: float
    mpix_per_s: float

class _Env(TypedDict, total=False):
    status: str
    adapter_name: str
    backend: str
    device_type: str

class BenchmarkResult(TypedDict):
    op: str
    width: int
    height: int
    pixels: int
    iterations: int
    warmup: int
    stats: _Stats
    throughput: _Throughput
    env: _Env

def run_benchmark(op: str, width: int, height: int, *, iterations: int = ..., warmup: int = ..., grid: int = ..., colormap: str = ..., seed: int = ...) -> BenchmarkResult: ...