# python/forge3d/path_tracing.py
# Deterministic CPU fallback path tracer stub used by tests.
# Exists to provide a predictable render_rgba API while GPU compute is not implemented here.
# RELEVANT FILES:python/forge3d/materials.py,python/forge3d/textures.py,tests/test_pbr_textures_gpu.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Iterable, Mapping

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
    _width: int = 0
    _height: int = 0
    _max_bounces: int = 1
    _seed: int = 1

    def __init__(self, width: int, height: int, *, max_bounces: int = 1, seed: int = 1) -> None:
        self._width = int(width)
        self._height = int(height)
        self._max_bounces = int(max_bounces)
        self._seed = int(seed)

        if self._width <= 0 or self._height <= 0:
            raise ValueError("invalid tracer size")

    @property
    def size(self) -> Tuple[int, int]:
        return (self._width, self._height)

    def add_sphere(self, center: tuple[float, float, float], radius: float, material_or_color) -> None:
        # Placeholder for API parity.
        return None

    def add_triangle(
        self,
        v0: tuple[float, float, float],
        v1: tuple[float, float, float],
        v2: tuple[float, float, float],
        material_or_color,
    ) -> None:
        # Placeholder for API parity.
        return None

    def render_rgba(self, *, spp: int = 1) -> np.ndarray:
        """Produce a deterministic RGBA image.

        Uses a simple gradient with optional procedural modulation.

        """
        width, height = self._width, self._height
        rng = np.random.default_rng(self._seed + int(spp))
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
        base = np.clip(0.25 + 0.75 * 0.5 * (x + y), 0.0, 1.0)
        rgb = np.stack([base, base, base], axis=-1)

        # Subtle variation to ensure deterministic but non-uniform output.
        ripple = 0.05 * np.sin(12.0 * x + 0.7) * np.cos(9.0 * y + 0.3)
        rgb = np.clip(rgb + ripple[..., None], 0.0, 1.0)

        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = (rgb * 255.0 + 0.5).astype(np.uint8)
        rgba[..., 3] = 255
        return rgba


def create_path_tracer(width: int, height: int, *, max_bounces: int = 1, seed: int = 1) -> PathTracer:
    return PathTracer(width, height, max_bounces=max_bounces, seed=seed)


def _synthetic_basis(width: int, height: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create simple deterministic basis fields for synthetic AOVs.

    Uses a gradient and a seeded random field to keep determinism.

    """
    rng = np.random.default_rng(int(seed))
    y = np.linspace(0.0, 1.0, int(height), dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, int(width), dtype=np.float32)[None, :]
    # Smooth gradient base
    base = np.clip(0.25 + 0.75 * 0.5 * (x + y), 0.0, 1.0).astype(np.float32)
    # Low-amplitude structured noise
    noise = rng.normal(0.0, 0.05, size=(int(height), int(width))).astype(np.float32)
    return base, noise


def render_aovs(
    width: int,
    height: int,
    scene: Any,
    camera: Optional[Dict[str, Any]] = None,
    *,
    aovs: Iterable[str] = ("albedo", "normal", "depth", "direct", "indirect", "emission", "visibility"),
    seed: int = 1,
    frames: int = 1,
    use_gpu: bool = True,
) -> Dict[str, np.ndarray]:
    """Render a deterministic set of AOVs for testing and API conformance.

    This CPU implementation returns arrays with the correct shapes and dtypes.
    Values are procedurally generated and deterministic with the given seed.

    """
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    # Normalize requested AOV names
    req = [str(k).lower() for k in aovs]
    base, noise = _synthetic_basis(width, height, seed=seed)

    out: Dict[str, np.ndarray] = {}

    if "albedo" in req:
        rgb = np.stack([
            base,
            np.clip(base * 0.8 + 0.1 + 0.3 * noise, 0.0, 1.0),
            np.clip(base * 0.6 + 0.2 - 0.2 * noise, 0.0, 1.0),
        ], axis=-1).astype(np.float32)
        out["albedo"] = rgb

    if "normal" in req:
        nx = (2.0 * base - 1.0).astype(np.float32)
        ny = (2.0 * (1.0 - base) - 1.0).astype(np.float32)
        nz = np.sqrt(np.clip(1.0 - np.clip(nx * nx + ny * ny, 0.0, 1.0), 0.0, 1.0)).astype(np.float32)
        out["normal"] = np.stack([nx, ny, nz], axis=-1)

    if "depth" in req:
        depth = (1.0 - base).astype(np.float32)
        out["depth"] = depth

    if "direct" in req:
        direct = np.clip(base + 0.2 * noise, 0.0, 10.0).astype(np.float32)
        out["direct"] = np.stack([direct, direct * 0.8, direct * 0.6], axis=-1)

    if "indirect" in req:
        indirect = np.clip(0.5 * base + 0.1 * noise, 0.0, 10.0).astype(np.float32)
        out["indirect"] = np.stack([indirect * 0.7, indirect, indirect * 0.9], axis=-1)

    if "emission" in req:
        emission = (0.1 + 0.2 * (np.sin(8.0 * base) + 1.0) * 0.5).astype(np.float32)
        out["emission"] = np.stack([emission, emission * 0.5, emission * 0.25], axis=-1)

    if "visibility" in req:
        vis = (base > 0.35).astype(np.uint8)
        out["visibility"] = vis

    return out


def save_aovs(
    aovs_map: Mapping[str, np.ndarray],
    basename: str,
    *,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Save AOVs to disk.

    HDR AOVs (albedo, normal, depth, direct, indirect, emission) are intended for EXR.
    Visibility is written as PNG. If EXR support is not available, this function
    skips HDR files and returns paths only for saved images.

    """
    from pathlib import Path

    out_paths: Dict[str, str] = {}
    out_dir = Path(output_dir) if output_dir is not None else Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to import numpy_to_png for uint8 saving.
    try:
        from . import numpy_to_png  # type: ignore
    except Exception:
        numpy_to_png = None  # type: ignore

    for key, arr in aovs_map.items():
        k = str(key).lower()
        if k == "visibility":
            # Expect (H,W) uint8
            filename = f"{basename}_aov-{k}.png"
            path = out_dir / filename
            if numpy_to_png is None:
                # Best-effort fallback using PIL if available
                try:
                    from PIL import Image  # type: ignore

                    im = Image.fromarray(arr.astype(np.uint8), mode="L")
                    im.save(str(path))
                except Exception:
                    # Skip if no writer available
                    continue
            else:
                numpy_to_png(str(path), arr)
            out_paths[k] = str(path)
        else:
            # HDR paths: prefer EXR, but skip if toolchain not present.
            # We still record the intended path for clarity.
            filename = f"{basename}_aov-{k}.exr"
            path = out_dir / filename
            # Optional: attempt OpenEXR via imageio if present.
            try:
                import imageio.v3 as iio  # type: ignore

                data = arr.astype(np.float32)
                # Some writers expect 3-channel for EXR, convert depth to 1-channel compatible
                if data.ndim == 2:
                    data = data[..., None]
                iio.imwrite(str(path), data, plugin="EXR")
                out_paths[k] = str(path)
            except Exception:
                # Skip silently if EXR pipeline is unavailable.
                continue

    return out_paths
