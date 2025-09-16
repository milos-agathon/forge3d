# python/forge3d/path_tracing.py
# Deterministic CPU fallback path tracer with basic features for tests and demos.
# Exists to provide a predictable render_rgba API and host-side utilities while GPU compute matures.
# RELEVANT FILES:python/forge3d/path_tracing.pyi,tests/test_a17_firefly_clamp.py,README.md

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Iterable, Mapping, Callable

import numpy as np
import time as _time

from .materials import PbrMaterial
from .denoise import atrous_denoise

# --- A19: Scene cache for HQ path tracing (Python fallback) ---
# Minimal in-memory cache to reuse scene-dependent precomputations across renders.
# On cache hit for unchanged (scene,camera,dimensions,seed,frames,denoiser) the re-render path
# avoids regenerating expensive fields (e.g., noise accumulation), improving latency while
# producing identical images.
class _SceneCache:
    def __init__(self, capacity: int = 8) -> None:
        from collections import OrderedDict
        self._cap = max(1, int(capacity))
        self._store: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _touch(self, key: str) -> None:
        # Move key to end (most-recent) if present
        if key in self._store:
            val = self._store.pop(key)
            self._store[key] = val

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self._store:
            self.hits += 1
            self._touch(key)
            return self._store[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        from collections import OrderedDict
        if key in self._store:
            self._store.pop(key)
        elif len(self._store) >= self._cap:
            # Evict least-recently-used
            try:
                self._store.popitem(last=False)
            except Exception:
                self._store = OrderedDict()
        self._store[key] = value


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


def make_sphere(*, center: Tuple[float, float, float], radius: float, albedo: Tuple[float, float, float]) -> Dict[str, Any]:
    """Minimal sphere descriptor for tests.

    This is a placeholder to keep API compatibility with tests that build small scenes.

    """
    return {
        "type": "sphere",
        "center": tuple(map(float, center)),
        "radius": float(radius),
        "albedo": tuple(map(float, albedo)),
    }


@dataclass
class PathTracer:
    _width: int = 0
    _height: int = 0
    _max_bounces: int = 1
    _seed: int = 1
    _tile: Optional[int] = None

    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        *,
        max_bounces: int = 1,
        seed: int = 1,
        tile: Optional[int] = None,
    ) -> None:
        self._width = int(width) if width is not None else 0
        self._height = int(height) if height is not None else 0
        self._max_bounces = int(max_bounces)
        self._seed = int(seed)
        self._tile = int(tile) if tile is not None else None
        # A19: scene cache controls and storage
        self._cache_enabled: bool = False
        self._scene_cache = _SceneCache(capacity=8)

        if (width is not None and self._width <= 0) or (height is not None and self._height <= 0):
            raise ValueError("invalid tracer size")

    @property
    def size(self) -> Tuple[int, int]:
        return (self._width, self._height)

    # A19: Public cache controls
    def enable_scene_cache(self, enabled: bool = True, *, capacity: Optional[int] = None) -> None:
        self._cache_enabled = bool(enabled)
        if capacity is not None:
            self._scene_cache = _SceneCache(capacity=int(capacity))

    def reset_scene_cache(self) -> None:
        self._scene_cache.clear()

    def cache_stats(self) -> Dict[str, int]:
        return {"hits": int(self._scene_cache.hits), "misses": int(self._scene_cache.misses)}

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

    def render_rgba(self, *args, spp: int = 1, **kwargs) -> np.ndarray:
        """Produce an RGBA image.

        Overloads:
          - render using internal size: render_rgba(spp=1)
          - path-tracing style: render_rgba(width,height,scene,camera,seed=...,frames=...,use_gpu=...,denoiser=...,svgf_iters=...)

        """
        # New-style call with explicit (w,h, ...)
        if len(args) >= 2 and isinstance(args[0], (int, np.integer)) and isinstance(args[1], (int, np.integer)):
            width = int(args[0]); height = int(args[1])
            scene = kwargs.get("scene", None)
            camera = kwargs.get("camera", None)
            seed = int(kwargs.get("seed", self._seed))
            frames = int(kwargs.get("frames", 1))
            denoiser = str(kwargs.get("denoiser", "off")).lower()
            svgf_iters = int(kwargs.get("svgf_iters", 5))
            # A17: optional luminance/throughput clamp to suppress fireflies with minimal bias.
            # Prefer "luminance_clamp"; accept legacy alias "firefly_clamp".
            lum_clamp = kwargs.get("luminance_clamp", kwargs.get("firefly_clamp", None))
            try:
                lum_clamp_f = float(lum_clamp) if lum_clamp is not None else None
            except Exception:
                lum_clamp_f = None

            # Build a stable cache key when enabled
            rgb: np.ndarray
            if self._cache_enabled:
                key = self._make_cache_key(width, height, scene, camera, seed, frames, denoiser, svgf_iters, lum_clamp_f)
                entry = self._scene_cache.get(key)
            else:
                key = ""
                entry = None

            # Synthesize a simple noisy HDR-like image (float32 0..1) deterministically
            y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
            x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
            base = np.clip(0.25 + 0.75 * 0.5 * (x + y), 0.0, 1.0)
            if entry is not None and "noise_accum" in entry and entry.get("dims") == (height, width) and entry.get("frames") == max(1, frames):
                noise_accum = entry["noise_accum"]  # precomputed sum over frames
            else:
                rgb_accum = np.zeros((height, width, 3), dtype=np.float32)
                for f in range(max(1, frames)):
                    rng = np.random.default_rng(seed + f)
                    noise = rng.normal(0.0, 0.08, size=(height, width, 3)).astype(np.float32)
                    rgb_accum += np.clip(np.stack([base, base, base], axis=-1) + noise, 0.0, 1.0)
                noise_accum = rgb_accum
                if self._cache_enabled:
                    self._scene_cache.put(key, {"noise_accum": noise_accum, "dims": (height, width), "frames": max(1, frames)})

            rgb = noise_accum / float(max(1, frames))

            # Apply luminance clamp if requested (scale color to limit luminance, minimizing bias)
            if lum_clamp_f is not None and lum_clamp_f > 0.0:
                # Compute luminance using Rec. 709 weights
                lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
                # Avoid div-by-zero; where lum <= clamp, scale=1; else scale=clamp/lum
                with np.errstate(divide='ignore', invalid='ignore'):
                    scale = np.where(lum > lum_clamp_f, (lum_clamp_f / np.maximum(lum, 1e-8)), 1.0).astype(np.float32)
                rgb = rgb * scale[..., None]

            if denoiser == "svgf":
                # Build guidance AOVs deterministically
                aovs = render_aovs(width, height, scene=None, camera=None, aovs=("albedo","normal","depth"), seed=seed)
                rgb = atrous_denoise(
                    rgb.astype(np.float32),
                    albedo=aovs.get("albedo"),
                    normal=aovs.get("normal"),
                    depth=aovs.get("depth"),
                    iterations=svgf_iters,
                    sigma_color=0.30,
                    sigma_albedo=0.30,
                    sigma_normal=0.60,
                    sigma_depth=0.80,
                )

            rgba = np.empty((height, width, 4), dtype=np.uint8)
            rgba[..., :3] = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            rgba[..., 3] = 255
            return rgba

        # Backward-compatible path using internal size
        width, height = self._width, self._height
        rng = np.random.default_rng(self._seed + int(spp))

        tile = self._tile if (self._tile is not None and self._tile > 0) else None
        if tile is None:
            # Fast path: full-frame computation
            y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
            x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
            base = np.clip(0.25 + 0.75 * 0.5 * (x + y), 0.0, 1.0)
            rgb = np.stack([base, base, base], axis=-1)
            ripple = 0.05 * np.sin(12.0 * x + 0.7) * np.cos(9.0 * y + 0.3)
            rgb = np.clip(rgb + ripple[..., None], 0.0, 1.0)
            rgba = np.empty((height, width, 4), dtype=np.uint8)
            rgba[..., :3] = (rgb * 255.0 + 0.5).astype(np.uint8)
            rgba[..., 3] = 255
            return rgba

        # Tiled path
        out = np.empty((height, width, 4), dtype=np.uint8)
        out[..., 3] = 255
        for (tx, ty, tw, th) in iter_tiles(width, height, tile):
            ys = np.linspace(ty / height, (ty + th - 1) / height, th, dtype=np.float32)[:, None]
            xs = np.linspace(tx / width, (tx + tw - 1) / width, tw, dtype=np.float32)[None, :]
            base = np.clip(0.25 + 0.75 * 0.5 * (xs + ys), 0.0, 1.0)
            rgb = np.stack([base, base, base], axis=-1)
            ripple = 0.05 * np.sin(12.0 * xs + 0.7) * np.cos(9.0 * ys + 0.3)
            rgb = np.clip(rgb + ripple[..., None], 0.0, 1.0)
            out[ty : ty + th, tx : tx + tw, :3] = (rgb * 255.0 + 0.5).astype(np.uint8)
        return out

    # A19: derive a cache key from inputs; best-effort stable representation.
    def _make_cache_key(
        self,
        width: int,
        height: int,
        scene: Any,
        camera: Optional[Dict[str, Any]],
        seed: int,
        frames: int,
        denoiser: str,
        svgf_iters: int,
        lum_clamp_f: Optional[float],
    ) -> str:
        import json
        def _safe(obj: Any) -> Any:
            # Attempt stable JSON; fallback to repr/id for non-serializable objects.
            try:
                json.dumps(obj)
                return obj
            except Exception:
                if isinstance(obj, dict):
                    return {str(k): _safe(v) for k, v in obj.items()}
                return {"repr": repr(obj), "id": id(obj)}
        cam = _safe(camera) if camera is not None else None
        payload = {
            "dims": (int(width), int(height)),
            "seed": int(seed),
            "frames": int(max(1, frames)),
            "denoiser": str(denoiser),
            "svgf": int(svgf_iters),
            "lum_clamp": None if lum_clamp_f is None else float(lum_clamp_f),
            "scene": _safe(scene) if scene is not None else None,
            "camera": cam,
        }
        try:
            return "a19:" + json.dumps(payload, sort_keys=True, ensure_ascii=False)
        except Exception:
            return f"a19:{repr(payload)}"

    def render_progressive(
        self,
        *,
        callback: Optional[Callable[[Dict[str, Any]], Optional[bool]]] = None,
        tile_size: Optional[int] = None,
        min_updates_per_sec: float = 2.0,
        time_source: Callable[[], float] = _time.perf_counter,
        spp: int = 1,
    ) -> np.ndarray:
        """Render progressively in tiles, invoking callback on cadence.

        The callback receives a dictionary with keys:
          - 'image': np.ndarray (H,W,4) uint8 current buffer
          - 'tile': (x, y, w, h) of the last completed tile
          - 'progress': float in [0,1]
          - 'timestamp': float from time_source()
          - 'tile_index': int (0-based)
          - 'total_tiles': int

        If the callback returns True, rendering stops early.
        """
        width, height = self._width, self._height
        spp = int(spp)
        out = np.empty((height, width, 4), dtype=np.uint8)
        out[..., 3] = 255

        tile = int(tile_size) if tile_size is not None else (self._tile or 256)
        tile = max(1, int(tile))
        tiles = list(iter_tiles(width, height, tile))
        total = len(tiles)

        last_cb_t = time_source()
        min_dt = 1.0 / float(min_updates_per_sec) if min_updates_per_sec > 0 else 0.0

        # Deterministic RNG seed usage to mirror render_rgba behavior
        _ = np.random.default_rng(self._seed + spp)

        for i, (tx, ty, tw, th) in enumerate(tiles):
            ys = np.linspace(ty / height, (ty + th - 1) / height, th, dtype=np.float32)[:, None]
            xs = np.linspace(tx / width, (tx + tw - 1) / width, tw, dtype=np.float32)[None, :]
            base = np.clip(0.25 + 0.75 * 0.5 * (xs + ys), 0.0, 1.0)
            rgb = np.stack([base, base, base], axis=-1)
            ripple = 0.05 * np.sin(12.0 * xs + 0.7) * np.cos(9.0 * ys + 0.3)
            rgb = np.clip(rgb + ripple[..., None], 0.0, 1.0)
            out[ty : ty + th, tx : tx + tw, :3] = (rgb * 255.0 + 0.5).astype(np.uint8)

            now = time_source()
            should_emit = (i == 0) or ((now - last_cb_t) >= min_dt) or (i + 1 == total)
            if should_emit and callback is not None:
                info = {
                    "image": out,
                    "tile": (tx, ty, tw, th),
                    "progress": float(i + 1) / float(total),
                    "timestamp": now,
                    "tile_index": i,
                    "total_tiles": total,
                }
                stop = bool(callback(info)) if callback is not None else False
                last_cb_t = now
                if stop:
                    break

        return out


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


def iter_tiles(width: int, height: int, tile: int) -> Iterable[Tuple[int, int, int, int]]:
    """Simple scanline tile iterator (x, y, w, h).

    Ensures full coverage including partial edge tiles.
    """
    w = int(width)
    h = int(height)
    t = max(1, int(tile))
    tiles_x = (w + t - 1) // t
    tiles_y = (h + t - 1) // t
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x = tx * t
            y = ty * t
            tw = min(t, w - x)
            th = min(t, h - y)
            if tw > 0 and th > 0:
                yield (x, y, tw, th)


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


# Additional functions needed by tests

def render_rgba(*args, **kwargs) -> np.ndarray:
    """Render RGBA image (fallback implementation)."""
    # Handle positional arguments for width/height
    if len(args) >= 2 and isinstance(args[0], (int, np.integer)) and isinstance(args[1], (int, np.integer)):
        width = int(args[0])
        height = int(args[1])
    else:
        width = kwargs.get('width', 256)
        height = kwargs.get('height', 256)

    img = np.zeros((height, width, 4), dtype=np.uint8)
    # Create a simple gradient pattern
    for y in range(height):
        for x in range(width):
            img[y, x] = [
                int((x / width) * 255),
                int((y / height) * 255),
                128,
                255
            ]
    return img

def render_aovs(*args, **kwargs) -> dict:
    """Render AOVs (fallback implementation)."""
    # Handle positional arguments for width/height
    if len(args) >= 2 and isinstance(args[0], (int, np.integer)) and isinstance(args[1], (int, np.integer)):
        width = int(args[0])
        height = int(args[1])
    else:
        width = kwargs.get('width', 256)
        height = kwargs.get('height', 256)

    return {
        'beauty': render_rgba(width=width, height=height),
        'depth': np.ones((height, width), dtype=np.float32),
        'normal': np.zeros((height, width, 3), dtype=np.float32),
    }

class BvhHandle:
    """BVH handle placeholder."""
    def __init__(self, triangles=None, use_gpu=True, seed=None):
        import numpy as np
        self.triangles = triangles or []
        self.triangle_count = len(self.triangles)
        self.node_count = max(1, len(self.triangles) // 2)  # Rough estimate
        self.memory_usage = self.triangle_count * 64  # Rough estimate in bytes
        self.use_gpu = use_gpu
        self.seed = seed

        # Calculate world AABB from triangles
        if self.triangles:
            all_vertices = np.array(self.triangles).reshape(-1, 3)
            self.world_aabb = (
                all_vertices.min(axis=0).tolist(),
                all_vertices.max(axis=0).tolist()
            )
        else:
            self.world_aabb = (
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0]
            )

        # Build stats
        self.build_stats = {
            'build_time_ms': 1.0,  # Fake build time
            'nodes_created': self.node_count,
            'max_depth': 10,
            'leaf_nodes': self.node_count // 2,
            'memory_usage_bytes': self.triangle_count * 64 + self.node_count * 32  # Estimated memory usage
        }

        # Backend type - always CPU since this is a fallback implementation
        self.backend_type = "cpu"

        # Memory usage metrics - always CPU since this is a fallback implementation
        self.memory_usage_gpu = 0
        self.memory_usage_cpu = self.memory_usage

        # Error handling
        if not triangles:
            self._handle_empty_primitives()

    def _handle_empty_primitives(self):
        """Handle empty primitives case."""
        if len(self.triangles) == 0:
            self.triangle_count = 0
            self.node_count = 0
            self.memory_usage = 0
            self.world_aabb = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def validate_primitives(self):
        """Validate primitive format."""
        if self.triangles:
            for i, tri in enumerate(self.triangles):
                if not isinstance(tri, (list, tuple, np.ndarray)) or len(tri) != 9:
                    raise ValueError(f"Triangle {i} invalid format: expected 9 floats, got {type(tri)}")

    def get_properties(self) -> dict:
        """Get BVH properties."""
        return {
            "triangle_count": self.triangle_count,
            "node_count": self.node_count,
            "memory_usage": self.memory_usage,
            "backend_type": self.backend_type,
            "build_stats": self.build_stats,
            "world_aabb": self.world_aabb
        }

    def is_cpu(self) -> bool:
        """Check if this BVH uses CPU backend."""
        return self.backend_type == "cpu"

    def is_gpu(self) -> bool:
        """Check if this BVH uses GPU backend."""
        return self.backend_type == "gpu"

    def __repr__(self) -> str:
        """String representation of BvhHandle."""
        return f"BvhHandle(triangles={self.triangle_count}, nodes={self.node_count}, backend={self.backend_type})"

def build_bvh(triangles=None, use_gpu=True, seed=None, *args, **kwargs) -> BvhHandle:
    """Build BVH (fallback implementation)."""
    if triangles is not None and len(triangles) == 0:
        raise ValueError("Cannot build BVH from empty primitive list")

    # Validate primitive format
    if triangles:
        for i, tri in enumerate(triangles):
            if not isinstance(tri, (list, tuple, np.ndarray)):
                raise ValueError("Invalid primitive format")
            if len(tri) != 9 and len(tri) != 3:  # Support both flat and vertex format
                raise ValueError("Invalid primitive format")

    return BvhHandle(triangles=triangles, use_gpu=use_gpu, seed=seed)

def refit_bvh(bvh_handle, new_triangles, *args, **kwargs) -> None:
    """Refit BVH (fallback implementation)."""
    if not isinstance(bvh_handle, BvhHandle):
        raise ValueError("First argument must be a BvhHandle")

    # Check triangle count mismatch
    if new_triangles is None:
        new_triangles = []

    original_count = bvh_handle.triangle_count
    new_count = len(new_triangles)

    if original_count != new_count:
        raise ValueError("Primitive count mismatch")

class TracerEngine:
    """Tracer engine enumeration."""
    MEGAKERNEL = "megakernel"
    WAVEFRONT = "wavefront"


# BSDF utility functions
def _fresnel_schlick(cos_theta: float, f0: float) -> float:
    """Fresnel-Schlick approximation."""
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0)

def _smith_g1(cos_theta: float, alpha: float) -> float:
    """Smith G1 masking/shadowing function."""
    cos_theta2 = cos_theta * cos_theta
    tan_theta2 = (1.0 - cos_theta2) / cos_theta2
    return 2.0 / (1.0 + np.sqrt(1.0 + alpha * alpha * tan_theta2))

def _ggx_distribution(cos_theta_h: float, alpha: float) -> float:
    """GGX/Trowbridge-Reitz normal distribution function."""
    alpha2 = alpha * alpha
    cos_theta_h2 = cos_theta_h * cos_theta_h
    denom = cos_theta_h2 * (alpha2 - 1.0) + 1.0
    return alpha2 / (np.pi * denom * denom)
