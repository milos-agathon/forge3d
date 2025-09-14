# python/forge3d/path_tracing.py
# Minimal CPU path tracing used by tests: RNG, spheres/triangles, simple BSDFs, HDR accumulation.
# This file exists to provide a deterministic, dependency-free path tracing reference for tests.
# RELEVANT FILES:python/forge3d/__init__.py,docs/user/path_tracing.rst,tests/test_path_tracing_*.py

"""CPU path tracing reference used by tests and examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Any, Optional
from enum import Enum
import numpy as np


class TracerEngine(Enum):
    """Path tracing engine selection."""
    MEGAKERNEL = "megakernel"
    WAVEFRONT = "wavefront"


@dataclass
class PathTracerConfig:
    width: int
    height: int
    max_bounces: int = 1
    seed: int = 1234
    tile: int = 32


def _fresnel_schlick(cos_theta: np.ndarray, F0: np.ndarray) -> np.ndarray:
    ct = np.clip(cos_theta, 0.0, 1.0).astype(np.float32)
    return (F0 + (1.0 - F0) * (1.0 - ct) ** 5.0).astype(np.float32)


class PathTracer:
    def __init__(self, width: int, height: int, *, max_bounces: int = 1, seed: int = 1234, tile: int = 32) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if tile <= 0:
            raise ValueError("tile must be positive")
        self.cfg = PathTracerConfig(width=width, height=height, max_bounces=max_bounces, seed=seed, tile=tile)
        self._rng_state = np.uint64(seed if seed >= 0 else 1234)
        self._spheres: List[Tuple[Tuple[float, float, float], float, Any]] = []
        self._triangles: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float], Any]] = []

    @property
    def size(self) -> Tuple[int, int]:
        return (self.cfg.width, self.cfg.height)

    def add_sphere(self, center: Tuple[float, float, float], radius: float, material_or_color: Any) -> None:
        if radius <= 0:
            raise ValueError("radius must be positive")
        self._spheres.append((tuple(map(float, center)), float(radius), material_or_color))

    def add_triangle(self, v0: Tuple[float, float, float], v1: Tuple[float, float, float], v2: Tuple[float, float, float], material_or_color: Any) -> None:
        self._triangles.append((tuple(map(float, v0)), tuple(map(float, v1)), tuple(map(float, v2)), material_or_color))

    def render_rgba(self, *, spp: int = 1) -> np.ndarray:
        h, w, t = self.cfg.height, self.cfg.width, self.cfg.tile
        spp = int(max(1, spp))
        hdr = np.zeros((h, w, 3), dtype=np.float32)

        cam_pos = np.array([0.0, 0.0, 1.5], dtype=np.float32)
        fov = 45.0 * np.pi / 180.0
        aspect = w / max(1.0, float(h))
        half_height = np.tan(0.5 * fov)
        half_width = aspect * half_height

        for _ in range(spp):
            for y0 in range(0, h, t):
                for x0 in range(0, w, t):
                    y1 = min(h, y0 + t)
                    x1 = min(w, x0 + t)
                    yy, xx = np.mgrid[y0:y1, x0:x1]
                    jx = self._rand_uniform(xx.shape)
                    jy = self._rand_uniform(xx.shape)
                    ndc_x = ((xx + jx + 0.5) / w) * 2.0 - 1.0
                    ndc_y = (1.0 - (yy + jy + 0.5) / h) * 2.0 - 1.0
                    dir_x = ndc_x * half_width
                    dir_y = ndc_y * half_height
                    dirs = np.stack([dir_x, dir_y, -np.ones_like(dir_x)], axis=-1).astype(np.float32)
                    dirs /= np.maximum(1e-6, np.linalg.norm(dirs, axis=-1, keepdims=True))
                    col = self._trace_scene(cam_pos, dirs)
                    hdr[y0:y1, x0:x1, :] += col

        hdr *= (1.0 / float(spp))
        ldr = hdr / (1.0 + hdr)
        out = np.clip(ldr * 255.0 + 0.5, 0, 255).astype(np.uint8)
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        return np.concatenate([out, alpha], axis=-1)

    def render_aovs_cpu(self, aovs: Tuple[str, ...] = (
        "albedo", "normal", "depth", "direct", "indirect", "emission", "visibility"
    ), *, spp: int = 1) -> dict:
        """Compute basic AOVs on CPU.

        Returns a dict of numpy arrays keyed by canonical names.

        - albedo, normal, direct, indirect, emission: float32, shape (H, W, 3)
        - depth: float32, shape (H, W), np.nan for miss
        - visibility: uint8 mask, shape (H, W), 255 hit / 0 miss
        """
        h, w, t = self.cfg.height, self.cfg.width, self.cfg.tile
        spp = int(max(1, spp))

        want = tuple(str(k) for k in aovs)
        out: dict = {}
        if "albedo" in want:
            out["albedo"] = np.zeros((h, w, 3), dtype=np.float32)
        if "normal" in want:
            out["normal"] = np.zeros((h, w, 3), dtype=np.float32)
        if "depth" in want:
            out["depth"] = np.full((h, w), np.nan, dtype=np.float32)
        if "direct" in want:
            out["direct"] = np.zeros((h, w, 3), dtype=np.float32)
        if "indirect" in want:
            out["indirect"] = np.zeros((h, w, 3), dtype=np.float32)
        if "emission" in want:
            out["emission"] = np.zeros((h, w, 3), dtype=np.float32)
        if "visibility" in want:
            out["visibility"] = np.zeros((h, w), dtype=np.uint8)

        cam_pos = np.array([0.0, 0.0, 1.5], dtype=np.float32)
        fov = 45.0 * np.pi / 180.0
        aspect = w / max(1.0, float(h))
        half_height = np.tan(0.5 * fov)
        half_width = aspect * half_height

        light_dir = np.array([0.5, 0.8, 0.2], dtype=np.float32)
        light_dir /= max(1e-6, np.linalg.norm(light_dir))

        for _ in range(spp):
            for y0 in range(0, h, t):
                for x0 in range(0, w, t):
                    y1 = min(h, y0 + t)
                    x1 = min(w, x0 + t)
                    yy, xx = np.mgrid[y0:y1, x0:x1]
                    jx = self._rand_uniform(xx.shape)
                    jy = self._rand_uniform(xx.shape)
                    ndc_x = ((xx + jx + 0.5) / w) * 2.0 - 1.0
                    ndc_y = (1.0 - (yy + jy + 0.5) / h) * 2.0 - 1.0
                    dir_x = ndc_x * half_width
                    dir_y = ndc_y * half_height
                    dirs = np.stack([dir_x, dir_y, -np.ones_like(dir_x)], axis=-1).astype(np.float32)
                    dirs /= np.maximum(1e-6, np.linalg.norm(dirs, axis=-1, keepdims=True))

                    # Intersect scene
                    t_s, n_s, col_s, hit_s = self._intersect_spheres(cam_pos, dirs)
                    t_t, n_t, col_t, hit_t = self._intersect_triangles(cam_pos, dirs)
                    use_t = (t_t < t_s) & hit_t
                    use_s = (~use_t) & hit_s
                    hit_any = use_t | use_s

                    t_best = np.where(use_t, t_t, np.where(use_s, t_s, np.inf))
                    n_best = np.where(use_t[..., None], n_t, np.where(use_s[..., None], n_s, 0.0)).astype(np.float32)
                    c_best = np.where(use_t[..., None], col_t, np.where(use_s[..., None], col_s, 0.0)).astype(np.float32)

                    if "albedo" in out:
                        out["albedo"][y0:y1, x0:x1, :] += c_best
                    if "normal" in out:
                        out["normal"][y0:y1, x0:x1, :] += n_best
                    if "depth" in out:
                        depth_tile = out["depth"][y0:y1, x0:x1]
                        depth_tile[hit_any] = np.where(hit_any, t_best, np.nan)[hit_any].astype(np.float32)
                    if "direct" in out:
                        ndotl = np.clip(np.sum(n_best * light_dir.reshape(1, 1, 3), axis=-1, keepdims=True), 0.0, 1.0)
                        out["direct"][y0:y1, x0:x1, :] += c_best * ndotl
                    if "indirect" in out:
                        # No GI in this CPU stub; keep zeros
                        pass
                    if "emission" in out:
                        # No emissive in this CPU stub; keep zeros
                        pass
                    if "visibility" in out:
                        vis_tile = out["visibility"][y0:y1, x0:x1]
                        vis_tile[:] = np.where(hit_any, 255, 0).astype(np.uint8)

        if spp > 1:
            w3 = 1.0 / float(spp)
            for k in ("albedo", "normal", "direct", "indirect", "emission"):
                if k in out:
                    out[k] *= w3
        return out

    def _rand_uniform(self, shape: Tuple[int, ...]) -> np.ndarray:
        n = int(np.prod(shape))
        x = self._rng_state + np.arange(n, dtype=np.uint64)
        x ^= (x >> np.uint64(12))
        x ^= (x << np.uint64(25))
        x ^= (x >> np.uint64(27))
        x *= np.uint64(0x2545F4914F6CDD1D)
        vals = (x >> np.uint64(11)).astype(np.float64) * (1.0 / float(1 << 53))
        return vals.astype(np.float32).reshape(shape)

    def _trace_scene(self, cam_pos: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        if not self._triangles and not self._spheres:
            rd = dirs
            tsky = 0.5 * (rd[..., 1:2] + 1.0)
            return (1.0 - tsky) * np.array([0.6, 0.7, 0.9], dtype=np.float32) + tsky * np.array([0.1, 0.2, 0.5], dtype=np.float32)
        t_s, n_s, col_s, hit_s = self._intersect_spheres(cam_pos, dirs)
        t_t, n_t, col_t, hit_t = self._intersect_triangles(cam_pos, dirs)
        use_t = (t_t < t_s) & hit_t
        use_s = (~use_t) & hit_s
        rgb_s = self._shade(col_s, n_s)
        rgb_t = self._shade(col_t, n_t)
        return np.where(use_t[..., None], rgb_t, np.where(use_s[..., None], rgb_s, 0.0)).astype(np.float32)

    def _intersect_spheres(self, cam_pos: np.ndarray, dirs: np.ndarray):
        h, w = dirs.shape[:2]
        ro = cam_pos.reshape(1, 1, 3)
        rd = dirs
        t_best = np.full((h, w), np.inf, dtype=np.float32)
        n_best = np.zeros((h, w, 3), dtype=np.float32)
        c_best = np.zeros((h, w, 3), dtype=np.float32)
        hit = np.zeros((h, w), dtype=bool)
        for (cx, cy, cz), r, mat in self._spheres:
            c = np.array([cx, cy, cz], dtype=np.float32).reshape(1, 1, 3)
            oc = ro - c
            b = np.sum(oc * rd, axis=-1)
            cterm = np.sum(oc * oc, axis=-1) - (r * r)
            disc = b * b - cterm
            mask = disc > 0.0
            if not np.any(mask):
                continue
            s = np.sqrt(np.maximum(0.0, disc))
            t0 = -b - s
            t1 = -b + s
            t = np.where(t0 > 1e-4, t0, np.where(t1 > 1e-4, t1, np.inf))
            closer = (t < t_best) & mask
            if np.any(closer):
                t_best = np.where(closer, t, t_best)
                p = ro + rd * t[..., None]
                n = (p - c) / max(1e-6, float(r))
                n_best = np.where(closer[..., None], n, n_best)
                c_best = np.where(closer[..., None], self._material_base_color(mat), c_best)
                hit = hit | closer
        return t_best, n_best, c_best, hit

    def _intersect_triangles(self, cam_pos: np.ndarray, dirs: np.ndarray):
        h, w = dirs.shape[:2]
        ro = cam_pos.astype(np.float32)
        t_best = np.full((h, w), np.inf, dtype=np.float32)
        n_best = np.zeros((h, w, 3), dtype=np.float32)
        c_best = np.zeros((h, w, 3), dtype=np.float32)
        hit = np.zeros((h, w), dtype=bool)
        for v0, v1, v2, mat in self._triangles:
            v0 = np.array(v0, dtype=np.float32)
            v1 = np.array(v1, dtype=np.float32)
            v2 = np.array(v2, dtype=np.float32)
            e1 = v1 - v0
            e2 = v2 - v0
            n = np.cross(e1, e2)
            n = n / max(1e-6, np.linalg.norm(n))
            denom = np.dot(-n, dirs.reshape(-1, 3).T).reshape(h, w)
            denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
            t = np.dot(v0 - ro, -n) / denom
            p = ro.reshape(1, 1, 3) + dirs * t[..., None]
            def edge(a, b, p):
                return np.cross(b - a, p - a)
            c0 = edge(v0, v1, p)
            c1 = edge(v1, v2, p)
            c2 = edge(v2, v0, p)
            same = (np.sum(c0 * n, axis=-1) >= 0) & (np.sum(c1 * n, axis=-1) >= 0) & (np.sum(c2 * n, axis=-1) >= 0)
            valid = (t > 1e-4) & same
            closer = (t < t_best) & valid
            if np.any(closer):
                t_best = np.where(closer, t, t_best)
                n_best = np.where(closer[..., None], n.reshape(1, 1, 3).astype(np.float32), n_best)
                c_best = np.where(closer[..., None], self._material_base_color(mat), c_best)
                hit = hit | closer
        return t_best, n_best, c_best, hit

    def _material_base_color(self, mat: Any) -> np.ndarray:
        if isinstance(mat, dict):
            mtype = str(mat.get("type", "lambert"))
            col = mat.get("base_color", (1.0, 1.0, 1.0))
            c = np.array(col, dtype=np.float32).reshape(1, 1, 3)
            if mtype == "metal":
                c = np.clip(c * 1.1, 0.0, 4.0)
            elif mtype == "dielectric":
                c = np.clip(c * 0.95 + 0.05, 0.0, 4.0)
            return c
        return np.array(mat, dtype=np.float32).reshape(1, 1, 3)

    def _shade(self, base_color: np.ndarray, normal: np.ndarray) -> np.ndarray:
        light_dir = np.array([0.5, 0.8, 0.2], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        ndotl = np.clip(np.sum(normal * light_dir.reshape(1, 1, 3), axis=-1, keepdims=True), 0.0, 1.0)
        rgb = base_color * ndotl
        F0 = np.array([0.04, 0.04, 0.04], dtype=np.float32).reshape(1, 1, 3)
        fres = _fresnel_schlick(ndotl, F0)
        rgb = np.maximum(rgb, base_color * 0.1) + 0.2 * fres
        return rgb.astype(np.float32)


def create_path_tracer(width: int, height: int, *, max_bounces: int = 1, seed: int = 1234) -> PathTracer:
    return PathTracer(width, height, max_bounces=max_bounces, seed=seed)


def render_rgba(width: int, height: int, scene, camera, seed: int, frames: int = 1, use_gpu: bool = True, engine: Optional[TracerEngine] = None):
    """Render RGBA image using GPU when available, else CPU fallback.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        scene: Scene description (list of spheres/primitives)
        camera: Camera parameters
        seed: Random seed for deterministic rendering
        frames: Number of samples per pixel
        use_gpu: Whether to use GPU rendering
        engine: Path tracing engine (MEGAKERNEL or WAVEFRONT), defaults to MEGAKERNEL

    Returns:
        RGBA image as numpy array (H, W, 4) uint8

    Matches the A1 bridge signature for deterministic smoke testing.
    """
    if use_gpu:
        try:
            from . import _forge3d as _f  # type: ignore
            from . import enumerate_adapters  # type: ignore
            if enumerate_adapters():
                return _f._pt_render_gpu(int(width), int(height), scene, camera, int(seed), int(frames))
        except Exception:
            pass
    # CPU fallback
    t = PathTracer(int(width), int(height), seed=int(seed))
    for sp in scene or []:
        c = sp.get('center', (0.0, 0.0, 0.0)); r = float(sp.get('radius', 1.0)); mat = sp.get('albedo', (1.0, 1.0, 1.0))
        t.add_sphere(tuple(c), float(r), mat)
    return t.render_rgba(spp=int(frames))


def render_aovs(
    width: int,
    height: int,
    scene,
    camera,
    *,
    aovs: Tuple[str, ...] = ("albedo", "normal", "depth", "direct", "indirect", "emission", "visibility"),
    seed: int = 1234,
    frames: int = 1,
    use_gpu: bool = True,
    engine: Optional[TracerEngine] = None,
):
    """Render AOVs on GPU when available, else CPU fallback.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        scene: Scene description (list of spheres/primitives)
        camera: Camera parameters
        aovs: Tuple of AOV names to render
        seed: Random seed for deterministic rendering
        frames: Number of samples per pixel
        use_gpu: Whether to use GPU rendering
        engine: Path tracing engine (MEGAKERNEL or WAVEFRONT), defaults to MEGAKERNEL

    Returns:
        Dict of numpy arrays keyed by canonical AOV names.
    """
    # GPU path: not implemented yet in this workstream scaffold; fall back
    # to CPU when GPU path is unavailable or fails gracefully.
    t = PathTracer(int(width), int(height), seed=int(seed))
    for sp in scene or []:
        c = sp.get('center', (0.0, 0.0, 0.0)); r = float(sp.get('radius', 1.0)); mat = sp.get('albedo', (1.0, 1.0, 1.0))
        t.add_sphere(tuple(c), float(r), mat)
    return t.render_aovs_cpu(tuple(aovs), spp=int(frames))


def save_aovs(prefix: str, aovs: dict) -> None:
    """Save AOVs to disk.

    - HDR AOVs (albedo, normal, depth, direct, indirect, emission) saved as .npy for portability.
    - visibility saved as .png if PNG writer is available; else as .npy as fallback.

    Note: EXR output is SKIPPED in this environment to avoid adding dependencies.
    To enable EXR, install an EXR writer (e.g., OpenEXR/imageio-exr) and adjust this helper.
    """
    import os
    from . import numpy_to_png  # type: ignore

    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    hdr_keys = ("albedo", "normal", "depth", "direct", "indirect", "emission")
    for k in hdr_keys:
        if k in aovs:
            np.save(f"{prefix}_{k}.npy", aovs[k])

    if "visibility" in aovs:
        vis = aovs["visibility"].astype(np.uint8)
        try:
            numpy_to_png(f"{prefix}_visibility.png", vis)
        except Exception:
            # Fallback to npy if PNG helper unavailable
            np.save(f"{prefix}_visibility.npy", vis)


# A7: BVH Construction API

class BvhHandle:
    """Handle to a built BVH acceleration structure."""
    
    def __init__(self, backend_type: str, triangle_count: int, node_count: int, 
                 world_aabb: tuple, build_stats: dict):
        self.backend_type = backend_type  # "GPU" or "CPU"
        self.triangle_count = triangle_count
        self.node_count = node_count
        self.world_aabb = world_aabb  # ((min_x, min_y, min_z), (max_x, max_y, max_z))
        self.build_stats = build_stats
        self._native_handle = None  # Will be set by Rust code
    
    def is_gpu(self) -> bool:
        return self.backend_type == "GPU"
    
    def is_cpu(self) -> bool:
        return self.backend_type == "CPU"
    
    def __repr__(self) -> str:
        return (f"BvhHandle(backend={self.backend_type}, triangles={self.triangle_count}, "
                f"nodes={self.node_count}, build_time={self.build_stats.get('build_time_ms', 0):.1f}ms)")


def build_bvh(primitives: list, *, use_gpu: bool = True, seed: int = 1) -> BvhHandle:
    """Build BVH acceleration structure from triangles.
    
    Args:
        primitives: List of triangles, where each triangle is either:
                   - A tuple of 3 vertices: ((x1,y1,z1), (x2,y2,z2), (x3,y3,z3))
                   - A dict with 'vertices' key containing the 3 vertices
        use_gpu: Whether to prefer GPU construction (falls back to CPU if unavailable)
        seed: Random seed for deterministic construction
        
    Returns:
        BvhHandle: Handle to the built BVH
        
    Example:
        triangles = [
            ((0,0,0), (1,0,0), (0,1,0)),  # Triangle 1
            ((1,0,0), (1,1,0), (0,1,0)),  # Triangle 2
        ]
        bvh = build_bvh(triangles, use_gpu=True, seed=42)
        print(f"Built BVH with {bvh.node_count} nodes using {bvh.backend_type}")
    """
    if not primitives:
        raise ValueError("Cannot build BVH from empty primitive list")
    
    # Convert primitives to consistent format
    triangles = []
    for i, prim in enumerate(primitives):
        if isinstance(prim, (tuple, list)) and len(prim) == 3:
            # Direct tuple/list of 3 vertices
            v0, v1, v2 = prim
            triangles.append((
                tuple(v0) if not isinstance(v0, tuple) else v0,
                tuple(v1) if not isinstance(v1, tuple) else v1,
                tuple(v2) if not isinstance(v2, tuple) else v2
            ))
        elif isinstance(prim, dict) and 'vertices' in prim:
            # Dict with vertices key
            vertices = prim['vertices']
            if len(vertices) != 3:
                raise ValueError(f"Triangle {i} must have exactly 3 vertices, got {len(vertices)}")
            triangles.append(tuple(tuple(v) for v in vertices))
        else:
            raise ValueError(f"Invalid primitive format at index {i}. Expected tuple of 3 vertices or dict with 'vertices' key")
    
    # For now, return a mock BVH handle since the Rust integration is not complete
    # In the full implementation, this would call into the Rust BVH builder
    
    # Compute scene bounds for mock data
    all_vertices = []
    for tri in triangles:
        all_vertices.extend(tri)
    
    if all_vertices:
        xs, ys, zs = zip(*all_vertices)
        min_bounds = (min(xs), min(ys), min(zs))
        max_bounds = (max(xs), max(ys), max(zs))
    else:
        min_bounds = (0.0, 0.0, 0.0)
        max_bounds = (0.0, 0.0, 0.0)
    
    world_aabb = (min_bounds, max_bounds)
    
    # Mock build statistics
    build_stats = {
        'build_time_ms': len(triangles) * 0.1,  # Fake timing
        'morton_time_ms': len(triangles) * 0.02,
        'sort_time_ms': len(triangles) * 0.03,
        'link_time_ms': len(triangles) * 0.05,
        'memory_usage_bytes': len(triangles) * 200,  # Rough estimate
        'leaf_count': len(triangles),
        'internal_count': max(0, len(triangles) - 1),
        'max_depth': max(1, int(np.log2(len(triangles))) + 1),
        'avg_leaf_size': 1.0,
    }
    
    # Choose backend based on availability and preference
    backend_type = "GPU" if use_gpu and _gpu_available() else "CPU"
    
    handle = BvhHandle(
        backend_type=backend_type,
        triangle_count=len(triangles),
        node_count=len(triangles) * 2 - 1 if triangles else 0,
        world_aabb=world_aabb,
        build_stats=build_stats
    )
    
    return handle


def refit_bvh(handle: BvhHandle, new_primitives: list) -> None:
    """Refit existing BVH with updated primitive positions.
    
    Args:
        handle: BvhHandle from build_bvh()
        new_primitives: Updated triangles in same format as build_bvh()
        
    Raises:
        ValueError: If primitive count doesn't match original BVH
        
    Example:
        # Build initial BVH
        triangles = [((0,0,0), (1,0,0), (0,1,0))]
        bvh = build_bvh(triangles)
        
        # Update triangle positions and refit
        updated_triangles = [((0,0,0), (1.5,0,0), (0,1.2,0))]
        refit_bvh(bvh, updated_triangles)
    """
    if len(new_primitives) != handle.triangle_count:
        raise ValueError(f"Primitive count mismatch: BVH has {handle.triangle_count}, "
                        f"got {len(new_primitives)} new primitives")
    
    # Convert primitives to consistent format (same as build_bvh)
    triangles = []
    for i, prim in enumerate(new_primitives):
        if isinstance(prim, (tuple, list)) and len(prim) == 3:
            v0, v1, v2 = prim
            triangles.append((
                tuple(v0) if not isinstance(v0, tuple) else v0,
                tuple(v1) if not isinstance(v1, tuple) else v1,
                tuple(v2) if not isinstance(v2, tuple) else v2
            ))
        elif isinstance(prim, dict) and 'vertices' in prim:
            vertices = prim['vertices']
            if len(vertices) != 3:
                raise ValueError(f"Triangle {i} must have exactly 3 vertices, got {len(vertices)}")
            triangles.append(tuple(tuple(v) for v in vertices))
        else:
            raise ValueError(f"Invalid primitive format at index {i}")
    
    # For now, just update the world AABB in the mock implementation
    all_vertices = []
    for tri in triangles:
        all_vertices.extend(tri)
    
    if all_vertices:
        xs, ys, zs = zip(*all_vertices)
        min_bounds = (min(xs), min(ys), min(zs))
        max_bounds = (max(xs), max(ys), max(zs))
        handle.world_aabb = (min_bounds, max_bounds)
    
    # In the full implementation, this would call into Rust refit code


def _gpu_available() -> bool:
    """Check if GPU BVH construction is available."""
    try:
        # This would check if wgpu context is available
        # For now, return False since GPU implementation is not complete
        return False
    except Exception:
        return False
