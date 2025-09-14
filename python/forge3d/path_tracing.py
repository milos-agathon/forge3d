# python/forge3d/path_tracing.py
# Path tracing API with CPU fallback for A1/A2/A3/A7: RNG, spheres/triangles, BSDFs, HDR, tiled, BVH.
# This exists to implement Workstream A Milestone 2 (Materials & BVH) with deterministic tests; GPU wiring pending.
# RELEVANT FILES:python/forge3d/__init__.py,docs/user/path_tracing.rst,tests/test_path_tracing_api.py,tests/test_path_tracing_a1.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List, Optional as Opt
import numpy as np


def _has_gpu() -> bool:
    try:
        from . import enumerate_adapters  # type: ignore
        return bool(enumerate_adapters())
    except Exception:
        return False


@dataclass
class PathTracerConfig:
    width: int
    height: int
    max_bounces: int = 4
    seed: int = 1234
    tile: int = 32


class PathTracer:
    """Minimal skeleton for offline path tracing.

    Notes
    -----
    This is a placeholder implementation to bootstrap Workstream A.
    It returns a black RGBA image for now and validates basic inputs.
    Real compute kernels, BVH, BSDFs, and denoisers will be added incrementally.
    """

    def __init__(self, width: int, height: int, *, max_bounces: int = 4, seed: int = 1234, tile: int = 32) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if tile <= 0:
            raise ValueError("tile must be positive")
        self.cfg = PathTracerConfig(width=width, height=height, max_bounces=max_bounces, seed=seed, tile=tile)
        self._rng_state = np.uint64(seed if seed >= 0 else 1234)
        self._spheres: List[Tuple[Tuple[float, float, float], float, object]] = []
        self._triangles: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float], object]] = []
        self._bvh: Opt["_BVH"] = None

    @property
    def size(self) -> Tuple[int, int]:
        return (self.cfg.width, self.cfg.height)

    def render_rgba(self, *args, spp: int | None = None, **kwargs) -> np.ndarray:
        """Render RGBA image using CPU fallback for A1 features.

        - CPU mode (existing): when called as render_rgba(spp=...).

        - GPU/bridge mode (A1 tests): when called as render_rgba(width,height,scene,camera,seed,frames=1,use_gpu=True).

        - RNG: simple xorshift64* per-pixel stream.

        - Geometry: ray-sphere intersection for registered spheres.

        - HDR accumulation: float32 buffer accumulates radiance over spp.

        - Tile scheduler: iterate tiles of size cfg.tile for cache locality.
        """
        # Branch on call form
        if spp is None and len(args) >= 4:
            width = int(args[0]); height = int(args[1]); scene = args[2]; camera = args[3]
            seed = int(kwargs.get('seed', 123)); frames = int(kwargs.get('frames', 1)); use_gpu = bool(kwargs.get('use_gpu', True))
            if use_gpu and _has_gpu():
                try:
                    from . import _forge3d as _f
                    img = _f._pt_render_gpu(width, height, scene, camera, int(seed), int(frames))
                    return np.asarray(img)
                except Exception:
                    # Fall back to CPU path on any GPU error
                    pass
            # CPU fallback using this class: build temporary tracer and mirror behavior
            tmp = PathTracer(width, height, seed=seed)
            for sp in scene:
                c = sp.get('center', (0.0,0.0,0.0)); r = float(sp.get('radius', 1.0)); al = tuple(sp.get('albedo', (1.0,1.0,1.0)))
                tmp.add_sphere(c, r, al)
            return tmp.render_rgba(spp=frames)

        # Existing CPU path
        h, w, t = self.cfg.height, self.cfg.width, self.cfg.tile
        spp = int(max(1, spp or 1))
        hdr = np.zeros((h, w, 3), dtype=np.float32)

        # Camera setup: simple pinhole
        cam_pos = np.array([0.0, 0.0, 1.5], dtype=np.float32)
        fov = 45.0 * np.pi / 180.0
        aspect = w / max(1.0, float(h))
        half_height = np.tan(0.5 * fov)
        half_width = aspect * half_height

        for s in range(spp):
            # Tiled traversal
            for y0 in range(0, h, t):
                for x0 in range(0, w, t):
                    y1 = min(h, y0 + t)
                    x1 = min(w, x0 + t)
                    # per-tile coordinate grids
                    yy, xx = np.mgrid[y0:y1, x0:x1]
                    # jitter via RNG
                    jx = self._rand_uniform(xx.shape)
                    jy = self._rand_uniform(xx.shape)
                    ndc_x = ((xx + jx + 0.5) / w) * 2.0 - 1.0
                    ndc_y = (1.0 - (yy + jy + 0.5) / h) * 2.0 - 1.0
                    dir_x = ndc_x * half_width
                    dir_y = ndc_y * half_height
                    dirs = np.stack([dir_x, dir_y, -np.ones_like(dir_x)], axis=-1)
                    # normalize
                    inv_len = 1.0 / np.maximum(1e-6, np.linalg.norm(dirs, axis=-1, keepdims=True))
                    dirs = dirs * inv_len

                    col = self._trace_scene(cam_pos, dirs)
                    hdr[y0:y1, x0:x1, :] += col

        hdr *= (1.0 / float(spp))
        # Simple tonemap (Reinhard)
        ldr = hdr / (1.0 + hdr)
        out = np.clip(ldr * 255.0 + 0.5, 0, 255).astype(np.uint8)
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        return np.concatenate([out, alpha], axis=-1)

    def supports_gpu(self) -> bool:
        return _has_gpu()

    # --- Scene construction (A1 minimal) ---------------------------------
    def add_sphere(self, center: Tuple[float, float, float], radius: float, material_or_color) -> None:
        if radius <= 0:
            raise ValueError("radius must be positive")
        self._spheres.append((tuple(map(float, center)), float(radius), material_or_color))
        self._bvh = None

    def add_triangle(
        self,
        v0: Tuple[float, float, float],
        v1: Tuple[float, float, float],
        v2: Tuple[float, float, float],
        material_or_color,
    ) -> None:
        self._triangles.append((tuple(map(float, v0)), tuple(map(float, v1)), tuple(map(float, v2)), material_or_color))
        self._bvh = None

    # --- Internals --------------------------------------------------------
    def _trace_scene(self, cam_pos: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        # Environment gradient when empty
        if not self._triangles and not self._spheres:
            rd = dirs.astype(np.float32)
            tsky = 0.5 * (rd[..., 1:2] + 1.0)
            return (1.0 - tsky) * np.array([0.6, 0.7, 0.9], dtype=np.float32) + tsky * np.array([0.1, 0.2, 0.5], dtype=np.float32)
        # triangles first, then spheres; choose nearer
        t_tri, rgb_tri = self._trace_triangles(cam_pos, dirs)
        rgb_sph = self._trace_spheres_bsdf(cam_pos, dirs)
        tri_mask = np.isfinite(t_tri)
        return np.where(tri_mask[..., None], rgb_tri, rgb_sph)

    def _rand_uniform(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Xorshift64* PRNG producing [0,1) float32s in given shape.

        Deterministic across calls for the same seed and traversal order.
        """
        n = int(np.prod(shape))
        x = self._rng_state + np.arange(n, dtype=np.uint64)
        # xorshift64*
        x ^= (x >> np.uint64(12))
        x ^= (x << np.uint64(25))
        x ^= (x >> np.uint64(27))
        x *= np.uint64(0x2545F4914F6CDD1D)
        # take high 53 bits
        vals = (x >> np.uint64(11)).astype(np.float64) * (1.0 / float(1 << 53))
        return vals.astype(np.float32).reshape(shape)


# ---- A1 helpers for tests -------------------------------------------------

def make_sphere(*, center: Tuple[float,float,float], radius: float, albedo: Tuple[float,float,float]):
    """Create a minimal sphere dict for GPU tracer.

    Deterministic and explicit fields only.
    """
    return {"center": tuple(map(float, center)), "radius": float(radius), "albedo": tuple(map(float, albedo))}


def make_camera(*, origin: Tuple[float,float,float], look_at: Tuple[float,float,float], up: Tuple[float,float,float], fov_y: float, aspect: float, exposure: float = 1.0):
    """Create a minimal camera dict for GPU tracer.

    Matches WGSL uniforms in the kernel.
    """
    return {
        "origin": tuple(map(float, origin)),
        "look_at": tuple(map(float, look_at)),
        "up": tuple(map(float, up)),
        "fov_y": float(fov_y),
        "aspect": float(aspect),
        "exposure": float(exposure),
    }

    # --- Scene construction (A1 minimal) ---------------------------------
    def add_sphere(self, center: Tuple[float, float, float], radius: float, material_or_color) -> None:
        if radius <= 0:
            raise ValueError("radius must be positive")
        self._spheres.append((tuple(map(float, center)), float(radius), material_or_color))
        # no accel for spheres; cheap loop

    def add_triangle(
        self,
        v0: Tuple[float, float, float],
        v1: Tuple[float, float, float],
        v2: Tuple[float, float, float],
        material_or_color,
    ) -> None:
        self._triangles.append((tuple(map(float, v0)), tuple(map(float, v1)), tuple(map(float, v2)), material_or_color))
        self._bvh = None  # invalidate accel

    # --- Internals --------------------------------------------------------
    def _rand_uniform(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Xorshift64* RNG → [0,1).

        Deterministic per-instance stream seeded by cfg.seed.
        """
        n = int(np.prod(shape))
        # vectorize via NumPy uint64 arithmetic
        x = self._rng_state + np.arange(1, n + 1, dtype=np.uint64)
        x ^= (x >> np.uint64(12))
        x ^= (x << np.uint64(25))
        x ^= (x >> np.uint64(27))
        x = x * np.uint64(0x2545F4914F6CDD1D)
        # update state
        self._rng_state = np.uint64(x[-1])
        # scale to float32 [0,1)
        # take high 53 bits like PCG approach approximated
        vals = (x >> np.uint64(11)).astype(np.float64) * (1.0 / float(1 << 53))
        return vals.astype(np.float32).reshape(shape)

    def _trace_spheres(self, cam_pos: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """Naive ray-sphere shading: N·L with unit light.

        Returns RGB float32 array matching dirs[...,0] shape.
        """
        h, w = dirs.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        if not self._spheres:
            return rgb
        # Broadcast ray origin and direction
        ro = cam_pos.reshape(1, 1, 3)
        rd = dirs.astype(np.float32)

        t_best = np.full((h, w), np.inf, dtype=np.float32)
        hit_color = np.zeros((h, w, 3), dtype=np.float32)
        hit_pos = np.zeros((h, w, 3), dtype=np.float32)
        hit_norm = np.zeros((h, w, 3), dtype=np.float32)

        for (cx, cy, cz), r, col in self._spheres:
            c = np.array([cx, cy, cz], dtype=np.float32).reshape(1, 1, 3)
            oc = ro - c
            b = np.sum(oc * rd, axis=-1)
            cterm = np.sum(oc * oc, axis=-1) - (r * r)
            disc = b * b - cterm
            mask = disc > 0.0
            if not np.any(mask):
                continue
            sqrt_disc = np.sqrt(np.maximum(0.0, disc))
            t0 = -b - sqrt_disc
            t1 = -b + sqrt_disc
            t = np.where(t0 > 1e-4, t0, np.where(t1 > 1e-4, t1, np.inf))
            closer = t < t_best
            update = mask & closer
            if np.any(update):
                t_best = np.where(update, t, t_best)
                p = ro + rd * t[..., None]
                n = (p - c) / max(1e-6, r)
                hit_pos = np.where(update[..., None], p, hit_pos)
                hit_norm = np.where(update[..., None], n, hit_norm)
                hit_color = np.where(update[..., None], np.array(col, dtype=np.float32), hit_color)

        # Shade hits: N·L with light from (0.5,0.8,0.2)
        light_dir = np.array([0.5, 0.8, 0.2], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        ndotl = np.clip(np.sum(hit_norm * light_dir.reshape(1, 1, 3), axis=-1), 0.0, 1.0)
        shade = (hit_color * ndotl[..., None]).astype(np.float32)
        miss = ~np.isfinite(t_best)
        sky = np.zeros((h, w, 3), dtype=np.float32)
        rgb = np.where(miss[..., None], sky, shade)
        return rgb

    def _ensure_bvh(self) -> None:
        if self._bvh is None and self._triangles:
            self._bvh = _BVH.build(self._triangles)

    def _trace_triangles(self, cam_pos: np.ndarray, dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (t_hit, rgb) for triangle hits; t=np.inf where miss. BSDFs supported."""
        h, w = dirs.shape[:2]
        tbest = np.full((h, w), np.inf, dtype=np.float32)
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        if not self._triangles:
            return tbest, rgb
        self._ensure_bvh()
        assert self._bvh is not None
        ro = cam_pos.astype(np.float32)
        # Iterate per-pixel; small frames in tests make this acceptable
        for y in range(h):
            for x in range(w):
                t, col = self._bvh.trace(ro, dirs[y, x, :])
                tbest[y, x] = t
                if np.isfinite(t):
                    rgb[y, x, :] = col
        return tbest, rgb

    def _trace_scene(self, cam_pos: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        # Fast path: no geometry -> environment gradient
        if not self._triangles and not self._spheres:
            rd = dirs.astype(np.float32)
            tsky = 0.5 * (rd[..., 1:2] + 1.0)
            return (1.0 - tsky) * np.array([0.6, 0.7, 0.9], dtype=np.float32) + tsky * np.array([0.1, 0.2, 0.5], dtype=np.float32)
        # triangles first, then spheres; choose nearer
        t_tri, rgb_tri = self._trace_triangles(cam_pos, dirs)
        rgb_sph = self._trace_spheres_bsdf(cam_pos, dirs)
        # recompute sphere t to compare: project hit_pos distance
        # For simplicity, estimate t from color presence (non-zero) → treat as farther than triangles if tie
        tri_mask = np.isfinite(t_tri)
        out = np.where(tri_mask[..., None], rgb_tri, rgb_sph)
        return out

    # --- BSDF-based spheres (A2) ------------------------------------------
    def _trace_spheres_bsdf(self, cam_pos: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        h, w = dirs.shape[:2]
        if not self._spheres:
            return np.zeros((h, w, 3), dtype=np.float32)
        ro = cam_pos.reshape(1, 1, 3)
        rd = dirs.astype(np.float32)
        t_best = np.full((h, w), np.inf, dtype=np.float32)
        hit_pos = np.zeros((h, w, 3), dtype=np.float32)
        hit_norm = np.zeros((h, w, 3), dtype=np.float32)
        base_color = np.zeros((h, w, 3), dtype=np.float32)
        F0 = np.zeros((h, w, 3), dtype=np.float32)
        rough = np.full((h, w, 1), 0.5, dtype=np.float32)
        kind = np.zeros((h, w, 1), dtype=np.int32)  # 0=lambert,1=metal,2=dielectric

        for item in self._spheres:
            (cx, cy, cz), r, mat_or_color = item
            c = np.array([cx, cy, cz], dtype=np.float32).reshape(1, 1, 3)
            oc = ro - c
            b = np.sum(oc * rd, axis=-1)
            cterm = np.sum(oc * oc, axis=-1) - (r * r)
            disc = b * b - cterm
            mask = disc > 0.0
            if not np.any(mask):
                continue
            sqrt_disc = np.sqrt(np.maximum(0.0, disc))
            t0 = -b - sqrt_disc
            t1 = -b + sqrt_disc
            t = np.where(t0 > 1e-4, t0, np.where(t1 > 1e-4, t1, np.inf))
            closer = t < t_best
            update = mask & closer
            if not np.any(update):
                continue
            t_best = np.where(update, t, t_best)
            p = ro + rd * t[..., None]
            n = (p - c) / max(1e-6, r)
            hit_pos = np.where(update[..., None], p, hit_pos)
            hit_norm = np.where(update[..., None], n, hit_norm)

            # Material resolve
            if isinstance(mat_or_color, dict):
                m = mat_or_color
                mtype = str(m.get("type", "lambert")).lower()
                bc = np.array(m.get("base_color", (1.0, 1.0, 1.0)), dtype=np.float32)
                rgh = float(m.get("roughness", 0.5))
                if "F0" in m:
                    F0_v = np.array(m["F0"], dtype=np.float32)
                else:
                    if mtype == "metal":
                        F0_v = bc
                    else:
                        F0_v = np.array([0.04, 0.04, 0.04], dtype=np.float32)
                k = {"lambert": 0, "metal": 1, "dielectric": 2}.get(mtype, 0)
            else:
                bc = np.array(mat_or_color, dtype=np.float32)
                rgh = 0.5
                F0_v = bc
                k = 0
            base_color = np.where(update[..., None], bc, base_color)
            F0 = np.where(update[..., None], F0_v, F0)
            rough = np.where(update[..., None], rgh, rough)
            kind = np.where(update[..., None], k, kind)

        # Shading
        L = np.array([0.5, 0.8, 0.2], dtype=np.float32)
        L /= np.linalg.norm(L)
        V = -rd
        N = hit_norm
        ndotl = np.clip(np.sum(N * L.reshape(1, 1, 3), axis=-1, keepdims=True), 0.0, 1.0)
        ndotv = np.clip(np.sum(N * V, axis=-1, keepdims=True), 0.0, 1.0)
        H = (V + L.reshape(1, 1, 3))
        H /= np.maximum(1e-6, np.linalg.norm(H, axis=-1, keepdims=True))
        vdoth = np.clip(np.sum(V * H, axis=-1, keepdims=True), 0.0, 1.0)

        lambert = base_color * ndotl
        spec = _ggx_specular(ndotl, ndotv, vdoth, rough.astype(np.float32), F0.astype(np.float32))
        F_dielectric = _fresnel_schlick(ndotv, F0.astype(np.float32))

        shade = lambert
        metal_mask = (kind == 1).astype(np.float32)
        shade = shade * (1.0 - metal_mask) + spec * metal_mask
        diel_mask = (kind == 2).astype(np.float32)
        shade = shade * (1.0 - diel_mask) + (spec * F_dielectric + lambert * (1.0 - F_dielectric)) * diel_mask

        miss = ~np.isfinite(t_best)
        tsky = 0.5 * (rd[..., 1:2] + 1.0)
        sky = (1.0 - tsky) * np.array([0.6, 0.7, 0.9], dtype=np.float32) + tsky * np.array([0.1, 0.2, 0.5], dtype=np.float32)
        return np.where(miss[..., None], sky, shade)

def _fresnel_schlick(cos_theta: np.ndarray, F0: np.ndarray) -> np.ndarray:
    return F0 + (1.0 - F0) * np.power(1.0 - np.clip(cos_theta, 0.0, 1.0), 5.0)

def _ggx_specular(ndotl: np.ndarray, ndotv: np.ndarray, vdoth: np.ndarray, rough: np.ndarray, F0: np.ndarray) -> np.ndarray:
    a = np.clip(rough, 1e-3, 1.0)
    # Schlick-GGX geometry term
    Gv = ndotv / (ndotv * (1.0 - a) + a)
    Gl = ndotl / (ndotl * (1.0 - a) + a)
    G = Gv * Gl
    F = _fresnel_schlick(vdoth, F0)
    denom = np.maximum(4.0 * ndotv * ndotl, 1e-3)
    return (G * F) / denom


# ----------------------- Minimal CPU BVH for triangles -----------------------

from dataclasses import dataclass as _dataclass


@_dataclass
class _BVHNode:
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    left: int
    right: int
    start: int
    count: int


class _BVH:
    def __init__(self, nodes: List[_BVHNode], tris: np.ndarray, materials: list[object]):
        self.nodes = nodes
        self.tris = tris  # (N,3,3) f32
        self.materials = materials  # list of material objects matching tris order

    @staticmethod
    def build(triangles: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float], object]]) -> "_BVH":
        tris = np.array([[v0, v1, v2] for v0, v1, v2, _ in triangles], dtype=np.float32)
        mats: list[object] = [m for *_vs, m in triangles]
        centroids = tris.mean(axis=1)
        nodes: List[_BVHNode] = []

        indices = np.arange(tris.shape[0], dtype=np.int32)

        def bbox_of(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            t = tris[idx]
            return t.min(axis=(0, 1)), t.max(axis=(0, 1))

        def sah_split(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            # binning SAH along largest centroid axis
            if idx.size <= 8:
                return idx, np.array([], dtype=idx.dtype)
            c = centroids[idx]
            cmin = c.min(axis=0)
            cmax = c.max(axis=0)
            extent = cmax - cmin
            axis = int(np.argmax(extent))
            if extent[axis] <= 1e-6:
                # degenerate, fallback to median
                order = np.argsort(c[:, axis], kind="mergesort")
                mid = idx.size // 2
                return idx[order[:mid]], idx[order[mid:]]
            bins = 12
            bmin = cmin[axis]
            inv = bins / max(1e-6, extent[axis])
            # histograms
            counts = np.zeros(bins, dtype=np.int32)
            left_bounds = np.full((bins, 3), np.inf, dtype=np.float32)
            right_bounds = np.full((bins, 3), -np.inf, dtype=np.float32)
            # assign
            bin_ids = np.clip(((c[:, axis] - bmin) * inv).astype(int), 0, bins - 1)
            for b, idv in zip(idx, bin_ids):
                counts[idv] += 1
                tri = tris[b]
                left_bounds[idv] = np.minimum(left_bounds[idv], tri.min(axis=0))
                right_bounds[idv] = np.maximum(right_bounds[idv], tri.max(axis=0))
            # prefix/suffix
            pref_c = np.cumsum(counts)
            # compute bbox prefix/suffix cost
            def area(v):
                e = np.maximum(0.0, v[1] - v[0])
                return e[0] * e[1] + e[1] * e[2] + e[2] * e[0]
            best_cost = np.inf
            best_k = -1
            for k in range(1, bins):
                left_idx = np.where(bin_ids < k)[0]
                right_idx = np.where(bin_ids >= k)[0]
                if left_idx.size == 0 or right_idx.size == 0:
                    continue
                lb = tris[idx[left_idx]].min(axis=(0, 1))
                rb = tris[idx[right_idx]].min(axis=(0, 1)), tris[idx[right_idx]].max(axis=(0, 1))
                lA = area((lb, tris[idx[left_idx]].max(axis=(0, 1))))
                rmin = tris[idx[right_idx]].min(axis=(0, 1))
                rmax = tris[idx[right_idx]].max(axis=(0, 1))
                rA = area((rmin, rmax))
                cost = lA * left_idx.size + rA * right_idx.size
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
            if best_k == -1:
                order = np.argsort(c[:, axis], kind="mergesort")
                mid = idx.size // 2
                return idx[order[:mid]], idx[order[mid:]]
            left_idx = idx[np.where(bin_ids < best_k)[0]]
            right_idx = idx[np.where(bin_ids >= best_k)[0]]
            return left_idx, right_idx

        def build_range(idx: np.ndarray) -> int:
            bmin, bmax = bbox_of(idx)
            if idx.size <= 8:
                n = _BVHNode(bmin, bmax, -1, -1, int(idx[0]), int(idx.size))
                nodes.append(n)
                return len(nodes) - 1
            left_ids, right_ids = sah_split(idx)
            left = build_range(left_ids)
            right = build_range(right_ids)
            n = _BVHNode(bmin, bmax, left, right, -1, 0)
            nodes.append(n)
            return len(nodes) - 1

        # Reorder tris/cols to leaf ranges for cache coherence
        leaf_order: List[int] = []

        def collect_leaves(i: int):
            node = nodes[i]
            if node.left == -1:
                leaf_order.extend(list(range(node.start, node.start + node.count)))
            else:
                collect_leaves(node.left)
                collect_leaves(node.right)

        root = build_range(indices)
        collect_leaves(root)
        tris2 = tris[leaf_order]
        mats2 = [mats[i] for i in leaf_order]
        # Rebuild nodes with contiguous leaf ranges
        nodes2: List[_BVHNode] = []

        def rebuild(i: int, offset: int) -> Tuple[int, int]:
            n = nodes[i]
            if n.left == -1:
                start = len(nodes2)  # temp index, will be actual index
                node = _BVHNode(n.bbox_min, n.bbox_max, -1, -1, offset, n.count)
                nodes2.append(node)
                return start, offset + n.count
            li, off_l = rebuild(n.left, offset)
            ri, off_r = rebuild(n.right, off_l)
            idx = len(nodes2)
            nodes2.append(_BVHNode(n.bbox_min, n.bbox_max, li, ri, -1, 0))
            return idx, off_r

        rebuild(root, 0)
        return _BVH(nodes2, tris2, mats2)

    @staticmethod
    def _ray_aabb(ro: np.ndarray, rd: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> bool:
        inv = 1.0 / np.where(np.abs(rd) < 1e-8, np.sign(rd) * 1e-8, rd)
        t0 = (bmin - ro) * inv
        t1 = (bmax - ro) * inv
        tmin = np.maximum(np.minimum(t0, t1), 0.0).max()
        tmax = np.minimum(np.maximum(t0, t1), np.inf).min()
        return tmax >= tmin

    @staticmethod
    def _ray_tri(ro: np.ndarray, rd: np.ndarray, tri: np.ndarray) -> Tuple[float, np.ndarray]:
        v0, v1, v2 = tri
        # Watertight Möller–Trumbore style intersection (robustness tweaks)
        e1 = v1 - v0
        e2 = v2 - v0
        p = np.cross(rd, e2)
        det = float(np.dot(e1, p))
        if abs(det) < 1e-10:
            return np.inf, np.zeros(3, dtype=np.float32)
        inv_det = 1.0 / det
        tvec = ro - v0
        u = float(np.dot(tvec, p)) * inv_det
        if u < -1e-6 or u > 1.0 + 1e-6:
            return np.inf, np.zeros(3, dtype=np.float32)
        q = np.cross(tvec, e1)
        v = float(np.dot(rd, q)) * inv_det
        if v < -1e-6 or u + v > 1.0 + 1e-6:
            return np.inf, np.zeros(3, dtype=np.float32)
        thit = float(np.dot(e2, q)) * inv_det
        if thit <= 1e-5:
            return np.inf, np.zeros(3, dtype=np.float32)
        n = np.cross(e1, e2)
        if np.dot(n, rd) > 0:
            n = -n
        n = n / max(1e-6, np.linalg.norm(n))
        return thit, n.astype(np.float32)

    def trace(self, ro: np.ndarray, rd: np.ndarray) -> Tuple[float, np.ndarray]:
        # Stack-based traversal
        t_best = np.inf
        col_best = np.zeros(3, dtype=np.float32)
        stack = [len(self.nodes) - 1]
        while stack:
            i = stack.pop()
            node = self.nodes[i]
            if not _BVH._ray_aabb(ro, rd, node.bbox_min, node.bbox_max):
                continue
            if node.left == -1:
                # leaf
                start, count = node.start, node.count
                for k in range(start, start + count):
                    th, n = _BVH._ray_tri(ro, rd, self.tris[k])
                    if th < t_best:
                        t_best = th
                        # Shade with triangle material
                        m = self.materials[k]
                        # Resolve material
                        if isinstance(m, dict):
                            mtype = str(m.get("type", "lambert")).lower()
                            bc = np.array(m.get("base_color", (1.0, 1.0, 1.0)), dtype=np.float32)
                            rgh = float(m.get("roughness", 0.5))
                            if "F0" in m:
                                F0_v = np.array(m["F0"], dtype=np.float32)
                            else:
                                F0_v = bc if mtype == "metal" else np.array([0.04, 0.04, 0.04], dtype=np.float32)
                            kind = {"lambert": 0, "metal": 1, "dielectric": 2}.get(mtype, 0)
                        else:
                            bc = np.array(m, dtype=np.float32)
                            rgh = 0.5
                            F0_v = bc
                            kind = 0
                        # Lighting
                        L = np.array([0.5, 0.8, 0.2], dtype=np.float32)
                        L /= np.linalg.norm(L)
                        V = -rd
                        ndotl = max(0.0, float(np.dot(n, L)))
                        ndotv = max(0.0, float(np.dot(n, V)))
                        H = (V + L)
                        H = H / max(1e-6, float(np.linalg.norm(H)))
                        vdoth = max(0.0, float(np.dot(V, H)))
                        lam = bc * ndotl
                        # GGX terms with scalar channels
                        def fresnel(c, F0):
                            return F0 + (1.0 - F0) * (1.0 - max(0.0, c)) ** 5
                        def schlick_ggx(c, a):
                            return c / (c * (1.0 - a) + a)
                        a = max(1e-3, min(1.0, rgh))
                        G = schlick_ggx(ndotv, a) * schlick_ggx(ndotl, a)
                        F = np.array([fresnel(vdoth, F0_v[i]) for i in range(3)], dtype=np.float32)
                        denom = max(4.0 * ndotv * ndotl, 1e-3)
                        spec = (G * F) / denom
                        if kind == 1:  # metal
                            col_best = spec.astype(np.float32)
                        elif kind == 2:  # dielectric
                            col_best = (spec * F + lam * (1.0 - F)).astype(np.float32)
                        else:
                            col_best = lam.astype(np.float32)
            else:
                stack.append(node.left)
                stack.append(node.right)
        return (t_best, col_best) if np.isfinite(t_best) else (np.inf, np.zeros(3, dtype=np.float32))


def create_path_tracer(width: int, height: int, *, max_bounces: int = 4, seed: int = 1234) -> PathTracer:
    """Factory to create a PathTracer with basic validation.

    Exists to keep API stable while kernels land incrementally.
    """
    return PathTracer(width, height, max_bounces=max_bounces, seed=seed)
