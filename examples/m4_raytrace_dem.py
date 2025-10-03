#!/usr/bin/env python3
"""
examples/m4_raytrace_dem.py

Ray trace a terrain built from a DEM heightmap using Forge3D's path tracing API.
- Loads a GeoTIFF DEM (same loader as m3_geopandas_demo.py)
- Triangulates the DEM into a grid mesh using pixel spacing
- Computes a palette-based albedo for a CPU preview overlay when GPU tracer is unavailable
- Uses GPU path tracer if available (falls back to a procedural image otherwise)

Notes
-----
- The current CPU fallback of the path tracer produces a deterministic procedural image and
  ignores mesh/materials. For visual verification without GPU, enable --preview to draw
  a point-cloud of the terrain colored by the same custom palette used in m3.
- When a GPU adapter is available (f3d.enumerate_adapters()), the tracer can consume the
  uploaded mesh handle and render via the native path tracer.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

import forge3d as f3d
from forge3d.path_tracing import PathTracer, make_camera
from forge3d.mesh import make_mesh, build_bvh_cpu, upload_mesh, validate_mesh_arrays


# --- Reuse palette from m3_geopandas_demo.py ---
CUSTOM_HEX_COLORS = [
    "#AABD8A", "#E6CE99", "#D4B388",
    "#C0A181", "#AC8D75", "#9B7B62",
]


def _hex_to_rgb01(s: str) -> Tuple[float, float, float]:
    s = s.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Invalid hex color: {s}")
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b)


def _interpolate_palette_rgba(hex_colors: list[str], n: int = 128) -> np.ndarray:
    cols = np.array([_hex_to_rgb01(h) for h in hex_colors], dtype=np.float32)
    stops = np.linspace(0.0, 1.0, cols.shape[0], dtype=np.float32)
    t = np.linspace(0.0, 1.0, int(n), dtype=np.float32)
    r = np.interp(t, stops, cols[:, 0])
    g = np.interp(t, stops, cols[:, 1])
    b = np.interp(t, stops, cols[:, 2])
    table = np.empty((t.shape[0], 4), dtype=np.uint8)
    table[:, 0] = (r * 255.0 + 0.5).astype(np.uint8)
    table[:, 1] = (g * 255.0 + 0.5).astype(np.uint8)
    table[:, 2] = (b * 255.0 + 0.5).astype(np.uint8)
    table[:, 3] = 255
    return table


def _normalize_robust(hm: np.ndarray, pct: float = 2.0) -> np.ndarray:
    h = np.asarray(hm, dtype=np.float32)
    low = float(np.percentile(h, max(0.0, min(50.0, pct)))) if pct > 0 else float(h.min())
    high = float(np.percentile(h, max(0.0, min(100.0, 100.0 - pct)))) if pct > 0 else float(h.max())
    if high <= low:
        return np.zeros_like(h, dtype=np.float32)
    h = np.clip(h, low, high)
    return (h - low) / (high - low)


def _equalize01(a: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
    hist, bin_edges = np.histogram(x, bins=256, range=(0.0, 1.0), density=False)
    cdf = np.cumsum(hist).astype(np.float32)
    if cdf[-1] <= 0:
        return x
    cdf /= cdf[-1]
    idx = np.minimum(np.searchsorted(bin_edges[1:], x, side='right'), 255)
    y = cdf[idx]
    return y.astype(np.float32)


def load_dem(src_path: Path) -> tuple[np.ndarray, tuple[float, float]]:
    try:
        import rasterio
    except Exception as exc:  # pragma: no cover
        raise ImportError("rasterio is required. Install with: pip install rasterio") from exc

    with rasterio.open(str(src_path)) as ds:
        band1 = ds.read(1, masked=True)
        data = np.array(band1.filled(np.nan), dtype=np.float32)
        if np.isnan(data).any():
            finite = data[np.isfinite(data)]
            fill_val = float(np.min(finite)) if finite.size else 0.0
            data = np.nan_to_num(data, nan=fill_val)
        try:
            sx = float(ds.transform.a)
            sy = float(-ds.transform.e)
        except Exception:
            sx, sy = 1.0, 1.0
        sx = abs(sx) or 1.0
        sy = abs(sy) or 1.0
        return data, (sx, sy)


def heightmap_to_mesh(hm: np.ndarray, spacing: tuple[float, float], z_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a heightmap grid to positions (N,3) and indices (M,3).

    The grid is laid out so that X increases with column index and Y increases with row index,
    using the provided pixel spacing. Z is elevation * z_scale.
    """
    H, W = hm.shape
    sx, sy = float(spacing[0]), float(spacing[1])
    # Vertex positions
    xs = np.arange(W, dtype=np.float32) * sx
    ys = np.arange(H, dtype=np.float32) * sy
    X, Y = np.meshgrid(xs, ys)
    Z = hm.astype(np.float32) * float(z_scale)
    V = np.stack([X, Z, Y], axis=-1).reshape(-1, 3).astype(np.float32)  # use X,Z,Y to keep Z up

    # Triangle indices: two per cell
    i = np.arange(H * W, dtype=np.uint32).reshape(H, W)
    i00 = i[:-1, :-1]
    i10 = i[1:, :-1]
    i01 = i[:-1, 1:]
    i11 = i[1:, 1:]
    # Triangles: (i00, i10, i11) and (i00, i11, i01)
    t0 = np.stack([i00, i10, i11], axis=-1).reshape(-1, 3)
    t1 = np.stack([i00, i11, i01], axis=-1).reshape(-1, 3)
    F = np.concatenate([t0, t1], axis=0).astype(np.uint32)
    return V, F


def palette_rgb_for_heightmap(hm: np.ndarray, equalize: bool = True, gamma: float = 1.1) -> np.ndarray:
    norm01 = _normalize_robust(hm, pct=1.0)
    if np.isfinite(gamma) and abs(gamma - 1.0) > 1e-3:
        norm01 = np.clip(norm01, 0.0, 1.0) ** (1.0 / gamma)
    if equalize:
        norm01 = _equalize01(norm01)
    table = _interpolate_palette_rgba(CUSTOM_HEX_COLORS, n=128)
    idx = np.clip((norm01 * (table.shape[0] - 1)).astype(np.int32), 0, table.shape[0] - 1)
    rgb = table[idx, :3].astype(np.float32) / 255.0
    return rgb


def main() -> int:
    p = argparse.ArgumentParser(description="Ray trace a terrain mesh built from a DEM")
    p.add_argument("--src", type=Path, default=Path("assets/Gore_Range_Albers_1m.tif"), help="Input GeoTIFF DEM")
    p.add_argument("--out", type=Path, default=Path("reports/m4_raytraced_dem.png"), help="Output PNG path")
    p.add_argument("--preview-size", type=int, nargs=2, default=(1200, 900), metavar=("W","H"), help="Render size (pixels)")
    p.add_argument("--z-scale", type=float, default=1.0, help="Vertical scale for terrain mesh (units match DEM)")
    p.add_argument("--frames", type=int, default=8, help="Accumulation frames for the path tracer")
    p.add_argument("--seed", type=int, default=7, help="Random seed")
    p.add_argument("--denoiser", type=str, choices=["off","svgf"], default="off", help="Denoiser for beauty")
    p.add_argument("--svgf-iters", type=int, default=5, help="SVGF iterations if enabled")
    p.add_argument("--luminance-clamp", dest="lum_clamp", type=float, default=None, help="Optional luminance clamp to suppress fireflies")
    p.add_argument("--preview", action="store_true", help="Overlay palette-colored terrain point cloud when GPU tracer is unavailable")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU fallback even if native GPU tracer is available")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load DEM
    hm, spacing = load_dem(args.src)

    # 2) Build mesh and upload
    V, F = heightmap_to_mesh(hm, spacing, z_scale=float(args.z_scale))
    validate_mesh_arrays(np.ascontiguousarray(V, dtype=np.float32), np.ascontiguousarray(F.astype(np.uint32)))
    mesh = make_mesh(np.ascontiguousarray(V, dtype=np.float32), np.ascontiguousarray(F.astype(np.uint32)))
    bvh = build_bvh_cpu(mesh, method="median")
    handle = upload_mesh(mesh, bvh)

    # 3) Camera setup: frame terrain from a nice oblique angle
    W, H = int(args.preview_size[0]), int(args.preview_size[1])
    aspect = float(W) / float(H)
    # Terrain AABB for framing
    minv = V.min(axis=0); maxv = V.max(axis=0)
    center = 0.5 * (minv + maxv)
    size = (maxv - minv)
    # Eye direction and distance heuristic
    dir_hint = np.array([0.9, 0.6, 1.2], dtype=np.float32)
    dir_hint /= (np.linalg.norm(dir_hint) + 1e-8)
    fovy = 35.0
    radius = float(np.linalg.norm(size)) * 0.6
    eye = center + dir_hint * radius
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    cam = make_camera(
        origin=tuple(map(float, eye.tolist())),
        look_at=tuple(map(float, center.tolist())),
        up=tuple(map(float, up.tolist())),
        fov_y=float(fovy),
        aspect=float(aspect),
        exposure=1.0,
    )

    # 4) Render via path tracer (GPU if available)
    # Prefer native probe (forge3d._forge3d.device_probe) if extension is built; fallback to Python shim
    prefer_gpu = not bool(args.no_gpu)
    adapters_count = 0
    probe_status = 'skipped'
    use_gpu = False
    if prefer_gpu:
        try:
            from forge3d import _forge3d as _native  # type: ignore
            try:
                probe_dict = _native.device_probe("metal")  # type: ignore[attr-defined]
                probe_status = str(probe_dict.get("status", "unknown"))
                use_gpu = (probe_status == "ok")
            except Exception:
                probe_status = 'error'
        except Exception:
            # Native module not available; rely on conservative Python shims
            try:
                adapters = f3d.enumerate_adapters()
                adapters_count = len(adapters)
                use_gpu = adapters_count > 0
                probe_status = getattr(f3d, 'device_probe', lambda: {'status': 'n/a'})().get('status', 'n/a')  # type: ignore
            except Exception:
                probe_status = 'error'
    else:
        probe_status = 'disabled'
    print(f"[m4_raytrace_dem] prefer_gpu={prefer_gpu} adapters={adapters_count} use_gpu={use_gpu} probe={probe_status}")
    # Prefer native GPU mesh tracer if available (HybridPathTracer with mesh)
    img = None
    if use_gpu:
        try:
            from forge3d import _forge3d as _native  # type: ignore
            V32 = np.ascontiguousarray(V, dtype=np.float32)
            F32 = np.ascontiguousarray(F.astype(np.uint32))
            img = _native._pt_render_gpu_mesh(int(W), int(H), V32, F32, cam, int(args.seed), int(max(1, args.frames)))
            print("[m4_raytrace_dem] used_native_gpu_mesh=True")
        except Exception as e:
            print(f"[m4_raytrace_dem] native GPU mesh render unavailable ({e!r}); trying sphere fallback")
            try:
                # Fallback to legacy GPU tracer (spheres + ground) to at least validate GPU path
                scene = []
                img = _native._pt_render_gpu(int(W), int(H), scene, cam, int(args.seed), int(max(1, args.frames)))
                print("[m4_raytrace_dem] used_native_gpu_legacy=True")
            except Exception as e2:
                print(f"[m4_raytrace_dem] legacy GPU render also unavailable ({e2!r}); falling back to CPU")

    if img is None:
        tracer = PathTracer()
        img = tracer.render_rgba(
            W, H,
            scene=[],
            camera=cam,
            seed=int(args.seed),
            frames=max(1, int(args.frames)),
            use_gpu=False,
            mesh=handle,
            denoiser=str(args.denoiser),
            svgf_iters=int(args.svgf_iters),
            luminance_clamp=(float(args.lum_clamp) if args.lum_clamp is not None else None),
        )

    # 5) Optional CPU preview overlay: palette-colored terrain points (only informative)
    if args.preview or not use_gpu:
        # Sample a subset of vertices for speed if very large
        pts = V
        step = max(1, int(round(max(1, V.shape[0] // (W * H // 6)))))  # rough thinning for huge meshes
        pts = pts[::step]
        # Color by palette
        rgb_raster = palette_rgb_for_heightmap(hm)
        # Use nearest mapping from vertices to heightmap grid indices
        Hm, Wm = hm.shape
        # Approximate: back-project XY to grid
        sx, sy = spacing
        xs = np.clip((pts[:, 0] / max(sx, 1e-8)).round().astype(int), 0, Wm - 1)
        ys = np.clip((pts[:, 2] / max(sy, 1e-8)).round().astype(int), 0, Hm - 1)
        colors = (rgb_raster[ys, xs] * 255.0 + 0.5).astype(np.uint8)

        # Project to screen using simple look-at/perspective (inline to avoid deps)
        def _normalize(v: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
            return v / n
        def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
            f = _normalize(target - eye)
            s = _normalize(np.cross(f, up))
            u = np.cross(s, f)
            M = np.eye(4, dtype=np.float32)
            M[0, :3] = s; M[1, :3] = u; M[2, :3] = -f
            M[0, 3] = -np.dot(s, eye); M[1, 3] = -np.dot(u, eye); M[2, 3] = np.dot(f, eye)
            return M
        def _perspective(fovy_deg: float, aspect: float, znear: float = 0.05, zfar: float = 1000.0) -> np.ndarray:
            f = 1.0 / np.tan(np.radians(float(fovy_deg)) * 0.5)
            a = float(aspect)
            M = np.zeros((4, 4), dtype=np.float32)
            M[0, 0] = f / max(a, 1e-8); M[1, 1] = f
            M[2, 2] = (zfar + znear) / (znear - zfar); M[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
            M[3, 2] = -1.0
            return M
        def _project_points(V: np.ndarray, Mview: np.ndarray, Mproj: np.ndarray, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
            Vh = np.concatenate([V.astype(np.float32), np.ones((V.shape[0], 1), dtype=np.float32)], axis=1)
            clip = (Mproj @ (Mview @ Vh.T)).T
            w = clip[:, 3:4]
            valid = np.abs(w[:, 0]) > 1e-6
            ndc = np.zeros((V.shape[0], 3), dtype=np.float32)
            if np.any(valid):
                ndc[valid] = clip[valid, :3] / w[valid]
            vis = valid & np.all(np.isfinite(ndc), axis=1) & (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
            xs2 = np.clip((ndc[:, 0] * 0.5 + 0.5) * (W - 1), 0, W - 1).astype(np.int32)
            ys2 = np.clip((1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (H - 1), 0, H - 1).astype(np.int32)
            return np.stack([xs2, ys2], axis=1), vis
        def _draw_points(img: np.ndarray, pts: np.ndarray, vis: np.ndarray, colors: np.ndarray, size: int = 1):
            h, w = img.shape[:2]
            r = max(1, int(size))
            for (x, y), ok, c in zip(pts, vis, colors):
                if not ok: continue
                x0 = max(0, x - r); x1 = min(w - 1, x + r)
                y0 = max(0, y - r); y1 = min(h - 1, y + r)
                img[y0:y1 + 1, x0:x1 + 1, 0] = int(c[0])
                img[y0:y1 + 1, x0:x1 + 1, 1] = int(c[1])
                img[y0:y1 + 1, x0:x1 + 1, 2] = int(c[2])
                img[y0:y1 + 1, x0:x1 + 1, 3] = 255

        # Build transforms and draw
        Mview = _look_at(eye, center, up)
        Mproj = _perspective(fovy, aspect)
        s2d, vis = _project_points(pts, Mview, Mproj, W, H)
        _draw_points(img, s2d, vis, colors, size=1)

    # 6) Save
    try:
        f3d.numpy_to_png(str(args.out), img)
        print(f"Wrote {args.out}")
    except Exception:
        try:
            from PIL import Image  # type: ignore
            Image.fromarray(img, mode="RGBA").save(str(args.out))
            print(f"Wrote {args.out}")
        except Exception as exc:
            print(f"Render/save failed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
