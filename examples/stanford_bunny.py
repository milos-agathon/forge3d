#!/usr/bin/env python3
"""
examples/stanford_bunny.py

Render the Stanford Bunny with simple Lambert shading and a blue background
using only Python + NumPy and the Forge3D helpers for I/O. This example does
not require GPU. It demonstrates how to:

- Load a Wavefront OBJ mesh (minimal parser for v/f records)
- Compute per-vertex normals (area-weighted)
- Project to screen with a perspective camera
- Software-rasterize triangles with a Z-buffer and barycentric interpolation
- Save the resulting RGBA image with forge3d.numpy_to_png

The script uses the bunny OBJ provided in this repository at:
  assets/bunny.obj
No network is required.

Output:
  out/stanford_bunny.png (512x512)

Note:
- This is a compact educational rasterizer (no perspective-correct attributes,
  no backface culling, and no clipping), sufficient to reproduce the classic
  shaded bunny on a blue background like the attached reference.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import argparse
from typing import Tuple, List

import numpy as np

try:
    import forge3d as f3d  # for numpy_to_png
except Exception:  # pragma: no cover
    f3d = None


# ----------------------------- OBJ Utilities -----------------------------

def _try_download(url: str, dst: Path, attempts: int = 3) -> bool:
    import urllib.request
    import time as _t
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for _ in range(max(1, int(attempts))):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/121.0 Safari/537.36"
                },
            )
            with urllib.request.urlopen(req, timeout=30) as r:  # nosec B310
                data = r.read()
            if not data or len(data) < 128:  # sanity check
                last_err = RuntimeError("download returned too little data")
                _t.sleep(0.5)
                continue
            dst.write_bytes(data)
            return True
        except Exception as e:  # pragma: no cover
            last_err = e
            _t.sleep(0.75)
    return False


def ensure_bunny_obj() -> Path:
    """Return the path to the bunny OBJ within this repo (assets/bunny.obj)."""
    repo_root = Path(__file__).resolve().parents[1]
    assets_dir = repo_root / "assets"
    local_path = assets_dir / "bunny.obj"
    if not local_path.exists():
        raise FileNotFoundError(
            f"assets/bunny.obj not found at {local_path}. Please place the OBJ there."
        )
    return local_path


def load_obj_vertices_indices(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Minimal OBJ loader for 'v' and 'f' lines; returns (V, F).

    - Vertices: (N,3) float32
    - Faces: (M,3) uint32 (triangulated fan for polygons/quads)
    """
    vs: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            t = line.strip().split()
            if not t:
                continue
            if t[0] == "v" and len(t) >= 4:
                vs.append((float(t[1]), float(t[2]), float(t[3])))
            elif t[0] == "f" and len(t) >= 4:
                # face entries like: v, v/t, v//n, v/t/n
                idx = []
                for w in t[1:]:
                    s = w.split("/")
                    try:
                        i = int(s[0])
                    except Exception:
                        continue
                    # OBJ is 1-based; allow negative indexing but clamp later
                    idx.append(i)
                # Triangulate a polygon fan: (0,i,i+1)
                if len(idx) >= 3:
                    base = idx[0]
                    for k in range(1, len(idx) - 1):
                        faces.append((base, idx[k], idx[k + 1]))

    if not vs or not faces:
        raise ValueError(f"OBJ parse failed or empty: {path}")

    V = np.array(vs, dtype=np.float32)
    # Convert indices to 0-based and clamp to valid range
    F = np.array(faces, dtype=np.int64)
    F[F > 0] -= 1
    F[F < 0] += len(V)
    F = np.clip(F, 0, len(V) - 1).astype(np.uint32)
    return V, F


# ------------------------------- Math Utils -------------------------------

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-8)
    u = up / (np.linalg.norm(up) + 1e-8)
    s = np.cross(f, u)
    s = s / (np.linalg.norm(s) + 1e-8)
    u2 = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u2
    M[2, :3] = -f
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    return M @ T


def perspective(fovy_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (zfar + znear) / (znear - zfar)
    P[2, 3] = (2 * zfar * znear) / (znear - zfar)
    P[3, 2] = -1.0
    return P


def project_vertices(V: np.ndarray, Mview: np.ndarray, Mproj: np.ndarray, W: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
    """Transform world vertices to screen coordinates and per-vertex depth.

    Returns (screen_xy, depth) where:
      - screen_xy: (N,2) float32 in pixel space
      - depth: (N,) float32 in 0..1 (after perspective divide)
    """
    N = V.shape[0]
    Vh = np.concatenate([V, np.ones((N, 1), dtype=np.float32)], axis=1)
    clip = (Mproj @ (Mview @ Vh.T)).T  # (N,4)
    w = np.clip(clip[:, 3], 1e-8, None)
    ndc = clip[:, :3] / w[:, None]
    x = (ndc[:, 0] * 0.5 + 0.5) * (W - 1)
    y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (H - 1)  # flip Y for image coords
    z = ndc[:, 2] * 0.5 + 0.5  # map [-1,1] to [0,1]
    return np.stack([x, y], axis=1).astype(np.float32), z.astype(np.float32)


def compute_vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    n = np.zeros_like(V, dtype=np.float32)
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # area-weighted
    for i in range(3):
        np.add.at(n, F[:, i], fn)
    l = np.linalg.norm(n, axis=1, keepdims=True)
    l = np.maximum(l, 1e-8)
    return (n / l).astype(np.float32)


# ---------------------------- Rasterizer Core -----------------------------

def rasterize(
    V: np.ndarray,
    F: np.ndarray,
    N: np.ndarray,
    W: int,
    H: int,
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    light_dir_world: np.ndarray,
    fovy: float = 35.0,
    znear: float = 0.1,
    zfar: float = 10.0,
    base_color: Tuple[int, int, int] = (245, 245, 245),
    bg_color: Tuple[int, int, int] = (45, 80, 210),
    ambient: float = 0.25,
    diffuse_scale: float = 0.85,
    enable_specular: bool = False,
    specular_intensity: float = 0.35,
    specular_power: int = 32,
    gamma: float = 2.2,
) -> np.ndarray:
    Mview = look_at(eye, target, up)
    Mproj = perspective(fovy, float(W) / float(H), znear, zfar)

    xy, z = project_vertices(V, Mview, Mproj, W, H)

    # Precompute vertex normals in view space for lighting
    # Transform normals by inverse-transpose of view (ignore non-uniform scale here)
    R = Mview[:3, :3]
    N_view = (R @ N.T).T
    N_view = N_view / (np.linalg.norm(N_view, axis=1, keepdims=True) + 1e-8)

    # Light direction in view space
    L_view = (R @ light_dir_world.astype(np.float32))
    L_view = L_view / (np.linalg.norm(L_view) + 1e-8)
    # View direction in view space (camera at origin looking -Z), approximate as +Z
    V_view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    V_view = V_view / (np.linalg.norm(V_view) + 1e-8)
    # Blinn-Phong half-vector (constant per frame with dir light + fixed view dir)
    H_blk = (L_view + V_view)
    H_blk = H_blk / (np.linalg.norm(H_blk) + 1e-8)

    img = np.zeros((H, W, 4), dtype=np.uint8)
    img[..., :3] = np.array(bg_color, dtype=np.uint8)[None, None, :]
    img[..., 3] = 255
    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    # Triangle rasterization per-face with vectorized bbox sampling
    X = xy[:, 0]; Y = xy[:, 1]
    for tri in F:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        x0, y0 = X[i0], Y[i0]
        x1, y1 = X[i1], Y[i1]
        x2, y2 = X[i2], Y[i2]
        # Bounding box (clamped)
        xmin = max(int(math.floor(min(x0, x1, x2))), 0)
        xmax = min(int(math.ceil(max(x0, x1, x2))), W - 1)
        ymin = max(int(math.floor(min(y0, y1, y2))), 0)
        ymax = min(int(math.ceil(max(y0, y1, y2))), H - 1)
        if xmin > xmax or ymin > ymax:
            continue
        # Edge function helpers
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) <= 1e-8:
            continue
        # Pixel grids
        xs = np.arange(xmin, xmax + 1, dtype=np.float32)
        ys = np.arange(ymin, ymax + 1, dtype=np.float32)
        XX, YY = np.meshgrid(xs, ys)
        # Barycentric coordinates
        w0 = ((x1 - x0) * (YY - y0) - (y1 - y0) * (XX - x0)) / area
        w1 = ((x2 - x1) * (YY - y1) - (y2 - y1) * (XX - x1)) / area
        w2 = 1.0 - w0 - w1
        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not np.any(mask):
            continue
        # Interpolate depth
        z_interp = (w0 * z[i0] + w1 * z[i1] + w2 * z[i2])
        # Z test
        zb_patch = zbuf[ymin:ymax + 1, xmin:xmax + 1]
        nearer = mask & (z_interp < zb_patch)
        if not np.any(nearer):
            continue
        # Interpolate normals (view space)
        n_interp = (
            w0[..., None] * N_view[i0][None, None, :] +
            w1[..., None] * N_view[i1][None, None, :] +
            w2[..., None] * N_view[i2][None, None, :]
        )
        n_len = np.linalg.norm(n_interp, axis=2, keepdims=True)
        n_len = np.maximum(n_len, 1e-8)
        n_unit = n_interp / n_len
        # Lambert term
        lambert = np.clip(np.sum(n_unit * L_view[None, None, :], axis=2), 0.0, 1.0)
        # Simple ambient + diffuse
        intensity = np.clip(ambient + diffuse_scale * lambert, 0.0, 1.0)
        base = np.array(base_color, dtype=np.float32) / 255.0
        rgb_patch = (intensity[..., None] * base[None, None, :])
        # Optional Blinn-Phong specular highlight (white tint)
        if enable_specular:
            ndoth = np.clip(np.sum(n_unit * H_blk[None, None, :], axis=2), 0.0, 1.0)
            spec = np.power(ndoth, float(max(1, int(specular_power)))).astype(np.float32)
            rgb_patch = np.clip(rgb_patch + specular_intensity * spec[..., None], 0.0, 1.0)
        # Gamma correction
        if gamma and gamma > 0:
            inv_g = 1.0 / float(gamma)
            rgb_patch = np.power(np.clip(rgb_patch, 0.0, 1.0), inv_g)
        rgb_u8 = (np.clip(rgb_patch, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        # Write where nearer
        zb_patch[nearer] = z_interp[nearer]
        img_patch = img[ymin:ymax + 1, xmin:xmax + 1, :3]
        img_patch[nearer] = rgb_u8[nearer]

    return img


# --------------------------------- Main ----------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Render Stanford Bunny (software rasterizer)")
    parser.add_argument("--exact-match", action="store_true", help="Use camera/light/colors tuned to match reference")
    parser.add_argument("--no-specular", action="store_true", help="Disable Blinn-Phong specular highlight")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--outfile", type=str, default="out/stanford_bunny.png")
    # Camera controls
    parser.add_argument("--fovy", type=float, default=None, help="Vertical field of view in degrees (overrides default)")
    parser.add_argument("--theta", type=float, default=None, help="Yaw angle in degrees around +Y (0=+Z, 90=+X) for orbital camera")
    parser.add_argument("--phi", type=float, default=None, help="Pitch angle in degrees (-90..90). 0=horizon, positive=up")
    parser.add_argument("--radius", type=float, default=None, help="Camera distance from target (dolly). If omitted, auto-framing computes it")
    parser.add_argument("--zoom", type=float, default=None, help="Scale auto-framed distance; 1.0 default, <1.0 closer, >1.0 farther. Ignored if --radius is set")
    parser.add_argument("--target", type=float, nargs=3, metavar=("X","Y","Z"), default=None, help="Look-at target position (x y z)")
    parser.add_argument("--eye", type=float, nargs=3, metavar=("X","Y","Z"), default=None, help="Camera eye position (x y z). Overrides spherical/orbit settings")
    parser.add_argument("--up", type=float, nargs=3, metavar=("X","Y","Z"), default=None, help="Up vector (x y z)")
    args = parser.parse_args()
    out_path = Path(args.outfile)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_path = ensure_bunny_obj()
    V, F = load_obj_vertices_indices(obj_path)
    # Normalize bunny to fit into [-0.5,0.5]^3
    minv = V.min(axis=0)
    maxv = V.max(axis=0)
    center = 0.5 * (minv + maxv)
    scale = 1.0 / (max(maxv - minv) + 1e-8)
    Vn = (V - center) * (1.6 * scale)  # scale up a bit after normalization

    N = compute_vertex_normals(Vn, F)

    # Camera and light matching the reference look (tuned under --exact-match)
    W, H = int(args.width), int(args.height)
    # Helper: auto-frame camera to fit object given a view direction hint
    def _auto_frame(target: np.ndarray, fovy_deg: float, dir_hint: np.ndarray) -> tuple[np.ndarray, float, float]:
        r = float(np.max(np.linalg.norm(Vn - target[None, :], axis=1)))
        margin = 1.15  # pad a bit to avoid cropping
        d = r * margin / max(1e-6, np.tan(np.radians(fovy_deg) * 0.5))
        v = np.asarray(dir_hint, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        eye = target + v * d
        znear = max(0.01, d - r * 2.0)
        zfar = d + r * 2.0
        return eye.astype(np.float32), float(znear), float(zfar)

    if args.exact_match:
        target = np.array([0.0, 0.05, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        light_dir_world = np.array([-0.28, 0.62, 1.02], dtype=np.float32)
        base_color = (242, 242, 242)
        bg_color = (43, 80, 210)
        ambient = 0.22
        diffuse_scale = 0.86
        fovy_base = 34.0
        specular_intensity = 0.32
        specular_power = 48
        dir_hint_default = np.array([0.82, 0.62, 1.58], dtype=np.float32) - target
    else:
        target = np.array([0.0, 0.05, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        light_dir_world = np.array([-0.3, 0.6, 1.0], dtype=np.float32)
        base_color = (245, 245, 245)
        bg_color = (44, 78, 210)
        ambient = 0.25
        diffuse_scale = 0.85
        fovy_base = 34.0
        specular_intensity = 0.35
        specular_power = 32
        dir_hint_default = np.array([0.8, 0.6, 1.6], dtype=np.float32) - target

    # Apply CLI overrides and compute camera
    if args.target is not None:
        target = np.array(args.target, dtype=np.float32)
    if args.up is not None:
        up = np.array(args.up, dtype=np.float32)
    # FOVY
    fovy = float(args.fovy) if args.fovy is not None else float(fovy_base)
    # Bounding radius around target
    r = float(np.max(np.linalg.norm(Vn - target[None, :], axis=1)))
    # View direction
    if (args.theta is not None) or (args.phi is not None):
        theta = math.radians(args.theta if args.theta is not None else 45.0)
        phi = math.radians(args.phi if args.phi is not None else 30.0)
        dirv = np.array([
            math.cos(phi) * math.sin(theta),
            math.sin(phi),
            math.cos(phi) * math.cos(theta),
        ], dtype=np.float32)
    else:
        dirv = dir_hint_default.astype(np.float32)
        dirv = dirv / (np.linalg.norm(dirv) + 1e-8)

    if args.eye is not None:
        eye = np.array(args.eye, dtype=np.float32)
        d = float(np.linalg.norm(eye - target))
        if d < 1e-6:
            d = r * 3.0
            eye = target + dirv * d
        znear = max(0.01, d - r * 2.0)
        zfar = d + r * 2.0
    else:
        if args.radius is not None:
            d = float(args.radius)
        else:
            margin = 1.15
            d = r * margin / max(1e-6, math.tan(math.radians(fovy) * 0.5))
        if args.zoom is not None:
            d = float(d) * float(args.zoom)
        eye = target + dirv * d
        znear = max(0.01, d - r * 2.0)
        zfar = d + r * 2.0

    # Optional: validate mesh via forge3d.mesh utilities
    try:
        import forge3d.mesh as fmesh  # type: ignore
        fmesh.validate_mesh_arrays(np.ascontiguousarray(Vn, dtype=np.float32), np.ascontiguousarray(F.astype(np.uint32)))
    except Exception:
        pass

    img = rasterize(
        Vn, F, N, W, H,
        eye, target, up,
        light_dir_world,
        fovy=fovy,
        znear=znear,
        zfar=zfar,
        base_color=base_color,
        bg_color=bg_color,
        ambient=ambient,
        diffuse_scale=diffuse_scale,
        enable_specular=(not args.no_specular),
        specular_intensity=specular_intensity,
        specular_power=specular_power,
    )

    # Save
    if f3d is not None and hasattr(f3d, "numpy_to_png"):
        f3d.numpy_to_png(str(out_path), img)
    else:  # fallback to PIL
        try:
            from PIL import Image  # type: ignore
            Image.fromarray(img, mode="RGBA").save(str(out_path))
        except Exception:
            out_path_rgba = out_path.with_suffix(".rgba")
            out_path_rgba.write_bytes(img.tobytes())
            print(f"Saved raw RGBA to {out_path_rgba}")
            return 0

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
