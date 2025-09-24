#!/usr/bin/env python3
"""
examples/stanford_bunny_raytrace.py

Ray trace the Stanford Bunny using Forge3D's Python API only.
- Loads OBJ from repository asset: assets/bunny.obj (no network)
- Builds a mesh + BVH with forge3d.mesh
- Renders via forge3d.path_tracing.render_rgba (GPU if available, CPU fallback otherwise)
- Saves PNG to out/stanford_bunny_rt.png

Controls:
- --width/--height: image size
- --fovy: vertical field of view in degrees (default 34)
- --theta/--phi: orbital yaw/pitch around the target
- --radius or --zoom: distance control (zoom scales auto-framed distance)
- --eye/--target/--up: direct camera overrides
- --frames/--seed: accumulation frames and RNG seed
- --cpu: force CPU fallback even if a GPU adapter exists

This example focuses on API usage and may render a procedural image in CPU fallback
mode while GPU compute matures. It still demonstrates the full Forge3D workflow
for triangle meshes including BVH creation and camera setup.
"""
from __future__ import annotations

from pathlib import Path
import math
import argparse
from typing import Tuple, List

import numpy as np

import forge3d as f3d
from forge3d.mesh import make_mesh, build_bvh_cpu, upload_mesh, validate_mesh_arrays
from forge3d.path_tracing import PathTracer, make_camera, render_aovs, save_aovs


# ----------------------------- OBJ Utilities -----------------------------

def ensure_bunny_obj() -> Path:
    """Return the path to the bunny OBJ within this repo (assets/bunny.obj)."""
    repo_root = Path(__file__).resolve().parents[1]
    local_path = repo_root / "assets" / "bunny.obj"
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


# ----------------------------- Camera helpers ----------------------------

def auto_frame(V: np.ndarray, target: np.ndarray, fovy_deg: float, dir_hint: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute camera eye position and distance to frame V with margin.

    Returns (eye, distance). znear/zfar are not needed for path tracing camera.
    """
    r = float(np.max(np.linalg.norm(V - target[None, :], axis=1)))
    margin = 1.15
    d = r * margin / max(1e-6, math.tan(math.radians(fovy_deg) * 0.5))
    v = np.asarray(dir_hint, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    eye = target + v * d
    return eye.astype(np.float32), float(d)


# ----------------------- CPU preview (points overlay) ----------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
    return v / n


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = _normalize(target - eye)  # forward
    s = _normalize(np.cross(f, up))
    u = np.cross(s, f)
    # Right-handed view matrix (OpenGL-style)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[0, 3] = -np.dot(s, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] = np.dot(f, eye)
    return M


def _perspective(fovy_deg: float, aspect: float, znear: float = 0.05, zfar: float = 100.0) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(float(fovy_deg)) * 0.5)
    a = float(aspect)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / max(a, 1e-8)
    M[1, 1] = f
    M[2, 2] = (zfar + znear) / (znear - zfar)
    M[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
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
    xs = np.clip((ndc[:, 0] * 0.5 + 0.5) * (W - 1), 0, W - 1).astype(np.int32)
    ys = np.clip((1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (H - 1), 0, H - 1).astype(np.int32)
    return np.stack([xs, ys], axis=1), vis


def _draw_points(img: np.ndarray, pts: np.ndarray, vis: np.ndarray, size: int, color: tuple[int, int, int]):
    h, w = img.shape[:2]
    r = max(1, int(size))
    for (x, y), ok in zip(pts, vis):
        if not ok:
            continue
        x0 = max(0, x - r)
        x1 = min(w - 1, x + r)
        y0 = max(0, y - r)
        y1 = min(h - 1, y + r)
        img[y0:y1 + 1, x0:x1 + 1, 0] = color[0]
        img[y0:y1 + 1, x0:x1 + 1, 1] = color[1]
        img[y0:y1 + 1, x0:x1 + 1, 2] = color[2]
        img[y0:y1 + 1, x0:x1 + 1, 3] = 255

# --------------------------------- Main ----------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Ray trace Stanford Bunny with Forge3D")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--outfile", type=str, default="out/stanford_bunny_rt.png")
    # Camera controls
    p.add_argument("--fovy", type=float, default=34.0, help="Vertical field of view in degrees")
    p.add_argument("--theta", type=float, default=None, help="Yaw angle in degrees around +Y (0=+Z, 90=+X)")
    p.add_argument("--phi", type=float, default=None, help="Pitch angle in degrees (-90..90), 0=horizon, +=up")
    p.add_argument("--radius", type=float, default=None, help="Camera distance from target (overrides --zoom)")
    p.add_argument("--zoom", type=float, default=None, help="Scale auto-framed distance; 1.0=default, <1.0 closer, >1.0 farther")
    p.add_argument("--target", type=float, nargs=3, metavar=("X","Y","Z"), default=None, help="Look-at target position")
    p.add_argument("--eye", type=float, nargs=3, metavar=("X","Y","Z"), default=None, help="Camera eye position (overrides theta/phi/radius)")
    p.add_argument("--up", type=float, nargs=3, metavar=("X","Y","Z"), default=None, help="Up vector")
    # Rendering
    p.add_argument("--frames", type=int, default=8, help="Accumulation frames (>=1)")
    p.add_argument("--seed", type=int, default=7, help="Random seed")
    p.add_argument("--cpu", action="store_true", help="Force CPU fallback (do not use GPU)")
    # Preview overlay (CPU point cloud) to visualize mesh when GPU is unavailable
    p.add_argument("--preview", action="store_true", help="Draw CPU point-cloud overlay of the bunny onto the output")
    p.add_argument("--preview-size", type=int, default=1, help="Point size (pixels radius)")
    p.add_argument("--preview-color", type=int, nargs=3, metavar=("R","G","B"), default=(255, 160, 40), help="Point color")
    # Denoiser & firefly controls
    p.add_argument("--denoiser", type=str, choices=["off", "svgf"], default="off", help="Apply denoiser to beauty")
    p.add_argument("--svgf-iters", type=int, default=5, help="Iterations for SVGF denoiser")
    p.add_argument("--luminance-clamp", dest="lum_clamp", type=float, default=None, help="Optional luminance clamp to suppress fireflies")
    # AOV export
    p.add_argument("--save-aovs", action="store_true", help="Render and save AOVs (EXR where available, PNG for visibility)")
    p.add_argument("--aovs", type=str, default="albedo,normal,depth,visibility", help="Comma-separated AOV list to render")
    p.add_argument("--aov-dir", type=str, default=None, help="Directory for AOV outputs (defaults to output file directory)")
    p.add_argument("--basename", type=str, default=None, help="Basename for outputs (defaults to output filename stem)")
    args = p.parse_args()

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load bunny OBJ
    obj_path = ensure_bunny_obj()
    V, F = load_obj_vertices_indices(obj_path)

    # 2) Normalize geometry to a stable unit scale around origin for predictable camera
    minv = V.min(axis=0)
    maxv = V.max(axis=0)
    center = 0.5 * (minv + maxv)
    scale = 1.0 / (max(maxv - minv) + 1e-8)
    Vn = (V - center).astype(np.float32) * (1.6 * float(scale))

    # 3) Validate and build mesh/BVH
    validate_mesh_arrays(np.ascontiguousarray(Vn, dtype=np.float32), np.ascontiguousarray(F.astype(np.uint32)))
    mesh = make_mesh(np.ascontiguousarray(Vn, dtype=np.float32), np.ascontiguousarray(F.astype(np.uint32)))
    bvh = build_bvh_cpu(mesh, method="median")
    handle = upload_mesh(mesh, bvh)

    # 4) Camera setup (auto-framed by default)
    W, H = int(args.width), int(args.height)
    aspect = float(W) / float(H)
    fovy = float(args.fovy)

    target = np.array([0.0, 0.05, 0.0], dtype=np.float32) if args.target is None else np.array(args.target, dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32) if args.up is None else np.array(args.up, dtype=np.float32)

    # Direction hint from a nice 3/4 view
    dir_hint_default = np.array([0.8, 0.6, 1.6], dtype=np.float32) - target

    # Spherical orbit direction if provided
    if (args.theta is not None) or (args.phi is not None):
        theta = math.radians(args.theta if args.theta is not None else 45.0)
        phi = math.radians(args.phi if args.phi is not None else 25.0)
        dirv = np.array([
            math.cos(phi) * math.sin(theta),
            math.sin(phi),
            math.cos(phi) * math.cos(theta),
        ], dtype=np.float32)
        dirv = dirv / (np.linalg.norm(dirv) + 1e-8)
    else:
        dirv = dir_hint_default / (np.linalg.norm(dir_hint_default) + 1e-8)

    if args.eye is not None:
        eye = np.array(args.eye, dtype=np.float32)
    else:
        if args.radius is not None:
            d = float(args.radius)
            eye = target + dirv * d
        else:
            eye, d = auto_frame(Vn, target, fovy, dirv)
            if args.zoom is not None:
                eye = target + dirv * (float(d) * float(args.zoom))

    cam = make_camera(
        origin=tuple(map(float, eye.tolist())),
        look_at=tuple(map(float, target.tolist())),
        up=tuple(map(float, up.tolist())),
        fov_y=float(fovy),
        aspect=float(aspect),
        exposure=1.0,
    )

    # 5) Render
    use_gpu = False if args.cpu else (len(f3d.enumerate_adapters()) > 0)
    tracer = PathTracer()
    img = tracer.render_rgba(
        W, H,
        scene=[],
        camera=cam,
        seed=int(args.seed),
        frames=max(1, int(args.frames)),
        use_gpu=bool(use_gpu),
        mesh=handle,
        denoiser=str(args.denoiser),
        svgf_iters=int(args.svgf_iters),
        luminance_clamp=(float(args.lum_clamp) if args.lum_clamp is not None else None),
    )

    # Optional CPU preview overlay of mesh vertices (helps when GPU tracer is unavailable)
    if args.preview or not use_gpu:
        Mview = _look_at(eye, target, up)
        Mproj = _perspective(fovy, aspect)
        pts, vis = _project_points(Vn, Mview, Mproj, W, H)
        color = tuple(int(c) for c in args.preview_color)
        _draw_points(img, pts, vis, size=int(args.preview_size), color=color)

    # 6) Save
    try:
        f3d.numpy_to_png(str(out_path), img)
    except Exception:
        # PIL fallback
        try:
            from PIL import Image  # type: ignore
            Image.fromarray(img, mode="RGBA").save(str(out_path))
        except Exception:
            raw = out_path.with_suffix(".rgba")
            raw.write_bytes(img.tobytes())
            print(f"Saved raw RGBA to {raw}")
            print("Done (CPU fallback render).")
            return 0

    print(f"Saved: {out_path} (use_gpu={use_gpu}, tris={handle.triangle_count})")
    if not use_gpu:
        print("Note: GPU backend not in use; the beauty image is a procedural fallback. Use --preview to overlay bunny points.")

    # 7) Optional AOVs
    if bool(args.save_aovs):
        req = [s.strip().lower() for s in str(args.aovs).split(',') if s.strip()]
        # Some fallback implementations may not accept 'mesh' parameter
        try:
            aov_map = render_aovs(
                W, H, scene=[], camera=cam,
                aovs=tuple(req), seed=int(args.seed), frames=max(1, int(args.frames)), use_gpu=bool(use_gpu), mesh=handle
            )
        except TypeError:
            aov_map = render_aovs(
                W, H, scene=[], camera=cam,
                aovs=tuple(req), seed=int(args.seed), frames=max(1, int(args.frames)), use_gpu=bool(use_gpu)
            )
        base = str(args.basename) if args.basename else out_path.stem
        out_dir = args.aov_dir if args.aov_dir else str(out_path.parent)
        saved = save_aovs(aov_map, base, output_dir=out_dir)
        if saved:
            print("Saved AOVs:")
            for k, pth in saved.items():
                print(f"  {k}: {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
