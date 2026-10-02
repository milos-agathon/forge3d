"""Deterministic first-order transport for the frozen NEPHELE fixture.

Mirrors the reference conventions (centered bilinear DEM, texel-center trilinear
f16 density, Henyey-Greenstein on dot(view_dir, to_sun), uniform environment)
and evaluates, per pixel:
  * camera transmittance and terrain hit (validated against reference AOVs),
  * single scattering of sun and environment along the camera segment,
  * first-bounce terrain radiance (direct sun + occluded, attenuated sky).
Higher orders (multiple scattering, terrain interreflection, surface-then-medium
paths) are deliberately excluded; reference minus this sum bounds them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
FIX = ROOT / "tests/nephele/fixture"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("first-order")
OUT.mkdir(parents=True, exist_ok=True)


def load(name):
    return json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))


medium, terrain_cfg, sun_cfg, atmos, material = (
    load("medium"), load("terrain"), load("sun"), load("atmosphere"), load("material"))
prov = load("reference-provenance")
cam = prov["camera_contract"]

# Density: unorm16 / 65535 -> f32 -> f16 RNE, indexed [z, y, x].
nx, ny, nz = medium["domain"]["grid_shape"]
r16 = np.asarray(medium["density_r16"], dtype=np.float64)
grid = (r16 / 65535.0).astype(np.float32).astype(np.float16).astype(np.float64).reshape(nz, ny, nx)
bmin = np.asarray(medium["domain"]["bounds_min"], dtype=np.float64)
bmax = np.asarray(medium["domain"]["bounds_max"], dtype=np.float64)
sigma_s = np.asarray(medium["sigma_s"], dtype=np.float64) * medium["density_scale"]
sigma_a = np.asarray(medium["sigma_a"], dtype=np.float64) * medium["density_scale"]
sigma_t = sigma_s + sigma_a
g = float(medium["phase"]["g"])


def density(p):
    """Trilinear texel-center density; zero outside the closed bounds. p[..., 3]."""
    inside = np.all((p >= bmin) & (p <= bmax), axis=-1)
    unit = np.clip((p - bmin) / (bmax - bmin), 0.0, 1.0)
    dims = np.asarray([nx, ny, nz], dtype=np.float64)
    c = unit * dims - 0.5
    lo = np.floor(c)
    f = c - lo
    lo = lo.astype(np.int64)
    hi = lo + 1
    lo = np.clip(lo, 0, dims.astype(np.int64) - 1)
    hi = np.clip(hi, 0, dims.astype(np.int64) - 1)
    v = np.zeros(p.shape[:-1])
    for z in (0, 1):
        iz = hi[..., 2] if z else lo[..., 2]
        wz = f[..., 2] if z else 1.0 - f[..., 2]
        for y in (0, 1):
            iy = hi[..., 1] if y else lo[..., 1]
            wy = f[..., 1] if y else 1.0 - f[..., 1]
            for x in (0, 1):
                ix = hi[..., 0] if x else lo[..., 0]
                wx = f[..., 0] if x else 1.0 - f[..., 0]
                v = v + grid[iz, iy, ix] * wx * wy * wz
    return np.where(inside, v, 0.0)


# Terrain: centered DEM, bilinear patches, heights dem[z, x] * exaggeration.
dem = np.load(FIX / terrain_cfg["dem"]).astype(np.float64) * terrain_cfg["exaggeration"]
H, W = dem.shape
sx, sz = terrain_cfg["spacing"]
ox, oz = -0.5 * (W - 1) * sx, -0.5 * (H - 1) * sz


def height(x, z):
    """Bilinear DEM height; NaN outside the DEM footprint."""
    u = (x - ox) / sx
    v = (z - oz) / sz
    inside = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    cu = np.clip(np.floor(u), 0, W - 2).astype(np.int64)
    cv = np.clip(np.floor(v), 0, H - 2).astype(np.int64)
    fu = np.clip(u - cu, 0, 1)
    fv = np.clip(v - cv, 0, 1)
    h = ((dem[cv, cu] * (1 - fu) + dem[cv, cu + 1] * fu) * (1 - fv)
         + (dem[cv + 1, cu] * (1 - fu) + dem[cv + 1, cu + 1] * fu) * fv)
    return np.where(inside, h, np.nan)


def normal_at(x, z):
    u = (x - ox) / sx
    v = (z - oz) / sz
    cu = np.clip(np.floor(u), 0, W - 2).astype(np.int64)
    cv = np.clip(np.floor(v), 0, H - 2).astype(np.int64)
    fu = np.clip(u - cu, 0, 1)
    fv = np.clip(v - cv, 0, 1)
    dh_du = (dem[cv, cu + 1] - dem[cv, cu]) * (1 - fv) + (dem[cv + 1, cu + 1] - dem[cv + 1, cu]) * fv
    dh_dv = (dem[cv + 1, cu] - dem[cv, cu]) * (1 - fu) + (dem[cv + 1, cu + 1] - dem[cv, cu + 1]) * fu
    n = np.stack([-dh_du / sx, np.ones_like(dh_du), -dh_dv / sz], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def first_hit(origin, direction, tmax=400.0, step=0.02):
    """First crossing of the bilinear surface (march + bisection). origin/direction [..., 3]."""
    shape = direction.shape[:-1]
    t_hit = np.full(shape, np.inf)
    prev_t = np.zeros(shape)
    prev_d = np.full(shape, np.nan)
    active = np.ones(shape, dtype=bool)
    for t in np.arange(step, tmax + step, step):
        if not active.any():
            break
        p = origin + direction * t
        d = p[..., 1] - height(p[..., 0], p[..., 2])
        # Primary rays only intersect from above: a ray that enters the
        # footprint already below the surface meets no skirt (reference
        # traversal semantics), so require a finite positive previous height.
        crossed = active & (d <= 0) & np.isfinite(d) & np.isfinite(prev_d) & (prev_d > 0)
        if crossed.any():
            lo = np.where(np.isfinite(prev_d), prev_t, t - step)
            hi = np.full(shape, t)
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                pm = origin + direction * mid[..., None]
                dm = pm[..., 1] - height(pm[..., 0], pm[..., 2])
                below = (dm <= 0) | ~np.isfinite(dm)
                hi = np.where(crossed & below, mid, hi)
                lo = np.where(crossed & ~below, mid, lo)
            t_hit = np.where(crossed, hi, t_hit)
            active &= ~crossed
        prev_t = np.where(active, t, prev_t)
        prev_d = np.where(active, d, prev_d)
        # Rays that left the DEM footprint above it and keep rising cannot hit.
        gone = active & ~np.isfinite(d) & (p[..., 1] > dem.max()) & (direction[..., 1] >= 0)
        active &= ~gone
    return t_hit


def optical_depth(origin, direction, length, step):
    """Midpoint-rule integral of sigma_t*density along [0, length]; returns [..., 3]."""
    n = np.maximum(np.ceil(length / step).astype(np.int64), 1)
    nmax = int(n.max())
    tau = np.zeros(origin.shape[:-1] + (3,))
    for i in range(nmax):
        valid = i < n
        dt = length / n
        t = (i + 0.5) * dt
        p = origin + direction * t[..., None]
        rho = np.where(valid, density(p), 0.0)
        tau += (rho * dt)[..., None] * sigma_t
    return tau


def medium_exit(origin, direction):
    """Distance to leave the medium AABB (or 0 if never inside ahead)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / direction
        t0 = (bmin - origin) * inv
        t1 = (bmax - origin) * inv
    tfar = np.nanmin(np.maximum(t0, t1), axis=-1)
    return np.clip(np.nan_to_num(tfar, nan=0.0, posinf=0.0), 0.0, None)


def hg(cos):
    return (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * cos) ** 1.5)


HMAX = float(dem.max())
X_LO, X_HI = ox, ox + (W - 1) * sx
Z_LO, Z_HI = oz, oz + (H - 1) * sz


def occluded(origin, direction, maximum=None, step=0.25):
    """Terrain occlusion with an analytic march bound.

    A ray can only meet the surface while it is inside the DEM footprint and
    below the DEM maximum, so the march stops at the first of: leaving the
    footprint, rising above HMAX, or reaching `maximum`.
    """
    o = origin + direction * 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        t_up = np.where(direction[..., 1] > 0, (HMAX - o[..., 1]) / direction[..., 1], np.inf)
        tx = np.where(direction[..., 0] > 0, (X_HI - o[..., 0]) / direction[..., 0],
                      np.where(direction[..., 0] < 0, (X_LO - o[..., 0]) / direction[..., 0], np.inf))
        tz = np.where(direction[..., 2] > 0, (Z_HI - o[..., 2]) / direction[..., 2],
                      np.where(direction[..., 2] < 0, (Z_LO - o[..., 2]) / direction[..., 2], np.inf))
    limit = np.minimum(np.minimum(t_up, tx), tz)
    limit = np.where(np.isfinite(limit), np.clip(limit, 0.0, None), 0.0)
    if maximum is not None:
        limit = np.minimum(limit, maximum)
    result = np.zeros(direction.shape[:-1], dtype=bool)
    # Points already above the surface but outside the footprint never occlude.
    tmax = float(limit.max()) if limit.size else 0.0
    for t in np.arange(0.0, tmax + step, step):
        live = (t <= limit) & ~result
        if not live.any():
            break
        p = o + direction * t
        d = p[..., 1] - height(p[..., 0], p[..., 2])
        result |= live & np.isfinite(d) & (d < 0)
    return result


# Camera rays (reference_pixel_ndc with full 64x64 viewport, aspect 1).
size = 64
tan_half = np.tan(np.radians(cam["fov_y"]) / 2)
ys, xs = np.mgrid[0:size, 0:size]
ndc_x = (2 * (xs + 0.5) / size - 1) * tan_half
ndc_y = (1 - 2 * (ys + 0.5) / size) * tan_half
fwd, right, up = (np.asarray(cam[k], dtype=np.float64) for k in ("forward", "right", "up"))
dirs = fwd + right * ndc_x[..., None] + up * ndc_y[..., None]
dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
eye = np.broadcast_to(np.asarray(cam["origin"], dtype=np.float64), dirs.shape)

az, el = np.radians(sun_cfg["azimuth_deg"]), np.radians(sun_cfg["elevation_deg"])
to_sun = np.asarray([np.cos(az) * np.cos(el), np.sin(el), np.sin(az) * np.cos(el)])
sun_L = np.asarray(sun_cfg["color"], dtype=np.float64) * sun_cfg["intensity"]
env_L = float(atmos["environment_intensity"])
albedo = np.asarray(material["albedo"], dtype=np.float64)

print("terrain hits ...", flush=True)
t_hit = first_hit(eye, dirs)
hit = np.isfinite(t_hit)
reach = np.where(hit, t_hit, np.maximum(medium_exit(eye, dirs), 1e-6))
ref_hit = np.load(FIX / "reference-terrain-hit.npy")
print("hit mask agreement", float(np.mean(hit == ref_hit)), "mismatches", int((hit != ref_hit).sum()), flush=True)

print("camera transmittance ...", flush=True)
tau_cam = optical_depth(eye, dirs, reach, 0.05)
T_cam = np.exp(-tau_cam)
ref_T = np.load(FIX / "reference-transmittance.npy").astype(np.float64)
print("camera T: mean |dT|", float(np.abs(T_cam - ref_T).mean()), "max", float(np.abs(T_cam - ref_T).max()), flush=True)

# Terrain sun transmittance (cloud_shadow AOV: medium only, no terrain test).
p_hit = eye + dirs * np.where(hit, t_hit, 0.0)[..., None]
sun_dirs = np.broadcast_to(to_sun, dirs.shape)
sun_len = medium_exit(p_hit, sun_dirs)
T_sun_hit = np.exp(-optical_depth(p_hit, sun_dirs, sun_len, 0.1))
ref_cs = np.load(FIX / "reference-cloud-shadow-aov.npy").astype(np.float64)
print("cloud shadow: mean |d| on hits", float(np.abs(T_sun_hit - ref_cs)[hit].mean()), flush=True)

np.savez(OUT / "geometry.npz", hit=hit, t_hit=t_hit, reach=reach, T_cam=T_cam, T_sun_hit=T_sun_hit)
if len(sys.argv) > 4 and sys.argv[4] == "geometry-only":
    raise SystemExit(0)

# Sphere directions (Fibonacci) for environment terms.
def fibonacci(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], axis=-1)


SPHERE = fibonacci(int(sys.argv[3]) if len(sys.argv) > 3 else 96)
dw = 4 * np.pi / len(SPHERE)

print("surface first-bounce ...", flush=True)
n_hit = normal_at(p_hit[..., 0], p_hit[..., 2])
cos_sun = np.clip(np.sum(n_hit * to_sun, axis=-1), 0, None)
vis_sun = ~occluded(p_hit + n_hit * 1e-3, sun_dirs, np.full(hit.shape, 400.0))
surf_sun = albedo / np.pi * sun_L * (cos_sun * vis_sun)[..., None] * T_sun_hit
surf_env = np.zeros(dirs.shape)
for w in SPHERE:
    cw = np.sum(n_hit * w, axis=-1)
    use = hit & (cw > 0)
    if not use.any():
        continue
    wd = np.broadcast_to(w, dirs.shape)
    vis = ~occluded(p_hit + n_hit * 1e-3, wd, np.full(hit.shape, 400.0))
    Tw = np.exp(-optical_depth(p_hit, wd, medium_exit(p_hit, wd), 0.25))
    surf_env += (albedo / np.pi * env_L * dw) * ((cw * vis * use)[..., None] * Tw)
surface1 = np.where(hit[..., None], surf_sun + surf_env, 0.0)

print("single scatter along camera segments ...", flush=True)
STEPS = int(__import__("os").environ.get("FO_STEPS", "48"))
ss_sun = np.zeros(dirs.shape)
ss_env = np.zeros(dirs.shape)
tau_acc = np.zeros(dirs.shape)
dt = reach / STEPS
for i in range(STEPS):
    t = (i + 0.5) * dt
    p = eye + dirs * t[..., None]
    rho = density(p)
    tau_mid = tau_acc + (rho * dt * 0.5)[..., None] * sigma_t
    weight = np.exp(-tau_mid) * (rho * dt)[..., None] * sigma_s
    tau_acc += (rho * dt)[..., None] * sigma_t
    live = rho > 0
    if not live.any():
        continue
    # Sun.
    vis = ~occluded(p, sun_dirs, np.full(hit.shape, 400.0))
    Ts = np.exp(-optical_depth(p, sun_dirs, medium_exit(p, sun_dirs), 0.25))
    ph = hg(np.sum(dirs * to_sun, axis=-1))
    ss_sun += weight * (ph * vis * live)[..., None] * sun_L * Ts
    # Environment (sphere quadrature, terrain-occluded, medium-attenuated).
    acc = np.zeros(dirs.shape)
    for w in SPHERE:
        wd = np.broadcast_to(w, dirs.shape)
        visw = ~occluded(p, wd, np.full(hit.shape, 400.0))
        Tw = np.exp(-optical_depth(p, wd, medium_exit(p, wd), 0.5))
        acc += (hg(np.sum(dirs * w, axis=-1)) * visw * live)[..., None] * Tw
    ss_env += weight * acc * env_L * dw
    print(f"  step {i + 1}/{STEPS}", flush=True)

background = np.where(hit[..., None], 0.0, env_L)
first_order = T_cam * (surface1 + background) + ss_sun + ss_env
np.savez(OUT / "first-order.npz", T_cam=T_cam, surface1=surface1, surf_sun=surf_sun, surf_env=surf_env,
         ss_sun=ss_sun, ss_env=ss_env, background=background, first_order=first_order, hit=hit)
print("saved", OUT, flush=True)
