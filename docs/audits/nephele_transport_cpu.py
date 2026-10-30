"""CPU geometry/first-order transport for the NEPHELE audit, never a renderer.

The two Gauss nodes integrate each cubic ray restriction of trilinear density
exactly. Terrain roots use the bilinear leaf equation from
hybrid_terrain_traversal.wgsl; cell boundaries, not a march cap, bound the work.
Numba is an existing local analysis dependency, not a product dependency.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numba import njit, prange


@njit
def slab(o, d, lo, hi):
    begin, end = 0., np.inf
    for axis in range(3):
        if d[axis] == 0:
            if o[axis] < lo[axis] or o[axis] > hi[axis]:
                return 0., 0.
        else:
            a, b = (lo[axis] - o[axis]) / d[axis], (hi[axis] - o[axis]) / d[axis]
            begin, end = max(begin, min(a, b)), min(end, max(a, b))
    return begin, max(begin, end)


@njit
def density_one(p, grid, lo, hi):
    xyz = np.empty(3, np.int64)
    frac = np.empty(3)
    dims = np.array([grid.shape[2], grid.shape[1], grid.shape[0]])
    for axis in range(3):
        if p[axis] < lo[axis] or p[axis] > hi[axis]:
            return 0.
        c = (p[axis] - lo[axis]) / (hi[axis] - lo[axis]) * dims[axis] - .5
        xyz[axis] = int(np.floor(c))
        frac[axis] = c - xyz[axis]
    value = 0.
    for z in range(2):
        iz = min(max(xyz[2] + z, 0), dims[2] - 1)
        wz = frac[2] if z else 1 - frac[2]
        for y in range(2):
            iy = min(max(xyz[1] + y, 0), dims[1] - 1)
            wy = frac[1] if y else 1 - frac[1]
            for x in range(2):
                ix = min(max(xyz[0] + x, 0), dims[0] - 1)
                wx = frac[0] if x else 1 - frac[0]
                value += grid[iz, iy, ix] * wx * wy * wz
    return value


@njit
def density_integral(o, d, reach, grid, lo, hi):
    begin, end = slab(o, d, lo, hi)
    end = min(end, reach)
    if end <= begin:
        return 0.
    dims = np.array([grid.shape[2], grid.shape[1], grid.shape[0]])
    cuts = np.empty(int(dims.sum()) + 2)
    cuts[0], cuts[1], n = begin, end, 2
    for axis in range(3):
        if d[axis] != 0:
            for j in range(dims[axis]):
                t = (lo[axis] + (j + .5) * (hi[axis] - lo[axis]) / dims[axis] - o[axis]) / d[axis]
                if begin < t < end:
                    cuts[n], n = t, n + 1
    ordered = np.sort(cuts[:n])
    value = 0.
    for j in range(n - 1):
        mid, half = .5 * (ordered[j] + ordered[j + 1]), .5 * (ordered[j + 1] - ordered[j])
        offset = half / np.sqrt(3.)
        value += half * (density_one(o + d * (mid - offset), grid, lo, hi) + density_one(o + d * (mid + offset), grid, lo, hi))
    return value


@njit
def height_one(p, dem, spacing):
    h, w = dem.shape
    u, v = p[0] / spacing[0] + .5 * (w - 1), p[2] / spacing[1] + .5 * (h - 1)
    if u < 0 or u > w - 1 or v < 0 or v > h - 1:
        return np.nan
    x, z = min(int(np.floor(u)), w - 2), min(int(np.floor(v)), h - 2)
    a, b = u - x, v - z
    return (1 - b) * ((1 - a) * dem[z, x] + a * dem[z, x + 1]) + b * ((1 - a) * dem[z + 1, x] + a * dem[z + 1, x + 1])


@njit
def normal_one(p, dem, spacing):
    h, w = dem.shape
    u, v = p[0] / spacing[0] + .5 * (w - 1), p[2] / spacing[1] + .5 * (h - 1)
    x, z = min(max(int(np.floor(u)), 0), w - 2), min(max(int(np.floor(v)), 0), h - 2)
    a, b = min(max(u - x, 0.), 1.), min(max(v - z, 0.), 1.)
    dx = ((1 - b) * (dem[z, x + 1] - dem[z, x]) + b * (dem[z + 1, x + 1] - dem[z + 1, x])) / spacing[0]
    dz = ((1 - a) * (dem[z + 1, x] - dem[z, x]) + a * (dem[z + 1, x + 1] - dem[z, x + 1])) / spacing[1]
    n = np.array([-dx, 1., -dz])
    return n / np.sqrt((n * n).sum())


@njit
def terrain_hit(o, d, dem, spacing, tmin):
    h, w = dem.shape
    lo = np.array([-.5 * (w - 1) * spacing[0], dem.min(), -.5 * (h - 1) * spacing[1]])
    hi = np.array([-lo[0], dem.max(), -lo[2]])
    if hi[1] == lo[1]:
        if d[1] == 0:
            return np.inf
        t = (lo[1] - o[1]) / d[1]
        p = o + d * t
        return t if t > tmin and lo[0] <= p[0] <= hi[0] and lo[2] <= p[2] <= hi[2] else np.inf
    begin, end = slab(o, d, lo, hi)
    if end <= begin or end <= tmin:
        return np.inf
    cuts = np.empty(h + w + 2)
    cuts[0], cuts[1], n = begin, end, 2
    for axis, count, step in ((0, w, spacing[0]), (2, h, spacing[1])):
        if d[axis] != 0:
            for j in range(count):
                t = (lo[axis] + j * step - o[axis]) / d[axis]
                if begin < t < end:
                    cuts[n], n = t, n + 1
    ordered = np.sort(cuts[:n])
    for j in range(n - 1):
        left, right = ordered[j], ordered[j + 1]
        mid = .5 * (left + right)
        p0, pm, p1 = o + d * left, o + d * mid, o + d * right
        # Evaluate within this cell rather than changing cells at shared edges.
        u, v = pm[0] / spacing[0] + .5 * (w - 1), pm[2] / spacing[1] + .5 * (h - 1)
        cx, cz = min(max(int(np.floor(u)), 0), w - 2), min(max(int(np.floor(v)), 0), h - 2)
        diff = np.empty(3)
        for k in range(3):
            p = p0 if k == 0 else pm if k == 1 else p1
            a = min(max(p[0] / spacing[0] + .5 * (w - 1) - cx, 0.), 1.)
            b = min(max(p[2] / spacing[1] + .5 * (h - 1) - cz, 0.), 1.)
            height = (1 - b) * ((1 - a) * dem[cz, cx] + a * dem[cz, cx + 1]) + b * ((1 - a) * dem[cz + 1, cx] + a * dem[cz + 1, cx + 1])
            diff[k] = p[1] - height
        a = 2 * diff[2] + 2 * diff[0] - 4 * diff[1]
        b, c = diff[2] - diff[0] - a, diff[0]
        roots = np.full(2, np.inf)
        # These degeneracy constants are the production leaf's contract.
        if abs(a) < 1e-12:
            if abs(b) > 1e-12:
                roots[0] = -c / b
        else:
            disc = b * b - 4 * a * c
            if disc >= 0:
                q = -.5 * (b + (np.sqrt(disc) if b >= 0 else -np.sqrt(disc)))
                roots[0] = q / a
                if abs(q) >= 1e-30:
                    roots[1] = c / q
        roots.sort()
        for s in roots:
            t = left + s * (right - left)
            if 0 <= s <= 1 and t > tmin:
                return t
    return np.inf


@njit(parallel=True)
def evaluate(points, incident, grid, lo, hi, dem, spacing, sigma_s, sigma_t, sun, sun_l, env_l, g, sphere, reject_interior):
    """All 192 environment directions at every actual point; both visibilities.

    Geometric valid-air classification is identical with visibility on/off.
    Returns direct sun and direct environment only, separately. Higher orders
    are never silently filled with a camera-direction or nearest-froxel value.
    """
    n = len(points)
    solar, environment = np.zeros((2, n, 3)), np.zeros((2, n, 3))
    rho, valid = np.zeros(n), np.zeros(n, np.bool_)
    for i in prange(n):
        p = points[i]
        ground = height_one(p, dem, spacing)
        valid[i] = not np.isfinite(ground) or p[1] >= ground
        if reject_interior and not valid[i]:
            continue
        rho[i] = density_one(p, grid, lo, hi)
        if rho[i] == 0:
            continue
        tau = density_integral(p, sun, np.inf, grid, lo, hi)
        sun_visible = not np.isfinite(terrain_hit(p, sun, dem, spacing, 1e-3))
        cosine = (incident[i] * sun).sum()
        phase = (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * cosine) ** 1.5)
        for c in range(3):
            value = sigma_s[c] * rho[i] * sun_l[c] * np.exp(-sigma_t[c] * tau) * phase
            solar[0, i, c], solar[1, i, c] = value, value * sun_visible
        for direction in sphere:
            tau = density_integral(p, direction, np.inf, grid, lo, hi)
            visible = not np.isfinite(terrain_hit(p, direction, dem, spacing, 1e-3))
            cosine = (incident[i] * direction).sum()
            phase = (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * cosine) ** 1.5)
            for c in range(3):
                value = sigma_s[c] * rho[i] * env_l * (4 * np.pi / len(sphere)) * phase * np.exp(-sigma_t[c] * tau)
                environment[0, i, c] += value
                environment[1, i, c] += value * visible
    return solar, environment, rho, valid


@njit(parallel=True)
def surface_first(points, normals, grid, lo, hi, dem, spacing, sigma_t, sun, sun_l, env_l, albedo, sphere):
    solar, environment = np.zeros((2, len(points), 3)), np.zeros((2, len(points), 3))
    for i in prange(len(points)):
        # ReferenceScene queries start at the hit position; query tmin, below,
        # excludes self-contact. Do not add a second normal displacement.
        p = points[i]
        tau = density_integral(p, sun, np.inf, grid, lo, hi)
        visible = not np.isfinite(terrain_hit(p, sun, dem, spacing, 1e-3))
        cosine = max((normals[i] * sun).sum(), 0.)
        for c in range(3):
            value = albedo[c] / np.pi * sun_l[c] * cosine * np.exp(-sigma_t[c] * tau)
            solar[0, i, c], solar[1, i, c] = value, value * visible
        for direction in sphere:
            cosine = max((normals[i] * direction).sum(), 0.)
            if cosine == 0:
                continue
            tau = density_integral(p, direction, np.inf, grid, lo, hi)
            visible = not np.isfinite(terrain_hit(p, direction, dem, spacing, 1e-3))
            for c in range(3):
                value = albedo[c] / np.pi * env_l * cosine * (4 * np.pi / len(sphere)) * np.exp(-sigma_t[c] * tau)
                environment[0, i, c] += value
                environment[1, i, c] += value * visible
    return solar, environment


@njit(parallel=True)
def hits_and_normals(origins, directions, dem, spacing):
    t, normals = np.empty(len(origins)), np.zeros_like(origins)
    for i in prange(len(origins)):
        t[i] = terrain_hit(origins[i], directions[i], dem, spacing, 1e-3)
        if np.isfinite(t[i]):
            normals[i] = normal_one(origins[i] + directions[i] * t[i], dem, spacing)
    return t, normals


class Scene:
    def __init__(self, root):
        fixture = Path(root) / "tests/nephele/fixture"
        read = lambda name: json.loads((fixture / (name + ".json")).read_text())
        medium, terrain = read("medium"), read("terrain")
        self.lo, self.hi = np.array(medium["domain"]["bounds_min"], float), np.array(medium["domain"]["bounds_max"], float)
        dims = medium["domain"]["grid_shape"]
        self.grid = (np.array(medium["density_r16"], float) / 65535).astype(np.float32).astype(np.float16).astype(float).reshape(tuple(reversed(dims)))
        self.dem = np.load(fixture / terrain["dem"]).astype(float) * terrain["exaggeration"]
        self.spacing = np.array(terrain["spacing"], float)
        self.sigma_s = np.array(medium["sigma_s"]) * medium["density_scale"]
        self.sigma_t = self.sigma_s + np.array(medium["sigma_a"]) * medium["density_scale"]
        sun = read("sun")
        az, el = np.radians(sun["azimuth_deg"]), np.radians(sun["elevation_deg"])
        self.sun = np.array([np.cos(az) * np.cos(el), np.sin(el), np.sin(az) * np.cos(el)])
        self.sun_l = np.array(sun["color"]) * sun["intensity"]
        self.env_l, self.albedo, self.g = read("atmosphere")["environment_intensity"], np.array(read("material")["albedo"]), medium["phase"]["g"]
        # Existing production surface-lighting quadrature, not an invented count.
        i = np.arange(192) + .5
        y = 1 - 2 * i / len(i)
        theta = np.pi * (1 + np.sqrt(5)) * i
        self.sphere = np.stack([np.cos(theta) * np.sqrt(1 - y * y), y, np.sin(theta) * np.sqrt(1 - y * y)], -1)
        self.cam = read("reference-provenance")["camera_contract"]

    def source(self, points, incident, *, reject_interior=True):
        return evaluate(np.ascontiguousarray(points), np.ascontiguousarray(incident), self.grid, self.lo, self.hi, self.dem, self.spacing, self.sigma_s, self.sigma_t, self.sun, self.sun_l, self.env_l, self.g, self.sphere, reject_interior)

    def surface(self, points, normals):
        return surface_first(points, normals, self.grid, self.lo, self.hi, self.dem, self.spacing, self.sigma_t, self.sun, self.sun_l, self.env_l, self.albedo, self.sphere)

    def camera(self, shape):
        h, w = shape
        y, x = np.indices(shape)
        tan = np.tan(np.radians(self.cam["fov_y"]) / 2)
        rays = np.array(self.cam["forward"]) + np.array(self.cam["right"]) * ((2 * (x + .5) / w - 1) * tan * w / h)[..., None] + np.array(self.cam["up"]) * ((1 - 2 * (y + .5) / h) * tan)[..., None]
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        origins = np.broadcast_to(self.cam["origin"], rays.shape).copy()
        hits, normals = hits_and_normals(origins.reshape(-1, 3), rays.reshape(-1, 3), self.dem, self.spacing)
        return origins, rays, hits.reshape(shape), normals.reshape((*shape, 3))
