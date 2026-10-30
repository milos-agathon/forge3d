"""CPU qualification of a concrete discrete-ordinate froxel algorithm.

No forge3d import, native calls, rendering, fixture writes, or fitted parameters.
The candidate uses 192 existing Fibonacci directions, the existing 64 depth
steps, nearest-froxel per-density continuation, bilinear DEM-vertex Lambertian
return, and RGBA16F source storage. Source iteration ends at a bitwise f16
fixed point (no selected iteration cap). The measured order count is reported.
Every new visibility query is controlled by the same ablation switch.

Run with the preserved analysis-venv and --evidence / --output (a new directory).
Historical baselines share this program's geometry and numerical helpers;
CSM and device arithmetic cannot be reproduced from the captured AOVs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import numba
from numba import njit, prange

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from nephele_transport_cpu import (
    Scene, density_one, density_integral, terrain_hit, height_one, normal_one, slab,
)

spec = importlib.util.spec_from_file_location("release_analysis", HERE / "nephele-release-analysis-2026-10-22.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

# Existing runtime/fixture depth contract, not an analysis quality setting.
NEAR, FAR, DEPTH, VISIBLE, WIDTH = .1, 6000., 64, 8, 10
LOG = np.log(FAR / NEAR)


def half(value):
    result = np.asarray(value).astype(np.float16).astype(np.float32)
    if not np.isfinite(result).all():
        raise ArithmeticError("Non-finite half-float source; no silent clipping")
    return result


def hg(cosine, g):
    return (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * np.clip(cosine, -1, 1)) ** 1.5)


@njit
def coord(point, origin, forward, right, up, tan):
    q = point - origin
    front = np.dot(q, forward)
    # Shader projection at w=0 has no finite address. No quadrature sample in
    # the recorded study hits that plane; fail instead of inventing a fallback.
    if front == 0:
        raise ValueError("Undefined froxel address on the camera plane")
    x = int(np.floor(min(max((.5 + .5 * np.dot(q, right) / (front * tan)) * VISIBLE + 1, 0), WIDTH - 1)))
    y = int(np.floor(min(max((.5 - .5 * np.dot(q, up) / (front * tan)) * VISIBLE + 1, 0), WIDTH - 1)))
    distance = np.sqrt(np.dot(q, q))
    z = int(np.floor(min(max(np.log(max(distance, NEAR) / NEAR) / LOG * DEPTH, 0), DEPTH - 1)))
    return x + WIDTH * (y + WIDTH * z)


@njit(parallel=True)
def point_properties(points, grid, lo, hi, dem, spacing, sun, sigma_t):
    rho = np.empty(len(points))
    sun_t = np.empty((len(points), 3))
    visible = np.empty(len(points), np.bool_)
    for i in prange(len(points)):
        rho[i] = density_one(points[i], grid, lo, hi)
        tau = density_integral(points[i], sun, np.inf, grid, lo, hi)
        sun_t[i] = np.exp(-sigma_t * tau)
        visible[i] = not np.isfinite(terrain_hit(points[i], sun, dem, spacing, 1e-3))
    return rho, sun_t, visible


@njit(parallel=True)
def ray_operator(points, directions, enabled, grid, lo, hi, dem, spacing,
                 sigma_t, origin, forward, right, up, tan, active_index, surface_shape):
    """Exact trilinear optical depths; 64 piecewise constant source gathers.

    Stored weights multiply source PER DENSITY. Density is integrated exactly
    in every segment; the source coefficient is fetched at its midpoint.
    Geometry alone bounds every ray. Open-surface media are never height-cut.
    Ground radiance uses a separate froxel-resolution diffuse surface cache.
    """
    n, nd = len(points), len(directions)
    indices = np.full((n, nd, DEPTH), -1, np.int32)
    weights = np.zeros((n, nd, DEPTH, 3), np.float32)
    terminal_t = np.empty((n, nd, 3), np.float32)
    ground = np.full((n, nd, 4), -1, np.int32)
    ground_w = np.zeros((n, nd, 4), np.float32)
    for i in prange(n):
        p = points[i]
        for j in range(nd):
            d = directions[j]
            hit = terrain_hit(p, d, dem, spacing, 1e-3) if enabled else np.inf
            begin, end = slab(p, d, lo, hi)
            end = min(end, hit)
            tau = np.zeros(3)
            if end > begin:
                step = (end - begin) / DEPTH
                for k in range(DEPTH):
                    left = p + d * (begin + k * step)
                    segment = sigma_t * density_integral(left, d, step, grid, lo, hi)
                    sample = p + d * (begin + (k + .5) * step)
                    at = active_index[coord(sample, origin, forward, right, up, tan)]
                    indices[i, j, k] = at
                    integral = density_integral(left, d, step, grid, lo, hi)
                    for c in range(3):
                        weights[i, j, k, c] = np.exp(-tau[c]) * (-np.expm1(-segment[c])) / sigma_t[c] if sigma_t[c] > 0 else integral
                    tau += segment
            terminal_t[i, j] = np.exp(-tau)
            if np.isfinite(hit):
                point = p + d * hit
                h, w = surface_shape
                u = min(max((point[0] / (spacing[0] * (dem.shape[1] - 1)) + .5) * (w - 1), 0), w - 1)
                v = min(max((point[2] / (spacing[1] * (dem.shape[0] - 1)) + .5) * (h - 1), 0), h - 1)
                x, y = min(int(np.floor(u)), w - 2), min(int(np.floor(v)), h - 2)
                a, b = u - x, v - y
                ground[i, j] = np.array([x + w * y, x + 1 + w * y, x + w * (y + 1), x + 1 + w * (y + 1)])
                ground_w[i, j] = np.array([(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b])
    return indices, weights, terminal_t, ground, ground_w


@njit(parallel=True)
def apply_operator(indices, weights, terminal_t, ground, ground_w, source, surface, env_l):
    total = np.zeros(terminal_t.shape, np.float32)
    for i in prange(len(indices)):
        for j in range(indices.shape[1]):
            terminal = np.full(3, env_l, np.float32)
            if ground[i, j, 0] >= 0:
                terminal[:] = 0
                for k in range(4):
                    terminal += ground_w[i, j, k] * surface[ground[i, j, k]]
            total[i, j] = terminal_t[i, j] * terminal
            for k in range(indices.shape[2]):
                at = indices[i, j, k]
                if at >= 0:
                    total[i, j] += weights[i, j, k] * source[at, j]
    return total


@njit(parallel=True)
def baseline_surface(points, normals, enabled, sphere, grid, lo, hi, dem, spacing, sigma_s, sigma_t, sun, sun_l, env_l, albedo, g):
    """Existing unattenuated IBL plus first-sun cloudlight, both arms.

    Same 192 directions and canonical 64^3 half-cell step as the production
    surface shader. Exact sun queries stand in for its unavailable world-sun
    texture and CSM; that modeling difference is reported, not fitted away.
    """
    out = np.empty((len(points), 3), np.float32)
    march_spacing = np.min((hi - lo) / 64) * .5
    for i in prange(len(points)):
        p = points[i] + normals[i] * 1e-3
        tau = density_integral(p, sun, np.inf, grid, lo, hi)
        # Existing direct surface CSM is not the media ablation switch.
        visible = not np.isfinite(terrain_hit(p, sun, dem, spacing, 1e-3))
        direct = sun_l * np.exp(-sigma_t * tau) * visible * max(np.dot(normals[i], sun), 0)
        cloud = np.zeros(3)
        for d in sphere:
            cosine = max(np.dot(normals[i], d), 0)
            if cosine == 0:
                continue
            begin, end = slab(p, d, lo, hi)
            if enabled:
                end = min(end, terrain_hit(p, d, dem, spacing, 1e-4))
            if end <= begin:
                continue
            count = int(np.ceil((end - begin) / march_spacing))
            step = (end - begin) / count
            trans, radiance = np.ones(3), np.zeros(3)
            phase = (1 - g*g) / (4*np.pi*(1 + g*g - 2*g*np.dot(d, sun))**1.5)
            for k in range(count):
                q = p + d * (begin + (k + .5) * step)
                rho = density_one(q, grid, lo, hi)
                ext = rho * sigma_t
                tau_sun = density_integral(q, sun, np.inf, grid, lo, hi)
                light_visible = not enabled or not np.isfinite(terrain_hit(q, sun, dem, spacing, 1e-4))
                source = sigma_s * rho * sun_l * np.exp(-sigma_t * tau_sun) * light_visible
                radiance += trans * np.exp(-ext * step * .5) * source * phase * step
                trans *= np.exp(-ext * step)
            cloud += radiance * cosine * (4 * np.pi / len(sphere))
        out[i] = albedo * (env_l + (direct + cloud) / np.pi)
    return out


class Model:
    def __init__(self, scene):
        self.scene = scene
        self.origin, self.forward, self.right, self.up = [np.array(scene.cam[k]) for k in ("origin", "forward", "right", "up")]
        self.tan = np.tan(np.radians(scene.cam["fov_y"]) / 2)
        z, y, x = np.indices((DEPTH, WIDTH, WIDTH))
        self.ids = np.stack([x, y, z], -1).reshape(-1, 3)
        uv = (self.ids[:, :2] + .5 - 1) / VISIBLE
        self.rays = self.forward + self.right * ((2 * uv[:, 0] - 1) * self.tan)[:, None] + self.up * ((1 - 2 * uv[:, 1]) * self.tan)[:, None]
        self.rays /= np.linalg.norm(self.rays, axis=-1, keepdims=True)
        distance = NEAR * np.exp(LOG * (self.ids[:, 2] + .5) / DEPTH)
        self.world = self.origin + self.rays * distance[:, None]
        density, sun_t, visible = self.properties(self.world)
        self.rho, self.sun_t, self.sun_visible = half(density), half(sun_t), visible
        self.extinction = half(density[:, None] * scene.sigma_t)
        self.active = np.flatnonzero(self.rho > 0)
        self.index = np.full(len(self.ids), -1, np.int32)
        self.index[self.active] = np.arange(len(self.active))
        self.origins, self.pixel_rays, self.hit_t, self.normals = scene.camera((64, 64))
        self.hit = np.isfinite(self.hit_t)
        self.primary_points = (self.origins + self.pixel_rays * np.where(self.hit, self.hit_t, 0)[..., None])[self.hit]
        # Reuse the actual froxel grid's horizontal dimensions. Increasing DEM
        # resolution alone changes geometry detail, never the cache allocation.
        self.surface_shape = (WIDTH, WIDTH)
        h, w = self.surface_shape
        tz, tx = np.indices((h, w))
        xz = np.stack([(tx / (w - 1) - .5) * (scene.dem.shape[1] - 1) * scene.spacing[0],
                       (tz / (h - 1) - .5) * (scene.dem.shape[0] - 1) * scene.spacing[1]], -1).reshape(-1, 2)
        self.vertices = np.stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]], -1)
        self.vertices[:, 1] = [height_one(p, scene.dem, scene.spacing) for p in self.vertices]
        self.vertex_normals = np.array([normal_one(p, scene.dem, scene.spacing) for p in self.vertices])
        ranks = np.loadtxt(ROOT / "assets/media/nephele_blue_noise_8x8.txt")
        p = self.ids[self.active]
        self.u1 = (ranks[(p[:, 1] + p[:, 2] * 5) % 8, (p[:, 0] + p[:, 2] * 3) % 8] + .5) / 64
        self.shifted_u2 = (ranks[(p[:, 1] + p[:, 2] * 5 + 7) % 8, (p[:, 0] + p[:, 2] * 3 + 1) % 8] + .5) / 64

    def properties(self, points):
        s = self.scene
        return point_properties(np.ascontiguousarray(points), s.grid, s.lo, s.hi, s.dem, s.spacing, s.sun, s.sigma_t)

    def operator(self, points, enabled):
        s = self.scene
        return ray_operator(np.ascontiguousarray(points), s.sphere, enabled, s.grid, s.lo, s.hi, s.dem, s.spacing, s.sigma_t,
                            self.origin, self.forward, self.right, self.up, self.tan, self.index, self.surface_shape)

    def coords(self, points):
        return np.array([coord(p, self.origin, self.forward, self.right, self.up, self.tan) for p in points.reshape(-1, 3)]).reshape(points.shape[:-1])

    def tent(self, field):
        out = np.zeros_like(field)
        for i in self.active:
            x, y, z = self.ids[i]
            total, weight = np.zeros(field.shape[1:]), 0.
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    j = np.clip(x + dx, 0, WIDTH - 1) + WIDTH * (np.clip(y + dy, 0, WIDTH - 1) + WIDTH * z)
                    if self.rho[j] > 0:
                        w = (2 - abs(dx)) * (2 - abs(dy))
                        total += w * field[j]
                        weight += w
            out[i] = total / weight
        return out

    def integrate(self, field):
        y, x = np.indices((64, 64))
        centre = np.stack([(x + .5) / 64 * VISIBLE + .5, (y + .5) / 64 * VISIBLE + .5], -1)
        base = np.clip(np.floor(centre), 0, WIDTH - 2).astype(int)
        f = centre - base
        stop = np.where(self.hit, self.hit_t, FAR)
        trans, value, previous = np.ones((64, 64, 3)), np.zeros((64, 64, 3)), NEAR
        for z in range(DEPTH):
            end = NEAR * np.exp(LOG * (z + 1) / DEPTH)
            step = np.maximum(np.minimum(end, stop) - previous, 0)
            previous = end
            ext, source = np.zeros_like(value), np.zeros_like(value)
            for dy in (0, 1):
                for dx in (0, 1):
                    j = base[..., 0] + dx + WIDTH * (base[..., 1] + dy + WIDTH * z)
                    w = (f[..., 0] if dx else 1 - f[..., 0]) * (f[..., 1] if dy else 1 - f[..., 1])
                    ext += self.extinction[j] * w[..., None]
                    source += field[j] * w[..., None]
            st = np.exp(-ext * step[..., None])
            weight = (1 - st) / np.maximum(ext, 1e-6)
            value += trans * source * weight
            trans *= st
        return half(value), half(trans)

    def candidate(self, enabled, output):
        s, n = self.scene, len(self.active)
        points = np.concatenate([self.world[self.active], self.vertices])
        print("candidate operator", enabled, len(points), flush=True)
        op = self.operator(points, enabled)
        phase = (4 * np.pi / len(s.sphere) * hg(s.sphere @ s.sphere.T, s.g)).astype(np.float32)
        phase_camera = (4 * np.pi / len(s.sphere) * hg(self.rays[self.active] @ s.sphere.T, s.g)).astype(np.float32)
        cosine = (4 * np.pi / len(s.sphere) * np.maximum(self.vertex_normals @ s.sphere.T, 0)).astype(np.float32)
        sun_t = self.sun_t[self.active] * (self.sun_visible[self.active, None] if enabled else 1)
        direct = (sun_t * s.sun_l)[:, None] * hg(s.sphere @ s.sun, s.g)[None, :, None]
        _, vertex_t, vertex_visible = self.properties(self.vertices)
        vertex_direct = s.albedo / np.pi * s.sun_l * vertex_t * np.maximum(self.vertex_normals @ s.sun, 0)[:, None] * (vertex_visible[:, None] if enabled else 1)
        source = np.zeros((n, len(s.sphere), 3), np.float32)
        surface = np.zeros((len(self.vertices), 3), np.float32)
        history, order, first_surface = [], 0, None
        rho = self.rho[self.active, None, None]
        while True:
            incoming = apply_operator(*op, source / rho, surface, s.env_l)
            scattered = np.matmul(incoming[:n].transpose(0, 2, 1), phase.T).transpose(0, 2, 1)
            next_source = half(s.sigma_s * rho * (direct + scattered))
            next_surface = half(vertex_direct + s.albedo / np.pi * np.einsum("ij,ijc->ic", cosine, incoming[n:]))
            order += 1
            if order == 1:
                first_surface = next_surface.copy()
            delta = float(max(np.max(np.abs(next_source - source)), np.max(np.abs(next_surface - surface))))
            if np.any(next_source < source) or np.any(next_surface < surface):
                raise ArithmeticError("Positive quantized source iteration lost monotonicity")
            history.append({"order": order, "maximum_stored_rgb_change": delta,
                            "changed_source_channels": int(np.count_nonzero(next_source != source)),
                            "changed_surface_channels": int(np.count_nonzero(next_surface != surface))})
            stable = np.array_equal(next_source, source) and np.array_equal(next_surface, surface)
            source, surface = next_source, next_surface
            if stable:
                break
        # No scalar angular closure: camera source is a fresh phase-weighted
        # gather of the same directional incoming radiance used by the shader.
        incoming = apply_operator(*op, source / rho, surface, s.env_l)
        camera = s.sigma_s * self.rho[self.active, None] * (
            sun_t * s.sun_l * hg(self.rays[self.active] @ s.sun, s.g)[:, None]
            + np.einsum("ij,ijc->ic", phase_camera, incoming[:n]))
        camera_field = np.zeros((len(self.ids), 3), np.float32)
        camera_field[self.active] = half(camera)
        ins, trans = self.integrate(camera_field)
        print("candidate converged", enabled, order, flush=True)
        del op, incoming
        primary_op = self.operator(self.primary_points, enabled)
        incoming_primary = apply_operator(*primary_op, source / rho, surface, s.env_l)
        environment = primary_op[2] * s.env_l * (primary_op[3][..., :1] < 0)
        returned = apply_operator(*primary_op, source / rho, surface, 0.)
        medium_return = apply_operator(*primary_op, source / rho, np.zeros_like(surface), 0.)
        normals = self.normals[self.hit]
        weights = 4 * np.pi / len(s.sphere) * np.maximum(normals @ s.sphere.T, 0)
        # Preserve the existing primary direct-sun term. Only the environment
        # boundary and medium return are replaced; secondary cache visibility
        # remains controlled by the ablation in every iteration above.
        _, primary_t, primary_visible = self.properties(self.primary_points + normals * 1e-3)
        solar = s.albedo / np.pi * s.sun_l * primary_t * np.maximum(normals @ s.sun, 0)[:, None] * primary_visible[:, None]
        surface_primary = half(solar + s.albedo / np.pi * np.einsum("ij,ijc->ic", weights, incoming_primary))
        primary = np.full((64, 64, 3), s.env_l, np.float32)
        primary[self.hit] = surface_primary
        np.savez(output / f"directional-{enabled}.npz", ins=ins, trans=trans, boundary=primary,
                 source=source, surface_cache=surface, first_surface_cache=first_surface, camera_field=camera_field,
                 environment=s.albedo / np.pi * np.einsum("ij,ijc->ic", weights, environment),
                 medium_return=s.albedo / np.pi * np.einsum("ij,ijc->ic", weights, medium_return),
                 all_return=s.albedo / np.pi * np.einsum("ij,ijc->ic", weights, returned), solar=solar)
        return {"ins": ins, "trans": trans, "boundary": primary, "history": history,
                "angular_source_range": np.ptp(source / rho, axis=1).mean(0).tolist()}


def baseline(model, state, enabled):
    """Source-faithful baseline: d550 cached field, ac world field, 68 CDF.

    Exact bilinear ray roots replace unavailable captured traversal buffers;
    exact sun integrals replace unavailable canonical/CSM texture values.
    These residuals are measured, never calibrated away.
    """
    s = model.scene
    p = model.world[model.active]
    incident = model.rays[model.active]
    u1 = model.u1
    if state == "68cc04e9":
        ranks = np.floor(u1 * 64).astype(int)
        u2 = (np.array([int(f"{r:06b}"[::-1], 2) for r in ranks]) + .5) / 64
    else:
        u2 = model.shifted_u2
    ct = np.clip((1 + s.g * s.g - ((1 - s.g * s.g) / (1 - s.g + 2 * s.g * u1)) ** 2) / (2 * s.g), -1, 1)
    helper = np.where((np.abs(incident[:, 1]) < .999)[:, None], [0., 1., 0.], [1., 0., 0.])
    tangent = np.cross(helper, incident)
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
    bitangent = np.cross(incident, tangent)
    rays = tangent * (np.cos(2 * np.pi * u2) * np.sqrt(1 - ct * ct))[:, None] + bitangent * (np.sin(2 * np.pi * u2) * np.sqrt(1 - ct * ct))[:, None] + incident * ct[:, None]
    hits = np.array([terrain_hit(o, d, s.dem, s.spacing, 1e-4) for o, d in zip(p, rays)])
    reaches = np.isfinite(hits) & enabled
    reach = np.where(reaches, np.minimum(hits, FAR), FAR)
    hp = p + rays * np.where(np.isfinite(hits), hits, 0)[:, None]
    normals = np.array([normal_one(q, s.dem, s.spacing) for q in hp])
    _, _, surface_visibility = model.properties(hp + normals * 1e-3)
    terminal_ground = s.albedo * (np.maximum(normals @ s.sun, 0)[:, None] * surface_visibility[:, None] * s.sun_l * model.sun_t[model.coords(hp)] / np.pi + s.env_l)
    terminal = np.where(reaches[:, None], terminal_ground, s.env_l)
    intervals = np.array([slab(o, d, s.lo, s.hi) for o, d in zip(p, rays)])
    if state == "d5507249":
        begin, end = np.zeros(len(p)), reach
    else:
        begin = intervals[:, 0]
        end = np.maximum(begin, np.minimum(reach, intervals[:, 1]))
    step = (end - begin) / DEPTH
    samples = p[:, None] + rays[:, None] * (begin[:, None] + (np.arange(DEPTH) + .5) * step[:, None])[..., None]
    js = model.coords(samples)
    sample_rho, sample_t, sample_vis = model.properties(samples.reshape(-1, 3))
    sample_rho, sample_t, sample_vis = sample_rho.reshape(js.shape), sample_t.reshape((*js.shape, 3)), sample_vis.reshape(js.shape)
    ext = model.extinction[js] if state == "d5507249" else sample_rho[..., None] * s.sigma_t
    st = np.exp(-ext * step[:, None, None])
    trans = np.concatenate([np.ones((len(p), 1, 3)), np.cumprod(st, axis=1)[:, :-1]], axis=1)
    kernel = trans * (1 - st) / np.maximum(ext, 1e-6)
    sun_source = half(s.sigma_s * model.rho[model.active, None] * s.sun_l * model.sun_t[model.active] * (model.sun_visible[model.active, None] if enabled else 1))
    camera_phase = half(hg(incident @ s.sun, s.g))
    raw = np.zeros((len(model.ids), 3), np.float32)
    raw[model.active] = half(sun_source * hg(incident @ s.sun, s.g)[:, None] + s.sigma_s * model.rho[model.active, None] * np.prod(st, axis=1) * terminal)
    single = half(model.tent(raw))
    sun_field = np.zeros_like(raw)
    sun_field[model.active] = sun_source * camera_phase[:, None]
    camera_sun = model.tent(sun_field)
    sun_field[model.active] = sun_source
    filtered_sun = model.tent(sun_field)
    filtered_rho = model.tent(model.rho[:, None])[:, 0]
    env = np.maximum(single - camera_sun, 0)
    if state == "d5507249":
        gathered = env[js] + filtered_sun[js] * hg(rays @ s.sun, s.g)[:, None, None]
    else:
        per_rho = np.divide(env, filtered_rho[:, None], out=np.zeros_like(env), where=filtered_rho[:, None] > 0)
        gathered = per_rho[js] * sample_rho[..., None] + s.sigma_s * sample_rho[..., None] * s.sun_l * sample_t * (sample_vis[..., None] if enabled else 1) * hg(rays @ s.sun, s.g)[:, None, None]
    field = single.copy()
    field[model.active] = half(single[model.active] + s.sigma_s * model.rho[model.active, None] * np.sum(kernel * gathered, axis=1))
    ins, transmittance = model.integrate(field)
    return ins, transmittance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output, evidence = args.output.resolve(), args.evidence.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    scene, report = Scene(ROOT), {}
    model = Model(scene)
    print("froxel model", len(model.active), "active; terrain vertices", len(model.vertices), flush=True)
    regions = {"sky": analysis.load(analysis.gate.FIXTURE, "sky-cloud-mask"),
               "terrain": analysis.load(analysis.gate.FIXTURE, "terrain-mask")}
    regions["shadow"] = regions["terrain"] & analysis.load(analysis.gate.FIXTURE, "cloud-shadow-mask")
    report.update({"head_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "environment": {"python": sys.version, "numpy": np.__version__, "numba": numba.__version__, "numba_threads": numba.get_num_threads(), "native_forge3d_imported": "forge3d" in sys.modules},
                   "contracts": {"grid": [WIDTH, WIDTH, DEPTH], "active_froxels_both_arms": len(model.active), "directions": len(scene.sphere), "segment_count": DEPTH,
                                 "direction_source": "main_nephele_surface_cloudlight existing 192-direction Fibonacci quadrature", "steps_source": "media.grid.z existing 64-depth contract",
                                 "iteration_stop": "bitwise identical stored f16 source and terrain caches; no cap", "domain": "untruncated medium on both sides; surface crossings control light only",
                                 "surface_cache_dimensions": list(model.surface_shape), "surface_cache_resolution_source": "froxel grid horizontal dimensions, independent of DEM resolution"}, "baselines": {}, "backcasts": {}})
    inputs = [Path(__file__), HERE / "nephele_transport_cpu.py", HERE / "nephele-release-analysis-2026-10-22.py", ROOT / "assets/media/nephele_blue_noise_8x8.txt", ROOT / "tests/nephele/gate4-policy.json",
              ROOT / "src/shaders/nephele_froxel.wgsl", ROOT / "src/shaders/nephele_realtime_terrain_trace.wgsl", evidence / "continuation-study/coupled-surface.npz"]
    inputs += [p for p in (ROOT / "tests/nephele/fixture").iterdir() if p.is_file()]
    runs = {"d5507249": 39545456967, "ac739096": 39566600606, "68cc04e9": 39580065706}
    captures, baselines = {}, {}
    for state, run in runs.items():
        capture = evidence / f"ci-{run}"
        inputs += [capture / f"{name}.npy" for name in ("realtime-rgb", "terrain-occlusion-disabled-rgb", "medium-disabled-rgb", "in-scatter", "transmittance")]
        values = {name: analysis.load(capture, name) for name in ("realtime-rgb", "terrain-occlusion-disabled-rgb", "medium-disabled-rgb", "in-scatter", "transmittance")}
        values["linear"] = analysis.gate._inverse_display_aces(values["realtime-rgb"])
        values["linear_off"] = analysis.gate._inverse_display_aces(values["terrain-occlusion-disabled-rgb"])
        captures[state] = values
        a, t = baseline(model, state, True)
        b, _ = baseline(model, state, False)
        baselines[state] = (a, b)
        report["baselines"][state] = {"run": run, "run_url": f"https://github.com/milos-agathon/forge3d/actions/runs/{run}",
                                    "metrics": analysis.score(values["realtime-rgb"], values["terrain-occlusion-disabled-rgb"], values["medium-disabled-rgb"])[0],
                                    "mean_absolute_in_scatter_rgb_error": {r: np.abs(a - values["in-scatter"])[mask].mean(0).tolist() for r, mask in regions.items()},
                                    "mean_absolute_transmittance_rgb_error": {r: np.abs(t - values["transmittance"])[mask].mean(0).tolist() for r, mask in regions.items()}}
        np.savez(output / f"baseline-{state}.npz", ins=a, ins_off=b, trans=t)
        print("baseline", state, report["baselines"][state]["mean_absolute_in_scatter_rgb_error"], flush=True)
    errors = []
    for old, new in (("ac739096", "68cc04e9"), ("d5507249", "ac739096")):
        oc, nc = captures[old], captures[new]
        prediction = oc["linear"] + baselines[new][0] - baselines[old][0]
        prediction_off = oc["linear_off"] + baselines[new][1] - baselines[old][1]
        error, error_off = nc["linear"] - prediction, nc["linear_off"] - prediction_off
        errors.append((f"{old}->{new}", error, error_off))
        metrics = analysis.score(analysis.display(prediction), analysis.display(prediction_off), nc["medium-disabled-rgb"])[0]
        measured = report["baselines"][new]["metrics"]
        report["backcasts"][f"{old}->{new}"] = {"predicted_metrics": metrics, "observed_metrics": measured,
            "metric_error_predicted_minus_observed": {k: metrics[k] - measured[k] for k in metrics if isinstance(metrics[k], float)},
            "mean_absolute_linear_rgb_error": {r: np.abs(error)[mask].mean(0).tolist() for r, mask in regions.items()}}
        np.savez(output / f"backcast-{old}-{new}.npz", error=error, error_off=error_off, prediction=prediction, prediction_off=prediction_off)
    current = captures["68cc04e9"]
    boundaries = []
    for enabled in (True, False):
        value = np.full((64, 64, 3), scene.env_l, np.float32)
        value[model.hit] = baseline_surface(model.primary_points, model.normals[model.hit], enabled, scene.sphere, scene.grid, scene.lo, scene.hi, scene.dem, scene.spacing, scene.sigma_s, scene.sigma_t, scene.sun, scene.sun_l, scene.env_l, scene.albedo, scene.g)
        boundaries.append(half(value))
        print("surface baseline", enabled, flush=True)
    boundary, boundary_off = boundaries
    np.savez(output / "baseline-surfaces.npz", boundary=boundary, boundary_off=boundary_off)
    candidates = [model.candidate(enabled, output) for enabled in (True, False)]
    new, new_off = candidates
    frame = current["linear"] + new["ins"] - baselines["68cc04e9"][0] + current["transmittance"] * (new["boundary"] - boundary) * model.hit[..., None]
    frame_off = current["linear_off"] + new_off["ins"] - baselines["68cc04e9"][1] + current["transmittance"] * (new_off["boundary"] - boundary_off) * model.hit[..., None]
    metrics, delta_e = analysis.score(analysis.display(frame), analysis.display(frame_off), current["medium-disabled-rgb"])
    sensitivity = {}
    for name, error, off_error in errors:
        scores = [analysis.score(analysis.display(frame + sign * error), analysis.display(frame_off + sign * off_error), current["medium-disabled-rgb"])[0] for sign in (-1, 1)]
        sensitivity[name] = {k: [min(s[k] for s in scores), max(s[k] for s in scores)] for k in metrics if isinstance(metrics[k], float)}
    old_surface = np.load(evidence / "continuation-study/coupled-surface.npz")["surface"]
    surface_difference = new["boundary"] - old_surface
    report["candidate"] = {"name": "directional-froxel-coupled-surface-f16-fixed-point-preserved-primary-sun", "metrics": metrics, "sensitivity": sensitivity,
                           "orders": {str(enabled): candidate["history"] for enabled, candidate in zip((True, False), candidates)},
                           "angular_source_range_mean_rgb": new["angular_source_range"],
                           "surface_vs_unreproducible_cache": {r: {"mean_signed_rgb": surface_difference[mask].mean(0).tolist(), "mean_absolute_rgb": np.abs(surface_difference[mask]).mean(0).tolist(), "max_absolute_rgb": np.abs(surface_difference[mask]).max(0).tolist()} for r, mask in regions.items() if r != "sky"}}
    report["decision"] = "NOMINAL_PASS_REQUIRES_IMPLEMENTATION" if metrics["all_five_pass"] else "STOP_NO_QUALIFIED_CANDIDATE"
    report["limitations"] = ["CPU arithmetic, not a physical render of HEAD; backend/CSM/traversal residuals remain.",
        "Backward predictions cover historical medium changes; primary surface changes are held at their observed boundary.",
        "Surface baseline uses exact world sun queries in place of captured CSM and 64^3 world-sun texture samples; their errors are retained.",
        "Directional and spatial quadrature are explicit shader algorithms, not a convergence certificate for the reference integral.",
        "Half-float fixed-point count is measured on this workload; other scenes require their own convergence test.",
        "Held-out plus/minus maps measure sensitivity, not a bound transferable to new physics.",
        "Frozen reference RNG dimension overlap is unchanged; reported separately."]
    report["input_sha256"] = {analysis.input_key(p): analysis.sha(p) for p in inputs}
    report["duration_seconds"] = time.perf_counter() - started
    report["environment"]["native_forge3d_imported"] = "forge3d" in sys.modules
    np.savez(output / "forecast.npz", rgb=analysis.display(frame), disabled=analysis.display(frame_off), linear=frame, linear_off=frame_off, delta_e2000=delta_e)
    def json_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unsupported audit JSON type: {type(value).__name__}")
    (output / "analysis.json").write_text(json.dumps(report, indent=2, default=json_scalar) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "metrics": metrics, "sensitivity": sensitivity, "seconds": report["duration_seconds"]}, indent=2, default=json_scalar), flush=True)


if __name__ == "__main__":
    main()
