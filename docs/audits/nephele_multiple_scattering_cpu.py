"""CPU path-integral check, independent of native/GPU execution.

Matches reference.rs RGB spectral tracking, HG, Lambert surfaces, environment
MIS and media_py.rs roulette (bounce >= 3, minimum .05), without a depth cap.
Exact optical-depth integration replaces stochastic shadow ratio tracking;
the unit fixture environment is sampled uniformly on the sphere. These are
unbiased variance reductions, not a bit-identical reference replay.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from numba import njit, prange

from nephele_transport_cpu import Scene, density_one, density_integral, slab, terrain_hit, normal_one


@njit
def uniform(frame, pixel, sample, bounce, dimension):
    p, s, b, k = np.uint64(pixel), np.uint64(sample), np.uint64(bounce), np.uint64(dimension)
    v = np.uint64(frame) ^ ((p << np.uint64(13)) | (p >> np.uint64(51))) ^ ((s << np.uint64(29)) | (s >> np.uint64(35)))
    v ^= ((b << np.uint64(47)) | (b >> np.uint64(17))) ^ k * np.uint64(0x9e3779b97f4a7c15)
    v ^= v >> np.uint64(30)
    v *= np.uint64(0xbf58476d1ce4e5b9)
    v ^= v >> np.uint64(27)
    v *= np.uint64(0x94d049bb133111eb)
    v ^= v >> np.uint64(31)
    return (float(v >> np.uint64(41)) + .5) / (1 << 23)


@njit
def phase(cosine, g):
    return (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * cosine) ** 1.5)


@njit
def local_direction(w, cosine, phi):
    helper = np.array([0., 0., 1.]) if abs(w[2]) < .999 else np.array([1., 0., 0.])
    tangent = np.cross(helper, w)
    tangent /= np.sqrt((tangent * tangent).sum())
    bitangent = np.cross(w, tangent)
    radius = np.sqrt(max(0., 1 - cosine * cosine))
    d = w * cosine + radius * (tangent * np.cos(phi) + bitangent * np.sin(phi))
    return d / np.sqrt((d * d).sum())


@njit
def sphere_direction(u, v):
    y, phi = 1 - 2 * u, 2 * np.pi * v
    r = np.sqrt(max(0., 1 - y * y))
    return np.array([r * np.cos(phi), y, r * np.sin(phi)])


@njit
def trace(o, d, frame, pixel, sample, occlusion, grid, lo, hi, dem, spacing,
          sigma_s, sigma_t, sun, sun_l, env_l, albedo, g, rate):
    throughput, radiance, first_order = np.ones(3), np.zeros(3), np.zeros(3)
    previous_pdf, bounce, from_medium = -1., 0, False
    while True:
        # Only terrain reached from a medium-scatter event is suppressed by
        # the disabled arm. Primary/surface transport keeps its geometry.
        hit = terrain_hit(o, d, dem, spacing, 1e-3) if occlusion or not from_medium else np.inf
        begin, end = slab(o, d, lo, hi)
        end = min(end, hit)
        distance, step, collided = begin, 0, False
        while rate > 0 and distance < end:
            distance += -np.log(uniform(frame, pixel, sample, bounce, 2 * step)) / rate
            if distance >= end:
                break
            rho = density_one(o + d * distance, grid, lo, hi)
            sigma = sigma_t * rho
            mean = sigma.mean()
            if uniform(frame, pixel, sample, bounce, 2 * step + 1) < mean / rate:
                # Real-event likelihood weight times scattering albedo.
                throughput *= sigma_s * rho / mean
                collided = True
                break
            throughput *= (rate - sigma) / (rate - mean)
            step += 1
        if collided:
            p = o + d * distance
            visible = not occlusion or not np.isfinite(terrain_hit(p, sun, dem, spacing, 1e-3))
            if visible:
                tau = density_integral(p, sun, np.inf, grid, lo, hi)
                value = throughput * sun_l * phase((d * sun).sum(), g) * np.exp(-sigma_t * tau)
                radiance += value
                if bounce == 0:
                    first_order += value
            light = sphere_direction(uniform(frame, pixel, sample, bounce, 10), uniform(frame, pixel, sample, bounce, 11))
            pdf = 1 / (4 * np.pi)
            pp = phase((d * light).sum(), g)
            visible = not occlusion or not np.isfinite(terrain_hit(p, light, dem, spacing, 1e-3))
            if visible:
                tau = density_integral(p, light, np.inf, grid, lo, hi)
                value = throughput * env_l * pp / pdf * np.exp(-sigma_t * tau)
                radiance += value * pdf * pdf / (pdf * pdf + pp * pp)
                if bounce == 0:
                    first_order += value
            q = 2 * uniform(frame, pixel, sample, bounce, 20) - 1
            numerator = 2 * q + g * (q * q + 3) + 2 * g * g * q + g ** 3 * (q * q - 1)
            cosine = min(1., max(-1., numerator / (2 * (1 + g * q) ** 2)))
            direction = local_direction(d, cosine, 2 * np.pi * uniform(frame, pixel, sample, bounce, 21))
            previous_pdf = phase(cosine, g)
            o, d, from_medium = p, direction, True
        elif np.isfinite(hit):
            p = o + d * hit
            n = normal_one(p, dem, spacing)
            cosine = max(0., (n * sun).sum())
            if cosine > 0 and not np.isfinite(terrain_hit(p, sun, dem, spacing, 1e-3)):
                tau = density_integral(p, sun, np.inf, grid, lo, hi)
                radiance += throughput * albedo * sun_l * cosine / np.pi * np.exp(-sigma_t * tau)
            light = sphere_direction(uniform(frame, pixel, sample, bounce, 32), uniform(frame, pixel, sample, bounce, 33))
            cosine = max(0., (n * light).sum())
            if cosine > 0 and not np.isfinite(terrain_hit(p, light, dem, spacing, 1e-3)):
                tau = density_integral(p, light, np.inf, grid, lo, hi)
                pdf, pp = 1 / (4 * np.pi), cosine / np.pi
                radiance += throughput * albedo * env_l * cosine / (np.pi * pdf) * np.exp(-sigma_t * tau) * pdf ** 2 / (pdf ** 2 + pp ** 2)
            cosine = np.sqrt(1 - uniform(frame, pixel, sample, bounce, 30))
            direction = local_direction(n, cosine, 2 * np.pi * uniform(frame, pixel, sample, bounce, 31))
            throughput *= albedo
            previous_pdf = cosine / np.pi
            o, d, from_medium = p, direction, False
        else:
            pdf = 1 / (4 * np.pi)
            weight = 1. if previous_pdf < 0 else previous_pdf ** 2 / (previous_pdf ** 2 + pdf ** 2)
            radiance += throughput * env_l * weight
            break
        if bounce >= 3:
            probability = min(1., max(.05, throughput.max()))
            if uniform(frame, pixel, sample, bounce, 40) >= probability:
                break
            throughput /= probability
        if throughput.max() == 0:
            break
        bounce += 1
    return radiance, first_order, bounce


@njit(parallel=True)
def render(origins, directions, pixels, start, count, seed, grid, lo, hi, dem, spacing,
           sigma_s, sigma_t, sun, sun_l, env_l, albedo, g):
    values = np.empty((2, len(pixels), count, 6))
    bounces = np.zeros((2, len(pixels), count), np.int64)
    rate = grid.max() * sigma_t.max()
    for i in prange(len(pixels)):
        for sample in range(count):
            for arm in range(2):
                full, first, depth = trace(origins[i], directions[i], seed, pixels[i], start + sample, arm == 1,
                                          grid, lo, hi, dem, spacing, sigma_s, sigma_t, sun, sun_l, env_l, albedo, g, rate)
                values[arm, i, sample, :3] = full
                values[arm, i, sample, 3:] = first
                bounces[arm, i, sample] = depth
    return values, bounces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-order-study", type=Path, required=True)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--samples", type=int, help="Default is the frozen reference's 640 spp; report the authority for an override")
    args = parser.parse_args()
    if args.sample_start < 0 or (args.samples is not None and args.samples <= 0):
        parser.error("sample identities must be nonnegative and count positive")
    root = Path(__file__).resolve().parents[2]
    fixture = root / "tests/nephele/fixture"
    provenance = json.loads((fixture / "reference-provenance.json").read_text())
    count = provenance["samples_per_pixel"] if args.samples is None else args.samples
    helper = Path(__file__).with_name("nephele-release-analysis-2026-10-22.py")
    first_report = args.first_order_study.resolve() / "analysis.json"
    inputs = [Path(__file__), Path(__file__).with_name("nephele_transport_cpu.py"), helper, first_report, root / "tests/nephele/gate4-policy.json"]
    inputs += [root / path for path in ("src/media/reference.rs", "src/media/tracking.rs", "src/media/model.rs", "src/media/density.rs", "src/media/lighting.rs", "src/media_py.rs", "src/shaders/nephele_froxel.wgsl", "src/path_tracing/hybrid_compute/media_reference.rs")]
    inputs += [p for p in sorted(fixture.iterdir()) if p.is_file()]
    hashes = {(p.relative_to(root).as_posix() if p.is_relative_to(root) else p.as_posix()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    scene = Scene(root)
    roi = np.load(fixture / "godray-roi-mask.npy").astype(bool)
    o, d, _, _ = scene.camera(roi.shape)
    pixels = np.flatnonzero(roi)
    print(f"CPU paired paths: {len(pixels)} ROI pixels x {count} samples x two arms", flush=True)
    values, bounces = render(o[roi], d[roi], pixels, args.sample_start, count, provenance["seed"], scene.grid, scene.lo, scene.hi,
                             scene.dem, scene.spacing, scene.sigma_s, scene.sigma_t, scene.sun, scene.sun_l, scene.env_l, scene.albedo, scene.g)
    assert np.isfinite(values).all() and (values >= 0).all()
    np.savez(args.output / "paths.npz", values=values, bounces=bounces, pixels=pixels, sample_start=args.sample_start)
    lum = np.array([.2126, .7152, .0722])
    policy = json.loads((root / "tests/nephele/gate4-policy.json").read_text())["occlusion_response"]
    target = policy["reference"]["mean_linear_luminance_change"]
    response = (values[0, :, :, :3] - values[1, :, :, :3]) @ lum
    first = (values[0, :, :, 3:] - values[1, :, :, 3:]) @ lum
    spec = importlib.util.spec_from_file_location("analysis", helper)
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    reference = analysis.gate._inverse_display_aces(np.load(fixture / "reference-rgb.npy"))[roi]
    enabled = values[1, :, :, :3]
    cpu_mean = enabled.mean(axis=(0, 1))
    cpu_sem = np.sqrt(enabled.var(axis=1, ddof=1).sum(axis=0) / count) / len(pixels)
    first_fine = json.loads(first_report.read_text())["levels"]["fine"]
    integrated_first = first_fine["sun_response"] + first_fine["environment_response"]
    # Independent sample identities within each fixed pixel; paired arms.
    sem = lambda x: float(np.sqrt(x.var(axis=1, ddof=1).sum() / count) / len(pixels)) if count > 1 else None
    summary = {"status": "CPU_EXPECTATION_CHECK_NOT_GPU_RELEASE_PROOF", "input_sha256": hashes,
               "python": sys.version, "numpy": np.__version__, "numba": importlib.metadata.version("numba"), "native_forge3d_imported": "forge3d" in sys.modules, "roi_pixels": len(pixels), "samples_per_pixel": count,
               "sample_start": args.sample_start, "sample_count_authority": "reference-provenance.json; any override is an explicitly reported follow-up",
               "model": "reference spectral delta tracking, HG, diffuse two-sided open terrain with upward normals, full surface/media recursion and MIS; exact shadow optical depth; uniform constant environment",
               "ablation": "Same untruncated medium in both arms. Disable sun/environment visibility at medium collisions and terrain intersection on medium continuation rays; primary and surface queries retain open-surface terrain.",
               "roulette_authority": "src/media_py.rs: roulette_start_bounce=3, minimum_probability=.05; no bounce cap",
               "response": float(response.mean()), "paired_standard_error": sem(response), "first_order_response": float(first.mean()),
               "first_order_paired_standard_error": sem(first), "beyond_direct_first_order_response": float((response - first).mean()),
               "beyond_direct_first_order_paired_standard_error": sem(response - first),
               "first_order_cross_check": {"deterministic_fine_response": integrated_first, "sampled_minus_integrated": float(first.mean() - integrated_first), "scope": "Independent sampled estimator versus the recorded 192-direction nested first-order integral; not a full-frame convergence certificate"},
               "enabled_frozen_reference_cross_check": {"frozen_mean_roi_linear_rgb": reference.mean(0).tolist(), "cpu_minus_frozen_mean_rgb": (cpu_mean - reference.mean(0)).tolist(), "cpu_mean_standard_error_rgb": cpu_sem.tolist(), "mean_absolute_per_pixel_linear_rgb_difference": np.abs(enabled.mean(axis=1) - reference).mean(0).tolist(), "scope": "Frozen image also has 640-spp noise; no unavailable frozen variance or bitwise equivalence is inferred"},
               "frozen_target": target, "allowed_response_interval": [target * (1 - policy["maximum_relative_error"]), target * (1 + policy["maximum_relative_error"])],
               "nominal_relative_error": float(abs(response.mean() - target) / target),
               "observed_maximum_bounce": int(bounces.max()), "mean_bounces": float(bounces.mean()),
               "enabled_mean_roi_linear_rgb": values[1, :, :, :3].mean(axis=(0, 1)).tolist(),
               "disabled_mean_roi_linear_rgb": values[0, :, :, :3].mean(axis=(0, 1)).tolist(),
               "duration_seconds": time.perf_counter() - started,
               "limitations": ["CPU expectation estimator, not native execution or bit-identical f32 replay", "Frozen reference has no disabled arm; ablation follows runtime medium visibility/terminal-light semantics", "ROI only; no all-five frame qualification", "Monte Carlo standard errors are empirical, not deterministic error bounds"]}
    summary["input_integrity_after_run"] = all(hashlib.sha256((root / key).read_bytes()).hexdigest() == digest for key, digest in hashes.items())
    assert summary["input_integrity_after_run"]
    (args.output / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "input_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
