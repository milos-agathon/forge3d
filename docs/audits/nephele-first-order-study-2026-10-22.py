"""Contract-spaced first-order CPU split; frozen references are read-only.

Both nested spacings come from the production 64^3 world-field cell. Angular
quadrature is the existing 192-direction contract. Optical depth is integrated
exactly through the trilinear density polynomial, with no ray/march cap.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time

import numpy as np
from numba import njit, prange

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from nephele_transport_cpu import Scene, slab, density_integral

spec = importlib.util.spec_from_file_location("analysis", HERE / "nephele-release-analysis-2026-10-22.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


@njit(parallel=True)
def tau_rays(origins, directions, reaches, grid, lo, hi):
    result = np.empty(len(origins))
    for i in prange(len(origins)):
        result[i] = density_integral(origins[i], directions[i], reaches[i], grid, lo, hi)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    scene = Scene(ROOT)
    terrain = a.load(a.gate.FIXTURE, "terrain-mask")
    origins, directions, hit_t, normals = scene.camera(terrain.shape)
    shape = terrain.shape
    o, d, t = origins.reshape(-1, 3), directions.reshape(-1, 3), hit_t.ravel()
    hit = np.isfinite(t)
    intervals = np.array([slab(o[i], d[i], scene.lo, scene.hi) for i in range(len(o))])
    begins, ends = intervals[:, 0], np.minimum(intervals[:, 1], t)
    lengths = np.maximum(ends - begins, 0)
    tau = tau_rays(o, d, t, scene.grid, scene.lo, scene.hi)
    camera_t = np.exp(-tau[:, None] * scene.sigma_t)
    points = o[hit] + d[hit] * t[hit, None]
    surface_sun, surface_env = scene.surface(points, normals.reshape(-1, 3)[hit])
    primary = np.full_like(o, scene.env_l)
    primary[hit] = surface_sun[1] + surface_env[1]
    cell = np.min((scene.hi - scene.lo) / 64)
    sky = a.load(a.gate.FIXTURE, "sky-cloud-mask")
    shadow = terrain & a.load(a.gate.FIXTURE, "cloud-shadow-mask")
    roi = a.load(a.gate.FIXTURE, "godray-roi-mask")
    capture = args.artifacts / "ci-39580065706"
    no_medium, disabled = a.load(capture, "medium-disabled-rgb"), a.load(capture, "terrain-occlusion-disabled-rgb")
    reference = a.gate._inverse_display_aces(a.load(a.gate.FIXTURE, "reference-rgb"))
    report = {"status": "FIRST_ORDER_ONLY_HIGHER_ORDERS_UNRESOLVED", "contract": {"world_field": "64^3 from renderer/media.rs upload_world_medium", "coarse_spacing": cell, "fine_spacing": cell / 2, "angular_directions": len(scene.sphere), "optical_depth": "piecewise cubic exact two-node Gauss", "geometry": "bilinear terrain leaf roots, reference query tmin 1e-3"}, "levels": {}, "input_sha256": {p.relative_to(ROOT).as_posix(): a.sha(p) for p in sorted(a.gate.FIXTURE.iterdir()) if p.is_file()}}
    report["input_sha256"].update({p.relative_to(ROOT).as_posix(): a.sha(p) for p in (Path(__file__), HERE / "nephele_transport_cpu.py", HERE / "nephele-release-analysis-2026-10-22.py")})
    values = {}
    for label, spacing in (("coarse", cell), ("fine", cell / 2)):
        counts = np.ceil(lengths / spacing).astype(int)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        ids = np.repeat(np.arange(len(o)), counts)
        sample = np.arange(offsets[-1]) - np.repeat(offsets[:-1], counts)
        dt = np.divide(lengths, counts, out=np.zeros_like(lengths), where=counts > 0)
        reach = begins[ids] + (sample + .5) * dt[ids]
        q = o[ids] + d[ids] * reach[:, None]
        # The reference has an open bilinear surface, not a closed solid.
        # Its medium is not clipped by terrain height. Retain the same source
        # domain in both lighting-visibility ablations.
        solar, env, _, valid = scene.source(q, d[ids], reject_interior=False)
        tau_point = tau_rays(o[ids], d[ids], reach, scene.grid, scene.lo, scene.hi)
        weight = np.exp(-tau_point[:, None] * scene.sigma_t) * dt[ids, None]
        integrated_sun, integrated_env = np.zeros((2, len(o), 3)), np.zeros((2, len(o), 3))
        for visibility in (0, 1):
            np.add.at(integrated_sun[visibility], ids, solar[visibility] * weight)
            np.add.at(integrated_env[visibility], ids, env[visibility] * weight)
        below_env, below_sun = np.zeros_like(o), np.zeros_like(o)
        np.add.at(below_env, ids, env[1] * weight * (~valid)[:, None])
        np.add.at(below_sun, ids, solar[1] * weight * (~valid)[:, None])
        first_order = (camera_t * primary + integrated_sun[1] + integrated_env[1]).reshape((*shape, 3))
        frame = a.display(first_order)
        first_order_off = (camera_t * primary + integrated_sun[0] + integrated_env[0]).reshape((*shape, 3))
        metrics = a.score(frame, a.display(first_order_off), no_medium)[0]
        delta_sun = (integrated_sun[0] - integrated_sun[1]).reshape((*shape, 3))
        delta_env = (integrated_env[0] - integrated_env[1]).reshape((*shape, 3))
        lum = np.array([.2126, .7152, .0722])
        report["levels"][label] = {"spacing": spacing, "source_points": len(q), "maximum_segments": int(counts.max()), "below_surface_points_retained_in_both_ablations": int(np.sum(~valid)), "sun_response": float((delta_sun @ lum)[roi].mean()), "environment_response": float((delta_env @ lum)[roi].mean()), "sun_response_roi_rgb": delta_sun[roi].mean(0).tolist(), "environment_response_roi_rgb": delta_env[roi].mean(0).tolist(), "first_order_frame_metrics_diagnostic_only": metrics, "reference_minus_first_order_mean_shadow_rgb": (reference - first_order)[shadow].mean(0).tolist()}
        values[label] = first_order
        report["levels"][label]["below_surface_enabled_environment_response"] = float((below_env.reshape((*shape, 3)) @ lum)[roi].mean())
        report["levels"][label]["below_surface_enabled_sun_response"] = float((below_sun.reshape((*shape, 3)) @ lum)[roi].mean())
        report["levels"][label]["legacy_solid_interior_response_estimate"] = report["levels"][label]["sun_response"] + report["levels"][label]["environment_response"] + report["levels"][label]["below_surface_enabled_environment_response"] + report["levels"][label]["below_surface_enabled_sun_response"]
        np.savez(args.output / (label + ".npz"), camera_t=camera_t.reshape((*shape, 3)), surface_sun=surface_sun, surface_env=surface_env, ss_sun=integrated_sun.reshape((2, *shape, 3)), ss_env=integrated_env.reshape((2, *shape, 3)), first_order=first_order, hit=hit.reshape(shape), hit_t=hit_t)
        print(label, len(q), "points; responses", report["levels"][label]["sun_response"], report["levels"][label]["environment_response"], flush=True)
    change = values["fine"] - values["coarse"]
    report["nested_difference"] = {"mean_absolute_sky_rgb": np.abs(change)[sky].mean(0).tolist(), "mean_absolute_shadow_rgb": np.abs(change)[shadow].mean(0).tolist(), "max_absolute_rgb": np.abs(change).max(axis=(0, 1)).tolist(), "g4a_fraction_change": report["levels"]["fine"]["first_order_frame_metrics_diagnostic_only"]["g4a_fraction"] - report["levels"]["coarse"]["first_order_frame_metrics_diagnostic_only"]["g4a_fraction"], "scope": "Observed coarse/fine numerical difference, not an infinite-resolution error bound or full transport proof"}
    report["duration_seconds"] = time.perf_counter() - started
    report["input_integrity_after_run"] = all(a.sha(ROOT / relative) == digest for relative, digest in report["input_sha256"].items())
    assert report["input_integrity_after_run"]
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["nested_difference"], indent=2), flush=True)


if __name__ == "__main__":
    main()
