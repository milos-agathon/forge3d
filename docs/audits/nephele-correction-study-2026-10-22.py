"""Read-only-input CPU study. --output must name a new directory.

Held-out error is empirical sensitivity, not a transferable confidence bound.
This does not certify a full angular renderer or an uncaptured HEAD.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import importlib.metadata
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from nephele_transport_cpu import Scene

spec = importlib.util.spec_from_file_location("release_analysis", HERE / "nephele-release-analysis-2026-10-22.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def historical_namespace(evidence):
    """Expose intermediates in memory; never rewrite archived source/inputs."""
    source = evidence / "continuation-study/source_operator.py"
    tree = ast.parse(source.read_text().split("blocked, unblocked = run(True)")[0])
    for node in tree.body:
        # Historical source inferred the checkout from the evidence folder's
        # depth. Inputs may now live anywhere; bind only that checkout lookup
        # in memory and keep all archived files/relative evidence reads intact.
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ROOT" for target in node.targets):
            node.value = ast.parse(f"Path({str(ROOT)!r})", mode="eval").body
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            ret = next(n for n in ast.walk(node) if isinstance(n, ast.Return))
            for name, expression in {"current_field": "current", "ground_field": "ground_field"}.items():
                ret.value.keys.append(ast.Constant(name))
                ret.value.values.append(ast.parse(expression, mode="eval").body)
    ast.fix_missing_locations(tree)
    namespace = {"__name__": "archived_cpu_model", "__file__": str(source)}
    previous = sys.argv
    try:
        sys.argv = [str(source), "hammersley"]
        exec(compile(tree, str(source), "exec"), namespace)
    finally:
        sys.argv = previous
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--review-models", required=True, type=Path)
    parser.add_argument("--first-order-study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-domain", choices=("reference-open", "height-rejected"), default="reference-open",
                        help="New first-order sources only; archived higher-order closures retain their historical rules")
    args = parser.parse_args()
    started = time.perf_counter()
    evidence, artifacts, output = args.evidence.resolve(), args.artifacts.resolve(), args.output.resolve()
    args.review_models = args.review_models.resolve()
    args.first_order_study = args.first_order_study.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scene = Scene(ROOT)
    ns = historical_namespace(evidence)
    source_inputs = [Path(__file__), HERE / "nephele_transport_cpu.py", HERE / "nephele-release-analysis-2026-10-22.py", evidence / "continuation-study/source_operator.py", evidence / "first-order/nephele_first_order_ablation.py"]
    source_inputs += sorted((ROOT / "tests/nephele/fixture").iterdir())
    source_inputs += sorted((evidence / "continuation-study").glob("*.npz"))
    source_inputs += sorted((evidence / "first-order").glob("*.npz"))
    source_inputs += sorted(args.review_models.glob("*.npz"))
    source_inputs += sorted(args.review_models.glob("*.py"))
    source_inputs += [args.first_order_study / "analysis.json", args.first_order_study / "fine.npz"]
    source_inputs += [artifacts / f"ci-{run}" / (name + ".npy") for run in (39566600606, 39580065706) for name in ("realtime-rgb", "terrain-occlusion-disabled-rgb", "in-scatter", "transmittance", "medium-disabled-rgb")]
    original_hashes = {analysis.input_key(p): analysis.sha(p) for p in source_inputs if p.is_file()}
    report = {"status": "STEP_3_INCOMPLETE", "input_sha256": original_hashes, "candidates": {}, "calibration": {}, "decomposition": {},
              "environment": {"python": sys.version, "numpy": np.__version__, "numba": importlib.metadata.version("numba"), "native_forge3d_imported": "forge3d" in sys.modules},
              "capture_sha": "68cc04e9ab87de3d90e92e76a8690e016e5d759b", "uncaptured_head_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "contracts": {"environment_directions": "192, nephele_realtime_terrain_trace.wgsl main_nephele_surface_cloudlight", "phase_directions": "one per froxel, nephele_froxel.wgsl cs_nephele_inject_multiple", "phase_steps": "media.grid.z = 64, nephele_froxel.wgsl continuation_transmittance/inject_multiple", "new_optical_depth": "two-node Gauss per trilinear interpolation interval; exact for its cubic ray polynomial", "new_terrain_intersection": "quadratic bilinear-cell roots, bounded only by the terrain AABB and cell boundaries"}}
    regions = {"sky": analysis.load(analysis.gate.FIXTURE, "sky-cloud-mask"), "shadow": analysis.load(analysis.gate.FIXTURE, "terrain-mask") & analysis.load(analysis.gate.FIXTURE, "cloud-shadow-mask")}
    terrain = analysis.load(analysis.gate.FIXTURE, "terrain-mask")
    capture = artifacts / "ci-39580065706"
    rgb, off_rgb, no_medium, observed_i, captured_t = [analysis.load(capture, name) for name in ("realtime-rgb", "terrain-occlusion-disabled-rgb", "medium-disabled-rgb", "in-scatter", "transmittance")]
    beauty, off_beauty = analysis.gate._inverse_display_aces(rgb), analysis.gate._inverse_display_aces(off_rgb)
    reference = analysis.gate._inverse_display_aces(analysis.load(analysis.gate.FIXTURE, "reference-rgb"))
    base = dict(np.load(evidence / "continuation-study/source-operator-hammersley.npz"))
    shifted = dict(np.load(evidence / "continuation-study/source-operator-shifted.npz"))
    baseline_score = analysis.score(rgb, off_rgb, no_medium)[0]
    boundary = (beauty - observed_i) / captured_t

    previous_capture = artifacts / "ci-39566600606"
    previous = analysis.gate._inverse_display_aces(analysis.load(previous_capture, "realtime-rgb"))
    previous_off = analysis.gate._inverse_display_aces(analysis.load(previous_capture, "terrain-occlusion-disabled-rgb"))
    predicted = previous + base["current"] - shifted["current"]
    predicted_off = previous_off + base["current_unblocked"] - shifted["current_unblocked"]
    error, off_error = beauty - predicted, off_beauty - predicted_off
    predicted_score = analysis.score(analysis.display(predicted), analysis.display(predicted_off), no_medium)[0]
    report["calibration"] = {"change": "ac739096 -> 68cc04e9 phase pairing", "predicted_metrics": predicted_score, "observed_metrics": baseline_score,
        "mean_absolute_linear_rgb_error": {r: np.abs(error)[mask].mean(0).tolist() for r, mask in regions.items()},
        "max_absolute_linear_rgb_error": {r: np.abs(error)[mask].max(0).tolist() for r, mask in regions.items()},
        "metric_error": {k: predicted_score[k] - baseline_score[k] for k in baseline_score if isinstance(baseline_score[k], float)},
        "scope": "Empirical held-out error for this sampler change only; not a confidence bound for different physics or reconstruction."}
    np.savez(output / "held-out-calibration.npz", error=error, off_error=off_error, predicted=analysis.display(predicted), predicted_off=analysis.display(predicted_off))

    def record(name, delta, off_delta, settings):
        metrics, de = analysis.score(analysis.display(beauty + delta), analysis.display(off_beauty + off_delta), no_medium)
        sensitivity = [analysis.score(analysis.display(beauty + delta + sign * error), analysis.display(off_beauty + off_delta + sign * off_error), no_medium)[0] for sign in (-1, 1)]
        entry = {"metrics": metrics, "settings": settings, "mean_added_linear_rgb": {key: delta[mask].mean(0).tolist() for key, mask in regions.items()},
                 "held_out_error_sensitivity": {k: [min(s[k] for s in sensitivity), max(s[k] for s in sensitivity)] for k in baseline_score if isinstance(baseline_score[k], float)}}
        report["candidates"][name] = entry
        np.savez(output / (name + ".npz"), rgb=analysis.display(beauty + delta), disabled=analysis.display(off_beauty + off_delta), delta_e2000=de, delta=delta, off_delta=off_delta)
        return entry

    coupled = np.load(evidence / "continuation-study/coupled-surface.npz")["surface"]
    coupled_delta = captured_t * (coupled - boundary)
    coupled_delta[~terrain] = 0
    profiles, seen_profiles = [], {}
    report["equivalent_profile_inputs"] = {}
    for directory, prefix in ((evidence / "continuation-study", "archived"), (args.review_models, "review")):
        for path in sorted(directory.glob("source-operator-*.npz")):
            model = dict(np.load(path))
            digest = hashlib.sha256(b"".join(model[key].tobytes() for key in ("current", "converged", "current_unblocked", "converged_unblocked"))).hexdigest()
            if digest in seen_profiles:
                report["equivalent_profile_inputs"][analysis.input_key(path)] = seen_profiles[digest]
                continue
            seen_profiles[digest] = analysis.input_key(path)
            profiles.append((path, prefix, model))
    for path, prefix, model in profiles:
        for order, surface in itertools.product(("current", "converged"), (False, True)):
            addition = coupled_delta if surface else np.zeros_like(beauty)
            record(prefix + "-" + path.stem + "-" + order + ("-surface" if surface else ""), model[order] - base["current"] + addition, model[order + "_unblocked"] - base["current_unblocked"] + addition,
                   {"family": "archived scalar model", "surface": surface, "order": order})
    print("held-out calibration and complete archived combinations scored", flush=True)

    origins, rays, hit_t, normals = scene.camera(terrain.shape)
    actual_hit = np.isfinite(hit_t)
    report["geometry_checks"] = {"primary_hit_mask_mismatches": int(np.sum(actual_hit != analysis.load(analysis.gate.FIXTURE, "reference-terrain-hit"))), "primary_hit_count": int(actual_hit.sum())}
    points = origins + rays * np.where(actual_hit, hit_t, 0)[..., None]
    surf_sun, surf_env = scene.surface(points[actual_hit], normals[actual_hit])
    surface_sun, surface_environment = np.zeros((2, *beauty.shape)), np.zeros((2, *beauty.shape))
    surface_sun[:, actual_hit], surface_environment[:, actual_hit] = surf_sun, surf_env
    print("physical surface evaluated", flush=True)

    depth = ns["DEPTH"]
    edges = ns["NEAR"] * np.exp(ns["LOG"] * np.arange(depth + 1) / depth)
    distances = ns["NEAR"] * np.exp(ns["LOG"] * (np.arange(depth) + .5) / depth)
    scene_distance = np.where(actual_hit, hit_t, ns["FAR"])
    step = np.maximum(np.minimum(edges[1:, None, None], scene_distance) - edges[:-1, None, None], 0)
    # Actual executed segment midpoint, including the clipped last segment.
    # A froxel-centre depth can lie behind the primary surface for that pixel.
    pixel_points = origins[None] + rays[None] * (edges[:-1, None, None] + .5 * step)[..., None]
    inside = np.all((pixel_points >= scene.lo) & (pixel_points <= scene.hi), axis=-1) & (step > 0)
    sun_pixel = np.zeros((2, depth, *beauty.shape))
    env_pixel = np.zeros_like(sun_pixel)
    reject_interior = args.source_domain == "height-rejected"
    solar, env, _, valid = scene.source(pixel_points[inside], np.broadcast_to(rays, pixel_points.shape)[inside], reject_interior=reject_interior)
    sun_pixel[:, inside], env_pixel[:, inside] = solar, env
    print("actual pixel sources evaluated", int(inside.sum()), "points x 192 directions", flush=True)
    solar_rejected, env_rejected, _, valid_air_f = scene.source(ns["world"], ns["rays"])
    solar_interior, env_interior, rho_f, _ = scene.source(ns["world"], ns["rays"], reject_interior=False)
    solar_f, env_f = (solar_rejected, env_rejected) if reject_interior else (solar_interior, env_interior)
    valid_f = (valid_air_f if reject_interior else np.ones_like(valid_air_f)) & (rho_f > 0)
    print("actual froxel sources evaluated", flush=True)

    py, px = np.indices(terrain.shape)
    centre = np.stack([(px + .5) / terrain.shape[1] * ns["VISIBLE"] + .5, (py + .5) / terrain.shape[0] * ns["VISIBLE"] + .5], -1)
    xy = np.floor(centre).astype(int)
    frac = centre - xy

    def coarse(field, tented):
        if tented:
            field = ns["tent"](field, valid_f)
        value = np.zeros((depth, *beauty.shape))
        for z in range(depth):
            for dy, dx in itertools.product((0, 1), repeat=2):
                j = xy[..., 0] + dx + ns["WIDTH"] * (xy[..., 1] + dy + ns["WIDTH"] * z)
                weight = (frac[..., 0] if dx else 1 - frac[..., 0]) * (frac[..., 1] if dy else 1 - frac[..., 1])
                value[z] += field[j] * weight[..., None]
        return value

    extinction = coarse(ns["extinction"], False)
    attenuation = np.exp(-extinction * step[..., None])
    prefix = np.concatenate([np.ones((1, *beauty.shape)), np.cumprod(attenuation, axis=0)[:-1]], 0)
    weight = prefix * np.divide(1 - attenuation, extinction, out=np.broadcast_to(step[..., None], extinction.shape).copy(), where=extinction > 0)
    integrate = lambda values: np.sum(values * weight, axis=0)
    old, old_off = ns["run"](True), ns["run"](False)
    old_sun_field = old["sun_field"]
    active = ns["active"]
    direct_off = scene.sigma_s * ns["rho"][active, None] * scene.sun_l * ns["camera_sun_t"][active]
    old_sun_off = np.zeros_like(old_sun_field)
    old_sun_off[active] = direct_off.astype(np.float16).astype(float) * ns["hg"](ns["incident"] @ scene.sun).astype(np.float16).astype(float)[:, None]
    old_sun_off = ns["tent"](old_sun_off)
    old_env = old["single_field"] - old_sun_field - old["ground_field"]
    old_env_off = old_off["single_field"] - old_sun_off - old_off["ground_field"]
    old_direct = [integrate(coarse(old_sun_off, False)), integrate(coarse(old_sun_field, False))]
    old_environment = [integrate(coarse(old_env_off, False)), integrate(coarse(old_env, False))]

    source_terms = {}
    for pixel, tented in itertools.product((False, True), repeat=2):
        sun_terms, env_terms = [], []
        for visibility in (0, 1):
            sun_terms.append(integrate(sun_pixel[visibility] if pixel else coarse(solar_f[visibility], tented)))
            env_terms.append(integrate(env_pixel[visibility] if pixel else coarse(env_f[visibility], tented)))
        source_terms[pixel, tented] = sun_terms, env_terms
    np.savez(output / "source-components.npz", sun_pixel=sun_pixel, env_pixel=env_pixel, solar_f=solar_f, env_f=env_f, env_interior=env_interior, valid_f=valid_f, integration_weight=weight, rho_f=rho_f)
    for angular, surface, pixel, tented in itertools.product((False, True), repeat=4):
        solar_new, env_new = source_terms[pixel, tented]
        delta, off_delta = solar_new[1] - old_direct[1], solar_new[0] - old_direct[0]
        if angular:
            delta += env_new[1] - old_environment[1]
            off_delta += env_new[0] - old_environment[0]
        if surface:
            env_delta = captured_t * (surface_environment[1] - scene.albedo * scene.env_l)
            env_delta[~terrain] = 0
            delta += env_delta
            off_delta += env_delta
        settings = dict(angular_environment=angular, surface_environment=surface, pixel_source=pixel, tent=tented, camera_t="captured unchanged", source_domain=args.source_domain, higher_orders="unchanged captured/model residual; not a reference-domain higher-order solution")
        name = "factorial-" + "-".join(f"{k}{int(v)}" for k, v in (("angular", angular), ("surface", surface), ("pixel", pixel), ("tent", tented)))
        record(name, delta, off_delta, settings)

    # Combine the actual-position correction with every archived continuation,
    # secondary-boundary and tent profile, with/without coupled surface return.
    for path, prefix_name, model in profiles:
        angular = "environment-quadrature" in path.stem
        tented = "tented" in path.stem or not angular
        sun_coarse, env_coarse = source_terms[False, tented]
        sun_actual, env_actual = source_terms[True, tented]
        correction = sun_actual[1] - sun_coarse[1]
        off_correction = sun_actual[0] - sun_coarse[0]
        if angular:
            correction += env_actual[1] - env_coarse[1]
            off_correction += env_actual[0] - env_coarse[0]
        for order, surface in itertools.product(("current", "converged"), (False, True)):
            addition = coupled_delta if surface else np.zeros_like(beauty)
            record(prefix_name + "-" + path.stem + "-" + order + ("-surface" if surface else "") + "-actual-pixel-source", model[order] - base["current"] + addition + correction, model[order + "_unblocked"] - base["current_unblocked"] + addition + off_correction,
                   {"family": "calibrated scalar continuation plus actual first-order pixel term", "angular_environment": angular, "surface": surface, "order": order, "pixel_source": True, "tent": tented, "camera_t": "captured unchanged", "source_domain": args.source_domain, "higher_orders": "historical closure; not reference-domain transport"})

    # Visibility x air, with the 192-direction estimator and production tent
    # held fixed. Both orderings and their interaction close algebraically.
    air_fields = {False: env_interior, True: env_rejected}
    factorial_values = {}
    for air, visibility in itertools.product((False, True), repeat=2):
        source_field = ns["tent"](air_fields[air][int(visibility)], ns["rho"] > 0)
        factorial_values[air, visibility] = integrate(coarse(source_field, False))
    lum = np.array([.2126, .7152, .0722])
    roi = analysis.load(analysis.gate.FIXTURE, "godray-roi-mask")
    direct_responses = {air: factorial_values[air, False] - factorial_values[air, True] for air in (False, True)}
    report["visibility_valid_air_factorial"] = {
        "estimator": "192 directions, exact trilinear optical depth and bilinear terrain roots in every cell", "tent": "same production tent with density support in every cell", "orders": {},
        "interaction_mean_shadow_rgb": (factorial_values[True, True] - factorial_values[True, False] - factorial_values[False, True] + factorial_values[False, False])[regions["shadow"]].mean(0).tolist(),
        "g4b_direct_environment": {str(air): {"roi_mean_linear_rgb": value[roi].mean(0).tolist(), "response": float((value @ lum)[roi].mean())} for air, value in direct_responses.items()},
        "g4b_change_due_to_air_at_fixed_visibility_estimator": float(((direct_responses[True] - direct_responses[False]) @ lum)[roi].mean()),
    }
    for label, ordering in {"visibility_then_air": [((False, False), (False, True)), ((False, True), (True, True))], "air_then_visibility": [((False, False), (True, False)), ((True, False), (True, True))]}.items():
        entries, closure_delta = [], np.zeros_like(beauty)
        for start_cell, end_cell in ordering:
            delta = factorial_values[end_cell] - factorial_values[start_cell]
            closure_delta += delta
            start_score = analysis.score(analysis.display(beauty + factorial_values[start_cell] - factorial_values[False, False]), off_rgb, no_medium)[0]
            end_score = analysis.score(analysis.display(beauty + factorial_values[end_cell] - factorial_values[False, False]), off_rgb, no_medium)[0]
            entries.append({"start": list(start_cell), "end": list(end_cell), "mean_shadow_linear_rgb": delta[regions["shadow"]].mean(0).tolist(), "g4a_fraction_change": end_score["g4a_fraction"] - start_score["g4a_fraction"], "signed_roi_linear_luminance_change": float((delta @ lum)[roi].mean())})
        report["visibility_valid_air_factorial"]["orders"][label] = {"contributions": entries, "max_closure_error": float(np.abs(closure_delta - (factorial_values[True, True] - factorial_values[False, False])).max())}

    # Algebraic closure includes an explicit unidentifiable transport remainder;
    # it is never mislabeled as measured first-order environment or camera light.
    first_data = dict(np.load(args.first_order_study / "fine.npz"))
    first_sun, first_env = np.zeros_like(beauty), np.zeros_like(beauty)
    first_sun[first_data["hit"]] = first_data["surface_sun"][1]
    first_env[first_data["hit"]] = first_data["surface_env"][1]
    first = dict(T_cam=first_data["camera_t"], surf_sun=first_sun, surf_env=first_env, ss_sun=first_data["ss_sun"][1], ss_env=first_data["ss_env"][1], background=np.where(first_data["hit"][..., None], 0., scene.env_l))
    lateral = source_terms[False, True][0][1] + source_terms[False, True][1][1] - source_terms[True, True][0][1] - source_terms[True, True][1][1]
    cloud_path = evidence / "first-order/surface-cloudlight.npy"
    cloud = np.load(cloud_path)
    report["input_sha256"][analysis.input_key(cloud_path)] = analysis.sha(cloud_path)
    sun_estimate = boundary - scene.albedo * scene.env_l - cloud
    sun_estimate[~terrain] = 0
    env_estimate = np.broadcast_to(scene.albedo * scene.env_l, beauty.shape).copy()
    env_estimate[~terrain] = 0
    contributions = {
        "camera_path_in_scatter_excluding_lateral": observed_i - first["ss_sun"] - first["ss_env"] - lateral,
        "camera_transmittance": (captured_t - first["T_cam"]) * (first["surf_sun"] + first["surf_env"] + first["background"]),
        "surface_sun_estimate_after_separating_archived_cloudlight": captured_t * (sun_estimate - first["surf_sun"]),
        "surface_environment": captured_t * (env_estimate - first["surf_env"]),
        "lateral_first_order_source_reconstruction": lateral,
    }
    residual = beauty - reference
    contributions["unresolved_higher_order_geometry_quantization_and_reference_sampling"] = residual - sum(contributions.values())
    closure = sum(contributions.values())
    report["decomposition"] = {"scope": "All five requested terms with explicit first-order/source-partition assumptions. Exact bookkeeping includes a named unresolved higher-order/reference/primary-cloudlight remainder, not an independent measurement of each physical path family.",
        "terms": {name: {"mean_shadow_linear_rgb": value[regions["shadow"]].mean(0).tolist(), "g4a_after_removing_term": analysis.score(analysis.display(beauty - value), off_rgb, no_medium)[0]["g4a_fraction"]} for name, value in contributions.items()},
        "total_residual_mean_shadow_rgb": residual[regions["shadow"]].mean(0).tolist(), "max_closure_error": float(np.abs(closure - residual).max()),
        "lateral_source_only": {"mean_shadow_linear_rgb": (source_terms[True, True][0][1] + source_terms[True, True][1][1] - source_terms[False, True][0][1] - source_terms[False, True][1][1])[regions["shadow"]].mean(0).tolist(), "camera_t_changed": False, "sun_estimator_identical_in_ablations": True}}
    candidates = report["candidates"]
    for term in report["decomposition"]["terms"].values():
        term["g4a_fraction_change_after_removing_term"] = term["g4a_after_removing_term"] - baseline_score["g4a_fraction"]
    best_name = max(candidates, key=lambda name: candidates[name]["metrics"]["g4a_fraction"])
    report["best_g4a"] = {"name": best_name, **candidates[best_name]}
    report["all_five_passing_nominal_models"] = [name for name, entry in candidates.items() if entry["metrics"]["all_five_pass"]]
    report["first_order_contract_study"] = json.loads((args.first_order_study / "analysis.json").read_text())
    report["source_domain_policy"] = {"controlled_source_comparison": args.source_domain, "reference_first_order": "untruncated medium on either side of the open bilinear surface; same domain for both lighting ablations", "higher_orders": "archived closures retain historical domain rules; hybrids do not certify full reference-domain physics", "production_decision": "NOT_MADE: terrain_occlusion_in_media=False must control any new visibility term; domain changes require explicit transport justification before implementation"}
    report["limitations"] = ["No full-angular higher-order solution or quantified error for changing that closure", "Archived candidates retain their historical, unqualified numerical parameters; fresh first-order terms use named contracts and the recorded nested comparison", "Geometry uses the reference bilinear surface, not runtime CSM; no CSM buffer was captured", "Held-out sensitivity cannot certify unrelated changes", "HEAD is uncaptured; source inspection cannot prove GPU equivalence", "Step 2 independent physical attribution and step 3 exact full candidate remain open; no release blocker or impossibility is established"]
    for relative, digest in original_hashes.items():
        assert analysis.sha(ROOT / relative) == digest, f"analysis modified input {relative}"
    report["input_integrity_after_run"] = "all hashes unchanged"
    report["duration_seconds"] = time.perf_counter() - started
    (output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "candidates": len(candidates), "best_g4a": report["best_g4a"], "duration_seconds": report["duration_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
