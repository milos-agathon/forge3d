"""CPU-only NEPHELE capture analysis. Never renders or changes acceptance inputs.

Download the four artifacts named in the release prompt, then run with
--artifacts .release-evidence/2026-10-22. Optional --models points at the
historical continuation-study directory and --first-order at first-order/.
All predictions are diagnostics; only the NVIDIA CI lane establishes a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import nephele_acceptance as gate

RUNS = (39517423981, 39545456967, 39566600606, 39580065706)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_key(path):
    """Repository-relative when possible; otherwise an explicit absolute path."""
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def load(directory, name):
    return np.load(directory / (name + ".npy"), allow_pickle=False)


def display(value):
    value = np.maximum(value, 0)
    mapped = np.clip(value * (2.51 * value + .03) /
                     (value * (2.43 * value + .59) + .14), 0, 1)
    srgb = np.where(mapped <= .0031308, 12.92 * mapped,
                    1.055 * mapped ** (1 / 2.4) - .055)
    return np.rint(srgb * 255).astype(np.uint8)


def score(frame, disabled, no_medium):
    reference = load(gate.FIXTURE, "reference-rgb")
    de = gate.delta_e_2000(gate.srgb_to_lab(frame), gate.srgb_to_lab(reference))
    sky = load(gate.FIXTURE, "sky-cloud-mask")
    terrain = load(gate.FIXTURE, "terrain-mask")
    shadow = terrain & load(gate.FIXTURE, "cloud-shadow-mask")
    medium = gate.delta_e_2000(gate.srgb_to_lab(no_medium), gate.srgb_to_lab(frame))
    response = gate._occlusion_response(*gate._roi(frame, disabled))
    policy = gate._load("gate4-policy.json", ROOT / "tests/nephele")
    expected = policy["occlusion_response"]["reference"]["mean_linear_luminance_change"]
    metrics = {
        "b1_fraction": float((de[sky] < 2.5).mean()),
        "shaft_ssim": gate.ssim(*gate._roi(reference, frame), data_range=255.),
        "g4a_fraction": float((de[shadow] < 2).mean()),
        "medium_changed_fraction": float((medium[terrain] > 5).mean()),
        "g4b_response": response,
        "g4b_relative_error": abs(response - expected) / expected,
    }
    metrics["margins"] = {
        "b1_fraction": metrics["b1_fraction"] - .95,
        "shaft_ssim": metrics["shaft_ssim"] - .95,
        "g4a_fraction": metrics["g4a_fraction"] - policy["aggregation"]["minimum_fraction"],
        "medium_changed_fraction": metrics["medium_changed_fraction"] - .10,
        "g4b_relative_error": policy["occlusion_response"]["maximum_relative_error"] - metrics["g4b_relative_error"],
    }
    metrics["all_five_pass"] = all(v >= 0 for v in metrics["margins"].values()) and metrics["shaft_ssim"] > .95
    return metrics, de


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--first-order", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sky = load(gate.FIXTURE, "sky-cloud-mask")
    terrain = load(gate.FIXTURE, "terrain-mask")
    shadow = terrain & load(gate.FIXTURE, "cloud-shadow-mask")
    reference = load(gate.FIXTURE, "reference-rgb")
    reference_linear = gate._inverse_display_aces(reference)
    ref_i = load(gate.FIXTURE, "reference-in-scatter").astype(float)
    ref_t = load(gate.FIXTURE, "reference-transmittance").astype(float)
    environment = gate._load("atmosphere.json")["environment_intensity"]
    report = {"kind": "CPU-only existing-capture measurements and unfitted predictions; no native execution",
              "numpy": np.__version__, "tool_sha256": sha(Path(__file__)),
              "input_sha256": {}, "captures": {}, "candidates": {},
              "reference_encoding": {
                  "beauty": "uint8 IEC sRGB after ACES fitted, exposure 1",
                  "aovs": "linear RGB in-scatter, dimensionless transmittance and optical depth",
                  "reference_in_scatter_semantics": "Total path radiance when collision_count > 0, including paths that first hit the primary terrain and scatter later (media_reference.rs). Not the same terrain path partition as the runtime camera-segment source integral.",
                  "reference_optical_depth_semantics": "Negative log of Monte Carlo mean transmittance; runtime integrates extinction deterministically.",
                  "sky_composite_mean_abs_linear_error": np.abs(reference_linear - ref_i - ref_t * environment)[sky].mean(0).tolist(),
                  "sky_composite_byte_agreement": float(np.mean(np.all(display(ref_i + ref_t * environment)[sky] == reference[sky], axis=-1))),
              }}
    for path in sorted(gate.FIXTURE.iterdir()):
        if path.suffix in (".npy", ".json"):
            report["input_sha256"][input_key(path)] = sha(path)
    report["input_sha256"]["tests/nephele/gate4-policy.json"] = sha(ROOT / "tests/nephele/gate4-policy.json")
    for name in ("scripts/nephele_acceptance.py", "tests/_deltae.py", "tests/_ssim.py"):
        report["input_sha256"][name] = sha(ROOT / name)
    data = []
    for run in RUNS:
        directory = args.artifacts / f"ci-{run}"
        rgb, off, no_medium, ins, trans, tau = [load(directory, n) for n in
            ("realtime-rgb", "terrain-occlusion-disabled-rgb", "medium-disabled-rgb", "in-scatter", "transmittance", "optical-depth")]
        linear = gate._inverse_display_aces(rgb)
        metrics, delta = score(rgb, off, no_medium)
        assert np.array_equal(display(linear), rgb), "presentation inverse must round-trip exactly"
        captures = [json.loads((directory / f"run-{label}.json").read_text(encoding="utf-8-sig")) for label in ("a", "b")]
        head = (directory / "checked-out-head.txt").read_text(encoding="utf-8-sig").strip()
        assert all(c["diagnostics"]["source_revision"] == head for c in captures)
        cases = ET.parse(directory / "junit.xml").getroot().findall(".//testcase")
        entry = {"head": head, "url": f"https://github.com/milos-agathon/forge3d/actions/runs/{run}",
                 "adapter": captures[0]["diagnostics"]["adapter"], "backend": captures[0]["diagnostics"]["backend"],
                 "counts": {outcome: sum(("failed" if case.find("failure") is not None else "skipped" if case.find("skipped") is not None else "passed") == outcome for case in cases) for outcome in ("passed", "failed", "skipped")},
                 "metrics_under_current_unchanged_policy": metrics,
                 "encoding_checks": {"inverse_round_trip_byte_identical": True,
                     "beer_mean_abs_error": np.abs(trans - np.exp(-tau)).mean(0).mean(0).tolist(),
                     "sky_composite_mean_abs_linear_error": np.abs(linear - ins - trans * environment)[sky].mean(0).tolist(),
                     "sky_composite_byte_agreement": float(np.mean(np.all(display(ins + trans * environment)[sky] == rgb[sky], axis=-1)))},
                 "regions": {}}
        for region, mask in (("sky", sky), ("shadow", shadow)):
            entry["regions"][region] = {
                "in_scatter_mean_error_rgb": (ins - ref_i)[mask].mean(0).tolist(),
                "transmittance_mean_error_rgb": (trans - ref_t)[mask].mean(0).tolist(),
                "optical_depth_mean_error_rgb": (tau - load(gate.FIXTURE, "reference-optical-depth"))[mask].mean(0).tolist(),
                "beauty_mean_linear_error_rgb": (linear - reference_linear)[mask].mean(0).tolist(),
            }
        if data:
            old = data[-1]
            previous_pass = old["delta"][sky] < 2.5
            entry["from_previous"] = {
                "sky_new_failures": int(np.sum(previous_pass & (delta[sky] >= 2.5))),
                "sky_new_passes": int(np.sum(~previous_pass & (delta[sky] < 2.5))),
                "transmittance_byte_identical": np.array_equal(trans, old["trans"]),
                "optical_depth_byte_identical": np.array_equal(tau, old["tau"]),
                "sky_in_scatter_delta_rgb": (ins - old["ins"])[sky].mean(0).tolist(),
                "sky_beauty_delta_rgb": (linear - old["linear"])[sky].mean(0).tolist(),
                "sky_b1_replacing_only_in_scatter": score(display(linear - ins + old["ins"]), off, no_medium)[0]["b1_fraction"],
            }
        for path in directory.iterdir():
            if path.suffix == ".npy" or path.name in ("run-a.json", "run-b.json", "checked-out-head.txt", "junit.xml"):
                report["input_sha256"][input_key(path)] = sha(path)
        np.save(args.output / f"delta-e2000-{run}.npy", delta)
        report["captures"][str(run)] = entry
        data.append(dict(rgb=rgb, off=off, no_medium=no_medium, ins=ins.astype(float), trans=trans.astype(float), tau=tau, linear=linear, delta=delta))
    current = data[-1]

    def candidate(name, delta, off_delta=None, limitation=None):
        frame = display(current["linear"] + delta)
        off_frame = display(gate._inverse_display_aces(current["off"]) + (delta if off_delta is None else off_delta))
        metrics, de = score(frame, off_frame, current["no_medium"])
        entry = {"metrics": metrics, "mean_added_shadow_linear_rgb": delta[shadow].mean(0).tolist(),
                 "mean_added_sky_linear_rgb": delta[sky].mean(0).tolist(), "limitation": limitation}
        report["candidates"][name] = entry
        np.savez(args.output / (name + ".npz"), rgb=frame, disabled=off_frame, delta_e2000=de)

    # Controlled AOV substitutions diagnose components; they are not candidate fixes.
    boundary = np.divide(current["linear"] - current["ins"], current["trans"], out=np.zeros_like(current["linear"]), where=current["trans"] > 0)
    candidate("diagnostic-reference-tagged-radiance", ref_i - current["ins"], limitation="Frozen reference AOV substitution. Terrain path partitions differ, so this is not a physically valid replacement or a runtime correction. Disabled response is not predicted.")
    candidate("diagnostic-reference-camera-transmittance", (ref_t - current["trans"]) * boundary, limitation="Frozen reference AOV substitution; diagnostic only. Disabled transmittance is unchanged by terrain-light occlusion.")
    report["boundary"] = {"sky_mean_rgb": boundary[sky].mean(0).tolist(), "shadow_mean_rgb": boundary[shadow].mean(0).tolist()}
    if args.models and args.first_order:
        first = dict(np.load(args.first_order / "first-order.npz"))
        baseline = dict(np.load(args.models / "source-operator-hammersley.npz"))
        for path in (args.models / "source_operator.py", args.first_order / "nephele_first_order_ablation.py"):
            report["input_sha256"][input_key(path)] = sha(path)
        report["model_capture_error"] = {region: np.abs(baseline["current"] - current["ins"])[mask].mean(0).tolist() for region, mask in (("sky", sky), ("shadow", shadow))}
        for path in sorted(args.models.glob("source-operator-*.npz")):
            model = dict(np.load(path))
            report["input_sha256"][input_key(path)] = sha(path)
            for key in ("current", "converged"):
                candidate(path.stem + "-" + key, model[key] - baseline["current"], model[key + "_unblocked"] - baseline["current_unblocked"], "Historical CPU transport model; scalar angular closure and approximate geometry/CSM, not an exact full-angular prediction.")
        for path in args.first_order.glob("*.npz"):
            report["input_sha256"][input_key(path)] = sha(path)
        t = current["trans"]
        sun_delta = t * (first["surf_sun"] - (boundary - np.asarray(gate._load("material.json")["albedo"]) * environment))
        sun_delta[~terrain] = 0
        candidate("surface-sun-first-order", sun_delta, limitation="First-order surface decomposition assumes the current environment boundary is albedo times environment intensity. Cloudlight remains in that estimate.")
        environment_delta = t * (first["surf_env"] - np.asarray(gate._load("material.json")["albedo"]) * environment)
        environment_delta[~terrain] = 0
        candidate("surface-environment-attenuation", environment_delta, limitation="Historical 96-direction first-order estimate. Existing 192-direction production term is already present, so this is a diagnostic substitution, not an additive physical fix.")
        cloud_blob = subprocess.check_output(["git", "show", "d5507249:docs/audits/nephele-bounded-round-2026-10-01/round2/first-order/surface-cloudlight.npy"], cwd=ROOT)
        cloud = np.load(io.BytesIO(cloud_blob), allow_pickle=False)
        report["git_blob_sha256"] = {"d5507249:docs/audits/nephele-bounded-round-2026-10-01/round2/first-order/surface-cloudlight.npy": hashlib.sha256(cloud_blob).hexdigest()}
        # Separate the archived physical surface-sun scattering from the
        # observed boundary before estimating primary sun. No fitted gain.
        surface_sun = boundary - np.asarray(gate._load("material.json")["albedo"]) * environment - cloud
        sun_delta = t * (first["surf_sun"] - surface_sun)
        sun_delta[~terrain] = 0
        candidate("surface-sun-separated-cloudlight", sun_delta, limitation="Diagnostic first-order sun replacement after subtracting the archived 192-direction cloudlight term; not an exact capture of the runtime cloudlight buffer or CSM.")
        first_order_scatter = first["ss_sun"] + first["ss_env"]
        candidate("diagnostic-first-order-camera-scatter", first_order_scatter - current["ins"], limitation="First-order camera source only, deliberately omits higher orders; not a complete candidate or an attribution of all reference error.")
        report["first_order_partition"] = {
            "shadow_reference_beauty_minus_first_order_rgb": (reference_linear - first["first_order"])[shadow].mean(0).tolist(),
            "shadow_reference_tagged_radiance_minus_first_order_camera_scatter_rgb": (ref_i - first_order_scatter)[shadow].mean(0).tolist(),
            "warning": "Neither residual uniquely separates camera scattering from surface-first multiple scattering. The archived captures do not contain that path decomposition.",
        }
        sun_ablation_path = args.models / "sun-ablation.npz"
        sun_ablation = dict(np.load(sun_ablation_path))
        report["input_sha256"][input_key(sun_ablation_path)] = sha(sun_ablation_path)
        roi = load(gate.FIXTURE, "godray-roi-mask")
        luminance = np.array([.2126, .7152, .0722])
        ref_sun_delta = sun_ablation["reference_unoccluded"] - sun_ablation["reference_sun"]
        coarse_sun_delta = sun_ablation["uncoarse"] - sun_ablation["coarse"]
        total_response = score(current["rgb"], current["off"], current["no_medium"])[0]["g4b_response"]
        report["g4b_first_order_split"] = {
            "reference_sun_mean_rgb": ref_sun_delta[roi].mean(0).tolist(),
            "coarse_sun_mean_rgb": coarse_sun_delta[roi].mean(0).tolist(),
            "reference_sun_response": float((ref_sun_delta @ luminance)[roi].mean()),
            "coarse_sun_response": float((coarse_sun_delta @ luminance)[roi].mean()),
            "observed_total_response": total_response,
            "observed_minus_coarse_sun": total_response - float((coarse_sun_delta @ luminance)[roi].mean()),
            "warning": "The remainder includes environment, terrain-reflected return, multiple scattering and display quantization; it is not a separately measured environment-only capture.",
        }
        coupled_path = args.models / "coupled-surface.npz"
        if coupled_path.exists():
            coupled = dict(np.load(coupled_path))
            report["input_sha256"][input_key(coupled_path)] = sha(coupled_path)
            surface_delta = t * (coupled["surface"] - boundary)
            surface_delta[~terrain] = 0
            candidate("surface-coupled-environment", surface_delta, limitation="192-direction historical surface-return estimate with scalar angular source; ablation applies the same primary boundary change.")
            angular = dict(np.load(args.models / "source-operator-hammersley-environment-quadrature.npz"))
            for key in ("current", "converged"):
                candidate("angular-environment-and-surface-" + key, surface_delta + angular[key] - baseline["current"], surface_delta + angular[key + "_unblocked"] - baseline["current_unblocked"], "Combined 192-direction environment visibility, valid-air source, scalar continuation, and physical surface return. Approximate model, not full-angular proof.")
    fixtures = sorted(gate.FIXTURE.glob("*.npy"))
    for path in fixtures:
        old = subprocess.check_output(["git", "show", "ac739096:" + path.relative_to(ROOT).as_posix()], cwd=ROOT)
        assert old == path.read_bytes(), f"forbidden fixture change: {path}"
    report["canonical_npy_files_identical_to_ac739096"] = len(fixtures)
    report["acceptance_diff_from_ac739096"] = subprocess.check_output(["git", "diff", "ac739096", "--", "scripts/nephele_acceptance.py", "tests/nephele/gate4-policy.json", "tests/test_nephele_physical.py", "tests/_deltae.py", "tests/_ssim.py"], cwd=ROOT, text=True)
    assert not report["acceptance_diff_from_ac739096"]
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"captures": {k: v["metrics_under_current_unchanged_policy"] for k, v in report["captures"].items()}, "candidates": {k: v["metrics"] for k, v in report["candidates"].items()}}, indent=2))


if __name__ == "__main__":
    main()
