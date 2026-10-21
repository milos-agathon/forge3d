"""CPU-only measurements of the cloudlight candidate; never changes gate policy.

Reuse the already measured, owner-accepted first-order reference response.
The original gate verifier still rejects this candidate at Gate 3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--baseline-artifact-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "tests"))
    from _deltae import delta_e_2000, srgb_to_lab
    from _ssim import ssim

    artifact, baseline = args.artifact_dir, args.baseline_artifact_dir
    load = lambda directory, name: np.load(directory / (name + ".npy"), allow_pickle=False)
    reference = load(artifact, "reference-rgb")
    realtime = load(artifact, "realtime-rgb")
    old_realtime = load(baseline, "realtime-rgb")
    disabled = load(artifact, "terrain-occlusion-disabled-rgb")
    shadow = load(artifact, "cloud-shadow-mask") & load(artifact, "terrain-mask")
    sky = load(artifact, "sky-cloud-mask")
    roi = load(artifact, "godray-roi-mask")
    ys, xs = np.nonzero(roi)
    crop = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if not roi[crop].all():
        raise ValueError("ROI must be rectangular")
    fixed = [path.name for path in (root / "tests/nephele/fixture").iterdir()
             if path.is_file() and (artifact / path.name).is_file()]
    for name in fixed:
        if (artifact / name).read_bytes() != (baseline / name).read_bytes():
            raise ValueError(f"frozen input changed: {name}")

    # Reuse round2's numerical sRGB + ACES inverse, unchanged from measure_gate4.py.
    aces_x = np.linspace(0.0, 64.0, 2_000_001)
    aces_y = np.clip((aces_x * (2.51 * aces_x + 0.03)) /
                     (aces_x * (2.43 * aces_x + 0.59) + 0.14), 0, 1)

    def linear(u8):
        value = u8.astype(np.float64) / 255.0
        display = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
        return np.interp(display, aces_y, aces_x)

    accepted_path = root / "tests/nephele/gate4-policy.json"
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    accepted_reference = accepted["occlusion_response"]["reference"]
    reference_change = accepted_reference["mean_linear_luminance_change"]
    luminance = np.array([0.2126, 0.7152, 0.0722])
    change = float(((linear(disabled) - linear(realtime)) @ luminance)[crop].mean())
    delta = delta_e_2000(srgb_to_lab(realtime), srgb_to_lab(reference))
    lab = srgb_to_lab(realtime) - srgb_to_lab(reference)
    report = {
        "head_sha": (artifact / "checked-out-head.txt").read_text(encoding="utf-8-sig").strip(),
        "gate_verifier_result": "Diagnostic measurements; consult the CI run for its acceptance verdict",
        "frozen_input_files_byte_identical": sorted(fixed),
        "sky_fraction_delta_e2000_lt_2_5": float((delta[sky] < 2.5).mean()),
        "godray_ssim": ssim(reference[crop], realtime[crop], data_range=255.0),
        "g4a_fraction_delta_e2000_lt_2": float((delta[shadow] < 2.0).mean()),
        "shadow_mean_lab_error": lab[shadow].mean(axis=0).tolist(),
        "sky_mean_lab_error": lab[sky].mean(axis=0).tolist(),
        "ablation_ssim": ssim(realtime[crop], disabled[crop], data_range=255.0),
        "delta_realtime_mean_linear_luminance": change,
        "approved_reference_delta": reference_change,
        "ablation_change_ratio": change / reference_change,
        "ablation_relative_error": abs(change - reference_change) / abs(reference_change),
        "sky_mean_added_linear_rgb": (linear(realtime) - linear(old_realtime))[sky].mean(axis=0).tolist(),
        "shadow_mean_added_linear_rgb": (linear(realtime) - linear(old_realtime))[shadow].mean(axis=0).tolist(),
        "optical_depth_byte_identical": (artifact / "optical-depth.npy").read_bytes() == (baseline / "optical-depth.npy").read_bytes(),
        "transmittance_byte_identical": (artifact / "transmittance.npy").read_bytes() == (baseline / "transmittance.npy").read_bytes(),
        "input_sha256": {name: hashlib.sha256((artifact / (name + ".npy")).read_bytes()).hexdigest()
                         for name in ("reference-rgb", "realtime-rgb", "terrain-occlusion-disabled-rgb",
                                      "sky-cloud-mask", "cloud-shadow-mask", "terrain-mask", "godray-roi-mask")},
        "accepted_reference_report_sha256": accepted_reference["source_git_blob_sha256"],
        "reference_policy_sha256": hashlib.sha256(accepted_path.read_bytes()).hexdigest(),
        "measurement_tool_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "measurement_tool_hash_convention": "LF-normalized Git source text",
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("input_sha256", "frozen_input_files_byte_identical")}, indent=2))


if __name__ == "__main__":
    main()
