"""CPU-only Gate 4 diagnosis; does not change the frozen reference or gate.

Run from the repository root:
  python docs/audits/nephele-gate4-measurement-2026-10-21/measure_gate4.py \
    --artifact-dir <downloaded CI artifact> --scratch <output directory> \
    --output docs/audits/nephele-gate4-measurement-2026-10-21/measurement.json

The historical first-order tool runs with its original quadrature and steps.
Only parallel camera single-scatter accumulators omit terrain visibility.
Their difference is added to the decoded 640-spp reference, so this is a
first-order estimate of the reference ablation, not a new path-traced reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "tests"))
    from _deltae import delta_e_2000, srgb_to_lab
    from _ssim import ssim

    source_path = root / "docs/audits/nephele-bounded-round-2026-10-01/round2/nephele_first_order.py"
    source = source_path.read_text(encoding="utf-8")
    replacements = {
        "ss_env = np.zeros(dirs.shape)":
            "ss_env = np.zeros(dirs.shape)\nss_sun_unoccluded = np.zeros(dirs.shape)\nss_env_unoccluded = np.zeros(dirs.shape)",
        "    ss_sun += weight * (ph * vis * live)[..., None] * sun_L * Ts":
            "    ss_sun += weight * (ph * vis * live)[..., None] * sun_L * Ts\n    ss_sun_unoccluded += weight * (ph * live)[..., None] * sun_L * Ts",
        "    acc = np.zeros(dirs.shape)":
            "    acc = np.zeros(dirs.shape)\n    acc_unoccluded = np.zeros(dirs.shape)",
        "        acc += (hg(np.sum(dirs * w, axis=-1)) * visw * live)[..., None] * Tw":
            "        acc += (hg(np.sum(dirs * w, axis=-1)) * visw * live)[..., None] * Tw\n        acc_unoccluded += (hg(np.sum(dirs * w, axis=-1)) * live)[..., None] * Tw",
        "    ss_env += weight * acc * env_L * dw":
            "    ss_env += weight * acc * env_L * dw\n    ss_env_unoccluded += weight * acc_unoccluded * env_L * dw",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError(f"historical tool no longer has exactly one patch site: {old}")
        source = source.replace(old, new)
    args.scratch.mkdir(parents=True, exist_ok=True)
    patched_path = args.scratch / "nephele_first_order_ablation.py"
    patched_path.write_text(source, encoding="utf-8", newline="\n")
    old_argv = sys.argv
    try:
        sys.argv = [str(patched_path), str(root), str(args.scratch)]
        namespace = {"__name__": "nephele_gate4_first_order", "__file__": str(patched_path)}
        exec(compile(source, str(patched_path), "exec"), namespace)
    finally:
        sys.argv = old_argv

    artifact = args.artifact_dir
    load = lambda name: np.load(artifact / f"{name}.npy", allow_pickle=False)
    reference, realtime, disabled = (load(n) for n in (
        "reference-rgb", "realtime-rgb", "terrain-occlusion-disabled-rgb"))
    shadow = load("cloud-shadow-mask") & load("terrain-mask")
    sky = load("sky-cloud-mask")
    roi = load("godray-roi-mask")
    ys, xs = np.nonzero(roi)
    crop = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if not roi[crop].all():
        raise ValueError("ROI must be rectangular")
    ref_lab, rt_lab = srgb_to_lab(reference), srgb_to_lab(realtime)
    de = delta_e_2000(rt_lab, ref_lab)
    lab_error = rt_lab - ref_lab
    residual = lab_error.copy()
    # Diagnostic only: subtract each existing 8x8 froxel tile's masked mean.
    for y in range(0, shadow.shape[0], 8):
        for x in range(0, shadow.shape[1], 8):
            tile = np.s_[y:y + 8, x:x + 8]
            mask = shadow[tile]
            if mask.any():
                residual[tile] -= lab_error[tile][mask].mean(axis=0)
    fail = shadow & (de >= 2.0)

    # Same numerical sRGB + ACES inverse as round2/first_order_compare.py.
    aces_x = np.linspace(0.0, 64.0, 2_000_001)
    aces_y = np.clip((aces_x * (2.51 * aces_x + 0.03)) /
                     (aces_x * (2.43 * aces_x + 0.59) + 0.14), 0, 1)

    def linear(u8):
        value = u8.astype(np.float64) / 255.0
        display = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
        return np.interp(display, aces_y, aces_x)

    def display(value):
        value = np.maximum(value, 0.0)
        mapped = np.clip((value * (2.51 * value + 0.03)) /
                         (value * (2.43 * value + 0.59) + 0.14), 0, 1)
        encoded = np.where(mapped <= 0.0031308, mapped * 12.92,
                           1.055 * mapped ** (1.0 / 2.4) - 0.055)
        return np.rint(encoded * 255.0).astype(np.uint8)

    n = namespace
    delta = n["ss_sun_unoccluded"] + n["ss_env_unoccluded"] - n["ss_sun"] - n["ss_env"]
    estimated_disabled = display(linear(reference) + delta)
    luma = np.asarray([0.2126, 0.7152, 0.0722])
    ref_change = float((delta[crop] @ luma).mean())
    rt_change = float(((linear(disabled) - linear(realtime))[crop] @ luma).mean())
    input_names = ["reference-rgb", "realtime-rgb", "terrain-occlusion-disabled-rgb",
                   "cloud-shadow-mask", "terrain-mask", "sky-cloud-mask", "godray-roi-mask"]
    report = {
        "schema": "forge3d.nephele.gate4_diagnosis/1",
        "ci_run": 39517423981,
        "captured_head": (artifact / "checked-out-head.txt").read_text(encoding="utf-8-sig").strip(),
        "reference_estimate": "first-order camera single-scatter delta added to decoded 640-spp reference; not a regenerated ablated reference",
        "quadrature_directions": len(n["SPHERE"]),
        "camera_steps": n["STEPS"],
        "FO_STEPS_environment": os.environ.get("FO_STEPS"),
        "numpy_version": np.__version__,
        "historical_tool_sha256": sha256(source_path),
        "patched_tool_sha256": sha256(patched_path),
        "measurement_tool_sha256": sha256(Path(__file__)),
        "input_sha256": {name: sha256(artifact / f"{name}.npy") for name in input_names},
        "fixture_sha256": {path.name: sha256(path) for path in sorted((root / "tests/nephele/fixture").iterdir()) if path.is_file()},
        "gate3": {"sky_fraction_delta_e2000_lt_2_5": float((de[sky] < 2.5).mean()),
                  "godray_ssim": ssim(reference[crop], realtime[crop], data_range=255.0)},
        "gate4a": {"pixels": int(shadow.sum()), "fraction_delta_e2000_lt_2": float((de[shadow] < 2.0).mean()),
                   "mean_lab_error": lab_error[shadow].mean(axis=0).tolist(),
                   "failing_pixels_too_dark_fraction": float((lab_error[..., 0][fail] < 0).mean()),
                   "tile_mean_removed_fraction_delta_e76_lt_2": float((np.linalg.norm(residual[shadow], axis=-1) < 2).mean())},
        "gate4b": {"roi_xyxy": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
                   "realtime_ablation_ssim": ssim(realtime[crop], disabled[crop], data_range=255.0),
                   "reference_first_order_ablation_ssim": ssim(reference[crop], estimated_disabled[crop], data_range=255.0),
                   "reference_first_order_mean_linear_luminance_change": ref_change,
                   "realtime_mean_linear_luminance_change": rt_change,
                   "realtime_to_reference_first_order_change_ratio": rt_change / ref_change},
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: report[key] for key in ("gate3", "gate4a", "gate4b")}, indent=2))


if __name__ == "__main__":
    main()
