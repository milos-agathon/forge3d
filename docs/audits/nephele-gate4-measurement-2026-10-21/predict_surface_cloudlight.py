"""Apply the saved physical 192-direction term to a captured frame, on CPU.

This is an offline prediction, not a modified renderer or physical-GPU pass.
Run from the repository root with --artifact-dir and --output.
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
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "tests"))
    from _deltae import delta_e_2000, srgb_to_lab

    term = root / "docs/audits/nephele-bounded-round-2026-10-01/round2/first-order/surface-cloudlight.npy"
    artifact = args.artifact_dir
    paths = {name: artifact / f"{name}.npy" for name in (
        "realtime-rgb", "reference-rgb", "transmittance", "terrain-mask", "cloud-shadow-mask", "sky-cloud-mask")}
    data = {name: np.load(path, allow_pickle=False) for name, path in paths.items()}
    realtime, reference = data["realtime-rgb"], data["reference-rgb"]
    shadow = data["terrain-mask"] & data["cloud-shadow-mask"]
    surface = np.load(term, allow_pickle=False)
    x = np.linspace(0, 64, 2_000_001)

    def aces(value):
        return np.clip(value * (2.51 * value + 0.03) /
                       (value * (2.43 * value + 0.59) + 0.14), 0, 1)

    def display(value):
        value = aces(value)
        srgb = np.where(value <= 0.0031308, value * 12.92,
                        1.055 * value ** (1.0 / 2.4) - 0.055)
        return np.rint(srgb * 255).astype(np.uint8)

    u = realtime.astype(np.float64) / 255.0
    linear = np.interp(np.where(u <= 0.04045, u / 12.92,
                               ((u + 0.055) / 1.055) ** 2.4), aces(x), x)
    if not np.array_equal(display(linear), realtime):
        raise ValueError("sRGB/ACES inverse does not round-trip the captured frame")
    addition = data["transmittance"][..., :3] * surface
    predicted = display(linear + addition)
    score = lambda image: float((delta_e_2000(srgb_to_lab(image), srgb_to_lab(reference))[shadow] < 2.0).mean())
    report = {
        "schema": "forge3d.nephele.surface_cloudlight_prediction/1",
        "kind": "offline prediction; not a renderer or GPU measurement",
        "quadrature_directions": 192,
        "g4a_existing": score(realtime),
        "g4a_predicted_with_physical_term": score(predicted),
        "mean_added_linear_rgb_on_shadowed_terrain": addition[shadow].mean(axis=0).tolist(),
        "sky_pixels_byte_identical": bool(np.array_equal(predicted[data["sky-cloud-mask"]], realtime[data["sky-cloud-mask"]])),
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "surface_cloudlight_sha256": hashlib.sha256(term.read_bytes()).hexdigest(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: value for key, value in report.items() if key.endswith("existing") or "predicted" in key or key.startswith("sky_")}, indent=2))


if __name__ == "__main__":
    main()
