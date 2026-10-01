"""Physical readbacks for the bounded NEPHELE environment remediation."""

from pathlib import Path
import json
import subprocess
import sys

import numpy as np

from scripts.generate_media_fixture import _aces_srgb8

ROOT = Path(__file__).resolve().parents[1]


def test_camera_visible_uniform_environment_readback(tmp_path: Path) -> None:
    output = tmp_path / "capture"
    control = ROOT / "docs/audits/nephele-vacuum-control-published-spp320-2026-09-30"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.nephele_b1_diagnostic", "--repo", str(ROOT),
         "--output", str(output), "--vacuum-reference", str(control / "vacuum-reference-spp320.npy"),
         "--control-registration", str(control / "vacuum-control-spp320.json")],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    actual = np.load(output / "medium-disabled-rgb.npy", allow_pickle=False)
    hit = np.load(ROOT / "tests/nephele/fixture/reference-terrain-hit.npy", allow_pickle=False)
    exposure = json.loads((ROOT / "tests/nephele/fixture/exposure.json").read_text(encoding="utf-8"))["value"]
    expected = _aces_srgb8(np.full(3, 0.25 * exposure, dtype=np.float32))
    background = hit == 0
    assert np.any(background)
    np.testing.assert_array_equal(actual[background], np.broadcast_to(expected, actual[background].shape))

def test_shadow_scattering_component_readback(tmp_path: Path) -> None:
    output = tmp_path / "shadow-components"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.nephele_shadow_inscatter_diagnostic", "--repo", str(ROOT),
         "--output", str(output)], cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["physical_acceptance"] is False
    assert report["mask_pixels"] > 0
    for label in ("primary", "isotropic", "occlusion-disabled"):
        assert report[label]["shadow_single_luminance"] > 0
        assert report[label]["shadow_multiple_luminance"] > 0
        assert report[label]["diagnostics"]["executed_multi_scatter"] is True
