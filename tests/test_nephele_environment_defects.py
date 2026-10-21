"""Physical readbacks for the bounded NEPHELE environment remediation."""

from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from _terrain_runtime import (
    _running_on_unsupported_hosted_macos_ci,
    _running_on_unsupported_hosted_windows_ci,
)
from scripts.generate_media_fixture import _aces_srgb8

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/nephele/fixture"
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)

# Owner-approved (Milos, 2026-10-20): hosted Windows and macOS runners have no
# Vulkan adapter for these Vulkan-pinned readbacks. Hosted Linux (lavapipe) and
# the opted-in physical NVIDIA lanes still run them.
pytestmark = pytest.mark.skipif(
    _running_on_unsupported_hosted_windows_ci() or _running_on_unsupported_hosted_macos_ci(),
    reason="hosted Windows/macOS CI has no Vulkan adapter for NEPHELE environment readbacks",
)

# Each capture runs in a fresh Vulkan-pinned process, as the physical lane does.
_CAPTURE = """
import json, sys, numpy as np
from scripts.nephele_acceptance import FixtureScene
phase, output, flags = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
capture = FixtureScene(phase).capture(**flags)
arrays = {key: np.asarray(value) for key, value in capture.items() if isinstance(value, np.ndarray)}
assert all(np.isfinite(value).all() for value in arrays.values()), "non-finite GPU readback"
np.savez(output, **arrays)
print(json.dumps(dict(capture["diagnostics"])))
"""


def _capture(tmp_path: Path, label: str, phase: str = "henyey_greenstein", **flags: bool):
    output = tmp_path / f"{label}.npz"
    env = {**os.environ, "FORGE3D_NO_BOOTSTRAP": "1", "FORGE3D_TEST_INSTALLED_WHEEL": "1",
           "WGPU_BACKEND": "vulkan", "WGPU_BACKENDS": "vulkan"}
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", _CAPTURE, phase, str(output), json.dumps(flags)],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    diagnostics = json.loads(result.stdout.strip().splitlines()[-1])
    return dict(np.load(output)), diagnostics


def test_camera_visible_uniform_environment_readback(tmp_path: Path) -> None:
    arrays, _ = _capture(tmp_path, "background", include_no_medium=True)
    actual = arrays["no_medium_beauty"]
    hit = np.load(FIXTURE / "reference-terrain-hit.npy", allow_pickle=False)
    exposure = json.loads((FIXTURE / "exposure.json").read_text(encoding="utf-8"))["value"]
    expected = _aces_srgb8(np.full(3, 0.25 * exposure, dtype=np.float32))
    background = hit == 0
    assert np.any(background)
    np.testing.assert_array_equal(actual[background], np.broadcast_to(expected, actual[background].shape))


def test_shadow_scattering_component_readback(tmp_path: Path) -> None:
    mask = np.load(FIXTURE / "cloud-shadow-mask.npy").astype(bool) & np.load(FIXTURE / "terrain-mask.npy").astype(bool)
    assert mask.sum() > 0
    for label, phase, occlusion in (
        ("primary", "henyey_greenstein", True),
        ("isotropic", "isotropic", True),
        ("occlusion-disabled", "henyey_greenstein", False),
    ):
        arrays, diagnostics = _capture(
            tmp_path, label, phase, terrain_occlusion_in_media=occlusion, capture_scatter_components=True,
        )
        multiple = arrays["in_scatter_multiple_luminance"].astype(np.float64)
        # The alpha channel and the native aggregate must describe the same GPU frame.
        np.testing.assert_allclose(float(multiple.mean()), diagnostics["multiple_scatter_luminance"],
                                   rtol=0, atol=np.finfo(np.float32).eps)
        total = arrays["in_scatter"].astype(np.float64) @ LUMA
        assert float((total - multiple)[mask].mean()) > 0, label
        assert float(multiple[mask].mean()) > 0, label
        assert diagnostics["executed_multi_scatter"] is True, label
