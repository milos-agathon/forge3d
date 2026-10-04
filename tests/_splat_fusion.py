# tests/_splat_fusion.py
# Shared helpers for the SPLAT-FUSED tests: the committed three-representation
# fixture (mini DEM, Gaussian splat cloud, COPC LiDAR swath) and the GPU
# availability convention.
# RELEVANT FILES: tests/test_splat_api.py, tests/test_splat_fusion_occlusion.py,
#                 src/splat/fixture.rs

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

import forge3d as f3d
from forge3d import splat

from _terrain_runtime import (
    _running_on_unsupported_hosted_macos_ci,
    _running_on_unsupported_hosted_windows_ci,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "splat_fusion"
GOLDEN_PATH = ROOT / "tests" / "golden" / "splat_fusion" / "fused_fixture.png"

#: Sun/sky of the acceptance renders (identical to the Rust acceptance test).
SUN_INTENSITY = 2.5
SUN_COLOR = (1.0, 0.97, 0.92)


def manifest() -> Dict[str, Any]:
    return json.loads((FIXTURE_DIR / "scene.json").read_text())


def heights(m: Dict[str, Any]) -> np.ndarray:
    return np.fromfile(FIXTURE_DIR / m["dem_file"], dtype="<f4").reshape(
        m["dem_height"], m["dem_width"]
    )


def gpu_unavailable_reason() -> "str | None":
    """None when the fused GPU tests can run, else why they cannot."""
    if _running_on_unsupported_hosted_macos_ci() or _running_on_unsupported_hosted_windows_ci():
        return "hosted CI runner without a supported GPU"
    if not splat.splat_fusion_available():
        return "wheel built without the splat-fusion feature"
    if not f3d.has_gpu():
        return "forge3d.has_gpu() reported no usable adapter"
    return None


def gpu_available() -> bool:
    """Skip-or-fail gate: FORGE3D_SPLAT_FUSION_REQUIRED_GPU=1 turns a missing
    GPU into a failure on the hardware lane."""
    reason = gpu_unavailable_reason()
    if reason is None:
        return True
    if os.environ.get("FORGE3D_SPLAT_FUSION_REQUIRED_GPU") == "1":
        raise RuntimeError(
            f"FORGE3D_SPLAT_FUSION_REQUIRED_GPU=1 but the fused GPU path is unavailable: {reason}"
        )
    return False


def scene_kwargs(m: Dict[str, Any]) -> Dict[str, Any]:
    """The fixture as `render_fused` / `render_fused_reference` inputs."""
    return dict(
        splats=str(FIXTURE_DIR / m["splat_file"]),
        pointcloud=splat.FusedPointCloud(
            path=FIXTURE_DIR / m["copc_file"],
            origin=tuple(m["copc_origin"]),
            z_up=True,
        ),
        terrain=splat.FusedTerrain(
            heights=heights(m),
            spacing=tuple(m["dem_spacing"]),
            albedo=tuple(m["terrain_albedo"]),
        ),
        camera=splat.FusedCamera(
            origin=tuple(m["cam_origin"]),
            look_at=tuple(m["cam_look_at"]),
            up=tuple(m["cam_up"]),
            fov_y_deg=m["fov_y_deg"],
        ),
        sun_azimuth_deg=m["sun_azimuth_deg"],
        sun_elevation_deg=m["sun_elevation_deg"],
        sun_intensity=SUN_INTENSITY,
        sun_color=SUN_COLOR,
        lidar_radius=m["lidar_radius"],
    )
