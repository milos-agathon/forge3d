#!/usr/bin/env python3
"""Temporary exact-test launcher for physical NVIDIA golden capture."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest


UNSAFE_ADAPTER_TOKENS = (
    "basic render driver",
    "lavapipe",
    "llvmpipe",
    "paravirtual",
    "software",
    "swiftshader",
    "virtual",
    "virtio",
    "warp",
)


def _require_nvidia_vulkan(probe: object) -> dict:
    if not isinstance(probe, dict):
        raise RuntimeError("capture-process adapter probe is not an object")
    name = str(probe.get("name", ""))
    if (
        probe.get("status") != "ok"
        or str(probe.get("backend", "")).lower() != "vulkan"
        or str(probe.get("device_type", "")).lower() != "discretegpu"
        or bool(probe.get("software_fallback", False))
        or (int(probe.get("vendor", 0)) != 0x10DE and "nvidia" not in name.lower())
        or any(token in name.lower() for token in UNSAFE_ADAPTER_TOKENS)
    ):
        raise RuntimeError(f"capture process did not select physical NVIDIA Vulkan: {probe}")
    return probe


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    import forge3d as f3d

    artifact_dir = Path(os.environ["FORGE3D_SUBSTRATIA_ARTIFACT_DIR"])
    workflow_envelope = json.loads((artifact_dir / "adapter-probe.json").read_text())
    workflow_probe = _require_nvidia_vulkan(workflow_envelope.get("probe"))
    # Initialize the production global context before probing so all in-process
    # test renders are bound to the exact adapter identity checked below.
    capture_session = f3d.Session(window=False)
    process_probe = _require_nvidia_vulkan(f3d.device_probe("vulkan"))
    identity_fields = ("backend", "device_type", "name", "vendor", "device")
    if any(
        str(process_probe.get(field, "")).lower()
        != str(workflow_probe.get(field, "")).lower()
        for field in identity_fields
    ):
        raise RuntimeError(
            f"capture-process adapter {process_probe} differs from workflow probe {workflow_probe}"
        )
    (artifact_dir / "capture-process-adapter.json").write_text(
        json.dumps(process_probe, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    status = pytest.main(
        [
            "tests/test_terrain_vt_pbr_families.py::TestTerrainVTPbrFamilies::test_normal_family_changes_lighting_ssim",
            "tests/test_terrain_vt_pbr_families.py::TestTerrainVTPbrFamilies::test_all_families_page_within_budget",
            "tests/test_terrain_vt_pbr_families.py::TestTerrainVTPbrFamilies::test_missing_family_is_fatal",
            "tests/test_terrain_vt_pbr_families.py::TestTerrainVTPbrFamilies::test_partial_normal_residency_degrades_gracefully",
            "tests/test_astro_night_golden.py::test_night_golden_matches_committed_vulkan_bytes",
            f"--junitxml={artifact_dir / 'pytest-junit.xml'}",
            "-v",
            "--tb=short",
        ]
    )
    _ = capture_session
    if status != pytest.ExitCode.OK:
        return int(status)

    sidera_probe = _require_nvidia_vulkan(
        json.loads((artifact_dir / "sidera-process-adapter.json").read_text())
    )
    if any(
        str(sidera_probe.get(field, "")).lower()
        != str(process_probe.get(field, "")).lower()
        for field in identity_fields
    ):
        raise RuntimeError(
            f"SIDERA subprocess adapter {sidera_probe} differs from capture process {process_probe}"
        )

    root = Path(__file__).resolve().parents[1]
    sidera_png = root / "tests" / "golden" / "sidera_night.png"
    sidera_sha = root / "tests" / "goldens" / "determinism" / "sidera_night.sha256"
    shutil.copy2(sidera_png, artifact_dir / "sidera_night.png")
    shutil.copy2(sidera_sha, artifact_dir / "sidera_night.sha256")
    mappings = {
        "golden_baseline.png": "tests/golden/terrain/substratia_grazing_baseline.nvidia-vulkan.png",
        "golden_normal.png": "tests/golden/terrain/substratia_grazing_normal.nvidia-vulkan.png",
        "sidera_night.png": "tests/golden/sidera_night.png",
        "sidera_night.sha256": "tests/goldens/determinism/sidera_night.sha256",
    }
    manifest = {
        "schema": "forge3d.nvidia-golden-capture.v1",
        "candidate_sha": os.environ["FORGE3D_SUBSTRATIA_CANDIDATE_SHA"],
        "adapter": process_probe,
        "sidera_adapter": sidera_probe,
        "files": {
            name: {"target": target, "sha256": _sha256(artifact_dir / name)}
            for name, target in mappings.items()
        },
    }
    (artifact_dir / "capture-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
