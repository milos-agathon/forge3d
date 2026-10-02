"""Run the non-acceptance NEPHELE T4/B1 terrain parity diagnostic.

This tool captures one no-medium candidate frame and compares it with the
committed, converged vacuum control. It never renders or replaces a reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


AUDIT_REFERENCE = Path("docs/audits/nephele-b1-2026-09-30/vacuum-reference-spp40.npy")
PASS_FRACTION = 0.95


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _save(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments],
        text=True,
        encoding="utf-8",
    ).strip()


def _require_committed_reference(repo: Path, supplied: Path) -> Path:
    expected = (repo / AUDIT_REFERENCE).resolve()
    reference = supplied.resolve()
    if reference != expected:
        raise ValueError(f"vacuum reference must be the committed audit artifact: {expected}")
    relative = AUDIT_REFERENCE.as_posix()
    _git(repo, "ls-files", "--error-unmatch", "--", relative)
    working_hash = _git(repo, "hash-object", relative)
    committed_hash = _git(repo, "rev-parse", f"HEAD:{relative}")
    if working_hash != committed_hash:
        raise ValueError("vacuum reference differs from its committed HEAD blob")
    return reference


def _metrics(candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    from tests._deltae import delta_e_2000, srgb_to_lab

    if candidate.shape != reference.shape or candidate.shape[:2] != mask.shape:
        raise ValueError("candidate, vacuum reference, and terrain mask shapes differ")
    values = delta_e_2000(srgb_to_lab(candidate), srgb_to_lab(reference))[mask]
    if not values.size:
        raise ValueError("terrain mask is empty")
    fraction = float(np.mean(values < 2.0))
    return {
        "pixels": int(values.size),
        "mean_delta_e": float(values.mean()),
        "p95_delta_e": float(np.quantile(values, 0.95)),
        "maximum_delta_e": float(values.max()),
        "fraction_delta_e_below_2": fraction,
        "pass": fraction >= PASS_FRACTION,
    }


def _manifest_path(repo: Path, record: dict[str, Any], label: str) -> Path:
    relative = Path(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}: invalid manifest path")
    path = repo / relative
    if path.name != record["artifact"] or _sha256(path) != record["sha256"]:
        raise ValueError(f"{label}: tracked artifact differs from the fixture manifest")
    return path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Forge3D checkout to capture")
    parser.add_argument("--output", type=Path, required=True, help="new diagnostic output directory")
    parser.add_argument(
        "--vacuum-reference",
        type=Path,
        required=True,
        help="committed docs/audits/.../vacuum-reference-spp40.npy",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    repo = arguments.repo.resolve()
    output = arguments.output.resolve()
    fixture = repo / "tests/nephele/fixture"
    if not (repo / ".git").exists():
        raise ValueError(f"not a Forge3D checkout: {repo}")
    vacuum_path = _require_committed_reference(repo, arguments.vacuum_reference)
    if output.exists():
        raise FileExistsError(f"refusing to replace diagnostic output: {output}")

    manifest_path = repo / "tests/nephele/fixture-manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("status") != "APPROVED":
        raise RuntimeError("T4/B1 runs only with an APPROVED frozen fixture")
    for role, item in manifest["scene_inputs"].items():
        _manifest_path(repo, item, f"scene input {role}")
    terrain_mask_path = _manifest_path(
        repo, manifest["files"]["terrain_mask"], "terrain mask"
    )

    os.environ.update(
        FORGE3D_NO_BOOTSTRAP="1",
        FORGE3D_TEST_INSTALLED_WHEEL="1",
        WGPU_BACKEND="vulkan",
        WGPU_BACKENDS="vulkan",
    )
    sys.path.insert(0, str(repo))
    import forge3d as f3d
    from forge3d.media import Medium
    from forge3d.terrain_params import (
        AovSettings,
        TonemapSettings,
        make_terrain_params_config,
    )
    from scripts.run_media_physical_capture import _camera_contract, _crop, _write_hdr

    def read_fixture(name: str) -> dict[str, Any]:
        return _json(fixture / name)

    camera_input = read_fixture("camera.json")
    terrain_data = read_fixture("terrain.json")
    medium_data = read_fixture("medium.json")
    material_input = read_fixture("material.json")
    sun = read_fixture("sun.json")
    atmosphere = read_fixture("atmosphere.json")
    exposure = read_fixture("exposure.json")
    crop = read_fixture("crop.json")
    provenance = read_fixture("reference-provenance.json")
    camera = camera_input["terrain_camera"]
    terrain = np.load(fixture / terrain_data["dem"], allow_pickle=False)
    shape = medium_data["domain"]["grid_shape"]
    density = np.asarray(medium_data["density_r16"], dtype=np.float32).reshape(
        shape[2], shape[1], shape[0]
    ) / 65535.0
    bounds = (
        medium_data["domain"]["bounds_min"],
        medium_data["domain"]["bounds_max"],
    )
    medium = Medium.grid3d(
        medium_data["sigma_a"],
        medium_data["sigma_s"],
        density,
        bounds,
        phase="henyey_greenstein",
        g=medium_data["phase"]["g"],
        density_scale=medium_data["density_scale"],
        version=1,
    )
    spans = [
        terrain_data["spacing"][index] * (terrain_data["dimensions"][index] - 1)
        for index in range(2)
    ]
    if spans[0] != spans[1]:
        raise ValueError("expected the tracked square terrain span")

    # This explicit physical Lambert model is the only intended B1 variable.
    config = make_terrain_params_config(
        size_px=tuple(crop["full_viewport"]),
        render_scale=1.0,
        terrain_span=spans[0],
        msaa_samples=1,
        z_scale=float(terrain_data["exaggeration"]),
        exposure=float(exposure["value"]),
        domain=(float(terrain.min()), float(terrain.max())),
        light_azimuth_deg=sun["azimuth_deg"],
        light_elevation_deg=sun["elevation_deg"],
        sun_intensity=sun["intensity"],
        sun_color=sun["color"],
        albedo_mode=material_input["albedo_mode"],
        colormap_strength=float(material_input["colormap_strength"]),
        cam_radius=camera["radius"],
        cam_phi_deg=camera["phi_deg"],
        cam_theta_deg=camera["theta_deg"],
        cam_target=camera["target"],
        fov_y_deg=float(camera_input["fov_y"]),
        camera_mode=camera["mode"],
        aa_samples=1,
        aa_seed=0x4E455048,
        tonemap=TonemapSettings(operator="aces"),
        aov=AovSettings(
            enabled=True,
            transmittance=True,
            in_scatter=True,
            cloud_shadow=True,
            optical_depth=True,
        ),
        media=medium,
        terrain_shading_model="lambert_physical",
    )
    params = f3d.TerrainRenderParams(config)
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material = f3d.MaterialSet.custom(
        tuple(material_input["albedo"]),
        float(material_input["metallic"]),
        float(material_input["roughness"]),
        triplanar_scale=float(material_input["triplanar_scale"]),
        normal_strength=float(material_input["normal_strength"]),
        blend_sharpness=float(material_input["blend_sharpness"]),
    )
    with tempfile.TemporaryDirectory(prefix="nephele-b1-ibl-") as temporary:
        hdr = Path(temporary) / "fixture.hdr"
        _write_hdr(hdr)
        ibl = f3d.IBL.from_hdr(
            str(hdr), intensity=float(atmosphere["environment_intensity"])
        )
        capture = renderer._capture_nephele_acceptance(
            material,
            ibl,
            params,
            terrain,
            terrain_occlusion_in_media=True,
            include_no_medium=True,
        )

    diagnostics = dict(capture["diagnostics"])
    if diagnostics.get("terrain_shading_model") != "lambert_physical":
        raise RuntimeError("B1 capture did not execute the physical terrain shading model")
    reference_diagnostics = provenance["diagnostics"]
    for key in ("adapter", "backend", "driver"):
        if diagnostics.get(key) != reference_diagnostics.get(key):
            raise RuntimeError(f"candidate/reference {key} identities differ")
    actual_camera = _camera_contract(capture["camera_contract"])
    expected_camera = _camera_contract(provenance["camera_contract"])
    if any(
        not np.array_equal(
            np.asarray(actual_camera[key], np.float32),
            np.asarray(expected_camera[key], np.float32),
        )
        for key in expected_camera
    ):
        raise RuntimeError("candidate/reference camera contracts differ")

    candidate = _crop(np.asarray(capture["no_medium_beauty"], dtype=np.uint8), crop)
    vacuum = np.load(vacuum_path, allow_pickle=False)
    terrain_mask = np.load(terrain_mask_path, allow_pickle=False)
    comparison = _metrics(candidate, vacuum, terrain_mask)
    dirty_lines = _git(repo, "status", "--porcelain=v1", "--untracked-files=no").splitlines()
    head = _git(repo, "rev-parse", "HEAD")
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "medium-disabled-rgb.npy", candidate, allow_pickle=False)
    # Retain the same executed primary frame and AOVs for the subsequent
    # B3-B9 diagnostics; this avoids a second capture with different inputs.
    for key, filename in (
        ("beauty", "realtime-rgb.npy"),
        ("in_scatter", "in-scatter.npy"),
        ("cloud_shadow", "cloud-shadow-aov.npy"),
        ("transmittance", "transmittance.npy"),
        ("optical_depth", "optical-depth.npy"),
    ):
        np.save(output / filename, _crop(np.asarray(capture[key]), crop), allow_pickle=False)
    _save(output / "candidate-diagnostics.json", diagnostics)
    _save(output / "camera-contract.json", actual_camera)
    report = {
        "schema": "forge3d.nephele.b1_diagnostic/2",
        "physical_acceptance": False,
        "repository": {
            "path": str(repo),
            "head": head,
            "tracked_worktree_clean": not dirty_lines,
            "tracked_dirty_entries": dirty_lines,
        },
        "candidate_native": {
            "source_revision": diagnostics.get("source_revision"),
            "source_matches_repository_head": diagnostics.get("source_revision") == head,
            "adapter": diagnostics.get("adapter"),
            "backend": diagnostics.get("backend"),
            "driver": diagnostics.get("driver"),
        },
        "diagnostic_tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__)),
        },
        "fixture_manifest_sha256": _sha256(manifest_path),
        "scene_inputs": manifest["scene_inputs"],
        "terrain_mask": manifest["files"]["terrain_mask"],
        "medium_control": (
            "candidate no-medium branch versus committed sigma_a=sigma_s=0 "
            "vacuum reference"
        ),
        "terrain_shading_model": "lambert_physical",
        "vacuum_reference": {
            "path": AUDIT_REFERENCE.as_posix(),
            "sha256": _sha256(vacuum_path),
            "samples_per_pixel": 40,
            "reused_committed_blob": True,
        },
        "terrain_comparison": comparison,
        "criterion": {
            "metric": "fraction_delta_e_below_2",
            "minimum": PASS_FRACTION,
            "pass": comparison["pass"],
        },
        "gate4_comparison1": "PRECHECK_PASS" if comparison["pass"] else "NOT_PROVEN",
    }
    _save(output / "b1-report.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0 if comparison["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
