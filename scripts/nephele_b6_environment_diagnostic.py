"""Isolate environment illumination under the frozen cloud-shadow mask.

The only lighting ablation is zero sun intensity. Reference sampling and seed
come from the approved fixture, and the retained reference wheel is verified
before importing native code. This diagnostic never replaces the fixture.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile

import numpy as np

from scripts import nephele_b1_diagnostic as b1


def scene(repo: Path):
    from forge3d.media import Medium

    fixture = repo / "tests/nephele/fixture"
    read = lambda name: b1._json(fixture / f"{name}.json")
    camera, terrain_data, medium_data = read("camera"), read("terrain"), read("medium")
    material, sun, atmosphere = read("material"), read("sun"), read("atmosphere")
    exposure, crop = read("exposure"), read("crop")
    terrain = np.ascontiguousarray(np.load(fixture / terrain_data["dem"], allow_pickle=False), dtype=np.float32)
    shape = medium_data["domain"]["grid_shape"]
    density = np.asarray(medium_data["density_r16"], dtype=np.float32).reshape(shape[2], shape[1], shape[0]) / 65535.0
    medium = Medium.grid3d(
        medium_data["sigma_a"], medium_data["sigma_s"], density,
        (medium_data["domain"]["bounds_min"], medium_data["domain"]["bounds_max"]),
        phase="henyey_greenstein", g=medium_data["phase"]["g"],
        density_scale=medium_data["density_scale"], version=1,
    )
    return camera, terrain_data, material, sun, atmosphere, exposure, crop, terrain, medium


def reference(repo: Path, output: Path) -> dict:
    _, provenance = b1._frozen_fixture(repo)
    wheel, wheel_record = b1._reference_wheel(repo, provenance["native_runtime"])
    if any(name == "forge3d" or name.startswith("forge3d.") for name in sys.modules):
        raise RuntimeError("reference diagnostic requires a fresh process")
    with tempfile.TemporaryDirectory(prefix="nephele-b6-reference-", ignore_cleanup_errors=True) as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(extracted)
        sys.path.insert(0, str(extracted))
        from forge3d.media import _native_module
        native = _native_module()
        imported = b1._verify_imported_native(native, provenance["native_runtime"], extracted)
        camera, terrain_data, material, sun, atmosphere, exposure, crop, terrain, medium = scene(repo)
        from scripts.generate_media_fixture import _aces_srgb8
        from scripts.run_media_physical_capture import _camera_contract
        started = time.monotonic()
        native_camera = {**camera, "terrain_camera": {**camera["terrain_camera"],
            "target": tuple(camera["terrain_camera"]["target"])}}
        capture = native._render_volumetric_reference(
            medium._native, terrain, crop["width"], crop["height"], native_camera,
            spacing=tuple(terrain_data["spacing"]), exaggeration=terrain_data["exaggeration"],
            albedo=tuple(material["albedo"]), sun_azimuth_deg=sun["azimuth_deg"],
            sun_elevation_deg=sun["elevation_deg"], sun_intensity=0.0, sun_color=tuple(sun["color"]),
            environment_intensity=atmosphere["environment_intensity"], exposure=exposure["value"],
            samples_per_pixel=provenance["samples_per_pixel"], homogeneous_medium_reach=120.0,
            seed=provenance["seed"], full_viewport=tuple(crop["full_viewport"]),
            crop=(crop["x"], crop["y"], crop["width"], crop["height"]),
        )
        elapsed = time.monotonic() - started
        diagnostics = dict(capture["diagnostics"])
        for key in ("source_revision", "adapter", "backend", "driver"):
            if diagnostics[key] != provenance["diagnostics"][key]:
                raise RuntimeError(f"reference {key} differs from retained runtime")
        expected = provenance["samples_per_pixel"] * crop["width"] * crop["height"]
        if diagnostics["sample_count"] != expected:
            raise RuntimeError("reference executed sample count differs from fixture")
        np.save(output / "reference-environment-linear.npy", capture["beauty"], allow_pickle=False)
        np.save(output / "reference-environment-rgb.npy", _aces_srgb8(np.asarray(capture["beauty"])), allow_pickle=False)
        return {"diagnostics": diagnostics, "camera_contract": _camera_contract(capture["camera_contract"]),
                "imported_native": imported, "wheel": wheel_record,
                "render_wall_seconds": elapsed, "samples_per_pixel": provenance["samples_per_pixel"]}


def environment_reference(repo: Path, reference_dir: Path) -> tuple[dict, Path]:
    """Validate the executed reference record before any candidate GPU work."""
    _, provenance = b1._frozen_fixture(repo)
    record = b1._json(reference_dir / "report.json")
    crop = b1._json(repo / "tests/nephele/fixture/crop.json")
    if (
        record.get("schema") != "forge3d.nephele.b6_environment_diagnostic/1"
        or record.get("sun_intensity") != 0.0
        or record.get("fixture_manifest_sha256") != b1._sha256(repo / "tests/nephele/fixture-manifest.json")
        or record.get("samples_per_pixel") != provenance["samples_per_pixel"]
        or record.get("imported_native", {}).get("sha256") != provenance["native_runtime"]["native_sha256"]
        or record.get("diagnostics", {}).get("sample_count") != provenance["samples_per_pixel"] * crop["width"] * crop["height"]
    ):
        raise RuntimeError("environment reference record differs from the approved fixture/runtime")
    for key in ("source_revision", "adapter", "backend", "driver"):
        if record["diagnostics"].get(key) != provenance["diagnostics"][key]:
            raise RuntimeError(f"environment reference {key} differs from the approved runtime")
    path = reference_dir / "reference-environment-rgb.npy"
    if b1._sha256(path) != record["artifact_hashes"][path.name]:
        raise RuntimeError("environment reference array differs from its executed record")
    return record, path


def candidate(repo: Path, output: Path, reference_dir: Path) -> dict:
    reference_report, reference_path = environment_reference(repo, reference_dir)
    import forge3d as f3d
    from forge3d.terrain_params import AovSettings, TonemapSettings, make_terrain_params_config
    from scripts.run_media_physical_capture import _camera_contract, _crop, _write_hdr

    camera, terrain_data, material_input, sun, atmosphere, exposure, crop, terrain, medium = scene(repo)
    camera = camera["terrain_camera"]
    span = terrain_data["spacing"][0] * (terrain_data["dimensions"][0] - 1)
    config = make_terrain_params_config(
        terrain_shading_model="lambert_physical", size_px=tuple(crop["full_viewport"]),
        render_scale=1.0, terrain_span=span, msaa_samples=1, z_scale=terrain_data["exaggeration"],
        exposure=exposure["value"], domain=(float(terrain.min()), float(terrain.max())),
        light_azimuth_deg=sun["azimuth_deg"], light_elevation_deg=sun["elevation_deg"],
        sun_intensity=0.0, sun_color=sun["color"], albedo_mode=material_input["albedo_mode"],
        colormap_strength=material_input["colormap_strength"], cam_radius=camera["radius"],
        cam_phi_deg=camera["phi_deg"], cam_theta_deg=camera["theta_deg"], cam_target=camera["target"],
        fov_y_deg=b1._json(repo / "tests/nephele/fixture/camera.json")["fov_y"],
        camera_mode=camera["mode"], aa_samples=1, aa_seed=0x4E455048,
        tonemap=TonemapSettings(operator="aces"),
        aov=AovSettings(enabled=True, transmittance=True, in_scatter=True, cloud_shadow=True, optical_depth=True),
        media=medium,
    )
    params = f3d.TerrainRenderParams(config)
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material = f3d.MaterialSet.custom(
        tuple(material_input["albedo"]), material_input["metallic"], material_input["roughness"],
        triplanar_scale=material_input["triplanar_scale"], normal_strength=material_input["normal_strength"],
        blend_sharpness=material_input["blend_sharpness"],
    )
    with tempfile.TemporaryDirectory(prefix="nephele-b6-environment-", ignore_cleanup_errors=True) as temporary:
        hdr = Path(temporary) / "environment.hdr"
        _write_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=atmosphere["environment_intensity"])
        capture = renderer._capture_nephele_acceptance(material, ibl, params, terrain,
            terrain_occlusion_in_media=True, include_no_medium=True, capture_radiance_provider=True)
    rgb = _crop(np.asarray(capture["beauty"], dtype=np.uint8), crop)
    np.save(output / "candidate-environment-rgb.npy", rgb, allow_pickle=False)
    np.save(output / "candidate-no-medium-environment-rgb.npy", _crop(np.asarray(capture["no_medium_beauty"]), crop), allow_pickle=False)
    np.save(output / "candidate-environment-in-scatter.npy", _crop(np.asarray(capture["in_scatter"]), crop), allow_pickle=False)
    np.save(output / "candidate-environment-transmittance.npy", _crop(np.asarray(capture["transmittance"]), crop), allow_pickle=False)
    manifest, _ = b1._frozen_fixture(repo)
    mask = np.load(b1._manifest_path(repo, manifest["files"]["cloud_shadow_mask"], "cloud shadow mask"), allow_pickle=False)
    if _camera_contract(capture["camera_contract"]) != reference_report["camera_contract"]:
        raise RuntimeError("environment candidate/reference cameras differ")
    for key in ("adapter", "backend", "driver"):
        if capture["diagnostics"][key] != reference_report["diagnostics"][key]:
            raise RuntimeError(f"environment candidate/reference {key} differ")
    reference_rgb = np.load(reference_path, allow_pickle=False)
    return {"diagnostics": dict(capture["diagnostics"]), "cloud_shadow_mask": manifest["files"]["cloud_shadow_mask"],
            "environment_under_cloud": b1._metrics(rgb, reference_rgb, mask),
            "reference_report_sha256": b1._sha256(reference_dir / "report.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("reference", "candidate"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    sys.path.insert(0, str(repo))
    os.environ.update(FORGE3D_NO_BOOTSTRAP="1", FORGE3D_TEST_INSTALLED_WHEEL="1", WGPU_BACKEND="vulkan", WGPU_BACKENDS="vulkan")
    if args.action == "candidate" and args.reference_dir is None:
        parser.error("candidate requires --reference-dir")
    output.mkdir(parents=True, exist_ok=False)
    result = reference(repo, output) if args.action == "reference" else candidate(repo, output, args.reference_dir.resolve())
    result.update(schema="forge3d.nephele.b6_environment_diagnostic/1", physical_acceptance=False,
                  sun_intensity=0.0, producer_sha256=b1._sha256(Path(__file__)),
                  fixture_manifest_sha256=b1._sha256(repo / "tests/nephele/fixture-manifest.json"),
                  repository_head=b1._git(repo, "rev-parse", "HEAD"),
                  tracked_worktree_clean=not bool(b1._git(repo, "status", "--porcelain=v1", "--untracked-files=no")))
    result["artifact_hashes"] = {p.name: b1._sha256(p) for p in output.glob("*.npy")}
    b1._save_new(output / "report.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("environment_under_cloud", {}).get("pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
