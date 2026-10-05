"""Measure shadow-mask scattering components without changing the frozen fixture."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from scripts import nephele_b1_diagnostic as b1
from scripts.nephele_b6_environment_diagnostic import scene
from scripts.run_media_physical_capture import _camera_contract, _crop, _write_hdr

LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)


def capture(repo: Path, output: Path, label: str, *, isotropic=False, occlusion=True, include_no_medium=False):
    import forge3d as f3d
    from forge3d.media import Medium
    from forge3d.terrain_params import AovSettings, TonemapSettings, make_terrain_params_config

    camera, terrain_data, material_input, sun, atmosphere, exposure, crop, terrain, medium = scene(repo)
    if isotropic:
        data = b1._json(repo / "tests/nephele/fixture/medium.json")
        shape = data["domain"]["grid_shape"]
        density = np.asarray(data["density_r16"], dtype=np.float32).reshape(shape[2], shape[1], shape[0]) / 65535.0
        medium = Medium.grid3d(data["sigma_a"], data["sigma_s"], density,
            (data["domain"]["bounds_min"], data["domain"]["bounds_max"]),
            phase="isotropic", density_scale=data["density_scale"], version=1)
    cam = camera["terrain_camera"]
    params = f3d.TerrainRenderParams(make_terrain_params_config(
        terrain_shading_model="lambert_physical", size_px=tuple(crop["full_viewport"]),
        render_scale=1.0, terrain_span=terrain_data["spacing"][0] * (terrain_data["dimensions"][0] - 1),
        msaa_samples=1, z_scale=terrain_data["exaggeration"], exposure=exposure["value"],
        domain=(float(terrain.min()), float(terrain.max())),
        light_azimuth_deg=sun["azimuth_deg"], light_elevation_deg=sun["elevation_deg"],
        sun_intensity=sun["intensity"], sun_color=sun["color"],
        albedo_mode=material_input["albedo_mode"], colormap_strength=material_input["colormap_strength"],
        cam_radius=cam["radius"], cam_phi_deg=cam["phi_deg"], cam_theta_deg=cam["theta_deg"],
        cam_target=cam["target"], fov_y_deg=camera["fov_y"], camera_mode=cam["mode"],
        aa_samples=1, aa_seed=0x4E455048, tonemap=TonemapSettings(operator="aces"),
        aov=AovSettings(enabled=True, transmittance=True, in_scatter=True, cloud_shadow=True, optical_depth=True),
        media=medium,
    ))
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material = f3d.MaterialSet.custom(tuple(material_input["albedo"]), material_input["metallic"],
        material_input["roughness"], triplanar_scale=material_input["triplanar_scale"],
        normal_strength=material_input["normal_strength"], blend_sharpness=material_input["blend_sharpness"])
    with tempfile.TemporaryDirectory(prefix="nephele-shadow-", ignore_cleanup_errors=True) as temporary:
        hdr = Path(temporary) / "environment.hdr"
        _write_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=atmosphere["environment_intensity"])
        capture_components = not include_no_medium
        raw = renderer._capture_nephele_acceptance(material, ibl, params, terrain,
            terrain_occlusion_in_media=occlusion, capture_scatter_components=capture_components,
            include_no_medium=include_no_medium)
    expected_camera = b1._json(repo / "tests/nephele/fixture/reference-provenance.json")["camera_contract"]
    if _camera_contract(raw["camera_contract"]) != expected_camera:
        raise RuntimeError("shadow diagnostic camera differs from the frozen reference")
    keys = ("beauty", "in_scatter", "transmittance", "cloud_shadow")
    if capture_components:
        keys += ("in_scatter_multiple_luminance",)
    if include_no_medium:
        keys += ("no_medium_beauty",)
    arrays = {key: _crop(np.asarray(raw[key]), crop) for key in keys}
    for key, array in arrays.items():
        if not np.isfinite(array).all():
            raise RuntimeError(f"non-finite GPU readback: {key}")
        np.save(output / f"{label}-{key}.npy", array, allow_pickle=False)
    # The alpha channel and the native aggregate must describe the same GPU frame.
    if capture_components:
        np.testing.assert_allclose(float(arrays["in_scatter_multiple_luminance"].mean()),
            raw["diagnostics"]["multiple_scatter_luminance"], rtol=0, atol=np.finfo(np.float32).eps)
    return arrays, raw["diagnostics"]


def measure(repo: Path, output: Path) -> dict:
    fixture = repo / "tests/nephele/fixture"
    mask = np.load(fixture / "cloud-shadow-mask.npy", allow_pickle=False).astype(bool)
    mask &= np.load(fixture / "terrain-mask.npy", allow_pickle=False).astype(bool)
    reference = np.load(fixture / "reference-in-scatter.npy", allow_pickle=False)
    captures = {}
    metrics = {}
    for label, kwargs in [("primary", {}), ("isotropic", {"isotropic": True}),
                           ("occlusion-disabled", {"occlusion": False})]:
        arrays, diagnostics = capture(repo, output, label, **kwargs)
        captures[label] = arrays
        total = arrays["in_scatter"].astype(np.float64) @ LUMA
        multiple = arrays["in_scatter_multiple_luminance"].astype(np.float64)
        metrics[label] = {
            "shadow_in_scatter_mean_rgb": arrays["in_scatter"][mask].mean(axis=0).tolist(),
            "shadow_total_luminance": float(total[mask].mean()),
            "shadow_multiple_luminance": float(multiple[mask].mean()),
            "shadow_single_luminance": float((total - multiple)[mask].mean()),
            "diagnostics": dict(diagnostics),
        }
    metrics["reference_shadow_total_luminance"] = float((reference.astype(np.float64) @ LUMA)[mask].mean())
    metrics["reference_shadow_mean_rgb"] = reference[mask].mean(axis=0).tolist()
    primary = metrics["primary"]["shadow_total_luminance"]
    metrics["primary_reference_ratio"] = primary / metrics["reference_shadow_total_luminance"]
    metrics["occlusion_shadow_luminance_change"] = metrics["occlusion-disabled"]["shadow_total_luminance"] - primary
    metrics["isotropic_shadow_luminance_change"] = metrics["isotropic"]["shadow_total_luminance"] - primary
    metrics["mask_pixels"] = int(mask.sum())
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--background-only", action="store_true",
        help="Read the primary frame and its no-medium camera background without a B1 control registration.")
    args = parser.parse_args()
    os.environ.update(FORGE3D_NO_BOOTSTRAP="1", FORGE3D_TEST_INSTALLED_WHEEL="1",
        WGPU_BACKEND="vulkan", WGPU_BACKENDS="vulkan")
    repo, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.background_only:
        _, diagnostics = capture(repo, output, "primary", include_no_medium=True)
        result = {"camera_background_capture": True, "diagnostics": dict(diagnostics)}
    else:
        result = measure(repo, output)
    result.update(schema="forge3d.nephele.shadow_inscatter_diagnostic/1", physical_acceptance=False,
        repository_head=b1._git(repo, "rev-parse", "HEAD"),
        tracked_worktree_clean=not bool(b1._git(repo, "status", "--porcelain=v1", "--untracked-files=no")),
        producer_sha256=b1._sha256(Path(__file__)),
        fixture_manifest_sha256=b1._sha256(repo / "tests/nephele/fixture-manifest.json"))
    result["artifact_hashes"] = {p.name: b1._sha256(p) for p in output.glob("*.npy")}
    b1._save_new(output / "report.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
