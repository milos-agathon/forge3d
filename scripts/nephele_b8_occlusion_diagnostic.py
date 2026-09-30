"""Capture the terrain-occlusion ablation with the frozen physical scene."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from scripts import nephele_b1_diagnostic as b1
from scripts.nephele_b6_environment_diagnostic import scene


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, primary, output = args.repo.resolve(), args.primary_dir.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace diagnostic output: {output}")
    os.environ.update(FORGE3D_NO_BOOTSTRAP="1", FORGE3D_TEST_INSTALLED_WHEEL="1", WGPU_BACKEND="vulkan", WGPU_BACKENDS="vulkan")
    import forge3d as f3d
    from forge3d.terrain_params import AovSettings, TonemapSettings, make_terrain_params_config
    from scripts.run_media_physical_capture import _camera_contract, _crop, _write_hdr
    from scripts.nephele_evidence_report import _roi_crop
    from tests._ssim import ssim

    camera_input, terrain_data, material_input, sun, atmosphere, exposure, crop, terrain, medium = scene(repo)
    camera = camera_input["terrain_camera"]
    config = make_terrain_params_config(
        terrain_shading_model="lambert_physical", size_px=tuple(crop["full_viewport"]), render_scale=1.0,
        terrain_span=terrain_data["spacing"][0] * (terrain_data["dimensions"][0] - 1),
        msaa_samples=1, z_scale=terrain_data["exaggeration"], exposure=exposure["value"],
        domain=(float(terrain.min()), float(terrain.max())), light_azimuth_deg=sun["azimuth_deg"],
        light_elevation_deg=sun["elevation_deg"], sun_intensity=sun["intensity"], sun_color=sun["color"],
        albedo_mode=material_input["albedo_mode"], colormap_strength=material_input["colormap_strength"],
        cam_radius=camera["radius"], cam_phi_deg=camera["phi_deg"], cam_theta_deg=camera["theta_deg"],
        cam_target=camera["target"], fov_y_deg=camera_input["fov_y"], camera_mode=camera["mode"],
        aa_samples=1, aa_seed=0x4E455048, tonemap=TonemapSettings(operator="aces"),
        aov=AovSettings(enabled=True, transmittance=True, in_scatter=True, cloud_shadow=True, optical_depth=True),
        media=medium,
    )
    params = f3d.TerrainRenderParams(config)
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material = f3d.MaterialSet.custom(tuple(material_input["albedo"]), material_input["metallic"], material_input["roughness"],
        triplanar_scale=material_input["triplanar_scale"], normal_strength=material_input["normal_strength"], blend_sharpness=material_input["blend_sharpness"])
    with tempfile.TemporaryDirectory(prefix="nephele-b8-", ignore_cleanup_errors=True) as temporary:
        hdr = Path(temporary) / "environment.hdr"
        _write_hdr(hdr)
        capture = renderer._capture_nephele_acceptance(material, f3d.IBL.from_hdr(str(hdr), intensity=atmosphere["environment_intensity"]),
            params, terrain, terrain_occlusion_in_media=False)
    diagnostics = dict(capture["diagnostics"])
    baseline = b1._json(primary / "candidate-diagnostics.json")
    for key in ("source_revision", "adapter", "backend", "driver", "terrain_shading_model"):
        if diagnostics[key] != baseline[key]:
            raise RuntimeError(f"occlusion-ablation {key} differs from primary")
    if _camera_contract(capture["camera_contract"]) != b1._json(primary / "camera-contract.json"):
        raise RuntimeError("occlusion-ablation camera differs from primary")
    manifest, _ = b1._frozen_fixture(repo)
    roi = np.load(b1._manifest_path(repo, manifest["files"]["godray_roi_mask"], "godray ROI"), allow_pickle=False)
    enabled = np.load(primary / "realtime-rgb.npy", allow_pickle=False)
    disabled = _crop(np.asarray(capture["beauty"], dtype=np.uint8), crop)
    enabled_roi, disabled_roi = _roi_crop(roi, enabled, disabled)
    score = float(ssim(enabled_roi, disabled_roi, data_range=255.0))
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "terrain-occlusion-disabled-rgb.npy", disabled, allow_pickle=False)
    report = {"schema": "forge3d.nephele.b8_diagnostic/1", "physical_acceptance": False,
        "terrain_occlusion_in_media": False, "diagnostics": diagnostics,
        "primary_frame_sha256": b1._sha256(primary / "realtime-rgb.npy"),
        "disabled_frame_sha256": b1._sha256(output / "terrain-occlusion-disabled-rgb.npy"),
        "producer_sha256": b1._sha256(Path(__file__)),
        "godray_roi_mask": manifest["files"]["godray_roi_mask"],
        "terrain_occlusion_ablation_ssim": score, "criterion": {"strict_maximum": 0.80, "pass": score < 0.80}}
    b1._save_new(output / "report.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0 if score < 0.80 else 1


if __name__ == "__main__":
    raise SystemExit(main())
