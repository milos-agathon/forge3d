"""Bisect the lavapipe no-medium capture crash by varying one feature per run.

Run from the repo root inside WSL with MESA_SHADER_CACHE_DISABLE=true:
    python -X faulthandler lvp_bisect.py <variant>
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/d/forge3d/.worktrees/nephele-172-completion")
sys.path.insert(0, str(ROOT))

from scripts.nephele_b6_environment_diagnostic import scene  # noqa: E402
from scripts.run_media_physical_capture import _write_hdr  # noqa: E402


def run(variant: str) -> None:
    import forge3d as f3d
    from forge3d.terrain_params import AovSettings, TonemapSettings, make_terrain_params_config

    camera, terrain_data, material_input, sun, atmosphere, exposure, crop, terrain, medium = scene(ROOT)
    cam = camera["terrain_camera"]
    aov = AovSettings(enabled=True, transmittance=True, in_scatter=True, cloud_shadow=True, optical_depth=True)
    shading = "lambert_physical"
    if variant == "no_media_aovs":
        aov = AovSettings(enabled=True)
    media_flags = dict(transmittance=True, in_scatter=True, cloud_shadow=True, optical_depth=True)
    if variant == "media_only_source_id":
        aov = AovSettings(enabled=True, source_id=True, **media_flags)
    if variant == "media_only_no_aux":
        aov = AovSettings(enabled=True, albedo=False, normal=False, depth=False, **media_flags)
    if variant.startswith("media_only"):
        variant_no_medium = False
    else:
        variant_no_medium = True
    if variant == "stylized":
        shading = "stylized"
    params = f3d.TerrainRenderParams(make_terrain_params_config(
        terrain_shading_model=shading, size_px=tuple(crop["full_viewport"]),
        render_scale=1.0, terrain_span=terrain_data["spacing"][0] * (terrain_data["dimensions"][0] - 1),
        msaa_samples=1, z_scale=terrain_data["exaggeration"], exposure=exposure["value"],
        domain=(float(terrain.min()), float(terrain.max())),
        light_azimuth_deg=sun["azimuth_deg"], light_elevation_deg=sun["elevation_deg"],
        sun_intensity=sun["intensity"], sun_color=sun["color"],
        albedo_mode=material_input["albedo_mode"], colormap_strength=material_input["colormap_strength"],
        cam_radius=cam["radius"], cam_phi_deg=cam["phi_deg"], cam_theta_deg=cam["theta_deg"],
        cam_target=cam["target"], fov_y_deg=camera["fov_y"], camera_mode=cam["mode"],
        aa_samples=1, aa_seed=0x4E455048, tonemap=TonemapSettings(operator="aces"),
        aov=aov, media=medium,
    ))
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material = f3d.MaterialSet.custom(tuple(material_input["albedo"]), material_input["metallic"],
        material_input["roughness"], triplanar_scale=material_input["triplanar_scale"],
        normal_strength=material_input["normal_strength"], blend_sharpness=material_input["blend_sharpness"])
    with tempfile.TemporaryDirectory() as temporary:
        hdr = Path(temporary) / "environment.hdr"
        _write_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=atmosphere["environment_intensity"])
        if variant.startswith("nomedia"):
            base = make_terrain_params_config(
                terrain_shading_model=shading, size_px=tuple(crop["full_viewport"]),
                render_scale=1.0, terrain_span=terrain_data["spacing"][0] * (terrain_data["dimensions"][0] - 1),
                msaa_samples=1, z_scale=terrain_data["exaggeration"], exposure=exposure["value"],
                domain=(float(terrain.min()), float(terrain.max())),
                light_azimuth_deg=sun["azimuth_deg"], light_elevation_deg=sun["elevation_deg"],
                sun_intensity=sun["intensity"], sun_color=sun["color"],
                albedo_mode=material_input["albedo_mode"], colormap_strength=material_input["colormap_strength"],
                cam_radius=cam["radius"], cam_phi_deg=cam["phi_deg"], cam_theta_deg=cam["theta_deg"],
                cam_target=cam["target"], fov_y_deg=camera["fov_y"],
                camera_mode=cam["mode"] if "screencam" not in variant else "screen",
                aa_samples=1, aa_seed=0x4E455048, tonemap=TonemapSettings(operator="aces"),
                aov=AovSettings(enabled=True), media=None,
            )
            plain = f3d.TerrainRenderParams(base)
            if "aov" in variant:
                renderer.render_with_aov(material, ibl, plain, terrain)
            else:
                renderer.render_terrain_pbr_pom(material, ibl, plain, terrain)
            print("OK", variant, flush=True)
            return
        include_no_medium = variant_no_medium
        if variant == "media_twice":
            renderer._capture_nephele_acceptance(material, ibl, params, terrain, include_no_medium=False)
            include_no_medium = False
        raw = renderer._capture_nephele_acceptance(material, ibl, params, terrain,
                                                   include_no_medium=include_no_medium)
    keys = sorted(k for k in raw if isinstance(raw[k], np.ndarray))
    print("OK", variant, keys, flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
