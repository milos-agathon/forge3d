#!/usr/bin/env python3
"""
Test heightmap input validation.
Per task.xml C3: ensure invalid heightmap inputs are rejected with clear errors.
"""
import pytest
import numpy as np


def test_invalid_heightmap_dict():
    """Test that passing a dict as heightmap raises clear error."""
    try:
        import forge3d as f3d
    except ImportError:
        pytest.skip("forge3d not available")

    sess = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(sess)

    # Create minimal test materials and IBL
    materials = f3d.MaterialSet.terrain_default(
        triplanar_scale=6.0,
        normal_strength=1.0,
        blend_sharpness=4.0,
    )

    # Create a minimal IBL (we need this for the API)
    # If from_hdr is not available, try to create a minimal IBL
    try:
        import tempfile
        import os

        # Create a minimal 1x1 HDR file for testing
        with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
            # Write minimal Radiance HDR header
            tmp.write(b"#?RADIANCE\n")
            tmp.write(b"FORMAT=32-bit_rle_rgbe\n\n")
            tmp.write(b"-Y 1 +X 1\n")
            tmp.write(b"\x01\x01\x01\x01")  # Minimal pixel data
            tmp_path = tmp.name

        ibl = f3d.IBL.from_hdr(tmp_path, intensity=1.0, rotate_deg=0.0)
        os.unlink(tmp_path)
    except Exception:
        pytest.skip("Cannot create test IBL")

    # Create minimal render params
    colormap = f3d.Colormap1D.from_stops(
        stops=[(0.0, "#000000"), (1.0, "#ffffff")],
        domain=(0.0, 1.0),
    )

    params_config = f3d.TerrainRenderParamsConfig(
        size_px=(256, 256),
        render_scale=1.0,
        msaa_samples=1,
        z_scale=1.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=100.0,
        cam_phi_deg=135.0,
        cam_theta_deg=45.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 1000.0),
        light=f3d.LightSettings(
            light_type="Directional",
            azimuth_deg=135.0,
            elevation_deg=35.0,
            intensity=1.0,
            color=[1.0, 1.0, 1.0],
        ),
        ibl=f3d.IblSettings(enabled=True, intensity=1.0, rotation_deg=0.0),
        shadows=f3d.ShadowSettings(
            enabled=False,
            technique="PCF",
            resolution=512,
            cascades=1,
            max_distance=100.0,
            softness=1.0,
            intensity=0.5,
            slope_scale_bias=0.5,
            depth_bias=0.001,
            normal_bias=0.5,
            min_variance=1e-4,
            light_bleed_reduction=0.5,
            evsm_exponent=40.0,
            fade_start=1.0,
        ),
        triplanar=f3d.TriplanarSettings(
            scale=6.0, blend_sharpness=4.0, normal_strength=1.0
        ),
        pom=f3d.PomSettings(
            enabled=False,
            mode="Occlusion",
            scale=0.04,
            min_steps=12,
            max_steps=40,
            refine_steps=4,
            shadow=False,
            occlusion=False,
        ),
        lod=f3d.LodSettings(level=0, bias=0.0, lod0_bias=0.0),
        sampling=f3d.SamplingSettings(
            mag_filter="Linear",
            min_filter="Linear",
            mip_filter="Linear",
            anisotropy=1,
            address_u="Repeat",
            address_v="Repeat",
            address_w="Repeat",
        ),
        clamp=f3d.ClampSettings(
            height_range=(0.0, 1.0),
            slope_range=(0.0, 1.0),
            ambient_range=(0.0, 1.0),
            shadow_range=(0.0, 1.0),
            occlusion_range=(0.0, 1.0),
        ),
        overlays=[
            f3d.OverlayLayer.from_colormap1d(
                colormap, strength=1.0, offset=0.0, blend_mode="Alpha", domain=(0.0, 1.0)
            )
        ],
        exposure=1.0,
        gamma=2.2,
        albedo_mode="mix",
        colormap_strength=0.5,
    )

    params = f3d.TerrainRenderParams(params_config)

    # Test 1: Pass a dict (should fail with clear error)
    invalid_heightmap_dict = {"width": 512, "height": 512, "data": None}

    with pytest.raises(ValueError) as exc_info:
        renderer.render_terrain_pbr_pom(
            material_set=materials,
            env_maps=ibl,
            params=params,
            target=None,
            heightmap=invalid_heightmap_dict,
        )

    error_msg = str(exc_info.value)
    assert "numpy array" in error_msg.lower(), f"Expected 'numpy array' in error message, got: {error_msg}"
    assert "dict" in error_msg.lower() or "invalid type" in error_msg.lower(), \
        f"Expected 'dict' or 'invalid type' in error message, got: {error_msg}"

    print(f"✓ Test passed: dict heightmap rejected with message: {error_msg}")


def test_valid_heightmap_numpy():
    """Test that passing a valid numpy array works."""
    try:
        import forge3d as f3d
    except ImportError:
        pytest.skip("forge3d not available")

    sess = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(sess)

    materials = f3d.MaterialSet.terrain_default(
        triplanar_scale=6.0,
        normal_strength=1.0,
        blend_sharpness=4.0,
    )

    try:
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
            tmp.write(b"#?RADIANCE\n")
            tmp.write(b"FORMAT=32-bit_rle_rgbe\n\n")
            tmp.write(b"-Y 1 +X 1\n")
            tmp.write(b"\x01\x01\x01\x01")
            tmp_path = tmp.name

        ibl = f3d.IBL.from_hdr(tmp_path, intensity=1.0, rotate_deg=0.0)
        os.unlink(tmp_path)
    except Exception:
        pytest.skip("Cannot create test IBL")

    colormap = f3d.Colormap1D.from_stops(
        stops=[(0.0, "#000000"), (1.0, "#ffffff")],
        domain=(0.0, 1.0),
    )

    params_config = f3d.TerrainRenderParamsConfig(
        size_px=(64, 64),
        render_scale=1.0,
        msaa_samples=1,
        z_scale=1.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=100.0,
        cam_phi_deg=135.0,
        cam_theta_deg=45.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 1000.0),
        light=f3d.LightSettings(
            light_type="Directional",
            azimuth_deg=135.0,
            elevation_deg=35.0,
            intensity=1.0,
            color=[1.0, 1.0, 1.0],
        ),
        ibl=f3d.IblSettings(enabled=True, intensity=1.0, rotation_deg=0.0),
        shadows=f3d.ShadowSettings(
            enabled=False,
            technique="PCF",
            resolution=512,
            cascades=1,
            max_distance=100.0,
            softness=1.0,
            intensity=0.5,
            slope_scale_bias=0.5,
            depth_bias=0.001,
            normal_bias=0.5,
            min_variance=1e-4,
            light_bleed_reduction=0.5,
            evsm_exponent=40.0,
            fade_start=1.0,
        ),
        triplanar=f3d.TriplanarSettings(
            scale=6.0, blend_sharpness=4.0, normal_strength=1.0
        ),
        pom=f3d.PomSettings(
            enabled=False,
            mode="Occlusion",
            scale=0.04,
            min_steps=12,
            max_steps=40,
            refine_steps=4,
            shadow=False,
            occlusion=False,
        ),
        lod=f3d.LodSettings(level=0, bias=0.0, lod0_bias=0.0),
        sampling=f3d.SamplingSettings(
            mag_filter="Linear",
            min_filter="Linear",
            mip_filter="Linear",
            anisotropy=1,
            address_u="Repeat",
            address_v="Repeat",
            address_w="Repeat",
        ),
        clamp=f3d.ClampSettings(
            height_range=(0.0, 1000.0),
            slope_range=(0.0, 1.0),
            ambient_range=(0.0, 1.0),
            shadow_range=(0.0, 1.0),
            occlusion_range=(0.0, 1.0),
        ),
        overlays=[
            f3d.OverlayLayer.from_colormap1d(
                colormap, strength=1.0, offset=0.0, blend_mode="Alpha", domain=(0.0, 1000.0)
            )
        ],
        exposure=1.0,
        gamma=2.2,
        albedo_mode="mix",
        colormap_strength=0.5,
    )

    params = f3d.TerrainRenderParams(params_config)

    # Test 2: Pass a valid numpy array (should succeed)
    valid_heightmap = np.random.rand(128, 128).astype(np.float32) * 1000.0

    try:
        frame = renderer.render_terrain_pbr_pom(
            material_set=materials,
            env_maps=ibl,
            params=params,
            target=None,
            heightmap=valid_heightmap,
        )
        assert frame is not None
        assert frame.shape == (64, 64, 4)  # H, W, C
        assert frame.dtype == np.uint8
        print("✓ Test passed: valid numpy heightmap accepted")
    except Exception as e:
        pytest.fail(f"Valid heightmap should be accepted, but got error: {e}")


if __name__ == "__main__":
    print("Testing heightmap validation...")
    test_invalid_heightmap_dict()
    test_valid_heightmap_numpy()
    print("\nAll tests passed!")
