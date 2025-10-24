# tests/test_terrain_render_color_space.py
# Test that terrain rendering produces correct color-space output without horizontal banding

import numpy as np
import pytest

forge3d = pytest.importorskip("forge3d")


def test_terrain_render_no_horizontal_banding():
    """Test that terrain rendering doesn't produce horizontal banding artifacts."""
    # Create session and renderer
    sess = forge3d.Session(window=False)
    renderer = forge3d.TerrainRenderer(sess)

    # Create a flat heightmap at constant elevation
    heights = np.ones((256, 256), dtype=np.float32) * 1500.0

    # Create materials and params
    materials = forge3d.MaterialSet.terrain_default()

    # Create simple IBL (just use default if available)
    try:
        ibl = forge3d.IBL.from_hdr("assets/snow_field_4k.hdr", intensity=1.0)
    except Exception:
        # If HDR not available, skip this test
        pytest.skip("HDR file not available for testing")

    # Create rendering params with full configuration
    from forge3d import (TerrainRenderParamsConfig, LightSettings, IblSettings,
                          ShadowSettings, TriplanarSettings, PomSettings,
                          LodSettings, SamplingSettings, ClampSettings)
    config = TerrainRenderParamsConfig(
        size_px=(512, 512),
        render_scale=1.0,
        msaa_samples=1,
        z_scale=1.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=1200.0,
        cam_phi_deg=135.0,
        cam_theta_deg=45.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 6000.0),
        light=LightSettings("Directional", 135.0, 35.0, 3.0, [1.0, 1.0, 1.0]),
        ibl=IblSettings(True, 1.0, 0.0),
        shadows=ShadowSettings(False, "PCSS", 2048, 3, 2000.0, 1.0, 0.8, 0.5, 0.002, 0.5, 1e-4, 0.5, 40.0, 1.0),
        triplanar=TriplanarSettings(6.0, 4.0, 1.0),
        pom=PomSettings(False, "Occlusion", 0.04, 8, 24, 2, False, False),
        lod=LodSettings(0, 0.0, -0.5),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 4, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 3000.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[],
        exposure=1.0,
        gamma=2.2,
        albedo_mode="mix",
        colormap_strength=0.5,
    )
    params = forge3d.TerrainRenderParams(config)

    # Render
    frame = renderer.render_terrain_pbr_pom(
        material_set=materials,
        env_maps=ibl,
        params=params,
        target=None,
        heightmap=heights,
    )

    # Verify output shape and type
    assert frame.shape == (512, 512, 4), f"Expected (512, 512, 4), got {frame.shape}"
    assert frame.dtype == np.uint8, f"Expected uint8, got {frame.dtype}"

    # Check that the image is not too dark (mean should be > 100)
    mean_brightness = frame[:, :, :3].mean()
    assert mean_brightness > 100, f"Image too dark: mean={mean_brightness:.1f}"

    # Check for horizontal banding by analyzing row consistency
    # Calculate mean brightness for each row
    row_means = frame[:, :, :3].mean(axis=(1, 2))

    # The middle 80% of rows should have similar values (ignore edges)
    middle_start = int(512 * 0.1)
    middle_end = int(512 * 0.9)
    middle_rows = row_means[middle_start:middle_end]

    # Standard deviation of row means should be small (< 10) for uniform rendering
    row_std = middle_rows.std()
    assert row_std < 10, f"Horizontal banding detected: row std={row_std:.2f}"

    # Check that we have reasonable color variation (not all one color)
    color_std = frame[:, :, :3].std()
    assert color_std > 1.0, f"No color variation: std={color_std:.2f}"


def test_terrain_render_color_space_correct():
    """Test that color-space conversion is working correctly."""
    sess = forge3d.Session(window=False)
    renderer = forge3d.TerrainRenderer(sess)

    # Create a flat heightmap
    heights = np.ones((128, 128), dtype=np.float32) * 1500.0

    materials = forge3d.MaterialSet.terrain_default()

    try:
        ibl = forge3d.IBL.from_hdr("assets/snow_field_4k.hdr", intensity=1.0)
    except Exception:
        pytest.skip("HDR file not available")

    from forge3d import (TerrainRenderParamsConfig, LightSettings, IblSettings,
                          ShadowSettings, TriplanarSettings, PomSettings,
                          LodSettings, SamplingSettings, ClampSettings)
    config = TerrainRenderParamsConfig(
        size_px=(256, 256),
        render_scale=1.0,
        msaa_samples=1,
        z_scale=1.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=1200.0,
        cam_phi_deg=135.0,
        cam_theta_deg=45.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 6000.0),
        light=LightSettings("Directional", 135.0, 35.0, 3.0, [1.0, 1.0, 1.0]),
        ibl=IblSettings(True, 1.0, 0.0),
        shadows=ShadowSettings(False, "PCSS", 2048, 3, 2000.0, 1.0, 0.8, 0.5, 0.002, 0.5, 1e-4, 0.5, 40.0, 1.0),
        triplanar=TriplanarSettings(6.0, 4.0, 1.0),
        pom=PomSettings(False, "Occlusion", 0.04, 8, 24, 2, False, False),
        lod=LodSettings(0, 0.0, -0.5),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 4, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 3000.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[],
        exposure=2.0,
        gamma=2.2,
        albedo_mode="mix",
        colormap_strength=0.5,
    )
    params = forge3d.TerrainRenderParams(config)

    frame = renderer.render_terrain_pbr_pom(
        material_set=materials,
        env_maps=ibl,
        params=params,
        target=None,
        heightmap=heights,
    )

    # With correct color-space handling, we should get reasonably bright mid-tones
    # Linear 0.35 (rock gray) → sRGB ~0.62 → u8 ~158
    # With lighting and exposure, should be in 100-200 range
    mean = frame[:, :, :3].mean()
    assert 100 < mean < 230, f"Color-space issue: mean={mean:.1f} (expected 100-230)"

    # Check that we're not clipped to white or black
    assert frame[:, :, :3].min() > 30, "Too much black clipping"
    assert frame[:, :, :3].max() < 250, "Too much white clipping"


def test_terrain_render_non_aligned_dimensions():
    """Test that non-256-aligned dimensions work correctly (padding test)."""
    sess = forge3d.Session(window=False)
    renderer = forge3d.TerrainRenderer(sess)

    # Use odd dimensions that aren't 256-aligned
    heights = np.ones((127, 127), dtype=np.float32) * 1500.0

    materials = forge3d.MaterialSet.terrain_default()

    try:
        ibl = forge3d.IBL.from_hdr("assets/snow_field_4k.hdr", intensity=1.0)
    except Exception:
        pytest.skip("HDR file not available")

    from forge3d import (TerrainRenderParamsConfig, LightSettings, IblSettings,
                          ShadowSettings, TriplanarSettings, PomSettings,
                          LodSettings, SamplingSettings, ClampSettings)
    config = TerrainRenderParamsConfig(
        size_px=(253, 251),  # Odd primes to stress padding
        render_scale=1.0,
        msaa_samples=1,
        z_scale=1.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=1200.0,
        cam_phi_deg=135.0,
        cam_theta_deg=45.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 6000.0),
        light=LightSettings("Directional", 135.0, 35.0, 3.0, [1.0, 1.0, 1.0]),
        ibl=IblSettings(True, 1.0, 0.0),
        shadows=ShadowSettings(False, "PCSS", 2048, 3, 2000.0, 1.0, 0.8, 0.5, 0.002, 0.5, 1e-4, 0.5, 40.0, 1.0),
        triplanar=TriplanarSettings(6.0, 4.0, 1.0),
        pom=PomSettings(False, "Occlusion", 0.04, 8, 24, 2, False, False),
        lod=LodSettings(0, 0.0, -0.5),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 4, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 3000.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[],
        exposure=1.0,
        gamma=2.2,
        albedo_mode="mix",
        colormap_strength=0.5,
    )
    params = forge3d.TerrainRenderParams(config)

    frame = renderer.render_terrain_pbr_pom(
        material_set=materials,
        env_maps=ibl,
        params=params,
        target=None,
        heightmap=heights,
    )

    # Verify correct output shape
    assert frame.shape == (251, 253, 4), f"Expected (251, 253, 4), got {frame.shape}"

    # Check for horizontal artifacts from padding issues
    row_means = frame[:, :, :3].mean(axis=(1, 2))
    middle_rows = row_means[50:200]
    row_std = middle_rows.std()
    assert row_std < 10, f"Padding artifacts detected: row std={row_std:.2f}"
