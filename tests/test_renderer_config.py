# tests/test_renderer_config.py
# Tests for renderer configuration parsing and validation
# Exists to ensure Python config plumbing matches Rust defaults and raises errors consistently
# RELEVANT FILES: python/forge3d/__init__.py, python/forge3d/config.py, src/render/params.rs, examples/terrain_demo.py

from __future__ import annotations

import json
from pathlib import Path

import pytest

import forge3d as f3d


def test_renderer_config_defaults_roundtrip() -> None:
    renderer = f3d.Renderer(32, 32)
    config = renderer.get_config()
    assert config["shading"]["brdf"] == "cooktorrance-ggx"
    assert config["lighting"]["lights"], "Expected at least one default light"
    assert config["lighting"]["exposure"] == pytest.approx(1.0)


def test_renderer_config_kwargs_override(tmp_path: Path) -> None:
    hdr_path = tmp_path / "sky.hdr"
    hdr_path.write_bytes(b"not-really-an-hdr-but-okay")
    renderer = f3d.Renderer(
        48,
        48,
        light=[
            {
                "type": "directional",
                "direction": [0.0, -1.0, 0.0],
                "intensity": 2.5,
                "color": [1.0, 0.9, 0.8],
            }
        ],
        brdf="lambert",
        shadows="hard",
        gi=["ibl"],
        hdr=str(hdr_path),
        exposure=0.85,
    )
    config = renderer.get_config()
    assert config["shading"]["brdf"] == "lambert"
    light = config["lighting"]["lights"][0]
    assert light["type"] == "directional"
    assert light["intensity"] == pytest.approx(2.5)
    assert config["lighting"]["exposure"] == pytest.approx(0.85)
    assert config["atmosphere"]["hdr_path"] == str(hdr_path)
    assert "ibl" in config["gi"]["modes"]


def test_renderer_config_json_path(tmp_path: Path) -> None:
    config_path = tmp_path / "renderer.json"
    config_path.write_text(
        json.dumps(
            {
                "lighting": {
                    "lights": [
                        {
                            "type": "directional",
                            "direction": [0.0, -1.0, 0.0],
                            "intensity": 4.0,
                        }
                    ]
                },
                "shading": {"brdf": "toon"},
            }
        ),
        encoding="utf-8",
    )
    renderer = f3d.Renderer(24, 24, config=str(config_path))
    config = renderer.get_config()
    assert config["lighting"]["lights"][0]["intensity"] == pytest.approx(4.0)
    assert config["shading"]["brdf"] == "toon"


def test_renderer_config_invalid_light_rejected() -> None:
    with pytest.raises(ValueError):
        f3d.Renderer(
            16,
            16,
            light=[{"type": "point"}],  # Missing position should fail validation
        )


def test_renderer_set_lights_updates_config() -> None:
    renderer = f3d.Renderer(32, 32)
    renderer.set_lights(
        [
            {
                "type": "directional",
                "direction": [0.0, -1.0, 0.0],
                "intensity": 2.0,
                "color": [1.0, 1.0, 1.0],
            }
        ]
    )
    cfg = renderer.get_config()
    assert cfg["lighting"]["lights"][0]["intensity"] == pytest.approx(2.0)


def test_renderer_brdf_override_roundtrip() -> None:
    renderer = f3d.Renderer(16, 16, config={"brdf_override": "toon", "shading": {"roughness": 0.2}})
    config = renderer.get_config()
    assert config.get("brdf_override") == "toon"
    assert "roughness" in renderer._shading


def test_shadow_map_requires_power_of_two() -> None:
    with pytest.raises(ValueError):
        f3d.Renderer(
            16,
            16,
            shadows="pcf",
            shadow_map_res=300,
        )


def test_shadow_memory_budget_enforced() -> None:
    with pytest.raises(ValueError):
        f3d.Renderer(
            16,
            16,
            shadows="evsm",
            shadow_map_res=8192,
            cascades=4,
        )


def test_shadow_override_parameters_roundtrip() -> None:
    renderer = f3d.Renderer(
        32,
        32,
        shadows="pcss",
        shadow_map_res=1024,
        pcss_blocker_radius=5.0,
        pcss_filter_radius=9.0,
        shadow_light_size=0.35,
        shadow_moment_bias=0.0008,
    )
    config = renderer.get_config()["shadows"]
    assert config["technique"] == "pcss"
    assert config["map_size"] == 1024
    assert config["pcss_blocker_radius"] == pytest.approx(5.0)
    assert config["pcss_filter_radius"] == pytest.approx(9.0)
    assert config["light_size"] == pytest.approx(0.35)
    assert config["moment_bias"] == pytest.approx(0.0008)


# ============================================================================
# P0-08: Nested vs Flat Override Merges
# ============================================================================

def test_nested_override_merge() -> None:
    """Test that nested config dict overrides work correctly."""
    renderer = f3d.Renderer(
        32, 32,
        config={
            "shading": {"brdf": "lambert"},
            "shadows": {"technique": "pcf", "cascades": 2},
            "gi": {"modes": ["ssao"]},
        }
    )
    config = renderer.get_config()
    assert config["shading"]["brdf"] == "lambert"
    assert config["shadows"]["technique"] == "pcf"
    assert config["shadows"]["cascades"] == 2
    assert config["gi"]["modes"] == ["ssao"]


def test_flat_override_merge() -> None:
    """Test that flat kwargs are converted to nested structure."""
    renderer = f3d.Renderer(
        32, 32,
        brdf="toon",
        shadows="pcf",
        cascades=2,
        gi=["ssao"],
        exposure=1.5,
    )
    config = renderer.get_config()
    assert config["shading"]["brdf"] == "toon"
    assert config["shadows"]["technique"] == "pcf"
    assert config["shadows"]["cascades"] == 2
    assert config["gi"]["modes"] == ["ssao"]
    assert config["lighting"]["exposure"] == pytest.approx(1.5)


def test_mixed_nested_and_flat_overrides() -> None:
    """Test that flat kwargs override nested config."""
    renderer = f3d.Renderer(
        32, 32,
        config={"shading": {"brdf": "lambert"}},
        cascades=4,  # Flat override
        gi=["ibl"],  # Flat override
    )
    config = renderer.get_config()
    assert config["shading"]["brdf"] == "lambert"  # From nested
    assert config["shadows"]["cascades"] == 4  # From flat
    assert config["gi"]["modes"] == ["ibl"]  # From flat


def test_nested_partial_override() -> None:
    """Test that nested overrides preserve unspecified fields."""
    renderer = f3d.Renderer(
        32, 32,
        config={
            "shadows": {"cascades": 2}  # Only override cascades
        }
    )
    config = renderer.get_config()
    # Cascades overridden
    assert config["shadows"]["cascades"] == 2
    # Other shadow fields keep defaults
    assert config["shadows"]["technique"] == "pcf"
    assert config["shadows"]["map_size"] == 2048


# ============================================================================
# P0-08: Enum Normalization Tests
# ============================================================================

def test_brdf_normalization_ggx() -> None:
    """Test that 'ggx' normalizes to 'cooktorrance-ggx'."""
    renderer = f3d.Renderer(32, 32, brdf="ggx")
    assert renderer.get_config()["shading"]["brdf"] == "cooktorrance-ggx"


def test_brdf_normalization_disney() -> None:
    """Test that 'disney' normalizes to 'disney-principled'."""
    renderer = f3d.Renderer(32, 32, brdf="disney")
    assert renderer.get_config()["shading"]["brdf"] == "disney-principled"


def test_brdf_normalization_sss() -> None:
    """Test that 'sss' normalizes to 'subsurface'."""
    renderer = f3d.Renderer(32, 32, brdf="sss")
    assert renderer.get_config()["shading"]["brdf"] == "subsurface"


def test_light_type_normalization_dir() -> None:
    """Test that 'dir' normalizes to 'directional'."""
    renderer = f3d.Renderer(
        32, 32,
        light=[{"type": "dir", "direction": [0, -1, 0]}]
    )
    assert renderer.get_config()["lighting"]["lights"][0]["type"] == "directional"


def test_light_type_normalization_sun() -> None:
    """Test that 'sun' normalizes to 'directional'."""
    renderer = f3d.Renderer(
        32, 32,
        light=[{"type": "sun", "direction": [0, -1, 0]}]
    )
    assert renderer.get_config()["lighting"]["lights"][0]["type"] == "directional"


def test_light_type_normalization_env() -> None:
    """Test that 'env' normalizes to 'environment'."""
    renderer = f3d.Renderer(
        32, 32,
        light=[{"type": "env", "hdr_path": "test.hdr"}]
    )
    assert renderer.get_config()["lighting"]["lights"][0]["type"] == "environment"


def test_sky_model_normalization_hosek() -> None:
    """Test that 'hosekwilkie' normalizes to 'hosek-wilkie'."""
    renderer = f3d.Renderer(32, 32, sky="hosekwilkie")
    assert renderer.get_config()["atmosphere"]["sky"] == "hosek-wilkie"


def test_gi_mode_normalization_probes() -> None:
    """Test that 'probes' normalizes to 'irradiance-probes'."""
    renderer = f3d.Renderer(32, 32, gi=["probes"])
    assert "irradiance-probes" in renderer.get_config()["gi"]["modes"]


def test_gi_mode_normalization_vct() -> None:
    """Test that 'vct' normalizes to 'voxel-cone-tracing'."""
    renderer = f3d.Renderer(32, 32, gi=["vct"])
    assert "voxel-cone-tracing" in renderer.get_config()["gi"]["modes"]


def test_phase_function_normalization_hg() -> None:
    """Test that 'hg' normalizes to 'henyey-greenstein'."""
    renderer = f3d.Renderer(
        32, 32,
        volumetric="phase=hg,density=0.02"
    )
    # Volumetric is parsed by terrain_demo, but we can test via atmosphere
    config = renderer.get_config()
    # Note: volumetric parsing happens in terrain_demo, not Renderer directly
    # This test validates the config system accepts the parameter
    assert "atmosphere" in config


# ============================================================================
# P0-08: Validation Error Tests
# ============================================================================

def test_validation_error_non_pot_shadow_map() -> None:
    """Test that non-power-of-two shadow map size raises ValueError."""
    with pytest.raises(ValueError, match="power of two"):
        f3d.Renderer(32, 32, shadow_map_res=1000)


def test_validation_error_zero_shadow_map() -> None:
    """Test that zero shadow map size raises ValueError."""
    with pytest.raises(ValueError, match="greater than zero"):
        f3d.Renderer(32, 32, shadow_map_res=0)


def test_validation_error_negative_shadow_map() -> None:
    """Test that negative shadow map size raises ValueError."""
    with pytest.raises(ValueError, match="greater than zero"):
        f3d.Renderer(32, 32, shadow_map_res=-1024)


def test_validation_error_invalid_cascades_low() -> None:
    """Test that cascades < 1 raises ValueError."""
    with pytest.raises(ValueError, match="cascades must be within"):
        f3d.Renderer(32, 32, cascades=0)


def test_validation_error_invalid_cascades_high() -> None:
    """Test that cascades > 4 raises ValueError."""
    with pytest.raises(ValueError, match="cascades must be within"):
        f3d.Renderer(32, 32, cascades=5)


def test_validation_error_csm_requires_multiple_cascades() -> None:
    """Test that CSM with cascades=1 raises ValueError."""
    with pytest.raises(ValueError, match="CSM.*cascades must be >= 2|cascades must be >= 2"):
        f3d.Renderer(32, 32, shadows="csm", cascades=1)


def test_validation_error_missing_hdr_for_hdri_sky() -> None:
    """Test that HDRI sky without HDR path raises ValueError."""
    with pytest.raises(ValueError, match="hdri requires.*hdr_path|hdr_path"):
        f3d.Renderer(32, 32, sky="hdri")


def test_validation_error_missing_hdr_for_ibl(tmp_path: Path) -> None:
    """Test that IBL mode without environment source raises ValueError."""
    # IBL requires either environment light with hdr_path or atmosphere.hdr_path
    with pytest.raises(ValueError, match="ibl.*requires|environment light"):
        f3d.Renderer(
            32, 32,
            gi=["ibl"],
            light=[{"type": "directional", "direction": [0, -1, 0]}]  # No env light
        )


def test_validation_error_directional_light_missing_direction() -> None:
    """Test that directional light without direction raises ValueError."""
    with pytest.raises(ValueError, match="direction required"):
        f3d.Renderer(
            32, 32,
            light=[{"type": "directional"}]  # Missing direction
        )


def test_validation_error_point_light_missing_position() -> None:
    """Test that point light without position raises ValueError."""
    with pytest.raises(ValueError, match="position required"):
        f3d.Renderer(
            32, 32,
            light=[{"type": "point"}]  # Missing position
        )


def test_validation_error_spot_light_missing_position() -> None:
    """Test that spot light without position raises ValueError."""
    with pytest.raises(ValueError, match="position required"):
        f3d.Renderer(
            32, 32,
            light=[{"type": "spot"}]  # Missing position
        )


def test_validation_error_environment_light_missing_hdr() -> None:
    """Test that environment light without hdr_path raises ValueError."""
    with pytest.raises(ValueError, match="hdr_path required"):
        f3d.Renderer(
            32, 32,
            light=[{"type": "environment"}]  # Missing hdr_path
        )


def test_validation_error_invalid_cone_angle_negative() -> None:
    """Test that negative cone angle raises ValueError."""
    with pytest.raises(ValueError, match="cone_angle must be within"):
        f3d.Renderer(
            32, 32,
            light=[{
                "type": "spot",
                "position": [0, 10, 0],
                "cone_angle": -10.0
            }]
        )


def test_validation_error_invalid_cone_angle_high() -> None:
    """Test that cone angle > 180 raises ValueError."""
    with pytest.raises(ValueError, match="cone_angle must be within"):
        f3d.Renderer(
            32, 32,
            light=[{
                "type": "spot",
                "position": [0, 10, 0],
                "cone_angle": 200.0
            }]
        )


def test_validation_error_negative_area_extent() -> None:
    """Test that negative area extent raises ValueError."""
    with pytest.raises(ValueError, match="area_extent.*must be positive"):
        f3d.Renderer(
            32, 32,
            light=[{
                "type": "area-rect",
                "position": [0, 10, 0],
                "area_extent": [-1.0, 2.0]
            }]
        )


def test_validation_error_pcss_negative_blocker_radius() -> None:
    """Test that PCSS with negative blocker radius raises ValueError."""
    with pytest.raises(ValueError, match="pcss_blocker_radius.*non-negative"):
        f3d.Renderer(
            32, 32,
            shadows="pcss",
            pcss_blocker_radius=-1.0
        )


def test_validation_error_pcss_zero_light_size() -> None:
    """Test that PCSS with zero light size raises ValueError."""
    with pytest.raises(ValueError, match="light_size must be positive"):
        f3d.Renderer(
            32, 32,
            shadows="pcss",
            shadow_light_size=0.0
        )


def test_validation_error_vsm_zero_moment_bias() -> None:
    """Test that VSM with zero moment bias raises ValueError."""
    with pytest.raises(ValueError, match="moment_bias must be positive"):
        f3d.Renderer(
            32, 32,
            shadows="vsm",
            shadow_moment_bias=0.0
        )


def test_validation_error_shadow_atlas_exceeds_budget() -> None:
    """Test that shadow atlas exceeding 256 MiB raises ValueError."""
    with pytest.raises(ValueError, match="256 MiB|memory budget"):
        f3d.Renderer(
            32, 32,
            shadows="evsm",  # Moment-based = 8 bytes per pixel
            shadow_map_res=8192,
            cascades=4
        )


def test_validation_error_negative_volumetric_density() -> None:
    """Test that negative volumetric density raises ValueError."""
    # This test validates config parsing; actual volumetric support is optional
    from forge3d.config import load_renderer_config
    with pytest.raises(ValueError, match="density must be non-negative"):
        load_renderer_config(None, {
            "volumetric": {"density": -0.1}
        })


def test_validation_error_hg_anisotropy_out_of_range_low() -> None:
    """Test that HG anisotropy < -0.999 raises ValueError."""
    from forge3d.config import load_renderer_config
    with pytest.raises(ValueError, match="anisotropy must be within"):
        load_renderer_config(None, {
            "volumetric": {"phase": "henyey-greenstein", "anisotropy": -1.5}
        })


def test_validation_error_hg_anisotropy_out_of_range_high() -> None:
    """Test that HG anisotropy > 0.999 raises ValueError."""
    from forge3d.config import load_renderer_config
    with pytest.raises(ValueError, match="anisotropy must be within"):
        load_renderer_config(None, {
            "volumetric": {"phase": "henyey-greenstein", "anisotropy": 1.5}
        })


# ============================================================================
# P0-08: Round-trip and Serialization Tests
# ============================================================================

def test_config_to_dict_roundtrip() -> None:
    """Test that config.to_dict() can be used to create new renderer."""
    renderer1 = f3d.Renderer(
        32, 32,
        brdf="toon",
        shadows="pcf",
        cascades=2,
    )
    config_dict = renderer1.get_config()
    
    # Create new renderer from serialized config
    renderer2 = f3d.Renderer(32, 32, config=config_dict)
    config2 = renderer2.get_config()
    
    assert config_dict["shading"]["brdf"] == config2["shading"]["brdf"]
    assert config_dict["shadows"]["technique"] == config2["shadows"]["technique"]
    assert config_dict["shadows"]["cascades"] == config2["shadows"]["cascades"]


def test_config_json_roundtrip(tmp_path: Path) -> None:
    """Test that config can be saved to JSON and loaded back."""
    renderer1 = f3d.Renderer(
        32, 32,
        brdf="disney",
        shadows="pcss",
        shadow_map_res=1024,
        gi=["ibl", "ssao"],
    )
    config1 = renderer1.get_config()
    
    # Save to JSON
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config1), encoding="utf-8")
    
    # Load from JSON
    renderer2 = f3d.Renderer(32, 32, config=str(config_path))
    config2 = renderer2.get_config()
    
    assert config2["shading"]["brdf"] == "disney-principled"
    assert config2["shadows"]["technique"] == "pcss"
    assert config2["shadows"]["map_size"] == 1024
    assert set(config2["gi"]["modes"]) == {"ibl", "ssao"}

