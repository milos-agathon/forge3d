# tests/test_terrain_overlay_stack.py
# Unit tests for terrain overlay layer stack and Python API
# Validates: OverlayLayerConfig, OverlaySettings, blend modes, z-ordering

import pytest
import numpy as np
from pathlib import Path

from forge3d.terrain_params import (
    OverlayBlendMode,
    OverlayLayerConfig,
    OverlaySettings,
    TerrainRenderParams,
    make_terrain_params_config,
)


class TestOverlayBlendMode:
    """Tests for OverlayBlendMode constants."""

    def test_blend_mode_values(self):
        """Verify blend mode constants have expected string values."""
        assert OverlayBlendMode.NORMAL == "normal"
        assert OverlayBlendMode.MULTIPLY == "multiply"
        assert OverlayBlendMode.OVERLAY == "overlay"


class TestOverlayLayerConfig:
    """Tests for OverlayLayerConfig dataclass."""

    def test_default_values(self):
        """Verify default values are correct."""
        layer = OverlayLayerConfig(name="test", source="test.png")
        assert layer.name == "test"
        assert layer.source == "test.png"
        assert layer.extent is None
        assert layer.opacity == 1.0
        assert layer.blend_mode == "normal"
        assert layer.visible is True
        assert layer.z_order == 0

    def test_custom_values(self):
        """Verify custom values are accepted."""
        layer = OverlayLayerConfig(
            name="satellite",
            source="satellite.png",
            extent=(0.1, 0.2, 0.8, 0.9),
            opacity=0.7,
            blend_mode="multiply",
            visible=False,
            z_order=5,
        )
        assert layer.name == "satellite"
        assert layer.extent == (0.1, 0.2, 0.8, 0.9)
        assert layer.opacity == 0.7
        assert layer.blend_mode == "multiply"
        assert layer.visible is False
        assert layer.z_order == 5

    def test_empty_name_rejected(self):
        """Verify empty name raises ValueError."""
        with pytest.raises(ValueError, match="name must be non-empty"):
            OverlayLayerConfig(name="", source="test.png")

    def test_opacity_range_validation(self):
        """Verify opacity must be in [0, 1]."""
        with pytest.raises(ValueError, match="opacity must be in"):
            OverlayLayerConfig(name="test", source="test.png", opacity=-0.1)
        with pytest.raises(ValueError, match="opacity must be in"):
            OverlayLayerConfig(name="test", source="test.png", opacity=1.5)
        # Edge cases should work
        layer_min = OverlayLayerConfig(name="test", source="test.png", opacity=0.0)
        layer_max = OverlayLayerConfig(name="test", source="test.png", opacity=1.0)
        assert layer_min.opacity == 0.0
        assert layer_max.opacity == 1.0

    def test_invalid_blend_mode_rejected(self):
        """Verify invalid blend mode raises ValueError."""
        with pytest.raises(ValueError, match="blend_mode must be one of"):
            OverlayLayerConfig(name="test", source="test.png", blend_mode="invalid")
        with pytest.raises(ValueError, match="blend_mode must be one of"):
            OverlayLayerConfig(name="test", source="test.png", blend_mode="add")

    def test_valid_blend_modes_accepted(self):
        """Verify all valid blend modes are accepted."""
        for mode in ["normal", "multiply", "overlay"]:
            layer = OverlayLayerConfig(name="test", source="test.png", blend_mode=mode)
            assert layer.blend_mode == mode

    def test_extent_validation(self):
        """Verify extent validation."""
        # Invalid length
        with pytest.raises(ValueError, match="extent must be"):
            OverlayLayerConfig(name="test", source="test.png", extent=(0.0, 0.0, 1.0))
        # u_min >= u_max
        with pytest.raises(ValueError, match="u_min < u_max"):
            OverlayLayerConfig(name="test", source="test.png", extent=(0.5, 0.0, 0.5, 1.0))
        # v_min >= v_max
        with pytest.raises(ValueError, match="v_min < v_max"):
            OverlayLayerConfig(name="test", source="test.png", extent=(0.0, 0.5, 1.0, 0.5))
        # Valid extent
        layer = OverlayLayerConfig(name="test", source="test.png", extent=(0.0, 0.0, 1.0, 1.0))
        assert layer.extent == (0.0, 0.0, 1.0, 1.0)


class TestOverlaySettings:
    """Tests for OverlaySettings dataclass."""

    def test_default_values(self):
        """Verify default values for backward compatibility."""
        settings = OverlaySettings()
        assert settings.enabled is False  # Default OFF for backward compatibility
        assert settings.global_opacity == 1.0
        assert settings.layers == []
        assert settings.resolution_scale == 1.0

    def test_enabled_can_be_set(self):
        """Verify enabled can be explicitly set."""
        settings = OverlaySettings(enabled=True)
        assert settings.enabled is True

    def test_global_opacity_validation(self):
        """Verify global_opacity must be in [0, 1]."""
        with pytest.raises(ValueError, match="global_opacity must be in"):
            OverlaySettings(global_opacity=-0.1)
        with pytest.raises(ValueError, match="global_opacity must be in"):
            OverlaySettings(global_opacity=1.5)

    def test_resolution_scale_validation(self):
        """Verify resolution_scale must be in [0.1, 2.0]."""
        with pytest.raises(ValueError, match="resolution_scale must be in"):
            OverlaySettings(resolution_scale=0.05)
        with pytest.raises(ValueError, match="resolution_scale must be in"):
            OverlaySettings(resolution_scale=3.0)
        # Edge cases
        settings_min = OverlaySettings(resolution_scale=0.1)
        settings_max = OverlaySettings(resolution_scale=2.0)
        assert settings_min.resolution_scale == 0.1
        assert settings_max.resolution_scale == 2.0

    def test_layers_default_empty(self):
        """Verify layers defaults to empty list when None."""
        settings = OverlaySettings(layers=None)
        assert settings.layers == []

    def test_has_visible_layers_empty(self):
        """Verify has_visible_layers is False when no layers."""
        settings = OverlaySettings()
        assert settings.has_visible_layers is False

    def test_has_visible_layers_with_visible_layer(self):
        """Verify has_visible_layers is True when visible layers exist."""
        layer = OverlayLayerConfig(name="test", source="test.png", visible=True, opacity=0.5)
        settings = OverlaySettings(layers=[layer])
        assert settings.has_visible_layers is True

    def test_has_visible_layers_hidden_layer(self):
        """Verify has_visible_layers is False when all layers hidden."""
        layer = OverlayLayerConfig(name="test", source="test.png", visible=False)
        settings = OverlaySettings(layers=[layer])
        assert settings.has_visible_layers is False

    def test_has_visible_layers_zero_opacity(self):
        """Verify has_visible_layers is False when opacity is zero."""
        layer = OverlayLayerConfig(name="test", source="test.png", visible=True, opacity=0.0)
        settings = OverlaySettings(layers=[layer])
        assert settings.has_visible_layers is False

    def test_layer_count(self):
        """Verify layer_count property."""
        settings_empty = OverlaySettings()
        assert settings_empty.layer_count == 0
        
        layer1 = OverlayLayerConfig(name="layer1", source="a.png")
        layer2 = OverlayLayerConfig(name="layer2", source="b.png")
        settings = OverlaySettings(layers=[layer1, layer2])
        assert settings.layer_count == 2


class TestOverlayLayerOrdering:
    """Tests for overlay layer z-ordering."""

    def test_z_order_sorting(self):
        """Verify layers can be sorted by z_order."""
        layers = [
            OverlayLayerConfig(name="top", source="top.png", z_order=10),
            OverlayLayerConfig(name="middle", source="mid.png", z_order=5),
            OverlayLayerConfig(name="bottom", source="bot.png", z_order=0),
        ]
        sorted_layers = sorted(layers, key=lambda l: l.z_order)
        assert [l.name for l in sorted_layers] == ["bottom", "middle", "top"]

    def test_negative_z_order(self):
        """Verify negative z_order values work."""
        layer = OverlayLayerConfig(name="behind", source="behind.png", z_order=-5)
        assert layer.z_order == -5


class TestTerrainRenderParamsOverlay:
    """Tests for overlay integration in TerrainRenderParams."""

    def test_default_overlay_disabled(self):
        """Verify overlay is disabled by default in TerrainRenderParams."""
        params = make_terrain_params_config(
            size_px=(512, 512),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 1000.0),
        )
        assert params.overlay is not None
        assert params.overlay.enabled is False
        assert params.overlay.layer_count == 0

    def test_overlay_can_be_configured(self):
        """Verify overlay can be configured via make_terrain_params_config."""
        overlay_settings = OverlaySettings(
            enabled=True,
            global_opacity=0.8,
            layers=[
                OverlayLayerConfig(name="satellite", source="sat.png", opacity=0.9),
            ],
        )
        params = make_terrain_params_config(
            size_px=(512, 512),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 1000.0),
            overlay=overlay_settings,
        )
        assert params.overlay.enabled is True
        assert params.overlay.global_opacity == 0.8
        assert params.overlay.layer_count == 1
        assert params.overlay.layers[0].name == "satellite"


class TestOverlayBlendModes:
    """Tests for blend mode configuration."""

    def test_blend_modes_match_constants(self):
        """Verify layer blend modes match OverlayBlendMode constants."""
        layer_normal = OverlayLayerConfig(
            name="normal", source="n.png", blend_mode=OverlayBlendMode.NORMAL
        )
        layer_multiply = OverlayLayerConfig(
            name="multiply", source="m.png", blend_mode=OverlayBlendMode.MULTIPLY
        )
        layer_overlay = OverlayLayerConfig(
            name="overlay", source="o.png", blend_mode=OverlayBlendMode.OVERLAY
        )
        assert layer_normal.blend_mode == "normal"
        assert layer_multiply.blend_mode == "multiply"
        assert layer_overlay.blend_mode == "overlay"
