# tests/test_presets.py
# Unit tests for P7-04: presets schema and merging
# RELEVANT FILES: python/forge3d/presets.py, python/forge3d/config.py
from __future__ import annotations

import pytest

from forge3d import presets
from forge3d.config import RendererConfig, load_renderer_config


@pytest.mark.parametrize("name", ["studio_pbr", "outdoor_sun", "toon_viz"])
def test_presets_available_and_schema_valid(name: str) -> None:
    # Ensure preset is listed and retrievable
    assert name in presets.available()
    mapping = presets.get(name)
    assert isinstance(mapping, dict)

    # Ensure the mapping merges into a RendererConfig without errors
    cfg = RendererConfig.from_mapping(mapping)
    # Validate constraints (e.g., shadow map sizes, cascade ranges, HDR requirements off by default)
    cfg.validate()


def test_outdoor_sun_expected_defaults() -> None:
    mapping = presets.get("outdoor_sun")
    cfg = RendererConfig.from_mapping(mapping)

    # Expected sky model
    assert cfg.atmosphere.sky == "hosek-wilkie"

    # Shadows technique and cascade count from preset
    assert cfg.shadows.technique == "csm"
    assert cfg.shadows.cascades == 3

    # BRDF default
    assert cfg.shading.brdf == "cooktorrance-ggx"


def test_studio_pbr_expected_defaults() -> None:
    mapping = presets.get("studio_pbr")
    cfg = RendererConfig.from_mapping(mapping)

    assert cfg.shading.brdf == "disney-principled"
    assert cfg.shadows.technique == "pcf"
    assert cfg.atmosphere.enabled is False


def test_toon_viz_expected_defaults() -> None:
    mapping = presets.get("toon_viz")
    cfg = RendererConfig.from_mapping(mapping)

    assert cfg.shading.brdf == "toon"
    assert cfg.shadows.technique == "hard"
    assert cfg.atmosphere.enabled is False
