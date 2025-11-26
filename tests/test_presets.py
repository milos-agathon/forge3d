# tests/test_presets.py
# Unit tests for P7-04: presets schema and merging
# RELEVANT FILES: python/forge3d/presets.py, python/forge3d/config.py
from __future__ import annotations

import pytest

from forge3d import presets
from forge3d.config import RendererConfig, load_renderer_config


@pytest.mark.parametrize("name", ["studiopbr", "outdoorsun", "toonviz"])
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


def test_presets_available_canonical_names_sorted() -> None:
    names = presets.available()
    # Canonical keys defined in python/forge3d/presets.py
    assert names == sorted(["studiopbr", "outdoorsun", "toonviz"])


@pytest.mark.parametrize(
    "alias, canonical",
    [
        ("studio", "studiopbr"),
        ("pbr", "studiopbr"),
        ("outdoor", "outdoorsun"),
        ("sun", "outdoorsun"),
        ("toon", "toonviz"),
    ],
)
def test_presets_aliases_resolve_to_canonical_configs(alias: str, canonical: str) -> None:
    mapping_alias = presets.get(alias)
    mapping_canonical = presets.get(canonical)

    cfg_alias = RendererConfig.from_mapping(mapping_alias)
    cfg_canonical = RendererConfig.from_mapping(mapping_canonical)

    assert cfg_alias.to_dict() == cfg_canonical.to_dict()


def test_presets_get_unknown_raises_valueerror_with_available_names() -> None:
    with pytest.raises(ValueError) as excinfo:
        presets.get("does-not-exist")

    msg = str(excinfo.value)
    assert "Unknown preset" in msg
    # Error message should list available preset names
    for name in presets.available():
        assert name in msg


def test_presets_get_returns_copy_not_shared_state() -> None:
    first = presets.get("outdoor_sun")
    first["shading"]["brdf"] = "lambert"

    second = presets.get("outdoor_sun")
    cfg = RendererConfig.from_mapping(second)
    # Preset definition must remain unchanged
    assert cfg.shading.brdf == "cooktorrance-ggx"


def test_presets_default_gi_modes_do_not_enable_ibl() -> None:
    for name in presets.available():
        mapping = presets.get(name)
        cfg = RendererConfig.from_mapping(mapping)
        assert "ibl" not in cfg.gi.modes
