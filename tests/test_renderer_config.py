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

