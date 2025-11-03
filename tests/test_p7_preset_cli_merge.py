# tests/test_p7_preset_cli_merge.py
# Small integration test to assert that `--preset outdoor_sun` plus CLI-style
# overrides yields the expected RendererConfig fields.
# RELEVANT FILES: python/forge3d/presets.py, python/forge3d/config.py, examples/terrain_demo.py
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from forge3d import presets
from forge3d.config import load_renderer_config


@pytest.mark.parametrize(
    "overrides",
    [
        {
            # Mirrors: --brdf cooktorrance-ggx --shadows csm --cascades 4 --hdr assets/sky.hdr
            "brdf": "cooktorrance-ggx",
            "shadows": "csm",
            "cascades": 4,
            "hdr": "assets/sky.hdr",
        }
    ],
)
def test_p7_outdoor_sun_cli_overrides_merge(overrides: dict) -> None:
    base = presets.get("outdoor_sun")
    cfg = load_renderer_config(base, overrides)

    # Shading override applied
    assert cfg.shading.brdf == "cooktorrance-ggx"

    # Shadows technique and cascades applied
    assert cfg.shadows.technique == "csm"
    assert cfg.shadows.cascades == 4

    # Atmosphere retains the preset sky model and accepts HDR override
    assert cfg.atmosphere.sky == "hosek-wilkie"
    assert cfg.atmosphere.hdr_path == "assets/sky.hdr"

    # Lighting preserved from preset (at least one directional light)
    assert len(cfg.lighting.lights) >= 1
    assert cfg.lighting.lights[0].type == "directional"
