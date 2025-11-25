# tests/test_p6_fog_config.py
# P6 Phase 2: Volumetric fog configuration and analytic behavior tests
#
# This module focuses on Python-side configuration plumbing for volumetric
# fog and basic depth-dependent fog behavior using the analytic helpers in
# forge3d.lighting. GPU correctness for cs_volumetric is covered by
# tests/test_p6_fog.rs.

from __future__ import annotations

import numpy as np
import pytest

from forge3d.config import load_renderer_config
import forge3d.lighting as lighting


def test_p6_volumetric_config_roundtrip_top_level_overrides() -> None:
    """Top-level `volumetric` overrides map into AtmosphereParams.volumetric.

    This exercises the CLI-style path:
        load_renderer_config(config=None, overrides={"volumetric": {...}})
    and validates that density/phase/g/mode/max_steps survive the round-trip.
    """

    cfg = load_renderer_config(
        None,
        {
            "volumetric": {
                "density": 0.015,
                "phase": "hg",
                "g": 0.7,
                "mode": "raymarch",
                "max_steps": 48,
            }
        },
    )

    vol = cfg.atmosphere.volumetric
    assert vol is not None
    assert vol.density == pytest.approx(0.015)
    # Phase should normalize via _PHASE_FUNCTIONS mapping
    assert vol.phase == "henyey-greenstein"
    assert vol.anisotropy == pytest.approx(0.7)
    assert vol.mode == "raymarch"
    assert vol.max_steps == 48


def test_p6_volumetric_config_roundtrip_nested_atmosphere() -> None:
    """Nested `atmosphere.volumetric` overrides behave as in the P6 design doc."""

    overrides = {
        "atmosphere": {
            "volumetric": {
                "density": 0.02,
                "phase": "hg",
                "g": 0.5,
                "mode": "raymarch",
                "max_steps": 64,
            },
        },
    }
    cfg = load_renderer_config(None, overrides)

    vol = cfg.atmosphere.volumetric
    assert vol is not None
    assert vol.density == pytest.approx(0.02)
    assert vol.phase == "henyey-greenstein"
    assert vol.anisotropy == pytest.approx(0.5)
    assert vol.mode == "raymarch"
    assert vol.max_steps == 64


def test_p6_volumetric_depth_fog_correlation_from_config() -> None:
    """Synthetic depth test: far pixels accumulate more fog than near pixels.

    Uses `height_fog_factor` with densities taken from RendererConfig so the
    behavior is driven by the same volumetric parameters that feed P6.
    """

    # Volumetric disabled via zero density
    cfg_off = load_renderer_config(None, {"volumetric": {"density": 0.0}})
    vol_off = cfg_off.atmosphere.volumetric
    assert vol_off is not None

    # Volumetric enabled with modest density
    cfg_on = load_renderer_config(None, {"volumetric": {"density": 0.03}})
    vol_on = cfg_on.atmosphere.volumetric
    assert vol_on is not None

    near = np.array([1.0, 5.0], dtype=np.float32)
    far = np.array([50.0, 100.0], dtype=np.float32)

    # With zero density, fog factor should be ~0 for all depths
    fog_off_near = lighting.height_fog_factor(near, density=vol_off.density)
    fog_off_far = lighting.height_fog_factor(far, density=vol_off.density)
    assert np.allclose(fog_off_near, 0.0)
    assert np.allclose(fog_off_far, 0.0)

    # With non-zero density, far pixels must be foggier than near pixels
    fog_on_near = lighting.height_fog_factor(near, density=vol_on.density)
    fog_on_far = lighting.height_fog_factor(far, density=vol_on.density)
    assert np.all(fog_on_far > fog_on_near)
