# tests/test_b2_tonemap.py
# Tone mapping and exposure unit tests for Workstream B2.
# Exists to validate Python API toggles and curve behavior.
# RELEVANT FILES:python/forge3d/pbr.py,python/forge3d/lighting.py,shaders/tone_map.wgsl,examples/pbr_spheres.py

from __future__ import annotations

import numpy as np
import pytest

from forge3d import pbr, lighting

_MODES = ("reinhard", "aces", "hable")


@pytest.mark.opbr
@pytest.mark.olighting
def test_set_tone_mapping_modes_roundtrip() -> None:
    for mode in _MODES:
        selected = pbr.set_tone_mapping(mode)
        assert selected == mode
        assert getattr(pbr, "_CURRENT_TONE_MAPPING") == mode
    pbr.set_tone_mapping("reinhard")


@pytest.mark.opbr
def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError):
        pbr.set_tone_mapping("linear")


@pytest.mark.opbr
def test_tone_curve_monotonic() -> None:
    samples = np.linspace(0.0, 8.0, 128, dtype=np.float32)
    for mode in _MODES:
        pbr.set_tone_mapping(mode)
        mapped = pbr._apply_current_tone_mapping(samples)
        diffs = np.diff(mapped)
        assert np.all(diffs >= -1e-5)
    pbr.set_tone_mapping("reinhard")


@pytest.mark.olighting
def test_exposure_stops_conversion() -> None:
    scale = lighting.set_exposure_stops(1.5)
    assert scale == pytest.approx(2.0 ** 1.5)
    assert lighting._EXPOSURE_STOPS == pytest.approx(1.5)
    lighting.set_exposure_stops(0.0)
