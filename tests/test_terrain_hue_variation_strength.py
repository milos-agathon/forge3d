from __future__ import annotations

from pathlib import Path

import pytest

import forge3d as f3d
from forge3d._native import NATIVE_AVAILABLE
from forge3d.terrain_params import make_terrain_params_config


ROOT = Path(__file__).resolve().parents[1]


def _make_params(*, hue_variation_strength: float = 0.08):
    return make_terrain_params_config(
        size_px=(256, 256),
        render_scale=1.0,
        terrain_span=1000.0,
        msaa_samples=1,
        z_scale=1.0,
        exposure=1.0,
        domain=(0.0, 1000.0),
        hue_variation_strength=hue_variation_strength,
    )


def test_hue_variation_strength_defaults_to_historical_value() -> None:
    assert _make_params().hue_variation_strength == pytest.approx(0.08)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(-1.0, 0.0), (0.0, 0.0), (0.125, 0.125), (0.2, 0.2), (1.0, 0.2)],
)
def test_hue_variation_strength_is_clamped(requested: float, expected: float) -> None:
    assert _make_params(hue_variation_strength=requested).hue_variation_strength == pytest.approx(
        expected
    )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_hue_variation_strength_rejects_non_finite_values(invalid: float) -> None:
    with pytest.raises(ValueError, match="hue_variation_strength must be finite"):
        _make_params(hue_variation_strength=invalid)


def test_hue_variation_strength_is_uploaded_and_consumed_as_a_uniform() -> None:
    upload = (ROOT / "src/terrain/renderer/upload.rs").read_text(encoding="utf-8")
    shader = (ROOT / "src/shaders/terrain_pbr_pom.wgsl").read_text(encoding="utf-8")

    assert "params.hue_variation_strength.clamp(0.0, 0.2)" in upload
    assert "clamp(u_overlay.params3.z, 0.0, 0.2)" in shader
    assert "let hue_variation_strength = 0.08" not in shader


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Requires compiled _forge3d extension")
def test_native_terrain_render_params_exposes_hue_variation_strength() -> None:
    native = f3d.TerrainRenderParams(_make_params(hue_variation_strength=0.0))
    assert native.hue_variation_strength == pytest.approx(0.0)
