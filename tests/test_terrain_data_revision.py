from __future__ import annotations

import pytest

import forge3d as f3d
from forge3d._native import NATIVE_AVAILABLE
from forge3d.terrain_params import make_terrain_params_config


def _make_params(*, terrain_data_revision: int | None = None):
    return make_terrain_params_config(
        size_px=(800, 600),
        render_scale=1.0,
        terrain_span=1000.0,
        msaa_samples=1,
        z_scale=1.0,
        exposure=1.0,
        domain=(0.0, 1000.0),
        terrain_data_revision=terrain_data_revision,
    )


def test_terrain_data_revision_default_none() -> None:
    params = _make_params()
    assert params.terrain_data_revision is None


def test_terrain_data_revision_can_be_set() -> None:
    params = _make_params(terrain_data_revision=42)
    assert params.terrain_data_revision == 42


def test_terrain_data_revision_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="terrain_data_revision must be >= 0"):
        _make_params(terrain_data_revision=-1)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Requires compiled _forge3d extension")
def test_native_terrain_render_params_exposes_revision() -> None:
    native = f3d.TerrainRenderParams(_make_params(terrain_data_revision=99))
    assert native.terrain_data_revision == 99
