from __future__ import annotations

import forge3d as f3d

from forge3d.terrain_params import ReflectionProbeSettings


def test_reflection_probe_settings_reexported() -> None:
    assert f3d.ReflectionProbeSettings is ReflectionProbeSettings


def test_reflection_probe_settings_listed_in_public_api() -> None:
    assert "ReflectionProbeSettings" in f3d.__all__
