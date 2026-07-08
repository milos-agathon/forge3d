from __future__ import annotations

from pathlib import Path


def test_water_surface_shader_uses_gerstner_wave_normals() -> None:
    shader = (Path(__file__).resolve().parents[1] / "src" / "shaders" / "water_surface.wgsl").read_text(
        encoding="utf-8"
    )

    assert "fn gerstner_height" in shader
    assert "fn gerstner_slope" in shader
    assert "gerstner_height(in.uv" in shader
    assert "let epsilon = 0.01" not in shader
    assert "fn simple_wave" not in shader
