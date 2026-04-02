from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_heightfield_ao_shader_reconstructs_scene_y_from_raw_height() -> None:
    shader = (REPO_ROOT / "src" / "shaders" / "heightfield_ao.wgsl").read_text()
    assert "return (h - u_ao.params1.w) * u_ao.params1.z;" in shader


def test_heightfield_sun_vis_shader_reconstructs_scene_y_from_raw_height() -> None:
    shader = (REPO_ROOT / "src" / "shaders" / "heightfield_sun_vis.wgsl").read_text()
    assert "return (h - u_sun.params1.w) * u_sun.params1.z;" in shader


def test_viewer_heightfield_compute_uses_z_scale_not_height_range_scaled_z() -> None:
    helpers = (REPO_ROOT / "src" / "viewer" / "terrain" / "render" / "helpers.rs").read_text()
    assert helpers.count("terrain_depth / height as f32,\n                    z_scale,") >= 2
    assert "h_range * z_scale" not in helpers
