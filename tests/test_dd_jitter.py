"""DUPLA absolute-coordinate jitter-demo contracts."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHADER = ROOT / "src" / "shaders" / "dd_jitter.wgsl"
RUST = ROOT / "src" / "core" / "dd" / "jitter.rs"
MODEL = ROOT / "src" / "core" / "dd" / "jitter_model.rs"


def test_jitter_shader_has_opt_in_dd_and_raw_vertex_paths() -> None:
    source = SHADER.read_text(encoding="utf-8")
    assert "@vertex\nfn vs_dd" in source
    assert "@vertex\nfn vs_raw_f32" in source
    assert "dd_sub_vec3(position, camera_dd)" in source
    assert "view_proj * vec4<f32>(local, 1.0)" in source
    assert "@compute" in source and "fn measure_jitter" in source


def test_jitter_demo_contract_is_1000_one_millimetre_frames() -> None:
    source = RUST.read_text(encoding="utf-8") + MODEL.read_text(encoding="utf-8")
    assert "const DEFAULT_FRAMES: u32 = 1_000" in source
    assert "const CAMERA_STEP_METRES: f64 = 0.001" in source
    assert "dd_errors_px" in source and "f32_errors_px" in source
    assert "raw_over_one_px" in source
    assert "dd_hash_a" in source and "dd_hash_b" in source
    assert "OneShotTiming" in source
    assert "tracked_create_buffer" in source


def test_jitter_demo_is_opt_in_only() -> None:
    source = RUST.read_text(encoding="utf-8")
    assert "pub fn jitter_demo" in source
    assert "initialize_for_context" not in source
