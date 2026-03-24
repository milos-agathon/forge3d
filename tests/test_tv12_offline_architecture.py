from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_OFFLINE = ROOT / "python" / "forge3d" / "offline.py"
TV12_DOC = ROOT / "docs" / "tv12-terrain-offline-render-quality.md"
TV12_SPEC = (
    ROOT / "docs" / "superpowers" / "specs" / "2026-03-22-tv12-terrain-offline-render-quality-design.md"
)
RUST_OFFLINE = ROOT / "src" / "terrain" / "renderer" / "offline.rs"
HDR_FRAME_RS = ROOT / "src" / "py_types" / "hdr_frame.rs"
SHADER_DIR = ROOT / "src" / "shaders"


def _module_definitions(path: Path) -> set[str]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def test_required_offline_compute_shaders_exist() -> None:
    required = [
        "offline_accumulate.wgsl",
        "offline_depth_extract.wgsl",
        "offline_depth_expand.wgsl",
        "offline_resolve.wgsl",
        "offline_luminance.wgsl",
        "tonemap_terrain_offline.wgsl",
    ]

    for shader_name in required:
        shader_path = SHADER_DIR / shader_name
        assert shader_path.exists(), f"Missing required offline shader: {shader_path}"


def test_python_offline_controller_exports_tv12_entrypoints() -> None:
    definitions = _module_definitions(PYTHON_OFFLINE)
    assert {"OfflineProgress", "OfflineResult", "render_offline"} <= definitions


def test_tv12_docs_reference_public_batch_primitives() -> None:
    source = TV12_DOC.read_text(encoding="utf-8")
    for api_name in (
        "begin_offline_accumulation",
        "accumulate_batch",
        "read_accumulation_metrics",
        "resolve_offline_hdr",
        "tonemap_offline_hdr",
        "end_offline_accumulation",
    ):
        assert api_name in source, f"TV12 docs do not mention public API {api_name}"


def test_tv12_spec_locks_linear_ldr_output_and_scalar_depth_reference() -> None:
    source = TV12_SPEC.read_text(encoding="utf-8")
    assert "Depth reference:** One R32Float texture" in source
    assert "Writes to Rgba8Unorm texture" in source
    assert "Writes to Rgba8UnormSrgb texture" not in source


def test_offline_renderer_bypasses_taa_reprojection_by_construction() -> None:
    source = RUST_OFFLINE.read_text(encoding="utf-8")
    lowered = source.lower()
    assert "offline_params.msaa_samples = 1;" in source
    assert "taa_" not in lowered
    assert "reproject" not in lowered


def test_hdr_frame_save_releases_gil_for_exr_export() -> None:
    source = HDR_FRAME_RS.read_text(encoding="utf-8")
    assert "fn save(&self, py: Python<'_>, path: &str)" in source
    assert "py.allow_threads(|| -> anyhow::Result<()> {" in source
