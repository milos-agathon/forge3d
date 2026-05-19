from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _doc(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_large_scene_docs_define_offline_diagnostics_scope():
    text = _doc("docs/guides/large_scene_support.md")

    for marker in (
        "`estimated_gpu_memory`",
        "`unavailable_cache_lod_stats`",
        "`unsupported_instancing_path`",
        "bottleneck layer types",
        "memory budget estimates",
        "cache/LOD stat availability",
        "offline map-production",
    ):
        assert marker in text


def test_large_scene_docs_reject_live_globe_and_engine_parity_claims():
    text = (
        _doc("docs/guides/large_scene_support.md")
        + "\n"
        + _doc("docs/guides/competitive_positioning.md")
        + "\n"
        + _doc("docs/guides/offline_3d_map_rendering.md")
    ).lower()

    for marker in (
        "live globe streaming | `non-goal`",
        "hosted tile-provider parity | `non-goal`",
        "blender parity | `non-goal`",
        "unreal parity | `non-goal`",
        "general dcc | `non-goal`",
    ):
        assert marker in text
    assert "replacement for blender" not in text
    assert "replacement for unreal" not in text
    assert "cesium parity" not in text.replace("cesium runtime parity | `non-goal`", "")
