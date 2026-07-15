"""Temporal rebase resets and scatter routine frames must be allocation-free."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _function(text: str, name: str) -> str:
    match = re.search(rf"\bfn\s+{name}\b[^{{]*\{{", text)
    assert match, name
    depth = 1
    cursor = match.end()
    while depth:
        depth += (text[cursor] == "{") - (text[cursor] == "}")
        cursor += 1
    return text[match.start():cursor]


def test_ssgi_and_taa_history_reset_only_mutate_validity_state():
    ssgi = _function(_read("src/core/screen_space_effects/ssgi/accessors.rs"), "reset_history")
    taa = _function(_read("src/core/taa.rs"), "reset_history")
    for body in (ssgi, taa):
        assert "tracked_create_" not in body
        assert "create_texture" not in body
        assert "set_half_res" not in body
    assert "history_valid = false" in _read(
        "src/core/screen_space_effects/ssgi/accessors.rs"
    )


def test_rebase_callback_invalidates_every_named_temporal_consumer():
    body = _function(
        _read("src/viewer/render/main_loop/frame_anchor.rs"), "refresh_after_rebase"
    )
    for token in (
        "terrain_viewer.invalidate_temporal_history()",
        "gi.invalidate_temporal_histories()",
        "taa.reset_history()",
        "self.prev_view_proj = current_vp",
        "self.fog_frame_index = 0",
        "self.fog_history_state.invalidate()",
        "self.history_invalidation_count",
    ):
        assert token in body


def test_scatter_render_path_rewrites_persistent_buffers_without_creating_them():
    scatter = _read("src/viewer/terrain/scene/scatter.rs")
    render = _function(scatter, "render_scatter_batches")
    assert "tracked_create_buffer" not in render
    assert "queue.write_buffer" in render
    assert "hlod_instance_buffer" in render
    assert "preallocate_instance_buffers" in scatter
    scene = _read("src/viewer/terrain/scene.rs")
    assert "scatter_hlod_instance_buffer: TrackedBuffer" in scene


def test_ssgi_first_frame_after_invalidation_bypasses_temporal_blend():
    runtime = _read("src/core/screen_space_effects/ssgi/runtime.rs")
    assert "self.settings.temporal_enabled != 0 && self.history_valid" in runtime
    assert "self.history_valid = true" in runtime


def test_ssr_and_fog_have_explicit_first_frame_history_bypass():
    ssr_runtime = _read("src/core/screen_space_effects/ssr/runtime.rs")
    ssr_accessors = _function(
        _read("src/core/screen_space_effects/ssr/accessors.rs"), "invalidate_history"
    )
    assert "history_state.invalidate()" in ssr_accessors
    assert "if self.history_state.is_valid()" in ssr_runtime
    assert "&self.ssr_final_view" in ssr_runtime
    assert "history_state.mark_populated()" in ssr_runtime

    fog = _read("src/viewer/render/main_loop/geometry/fog.rs")
    assert "fog_history_state" in fog
    assert ".blend_alpha(" in fog
    assert "fog_history_state.mark_populated()" in fog


def test_density_rebase_metadata_is_excluded_from_voxel_identity():
    density = _function(
        _read("src/viewer/terrain/volume_density.rs"), "fingerprint_configs"
    )
    assert "for value in context.render_origin_span" not in density
    reuse = _function(
        _read("src/viewer/terrain/volumetrics.rs"), "ensure_density_volume_atlas"
    )
    assert "atlas.metadata = data.metadata" in reuse
    assert "DensityVolumeAtlasGpu::upload" in reuse
