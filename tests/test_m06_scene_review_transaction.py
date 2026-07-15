"""Scene-review apply is a detached, failure-injectable transaction."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/viewer/scene_review.rs").read_text(encoding="utf-8")


def _between(start: str, end: str) -> str:
    return SOURCE.split(start, 1)[1].split(end, 1)[0]


def test_registry_snapshot_is_published_only_after_runtime_commit():
    body = _between("pub(crate) fn set_scene_review_state", "pub(crate) fn apply_scene_variant")
    assert body.index("let previous = self.scene_review_registry.clone()") < body.index(
        "self.scene_review_registry.install(state)?"
    )
    assert body.index("self.reapply_scene_review_state()") < body.index(
        "update_scene_review_state"
    )
    assert "self.scene_review_registry = previous" in body


def test_all_parse_validation_and_detached_allocation_precede_commit():
    body = _between("pub(crate) fn reapply_scene_review_state", "fn validate_scene_review_effective")
    validate = body.index("validate_scene_review_effective")
    labels = body.index("stage_review_label")
    raster = body.index("stage_review_raster_replacement")
    vector = body.index("stage_review_vector_replacement")
    commit = body.index("commit_review_stacks")
    registry = body.index("managed_raster_overlay_ids = raster_ids")
    assert validate < labels < raster < vector < commit < registry
    assert body.index("build_layer_bvh") < commit


def test_staging_never_mutates_live_label_or_stack_allocators():
    labels = _between("fn stage_review_label", "#[cfg(test)]")
    assert "add_label(" not in labels
    assert "add_line_label(" not in labels
    assert "label_manager" not in labels
    raster = (ROOT / "src/viewer/terrain/overlay/stack/core.rs").read_text(encoding="utf-8")
    vector = (ROOT / "src/viewer/terrain/vector_overlay/core.rs").read_text(encoding="utf-8")
    for source in (raster, vector):
        stage = source.split("fn stage_replacement", 1)[1]
        assert "let mut candidate = Self::new" in stage
        assert "candidate.next_id = self.next_id" in stage


def test_composite_failure_is_propagated_before_atomic_swap():
    wrapper = (ROOT / "src/viewer/terrain/scene/overlays.rs").read_text(encoding="utf-8")
    stage = wrapper.split("stage_review_raster_replacement", 1)[1].split(
        "stage_review_vector_replacement", 1
    )[0]
    assert "candidate.build_composite" in stage
    assert "?" in stage
    apply = _between("pub(crate) fn reapply_scene_review_state", "fn validate_scene_review_effective")
    assert apply.index("after_raster_composite") < apply.index("commit_review_stacks")


def test_required_failure_points_are_before_commit_and_registry_publication():
    body = _between("pub(crate) fn reapply_scene_review_state", "fn validate_scene_review_effective")
    commit = body.index("commit_review_stacks")
    for point in (
        "after_label_staging",
        "after_vector_allocation",
        "after_raster_composite",
    ):
        assert body.index(f'scene_review_injected_failure("{point}")') < commit
        assert point in SOURCE


def test_old_bvhs_and_labels_are_removed_only_after_staging_succeeds():
    body = _between("pub(crate) fn reapply_scene_review_state", "fn validate_scene_review_effective")
    final_stage = body.index("after_vector_allocation")
    assert final_stage < body.index("remove_layer_bvh")
    assert final_stage < body.index("self.remove_label")


def test_ipc_waits_for_correlated_event_loop_completion():
    runner = (ROOT / "src/viewer/event_loop/runner.rs").read_text(encoding="utf-8")
    state = (ROOT / "src/viewer/event_loop/ipc_state.rs").read_text(encoding="utf-8")
    server = (ROOT / "src/viewer/ipc/server.rs").read_text(encoding="utf-8")
    assert "SyncSender<Result<(), String>>" in state
    assert "recv_timeout" in runner
    assert "item.completion.send(outcome)" in runner
    assert "only acknowledges enqueue" not in server
