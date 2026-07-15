"""Scene-review apply must stage, validate, and roll back as one transaction."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/viewer/scene_review.rs").read_text(encoding="utf-8")


def _between(start: str, end: str) -> str:
    return SOURCE.split(start, 1)[1].split(end, 1)[0]


def test_registry_snapshot_is_published_only_after_runtime_apply_succeeds():
    body = _between("pub(crate) fn set_scene_review_state", "pub(crate) fn apply_scene_variant")
    assert body.index("let previous = self.scene_review_registry.clone()") < body.index(
        "self.scene_review_registry.install(state)?"
    )
    assert body.index("self.reapply_scene_review_state()") < body.index(
        "update_scene_review_state"
    )
    assert "self.scene_review_registry = previous" in body


def test_new_runtime_is_staged_before_old_runtime_is_removed():
    body = _between("pub(crate) fn reapply_scene_review_state", "fn remove_scene_review_runtime_ids")
    stage = body.index("let mut raster_ids")
    old = body.index("let old_raster_ids")
    commit = body.index("managed_raster_overlay_ids = raster_ids")
    assert stage < old < commit
    assert body.count("self.remove_scene_review_runtime_ids(&raster_ids") >= 3


def test_all_world_content_is_validated_before_any_runtime_mutation():
    body = _between("pub(crate) fn reapply_scene_review_state", "fn remove_scene_review_runtime_ids")
    assert body.index("validate_scene_review_effective") < body.index("add_overlay_image")
    validation = _between("fn validate_scene_review_effective", "fn add_managed_vector_overlay")
    assert "for overlay in &effective.vector_overlays" in validation
    assert "for payload in &effective.labels" in validation
    assert "validate_world_point" in validation


def test_managed_vector_bvh_is_removed_and_rebuilt_with_runtime_layer():
    remove = _between("fn remove_scene_review_runtime_ids", "fn validate_scene_review_effective")
    assert "remove_vector_overlay" in remove
    assert "remove_layer_bvh" in remove
    assert "build_layer_bvh" in SOURCE


def test_ipc_does_not_publish_scene_review_state_before_runtime_commit():
    server = (ROOT / "src/viewer/ipc/server.rs").read_text(encoding="utf-8")
    arm = server.split("req @ IpcRequest::SetSceneReviewState", 1)[1].split(
        "IpcRequest::ApplySceneVariant", 1
    )[0]
    assert "update_scene_review_state" not in arm
    assert "only acknowledges enqueue" in arm
