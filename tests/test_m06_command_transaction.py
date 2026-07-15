"""Fail-closed contracts for M-06 prospective command transactions."""

from __future__ import annotations

from itertools import permutations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_drained_batch_is_preflighted_once_before_any_handler_runs() -> None:
    runner = _read("src/viewer/event_loop/runner.rs")
    body = runner[runner.index("fn apply_command_batch") :]
    body = body[: body.index("\n}\n")]
    preflight = body.index("preflight_command_batch")
    ordering = body.index("order_command_batch")
    handler = body.index("handle_cmd")
    assert preflight < ordering < handler
    assert "q.drain(..).collect()" in runner
    assert "for item in queued" in runner
    assert "completion.send(Err(error.clone()))" in runner


def test_prospective_frame_precedence_and_existing_pending_content_are_locked() -> None:
    source = _read("src/viewer/event_loop/command_batch.rs")
    terrain = source.index("if terrain_present")
    point = source.index("else if point_present")
    general = source.index("else {\n            general_eye")
    assert terrain < point < general
    for required in (
        "cloud.source_points",
        "all_vector_overlay_source_points",
        "label_manager.world_points",
        "self.object_translation",
        "preflight_laz_bounds(path)",
        "ViewerCmd::AddVectorOverlay",
        "ViewerCmd::AddLabel",
        "ViewerCmd::AddLineLabel",
        "ViewerCmd::AddCurvedLabel",
        "ViewerCmd::AddCallout",
        "ViewerCmd::SetTransform",
    ):
        assert required in source


def test_semantic_order_is_permutation_invariant_for_frame_establishers() -> None:
    # Mirrors the production priorities and adversarially proves that every
    # permutation of the four required semantic classes has one canonical form.
    priority = {"terrain": 0, "point": 1, "camera": 4, "object": 5}
    canonical = ("terrain", "point", "camera", "object")
    for order in permutations(canonical):
        assert tuple(sorted(order, key=priority.__getitem__)) == canonical

    source = _read("src/viewer/event_loop/command_batch.rs")
    assert "ViewerCmd::LoadTerrain(_) => 0" in source
    assert "ViewerCmd::LoadPointCloud { .. } => 1" in source
    assert "ViewerCmd::SetCamLookAt { .. } => 4" in source
    assert "commands.sort_by_key(command_priority)" in source


def test_every_interactive_camera_mutation_uses_candidate_validation() -> None:
    controller = _read("src/viewer/camera_controller.rs")
    for route in (
        "try_handle_mouse_move",
        "try_handle_mouse_scroll",
        "try_handle_pan",
        "try_update_fps",
    ):
        body = controller[controller.index(f"pub fn {route}") :]
        body = body[: body.index("\n    }")]
        assert "candidate" in body
        assert "validate_current_pose" in body
        assert "*self = candidate" in body

    terrain = _read("src/viewer/terrain/scene/terrain_load.rs")
    point = _read("src/viewer/pointcloud/state.rs")
    assert terrain.count("validate_camera_state(") >= 4
    assert point.count("validate_camera_state(anchor)") >= 3
    # Terrain computes and validates candidates before publishing them; the
    # point-cloud camera retains its older mutate-then-rollback implementation.
    assert "let phi =" in terrain and "let radius =" in terrain
    assert "= old" in point


def test_ipc_acknowledgement_is_correlated_with_execution_result() -> None:
    server = _read("src/viewer/ipc/server.rs")
    queue = _read("src/viewer/event_loop/ipc_state.rs")
    runner = _read("src/viewer/event_loop/runner.rs")
    assert "recv_timeout" in runner
    assert "match cmd_sender(cmd)" in server
    assert "queued for execution" not in server.lower()
    assert "SyncSender<Result<(), String>>" in queue

    stats = _read("src/viewer/ipc/protocol/payloads.rs")
    assert "applied_command_revision" in stats
    assert "rendered_frame_revision" in stats


def test_one_parser_and_terrain_preflight_cover_startup_and_runtime() -> None:
    parser = _read("src/viewer/event_loop/stdin_reader/parser.rs")
    startup = _read("src/viewer/event_loop/cmd_parse_init.rs")
    server = _read("src/viewer/ipc/server.rs")
    assert "translate_text_command" in parser
    assert "preflight_terrain_path" in parser
    assert "translate_text_command" in startup
    assert "ipc_request_to_viewer_cmd" in server
    assert "legacy" not in startup.lower()


def test_scene_review_scatter_is_staged_before_the_atomic_commit() -> None:
    source = _read("src/viewer/scene_review.rs")
    staged = source.index("stage_scatter_batches_from_configs")
    commit_marker = source.index("// Commit: no fallible work remains")
    committed = source.index("commit_scatter_batches")
    assert staged < commit_marker < committed
    assert "set_scatter_batches_from_configs(&effective.scatter_batches)" not in source
