"""Behavioral proof for the public M-06 Python viewer contracts."""

from __future__ import annotations

import math

import pytest

import forge3d
from forge3d.viewer import ViewerHandle


def _capturing_handle() -> tuple[ViewerHandle, list[dict]]:
    handle = object.__new__(ViewerHandle)
    captured: list[dict] = []

    def send(command: dict) -> dict:
        captured.append(command)
        return {"ok": True, "id": command.get("id")}

    handle._send_command = send  # type: ignore[method-assign]
    return handle, captured


def test_authoritative_aliases_are_exported_from_package_and_viewer_module():
    assert forge3d.WorldPosition is not None
    assert forge3d.VectorOverlayVertex is not None
    assert forge3d.NormalizedExtent is not None


def test_vector_helper_preserves_earth_scale_submillimetre_xyz_and_large_u32_id():
    handle, captured = _capturing_handle()
    x = 6_378_137.0 + 0.000_25
    created = handle.add_vector_overlay(
        "precision",
        [(x, 2.0, 3.0, 1.0, 0.5, 0.0, 1.0, 16_777_217)],
        [0],
        primitive="points",
    )
    assert created == 1
    assert captured[0]["vertices"][0][0] == x
    assert captured[0]["vertices"][0][7] == 16_777_217
    assert handle._next_public_vector_overlay_id == 2


@pytest.mark.parametrize(
    "row",
    [
        (0.0,) * 7,
        (0.0,) * 9,
        (math.nan, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1),
        (0.0, 0.0, 0.0, 1e300, 1.0, 1.0, 1.0, 1),
        (0.0, 0.0, 0.0, -0.1, 1.0, 1.0, 1.0, 1),
        (0.0, 0.0, 0.0, 1.1, 1.0, 1.0, 1.0, 1),
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.5),
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1),
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2**32),
    ],
)
def test_vector_rejection_is_synchronous_and_allocator_atomic(row):
    handle, captured = _capturing_handle()
    with pytest.raises((TypeError, ValueError)):
        handle.add_vector_overlay("bad", [row], [0], primitive="points")
    assert captured == []
    assert getattr(handle, "_next_public_vector_overlay_id", 1) == 1


def test_normalized_extent_rejects_non_normalized_or_empty_ranges_before_ipc():
    handle, captured = _capturing_handle()
    with pytest.raises(ValueError):
        handle.load_overlay("bad", "overlay.png", extent=(0.0, 0.0, 2.0, 1.0))
    with pytest.raises(ValueError):
        handle.load_overlay("bad", "overlay.png", extent=(0.5, 0.0, 0.5, 1.0))
    assert captured == []


def test_high_level_pick_is_execution_correlated_and_returns_absolute_results():
    handle = object.__new__(ViewerHandle)
    commands: list[dict] = []
    responses = iter(
        (
            {"ok": True, "pick_events": [{"screen_pos": [1, 2], "results": []}]},
            {"ok": True},
            {
                "ok": True,
                "pick_events": [
                    {
                        "screen_pos": [40, 50],
                        "results": [
                            {
                                "kind": "Object",
                                "world_pos": [6_378_137.000_25, 2.0, 3.0],
                            }
                        ],
                    }
                ],
            },
        )
    )

    def send(command: dict) -> dict:
        commands.append(command)
        return next(responses)

    handle._send_command = send  # type: ignore[method-assign]
    results = handle.pick_at(40, 50, shift=True)
    assert [command["cmd"] for command in commands] == [
        "poll_pick_events",
        "pick_at",
        "poll_pick_events",
    ]
    assert commands[1] == {
        "cmd": "pick_at",
        "x": 40,
        "y": 50,
        "shift": True,
        "ctrl": False,
    }
    assert results[0]["world_pos"][0] == 6_378_137.000_25


def test_manual_label_update_uses_the_execution_correlated_command():
    handle, captured = _capturing_handle()
    handle.update_labels()
    assert captured == [{"cmd": "update_labels"}]
