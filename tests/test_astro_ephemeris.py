from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path

import pytest

from forge3d import astro, sky
from forge3d._native import get_native_module

DATA = Path(__file__).parent / "data" / "horizons_vectors.dat"
THRESHOLDS = {
    "sun": 10.0,
    "moon": 30.0,
    "mercury": 60.0,
    "venus": 60.0,
    "mars": 60.0,
    "jupiter": 60.0,
    "saturn": 60.0,
}


def _separation_arcsec(a, b):
    azimuth_delta = math.radians(a[0] - b[0])
    altitude_a = math.radians(a[1])
    altitude_b = math.radians(b[1])
    cosine = (
        math.sin(altitude_a) * math.sin(altitude_b)
        + math.cos(altitude_a) * math.cos(altitude_b) * math.cos(azimuth_delta)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) * 3_600.0


def _utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_public_ephemeris_meets_committed_horizons_gates():
    maxima = {body: 0.0 for body in THRESHOLDS}
    phase_max = semidiameter_max = 0.0
    for line in DATA.read_text(encoding="ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] == "@moon_phase":
            phase = astro.moon_phase(_utc(fields[1]))
            phase_max = max(phase_max, abs(phase[0] - float(fields[2]) / 100.0))
            semidiameter_max = max(semidiameter_max, abs(phase[2] - float(fields[3]) * 0.5))
            continue
        body = fields[2]
        actual = astro.body_position(
            body,
            _utc(fields[1]),
            float(fields[3]),
            float(fields[4]),
            height_m=float(fields[5]),
        )
        maxima[body] = max(
            maxima[body],
            _separation_arcsec(actual, (float(fields[6]), float(fields[7]))),
        )
    assert all(maxima[body] <= threshold for body, threshold in THRESHOLDS.items()), maxima
    assert phase_max <= 0.005
    assert semidiameter_max <= 1.0


def test_observation_sets_every_rendered_body_and_star_catalog():
    summary = sky.set_observation(
        datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
    )
    assert summary["star_count"] == 9_096
    assert set(THRESHOLDS) <= summary.keys()
    assert len(summary["moon_phase"]) == 3


def test_observation_routes_to_active_subprocess_viewer():
    class FakeViewer:
        is_running = True

        def __init__(self):
            self.command = None

        def send_ipc(self, command):
            self.command = command

    fake = FakeViewer()
    sky._set_active_viewer(fake)
    try:
        sky.set_observation(
            datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
        )
        assert fake.command["cmd"] == "set_observation"
    finally:
        sky._set_active_viewer(None)


def test_observation_replays_when_viewer_opens_later():
    sky.set_observation(
        datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
    )

    class FakeViewer:
        is_running = True

        def __init__(self):
            self.command = None

        def send_ipc(self, command):
            self.command = command

    fake = FakeViewer()
    try:
        sky._set_active_viewer(fake)
        assert fake.command["cmd"] == "set_observation"
    finally:
        sky._set_active_viewer(None)


def test_observation_replay_survives_viewer_restart():
    class FakeViewer:
        is_running = True

        def __init__(self):
            self.command = None

        def send_ipc(self, command):
            self.command = command

    first = FakeViewer()
    second = FakeViewer()
    sky._set_active_viewer(first)
    try:
        sky.set_observation(
            datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
        )
        sky._remove_active_viewer(first)
        first.is_running = False
        sky._set_active_viewer(second)
        assert second.command["cmd"] == "set_observation"
    finally:
        sky._set_active_viewer(None)


def test_temporary_viewer_restores_previous_observation_target():
    class FakeViewer:
        is_running = True

        def __init__(self):
            self.command = None

        def send_ipc(self, command):
            self.command = command

    previous = FakeViewer()
    temporary = FakeViewer()
    sky._set_active_viewer(previous)
    try:
        sky._set_active_viewer(temporary)
        sky._remove_active_viewer(temporary)
        assert sky._get_active_viewer() is previous
    finally:
        sky._set_active_viewer(None)


def test_viewer_close_keeps_replay_cleanup_best_effort(monkeypatch):
    from forge3d import viewer as viewer_module

    handle = object.__new__(viewer_module.ViewerHandle)
    handle._socket = None
    handle._process = None
    handle._cleanup_paths = []
    monkeypatch.setattr(
        sky,
        "_remove_active_viewer",
        lambda _viewer: (_ for _ in ()).throw(RuntimeError("broken replay")),
    )
    handle.close()


def test_blocking_viewer_keeps_direct_launch_without_observation(monkeypatch):
    from forge3d import viewer as viewer_module

    class Process:
        def wait(self):
            return 7

    launched = []
    sky._clear_observation_replay()
    monkeypatch.setattr(viewer_module, "_find_viewer_binary", lambda: "viewer")
    monkeypatch.setattr(
        viewer_module.subprocess,
        "Popen",
        lambda command: launched.append(command) or Process(),
    )
    assert viewer_module.open_viewer(width=320, height=200) == 7
    assert launched == [["viewer", "--size", "320x200", "--fov", "60.0"]]


def test_observation_aware_blocking_snapshot_closes_after_capture(monkeypatch):
    from forge3d import viewer as viewer_module

    class Process:
        def wait(self):
            raise AssertionError("snapshot flow must not wait for an interactive close")

    class Viewer:
        _process = Process()

        def __init__(self):
            self.snapshot_path = None
            self.closed = False

        def snapshot(self, path, *, width, height):
            self.snapshot_path = (path, width, height)

        def close(self):
            self.closed = True

    handle = Viewer()
    sky.set_observation(
        datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
    )
    monkeypatch.setattr(viewer_module, "open_viewer_async", lambda **_kwargs: handle)
    assert viewer_module.open_viewer(snapshot_path="night.png") == 0
    assert handle.snapshot_path == ("night.png", 1280, 720)
    assert handle.closed
    sky._clear_observation_replay()


def test_manual_sun_success_clears_stale_observation_replay():
    from forge3d import viewer as viewer_module

    class Socket:
        def sendall(self, _request):
            pass

        def recv(self, _size):
            return b'{"ok": true}\n'

    class FakeViewer:
        is_running = True
        command = None

        def send_ipc(self, command):
            self.command = command

    sky.set_observation(
        datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
    )
    handle = object.__new__(viewer_module.ViewerHandle)
    handle._socket = Socket()
    handle._send_command(
        {"cmd": "lit_sun", "azimuth_deg": 180.0, "elevation_deg": 30.0}
    )
    viewer = FakeViewer()
    try:
        sky._set_active_viewer(viewer)
        assert viewer.command is None
    finally:
        sky._set_active_viewer(None)


def test_observation_can_preseed_viewer_without_native(monkeypatch):
    monkeypatch.setattr(sky, "get_native_module", lambda: None)
    summary = sky.set_observation(
        datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
    )
    assert summary["status"] == "pending_viewer"
    with pytest.raises(ValueError):
        sky.set_observation(
            datetime(2051, 1, 1, tzinfo=timezone.utc), 0.0, 0.0
        )

    class ClosedViewer:
        is_running = False

    closed = ClosedViewer()
    sky._set_active_viewer(closed)
    try:
        summary = sky.set_observation(
            datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
        )
        assert summary["status"] == "pending_viewer"
    finally:
        sky._set_active_viewer(None)


def test_invalid_or_out_of_window_inputs_raise():
    with pytest.raises(ValueError):
        astro.body_position("sun", datetime(2026, 1, 1), 0.0, 0.0)
    with pytest.raises(ValueError):
        astro.body_position(
            "sun", datetime(2051, 1, 1, tzinfo=timezone.utc), 0.0, 0.0
        )


def test_sidereal_refraction_and_reduction_ablation_gates():
    metrics = get_native_module().astro_validation_metrics()
    assert metrics["gmst_max_seconds"] <= 0.1
    assert metrics["refraction_error_arcminutes"] <= 0.2
    assert metrics["precession_arcminutes"] > 20.0
    assert metrics["lunar_parallax_arcminutes"] > 30.0


def test_set_sun_handler_is_live_not_print_only():
    frame_setup = (
        Path(__file__).parents[1]
        / "src"
        / "viewer"
        / "render"
        / "main_loop"
        / "frame_setup.rs"
    ).read_text(encoding="utf-8")
    assert "sun_dir_ws.y > 0.0" in frame_setup
    helpers = (
        Path(__file__).parents[1]
        / "src"
        / "viewer"
        / "state"
        / "viewer_helpers"
        / "core.rs"
    ).read_text(encoding="utf-8")
    assert "lit_sun_direction_ws[1] > 0.0" in helpers
    scene_command = (
        Path(__file__).parents[1]
        / "src"
        / "viewer"
        / "cmd"
        / "scene_command.rs"
    ).read_text(encoding="utf-8")
    lit_sun = scene_command.split("ViewerCmd::SetLitSun", 1)[1].split(
        "ViewerCmd::SetLitIbl", 1
    )[0]
    assert "viewer.sync_terrain_sun_to_lit()" in lit_sun
    terrain_command = (
        Path(__file__).parents[1]
        / "src"
        / "viewer"
        / "cmd"
        / "terrain_command.rs"
    ).read_text(encoding="utf-8")
    load_handler = terrain_command.split("ViewerCmd::LoadTerrain", 1)[1].split(
        "ViewerCmd::SetTerrainCamera", 1
    )[0]
    assert "viewer.sync_terrain_sun_to_lit()" in load_handler

    viewer = (Path(__file__).parents[1] / "python" / "forge3d" / "viewer.py").read_text(
        encoding="utf-8"
    )
    replay = viewer.split("sky._set_active_viewer(handle)", 1)
    assert "handle.close()" in replay[1]
    blocking = viewer.split("def open_viewer(", 1)[1]
    assert "handle = open_viewer_async(" in blocking
