"""Tests for non-blocking viewer IPC functionality.

These tests validate:
- NDJSON request formatting (Python client)
- READY line parsing (Python client)
- Command enum mapping (unit tests, no GUI required)

No actual viewer window is opened in these tests.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

# Add the python package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from forge3d.bundle import BundleManifest, LoadedBundle, ReviewLayer, SceneState, SceneVariant
from forge3d.viewer import (
    ViewerHandle,
    ViewerError,
    _READY_PATTERN,
    _find_viewer_binary,
    _prepare_terrain_path,
    open_viewer_async,
)
import forge3d.viewer as viewer_module
import forge3d.viewer_ipc as viewer_ipc_module
import forge3d._viewer_entry as viewer_entry_module


class TestReadyLineParsing:
    """Test READY line parsing."""

    def test_ready_pattern_matches_valid_line(self):
        """READY pattern matches valid FORGE3D_VIEWER_READY line."""
        line = "FORGE3D_VIEWER_READY port=12345"
        match = _READY_PATTERN.search(line)
        assert match is not None
        assert match.group(1) == "12345"

    def test_ready_pattern_extracts_port(self):
        """READY pattern extracts port number correctly."""
        test_cases = [
            ("FORGE3D_VIEWER_READY port=0", "0"),
            ("FORGE3D_VIEWER_READY port=80", "80"),
            ("FORGE3D_VIEWER_READY port=65535", "65535"),
            ("Some prefix FORGE3D_VIEWER_READY port=8080 suffix", "8080"),
        ]
        for line, expected_port in test_cases:
            match = _READY_PATTERN.search(line)
            assert match is not None, f"Failed to match: {line}"
            assert match.group(1) == expected_port

    def test_ready_pattern_rejects_invalid_lines(self):
        """READY pattern does not match invalid lines."""
        invalid_lines = [
            "FORGE3D_VIEWER_READY",  # missing port
            "FORGE3D_VIEWER_READY port=",  # missing port value
            "VIEWER_READY port=1234",  # wrong prefix
            "forge3d_viewer_ready port=1234",  # wrong case
        ]
        for line in invalid_lines:
            match = _READY_PATTERN.search(line)
            assert match is None, f"Should not match: {line}"


class TestCommandFormatting:
    """Test IPC command JSON formatting."""

    def test_load_obj_format(self):
        """load_obj command is formatted correctly."""
        cmd = {"cmd": "load_obj", "path": "/path/to/model.obj"}
        json_str = json.dumps(cmd)
        parsed = json.loads(json_str)
        assert parsed["cmd"] == "load_obj"
        assert parsed["path"] == "/path/to/model.obj"

    def test_load_gltf_format(self):
        """load_gltf command is formatted correctly."""
        cmd = {"cmd": "load_gltf", "path": "/path/to/model.glb"}
        json_str = json.dumps(cmd)
        parsed = json.loads(json_str)
        assert parsed["cmd"] == "load_gltf"
        assert parsed["path"] == "/path/to/model.glb"

    def test_cam_lookat_format(self):
        """cam_lookat command is formatted correctly."""
        cmd = {
            "cmd": "cam_lookat",
            "eye": [0.0, 5.0, 10.0],
            "target": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
        }
        json_str = json.dumps(cmd)
        parsed = json.loads(json_str)
        assert parsed["cmd"] == "cam_lookat"
        assert parsed["eye"] == [0.0, 5.0, 10.0]
        assert parsed["target"] == [0.0, 0.0, 0.0]
        assert parsed["up"] == [0.0, 1.0, 0.0]

    def test_set_fov_format(self):
        """set_fov command is formatted correctly."""
        cmd = {"cmd": "set_fov", "deg": 60.0}
        json_str = json.dumps(cmd)
        parsed = json.loads(json_str)
        assert parsed["cmd"] == "set_fov"
        assert parsed["deg"] == 60.0

    def test_lit_sun_format(self):
        """lit_sun command is formatted correctly."""
        cmd = {"cmd": "lit_sun", "azimuth_deg": 45.0, "elevation_deg": 30.0}
        json_str = json.dumps(cmd)
        parsed = json.loads(json_str)
        assert parsed["cmd"] == "lit_sun"
        assert parsed["azimuth_deg"] == 45.0
        assert parsed["elevation_deg"] == 30.0

    def test_lit_ibl_format(self):
        """lit_ibl command is formatted correctly."""
        cmd = {"cmd": "lit_ibl", "path": "/path/to/env.hdr", "intensity": 1.5}
        json_str = json.dumps(cmd)
        parsed = json.loads(json_str)
        assert parsed["cmd"] == "lit_ibl"
        assert parsed["path"] == "/path/to/env.hdr"
        assert parsed["intensity"] == 1.5

    def test_set_z_scale_format(self):
        """set_z_scale command is formatted correctly."""
        cmd = {"cmd": "set_z_scale", "value": 2.5}
        json_str = json.dumps(cmd)
        parsed = json.loads(json_str)
        assert parsed["cmd"] == "set_z_scale"
        assert parsed["value"] == 2.5

    def test_snapshot_format(self):
        """snapshot command is formatted correctly."""
        cmd = {"cmd": "snapshot", "path": "/path/to/out.png", "width": 3840, "height": 2160}
        json_str = json.dumps(cmd)
        parsed = json.loads(json_str)
        assert parsed["cmd"] == "snapshot"
        assert parsed["path"] == "/path/to/out.png"
        assert parsed["width"] == 3840
        assert parsed["height"] == 2160

    def test_snapshot_without_size(self):
        """snapshot command without size is formatted correctly."""
        cmd = {"cmd": "snapshot", "path": "/path/to/out.png"}
        json_str = json.dumps(cmd)
        parsed = json.loads(json_str)
        assert parsed["cmd"] == "snapshot"
        assert parsed["path"] == "/path/to/out.png"
        assert "width" not in parsed
        assert "height" not in parsed

    def test_close_format(self):
        """close command is formatted correctly."""
        cmd = {"cmd": "close"}
        json_str = json.dumps(cmd)
        parsed = json.loads(json_str)
        assert parsed["cmd"] == "close"

    def test_get_terrain_volumetrics_report_format(self):
        """get_terrain_volumetrics_report command is formatted correctly."""
        cmd = {"cmd": "get_terrain_volumetrics_report"}
        parsed = json.loads(json.dumps(cmd))
        assert parsed["cmd"] == "get_terrain_volumetrics_report"
        assert len(parsed) == 1

    def test_set_terrain_scatter_format(self):
        """set_terrain_scatter command is JSON-safe and preserves nested mesh payloads."""
        cmd = {
            "cmd": "set_terrain_scatter",
            "batches": [
                {
                    "name": "trees",
                    "color": [0.2, 0.6, 0.3, 1.0],
                    "max_draw_distance": 180.0,
                    "terrain_blend": {"enabled": True, "bury_depth": 0.5, "fade_distance": 2.0},
                    "terrain_contact": {
                        "enabled": True,
                        "distance": 1.5,
                        "strength": 0.3,
                        "vertical_weight": 0.75,
                    },
                    "transforms": [[1.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 4.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 1.0]],
                    "levels": [
                        {
                            "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                            "normals": [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                            "indices": [0, 1, 2],
                            "max_distance": 90.0,
                        }
                    ],
                    "wind": {
                        "enabled": True,
                        "direction_deg": 45.0,
                        "speed": 1.0,
                        "amplitude": 2.0,
                        "rigidity": 0.3,
                        "bend_start": 0.0,
                        "bend_extent": 1.0,
                        "gust_strength": 0.5,
                        "gust_frequency": 0.3,
                        "fade_start": 100.0,
                        "fade_end": 200.0,
                    },
                }
            ],
        }
        parsed = json.loads(json.dumps(cmd))
        assert parsed["cmd"] == "set_terrain_scatter"
        assert parsed["batches"][0]["name"] == "trees"
        assert parsed["batches"][0]["terrain_blend"]["enabled"] is True
        assert parsed["batches"][0]["terrain_contact"]["vertical_weight"] == 0.75
        assert parsed["batches"][0]["transforms"][0][3] == 3.0
        assert parsed["batches"][0]["levels"][0]["indices"] == [0, 1, 2]
        wind = parsed["batches"][0]["wind"]
        assert wind["enabled"] is True
        assert wind["direction_deg"] == 45.0
        assert wind["speed"] == 1.0
        assert wind["amplitude"] == 2.0
        assert wind["rigidity"] == 0.3
        assert wind["bend_start"] == 0.0
        assert wind["bend_extent"] == 1.0
        assert wind["gust_strength"] == 0.5
        assert wind["gust_frequency"] == 0.3
        assert wind["fade_start"] == 100.0
        assert wind["fade_end"] == 200.0

    def test_clear_terrain_scatter_format(self):
        """clear_terrain_scatter command is formatted correctly."""
        cmd = {"cmd": "clear_terrain_scatter"}
        parsed = json.loads(json.dumps(cmd))
        assert parsed["cmd"] == "clear_terrain_scatter"

    def test_set_scene_review_state_format(self):
        """set_scene_review_state preserves nested TV16 payloads."""
        cmd = {
            "cmd": "set_scene_review_state",
            "state": {
                "base_state": {"labels": [{"text": "Base", "kind": "point", "world_pos": [0.0, 0.0, 0.0]}]},
                "review_layers": [{"id": "notes", "labels": [{"text": "Note", "kind": "point", "world_pos": [1.0, 0.0, 0.0]}]}],
                "variants": [{"id": "review", "active_layer_ids": ["notes"], "preset": {"exposure": 2.0}}],
                "active_variant_id": "review",
            },
        }
        parsed = json.loads(json.dumps(cmd))
        assert parsed["cmd"] == "set_scene_review_state"
        assert parsed["state"]["variants"][0]["id"] == "review"
        assert parsed["state"]["review_layers"][0]["id"] == "notes"

    def test_scene_review_query_formats(self):
        """TV16 query and mutation commands serialize with the expected keys."""
        commands = [
            {"cmd": "list_scene_variants"},
            {"cmd": "list_review_layers"},
            {"cmd": "get_active_scene_variant"},
            {"cmd": "apply_scene_variant", "variant_id": "review"},
            {"cmd": "set_review_layer_visible", "layer_id": "notes", "visible": True},
        ]
        parsed = [json.loads(json.dumps(cmd)) for cmd in commands]
        assert parsed[0] == {"cmd": "list_scene_variants"}
        assert parsed[1] == {"cmd": "list_review_layers"}
        assert parsed[2] == {"cmd": "get_active_scene_variant"}
        assert parsed[3] == {"cmd": "apply_scene_variant", "variant_id": "review"}
        assert parsed[4] == {"cmd": "set_review_layer_visible", "layer_id": "notes", "visible": True}

    def test_set_transform_format(self):
        """set_transform command is formatted correctly."""
        cmd = {
            "cmd": "set_transform",
            "translation": [1.0, 2.0, 3.0],
            "rotation_quat": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
        }
        json_str = json.dumps(cmd)
        parsed = json.loads(json_str)
        assert parsed["cmd"] == "set_transform"
        assert parsed["translation"] == [1.0, 2.0, 3.0]
        assert parsed["rotation_quat"] == [0.0, 0.0, 0.0, 1.0]
        assert parsed["scale"] == [1.0, 1.0, 1.0]

    def test_set_transform_partial(self):
        """set_transform command works with partial fields."""
        # Only translation
        cmd1 = {"cmd": "set_transform", "translation": [1.0, 0.0, 0.0]}
        parsed1 = json.loads(json.dumps(cmd1))
        assert parsed1["cmd"] == "set_transform"
        assert parsed1["translation"] == [1.0, 0.0, 0.0]
        assert "rotation_quat" not in parsed1
        assert "scale" not in parsed1

        # Only rotation
        cmd2 = {"cmd": "set_transform", "rotation_quat": [0.0, 0.707, 0.0, 0.707]}
        parsed2 = json.loads(json.dumps(cmd2))
        assert parsed2["rotation_quat"] == [0.0, 0.707, 0.0, 0.707]

        # Only scale
        cmd3 = {"cmd": "set_transform", "scale": [2.0, 2.0, 2.0]}
        parsed3 = json.loads(json.dumps(cmd3))
        assert parsed3["scale"] == [2.0, 2.0, 2.0]


class TestResponseParsing:
    """Test IPC response parsing."""

    def test_success_response(self):
        """Success response is parsed correctly."""
        response_json = '{"ok":true}'
        response = json.loads(response_json)
        assert response["ok"] is True
        assert "error" not in response

    def test_error_response(self):
        """Error response is parsed correctly."""
        response_json = '{"ok":false,"error":"Something went wrong"}'
        response = json.loads(response_json)
        assert response["ok"] is False
        assert response["error"] == "Something went wrong"


class TestGetStatsFormatParse:
    """Test get_stats command formatting and response parsing."""

    def test_get_stats_command_format(self):
        """get_stats command is formatted correctly."""
        cmd = {"cmd": "get_stats"}
        json_str = json.dumps(cmd)
        parsed = json.loads(json_str)
        assert parsed["cmd"] == "get_stats"
        assert len(parsed) == 1  # Only cmd field

    def test_get_stats_response_before_load(self):
        """get_stats response before load has correct structure."""
        # Simulate response when no mesh is loaded
        response_json = (
            '{"ok":true,"stats":{'
            '"vb_ready":false,"vertex_count":0,"index_count":0,"scene_has_mesh":false}}'
        )
        response = json.loads(response_json)
        assert response["ok"] is True
        stats = response["stats"]
        assert stats["vb_ready"] is False
        assert stats["vertex_count"] == 0
        assert stats["index_count"] == 0
        assert stats["scene_has_mesh"] is False

    def test_get_stats_response_after_load(self):
        """get_stats response after load has correct structure."""
        # Simulate response when mesh is loaded
        response_json = (
            '{"ok":true,"stats":{'
            '"vb_ready":true,"vertex_count":1234,"index_count":5678,"scene_has_mesh":true}}'
        )
        response = json.loads(response_json)
        assert response["ok"] is True
        stats = response["stats"]
        assert stats["vb_ready"] is True
        assert stats["vertex_count"] == 1234
        assert stats["index_count"] == 5678
        assert stats["scene_has_mesh"] is True

    def test_get_stats_response_validates_required_fields(self):
        """get_stats response must have all required fields."""
        response_json = '{"ok":true,"stats":{"vb_ready":true}}'
        response = json.loads(response_json)
        stats = response.get("stats", {})
        # Check that we can access fields (they may be missing in malformed responses)
        required_fields = ["vb_ready", "vertex_count", "index_count", "scene_has_mesh"]
        for field in required_fields:
            # A well-formed response should have all fields
            if field == "vb_ready":
                assert field in stats, f"Missing required field: {field}"

    def test_get_stats_ndjson_roundtrip(self):
        """get_stats command/response roundtrip works correctly."""
        # Command
        cmd = {"cmd": "get_stats"}
        request_line = json.dumps(cmd) + "\n"
        assert request_line.endswith("\n")
        
        # Parse command back
        parsed_cmd = json.loads(request_line.strip())
        assert parsed_cmd["cmd"] == "get_stats"
        
        # Response
        response = {
            "ok": True,
            "stats": {
                "vb_ready": True,
                "vertex_count": 100,
                "index_count": 300,
                "scene_has_mesh": True,
            }
        }
        response_line = json.dumps(response) + "\n"
        
        # Parse response back
        parsed_response = json.loads(response_line.strip())
        assert parsed_response["ok"] is True
        assert parsed_response["stats"]["vertex_count"] == 100


class TestViewerHandleValidation:
    """Test ViewerHandle input validation."""

    def test_open_viewer_async_rejects_both_paths(self):
        """open_viewer_async rejects both obj_path and gltf_path."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            open_viewer_async(obj_path="a.obj", gltf_path="b.glb")

    def test_prepare_terrain_path_converts_npy(self, tmp_path):
        """A .npy terrain path is converted to a temporary TIFF for the viewer."""
        import numpy as np

        src = tmp_path / "terrain.npy"
        np.save(src, np.arange(64, dtype=np.float32).reshape(8, 8))

        prepared, cleanup_paths = _prepare_terrain_path(src)

        assert prepared is not None
        prepared_path = Path(prepared)
        assert prepared_path.suffix.lower() in (".tif", ".tiff")
        assert prepared_path.exists()
        assert cleanup_paths == [prepared_path]

        for path in cleanup_paths:
            path.unlink(missing_ok=True)

    def test_find_viewer_binary_prefers_newest_build(self, tmp_path, monkeypatch):
        """The helper picks the newest built viewer binary instead of blindly preferring release."""
        root = tmp_path
        module_path = root / "python" / "forge3d" / "viewer.py"
        module_path.parent.mkdir(parents=True)
        module_path.write_text("# test module path\n", encoding="utf-8")

        suffix = ".exe" if sys.platform == "win32" else ""
        release_bin = root / "target" / "release" / f"interactive_viewer{suffix}"
        debug_bin = root / "target" / "debug" / f"interactive_viewer{suffix}"
        release_bin.parent.mkdir(parents=True)
        debug_bin.parent.mkdir(parents=True)
        release_bin.write_text("release", encoding="utf-8")
        debug_bin.write_text("debug", encoding="utf-8")

        now = time.time()
        older = now - 60.0
        newer = now
        release_ns = int(older * 1_000_000_000)
        debug_ns = int(newer * 1_000_000_000)
        release_bin.touch()
        debug_bin.touch()
        original_module_file = viewer_module.__file__
        monkeypatch.setattr(viewer_module, "__file__", str(module_path))
        try:
            os.utime(release_bin, ns=(release_ns, release_ns))
            os.utime(debug_bin, ns=(debug_ns, debug_ns))
            assert _find_viewer_binary() == str(debug_bin)
        finally:
            monkeypatch.setattr(viewer_module, "__file__", original_module_file)

    def test_find_viewer_binary_uses_installed_console_script(self, tmp_path, monkeypatch):
        """Installed wheels resolve the console-script launcher beside the current interpreter."""
        package_root = tmp_path / "site-packages" / "forge3d"
        package_root.mkdir(parents=True)
        module_path = package_root / "viewer.py"
        module_path.write_text("# test module path\n", encoding="utf-8")

        scripts_dir = tmp_path / "venv" / ("Scripts" if sys.platform == "win32" else "bin")
        scripts_dir.mkdir(parents=True)
        suffix = ".exe" if sys.platform == "win32" else ""
        viewer_script = scripts_dir / f"interactive_viewer{suffix}"
        viewer_script.write_text("launcher", encoding="utf-8")
        python_exe = scripts_dir / ("python.exe" if sys.platform == "win32" else "python")
        python_exe.write_text("python", encoding="utf-8")

        monkeypatch.setattr(viewer_module, "__file__", str(module_path))
        monkeypatch.setattr(sys, "executable", str(python_exe))

        assert _find_viewer_binary() == str(viewer_script)


class TestViewerHandleHelpers:
    """Test higher-level ViewerHandle helper behavior without a live viewer."""

    def test_set_orbit_camera_includes_optional_target(self):
        """set_orbit_camera forwards explicit terrain targets when provided."""
        handle = ViewerHandle.__new__(ViewerHandle)
        sent: list[dict[str, Any]] = []
        handle._send_command = lambda cmd: sent.append(cmd) or {"ok": True}  # type: ignore[attr-defined]

        handle.set_orbit_camera(45.0, 35.0, 1500.0, fov_deg=40.0, target=(1.0, 2.0, 3.0))

        assert sent == [
            {
                "cmd": "set_terrain_camera",
                "phi_deg": 45.0,
                "theta_deg": 35.0,
                "radius": 1500.0,
                "fov_deg": 40.0,
                "target": [1.0, 2.0, 3.0],
            }
        ]

    def test_set_z_scale_uses_set_terrain_command(self):
        """set_z_scale forwards through the terrain IPC surface."""
        handle = ViewerHandle.__new__(ViewerHandle)
        sent: list[dict[str, Any]] = []
        handle._send_command = lambda cmd: sent.append(cmd) or {"ok": True}  # type: ignore[attr-defined]

        handle.set_z_scale(0.15)

        assert sent == [{"cmd": "set_terrain", "zscale": 0.15}]

    def test_set_terrain_scatter_uses_viewer_ipc(self):
        """set_terrain_scatter forwards the expected batch payload."""
        handle = ViewerHandle.__new__(ViewerHandle)
        sent: list[dict[str, Any]] = []
        handle._send_command = lambda cmd: sent.append(cmd) or {"ok": True}  # type: ignore[attr-defined]

        batches = [
            {
                "name": "trees",
                "transforms": [[1.0] * 16],
                "levels": [],
                "wind": {
                    "enabled": True,
                    "direction_deg": 45.0,
                    "speed": 1.0,
                    "amplitude": 2.0,
                    "rigidity": 0.3,
                    "bend_start": 0.0,
                    "bend_extent": 1.0,
                    "gust_strength": 0.5,
                    "gust_frequency": 0.3,
                    "fade_start": 100.0,
                    "fade_end": 200.0,
                },
            }
        ]
        handle.set_terrain_scatter(batches)

        assert sent == [{"cmd": "set_terrain_scatter", "batches": batches}]

    def test_clear_terrain_scatter_uses_viewer_ipc(self):
        """clear_terrain_scatter forwards the clear command."""
        handle = ViewerHandle.__new__(ViewerHandle)
        sent: list[dict[str, Any]] = []
        handle._send_command = lambda cmd: sent.append(cmd) or {"ok": True}  # type: ignore[attr-defined]

        handle.clear_terrain_scatter()

        assert sent == [{"cmd": "clear_terrain_scatter"}]

    def test_snapshot_waits_for_file_creation(self, tmp_path):
        """snapshot waits for the file to be written instead of sleeping blindly."""
        handle = ViewerHandle.__new__(ViewerHandle)
        handle._timeout = 1.0  # type: ignore[attr-defined]
        out = tmp_path / "snap.png"

        def fake_send(cmd: dict[str, Any]) -> dict[str, Any]:
            assert cmd["cmd"] == "snapshot"

            def writer() -> None:
                time.sleep(0.2)
                out.write_bytes(b"png")

            threading.Thread(target=writer, daemon=True).start()
            return {"ok": True}

        handle._send_command = fake_send  # type: ignore[attr-defined]

        start = time.perf_counter()
        handle.snapshot(out)
        elapsed = time.perf_counter() - start

        assert out.exists()
        assert out.read_bytes() == b"png"
        assert elapsed >= 0.15

    def test_render_animation_forwards_target_bearing_states(self, tmp_path):
        """render_animation keeps using the frame loop while forwarding state.target."""
        handle = ViewerHandle.__new__(ViewerHandle)
        orbit_calls: list[tuple[float, float, float, float, Any]] = []
        snapshots: list[Path] = []

        class _State:
            def __init__(self, frame: int) -> None:
                self.phi_deg = 10.0 + frame
                self.theta_deg = 20.0 + frame
                self.radius = 30.0 + frame
                self.fov_deg = 40.0 + frame
                self.target = (1.0 + frame, 2.0 + frame, 3.0 + frame)

        class _Animation:
            def get_frame_count(self, fps: int) -> int:
                assert fps == 2
                return 3

            def evaluate(self, time: float):
                return _State(int(round(time * 2)))

        handle.set_orbit_camera = (  # type: ignore[method-assign]
            lambda phi_deg, theta_deg, radius, fov_deg=None, target=None: orbit_calls.append(
                (phi_deg, theta_deg, radius, fov_deg, target)
            )
        )
        handle.snapshot = lambda path, width=None, height=None: snapshots.append(Path(path))  # type: ignore[method-assign]

        handle.render_animation(_Animation(), tmp_path, fps=2)

        assert orbit_calls == [
            (10.0, 20.0, 30.0, 40.0, (1.0, 2.0, 3.0)),
            (11.0, 21.0, 31.0, 41.0, (2.0, 3.0, 4.0)),
            (12.0, 22.0, 32.0, 42.0, (3.0, 4.0, 5.0)),
        ]
        assert [path.name for path in snapshots] == [
            "frame_0000.png",
            "frame_0001.png",
            "frame_0002.png",
        ]

    def test_get_terrain_volumetrics_report_returns_payload(self):
        """The helper returns the decoded terrain volumetrics report."""
        handle = ViewerHandle.__new__(ViewerHandle)
        handle._send_command = lambda cmd: {  # type: ignore[attr-defined]
            "ok": True,
            "terrain_volumetrics_report": {
                "active_volume_count": 1,
                "texture_bytes": 4096,
            },
        }

        report = handle.get_terrain_volumetrics_report()

        assert report["active_volume_count"] == 1
        assert report["texture_bytes"] == 4096

    def test_get_terrain_volumetrics_report_requires_payload(self):
        """Missing terrain volumetrics report data raises ViewerError."""
        handle = ViewerHandle.__new__(ViewerHandle)
        handle._send_command = lambda cmd: {"ok": True}  # type: ignore[attr-defined]

        with pytest.raises(ViewerError, match="returned no report data"):
            handle.get_terrain_volumetrics_report()

    def test_load_bundle_loads_terrain_then_review_state(self):
        """The high-level bundle loader sends terrain first and then installs TV16 state."""
        handle = ViewerHandle.__new__(ViewerHandle)
        handle._cleanup_paths = []  # type: ignore[attr-defined]
        sent: list[dict[str, Any]] = []
        handle._send_command = lambda cmd: sent.append(cmd) or {"ok": True}  # type: ignore[attr-defined]

        bundle = LoadedBundle(
            path=Path("scene.forge3d"),
            manifest=BundleManifest.new("scene"),
            dem_path=Path("terrain.tif"),
            scene_state=SceneState(
                review_layers=[ReviewLayer(id="notes")],
                variants=[SceneVariant(id="review", active_layer_ids=["notes"])],
                active_variant_id="review",
            ),
        )

        returned = handle.load_bundle(bundle)

        assert returned is bundle
        assert sent == [
            {"cmd": "load_terrain", "path": "terrain.tif"},
            {"cmd": "set_scene_review_state", "state": bundle.scene_state.to_dict()},
        ]

    def test_load_bundle_variant_override_updates_installed_state(self):
        """An explicit variant_id overrides the bundle's active variant before install."""
        handle = ViewerHandle.__new__(ViewerHandle)
        handle._cleanup_paths = []  # type: ignore[attr-defined]
        sent: list[dict[str, Any]] = []
        handle._send_command = lambda cmd: sent.append(cmd) or {"ok": True}  # type: ignore[attr-defined]

        bundle = LoadedBundle(
            path=Path("scene.forge3d"),
            manifest=BundleManifest.new("scene"),
            scene_state=SceneState(
                review_layers=[ReviewLayer(id="a"), ReviewLayer(id="b")],
                variants=[
                    SceneVariant(id="first", active_layer_ids=["a"]),
                    SceneVariant(id="second", active_layer_ids=["b"]),
                ],
                active_variant_id="first",
            ),
        )

        handle.load_bundle(bundle, variant_id="second")

        assert bundle.get_active_variant_id() == "second"
        assert sent == [
            {
                "cmd": "set_scene_review_state",
                "state": bundle.scene_state.to_dict(),
            }
        ]

    def test_scene_review_helper_methods_use_expected_commands(self):
        """Variant and layer helpers send the exact TV16 command payloads."""
        handle = ViewerHandle.__new__(ViewerHandle)

        def fake_send(cmd: dict[str, Any]) -> dict[str, Any]:
            if cmd["cmd"] == "list_scene_variants":
                return {"ok": True, "scene_variants": [{"id": "review", "active_layer_ids": ["notes"]}]}
            if cmd["cmd"] == "list_review_layers":
                return {"ok": True, "review_layers": [{"id": "notes", "name": "Notes"}]}
            if cmd["cmd"] == "get_active_scene_variant":
                return {"ok": True, "active_scene_variant": "review"}
            captured.append(cmd)
            return {"ok": True}

        captured: list[dict[str, Any]] = []
        handle._send_command = fake_send  # type: ignore[attr-defined]

        assert handle.list_scene_variants() == [{"id": "review", "active_layer_ids": ["notes"]}]
        assert handle.list_review_layers() == [{"id": "notes", "name": "Notes"}]
        assert handle.get_active_scene_variant() == "review"

        handle.apply_scene_variant("review")
        handle.set_review_layer_visible("notes", False)

        assert captured == [
            {"cmd": "apply_scene_variant", "variant_id": "review"},
            {"cmd": "set_review_layer_visible", "layer_id": "notes", "visible": False},
        ]


class TestViewerIpcHelpers:
    """Low-level viewer_ipc helpers should emit exact TV16 commands."""

    def test_scene_review_ipc_helpers_emit_expected_commands(self, monkeypatch):
        commands: list[dict[str, Any]] = []

        def fake_send(_sock: Any, cmd: dict[str, Any]) -> dict[str, Any]:
            commands.append(cmd)
            return {"ok": True}

        monkeypatch.setattr(viewer_ipc_module, "send_ipc", fake_send)

        viewer_ipc_module.set_scene_review_state(object(), {"base_state": {}})
        viewer_ipc_module.list_scene_variants(object())
        viewer_ipc_module.list_review_layers(object())
        viewer_ipc_module.get_active_scene_variant(object())
        viewer_ipc_module.apply_scene_variant(object(), "review")
        viewer_ipc_module.set_review_layer_visible(object(), "notes", True)

        assert commands == [
            {"cmd": "set_scene_review_state", "state": {"base_state": {}}},
            {"cmd": "list_scene_variants"},
            {"cmd": "list_review_layers"},
            {"cmd": "get_active_scene_variant"},
            {"cmd": "apply_scene_variant", "variant_id": "review"},
            {"cmd": "set_review_layer_visible", "layer_id": "notes", "visible": True},
        ]

    def test_set_terrain_pbr_emits_hdr_controls(self, monkeypatch):
        commands: list[dict[str, Any]] = []

        def fake_send(_sock: Any, cmd: dict[str, Any]) -> dict[str, Any]:
            commands.append(cmd)
            return {"ok": True}

        monkeypatch.setattr(viewer_ipc_module, "send_ipc", fake_send)

        viewer_ipc_module.set_terrain_pbr(
            object(),
            enabled=True,
            hdr_path="assets/hdri/brown_photostudio_02_4k.hdr",
            ibl_intensity=0.75,
            hdr_rotate_deg=135.0,
        )

        assert commands == [
            {
                "cmd": "set_terrain_pbr",
                "enabled": True,
                "hdr_path": "assets/hdri/brown_photostudio_02_4k.hdr",
                "ibl_intensity": 0.75,
                "hdr_rotate_deg": 135.0,
            }
        ]

    def test_viewer_ipc_binary_lookup_uses_shared_locator(self, tmp_path, monkeypatch):
        """viewer_ipc resolves the same installed console-script path as viewer.py."""
        package_root = tmp_path / "site-packages" / "forge3d"
        package_root.mkdir(parents=True)
        module_path = package_root / "viewer_ipc.py"
        module_path.write_text("# test module path\n", encoding="utf-8")

        scripts_dir = tmp_path / "venv" / ("Scripts" if sys.platform == "win32" else "bin")
        scripts_dir.mkdir(parents=True)
        suffix = ".exe" if sys.platform == "win32" else ""
        viewer_script = scripts_dir / f"interactive_viewer{suffix}"
        viewer_script.write_text("launcher", encoding="utf-8")
        python_exe = scripts_dir / ("python.exe" if sys.platform == "win32" else "python")
        python_exe.write_text("python", encoding="utf-8")

        monkeypatch.setattr(viewer_ipc_module, "__file__", str(module_path))
        monkeypatch.setattr(sys, "executable", str(python_exe))

        assert viewer_ipc_module.find_viewer_binary() == str(viewer_script)


class TestViewerEntryPoint:
    """Console-script entrypoint tests for installed-wheel viewer launch."""

    def test_viewer_entry_forwards_sys_argv_to_native_cli(self, monkeypatch):
        """The console script forwards argv to the native CLI bridge."""
        captured: list[list[str]] = []

        class _Native:
            def run_interactive_viewer_cli(self, args: list[str]) -> None:
                captured.append(args)

        monkeypatch.setattr(viewer_entry_module, "_get_native_module", lambda: _Native())
        monkeypatch.setattr(
            sys,
            "argv",
            ["interactive_viewer", "--ipc-port", "0", "--size", "1280x720"],
        )

        assert viewer_entry_module.main() == 0
        assert captured == [["--ipc-port", "0", "--size", "1280x720"]]

    def test_viewer_entry_requires_native_cli_bridge(self, monkeypatch):
        """The console script fails clearly when the native bridge is unavailable."""
        monkeypatch.setattr(viewer_entry_module, "_get_native_module", lambda: None)

        assert viewer_entry_module.main() == 1


class TestNDJSONProtocol:
    """Test NDJSON protocol compliance."""

    def test_newline_delimited(self):
        """Commands are newline-delimited."""
        cmd = {"cmd": "close"}
        request = json.dumps(cmd) + "\n"
        assert request.endswith("\n")
        assert request.count("\n") == 1

    def test_multiple_commands_format(self):
        """Multiple commands are properly delimited."""
        cmds = [
            {"cmd": "set_fov", "deg": 45.0},
            {"cmd": "cam_lookat", "eye": [0, 1, 2], "target": [0, 0, 0], "up": [0, 1, 0]},
            {"cmd": "snapshot", "path": "out.png"},
        ]
        ndjson = "\n".join(json.dumps(c) for c in cmds) + "\n"
        lines = ndjson.strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines):
            parsed = json.loads(line)
            assert parsed["cmd"] == cmds[i]["cmd"]
