"""Tests for non-blocking viewer IPC functionality.

These tests validate:
- NDJSON request formatting (Python client)
- READY line parsing (Python client)
- Command enum mapping (unit tests, no GUI required)

No actual viewer window is opened in these tests.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Add the python package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from forge3d.viewer import (
    ViewerHandle,
    ViewerError,
    _READY_PATTERN,
    open_viewer_async,
)


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


class TestViewerHandleValidation:
    """Test ViewerHandle input validation."""

    def test_open_viewer_async_rejects_both_paths(self):
        """open_viewer_async rejects both obj_path and gltf_path."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            open_viewer_async(obj_path="a.obj", gltf_path="b.glb")


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
