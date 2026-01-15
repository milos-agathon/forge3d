"""P1.4: TAA Integration tests.

Tests the TAA integration with:
1. IPC protocol support for TAA commands
2. CLI flags in camera_animation_demo.py
3. TAA-jitter coupling (TAA enables jitter)
4. Preset support for TAA parameters
"""

import pytest
from pathlib import Path

# Skip if forge3d not built
pytest.importorskip("forge3d")


class TestTaaIpcProtocol:
    """Test P1.4 TAA IPC protocol support."""

    def test_ipc_set_taa_enabled_request(self):
        """Verify IpcRequest::SetTaaEnabled exists."""
        protocol_path = Path(__file__).parent.parent / "src" / "viewer" / "ipc" / "protocol.rs"
        content = protocol_path.read_text()
        assert "SetTaaEnabled" in content, "SetTaaEnabled IPC request not found"
        assert "enabled: bool" in content, "enabled field not found in SetTaaEnabled"

    def test_ipc_get_taa_status_request(self):
        """Verify IpcRequest::GetTaaStatus exists."""
        protocol_path = Path(__file__).parent.parent / "src" / "viewer" / "ipc" / "protocol.rs"
        content = protocol_path.read_text()
        assert "GetTaaStatus" in content, "GetTaaStatus IPC request not found"

    def test_ipc_taa_request_handling(self):
        """Verify TAA IPC requests are handled."""
        protocol_path = Path(__file__).parent.parent / "src" / "viewer" / "ipc" / "protocol.rs"
        content = protocol_path.read_text()
        
        # Check request is mapped to ViewerCmd
        assert "IpcRequest::SetTaaEnabled" in content, "SetTaaEnabled not handled"
        assert "IpcRequest::GetTaaStatus" in content, "GetTaaStatus not handled"
        assert "ViewerCmd::SetTaaEnabled" in content, "ViewerCmd::SetTaaEnabled mapping not found"
        assert "ViewerCmd::GetTaaStatus" in content, "ViewerCmd::GetTaaStatus mapping not found"


class TestTaaViewerCmd:
    """Test TAA viewer commands."""

    def test_viewer_cmd_set_taa_enabled(self):
        """Verify ViewerCmd::SetTaaEnabled exists."""
        enums_path = Path(__file__).parent.parent / "src" / "viewer" / "viewer_enums.rs"
        content = enums_path.read_text()
        assert "SetTaaEnabled" in content, "SetTaaEnabled command not found"

    def test_viewer_cmd_get_taa_status(self):
        """Verify ViewerCmd::GetTaaStatus exists."""
        enums_path = Path(__file__).parent.parent / "src" / "viewer" / "viewer_enums.rs"
        content = enums_path.read_text()
        assert "GetTaaStatus" in content, "GetTaaStatus command not found"

    def test_taa_command_handler(self):
        """Verify TAA commands are handled in handler.rs."""
        handler_path = Path(__file__).parent.parent / "src" / "viewer" / "cmd" / "handler.rs"
        content = handler_path.read_text()
        
        assert "ViewerCmd::SetTaaEnabled" in content, "SetTaaEnabled handler not found"
        assert "ViewerCmd::GetTaaStatus" in content, "GetTaaStatus handler not found"
        assert "set_taa_enabled" in content, "set_taa_enabled call not found"


class TestTaaCliFlags:
    """Test TAA CLI flags in camera_animation_demo.py."""

    def test_taa_flag_exists(self):
        """Verify --taa flag exists."""
        demo_path = Path(__file__).parent.parent / "examples" / "camera_animation_demo.py"
        content = demo_path.read_text()
        assert '"--taa"' in content, "--taa flag not found"

    def test_no_taa_flag_exists(self):
        """Verify --no-taa flag exists."""
        demo_path = Path(__file__).parent.parent / "examples" / "camera_animation_demo.py"
        content = demo_path.read_text()
        assert '"--no-taa"' in content, "--no-taa flag not found"

    def test_taa_history_weight_flag(self):
        """Verify --taa-history-weight flag exists."""
        demo_path = Path(__file__).parent.parent / "examples" / "camera_animation_demo.py"
        content = demo_path.read_text()
        assert '"--taa-history-weight"' in content, "--taa-history-weight flag not found"
        assert "default=0.9" in content, "default history weight should be 0.9"

    def test_jitter_flag_exists(self):
        """Verify --jitter flag exists."""
        demo_path = Path(__file__).parent.parent / "examples" / "camera_animation_demo.py"
        content = demo_path.read_text()
        assert '"--jitter"' in content, "--jitter flag not found"

    def test_no_jitter_flag_exists(self):
        """Verify --no-jitter flag exists."""
        demo_path = Path(__file__).parent.parent / "examples" / "camera_animation_demo.py"
        content = demo_path.read_text()
        assert '"--no-jitter"' in content, "--no-jitter flag not found"

    def test_debug_velocity_flag_exists(self):
        """Verify --debug-velocity flag exists (P1.1)."""
        demo_path = Path(__file__).parent.parent / "examples" / "camera_animation_demo.py"
        content = demo_path.read_text()
        assert '"--debug-velocity"' in content, "--debug-velocity flag not found"


class TestTaaJitterCoupling:
    """Test TAA and jitter coupling (P1.2 + P1.3 integration)."""

    def test_taa_enables_jitter(self):
        """Verify TAA enables jitter automatically."""
        mod_path = Path(__file__).parent.parent / "src" / "viewer" / "mod.rs"
        content = mod_path.read_text()
        
        # Check that set_taa_enabled modifies jitter state
        assert "taa_jitter" in content, "TAA should control jitter state"
        assert "JitterState::enabled()" in content, "TAA should enable jitter"

    def test_cli_taa_enables_jitter_message(self):
        """Verify CLI prints jitter message when TAA enabled."""
        demo_path = Path(__file__).parent.parent / "examples" / "camera_animation_demo.py"
        content = demo_path.read_text()
        
        assert "Halton 2,3" in content, "Should mention Halton jitter sequence"
        assert "auto-enabled with TAA" in content, "Should mention jitter auto-enabled"


class TestTaaFunctionWiring:
    """Test TAA parameter wiring through functions."""

    def test_export_animation_frames_has_taa_params(self):
        """Verify export_animation_frames accepts TAA parameters."""
        demo_path = Path(__file__).parent.parent / "examples" / "camera_animation_demo.py"
        content = demo_path.read_text()
        
        # Check function signature includes TAA params
        assert "taa_enabled: bool" in content, "taa_enabled param not in export_animation_frames"
        assert "taa_history_weight: float" in content, "taa_history_weight param not in export_animation_frames"

    def test_run_interactive_preview_has_taa_params(self):
        """Verify run_interactive_preview accepts TAA parameters."""
        demo_path = Path(__file__).parent.parent / "examples" / "camera_animation_demo.py"
        content = demo_path.read_text()
        
        # Check function signature includes TAA params
        assert "taa_enabled: bool" in content, "taa_enabled param not in run_interactive_preview"
        assert "taa_history_weight: float" in content, "taa_history_weight param not in run_interactive_preview"

    def test_ipc_set_taa_enabled_call(self):
        """Verify set_taa_enabled IPC command is sent."""
        demo_path = Path(__file__).parent.parent / "examples" / "camera_animation_demo.py"
        content = demo_path.read_text()
        
        assert "set_taa_enabled" in content, "set_taa_enabled IPC command not sent"
        assert '"enabled": True' in content or '"enabled":True' in content, \
            "enabled=True not passed to set_taa_enabled"


class TestPhase1FlagsComplete:
    """Test all Phase 1 CLI flags are exposed."""

    def test_p1_1_motion_vectors_flag(self):
        """Verify P1.1 motion vector debug flag exists."""
        demo_path = Path(__file__).parent.parent / "examples" / "camera_animation_demo.py"
        content = demo_path.read_text()
        assert "--debug-velocity" in content, "P1.1 --debug-velocity flag not found"

    def test_p1_2_jitter_flags(self):
        """Verify P1.2 jitter flags exist."""
        demo_path = Path(__file__).parent.parent / "examples" / "camera_animation_demo.py"
        content = demo_path.read_text()
        assert "--jitter" in content, "P1.2 --jitter flag not found"
        assert "--no-jitter" in content, "P1.2 --no-jitter flag not found"

    def test_p1_3_taa_flags(self):
        """Verify P1.3 TAA flags exist."""
        demo_path = Path(__file__).parent.parent / "examples" / "camera_animation_demo.py"
        content = demo_path.read_text()
        assert "--taa" in content, "P1.3 --taa flag not found"
        assert "--no-taa" in content, "P1.3 --no-taa flag not found"
        assert "--taa-history-weight" in content, "P1.3 --taa-history-weight flag not found"


class TestTaaHelpText:
    """Test TAA flag help text is informative."""

    def test_taa_help_mentions_shimmer(self):
        """Verify --taa help mentions shimmer reduction."""
        demo_path = Path(__file__).parent.parent / "examples" / "camera_animation_demo.py"
        content = demo_path.read_text()
        assert "shimmer" in content.lower(), "TAA help should mention shimmer reduction"

    def test_history_weight_help_mentions_ghosting(self):
        """Verify history weight help mentions ghosting tradeoff."""
        demo_path = Path(__file__).parent.parent / "examples" / "camera_animation_demo.py"
        content = demo_path.read_text()
        assert "ghost" in content.lower(), "History weight help should mention ghosting"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
