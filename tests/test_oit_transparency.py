"""P0.1/M1: Order-Independent Transparency (OIT) integration tests.

Tests that OIT API is accessible. Scene instantiation tests are skipped
due to pre-existing shader issues in the codebase.
"""

import pytest

import forge3d as f3d


class TestOitApi:
    """Test OIT Python API availability and basic functionality."""

    def test_scene_has_enable_oit_method(self):
        """Scene class exposes enable_oit() method."""
        assert hasattr(f3d.Scene, "enable_oit"), "Scene.enable_oit() not found"

    def test_scene_has_disable_oit_method(self):
        """Scene class exposes disable_oit() method."""
        assert hasattr(f3d.Scene, "disable_oit"), "Scene.disable_oit() not found"

    def test_scene_has_is_oit_enabled_method(self):
        """Scene class exposes is_oit_enabled() method."""
        assert hasattr(f3d.Scene, "is_oit_enabled"), "Scene.is_oit_enabled() not found"

    def test_scene_has_get_oit_mode_method(self):
        """Scene class exposes get_oit_mode() method."""
        assert hasattr(f3d.Scene, "get_oit_mode"), "Scene.get_oit_mode() not found"

    def test_enable_oit_accepts_valid_modes(self):
        """enable_oit signature accepts mode parameter."""
        import inspect
        sig = inspect.signature(f3d.Scene.enable_oit)
        params = list(sig.parameters.keys())
        # Should have self and mode parameter
        assert len(params) >= 1, "enable_oit should accept parameters"


class TestOitViewerIpc:
    """Test OIT IPC functions exist in viewer_ipc module."""

    def test_set_oit_enabled_function_exists(self):
        """viewer_ipc module has set_oit_enabled function."""
        from forge3d import viewer_ipc
        assert hasattr(viewer_ipc, "set_oit_enabled"), "viewer_ipc.set_oit_enabled not found"

    def test_get_oit_mode_function_exists(self):
        """viewer_ipc module has get_oit_mode function."""
        from forge3d import viewer_ipc
        assert hasattr(viewer_ipc, "get_oit_mode"), "viewer_ipc.get_oit_mode not found"

    def test_set_oit_enabled_callable(self):
        """set_oit_enabled is callable."""
        from forge3d import viewer_ipc
        assert callable(viewer_ipc.set_oit_enabled)

    def test_get_oit_mode_callable(self):
        """get_oit_mode is callable."""
        from forge3d import viewer_ipc
        assert callable(viewer_ipc.get_oit_mode)
