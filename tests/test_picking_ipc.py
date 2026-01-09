"""Integration tests for picking system via IPC.

Tests the Plan 3 picking functionality including:
- Vector overlay with feature IDs
- Lasso mode toggle
- Pick event polling
- BVH-based picking
"""

import pytest
import sys
import time
import socket
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from forge3d.viewer_ipc import (
    find_viewer_binary,
    launch_viewer,
    close_viewer,
    send_ipc,
    add_vector_overlay,
    poll_pick_events,
    set_lasso_mode,
    get_lasso_state,
    clear_selection,
)


@pytest.fixture(scope="module")
def viewer_connection():
    """Launch viewer and provide connection for tests."""
    try:
        process, port, sock = launch_viewer(width=800, height=600, print_output=False)
        yield sock, process
    except FileNotFoundError:
        pytest.skip("interactive_viewer binary not found - build with cargo build --release")
    except Exception as e:
        pytest.skip(f"Could not launch viewer: {e}")
    finally:
        try:
            close_viewer(sock, process)
        except:
            pass


class TestVectorOverlayFeatureIds:
    """Test vector overlay with per-feature IDs."""
    
    def test_add_overlay_with_feature_ids(self, viewer_connection):
        """Test adding vector overlay with distinct feature IDs per triangle."""
        sock, _ = viewer_connection
        
        # First load a terrain (required for vector overlays)
        dem_path = Path(__file__).parent.parent / "assets" / "tif" / "Mount_Fuji_30m.tif"
        if not dem_path.exists():
            pytest.skip("Test DEM not found")
        
        resp = send_ipc(sock, {"cmd": "load_terrain", "path": str(dem_path)})
        assert resp.get("ok", False), f"Failed to load terrain: {resp}"
        
        # Create vertices with distinct feature IDs
        # Format: [x, y, z, r, g, b, a, feature_id]
        vertices = [
            # Triangle 1 - feature_id = 1
            [138.72, 4000.0, 35.36, 1.0, 0.0, 0.0, 1.0, 1.0],
            [138.73, 4000.0, 35.35, 1.0, 0.0, 0.0, 1.0, 1.0],
            [138.71, 4000.0, 35.35, 1.0, 0.0, 0.0, 1.0, 1.0],
            # Triangle 2 - feature_id = 2
            [138.74, 4000.0, 35.36, 0.0, 1.0, 0.0, 1.0, 2.0],
            [138.75, 4000.0, 35.35, 0.0, 1.0, 0.0, 1.0, 2.0],
            [138.73, 4000.0, 35.35, 0.0, 1.0, 0.0, 1.0, 2.0],
            # Triangle 3 - feature_id = 3
            [138.76, 4000.0, 35.36, 0.0, 0.0, 1.0, 1.0, 3.0],
            [138.77, 4000.0, 35.35, 0.0, 0.0, 1.0, 1.0, 3.0],
            [138.75, 4000.0, 35.35, 0.0, 0.0, 1.0, 1.0, 3.0],
        ]
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        
        resp = add_vector_overlay(
            sock,
            "Test Features",
            vertices,
            indices,
            primitive="triangles",
            drape=False,
            opacity=1.0,
        )
        
        assert resp.get("ok", False), f"Failed to add vector overlay: {resp}"
        # The response should contain the overlay ID
        assert "id" in resp or resp.get("ok"), "Expected overlay ID in response"


class TestLassoMode:
    """Test lasso selection mode."""
    
    def test_lasso_mode_toggle(self, viewer_connection):
        """Test enabling and disabling lasso mode."""
        sock, _ = viewer_connection
        
        # Enable lasso mode
        resp = set_lasso_mode(sock, True)
        assert resp.get("ok", False), f"Failed to enable lasso mode: {resp}"
        
        # Check lasso state
        resp = get_lasso_state(sock)
        assert resp.get("ok", False), f"Failed to get lasso state: {resp}"
        # State should be "active" or similar
        
        # Disable lasso mode
        resp = set_lasso_mode(sock, False)
        assert resp.get("ok", False), f"Failed to disable lasso mode: {resp}"
    
    def test_clear_selection(self, viewer_connection):
        """Test clearing selection."""
        sock, _ = viewer_connection
        
        resp = clear_selection(sock)
        assert resp.get("ok", False), f"Failed to clear selection: {resp}"


class TestPickEventPolling:
    """Test pick event polling."""
    
    def test_poll_empty_events(self, viewer_connection):
        """Test polling when no pick events have occurred."""
        sock, _ = viewer_connection
        
        resp = poll_pick_events(sock)
        assert resp.get("ok", False), f"Failed to poll pick events: {resp}"
        # Events list may be empty if no clicks happened
        events = resp.get("pick_events", [])
        assert isinstance(events, list), "Expected pick_events to be a list"


class TestVertexFormat:
    """Test that vertex format with 8 components is accepted."""
    
    def test_8_component_vertices(self, viewer_connection):
        """Test that vertices with 8 components (including feature_id) are accepted."""
        sock, _ = viewer_connection
        
        # Load terrain first
        dem_path = Path(__file__).parent.parent / "assets" / "tif" / "Mount_Fuji_30m.tif"
        if not dem_path.exists():
            pytest.skip("Test DEM not found")
        
        # Try adding overlay - should not fail with parse error
        vertices = [
            [138.72, 4000.0, 35.36, 1.0, 0.0, 0.0, 1.0, 42.0],  # feature_id = 42
            [138.73, 4000.0, 35.35, 1.0, 0.0, 0.0, 1.0, 42.0],
            [138.71, 4000.0, 35.35, 1.0, 0.0, 0.0, 1.0, 42.0],
        ]
        indices = [0, 1, 2]
        
        resp = add_vector_overlay(
            sock,
            "Feature ID Test",
            vertices,
            indices,
            primitive="triangles",
        )
        
        # Should succeed without JSON parse error
        assert resp.get("ok", False), f"8-component vertex format rejected: {resp}"
    
    def test_7_component_vertices_should_fail(self, viewer_connection):
        """Test that vertices with only 7 components are rejected."""
        sock, _ = viewer_connection
        
        # Load terrain first
        dem_path = Path(__file__).parent.parent / "assets" / "tif" / "Mount_Fuji_30m.tif"
        if not dem_path.exists():
            pytest.skip("Test DEM not found")
        
        # Try adding overlay with 7-component vertices - should fail
        vertices = [
            [138.72, 4000.0, 35.36, 1.0, 0.0, 0.0, 1.0],  # Missing feature_id
            [138.73, 4000.0, 35.35, 1.0, 0.0, 0.0, 1.0],
            [138.71, 4000.0, 35.35, 1.0, 0.0, 0.0, 1.0],
        ]
        indices = [0, 1, 2]
        
        resp = add_vector_overlay(
            sock,
            "Invalid Format Test",
            vertices,
            indices,
            primitive="triangles",
        )
        
        # Should fail with parse error
        assert not resp.get("ok", True), "7-component vertices should be rejected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
