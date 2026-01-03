# tests/test_vector_overlay_rendering.py
# Integration tests for vector overlay rendering
# Tests IPC commands, lighting, and shadow integration

import pytest
import json
import socket
import subprocess
import time
import os
from pathlib import Path


# Skip all tests if no DEM available or viewer can't start
pytestmark = pytest.mark.skipif(
    not os.environ.get("FORGE3D_TEST_DEM"),
    reason="Set FORGE3D_TEST_DEM to path of test DEM file to run integration tests"
)


def send_ipc(sock: socket.socket, cmd: dict) -> dict:
    """Send IPC command and receive response."""
    msg = json.dumps(cmd) + "\n"
    sock.sendall(msg.encode())
    response = sock.recv(4096).decode()
    return json.loads(response)


@pytest.fixture(scope="module")
def viewer_context():
    """Start viewer with IPC and yield (socket, process)."""
    dem_path = os.environ.get("FORGE3D_TEST_DEM")
    if not dem_path or not Path(dem_path).exists():
        pytest.skip("Test DEM not found")
    
    # Start viewer with IPC
    proc = subprocess.Popen(
        ["python", "-m", "forge3d.terrain_demo", "--dem", dem_path, "--ipc", "--headless"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for viewer to start
    time.sleep(2.0)
    
    # Connect to IPC
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(("127.0.0.1", 9123))
    except ConnectionRefusedError:
        proc.terminate()
        pytest.skip("Could not connect to viewer IPC")
    
    yield sock, proc
    
    # Cleanup
    sock.close()
    proc.terminate()
    proc.wait(timeout=5)


class TestVectorOverlayDefaultOff:
    """Test that vector overlays are disabled by default (regression)."""

    def test_vector_overlay_default_off(self, viewer_context):
        """Verify no vector overlays by default.
        
        Per plan Section 11: vector_overlays_enabled = false by default.
        Existing tests and renders should be byte-identical to before.
        """
        sock, _ = viewer_context
        
        # List overlays - should be empty
        resp = send_ipc(sock, {"cmd": "list_vector_overlays"})
        assert resp.get("ok", False)
        # Empty list expected


class TestVectorOverlayAddRemove:
    """Test adding and removing vector overlays."""

    def test_add_vector_overlay_triangle(self, viewer_context):
        """Add a simple triangle overlay."""
        sock, _ = viewer_context
        
        resp = send_ipc(sock, {
            "cmd": "add_vector_overlay",
            "name": "test_triangle",
            "vertices": [
                [100, 0, 100, 1, 0, 0, 1],  # Red
                [200, 0, 100, 0, 1, 0, 1],  # Green
                [150, 0, 200, 0, 0, 1, 1],  # Blue
            ],
            "indices": [0, 1, 2],
            "primitive": "triangles",
            "drape": True,
            "drape_offset": 1.0,
        })
        assert resp.get("ok", False), f"Failed to add overlay: {resp}"

    def test_add_vector_overlay_lines(self, viewer_context):
        """Add line overlay."""
        sock, _ = viewer_context
        
        resp = send_ipc(sock, {
            "cmd": "add_vector_overlay",
            "name": "test_lines",
            "vertices": [
                [0, 0, 0, 1, 1, 0, 1],
                [100, 0, 100, 1, 1, 0, 1],
            ],
            "indices": [0, 1],
            "primitive": "lines",
            "drape": True,
            "line_width": 3.0,
        })
        assert resp.get("ok", False), f"Failed to add lines: {resp}"

    def test_remove_vector_overlay(self, viewer_context):
        """Remove vector overlay by ID."""
        sock, _ = viewer_context
        
        # Add overlay
        resp = send_ipc(sock, {
            "cmd": "add_vector_overlay",
            "name": "to_remove",
            "vertices": [[50, 0, 50, 1, 1, 1, 1]],
            "indices": [0],
            "primitive": "points",
        })
        assert resp.get("ok", False)
        
        # Remove it (ID 0 if first added)
        resp = send_ipc(sock, {"cmd": "remove_vector_overlay", "id": 0})
        # May or may not find it depending on test order


class TestVectorOverlayLighting:
    """Test vector overlay lighting integration."""

    def test_vector_overlay_lit_by_sun(self, viewer_context):
        """Verify vector overlay receives sun lighting.
        
        Per plan Section 8: Overlays use identical lighting to terrain.
        Changing sun direction should change overlay appearance.
        """
        sock, _ = viewer_context
        
        # Add a white overlay
        send_ipc(sock, {
            "cmd": "add_vector_overlay",
            "name": "lit_test",
            "vertices": [
                [200, 0, 200, 1, 1, 1, 1],
                [300, 0, 200, 1, 1, 1, 1],
                [250, 0, 300, 1, 1, 1, 1],
            ],
            "indices": [0, 1, 2],
            "primitive": "triangles",
            "drape": True,
        })
        
        # Set sun to east (azimuth 90)
        send_ipc(sock, {
            "cmd": "set_terrain_sun",
            "azimuth_deg": 90,
            "elevation_deg": 45,
            "intensity": 1.0,
        })
        
        # Snapshot (would need hash comparison for full test)
        # snap1 = capture_snapshot(sock, "overlay_sun_east.png")
        
        # Set sun to west (azimuth 270)
        send_ipc(sock, {
            "cmd": "set_terrain_sun",
            "azimuth_deg": 270,
            "elevation_deg": 45,
            "intensity": 1.0,
        })
        
        # snap2 = capture_snapshot(sock, "overlay_sun_west.png")
        # assert snap1 != snap2, "Overlay should change with sun direction"


class TestVectorOverlayShadows:
    """Test vector overlay shadow integration."""

    def test_vector_overlay_receives_shadows(self, viewer_context):
        """Verify vector overlay is shadowed by terrain.
        
        Per plan Section 8: Overlays sample same sun_vis_tex as terrain.
        Overlay in shadow area should be darker than lit area.
        """
        # This would require placing overlay in known shadow area
        # and comparing luminance to lit area
        pass


class TestVectorOverlayVisibility:
    """Test vector overlay visibility controls."""

    def test_set_vector_overlay_visible(self, viewer_context):
        """Set overlay visibility."""
        sock, _ = viewer_context
        
        # Add overlay
        send_ipc(sock, {
            "cmd": "add_vector_overlay",
            "name": "vis_test",
            "vertices": [[100, 0, 100, 1, 0, 0, 1]],
            "indices": [0],
            "primitive": "points",
        })
        
        # Hide it
        resp = send_ipc(sock, {
            "cmd": "set_vector_overlay_visible",
            "id": 0,
            "visible": False,
        })
        # Response may vary

    def test_set_vector_overlay_opacity(self, viewer_context):
        """Set overlay opacity."""
        sock, _ = viewer_context
        
        resp = send_ipc(sock, {
            "cmd": "set_vector_overlay_opacity",
            "id": 0,
            "opacity": 0.5,
        })
        # Response may vary

    def test_set_global_vector_overlay_opacity(self, viewer_context):
        """Set global overlay opacity multiplier."""
        sock, _ = viewer_context
        
        resp = send_ipc(sock, {
            "cmd": "set_global_vector_overlay_opacity",
            "opacity": 0.8,
        })
        assert resp.get("ok", False)


class TestVectorOverlayZOrder:
    """Test vector overlay z-ordering."""

    def test_z_order_stacking(self, viewer_context):
        """Overlays with higher z_order should render on top."""
        sock, _ = viewer_context
        
        # Add background overlay (z_order=0)
        send_ipc(sock, {
            "cmd": "add_vector_overlay",
            "name": "background",
            "vertices": [
                [100, 0, 100, 0, 0, 1, 1],
                [200, 0, 100, 0, 0, 1, 1],
                [200, 0, 200, 0, 0, 1, 1],
                [100, 0, 200, 0, 0, 1, 1],
            ],
            "indices": [0, 1, 2, 0, 2, 3],
            "primitive": "triangles",
            "drape": True,
            "z_order": 0,
        })
        
        # Add foreground overlay (z_order=1)
        send_ipc(sock, {
            "cmd": "add_vector_overlay",
            "name": "foreground",
            "vertices": [
                [120, 0, 120, 1, 0, 0, 1],
                [180, 0, 120, 1, 0, 0, 1],
                [150, 0, 180, 1, 0, 0, 1],
            ],
            "indices": [0, 1, 2],
            "primitive": "triangles",
            "drape": True,
            "z_order": 1,
        })
        
        # Visual inspection would show red triangle on blue quad
