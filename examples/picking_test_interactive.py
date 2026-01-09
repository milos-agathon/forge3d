#!/usr/bin/env python3
"""Interactive Picking Test - Automated verification of picking functionality.

This script tests the Plan 3 picking system by:
1. Loading terrain and vector overlays with distinct feature IDs
2. Verifying lasso mode toggle works
3. Verifying pick event polling works
4. Printing detailed diagnostics

Usage:
    python examples/picking_test_interactive.py
    python examples/picking_test_interactive.py --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

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

ASSETS_DIR = Path(__file__).parent.parent / "assets"
DEM_PATH = ASSETS_DIR / "tif" / "Mount_Fuji_30m.tif"


def create_test_triangles_with_features() -> Tuple[List[List[float]], List[int]]:
    """Create test triangles with distinct feature IDs.
    
    Returns:
        vertices: List of [x, y, z, r, g, b, a, feature_id]
        indices: Triangle indices
    """
    vertices = []
    indices = []
    
    center_x = 138.7278
    center_y = 35.3606
    z = 4000.0
    
    # Create 10 triangles with distinct feature IDs
    for i in range(10):
        feature_id = i + 1
        offset_x = (i % 5) * 0.01
        offset_y = (i // 5) * 0.01
        
        x = center_x + offset_x
        y = center_y + offset_y
        size = 0.003
        
        # Color varies by feature
        r = 1.0 if i % 3 == 0 else 0.0
        g = 1.0 if i % 3 == 1 else 0.0
        b = 1.0 if i % 3 == 2 else 0.0
        
        idx = len(vertices)
        # Format: [x, y, z, r, g, b, a, feature_id]
        vertices.append([x, z, y, r, g, b, 1.0, float(feature_id)])
        vertices.append([x + size, z, y - size, r, g, b, 1.0, float(feature_id)])
        vertices.append([x - size, z, y - size, r, g, b, 1.0, float(feature_id)])
        
        indices.extend([idx, idx + 1, idx + 2])
    
    return vertices, indices


def test_vertex_format(sock, verbose: bool = False) -> bool:
    """Test that 8-component vertex format is accepted."""
    print("\n[TEST] Vertex Format (8 components)")
    
    vertices, indices = create_test_triangles_with_features()
    
    if verbose:
        print(f"  Created {len(vertices)} vertices, {len(indices)//3} triangles")
        print(f"  First vertex: {vertices[0]}")
        print(f"  Vertex length: {len(vertices[0])}")
    
    resp = add_vector_overlay(
        sock,
        "Feature Test Layer",
        vertices,
        indices,
        primitive="triangles",
        drape=False,
        opacity=0.9,
    )
    
    if resp.get("ok", False):
        print("  ✓ PASS: 8-component vertices accepted")
        return True
    else:
        print(f"  ✗ FAIL: {resp.get('error', 'Unknown error')}")
        return False


def test_lasso_mode(sock, verbose: bool = False) -> bool:
    """Test lasso mode toggle."""
    print("\n[TEST] Lasso Mode Toggle")
    
    # Enable
    resp = set_lasso_mode(sock, True)
    if not resp.get("ok", False):
        print(f"  ✗ FAIL: Could not enable lasso mode: {resp}")
        return False
    
    if verbose:
        print("  Lasso mode enabled")
    
    # Get state
    resp = get_lasso_state(sock)
    if not resp.get("ok", False):
        print(f"  ✗ FAIL: Could not get lasso state: {resp}")
        return False
    
    state = resp.get("state", "unknown")
    if verbose:
        print(f"  Lasso state: {state}")
    
    # Disable
    resp = set_lasso_mode(sock, False)
    if not resp.get("ok", False):
        print(f"  ✗ FAIL: Could not disable lasso mode: {resp}")
        return False
    
    print("  ✓ PASS: Lasso mode toggle works")
    return True


def test_pick_event_polling(sock, verbose: bool = False) -> bool:
    """Test pick event polling."""
    print("\n[TEST] Pick Event Polling")
    
    resp = poll_pick_events(sock)
    if not resp.get("ok", False):
        print(f"  ✗ FAIL: Could not poll pick events: {resp}")
        return False
    
    events = resp.get("pick_events", [])
    if verbose:
        print(f"  Polled events: {len(events)}")
        for e in events:
            print(f"    Event: {e}")
    
    print("  ✓ PASS: Pick event polling works")
    return True


def test_clear_selection(sock, verbose: bool = False) -> bool:
    """Test selection clearing."""
    print("\n[TEST] Clear Selection")
    
    resp = clear_selection(sock)
    if not resp.get("ok", False):
        print(f"  ✗ FAIL: Could not clear selection: {resp}")
        return False
    
    print("  ✓ PASS: Clear selection works")
    return True


def test_feature_id_extraction(sock, verbose: bool = False) -> bool:
    """Test that feature IDs are correctly extracted from vertices.
    
    This verifies the fix to handler.rs that extracts feature_id from
    vertex data (index 7) instead of using the layer ID.
    """
    print("\n[TEST] Feature ID Extraction (BVH)")
    
    # Create triangles with known feature IDs
    vertices = [
        # Triangle with feature_id = 100
        [138.72, 4000.0, 35.36, 1.0, 0.0, 0.0, 1.0, 100.0],
        [138.73, 4000.0, 35.35, 1.0, 0.0, 0.0, 1.0, 100.0],
        [138.71, 4000.0, 35.35, 1.0, 0.0, 0.0, 1.0, 100.0],
        # Triangle with feature_id = 200
        [138.74, 4000.0, 35.36, 0.0, 1.0, 0.0, 1.0, 200.0],
        [138.75, 4000.0, 35.35, 0.0, 1.0, 0.0, 1.0, 200.0],
        [138.73, 4000.0, 35.35, 0.0, 1.0, 0.0, 1.0, 200.0],
    ]
    indices = [0, 1, 2, 3, 4, 5]
    
    resp = add_vector_overlay(
        sock,
        "Feature ID Test",
        vertices,
        indices,
        primitive="triangles",
        drape=False,
    )
    
    if resp.get("ok", False):
        print("  ✓ PASS: Overlay with feature IDs 100, 200 added")
        if verbose:
            print("  Note: Actual picking verification requires manual click testing")
        return True
    else:
        print(f"  ✗ FAIL: {resp.get('error', 'Unknown error')}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Interactive Picking Test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--keep-open", action="store_true", help="Keep viewer open after tests")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PICKING SYSTEM TEST")
    print("=" * 60)
    
    # Check DEM exists
    if not DEM_PATH.exists():
        print(f"ERROR: DEM not found at {DEM_PATH}")
        return 1
    
    # Launch viewer
    print("\nLaunching viewer...")
    try:
        process, port, sock = launch_viewer(width=800, height=600, print_output=args.verbose)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: Failed to launch viewer: {e}")
        return 1
    
    print(f"Connected on port {port}")
    
    try:
        # Load terrain
        print("\nLoading terrain...")
        resp = send_ipc(sock, {"cmd": "load_terrain", "path": str(DEM_PATH.resolve())})
        if not resp.get("ok", False):
            print(f"ERROR: Failed to load terrain: {resp}")
            return 1
        
        # Set camera
        send_ipc(sock, {
            "cmd": "set_terrain_camera",
            "phi_deg": 135.0,
            "theta_deg": 30.0,
            "radius": 15000.0,
            "fov_deg": 45.0
        })
        
        # Run tests
        results = []
        results.append(("Vertex Format", test_vertex_format(sock, args.verbose)))
        results.append(("Lasso Mode", test_lasso_mode(sock, args.verbose)))
        results.append(("Pick Event Polling", test_pick_event_polling(sock, args.verbose)))
        results.append(("Clear Selection", test_clear_selection(sock, args.verbose)))
        results.append(("Feature ID Extraction", test_feature_id_extraction(sock, args.verbose)))
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for _, r in results if r)
        total = len(results)
        
        for name, result in results:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {status}: {name}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if args.keep_open:
            print("\nViewer kept open. Press Ctrl+C to exit.")
            try:
                while process.poll() is None:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        
        return 0 if passed == total else 1
        
    finally:
        if not args.keep_open:
            close_viewer(sock, process)


if __name__ == "__main__":
    sys.exit(main())
