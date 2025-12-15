#!/usr/bin/env python3
"""Journey 1: Open populated scene -> interact -> live Python updates -> snapshot.

This example demonstrates the non-blocking viewer workflow where:
1. Python opens viewer already populated with a model
2. User can manipulate the view with mouse
3. User can run Python commands while viewer stays open to update scene live
4. User can save snapshots at arbitrary resolution

Run from the repository root:
    python examples/interactive_viewer_journey1.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add the python package to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from forge3d.viewer import open_viewer_async, ViewerError


def main() -> int:
    # Ensure output directory exists
    out_dir = Path("examples/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find an OBJ file to use
    obj_path = Path("assets/objects/bunny.obj")
    if not obj_path.exists():
        print(f"Warning: {obj_path} not found, starting with empty scene")
        obj_path = None

    print("=== Journey 1: Open populated -> interact -> live Python updates -> snapshot ===")
    print()

    try:
        # Step 1: Open viewer with populated scene
        print("Step 1: Opening viewer with scene...")
        viewer = open_viewer_async(
            width=1280,
            height=720,
            obj_path=obj_path,
            fov_deg=60.0,
            timeout=30.0,
        )
        print(f"  Viewer started on port {viewer.port}")
        print("  (Window is interactive - you can orbit with mouse)")
        print()

        # Give viewer time to initialize and render
        time.sleep(2.0)

        # Step 2: Set initial camera position via Python
        print("Step 2: Setting camera via Python...")
        viewer.set_camera_lookat(
            eye=(0.0, 2.0, 5.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
        )
        print("  Camera set to eye=(0, 2, 5), target=(0, 0, 0)")
        time.sleep(0.5)

        # Step 3: Set FOV
        print("Step 3: Setting FOV to 45 degrees...")
        viewer.set_fov(45.0)
        time.sleep(0.5)

        # Step 4: Take first snapshot at arbitrary resolution
        snapshot1 = out_dir / "journey1_snapshot1.png"
        print(f"Step 4: Taking snapshot at 1920x1080 -> {snapshot1}")
        viewer.snapshot(snapshot1, width=1920, height=1080)
        print(f"  Saved: {snapshot1}")
        print()

        # Step 5: Change camera position (demonstrating live updates)
        print("Step 5: Changing camera position via Python...")
        viewer.set_camera_lookat(
            eye=(3.0, 1.5, 3.0),
            target=(0.0, 0.5, 0.0),
        )
        print("  Camera moved to eye=(3, 1.5, 3), target=(0, 0.5, 0)")
        time.sleep(0.5)

        # Step 6: Change sun direction
        print("Step 6: Setting sun direction...")
        viewer.set_sun(azimuth_deg=45.0, elevation_deg=30.0)
        print("  Sun set to azimuth=45°, elevation=30°")
        time.sleep(0.5)

        # Step 7: Take second snapshot at higher resolution
        snapshot2 = out_dir / "journey1_snapshot2.png"
        print(f"Step 7: Taking snapshot at 3840x2160 -> {snapshot2}")
        viewer.snapshot(snapshot2, width=3840, height=2160)
        print(f"  Saved: {snapshot2}")
        print()

        # Step 8: Close viewer
        print("Step 8: Closing viewer...")
        viewer.close()
        print("  Viewer closed")
        print()

        # Report results
        print("=== Journey 1 Complete ===")
        print("Snapshots created:")
        for snap in [snapshot1, snapshot2]:
            if snap.exists():
                size_kb = snap.stat().st_size / 1024
                print(f"  {snap}: {size_kb:.1f} KB")
            else:
                print(f"  {snap}: NOT FOUND")

        return 0

    except ViewerError as e:
        print(f"Viewer error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"Binary not found: {e}", file=sys.stderr)
        print("Build with: cargo build --release --bin interactive_viewer", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
