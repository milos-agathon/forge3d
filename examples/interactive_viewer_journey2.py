#!/usr/bin/env python3
"""Journey 2: Open blank viewer -> build scene from Python -> snapshot.

This example demonstrates the non-blocking viewer workflow where:
1. Python opens viewer with no initial content
2. User runs Python commands to load objects and adjust scene
3. User saves snapshots at arbitrary resolution

Run from the repository root:
    python examples/interactive_viewer_journey2.py
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

    print("=== Journey 2: Open blank -> build scene from Python -> snapshot ===")
    print()

    try:
        # Step 1: Open viewer blank (no initial content)
        print("Step 1: Opening blank viewer...")
        viewer = open_viewer_async(
            width=1280,
            height=720,
            fov_deg=60.0,
            timeout=30.0,
        )
        print(f"  Viewer started on port {viewer.port}")
        print("  (Window is blank - we'll add content via Python)")
        print()

        # Give viewer time to initialize
        time.sleep(2.0)

        # Step 2: Load an object via Python
        print("Step 2: Loading object via Python...")
        obj_path = Path("assets/objects/bunny.obj")
        if obj_path.exists():
            viewer.load_obj(obj_path)
            print(f"  Loaded: {obj_path}")
        else:
            print(f"  Warning: {obj_path} not found, trying cornell_box.obj")
            alt_path = Path("assets/objects/cornell_box.obj")
            if alt_path.exists():
                viewer.load_obj(alt_path)
                print(f"  Loaded: {alt_path}")
            else:
                print("  No OBJ files found, continuing with empty scene")
        time.sleep(0.5)

        # Step 3: Set camera to view the loaded object
        print("Step 3: Setting camera position...")
        viewer.set_camera_lookat(
            eye=(0.0, 1.0, 4.0),
            target=(0.0, 0.5, 0.0),
            up=(0.0, 1.0, 0.0),
        )
        print("  Camera set to eye=(0, 1, 4), target=(0, 0.5, 0)")
        time.sleep(0.5)

        # Step 4: Adjust sun lighting
        print("Step 4: Setting sun direction...")
        viewer.set_sun(azimuth_deg=135.0, elevation_deg=45.0)
        print("  Sun set to azimuth=135°, elevation=45°")
        time.sleep(0.5)

        # Step 5: Take snapshot at arbitrary resolution
        snapshot1 = out_dir / "journey2_snapshot1.png"
        print(f"Step 5: Taking snapshot at 2560x1440 -> {snapshot1}")
        viewer.snapshot(snapshot1, width=2560, height=1440)
        print(f"  Saved: {snapshot1}")
        print()

        # Step 6: Try different camera angle
        print("Step 6: Changing to different camera angle...")
        viewer.set_camera_lookat(
            eye=(-2.0, 2.0, 2.0),
            target=(0.0, 0.0, 0.0),
        )
        viewer.set_fov(75.0)
        print("  Camera: eye=(-2, 2, 2), FOV=75°")
        time.sleep(0.5)

        # Step 7: Take another snapshot
        snapshot2 = out_dir / "journey2_snapshot2.png"
        print(f"Step 7: Taking snapshot at 1920x1080 -> {snapshot2}")
        viewer.snapshot(snapshot2, width=1920, height=1080)
        print(f"  Saved: {snapshot2}")
        print()

        # Step 8: Close viewer
        print("Step 8: Closing viewer...")
        viewer.close()
        print("  Viewer closed")
        print()

        # Report results
        print("=== Journey 2 Complete ===")
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
