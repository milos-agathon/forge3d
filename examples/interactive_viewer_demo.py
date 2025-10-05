# examples/interactive_viewer_demo.py
# Workstream I1: Interactive Viewer Python demonstration
# Shows how to use the forge3d interactive viewer from Python

"""
Interactive Viewer Demo

This example demonstrates the forge3d interactive viewer - a real-time windowed
viewer with orbit and FPS camera controls.

The viewer provides:
- Windowed rendering with winit and wgpu
- Orbit camera mode for rotating around a target
- FPS camera mode for free movement
- 60+ FPS rendering on simple scenes
- DPI scaling support

Controls:
  Tab       - Toggle camera mode (Orbit/FPS)

  Orbit mode:
    Drag    - Rotate camera around target
    Scroll  - Zoom in/out
    Shift+Drag - Pan the target point

  FPS mode:
    WASD    - Move forward/left/backward/right
    Q/E     - Move down/up
    Drag    - Look around (hold left mouse button)
    Shift   - Move faster (2x speed)

  Esc       - Exit viewer
"""

import argparse
from pathlib import Path

try:
    import forge3d as f3d
    FORGE3D_AVAILABLE = True
except ImportError:
    FORGE3D_AVAILABLE = False


def demo_basic_viewer():
    """Demonstrate basic viewer with default settings."""
    print("\n=== Basic Viewer Demo ===")
    print("Opening viewer with default settings...")
    print("- Resolution: 1024x768")
    print("- Camera: Orbit mode (press Tab to switch to FPS)")
    print("- Close window or press Esc to exit\n")

    # Open viewer with default settings
    f3d.open_viewer()


def demo_custom_viewer(width: int, height: int):
    """Demonstrate viewer with custom settings."""
    print(f"\n=== Custom Viewer Demo ===")
    print(f"Opening viewer with custom settings...")
    print(f"- Resolution: {width}x{height}")
    print(f"- Title: Custom Scene Viewer")
    print(f"- FOV: 60 degrees")
    print("- Close window or press Esc to exit\n")

    # Open viewer with custom settings
    f3d.open_viewer(
        width=width,
        height=height,
        title="Custom Scene Viewer",
        vsync=True,
        fov_deg=60.0,
        znear=0.1,
        zfar=1000.0,
    )


def demo_no_vsync():
    """Demonstrate viewer with VSync disabled for maximum FPS."""
    print("\n=== High FPS Viewer Demo ===")
    print("Opening viewer with VSync disabled...")
    print("- VSync: OFF (uncapped frame rate)")
    print("- Watch the FPS counter in the window title")
    print("- Close window or press Esc to exit\n")

    f3d.open_viewer(
        width=1280,
        height=720,
        title="High FPS Viewer - VSync OFF",
        vsync=False,  # Disable VSync for maximum FPS
    )


def main():
    parser = argparse.ArgumentParser(
        description="forge3d interactive viewer demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "custom", "no-vsync"],
        default="basic",
        help="Demo mode to run (default: basic)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Window width for custom mode (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Window height for custom mode (default: 720)",
    )

    args = parser.parse_args()

    if not FORGE3D_AVAILABLE:
        print("ERROR: forge3d module not available")
        print("Install with: pip install -e .")
        print("Or build with: maturin develop --release")
        return 1

    print("="*70)
    print("INTERACTIVE VIEWER DEMO")
    print("="*70)
    print("\nThe forge3d interactive viewer provides real-time 3D exploration.")
    print("This demo will open a window - use the controls shown above.\n")

    try:
        if args.mode == "basic":
            demo_basic_viewer()
        elif args.mode == "custom":
            demo_custom_viewer(args.width, args.height)
        elif args.mode == "no-vsync":
            demo_no_vsync()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n=== Demo Complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
