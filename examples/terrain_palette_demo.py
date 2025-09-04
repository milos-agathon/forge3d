#!/usr/bin/env python3
"""
Terrain palette switching demo for forge3d

Demonstrates runtime palette switching for terrain rendering,
showing different colormap effects and performance characteristics.
"""

import numpy as np
import os
import time
import sys
from pathlib import Path

# Add repo root to sys.path for forge3d import
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "python"))

try:
    import forge3d as f3d
except ImportError as e:
    print(f"Failed to import forge3d: {e}")
    print("Make sure the package is installed or run 'maturin develop' first")
    sys.exit(1)


def create_varied_terrain(width=64, height=64):
    """Create a varied terrain heightmap for palette demonstration"""
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    X, Y = np.meshgrid(x, y)

    # Combine multiple terrain features
    base_terrain = (
        np.sin(X * 0.5) * np.cos(Y * 0.5) * 0.3 +  # Large hills
        np.sin(X * 2) * np.cos(Y * 2) * 0.1 +      # Medium features
        np.sin(X * 8) * np.cos(Y * 8) * 0.03       # Small details
    )

    # Add some peaks and valleys
    peaks = np.exp(-((X - 6)**2 + (Y - 6)**2) / 4) * 0.4
    valleys = -np.exp(-((X - 12)**2 + (Y - 12)**2) / 3) * 0.2

    terrain = base_terrain + peaks + valleys

    return terrain.astype(np.float32)


def demo_available_palettes():
    """Demo listing and examining available palettes"""
    print("=== Available Terrain Palettes ===")

    try:
        palettes = f3d.list_palettes()
        current = f3d.get_current_palette()

        print(f"Found {len(palettes)} available palettes:")
        for i, palette in enumerate(palettes):
            marker = " <- CURRENT" if palette == current else ""
            print(f"  {i+1:2d}. {palette['name']} ({palette['description']}){marker}")

        print(f"\nCurrent palette: {current}")

        return palettes

    except Exception as e:
        print(f"ERROR: Failed to list palettes: {e}")
        return []


def demo_basic_palette_switching():
    """Demo basic palette switching functionality"""
    print("\n=== Basic Palette Switching ===")

    try:
        palettes = f3d.list_palettes()
        if len(palettes) < 2:
            print("ERROR: Need at least 2 palettes for switching demo")
            return

        # Test switching between first few palettes
        test_palettes = palettes[:min(3, len(palettes))]

        print("Testing palette switching:")
        for palette in test_palettes:
            # Switch palette
            f3d.set_palette(palette)
            current = f3d.get_current_palette()

            # Verify the switch worked
            if current == palette:
                print(f"  OK   Switched to: {palette['name']}")
            else:
                print(f"  WARN Switch failed: requested {palette['name']}, got {current}")

        # Switch back to first palette
        f3d.set_palette(test_palettes[0])
        print(f"\nReset to: {f3d.get_current_palette()}")

    except Exception as e:
        print(f"ERROR: Palette switching failed: {e}")


def demo_visual_palette_comparison():
    """Create visual comparison of different palettes"""
    print("\n=== Visual Palette Comparison ===")

    try:
        palettes = f3d.list_palettes()
        if not palettes:
            print("ERROR: No palettes available")
            return

        # Create terrain heightmap
        terrain = create_varied_terrain(64, 64)
        print(f"Created terrain: {terrain.shape}, range: [{terrain.min():.3f}, {terrain.max():.3f}]")

        # Create renderer
        r = f3d.Renderer(512, 512)
        r.upload_height_r32f(terrain)

        output_dir = "terrain_palette_demo_output"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nGenerating visual comparison for {len(palettes)} palettes:")

        for i, palette in enumerate(palettes):
            try:
                # Switch to this palette
                f3d.set_palette(palette)

                # Render terrain
                rgba = r.render_terrain_rgba()

                # Save image
                output_file = f"{output_dir}/terrain_{i+1:02d}_{palette['name']}.png"
                f3d.numpy_to_png(output_file, rgba)

                print(f"  {i+1:2d}. {palette['name']:<15} -> {output_file}")

            except Exception as e:
                print(f"  {i+1:2d}. {palette['name']:<15} -> ERROR: {e}")

        print(f"\nVisual comparison saved to {output_dir}/")

    except Exception as e:
        print(f"ERROR: Visual comparison failed: {e}")


def demo_palette_performance():
    """Measure palette switching performance"""
    print("\n=== Palette Switching Performance ===")

    try:
        palettes = f3d.list_palettes()
        if len(palettes) < 2:
            print("ERROR: Need at least 2 palettes for performance test")
            return

        # Create terrain
        terrain = create_varied_terrain(32, 32)

        # Create renderer
        r = f3d.Renderer(256, 256)
        r.upload_height_r32f(terrain)

        # Warm up
        f3d.set_palette(palettes[0])
        for _ in range(3):
            r.render_terrain_rgba()

        print("Testing palette switching performance:")

        # Test rapid palette switching
        num_switches = 20
        start_time = time.time()

        for i in range(num_switches):
            palette_idx = i % len(palettes)
            f3d.set_palette(palettes[palette_idx])
            _ = r.render_terrain_rgba()

        elapsed_time = time.time() - start_time

        avg_time = elapsed_time / num_switches
        fps = 1.0 / avg_time if avg_time > 0 else float('inf')

        print(f"  {num_switches} renders with palette switching:")
        print(f"  Total time: {elapsed_time:.3f} seconds")
        print(f"  Average per render: {avg_time:.3f} seconds")
        print(f"  Effective FPS: {fps:.1f}")

        # Compare with no palette switching
        start_time = time.time()
        f3d.set_palette(palettes[0])

        for _ in range(num_switches):
            _ = r.render_terrain_rgba()

        elapsed_no_switch = time.time() - start_time
        avg_no_switch = elapsed_no_switch / num_switches
        fps_no_switch = 1.0 / avg_no_switch if avg_no_switch > 0 else float('inf')

        print(f"\n  {num_switches} renders without palette switching:")
        print(f"  Total time: {elapsed_no_switch:.3f} seconds")
        print(f"  Average per render: {avg_no_switch:.3f} seconds")
        print(f"  Effective FPS: {fps_no_switch:.1f}")

        # Calculate overhead
        overhead = (avg_time - avg_no_switch) / avg_no_switch * 100 if avg_no_switch > 0 else 0
        print(f"\n  Palette switching overhead: {overhead:.1f}%")

        if overhead < 20:
            print("  OK   Low overhead - efficient palette switching")
        elif overhead < 50:
            print("  ~    Moderate overhead - acceptable for most uses")
        else:
            print("  SLOW High overhead - consider optimizations")

    except Exception as e:
        print(f"ERROR: Performance test failed: {e}")


def demo_palette_characteristics():
    """Analyze characteristics of different palettes"""
    print("\n=== Palette Characteristics Analysis ===")

    try:
        palettes = f3d.list_palettes()
        if not palettes:
            print("ERROR: No palettes available")
            return

        # Create a gradient terrain for palette analysis
        gradient_terrain = np.linspace(-0.5, 0.5, 32*32).reshape(32, 32).astype(np.float32)

        r = f3d.Renderer(256, 256)
        r.upload_height_r32f(gradient_terrain)

        print("Analyzing palette color characteristics:")
        print("Palette           | Brightness | Contrast | Dominant Colors")
        print("------------------|------------|----------|----------------")

        for palette in palettes:
            try:
                # Switch palette and render gradient
                f3d.set_palette(palette)
                rgba = r.render_terrain_rgba()

                # Analyze color properties
                rgb = rgba[:, :, :3].astype(np.float32) / 255.0
                brightness = np.mean(rgb)
                contrast = np.std(rgb)

                # Find dominant colors (simplified)
                r_avg = np.mean(rgba[:, :, 0])
                g_avg = np.mean(rgba[:, :, 1])
                b_avg = np.mean(rgba[:, :, 2])

                dominant_channel = max([('R', r_avg), ('G', g_avg), ('B', b_avg)], key=lambda x: x[1])

                print(f"{palette['name']:<17} | {brightness:10.3f} | {contrast:8.3f} | {dominant_channel[0]} dominant")

            except Exception as e:
                print(f"{palette['name']:<17} | ERROR: {str(e)[:30]}")

    except Exception as e:
        print(f"ERROR: Palette analysis failed: {e}")


def demo_terrain_height_mapping():
    """Demonstrate how different palettes map to terrain heights"""
    print("\n=== Terrain Height Mapping Demo ===")

    try:
        # Create terrain with specific height features
        size = 48
        heights = np.zeros((size, size), dtype=np.float32)

        # Create bands of different heights
        band_height = size // 6
        for i in range(6):
            start_row = i * band_height
            end_row = min((i + 1) * band_height, size)
            height_value = -0.4 + (i / 5.0) * 0.8  # Range from -0.4 to +0.4
            heights[start_row:end_row, :] = height_value

        print(f"Created height bands terrain: {heights.shape}")
        print(f"Height bands: {np.unique(heights)}")

        # Render with different palettes to show height mapping
        r = f3d.Renderer(384, 384)
        r.upload_height_r32f(heights)

        palettes = f3d.list_palettes()
        test_palettes = palettes[:min(4, len(palettes))]  # Test first 4 palettes

        output_dir = "terrain_palette_demo_output"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nHeight mapping visualization for {len(test_palettes)} palettes:")

        for palette in test_palettes:
            f3d.set_palette(palette)
            rgba = r.render_terrain_rgba()

            output_file = f"{output_dir}/height_mapping_{palette['name']}.png"
            f3d.numpy_to_png(output_file, rgba)

            print(f"  {palette['name']:<15} -> {output_file}")

        print("\nHeight mapping images show how each palette represents different elevations")

    except Exception as e:
        print(f"ERROR: Height mapping demo failed: {e}")


def demo_interactive_palette_switching():
    """Demonstrate interactive palette switching (simulation)"""
    print("\n=== Interactive Palette Switching Demo ===")

    try:
        palettes = f3d.list_palettes()
        if len(palettes) < 3:
            print("ERROR: Need at least 3 palettes for interactive demo")
            return

        # Simulate interactive switching
        terrain = create_varied_terrain(48, 48)
        r = f3d.Renderer(256, 256)
        r.upload_height_r32f(terrain)

        print("Simulating interactive palette switching sequence:")

        # Simulate a user cycling through palettes
        switch_sequence = [
            (palettes[0], "Initial terrain view"),
            (palettes[1], "User switches to warmer colors"),
            (palettes[2] if len(palettes) > 2 else palettes[0], "User tries cooler colors"),
            (palettes[0], "User returns to original"),
        ]

        for i, (palette, action) in enumerate(switch_sequence):
            print(f"\nStep {i+1}: {action}")

            start_time = time.time()
            f3d.set_palette(palette)
            _ = r.render_terrain_rgba()
            switch_time = time.time() - start_time

            current = f3d.get_current_palette()
            print(f"  Switched to: {current}")
            print(f"  Switch + render time: {switch_time:.3f} seconds")

            if switch_time < 0.1:
                print("  OK   Responsive - good for real-time interaction")
            elif switch_time < 0.5:
                print("  ~    Acceptable - usable for interaction")
            else:
                print("  SLOW Slow - may impact user experience")

        print("\nInteractive demo complete - palette switching is ready for real-time use")

    except Exception as e:
        print(f"ERROR: Interactive demo failed: {e}")


def main():
    """Run all terrain palette demos"""
    print("forge3d Terrain Palette Switching Demo")
    print("======================================")

    try:
        demo_available_palettes()
        demo_basic_palette_switching()
        demo_visual_palette_comparison()
        demo_palette_performance()
        demo_palette_characteristics()
        demo_terrain_height_mapping()
        demo_interactive_palette_switching()

        print("\n=== Demo Complete ===")
        print("Generated images saved to terrain_palette_demo_output/")
        print("Compare the visual effects of different palettes on terrain rendering.")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
