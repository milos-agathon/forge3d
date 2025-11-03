#!/usr/bin/env python3
"""
examples/lights_ssbo_debug.py
P1-12: Light SSBO debug utility

Parses --light CLI specs, uploads via Renderer.set_lights(), and prints debug info.
Requires P1-06 and P1-08 integration to be complete for full functionality.

Usage:
    python examples/lights_ssbo_debug.py \
        --light type=directional,intensity=3,color=1,0.9,0.8 \
        --light type=point,pos=10,5,0,intensity=10,range=50
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def _split_key_value_string(spec: str) -> dict[str, str]:
    """Parse key=value,key=value string into dictionary."""
    tokens = spec.split(",")
    result: dict[str, str] = {}
    current_key: str | None = None
    for raw in tokens:
        segment = raw.strip()
        if not segment:
            continue
        if "=" in segment:
            key, value = segment.split("=", 1)
            current_key = key.strip().lower()
            result[current_key] = value.strip()
        elif current_key is not None:
            result[current_key] = f"{result[current_key]},{segment}"
        else:
            raise ValueError(f"Invalid segment '{segment}' in specification '{spec}'")
    return result


def _parse_float_list(value: str, length: int, label: str) -> tuple[float, ...]:
    """Parse comma-separated float list."""
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != length:
        raise ValueError(f"{label} requires exactly {length} comma-separated floats")
    return tuple(parts)


def _parse_light_spec(spec: str) -> dict[str, Any]:
    """
    Parse light specification from CLI.
    
    Reuses parsing logic from terrain_demo.py for consistency.
    
    Supported keys:
        - type, light: Light type (directional, point, spot, area_rect, area_disk, etc.)
        - dir, direction: 3D direction vector
        - pos, position: 3D position
        - intensity, power: Light intensity
        - color, rgb: 3D RGB color
        - cone, cone_angle, angle: Spot cone angle (degrees)
        - area, extent, area_extent: Area light size (2D)
        - range: Light range (point/spot)
        - hdr, hdr_path: Environment HDR path
    
    Example:
        type=directional,dir=0.2,0.8,-0.55,intensity=8,color=1,0.96,0.9
    """
    mapping = _split_key_value_string(spec)
    out: dict[str, Any] = {}
    
    for key, val in mapping.items():
        if key in {"type", "light"}:
            out["type"] = val
        elif key in {"dir", "direction"}:
            out["direction"] = _parse_float_list(val, 3, "direction")
        elif key in {"pos", "position"}:
            out["position"] = _parse_float_list(val, 3, "position")
        elif key in {"intensity", "power"}:
            out["intensity"] = float(val)
        elif key in {"color", "rgb"}:
            out["color"] = _parse_float_list(val, 3, "color")
        elif key in {"cone", "cone_angle", "angle"}:
            out["cone_angle"] = float(val)
        elif key in {"area", "extent", "area_extent"}:
            out["area_extent"] = _parse_float_list(val, 2, "area extent")
        elif key in {"range"}:
            out["range"] = float(val)
        elif key in {"hdr", "hdr_path"}:
            out["hdr_path"] = val
    
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Debug utility for light SSBO inspection (P1-12)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single directional light
    python examples/lights_ssbo_debug.py \\
        --light type=directional,intensity=3,color=1,0.9,0.8

    # Multiple heterogeneous lights
    python examples/lights_ssbo_debug.py \\
        --light type=directional,dir=0.3,0.8,0.5,intensity=5 \\
        --light type=point,pos=0,10,0,intensity=10,color=1,1,0.8 \\
        --light type=spot,pos=5,5,5,dir=0,-1,0,cone=30,intensity=8

    # Area lights
    python examples/lights_ssbo_debug.py \\
        --light type=area_rect,pos=0,10,0,dir=0,-1,0,area=2,1.5,intensity=15
        """,
    )
    
    parser.add_argument(
        "--light",
        dest="lights",
        action="append",
        default=[],
        metavar="SPEC",
        help="Light specification in key=value form (repeatable). "
             "Example: type=directional,dir=0.2,0.8,-0.55,intensity=8,color=1,0.96,0.9",
    )
    
    args = parser.parse_args()
    
    if not args.lights:
        print("Error: No lights specified. Use --light to add lights.", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print("  python examples/lights_ssbo_debug.py \\", file=sys.stderr)
        print("    --light type=directional,intensity=3", file=sys.stderr)
        parser.print_help(sys.stderr)
        return 1
    
    # Parse light specifications
    parsed_lights = []
    for i, spec in enumerate(args.lights):
        try:
            light = _parse_light_spec(spec)
            parsed_lights.append(light)
            print(f"Parsed light {i}: {light}")
        except Exception as e:
            print(f"Error parsing light {i} ('{spec}'): {e}", file=sys.stderr)
            return 1
    
    print(f"\n✓ Successfully parsed {len(parsed_lights)} light(s)\n")
    
    # ========================================================================
    # P1-07/P1-12: Native Debug Info Integration
    # ========================================================================
    # Try to use native renderer.light_debug_info() when available
    # Fall back to CPU-only mock output if forge3d is unavailable
    # ========================================================================
    
    print("=" * 70)
    print("SSBO Debug Output (P1-12)")
    print("=" * 70)
    print()
    
    # Try native path first
    native_available = False
    try:
        import forge3d as f3d
        
        # Check if TerrainRenderer and light_debug_info are available
        if hasattr(f3d, 'Session') and hasattr(f3d, 'TerrainRenderer'):
            print("[P1-07] Native debug info available, attempting GPU path...")
            
            try:
                # Create session and renderer
                session = f3d.Session()
                renderer = f3d.TerrainRenderer(session)
                
                # Upload lights to GPU
                renderer.set_lights(parsed_lights)
                
                # Get native debug info
                debug_info = renderer.light_debug_info()
                print()
                print(debug_info)
                
                native_available = True
            except Exception as render_err:
                # Renderer creation or light upload failed (e.g., shader errors)
                print(f"        Renderer creation failed: {type(render_err).__name__}")
                print("        Falling back to CPU-only mode...")
                print()
        else:
            print("[P1-07] forge3d found but TerrainRenderer unavailable")
            print("        Using CPU-only fallback...")
            print()
    except ImportError:
        print("[P1-07] forge3d not available, using CPU-only fallback...")
        print()
    except Exception as e:
        print(f"[P1-07] Unexpected error: {type(e).__name__}")
        print("        Using CPU-only fallback...")
        print()
    
    # CPU fallback if native not available
    if not native_available:
        # Mock debug output format (matches P1-07 LightBuffer.debug_info() format)
        print("LightBuffer Debug Info:")
        print(f"  Count: {len(parsed_lights)} lights")
        print(f"  Frame: 0 (seed: [0.500, 0.755])")
        print()
        
        for i, light in enumerate(parsed_lights):
            light_type = light.get("type", "unknown").capitalize()
            intensity = light.get("intensity", 1.0)
            color = light.get("color", (1.0, 1.0, 1.0))
            
            print(f"  Light {i}: {light_type}")
            print(f"    Intensity: {intensity:.2f}, Color: [{color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f}]")
            
            # Type-specific fields
            if light_type.lower() == "directional":
                direction = light.get("direction")
                if direction:
                    print(f"    Direction: [{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]")
            
            elif light_type.lower() == "point":
                position = light.get("position", (0.0, 0.0, 0.0))
                range_val = light.get("range", 100.0)
                print(f"    Position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}], Range: {range_val:.2f}")
            
            elif light_type.lower() == "spot":
                position = light.get("position", (0.0, 0.0, 0.0))
                direction = light.get("direction", (0.0, -1.0, 0.0))
                cone_angle = light.get("cone_angle", 30.0)
                range_val = light.get("range", 100.0)
                print(f"    Position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}], "
                      f"Direction: [{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]")
                print(f"    Cone: {cone_angle:.1f}°, Range: {range_val:.2f}")
            
            elif light_type.lower() in ("area_rect", "arearect"):
                position = light.get("position", (0.0, 0.0, 0.0))
                direction = light.get("direction", (0.0, -1.0, 0.0))
                area_extent = light.get("area_extent", (1.0, 1.0))
                print(f"    Position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}], "
                      f"Normal: [{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]")
                print(f"    Half-extents: width={area_extent[0]:.2f}, height={area_extent[1]:.2f}")
            
            elif light_type.lower() in ("area_disk", "areadisk"):
                position = light.get("position", (0.0, 0.0, 0.0))
                direction = light.get("direction", (0.0, -1.0, 0.0))
                radius = light.get("area_extent", (1.0,))[0] if "area_extent" in light else light.get("range", 1.0)
                print(f"    Position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}], "
                      f"Normal: [{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]")
                print(f"    Radius: {radius:.2f}")
            
            print()
    
    print("=" * 70)
    print()
    if native_available:
        print("✓ Debug utility completed (native GPU path)")
    else:
        print("✓ Debug utility completed (CPU-only fallback)")
        print()
        print("To enable full native SSBO inspection:")
        print("  1. Build forge3d with: maturin develop --release")
        print("  2. Ensure P1-06, P1-08, P1-09 are integrated")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
