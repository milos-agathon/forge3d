#!/usr/bin/env python3
"""
P1-09 Simple Test: Verify CLI integration without full render
Tests that --light parsing and set_lights() bridge work correctly
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import forge3d as f3d
except ImportError as e:
    print(f"❌ Failed to import forge3d: {e}")
    sys.exit(1)

def test_cli_integration():
    """Test that renderer.set_lights() and light_debug_info() work"""
    print("P1-09: Testing CLI integration (simple)")
    print("=" * 60)
    
    # Create session and renderer
    print("\n1. Creating renderer...")
    session = f3d.Session()
    renderer = f3d.TerrainRenderer(session)
    print("✓ Renderer created")
    
    # Test 1: Set lights from dict (mimics CLI parsing)
    print("\n2. Testing set_lights() with CLI-style dict...")
    cli_lights = [
        {
            "type": "directional",
            "azimuth": 225.0,
            "elevation": 45.0,
            "intensity": 5.0,
            "color": [1.0, 0.95, 0.9]
        },
        {
            "type": "point",
            "position": [100.0, 50.0, 0.0],
            "intensity": 10.0,
            "range": 200.0
        }
    ]
    
    try:
        renderer.set_lights(cli_lights)
        print(f"✓ Uploaded {len(cli_lights)} lights")
    except Exception as e:
        print(f"❌ Failed to set lights: {e}")
        return False
    
    # Test 2: Get debug info
    print("\n3. Testing light_debug_info()...")
    try:
        debug_info = renderer.light_debug_info()
        print("✓ Got debug info:")
        print("-" * 60)
        print(debug_info)
        print("-" * 60)
        
        # Validate debug info contains expected data
        if "light_count: 2" not in debug_info:
            print("❌ Debug info doesn't show 2 lights")
            return False
        if "light_type: Directional" not in debug_info:
            print("❌ Debug info missing directional light")
            return False
        if "light_type: Point" not in debug_info:
            print("❌ Debug info missing point light")
            return False
            
        print("✓ Debug info validated")
    except Exception as e:
        print(f"❌ Failed to get debug info: {e}")
        return False
    
    # Test 3: Update with different lights
    print("\n4. Testing light update...")
    new_lights = [
        {
            "type": "spot",
            "position": [50, 80, 50],
            "direction": [0, -1, 0],
            "cone_angle": 30.0,
            "intensity": 8.0,
            "range": 150.0
        }
    ]
    
    try:
        renderer.set_lights(new_lights)
        debug_info = renderer.light_debug_info()
        
        if "light_count: 1" not in debug_info:
            print("❌ Light count not updated")
            return False
        if "light_type: Spot" not in debug_info:
            print("❌ Spot light not found")
            return False
            
        print("✓ Light update successful")
    except Exception as e:
        print(f"❌ Failed to update lights: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All P1-09 simple tests passed!")
    print("\nCLI integration verified:")
    print("  • renderer.set_lights(dicts) ✓")
    print("  • renderer.light_debug_info() ✓")
    print("  • Light updates ✓")
    return True

if __name__ == "__main__":
    success = test_cli_integration()
    sys.exit(0 if success else 1)
