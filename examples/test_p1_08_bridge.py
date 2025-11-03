#!/usr/bin/env python3
"""
P1-08 Test: Python bridge for set_lights()
Tests that Python dicts are correctly parsed and uploaded to LightBuffer
"""

import sys
from pathlib import Path

# Add forge3d to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import forge3d as f3d
except ImportError as e:
    print(f"❌ Failed to import forge3d: {e}")
    print("Build with: maturin develop --release")
    sys.exit(1)

def test_set_lights():
    """Test set_lights() with various light types"""
    print("P1-08: Testing Python light bridge")
    print("=" * 50)
    
    # Create session and renderer
    print("\n1. Creating renderer...")
    session = f3d.Session()
    renderer = f3d.TerrainRenderer(session)
    print("✓ Renderer created")
    
    # Test 1: Directional light
    print("\n2. Testing directional light...")
    lights = [
        {
            "type": "directional",
            "azimuth": 135.0,
            "elevation": 35.0,
            "intensity": 3.0,
            "color": [1.0, 0.9, 0.8]
        }
    ]
    
    try:
        renderer.set_lights(lights)
        print("✓ Directional light uploaded")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 2: Point light
    print("\n3. Testing point light...")
    lights = [
        {
            "type": "point",
            "position": [0.0, 10.0, 0.0],
            "intensity": 10.0,
            "range": 50.0,
            "color": [1.0, 1.0, 1.0]
        }
    ]
    
    try:
        renderer.set_lights(lights)
        print("✓ Point light uploaded")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 3: Multiple heterogeneous lights
    print("\n4. Testing multiple lights...")
    lights = [
        {
            "type": "directional",
            "azimuth": 225.0,
            "elevation": 45.0,
            "intensity": 5.0,
            "color": [1.0, 0.95, 0.9]
        },
        {
            "type": "point",
            "pos": [10.0, 5.0, 0.0],  # Test 'pos' alias
            "intensity": 10.0,
            "range": 100.0
        },
        {
            "type": "spot",
            "position": [5.0, 8.0, 5.0],
            "direction": [0.0, -1.0, 0.0],
            "cone_angle": 30.0,  # Single angle (inner computed as 75%)
            "intensity": 8.0,
            "range": 100.0
        },
        {
            "type": "area_rect",
            "position": [0.0, 10.0, 0.0],
            "direction": [0.0, -1.0, 0.0],
            "area_extent": [2.0, 1.5],
            "intensity": 15.0
        }
    ]
    
    try:
        renderer.set_lights(lights)
        print(f"✓ {len(lights)} heterogeneous lights uploaded")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 4: Flexible key names
    print("\n5. Testing flexible key aliases...")
    lights = [
        {
            "type": "point",
            "pos": [0, 0, 0],  # 'pos' instead of 'position'
            "power": 5.0,       # 'power' instead of 'intensity'
            "rgb": [1, 0.5, 0]  # 'rgb' instead of 'color'
        }
    ]
    
    try:
        renderer.set_lights(lights)
        print("✓ Flexible key names work")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 5: MAX_LIGHTS enforcement (should fail with >16 lights)
    print("\n6. Testing MAX_LIGHTS enforcement...")
    lights = [
        {"type": "point", "position": [i, 0, 0], "intensity": 1.0}
        for i in range(17)  # 17 lights (>16)
    ]
    
    try:
        renderer.set_lights(lights)
        print("❌ Should have failed with >16 lights")
        return False
    except Exception as e:
        if "Too many lights" in str(e) or "17" in str(e):
            print(f"✓ MAX_LIGHTS enforced: {e}")
        else:
            print(f"❌ Wrong error: {e}")
            return False
    
    print("\n" + "=" * 50)
    print("✅ All P1-08 tests passed!")
    return True

if __name__ == "__main__":
    success = test_set_lights()
    sys.exit(0 if success else 1)
