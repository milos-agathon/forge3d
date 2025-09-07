#!/usr/bin/env python3
"""
Environment Mapping and Image-Based Lighting (IBL) Demo

Demonstrates environment mapping with roughness-based lighting effects by:
1. Creating a synthetic HDR environment map
2. Computing luminance values for different roughness levels: L(0.1), L(0.5), L(0.9)
3. Saving the environment map as a tone-mapped PNG image

Usage:
    python examples/environment_mapping.py --headless --out out/environment_demo.png
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import logging

# Add repository root to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import forge3d as f3d
    import forge3d.envmap as envmap
except ImportError as e:
    print(f"ERROR: Could not import forge3d: {e}")
    print("Run: maturin develop --release")
    sys.exit(1)


def create_test_environment(size=256):
    """Create test environment map with specified size."""
    print(f"Creating synthetic HDR environment map ({size}x{size})...")
    
    # Create environment with interesting lighting variation
    env = envmap.EnvironmentMap.create_test_envmap(size)
    
    # Validate the environment map
    validation = envmap.validate_environment_map(env)
    if not validation['valid']:
        raise RuntimeError(f"Invalid environment map: {validation['errors']}")
    
    print(f"Environment map: {env.width}x{env.height}, {validation['statistics']['memory_mb']:.1f} MB")
    print(f"HDR range: {validation['statistics']['min_value']:.3f} - {validation['statistics']['max_value']:.3f}")
    
    return env


def compute_and_print_luminance_values(env):
    """Compute and print L(0.1), L(0.5), L(0.9) values."""
    print("\nComputing roughness luminance values...")
    
    # Test specific roughness values as required by the task
    roughness_values = [0.1, 0.5, 0.9]
    
    # Compute luminance series
    luminances = envmap.compute_roughness_luminance_series(env, roughness_values)
    
    l_01, l_05, l_09 = luminances
    
    print(f"L(0.1) = {l_01:.6f}")
    print(f"L(0.5) = {l_05:.6f}")
    print(f"L(0.9) = {l_09:.6f}")
    
    return luminances


def main():
    parser = argparse.ArgumentParser(description="Environment mapping demonstration")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--out", type=str, default="out/environment_demo.png", help="Output file path")
    parser.add_argument("--size", type=int, default=256, help="Environment map size")
    
    args = parser.parse_args()
    
    print("=== Environment Mapping Demo ===")
    print(f"Mode: {'headless' if args.headless else 'interactive'}")
    print(f"Output: {args.out}")
    print(f"Size: {args.size}x{args.size}")
    
    # Check feature availability
    if not envmap.has_envmap_support():
        print("ERROR: Environment mapping module not available")
        print("This might require additional feature flags or dependencies")
        return 1
    
    try:
        # Step 1: Create test environment with specified size
        env = create_test_environment(args.size)
        
        # Step 2: Compute and print L(0.1), L(0.5), L(0.9) values
        compute_and_print_luminance_values(env)
        
        # Step 3: Save environment map to specified output path
        output_path = Path(args.out)
        
        # Create output directory if missing
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving environment map to: {output_path}")
        envmap.save_environment_map(env, output_path)
        
        print(f"\n=== Environment Mapping Demo Complete ===")
        print(f"Environment map saved: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())