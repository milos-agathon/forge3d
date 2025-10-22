"""Debug test to verify sampling modes are actually different"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_sampling_patterns():
    """Test that different sampling modes produce different ray patterns"""
    from forge3d.render import render_raytrace_mesh
    
    # Simple cube mesh (12 triangles to avoid GPU LBVH edge cases)
    vertices = np.array([
        # Bottom face (Y=0)
        [-0.5, 0.0, -0.5], [0.5, 0.0, -0.5], [0.5, 0.0, 0.5], [-0.5, 0.0, 0.5],
        # Top face (Y=1)
        [-0.5, 1.0, -0.5], [0.5, 1.0, -0.5], [0.5, 1.0, 0.5], [-0.5, 1.0, 0.5],
    ], dtype=np.float32)
    
    indices = np.array([
        # Bottom
        [0, 1, 2], [0, 2, 3],
        # Top
        [4, 6, 5], [4, 7, 6],
        # Sides
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ], dtype=np.uint32)
    
    camera = {
        'origin': (1.5, 1.5, 2.0),
        'look_at': (0.0, 0.5, 0.0),
        'up': (0.0, 1.0, 0.0),
        'fov_y': 45.0,
    }
    
    # Small image, single SPP to see per-pixel sampling
    width, height = 64, 64
    spp = 1  # Single sample to see the pattern clearly
    
    print("\nSampling Pattern Test (1 SPP)")
    print("="*60)
    
    images = {}
    for mode in ['rng', 'sobol', 'cmj']:
        print(f"\nRendering with {mode.upper()}...")
        img, _ = render_raytrace_mesh(
            mesh=(vertices, indices),
            width=width,
            height=height,
            camera=camera,
            frames=spp,
            seed=42,
            sampling_mode=mode,
            prefer_gpu=True,
            verbose=False,
        )
        images[mode] = img
        
        # Check first 10 pixels
        print(f"First 10 pixels (row 0):")
        for i in range(10):
            r, g, b = img[0, i, :3]
            print(f"  Pixel {i}: RGB=({r:3d}, {g:3d}, {b:3d})")
    
    # Compare images
    print("\n" + "="*60)
    print("Comparing patterns...")
    print("="*60)
    
    rng_vs_sobol = np.sum(np.abs(images['rng'].astype(int) - images['sobol'].astype(int)))
    rng_vs_cmj = np.sum(np.abs(images['rng'].astype(int) - images['cmj'].astype(int)))
    sobol_vs_cmj = np.sum(np.abs(images['sobol'].astype(int) - images['cmj'].astype(int)))
    
    print(f"\nTotal pixel differences:")
    print(f"  RNG vs Sobol: {rng_vs_sobol}")
    print(f"  RNG vs CMJ:   {rng_vs_cmj}")
    print(f"  Sobol vs CMJ: {sobol_vs_cmj}")
    
    if rng_vs_sobol == 0 and rng_vs_cmj == 0:
        print("\n⚠️  WARNING: All sampling modes produce IDENTICAL output!")
        print("    The sampling_mode parameter may not be reaching the shader.")
        return False
    else:
        print("\n✓ Sampling modes produce different patterns (working correctly)")
        return True

if __name__ == '__main__':
    success = test_sampling_patterns()
    sys.exit(0 if success else 1)
