"""Test sampling quality: RNG vs Sobol vs CMJ"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_sampling_modes():
    """Compare RNG, Sobol, and CMJ sampling at low SPP"""
    from forge3d.render import render_raytrace_mesh
    
    # Create simple test mesh (sphere approximation)
    n_theta = 20
    n_phi = 20
    vertices = []
    for i in range(n_theta):
        theta = np.pi * i / (n_theta - 1)
        for j in range(n_phi):
            phi = 2 * np.pi * j / n_phi
            x = np.sin(theta) * np.cos(phi)
            y = np.cos(theta)
            z = np.sin(theta) * np.sin(phi)
            vertices.append([x, y, z])
    
    vertices = np.array(vertices, dtype=np.float32)
    
    # Create triangles
    indices = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            j_next = (j + 1) % n_phi
            
            # Two triangles per quad
            v0 = i * n_phi + j
            v1 = i * n_phi + j_next
            v2 = (i + 1) * n_phi + j
            v3 = (i + 1) * n_phi + j_next
            
            indices.append([v0, v1, v2])
            indices.append([v1, v3, v2])
    
    indices = np.array(indices, dtype=np.uint32)
    
    # Camera setup
    camera = {
        'origin': (3.0, 2.0, 3.0),
        'look_at': (0.0, 0.0, 0.0),
        'up': (0.0, 1.0, 0.0),
        'fov_y': 45.0,
        'lighting_azimuth': 315.0,
        'lighting_elevation': 45.0,
    }
    
    width, height = 512, 512
    spp = 16  # Low SPP to see noise differences
    seed = 42
    
    print(f"\n{'='*60}")
    print(f"Sampling Quality Test: {width}x{height} @ {spp} SPP")
    print(f"{'='*60}\n")
    
    results = {}
    
    for sampling_mode in ['rng', 'sobol', 'cmj']:
        print(f"Testing {sampling_mode.upper()} sampling...")
        
        try:
            img, meta = render_raytrace_mesh(
                mesh=(vertices, indices),
                width=width,
                height=height,
                camera=camera,
                frames=spp,
                seed=seed,
                sampling_mode=sampling_mode,
                prefer_gpu=True,
                denoiser=None,
                verbose=False,
            )
            
            # Calculate noise metrics
            center_crop = img[height//4:3*height//4, width//4:3*width//4]
            mean_intensity = center_crop.mean()
            std_intensity = center_crop.std()
            
            # Calculate variance (proxy for noise)
            variance = np.var(center_crop.astype(np.float32))
            
            results[sampling_mode] = {
                'mean': mean_intensity,
                'std': std_intensity,
                'variance': variance,
            }
            
            print(f"  Mean intensity: {mean_intensity:.2f}")
            print(f"  Std deviation:  {std_intensity:.2f}")
            print(f"  Variance:       {variance:.2f}")
            print()
            
        except Exception as e:
            print(f"  ERROR: {e}\n")
            results[sampling_mode] = None
    
    # Compare results
    print(f"\n{'='*60}")
    print("Quality Comparison (lower variance = less noise)")
    print(f"{'='*60}\n")
    
    if all(results.values()):
        rng_var = results['rng']['variance']
        
        for mode in ['sobol', 'cmj']:
            if results[mode]:
                improvement = (rng_var - results[mode]['variance']) / rng_var * 100
                print(f"{mode.upper()} vs RNG: {improvement:+.1f}% variance reduction")
        
        print()
        
        # Determine winner
        variances = [(mode, r['variance']) for mode, r in results.items() if r]
        variances.sort(key=lambda x: x[1])
        
        print(f"Best quality: {variances[0][0].upper()} (variance={variances[0][1]:.2f})")
        print(f"Worst quality: {variances[-1][0].upper()} (variance={variances[-1][1]:.2f})")
    
    return results

if __name__ == '__main__':
    test_sampling_modes()
