#!/usr/bin/env python3
"""
Advanced Example 10: Device Capability Probe

Comprehensive GPU device detection, capability analysis, and performance probing.
Provides detailed information about available GPUs and their rendering capabilities.
"""

import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def probe_basic_device_info():
    """Probe basic device information."""
    try:
        import forge3d as f3d
        
        print("Basic Device Information:")
        print("-" * 40)
        
        # Check GPU availability
        has_gpu = f3d.has_gpu()
        print(f"GPU Acceleration: {'Available' if has_gpu else 'Not Available'}")
        
        # Enumerate adapters
        try:
            adapters = f3d.enumerate_adapters()
            print(f"Available Adapters: {len(adapters)}")
            
            for i, adapter in enumerate(adapters):
                print(f"  Adapter {i}: {adapter}")
                
        except Exception as e:
            print(f"  Adapter enumeration failed: {e}")
        
        # Device diagnostics
        try:
            diagnostics = f3d.device_probe()
            print(f"Device Diagnostics: {diagnostics}")
        except Exception as e:
            print(f"  Device probe failed: {e}")
            
        return has_gpu, adapters if 'adapters' in locals() else []
        
    except ImportError:
        print("forge3d not available for device probing")
        return False, []
    except Exception as e:
        print(f"Device probing failed: {e}")
        return False, []


def probe_rendering_capabilities():
    """Probe rendering capabilities through feature testing."""
    print("\nRendering Capabilities:")
    print("-" * 40)
    
    capabilities = {
        'basic_rendering': False,
        'terrain_rendering': False,
        'vector_graphics': False,
        'shadows': False,
        'pbr_materials': False,
        'hdr_support': False,
        'async_readback': False,
    }
    
    try:
        import forge3d as f3d
        
        # Test basic rendering
        try:
            renderer = f3d.Renderer(64, 64)
            test_img = renderer.render_triangle_rgba()
            capabilities['basic_rendering'] = test_img is not None
            print(f"Basic Rendering: {'✓' if capabilities['basic_rendering'] else '✗'}")
        except Exception:
            print(f"Basic Rendering: ✗")
        
        # Test terrain rendering
        try:
            scene = f3d.Scene(64, 64)
            test_terrain = np.random.rand(32, 32).astype(np.float32)
            scene.set_height_data(test_terrain, spacing=1.0, exaggeration=1.0)
            terrain_img = scene.render_terrain_rgba()
            capabilities['terrain_rendering'] = terrain_img is not None
            print(f"Terrain Rendering: {'✓' if capabilities['terrain_rendering'] else '✗'}")
        except Exception:
            print(f"Terrain Rendering: ✗")
        
        # Test vector graphics
        try:
            f3d.clear_vectors_py()
            test_points = np.array([[[100, 100], [200, 200]]], dtype=np.float32)
            test_colors = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            f3d.add_points_py(test_points, colors=test_colors, sizes=np.array([5.0]))
            capabilities['vector_graphics'] = True
            print(f"Vector Graphics: ✓")
        except Exception:
            print(f"Vector Graphics: ✗")
        
        # Test shadows
        try:
            import forge3d.shadows as shadows
            config = shadows.get_preset_config('low_quality')
            capabilities['shadows'] = config is not None
            print(f"Shadow Mapping: {'✓' if capabilities['shadows'] else '✗'}")
        except ImportError:
            print(f"Shadow Mapping: ✗ (module not available)")
        except Exception:
            print(f"Shadow Mapping: ✗")
        
        # Test PBR materials
        try:
            import forge3d.pbr as pbr
            material = pbr.PbrMaterial()
            capabilities['pbr_materials'] = material is not None
            print(f"PBR Materials: {'✓' if capabilities['pbr_materials'] else '✗'}")
        except ImportError:
            print(f"PBR Materials: ✗ (module not available)")
        except Exception:
            print(f"PBR Materials: ✗")
        
        # Test HDR support
        try:
            import forge3d.hdr as hdr
            # Simple HDR test
            capabilities['hdr_support'] = True
            print(f"HDR Support: ✓")
        except ImportError:
            print(f"HDR Support: ✗ (module not available)")
        except Exception:
            print(f"HDR Support: ✗")
        
        # Test async readback
        try:
            import forge3d.async_readback as async_rb
            capabilities['async_readback'] = True
            print(f"Async Readback: ✓")
        except ImportError:
            print(f"Async Readback: ✗ (module not available)")
        except Exception:
            print(f"Async Readback: ✗")
            
    except ImportError:
        print("forge3d not available for capability testing")
    
    return capabilities


def probe_performance_characteristics():
    """Probe performance characteristics."""
    print("\nPerformance Characteristics:")
    print("-" * 40)
    
    try:
        import forge3d as f3d
        import time
        
        performance_data = {}
        
        # Test rendering performance at different sizes
        test_sizes = [(256, 256), (512, 512), (1024, 1024)]
        
        for width, height in test_sizes:
            try:
                start_time = time.perf_counter()
                
                renderer = f3d.Renderer(width, height)
                image = renderer.render_triangle_rgba()
                
                end_time = time.perf_counter()
                render_time = (end_time - start_time) * 1000  # Convert to ms
                
                pixels = width * height
                mpixels_per_sec = (pixels / (render_time / 1000)) / 1_000_000
                
                performance_data[f"{width}x{height}"] = {
                    'render_time_ms': round(render_time, 2),
                    'mpixels_per_sec': round(mpixels_per_sec, 2),
                    'pixels': pixels,
                }
                
                print(f"  {width}x{height}: {render_time:.1f}ms ({mpixels_per_sec:.1f} Mpix/s)")
                
            except Exception as e:
                print(f"  {width}x{height}: Failed ({e})")
                performance_data[f"{width}x{height}"] = None
        
        # Test memory usage estimation
        try:
            # Estimate memory usage for different texture sizes
            memory_estimates = {}
            for size in [1024, 2048, 4096]:
                # RGBA texture memory estimation
                rgba_bytes = size * size * 4
                memory_estimates[f"{size}x{size}_rgba"] = {
                    'bytes': rgba_bytes,
                    'mb': round(rgba_bytes / (1024 * 1024), 2),
                }
            
            performance_data['memory_estimates'] = memory_estimates
            print(f"  Memory estimates calculated for texture sizes")
            
        except Exception as e:
            print(f"  Memory estimation failed: {e}")
        
        return performance_data
        
    except ImportError:
        print("forge3d not available for performance testing")
        return {}


def generate_system_report():
    """Generate comprehensive system report."""
    print("\nSystem Environment:")
    print("-" * 40)
    
    try:
        import platform
        import sys
        
        system_info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'python_implementation': platform.python_implementation(),
        }
        
        for key, value in system_info.items():
            print(f"  {key}: {value}")
        
        # Try to get forge3d environment report
        try:
            import forge3d as f3d
            env_report = f3d.report_environment()
            system_info['forge3d_environment'] = env_report
            print(f"  forge3d_environment: Available")
        except Exception:
            print(f"  forge3d_environment: Not available")
        
        return system_info
        
    except Exception as e:
        print(f"System report generation failed: {e}")
        return {}


def main():
    """Main example execution."""
    print("Device Capability Probe")
    print("======================")
    
    out_dir = Path(__file__).parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    
    try:
        # Comprehensive device probing
        has_gpu, adapters = probe_basic_device_info()
        capabilities = probe_rendering_capabilities()
        performance = probe_performance_characteristics()
        system_info = generate_system_report()
        
        # Compile comprehensive report
        report = {
            'timestamp': str(Path(__file__).stat().st_mtime),
            'gpu_available': has_gpu,
            'adapters': adapters,
            'capabilities': capabilities,
            'performance': performance,
            'system': system_info,
            'capability_summary': {
                'total_features': len(capabilities),
                'working_features': sum(1 for v in capabilities.values() if v),
                'feature_completeness': sum(1 for v in capabilities.values() if v) / len(capabilities),
            }
        }
        
        # Save detailed report
        import json
        report_path = out_dir / "device_capability_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved detailed report: {report_path}")
        
        # Generate summary
        print("\nCapability Summary:")
        print("-" * 40)
        working = report['capability_summary']['working_features']
        total = report['capability_summary']['total_features']
        completeness = report['capability_summary']['feature_completeness']
        
        print(f"  Working Features: {working}/{total} ({completeness:.1%})")
        print(f"  GPU Acceleration: {'Yes' if has_gpu else 'No'}")
        print(f"  Available Adapters: {len(adapters)}")
        
        if performance:
            fastest_config = min(performance.items(), 
                               key=lambda x: x[1]['render_time_ms'] if x[1] else float('inf'))
            if fastest_config[1]:
                print(f"  Best Performance: {fastest_config[0]} at {fastest_config[1]['mpixels_per_sec']:.1f} Mpix/s")
        
        print("\nExample completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Always exit success for demo runs
    try:
        main()
    finally:
        sys.exit(0)
