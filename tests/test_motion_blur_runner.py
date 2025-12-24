#!/usr/bin/env python3
"""
Test script to run motion blur tests and validate output.
This script runs the viewer commands and checks for proper output.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd: str, timeout: int = 120) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout}s")
        return -1, "", "Timeout"
    except Exception as e:
        print(f"ERROR: {e}")
        return -1, "", str(e)

def check_output_image(path: str) -> bool:
    """Check if output image exists and has reasonable size."""
    p = Path(path)
    if not p.exists():
        print(f"  ERROR: Output file not found: {path}")
        return False
    size = p.stat().st_size
    if size < 1000:
        print(f"  ERROR: Output file too small ({size} bytes): {path}")
        return False
    print(f"  OK: {path} ({size} bytes)")
    return True

def main():
    print("Motion Blur Test Runner")
    print("="*60)
    
    # Check if DEM file exists
    dem_path = Path(__file__).parent / "assets" / "dem_rainier.tif"
    if not dem_path.exists():
        print(f"ERROR: DEM file not found: {dem_path}")
        return 1
    
    # Check if binary exists
    binary_path = Path(__file__).parent / "target" / "release" / "interactive_viewer.exe"
    if not binary_path.exists():
        print(f"ERROR: Viewer binary not found: {binary_path}")
        print("Please build with: cargo build --release --bin interactive_viewer")
        return 1
    
    print(f"DEM: {dem_path}")
    print(f"Binary: {binary_path}")
    
    # Test cases
    test_cases = [
        {
            "name": "Camera pan motion blur (horizontal sweep)",
            "cmd": 'python examples/terrain_viewer_interactive.py --dem assets/dem_rainier.tif --pbr --motion-blur --mb-samples 16 --mb-shutter-angle 180 --mb-cam-phi-delta 5 --snapshot motion_blur_pan.png --width 800 --height 600',
            "output": "motion_blur_pan.png",
        },
        {
            "name": "Camera tilt motion blur (vertical sweep)",
            "cmd": 'python examples/terrain_viewer_interactive.py --dem assets/dem_rainier.tif --pbr --motion-blur --mb-samples 16 --mb-shutter-angle 180 --mb-cam-theta-delta 3 --snapshot motion_blur_tilt.png --width 800 --height 600',
            "output": "motion_blur_tilt.png",
        },
        {
            "name": "Zoom blur (dolly in/out)",
            "cmd": 'python examples/terrain_viewer_interactive.py --dem assets/dem_rainier.tif --pbr --motion-blur --mb-samples 16 --mb-shutter-angle 180 --mb-cam-radius-delta -50 --snapshot motion_blur_zoom.png --width 800 --height 600',
            "output": "motion_blur_zoom.png",
        },
        {
            "name": "High-quality motion blur (32 samples)",
            "cmd": 'python examples/terrain_viewer_interactive.py --dem assets/dem_rainier.tif --pbr --motion-blur --mb-samples 32 --mb-shutter-angle 270 --mb-cam-phi-delta 10 --snapshot motion_blur_hq.png --width 800 --height 600',
            "output": "motion_blur_hq.png",
        },
    ]
    
    results = []
    for tc in test_cases:
        print(f"\n\n{'#'*60}")
        print(f"# TEST: {tc['name']}")
        print(f"{'#'*60}")
        
        # Clean up previous output
        out_path = Path(__file__).parent / tc['output']
        if out_path.exists():
            out_path.unlink()
        
        code, stdout, stderr = run_command(tc['cmd'])
        
        success = False
        if code == 0:
            success = check_output_image(str(out_path))
        
        results.append({
            "name": tc['name'],
            "success": success,
            "code": code,
        })
    
    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for r in results:
        status = "PASS" if r['success'] else "FAIL"
        print(f"  [{status}] {r['name']}")
        if not r['success']:
            all_passed = False
    
    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
