#!/usr/bin/env python3
"""
Canary test for debug mode plumbing.

Mode 110 should produce a pure red image.
If it doesn't, the debug mode isn't reaching the shader.
"""
import os
import subprocess
import sys
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "examples" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Test 1: Mode 110 (pure red canary)
    print("Testing debug mode 110 (pure red canary)...")
    env = os.environ.copy()
    env["VF_COLOR_DEBUG_MODE"] = "110"
    
    cmd = [
        sys.executable,
        str(repo_root / "examples" / "terrain_demo.py"),
        "--dem", str(repo_root / "assets" / "Gore_Range_Albers_1m.tif"),
        "--hdr", str(repo_root / "assets" / "snow_field_4k.hdr"),
        "--size", "200", "150",
        "--msaa", "1",
        "--z-scale", "5.0",
        "--albedo-mode", "material",
        "--gi", "ibl",
        "--output", str(out_dir / "canary_mode_110.png"),
        "--overwrite",
    ]
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(repo_root))
    print(f"  Return code: {result.returncode}")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:500]}")
        return 1
    
    output_path = out_dir / "canary_mode_110.png"
    if not output_path.exists():
        print(f"  FAIL: Output not created at {output_path}")
        return 1
    
    # Check if image is predominantly red
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(output_path).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        
        # Check average color
        avg_r = np.mean(arr[:, :, 0])
        avg_g = np.mean(arr[:, :, 1])
        avg_b = np.mean(arr[:, :, 2])
        
        print(f"  Average RGB: ({avg_r:.1f}, {avg_g:.1f}, {avg_b:.1f})")
        
        # For pure red, R should be ~255, G and B should be ~0
        if avg_r > 200 and avg_g < 50 and avg_b < 50:
            print("  PASS: Image is predominantly RED - debug mode 110 is working!")
            return 0
        else:
            print("  FAIL: Image is NOT red - debug mode is NOT reaching the shader!")
            print("        This means VF_COLOR_DEBUG_MODE env var is not being read or")
            print("        the overlay uniform isn't being bound to the shader.")
            return 1
    except ImportError:
        print("  WARNING: PIL/numpy not available, cannot verify image content")
        print(f"  Output created: {output_path} ({output_path.stat().st_size} bytes)")
        return 0

if __name__ == "__main__":
    sys.exit(main())
