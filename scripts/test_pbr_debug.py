#!/usr/bin/env python3
"""Quick test of PBR debug modes."""
import os
import subprocess
import sys
from pathlib import Path

OUT_DIR = Path("examples/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_CMD = [
    sys.executable, "examples/terrain_demo.py",
    "--dem", "assets/Gore_Range_Albers_1m.tif",
    "--hdr", "assets/snow_field_4k.hdr",
    "--size", "400", "225",
    "--msaa", "1",
    "--z-scale", "5.0",
    "--albedo-mode", "material",
    "--gi", "ibl",
    "--ibl-intensity", "1.0",
    "--overwrite",
]

def run_test(name: str, debug_mode: int):
    """Run a single test render."""
    output = OUT_DIR / f"pbr_test_{name}.png"
    cmd = BASE_CMD + ["--output", str(output)]
    
    env = os.environ.copy()
    env["VF_COLOR_DEBUG_MODE"] = str(debug_mode)
    
    print(f"Running {name} (mode {debug_mode})...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:200]}")
        return False
    
    if output.exists():
        print(f"  OK: {output} ({output.stat().st_size} bytes)")
        return True
    else:
        print(f"  FAILED: output not created")
        return False

if __name__ == "__main__":
    tests = [
        ("diffuse_only", 7),
        ("specular_only", 8),
        ("fresnel", 9),
        ("ndotv", 10),
        ("roughness", 11),
        ("energy", 12),
    ]
    
    results = []
    for name, mode in tests:
        results.append(run_test(name, mode))
    
    print(f"\nResults: {sum(results)}/{len(results)} passed")
