#!/usr/bin/env python3
"""
Test PBR debug modes 7 vs 8 to verify they produce different outputs.
"""
import os
import subprocess
import sys
import hashlib
from pathlib import Path

repo_root = Path(__file__).parents[1]
out_dir = repo_root / "examples" / "out"
out_dir.mkdir(parents=True, exist_ok=True)

def render_with_mode(mode: int, output_name: str) -> str:
    """Render with specified debug mode, return SHA256 of output."""
    env = os.environ.copy()
    for k in list(env.keys()):
        if k.startswith("VF_"):
            env.pop(k, None)
    env["VF_COLOR_DEBUG_MODE"] = str(mode)
    
    out_path = out_dir / output_name
    cmd = [
        sys.executable,
        str(repo_root / "examples" / "terrain_demo.py"),
        "--dem", str(repo_root / "assets" / "Gore_Range_Albers_1m.tif"),
        "--hdr", str(repo_root / "assets" / "snow_field_4k.hdr"),
        "--size", "400", "300",
        "--msaa", "1",
        "--z-scale", "5.0",
        "--albedo-mode", "material",
        "--gi", "ibl",
        "--output", str(out_path),
        "--overwrite",
    ]
    
    print(f"  Rendering mode {mode} -> {output_name}...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(repo_root))
    
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        return "ERROR"
    
    # Get SHA256
    if out_path.exists():
        h = hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]
        return h
    return "NOT_FOUND"

print("=" * 60)
print("Testing PBR Debug Modes")
print("=" * 60)

# Test various modes
modes = [
    (0, "mode_0_beauty.png", "Beauty (normal)"),
    (7, "mode_7_diffuse.png", "Diffuse IBL only"),
    (8, "mode_8_specular.png", "Specular IBL only"),
    (110, "mode_110_red.png", "Pure red (canary)"),
]

results = {}
for mode, filename, desc in modes:
    hash_val = render_with_mode(mode, filename)
    results[mode] = (hash_val, desc)
    print(f"  Mode {mode} ({desc}): {hash_val}")

print("\n" + "=" * 60)
print("Results Summary")
print("=" * 60)

# Check if modes are different
if results[7][0] == results[8][0]:
    print("[FAIL] Mode 7 (diffuse) == Mode 8 (specular) - PBR debug modes not working!")
else:
    print("[PASS] Mode 7 (diffuse) != Mode 8 (specular) - PBR debug modes working!")

if results[0][0] == results[7][0]:
    print("[WARN] Mode 0 (beauty) == Mode 7 (diffuse) - might indicate issue")
    
if results[110][0] != results[0][0]:
    print("[PASS] Mode 110 (red) != Mode 0 (beauty) - canary works!")
else:
    print("[FAIL] Mode 110 (red) == Mode 0 (beauty) - debug modes broken!")

print(f"\nImages saved to: {out_dir}")
