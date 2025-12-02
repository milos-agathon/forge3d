#!/usr/bin/env python3
"""
Simple diagnostic to verify VF_COLOR_DEBUG_MODE plumbing.
Captures stderr where the Rust diagnostic prints.
"""
import os
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
out_path = repo_root / "examples" / "out" / "diag_mode_7.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
diag_log = repo_root / "diag_log.txt"

env = os.environ.copy()
env["VF_COLOR_DEBUG_MODE"] = "7"
env["RUST_LOG"] = "debug"

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
    "--output", str(out_path),
    "--overwrite",
]

lines = []
lines.append(f"Running with VF_COLOR_DEBUG_MODE=7")
lines.append(f"Command: {' '.join(cmd)}")
lines.append("-" * 60)

result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(repo_root))

lines.append("STDOUT:")
lines.append(result.stdout)
lines.append("-" * 60)
lines.append("STDERR:")
lines.append(result.stderr)
lines.append("-" * 60)
lines.append(f"Return code: {result.returncode}")
lines.append(f"Output exists: {out_path.exists()}")

if out_path.exists():
    lines.append(f"Output size: {out_path.stat().st_size} bytes")

# Write to file so we can check even if console doesn't show output
with open(diag_log, "w") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"\nLog written to: {diag_log}")
