#!/usr/bin/env python3
"""
Test VF_COLOR_DEBUG_MODE via subprocess - mimics proof pack behavior.
"""
import os
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).parents[1]

# Create the environment with VF_COLOR_DEBUG_MODE set
env = os.environ.copy()
# Clear any existing VF_* vars
for k in list(env.keys()):
    if k.startswith("VF_"):
        env.pop(k, None)
# Set the debug mode
env["VF_COLOR_DEBUG_MODE"] = "110"

print(f"[SUBPROCESS TEST] Setting VF_COLOR_DEBUG_MODE=110 in subprocess env")

# Run the direct test script as a subprocess
cmd = [sys.executable, str(repo_root / "tools" / "test_env_direct.py")]
print(f"[SUBPROCESS TEST] Running: {' '.join(cmd)}")
print("-" * 60)

result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(repo_root))

print("STDOUT:")
print(result.stdout)
print("-" * 60)
print("STDERR:")
print(result.stderr)
print("-" * 60)
print(f"Return code: {result.returncode}")

# Check if it passed
if "[PASS]" in result.stdout:
    print("\n[SUBPROCESS TEST] SUCCESS - env var works via subprocess!")
else:
    print("\n[SUBPROCESS TEST] FAILED - env var NOT reaching subprocess!")
