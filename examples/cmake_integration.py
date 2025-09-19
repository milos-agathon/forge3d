#!/usr/bin/env python3
"""CMake Integration Example (Python driver)

This Python script configures and builds the CMake integration example under
`examples/cmake_integration/` and runs the provided CMake targets.

Usage:
  python examples/cmake_integration.py
  python examples/cmake_integration.py --build-dir build/cmake_integration --no-subdir
  python examples/cmake_integration.py --run-cpp

What it does:
- Configures CMake for the example project
- Builds the `run_python_examples` target to validate Python import and a small render
- Optionally runs the `cpp_example` binary (for demonstration)
- Runs `ctest` for the `forge3d_import_test`
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> int:
    p = argparse.ArgumentParser(description="CMake integration example driver")
    p.add_argument("--build-dir", type=str, default="build/cmake_integration",
                   help="Out-of-source build directory (default: build/cmake_integration)")
    p.add_argument("--no-subdir", action="store_true",
                   help="Set -DUSE_FORGE3D_SUBDIR=OFF to use an installed forge3d instead of add_subdirectory")
    p.add_argument("--run-cpp", action="store_true",
                   help="After building, also run the cpp_example binary")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ex_dir = repo_root / "examples" / "cmake_integration"
    build_dir = repo_root / args.build_dir
    build_dir.mkdir(parents=True, exist_ok=True)

    use_subdir = "OFF" if args.no_subdir else "ON"

    # Configure
    run([
        "cmake",
        "-S", str(ex_dir),
        "-B", str(build_dir),
        f"-DUSE_FORGE3D_SUBDIR={use_subdir}",
    ])

    # Build Python examples target
    run(["cmake", "--build", str(build_dir), "--target", "run_python_examples", "-j"])

    # Optionally run the C++ example
    if args.run_cpp:
        cpp_bin = build_dir / "cpp_example"
        if not cpp_bin.exists():
            # On Windows, extension is .exe
            if (build_dir / "cpp_example.exe").exists():
                cpp_bin = build_dir / "cpp_example.exe"
        if cpp_bin.exists():
            run([str(cpp_bin)], cwd=build_dir)
        else:
            print("Warning: cpp_example binary not found; did the generator name it differently?")

    # Run ctest for the simple import test
    run(["ctest", "--output-on-failure"], cwd=build_dir)
    print("CMake integration example completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
