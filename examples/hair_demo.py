#!/usr/bin/env python3
"""Hair demo (Python example) that invokes the wavefront GPU example under the hood.

Usage examples:
  python examples/hair_demo.py                     # baseline render (no hair)
  python examples/hair_demo.py --enable-hair       # render with hair demo (default blond)
  python examples/hair_demo.py --enable-hair --hair-width=0.05 --hair-mat=0

This script keeps examples in Python as requested and shells out to the GPU backend.
It writes the example's PNG to out/wavefront_instances.png (same as the Rust example)
so existing tests and tooling continue to work.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Hair demo wrapper (Python example)")
    parser.add_argument("--enable-hair", action="store_true", help="Enable hair demo")
    parser.add_argument("--hair-width", type=float, default=0.02, help="Hair radius (default: 0.02)")
    parser.add_argument(
        "--hair-mat",
        type=int,
        default=1,
        choices=[0, 1],
        help="Hair material: 0=dark, 1=blond (default: 1)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build only; skip GPU execution")
    parser.add_argument("--restir", action="store_true", help="Enable ReSTIR")
    parser.add_argument("--restir-debug", action="store_true", help="Enable ReSTIR debug AOV mode")
    parser.add_argument("--restir-spatial", action="store_true", help="Enable ReSTIR spatial reuse stage")
    parser.add_argument(
        "--output",
        type=str,
        default="out/hair_demo.png",
        help="Optional copy of rendered PNG to this path (default: out/hair_demo.png)",
    )
    args = parser.parse_args()

    # Ensure cargo is present
    if shutil.which("cargo") is None:
        raise RuntimeError("cargo not found in PATH; cannot run GPU example")

    # GPU gating: only run the example if FORGE3D_RUN_WAVEFRONT=1
    run_gpu = os.environ.get("FORGE3D_RUN_WAVEFRONT", "0") == "1" and not args.dry_run
    if not run_gpu:
        print("FORGE3D_RUN_WAVEFRONT != 1, performing build-only dry run (skipping GPU execution)")
        repo_root = Path(__file__).resolve().parents[1]
        r = subprocess.run([
            "cargo", "build", "--no-default-features", "--features", "images"
        ], cwd=str(repo_root))
        if r.returncode != 0:
            raise SystemExit(r.returncode)
        return 0

    # Build base command
    cmd = [
        "cargo",
        "run",
        "--no-default-features",
        "--features",
        "images",
        "--example",
        "wavefront_instances",
    ]
    example_args: list[str] = []
    if args.restir:
        example_args.append("--restir")
    if args.restir_debug:
        example_args.append("--restir-debug")
    if args.restir_spatial:
        example_args.append("--restir-spatial")
    if args.enable_hair:
        example_args.extend([
            "--hair-demo",
            f"--hair-width={args.hair_width}",
            f"--hair-mat={args.hair_mat}",
        ])

    if example_args:
        cmd.append("--")
        cmd.extend(example_args)

    repo_root = Path(__file__).resolve().parents[1]
    out_png = repo_root / "out" / "wavefront_instances.png"

    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(repo_root))
    if r.returncode != 0:
        raise SystemExit(r.returncode)

    if out_png.exists():
        # Optionally copy to requested path for convenience
        dst = Path(args.output)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_png, dst)
        print(f"Saved: {out_png}")
        if dst != out_png:
            print(f"Copied to: {dst}")
    else:
        print("Warning: expected output not found:", out_png)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
