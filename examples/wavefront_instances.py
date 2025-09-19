#!/usr/bin/env python3
"""Python wrapper for the Wavefront Path Tracer example.

This keeps examples in Python while invoking the GPU backend via cargo.
All original flags are supported and forwarded to the Rust example.

Usage:
  python examples/wavefront_instances.py [flags]

Examples:
  # Baseline
  python examples/wavefront_instances.py

  # Enable ReSTIR + spatial, dump AOVs with header
  python examples/wavefront_instances.py --restir --restir-spatial \
      --dump-aov-depth=out/depth.bin --dump-aov-with-header

  # Enable media and hair demo
  python examples/wavefront_instances.py --medium-enable --medium-sigma-t=0.2 --medium-density=1.0 \
      --hair-demo --hair-width=0.05 --hair-mat=1
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Wavefront Path Tracer (Python example wrapper)")
    # ReSTIR
    p.add_argument("--restir", action="store_true")
    p.add_argument("--restir-debug", action="store_true")
    p.add_argument("--restir-spatial", action="store_true")
    # P2 validation toggles
    p.add_argument("--swap-materials", action="store_true")
    p.add_argument("--skinny-blas1", action="store_true")
    p.add_argument("--camera-jitter", type=float, default=0.0)
    p.add_argument("--force-blas", type=int, choices=[0, 1], default=None)
    # AOV dumping
    p.add_argument("--dump-aov-depth", type=str, default=None)
    p.add_argument("--dump-aov-albedo", type=str, default=None)
    p.add_argument("--dump-aov-normal", type=str, default=None)
    p.add_argument("--dump-aov-with-header", action="store_true")
    # P4 media
    p.add_argument("--medium-enable", action="store_true")
    p.add_argument("--medium-g", type=float, default=0.0)
    p.add_argument("--medium-sigma-t", type=float, default=0.0)
    p.add_argument("--medium-density", type=float, default=0.0)
    # P5 AO compute
    p.add_argument("--compute-ao", action="store_true")
    p.add_argument("--ao-samples", type=int, default=16)
    p.add_argument("--ao-intensity", type=float, default=1.0)
    p.add_argument("--ao-bias", type=float, default=0.025)
    # P5 hair demo
    p.add_argument("--hair-demo", action="store_true")
    p.add_argument("--hair-width", type=float, default=0.02)
    p.add_argument("--hair-mat", type=int, choices=[0, 1], default=1)
    # P6 QMC / adaptive SPP
    p.add_argument("--qmc-mode", type=int, choices=[0, 1], default=0)
    p.add_argument("--spp-limit", type=int, default=None)

    args = p.parse_args()

    if shutil.which("cargo") is None:
        raise RuntimeError("cargo not found in PATH; cannot run GPU example")

    repo_root = Path(__file__).resolve().parents[1]

    # GPU gating: only run the example if FORGE3D_RUN_WAVEFRONT=1
    if os.environ.get("FORGE3D_RUN_WAVEFRONT", "0") != "1":
        print("FORGE3D_RUN_WAVEFRONT != 1, performing build-only dry run (skipping GPU execution)")
        r = subprocess.run([
            "cargo", "build", "--no-default-features", "--features", "images"
        ], cwd=str(repo_root))
        if r.returncode != 0:
            raise SystemExit(r.returncode)
        raise SystemExit(0)

    cmd = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "wavefront_instances",
    ]

    # Forward flags
    ef: list[str] = []
    if args.restir:
        ef.append("--restir")
    if args.restir_debug:
        ef.append("--restir-debug")
    if args.restir_spatial:
        ef.append("--restir-spatial")
    if args.swap_materials:
        ef.append("--swap-materials")
    if args.skinny_blas1:
        ef.append("--skinny-blas1")
    if abs(args.camera_jitter) > 0.0:
        ef.append(f"--camera-jitter={args.camera_jitter}")
    if args.force_blas is not None:
        ef.append(f"--force-blas={args.force_blas}")

    if args.dump_aov_depth:
        ef.append(f"--dump-aov-depth={args.dump_aov_depth}")
    if args.dump_aov_albedo:
        ef.append(f"--dump-aov-albedo={args.dump_aov_albedo}")
    if args.dump_aov_normal:
        ef.append(f"--dump-aov-normal={args.dump_aov_normal}")
    if args.dump_aov_with_header:
        ef.append("--dump-aov-with-header")

    if args.medium_enable:
        ef.append("--medium-enable")
    if args.medium_g != 0.0:
        ef.append(f"--medium-g={args.medium_g}")
    if args.medium_sigma_t != 0.0:
        ef.append(f"--medium-sigma-t={args.medium_sigma_t}")
    if args.medium_density != 0.0:
        ef.append(f"--medium-density={args.medium_density}")

    if args.compute_ao:
        ef.append("--compute-ao")
        if args.ao_samples != 16:
            ef.append(f"--ao-samples={args.ao_samples}")
        if args.ao_intensity != 1.0:
            ef.append(f"--ao-intensity={args.ao_intensity}")
        if args.ao_bias != 0.025:
            ef.append(f"--ao-bias={args.ao_bias}")

    if args.hair_demo:
        ef.append("--hair-demo")
        if args.hair_width != 0.02:
            ef.append(f"--hair-width={args.hair_width}")
        if args.hair_mat != 1:
            ef.append(f"--hair-mat={args.hair_mat}")

    # QMC / adaptive SPP forwarding
    if args.qmc_mode in (0, 1):
        ef.append(f"--qmc-mode={args.qmc_mode}")
    if args.spp_limit is not None:
        ef.append(f"--spp-limit={int(args.spp_limit)}")

    if ef:
        cmd.append("--")
        cmd.extend(ef)

    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(repo_root))
    if r.returncode != 0:
        raise SystemExit(r.returncode)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
