#!/usr/bin/env python3
"""
Generate golden reference images for P9 regression testing

Creates 12 golden images with different BRDF × shadow × GI combinations
at 1280×920 resolution for SSIM validation (epsilon ≥ 0.98).

This script launches the interactive viewer example headlessly, issues
commands over stdin to configure GI and scene, captures a snapshot PNG,
and quits.

Usage:
    python scripts/generate_golden_images.py \
        [--output-dir tests/golden] [--overwrite] \
        [--filter SUBSTR] [--ibl path.hdr] [--obj path.obj]
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path

# Add parent directory to path for forge3d import
sys.path.insert(0, str(Path(__file__).parent.parent))

# Python module is not required; we drive the Rust viewer example.


GOLDEN_CONFIGS = [
    # (name, width, height, brdf, shadows, gi_modes, ibl_enabled, description)
    (
        "lambert_hard_nogi",
        1280,
        920,
        "lambert",
        "hard",
        [],
        False,
        "Lambert diffuse with hard shadows, no GI",
    ),
    (
        "phong_pcf_nogi",
        1280,
        920,
        "phong",
        "pcf",
        [],
        False,
        "Phong specular with PCF soft shadows",
    ),
    (
        "ggx_pcf_ibl",
        1280,
        920,
        "cooktorrance-ggx",
        "pcf",
        ["ibl"],
        True,
        "Cook-Torrance GGX with PCF and IBL",
    ),
    (
        "disney_pcss_ibl_ssao",
        1280,
        920,
        "disney-principled",
        "pcss",
        ["ibl", "ssao"],
        True,
        "Disney Principled with PCSS, IBL, and SSAO",
    ),
    (
        "orennayar_vsm_nogi",
        1280,
        920,
        "oren-nayar",
        "vsm",
        [],
        False,
        "Oren-Nayar rough diffuse with VSM shadows",
    ),
    (
        "toon_hard_nogi",
        1280,
        920,
        "toon",
        "hard",
        [],
        False,
        "Toon shading with hard shadows (stylized)",
    ),
    (
        "ashikhmin_pcss_ibl",
        1280,
        920,
        "ashikhmin-shirley",
        "pcss",
        ["ibl"],
        True,
        "Ashikhmin-Shirley anisotropic with PCSS and IBL",
    ),
    (
        "ward_evsm_nogi",
        1280,
        920,
        "ward",
        "evsm",
        [],
        False,
        "Ward anisotropic with EVSM shadows",
    ),
    (
        "blinnphong_msm_nogi",
        1280,
        920,
        "blinn-phong",
        "msm",
        [],
        False,
        "Blinn-Phong with MSM shadows",
    ),
    (
        "ggx_csm_ibl_gtao",
        1280,
        920,
        "cooktorrance-ggx",
        "csm",
        ["ibl", "gtao"],
        True,
        "GGX with cascaded shadow maps, IBL, and GTAO",
    ),
    (
        "disney_pcf_ibl_ssgi",
        1280,
        920,
        "disney-principled",
        "pcf",
        ["ibl", "ssgi"],
        True,
        "Disney Principled with PCF, IBL, and SSGI",
    ),
    (
        "ggx_pcss_ibl_ssr",
        1280,
        920,
        "cooktorrance-ggx",
        "pcss",
        ["ibl", "ssr"],
        True,
        "GGX with PCSS, IBL, and screen-space reflections",
    ),
]


def _camera_for_config(name: str):
    """Return (fov_deg, eye(x,y,z), target(x,y,z), up(x,y,z)) for a config name.

    Defaults chosen to frame assets/cornell_box.obj nicely.
    """
    # Default camera
    fov = 45.0
    eye = [0.0, 1.0, -3.0]
    target = [0.0, 1.0, 0.0]
    up = [0.0, 1.0, 0.0]

    # Slightly wider for SSR to catch reflections
    if name.endswith("_ssr"):
        fov = 55.0

    return fov, eye, target, up


def _viewer_args_for_config(name, width: int, height: int, gi_modes, ibl_enabled, ibl_path: Path|None, obj_path: Path|None, snapshot: Path):
    args: list[str] = []
    # Ensure window size matches golden target
    args += ["--size", f"{int(width)}x{int(height)}"]
    # Load mesh if available
    if obj_path and obj_path.exists():
        args += ["--obj", str(obj_path)]
    # IBL setup
    if ibl_enabled and ibl_path and ibl_path.exists():
        args += ["--ibl", str(ibl_path)]
    # Enable GI modes
    for mode in gi_modes:
        if mode in ("ssao", "ssgi", "ssr"):
            args += ["--gi", f"{mode}:on"]
        elif mode == "gtao":
            args += ["--ssao-technique", "gtao", "--gi", "ssao:on"]
        elif mode == "ibl":
            pass
    # Choose visualization:
    # - SSGI/SSR: GI debug view
    # - IBL-only: Lit viz (simple shading using albedo+normal with IBL)
    # - Otherwise: Material; if AO/GTAO present, composite AO for visibility
    has_ssgi_or_ssr = any(m in ("ssgi", "ssr") for m in gi_modes)
    has_ao = any(m in ("ssao", "gtao") for m in gi_modes)
    has_ibl = any(m == "ibl" for m in gi_modes)
    has_ibl_only = has_ibl and not has_ssgi_or_ssr and not has_ao
    if has_ssgi_or_ssr:
        args += ["--viz", "gi"]
    elif has_ibl_only:
        args += ["--viz", "lit", "--lit-sun", "1.0", "--lit-ibl", "0.6"]
    else:
        args += ["--viz", "material"]
        if has_ao:
            # Enable AO composite and set a default multiplier
            args += ["--ssao-composite", "on", "--ssao-mul", "0.8"]
    # Camera & FOV for consistent framing
    fov, eye, target, up = _camera_for_config(name)
    args += ["--fov", f"{fov}"]
    args += ["--cam-lookat", f"{eye[0]},{eye[1]},{eye[2]},{target[0]},{target[1]},{target[2]},{up[0]},{up[1]},{up[2]}"]
    # Take snapshot
    args += ["--snapshot", str(snapshot)]
    return args


def _run_viewer_and_snapshot(example_args: list[str], cwd: Path, snapshot_path: Path, timeout_sec: float = 45.0) -> None:
    env = os.environ.copy()
    env["FORGE3D_AUTO_SNAPSHOT_PATH"] = str(snapshot_path)
    proc = subprocess.Popen(
        ["cargo", "run", "--quiet", "--release", "--example", "interactive_viewer", "--", *example_args],
        cwd=str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    # Asynchronously drain stdout to avoid deadlocks and detect readiness/snapshot
    saved_snapshot = {"done": False}
    ready = {"seen": False}

    def _reader():
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                s = line.rstrip()
                print(f"[viewer] {s}")
                if "Interactive Viewer" in s or s.startswith("Controls:"):
                    ready["seen"] = True
                if s.startswith("Saved snapshot to "):
                    # Matches viewer log in render()
                    saved_snapshot["done"] = True
        except Exception:
            pass

    import threading, sys
    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    # If running interactively, forward our terminal stdin to the viewer's stdin
    forwarder_stop = {"stop": False}
    def _forward_stdin():
        try:
            if not sys.stdin.isatty():
                return
            assert proc.stdin is not None
            for line in sys.stdin:
                if forwarder_stop["stop"]:
                    break
                try:
                    proc.stdin.write(line)
                    proc.stdin.flush()
                    try:
                        sys.stdout.write(f"[forward] {line}")
                        sys.stdout.flush()
                    except Exception:
                        pass
                except Exception:
                    break
        except Exception:
            pass

    tf = threading.Thread(target=_forward_stdin, daemon=True)
    tf.start()
    try:
        # Wait up to a few seconds for startup banner to ensure input thread alive
        t0 = time.time()
        while not ready["seen"] and (time.time() - t0) < 5.0:
            time.sleep(0.1)
        # Extra settle time for first frame
        time.sleep(0.5)
        # Proactively request snapshot via stdin as a fallback
        try:
            assert proc.stdin is not None
            proc.stdin.write(f"snapshot {snapshot_path}\n")
            proc.stdin.flush()
        except Exception:
            pass
        # After initial-commands snapshot, wait for file or log
        t0 = time.time()
        while not (snapshot_path.exists() or saved_snapshot["done"]):
            if time.time() - t0 > 120.0:
                raise RuntimeError(f"Snapshot not created within timeout: {snapshot_path}")
            time.sleep(0.2)
        # Request quit and wait
        try:
            assert proc.stdin is not None
            proc.stdin.write(":quit\n")
            proc.stdin.flush()
        except Exception:
            pass
        try:
            forwarder_stop["stop"] = True
        except Exception:
            pass
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        # If we have our snapshot already, force-kill and continue. Otherwise, fail.
        try:
            if snapshot_path.exists() or saved_snapshot["done"]:
                print("  [generator] Viewer did not exit in time; killing after snapshot...")
                proc.kill()
            else:
                proc.kill()
                raise RuntimeError("Viewer did not exit in time")
        except Exception:
            # Best-effort cleanup; propagate failure only if no snapshot
            if not snapshot_path.exists():
                raise RuntimeError("Viewer did not exit in time")
    finally:
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass


def generate_golden_image(
    name: str,
    width: int,
    height: int,
    brdf: str,
    shadows: str,
    gi_modes: list,
    ibl_enabled: bool,
    description: str,
    output_dir: Path,
    *,
    ibl_path: Path|None,
    obj_path: Path|None,
) -> None:
    """Generate a single golden reference image using the interactive viewer."""

    print(f"\nGenerating: {name}")
    print(f"  Resolution: {width}×{height}")
    print(f"  BRDF: {brdf}")
    print(f"  Shadows: {shadows}")
    print(f"  GI: {gi_modes if gi_modes else 'none'}")
    print(f"  Description: {description}")

    output_path = output_dir / f"{name}.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build commands for viewer
    project_root = Path(__file__).parent.parent
    args_list = _viewer_args_for_config(name, width, height, gi_modes, ibl_enabled, ibl_path, obj_path, output_path)

    # Launch viewer and send commands
    _run_viewer_and_snapshot(args_list, project_root, output_path)

    # Ensure file exists
    t0 = time.time()
    while not output_path.exists():
        if time.time() - t0 > 10.0:
            raise RuntimeError(f"Snapshot not created: {output_path}")
        time.sleep(0.1)
    print(f"  ✓ Snapshot written: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden reference images for P9 testing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/golden"),
        help="Output directory for golden images (default: tests/golden)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing golden images",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Substring filter to select a subset of configs by name",
    )
    parser.add_argument(
        "--ibl",
        type=Path,
        default=None,
        help="Path to environment HDR/EXR to use for IBL scenes",
    )
    parser.add_argument(
        "--obj",
        type=Path,
        default=Path("assets/bunny.obj"),
        help="Path to OBJ mesh to load (default: assets/bunny.obj)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Golden Image Generator (P9)")
    print("=" * 70)
    print(f"Output directory: {args.output_dir}")
    # Filter configs if requested
    selected = [c for c in GOLDEN_CONFIGS if (args.filter is None or args.filter in c[0])]
    print("Number of images: {}".format(len(selected)))
    print(f"Resolution: 1280×920 (all images)")
    print("=" * 70)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate each golden image
    success_count = 0
    for config in selected:
        name, width, height, brdf, shadows, gi_modes, ibl_enabled, description = config

        output_path = args.output_dir / f"{name}.png"

        # Skip if exists and not overwriting
        if output_path.exists() and not args.overwrite:
            print(f"\nSkipping: {name} (already exists)")
            continue

        try:
            generate_golden_image(
                name,
                width,
                height,
                brdf,
                shadows,
                gi_modes,
                ibl_enabled,
                description,
                args.output_dir,
                ibl_path=args.ibl,
                obj_path=args.obj if args.obj is not None else None,
            )
            success_count += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Generated {success_count}/{len(selected)} golden images")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    if success_count < len(selected):
        print("\n⚠️  Some images failed to generate")
        sys.exit(1)


if __name__ == "__main__":
    main()
