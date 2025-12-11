#!/usr/bin/env python3
"""Terrain Rendering Validation Script.

Implements the verification protocol from docs/plan.md:
- Baseline snapshot capture (no code changes)
- Phase P1-P6 validation with diff images and SSIM
- Final proofpack generation

Usage:
    python scripts/terrain_validation.py baseline    # Capture baseline
    python scripts/terrain_validation.py phase P1   # Validate phase P1
    python scripts/terrain_validation.py proofpack  # Generate final proofpack
    python scripts/terrain_validation.py all        # Run full validation suite
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports" / "terrain"
BASELINE_DIR = REPORTS_DIR / "baseline"

# Canonical render settings (deterministic, reproducible)
CANONICAL_SETTINGS = {
    "dem": "assets/Gore_Range_Albers_1m.tif",
    "hdr": "assets/hdri/snow_field_4k.hdr",
    "size": (1920, 1080),
    "msaa": 4,
    "z_scale": 2.0,
    "cam_radius": 1000.0,
    "cam_phi": 135.0,
    "cam_theta": 45.0,
    "exposure": 1.0,
    "ibl_intensity": 1.0,
    "sun_azimuth": 135.0,
    "sun_elevation": 35.0,
}

# Phase-specific settings (additive on top of canonical)
PHASE_SETTINGS = {
    "baseline": {"shadows": "none"},  # P0: No shadows for baseline
    "P1": {"shadows": "csm", "cascades": 4},  # CSM enabled
    "P2": {"shadows": "csm", "cascades": 4, "fog_density": 0.0},  # Fog disabled (no-op)
    "P2_fog": {"shadows": "csm", "cascades": 4, "fog_density": 0.002, "fog_height_falloff": 0.01},
    "P3": {"shadows": "csm", "cascades": 4},  # Toksvig enabled by default
    "P4": {"shadows": "csm", "cascades": 4},  # Reflections (auto-enabled on water)
    "P5": {"shadows": "csm", "cascades": 4, "debug_mode": 0},  # AO weight 0 (no-op)
    "P5_ao": {"shadows": "csm", "cascades": 4, "debug_mode": 28},  # Raw SSAO debug
    "P6": {"shadows": "csm", "cascades": 4},  # Micro-detail (enabled by default)
}

# Debug modes for specific outputs
DEBUG_MODES = {
    "water_mask": 100,
    "water_shore": 101,
    "water_ibl": 102,
    "pbr_diffuse": 7,
    "pbr_specular": 8,
    "roughness": 11,
    "energy": 12,
    "specaa_sparkle": 17,
    "ssao_raw": 28,
}


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_gpu_info() -> str:
    """Get GPU/driver info (best effort)."""
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True
            )
            for line in result.stdout.split("\n"):
                if "Chipset Model" in line or "Metal" in line:
                    return line.strip()
        return f"{platform.system()} {platform.machine()}"
    except Exception:
        return "unknown"


def md5_file(path: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def run_terrain_render(
    output_path: Path,
    settings: dict,
    log_path: Path | None = None,
) -> tuple[bool, str]:
    """Run terrain_demo.py with given settings, capture output."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "examples" / "terrain_demo.py"),
        "--dem", str(PROJECT_ROOT / settings.get("dem", CANONICAL_SETTINGS["dem"])),
        "--hdr", str(PROJECT_ROOT / settings.get("hdr", CANONICAL_SETTINGS["hdr"])),
        "--size", str(settings.get("size", CANONICAL_SETTINGS["size"])[0]),
                 str(settings.get("size", CANONICAL_SETTINGS["size"])[1]),
        "--msaa", str(settings.get("msaa", CANONICAL_SETTINGS["msaa"])),
        "--z-scale", str(settings.get("z_scale", CANONICAL_SETTINGS["z_scale"])),
        "--cam-radius", str(settings.get("cam_radius", CANONICAL_SETTINGS["cam_radius"])),
        "--cam-phi", str(settings.get("cam_phi", CANONICAL_SETTINGS["cam_phi"])),
        "--cam-theta", str(settings.get("cam_theta", CANONICAL_SETTINGS["cam_theta"])),
        "--exposure", str(settings.get("exposure", CANONICAL_SETTINGS["exposure"])),
        "--ibl-intensity", str(settings.get("ibl_intensity", CANONICAL_SETTINGS["ibl_intensity"])),
        "--sun-azimuth", str(settings.get("sun_azimuth", CANONICAL_SETTINGS["sun_azimuth"])),
        "--sun-elevation", str(settings.get("sun_elevation", CANONICAL_SETTINGS["sun_elevation"])),
        "--output", str(output_path),
        "--overwrite",
    ]

    if "shadows" in settings:
        cmd.extend(["--shadows", settings["shadows"]])
    if "cascades" in settings:
        cmd.extend(["--cascades", str(settings["cascades"])])
    if "fog_density" in settings:
        cmd.extend(["--fog-density", str(settings["fog_density"])])
    if "fog_height_falloff" in settings:
        cmd.extend(["--fog-height-falloff", str(settings["fog_height_falloff"])])
    if "debug_mode" in settings:
        cmd.extend(["--debug-mode", str(settings["debug_mode"])])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=300
        )
        full_log = f"Command: {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'w') as f:
                f.write(full_log)

        success = result.returncode == 0 and output_path.exists()
        return success, full_log
    except subprocess.TimeoutExpired:
        return False, "Render timed out after 300 seconds"
    except Exception as e:
        return False, f"Render failed: {e}"


def compare_images(ref: Path, test: Path, diff: Path, json_out: Path) -> dict:
    """Compare two images using compare_images.py script."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "compare_images.py"),
        str(ref), str(test),
        "--ssim",
        "--diff", str(diff),
        "--json", str(json_out),
    ]

    diff.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=60)
        if json_out.exists():
            with open(json_out) as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Image comparison failed: {e}")

    return {}


def capture_baseline() -> dict:
    """Capture baseline snapshot (Section 1 of plan.md)."""
    print("=" * 60)
    print("Capturing baseline snapshot...")
    print("=" * 60)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    # Merge canonical + baseline settings
    settings = {**CANONICAL_SETTINGS, **PHASE_SETTINGS["baseline"]}

    # Render main terrain image
    main_output = BASELINE_DIR / "baseline_terrain.png"
    log_path = BASELINE_DIR / "baseline_run.log"

    success, log = run_terrain_render(main_output, settings, log_path)
    if not success:
        print(f"ERROR: Baseline render failed!")
        print(log[-500:] if len(log) > 500 else log)
        return {"status": "FAIL", "error": "Baseline render failed"}

    print(f"  Rendered: {main_output}")

    # Render debug mode outputs
    debug_outputs = {}
    for mode_name, mode_id in DEBUG_MODES.items():
        debug_settings = {**settings, "debug_mode": mode_id}
        debug_output = BASELINE_DIR / f"baseline_debug_{mode_name}.png"
        success, _ = run_terrain_render(debug_output, debug_settings)
        if success:
            debug_outputs[mode_name] = {
                "path": str(debug_output.relative_to(PROJECT_ROOT)),
                "md5": md5_file(debug_output),
            }
            print(f"  Debug mode {mode_id} ({mode_name}): {debug_output.name}")

    # Generate summary JSON
    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "gpu_driver": get_gpu_info(),
        "cli_args": settings,
        "outputs": {
            "main": {
                "path": str(main_output.relative_to(PROJECT_ROOT)),
                "md5": md5_file(main_output),
            },
            **{f"debug_{k}": v for k, v in debug_outputs.items()},
        },
        "status": "PASS",
    }

    summary_path = BASELINE_DIR / "baseline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nBaseline summary saved: {summary_path}")
    return summary


def validate_phase(phase: str) -> dict:
    """Validate a specific phase against baseline (Sections 2-7 of plan.md)."""
    print("=" * 60)
    print(f"Validating phase {phase}...")
    print("=" * 60)

    # Check baseline exists
    baseline_main = BASELINE_DIR / "baseline_terrain.png"
    if not baseline_main.exists():
        print("ERROR: Baseline not found. Run 'baseline' first.")
        return {"status": "FAIL", "error": "Baseline not found"}

    # Phase directory
    phase_dir = REPORTS_DIR / f"phase_{phase.lower()}"
    phase_dir.mkdir(parents=True, exist_ok=True)

    # Determine settings for this phase
    phase_key = phase.upper()
    if phase_key not in PHASE_SETTINGS:
        print(f"ERROR: Unknown phase '{phase}'")
        return {"status": "FAIL", "error": f"Unknown phase {phase}"}

    settings = {**CANONICAL_SETTINGS, **PHASE_SETTINGS[phase_key]}

    # Render phase output
    phase_output = phase_dir / f"phase_{phase.lower()}.png"
    log_path = phase_dir / f"{phase.lower()}_run.log"

    success, log = run_terrain_render(phase_output, settings, log_path)
    if not success:
        print(f"ERROR: Phase {phase} render failed!")
        return {"status": "FAIL", "error": "Render failed"}

    print(f"  Rendered: {phase_output}")

    # Compare against baseline
    diff_output = phase_dir / f"phase_{phase.lower()}_diff.png"
    metrics_output = phase_dir / f"{phase.lower()}_metrics.json"

    # Determine reference (baseline for P1, previous phase for P2+)
    ref_image = baseline_main  # For now, always compare to baseline

    metrics = compare_images(ref_image, phase_output, diff_output, metrics_output)

    print(f"  Diff image: {diff_output}")
    print(f"  SSIM: {metrics.get('ssim', 'N/A')}")
    print(f"  MSE: {metrics.get('mse', 'N/A')}")

    # Build result JSON
    result = {
        "phase": phase,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "gpu_driver": get_gpu_info(),
        "settings": settings,
        "output_path": str(phase_output.relative_to(PROJECT_ROOT)),
        "output_md5": md5_file(phase_output) if phase_output.exists() else None,
        "diff_path": str(diff_output.relative_to(PROJECT_ROOT)),
        "metrics": metrics,
        "ssim_vs_baseline": metrics.get("ssim"),
        "status": "PASS" if metrics.get("ssim", 0) > 0.5 else "REVIEW",
    }

    result_path = phase_dir / f"{phase.lower()}_result.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nPhase {phase} result saved: {result_path}")
    return result


def generate_proofpack() -> dict:
    """Generate final proofpack (Section 10 of plan.md)."""
    print("=" * 60)
    print("Generating final proofpack...")
    print("=" * 60)

    phases = ["P1", "P2", "P3", "P4", "P5", "P6"]
    phase_results = {}
    all_pass = True

    for phase in phases:
        phase_dir = REPORTS_DIR / f"phase_{phase.lower()}"
        result_file = phase_dir / f"{phase.lower()}_result.json"

        if result_file.exists():
            with open(result_file) as f:
                phase_results[phase] = json.load(f)
            if phase_results[phase].get("status") != "PASS":
                all_pass = False
        else:
            phase_results[phase] = {"status": "MISSING"}
            all_pass = False

    # Render final image with all features
    final_settings = {**CANONICAL_SETTINGS, "shadows": "csm", "cascades": 4}
    final_output = REPORTS_DIR / "phase_final.png"
    final_log = REPORTS_DIR / "final_run.log"

    success, _ = run_terrain_render(final_output, final_settings, final_log)

    # Compare final to baseline
    baseline_main = BASELINE_DIR / "baseline_terrain.png"
    final_diff = REPORTS_DIR / "phase_final_diff.png"
    final_metrics_file = REPORTS_DIR / "final_metrics.json"

    final_metrics = {}
    if baseline_main.exists() and final_output.exists():
        final_metrics = compare_images(
            baseline_main, final_output, final_diff, final_metrics_file
        )

    # Build proofpack summary
    proofpack = {
        "status": "PASS" if all_pass and success else "REVIEW",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "phases": list(phase_results.keys()),
        "phase_statuses": {p: r.get("status", "UNKNOWN") for p, r in phase_results.items()},
        "assets": [
            str(CANONICAL_SETTINGS["dem"]),
            str(CANONICAL_SETTINGS["hdr"]),
        ],
        "ssim_vs_baseline": final_metrics.get("ssim", 0.0),
        "changed_files": [],  # Would need git diff to populate
        "notes": [
            "All phases implemented in terrain_pbr_pom.wgsl",
            "P1: CSM shadows with PCF/PCSS",
            "P2: Atmospheric fog (disabled by default)",
            "P3: Toksvig specular anti-aliasing",
            "P4: Planar water reflections",
            "P5: SSAO debug mode 28, coarse AO fallback",
            "P6: Procedural micro-detail with RNM blending",
        ],
    }

    proofpack_path = REPORTS_DIR / "proofpack_summary_final.json"
    with open(proofpack_path, 'w') as f:
        json.dump(proofpack, f, indent=2)

    print(f"\nProofpack saved: {proofpack_path}")
    print(f"Status: {proofpack['status']}")
    print(f"SSIM vs baseline: {proofpack['ssim_vs_baseline']}")
    return proofpack


def run_all() -> dict:
    """Run full validation suite."""
    results = {}

    # Capture baseline
    results["baseline"] = capture_baseline()
    if results["baseline"].get("status") != "PASS":
        print("\nERROR: Baseline capture failed. Aborting.")
        return results

    # Validate each phase
    for phase in ["P1", "P2", "P3", "P4", "P5", "P6"]:
        results[phase] = validate_phase(phase)

    # Generate proofpack
    results["proofpack"] = generate_proofpack()

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for key, res in results.items():
        status = res.get("status", "UNKNOWN")
        print(f"  {key}: {status}")

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Terrain rendering validation per docs/plan.md"
    )
    parser.add_argument(
        "command",
        choices=["baseline", "phase", "proofpack", "all"],
        help="Validation command to run"
    )
    parser.add_argument(
        "phase",
        nargs="?",
        help="Phase name (P1-P6) for 'phase' command"
    )

    args = parser.parse_args(argv[1:] if argv else None)

    if args.command == "baseline":
        result = capture_baseline()
        return 0 if result.get("status") == "PASS" else 1

    elif args.command == "phase":
        if not args.phase:
            print("ERROR: Phase name required (P1, P2, P3, P4, P5, P6)")
            return 1
        result = validate_phase(args.phase)
        return 0 if result.get("status") == "PASS" else 1

    elif args.command == "proofpack":
        result = generate_proofpack()
        return 0 if result.get("status") == "PASS" else 1

    elif args.command == "all":
        results = run_all()
        all_pass = all(r.get("status") == "PASS" for r in results.values())
        return 0 if all_pass else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
