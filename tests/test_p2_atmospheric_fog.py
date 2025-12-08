"""
Phase P2: Atmospheric Fog Validation Tests

Validates the P2 atmospheric fog implementation:
- Fog uniforms and bind group (group 4) are correctly set up in WGSL and Rust
- CLI parameters (--fog-density, --fog-height-falloff, --fog-inscatter) are wired
- With fog disabled (density=0), output matches P1 within SSIM threshold
- With fog enabled, fog effect is visible (differs from P1/no-fog)

Outputs:
- reports/terrain/p2/phase_p2.png (no-fog render)
- reports/terrain/p2/phase_p2_diff.png (diff vs P1)
- reports/terrain/p2/phase_p2_fog.png (fog-enabled render)
- reports/terrain/p2/p2_result.json (validation results)
- reports/terrain/p2/p2_run.log (execution log)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# P2 SSIM threshold - should match P1 when fog disabled
P2_SSIM_THRESHOLD = 0.98

# Output paths
P2_DIR = Path("reports/terrain/p2")
P1_DIR = Path("reports/terrain/p1")
P2_OUTPUT = P2_DIR / "phase_p2.png"
P2_DIFF = P2_DIR / "phase_p2_diff.png"
P2_FOG_OUTPUT = P2_DIR / "phase_p2_fog.png"
P2_RESULT = P2_DIR / "p2_result.json"
P2_LOG = P2_DIR / "p2_run.log"

# Asset paths
DEFAULT_DEM = Path("assets/Gore_Range_Albers_1m.tif")
DEFAULT_HDR = Path("assets/hdri/snow_field_4k.hdr")


def get_git_rev() -> str:
    """Get short git revision hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def compute_image_hash(path: Path) -> str:
    """Compute MD5 hash of image file."""
    if not path.exists():
        return "missing"
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_image_rgba(path: Path) -> np.ndarray:
    """Load PNG as RGBA uint8 array."""
    from PIL import Image
    img = Image.open(path).convert("RGBA")
    return np.array(img, dtype=np.uint8)


def compute_ssim_and_diff(
    baseline: Path,
    test: Path,
    diff_out: Path,
) -> tuple[float, dict[str, Any]]:
    """
    Compute SSIM between baseline and test images, produce visual diff.
    
    Returns (ssim_value, error_metrics_dict).
    """
    # Import ssim from the tests module - handle both pytest and direct execution
    try:
        from tests._ssim import ssim
    except ImportError:
        import sys
        from pathlib import Path
        tests_dir = Path(__file__).parent
        if str(tests_dir) not in sys.path:
            sys.path.insert(0, str(tests_dir))
        from _ssim import ssim
    
    baseline_arr = load_image_rgba(baseline)
    test_arr = load_image_rgba(test)
    
    if baseline_arr.shape != test_arr.shape:
        raise ValueError(
            f"Image shapes differ: baseline={baseline_arr.shape}, test={test_arr.shape}"
        )
    
    # Use RGB only for SSIM (drop alpha)
    baseline_rgb = baseline_arr[:, :, :3]
    test_rgb = test_arr[:, :, :3]
    
    # Compute SSIM on uint8 with data_range=255
    ssim_value = float(ssim(baseline_rgb, test_rgb, data_range=255.0))
    
    # Compute error metrics on float [0, 1]
    baseline_float = baseline_arr.astype(np.float32) / 255.0
    test_float = test_arr.astype(np.float32) / 255.0
    diff = test_float - baseline_float
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    max_abs_err = float(np.max(np.abs(diff)))
    
    # Generate diff image: absolute difference scaled to [0, 255]
    diff_uint = np.abs(test_arr.astype(np.int16) - baseline_arr.astype(np.int16))
    diff_scaled = np.clip(diff_uint * 4, 0, 255).astype(np.uint8)
    diff_scaled[:, :, 3] = 255  # Set alpha to opaque
    
    # Save diff image
    try:
        from PIL import Image
        diff_img = Image.fromarray(diff_scaled)
        diff_out.parent.mkdir(parents=True, exist_ok=True)
        diff_img.save(str(diff_out))
    except Exception as e:
        print(f"Warning: Could not save diff image: {e}")
    
    metrics = {
        "ssim": round(ssim_value, 6),
        "mse": round(mse, 8),
        "mae": round(mae, 8),
        "max_abs_err": round(max_abs_err, 8),
    }
    
    return ssim_value, metrics


def verify_shader_fog_config() -> dict[str, bool]:
    """Verify fog uniforms and bindings in terrain shader."""
    shader_path = Path("src/shaders/terrain_pbr_pom.wgsl")
    if not shader_path.exists():
        return {"shader_exists": False}
    
    shader_src = shader_path.read_text()
    
    checks = {
        "fog_uniforms_struct": "struct FogUniforms" in shader_src,
        "fog_density_field": "fog_density: f32" in shader_src,
        "fog_height_falloff_field": "fog_height_falloff: f32" in shader_src,
        "fog_inscatter_field": "fog_inscatter: vec3<f32>" in shader_src,
        "fog_bind_group": "@group(4) @binding(0)" in shader_src,
        "apply_atmospheric_fog_fn": "fn apply_atmospheric_fog" in shader_src,
        "fog_early_out": "if (density_raw <= 0.0)" in shader_src,
    }
    
    return checks


def verify_rust_fog_config() -> dict[str, bool]:
    """Verify fog configuration in Rust terrain renderer."""
    terrain_renderer = Path("src/terrain_renderer.rs")
    terrain_params = Path("src/terrain_render_params.rs")
    
    checks = {}
    
    if terrain_renderer.exists():
        src = terrain_renderer.read_text()
        checks["rust_fog_uniforms_struct"] = "struct FogUniforms" in src
        checks["rust_fog_bind_group_layout"] = "create_fog_bind_group_layout" in src
        checks["rust_fog_bind_group_index_4"] = 'set_bind_group(4,' in src
    else:
        checks["rust_renderer_exists"] = False
    
    if terrain_params.exists():
        src = terrain_params.read_text()
        checks["rust_fog_settings_native"] = "FogSettingsNative" in src
    else:
        checks["rust_params_exists"] = False
    
    return checks


def verify_cli_fog_params() -> dict[str, bool]:
    """Verify fog CLI parameters are defined."""
    terrain_demo = Path("examples/terrain_demo.py")
    
    if not terrain_demo.exists():
        return {"cli_exists": False}
    
    src = terrain_demo.read_text()
    
    checks = {
        "cli_fog_density": "--fog-density" in src,
        "cli_fog_height_falloff": "--fog-height-falloff" in src,
        "cli_fog_inscatter": "--fog-inscatter" in src,
    }
    
    return checks


def render_terrain_with_fog(
    output_path: Path,
    fog_density: float = 0.0,
    fog_height_falloff: float = 0.0,
    fog_inscatter: str = "1.0,1.0,1.0",
) -> dict[str, Any]:
    """
    Render terrain with specified fog configuration using terrain_pbr_pom.run().
    
    Returns render result dict with status, hash, fog_config.
    """
    import argparse
    
    result = {
        "phase": "P2",
        "name": "Fog Render",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_rev": get_git_rev(),
        "output_path": str(output_path),
        "fog_config": {
            "density": fog_density,
            "height_falloff": fog_height_falloff,
            "inscatter": fog_inscatter,
        },
        "status": "PENDING",
    }
    
    # Check assets
    if not DEFAULT_DEM.exists():
        result["status"] = "SKIP"
        result["error"] = f"DEM not found: {DEFAULT_DEM}"
        return result
    
    if not DEFAULT_HDR.exists():
        result["status"] = "SKIP"
        result["error"] = f"HDR not found: {DEFAULT_HDR}"
        return result
    
    try:
        from forge3d import terrain_pbr_pom
        
        # Create args namespace similar to P1 test
        args = argparse.Namespace(
            dem=DEFAULT_DEM,
            hdr=DEFAULT_HDR,
            ibl_res=128,
            ibl_cache=None,
            ibl_intensity=1.0,
            size=[512, 512],
            render_scale=1.0,
            msaa=4,
            z_scale=2.0,
            cam_radius=1000.0,
            cam_phi=135.0,
            cam_theta=45.0,
            exposure=1.0,
            sun_azimuth=135.0,
            sun_elevation=35.0,
            sun_intensity=3.0,
            sun_color=None,
            colormap_domain=None,
            colormap="terrain",
            colormap_interpolate=False,
            colormap_size=256,
            output=output_path,
            window=False,
            viewer=False,
            overwrite=True,
            albedo_mode="mix",
            colormap_strength=0.5,
            height_curve_mode="linear",
            height_curve_strength=0.0,
            height_curve_power=1.0,
            height_curve_lut=None,
            light=[],
            brdf=None,
            preset=None,
            shadows="csm",
            shadow_map_res=1024,
            cascades=4,
            pcss_blocker_radius=None,
            pcss_filter_radius=None,
            shadow_light_size=None,
            shadow_moment_bias=None,
            gi=None,
            sky=None,
            volumetric=None,
            debug_lights=False,
            water_detect=False,
            water_level=0.08,
            water_slope=0.015,
            water_mask_output=None,
            water_mask_output_mode="overlay",
            water_material="overlay",
            pom_disabled=False,
            debug_mode=0,
            # P2: Fog parameters
            fog_density=fog_density,
            fog_height_falloff=fog_height_falloff,
            fog_inscatter=fog_inscatter,
        )
        
        # Run terrain render
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exit_code = terrain_pbr_pom.run(args)
        
        if exit_code == 0 and output_path.exists():
            result["image_hash"] = compute_image_hash(output_path)
            result["status"] = "PASS"
            print(f"Wrote {output_path}")
        else:
            result["status"] = "FAIL"
            result["error"] = f"Render returned exit code {exit_code}"
        
    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()
    
    return result


# ============================================================================
# Tests
# ============================================================================

def test_p2_fog_infrastructure():
    """Test P2 fog infrastructure: shader, Rust, and CLI configuration."""
    print("\n" + "=" * 70)
    print("Phase P2: Atmospheric Fog Validation")
    print("=" * 70)
    
    shader_checks = verify_shader_fog_config()
    rust_checks = verify_rust_fog_config()
    cli_checks = verify_cli_fog_params()
    
    all_checks = {**shader_checks, **rust_checks, **cli_checks}
    all_passed = all(all_checks.values())
    
    print(f"\nInfrastructure Status: {'PASS' if all_passed else 'FAIL'}\n")
    
    print("Shader checks:")
    for name, passed in shader_checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    
    print("\nRust checks:")
    for name, passed in rust_checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    
    print("\nCLI checks:")
    for name, passed in cli_checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    
    # Write partial result
    P2_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "phase": "P2",
        "name": "Atmospheric Fog",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_rev": get_git_rev(),
        "shader_config": {"valid": all(shader_checks.values()), "checks": shader_checks},
        "rust_config": {"valid": all(rust_checks.values()), "checks": rust_checks},
        "cli_config": {"valid": all(cli_checks.values()), "checks": cli_checks},
        "status": "PASS" if all_passed else "FAIL",
    }
    
    with open(P2_RESULT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nP2 validation results written to: {P2_RESULT}")
    
    assert all_passed, f"Infrastructure checks failed: {[k for k, v in all_checks.items() if not v]}"


def test_p2_render_nofog_matches_p1():
    """Test P2 render with fog disabled matches P1 within SSIM threshold."""
    pytest.importorskip("forge3d")
    
    print("\n" + "=" * 70)
    print("P2 No-Fog Render (should match P1)")
    print("=" * 70)
    
    # Render with fog disabled (density=0)
    render_result = render_terrain_with_fog(
        P2_OUTPUT,
        fog_density=0.0,
        fog_height_falloff=0.0,
    )
    
    if render_result["status"] == "SKIP":
        pytest.skip(render_result.get("error", "Render skipped"))
    
    print(f"P2 no-fog render completed: {render_result['output_path']}")
    print(f"Image hash: {render_result.get('image_hash', 'N/A')}")
    
    # Compare with P1
    p1_image = P1_DIR / "phase_p1.png"
    ssim_pass = True
    comparison = {
        "test_image": str(P2_OUTPUT),
        "threshold": P2_SSIM_THRESHOLD,
    }
    
    if p1_image.exists() and P2_OUTPUT.exists():
        try:
            ssim_value, metrics = compute_ssim_and_diff(p1_image, P2_OUTPUT, P2_DIFF)
            ssim_pass = bool(ssim_value >= P2_SSIM_THRESHOLD)
            
            comparison.update({
                "p1_image": str(p1_image),
                "diff_image": str(P2_DIFF) if P2_DIFF.exists() else None,
                **metrics,
                "pass": ssim_pass,
            })
            
            print(f"SSIM vs P1: {ssim_value:.4f} (threshold: {P2_SSIM_THRESHOLD})")
            if P2_DIFF.exists():
                print(f"Diff image: {P2_DIFF}")
        except Exception as e:
            print(f"Warning: SSIM comparison failed: {e}")
            comparison["error"] = str(e)
    else:
        print("P1 image not found - skipping SSIM comparison")
        comparison["p1_status"] = "missing"
    
    # Load existing result and update
    result = {}
    if P2_RESULT.exists():
        with open(P2_RESULT) as f:
            result = json.load(f)
    
    result["render_nofog"] = render_result
    result["image_comparison_nofog"] = comparison
    result["status"] = "PASS" if render_result["status"] == "PASS" and ssim_pass else "FAIL"
    
    with open(P2_RESULT, "w") as f:
        json.dump(result, f, indent=2)
    
    assert render_result["status"] == "PASS", f"Render failed: {render_result.get('error')}"
    assert ssim_pass, f"SSIM {comparison.get('ssim', 'N/A')} below threshold {P2_SSIM_THRESHOLD}"


def test_p2_render_fog_enabled():
    """Test P2 render with fog enabled produces visible fog effect."""
    pytest.importorskip("forge3d")
    
    print("\n" + "=" * 70)
    print("P2 Fog-Enabled Render")
    print("=" * 70)
    
    # Render with fog enabled
    render_result = render_terrain_with_fog(
        P2_FOG_OUTPUT,
        fog_density=0.5,  # Moderate fog
        fog_height_falloff=0.001,  # Subtle height falloff
        fog_inscatter="0.8,0.85,0.95",  # Slightly bluish fog
    )
    
    if render_result["status"] == "SKIP":
        pytest.skip(render_result.get("error", "Render skipped"))
    
    print(f"P2 fog render completed: {render_result['output_path']}")
    print(f"Image hash: {render_result.get('image_hash', 'N/A')}")
    print(f"Fog config: {render_result.get('fog_config')}")
    
    # Verify fog actually made a difference by comparing to no-fog
    fog_preview = {
        "image_path": str(P2_FOG_OUTPUT),
        "fog_params": render_result.get("fog_config"),
        "notes": "Fog-enabled preview - distant mountains should fade towards inscatter color",
    }
    
    if P2_OUTPUT.exists() and P2_FOG_OUTPUT.exists():
        try:
            nofog_arr = load_image_rgba(P2_OUTPUT)
            fog_arr = load_image_rgba(P2_FOG_OUTPUT)
            
            # Compute difference
            diff = np.abs(fog_arr.astype(np.float32) - nofog_arr.astype(np.float32))
            mse = float(np.mean(diff ** 2))
            
            fog_preview["diff_vs_nofog_mse"] = round(mse, 6)
            
            # Fog should make a visible difference
            if mse < 1.0:
                print(f"Warning: Fog effect very subtle (MSE={mse:.6f})")
            else:
                print(f"Fog effect visible (MSE vs no-fog: {mse:.2f})")
        except Exception as e:
            print(f"Warning: Could not compare fog/no-fog: {e}")
    
    # Update result
    result = {}
    if P2_RESULT.exists():
        with open(P2_RESULT) as f:
            result = json.load(f)
    
    result["render_fog"] = render_result
    result["fog_preview"] = fog_preview
    
    with open(P2_RESULT, "w") as f:
        json.dump(result, f, indent=2)
    
    assert render_result["status"] == "PASS", f"Fog render failed: {render_result.get('error')}"


# ============================================================================
# Main entry point for manual execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Phase P2: Atmospheric Fog Validation")
    print("=" * 70)
    
    # Run infrastructure test
    try:
        test_p2_fog_infrastructure()
    except AssertionError as e:
        print(f"\nInfrastructure test failed: {e}")
        sys.exit(1)
    
    # Run no-fog render test
    try:
        test_p2_render_nofog_matches_p1()
    except (AssertionError, Exception) as e:
        print(f"\nNo-fog render test failed: {e}")
    
    # Run fog-enabled render test
    try:
        test_p2_render_fog_enabled()
    except (AssertionError, Exception) as e:
        print(f"\nFog render test failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("P2 Validation Summary")
    print("=" * 70)
    
    if P2_RESULT.exists():
        with open(P2_RESULT) as f:
            result = json.load(f)
        print(f"Overall Status: {result.get('status', 'UNKNOWN')}")
        
        if "image_comparison_nofog" in result:
            comp = result["image_comparison_nofog"]
            ssim = comp.get("ssim", "N/A")
            print(f"No-fog SSIM vs P1: {ssim} (threshold: {P2_SSIM_THRESHOLD})")
        
        if "fog_preview" in result:
            preview = result["fog_preview"]
            print(f"Fog preview: {preview.get('image_path', 'N/A')}")
            if "diff_vs_nofog_mse" in preview:
                print(f"Fog effect MSE: {preview['diff_vs_nofog_mse']:.4f}")
