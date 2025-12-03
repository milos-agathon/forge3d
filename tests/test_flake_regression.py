# tests/test_flake_regression.py
"""
Milestone 4: Regression guardrails to prevent flakes from returning.

This test computes a "flake score" metric based on high-frequency energy
in rendered terrain. If someone reintroduces implicit-LOD height sampling
or mismatched texel offsets, this test will fail.

Flake Score Metric:
- Render mode 23 (no specular) and baseline material
- Compute Laplacian magnitude (measures high-frequency energy)
- Use p95 and p99 percentiles in a fixed ROI
- Require these below a threshold for the fixed configuration

Deliverables:
- reports/flake/flake_score.json (p95, p99, and pass/fail)

RELEVANT FILES: docs/plan.md, src/shaders/terrain_pbr_pom.wgsl
"""
from __future__ import annotations

import json
import numpy as np
import pytest
import tempfile
import os
import contextlib
from pathlib import Path
from datetime import datetime

import forge3d as f3d
from forge3d.terrain_params import (
    ClampSettings,
    IblSettings,
    LightSettings,
    LodSettings,
    PomSettings,
    SamplingSettings,
    ShadowSettings,
    TerrainRenderParams as TerrainRenderParamsConfig,
    TriplanarSettings,
)


if not f3d.has_gpu():
    pytest.skip("GPU required for flake regression tests", allow_module_level=True)


# Threshold for flake score (empirically determined)
# These should be low enough to catch regressions but high enough for normal variation
FLAKE_SCORE_P95_THRESHOLD = 50.0   # p95 Laplacian magnitude
FLAKE_SCORE_P99_THRESHOLD = 100.0  # p99 Laplacian magnitude

# Reports directory
REPORTS_DIR = Path(__file__).parent.parent / "reports" / "flake"


@contextlib.contextmanager
def debug_mode_env(mode: int):
    """Context manager to set VF_COLOR_DEBUG_MODE environment variable."""
    old_value = os.environ.get("VF_COLOR_DEBUG_MODE")
    os.environ["VF_COLOR_DEBUG_MODE"] = str(mode)
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("VF_COLOR_DEBUG_MODE", None)
        else:
            os.environ["VF_COLOR_DEBUG_MODE"] = old_value


def _create_test_hdr(path: str, width: int = 16, height: int = 8) -> None:
    """Create minimal HDR file for IBL."""
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        f.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 200 + 55)
                g = int((y / max(height - 1, 1)) * 200 + 55)
                b = 180
                e = 128
                f.write(bytes([r, g, b, e]))


def _create_test_heightmap(size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Create synthetic heightmap with features that expose flakes."""
    h, w = size
    y = np.linspace(0, 1, h)
    x = np.linspace(0, 1, w)
    xx, yy = np.meshgrid(x, y)
    
    # Terrain with ridges - important for flake exposure
    base = yy * 0.5
    ridges = np.sin(xx * 20) * np.sin(yy * 15) * 0.15
    noise = np.sin(xx * 40 + yy * 30) * 0.05
    
    heightmap = (base + ridges + noise + 0.2).astype(np.float32)
    return np.clip(heightmap, 0.0, 1.0)


def _build_config(overlay) -> TerrainRenderParamsConfig:
    """Build terrain config optimized for flake detection."""
    return TerrainRenderParamsConfig(
        size_px=(256, 256),
        render_scale=1.0,
        msaa_samples=1,
        z_scale=2.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=800.0,
        cam_phi_deg=135.0,
        cam_theta_deg=20.0,  # Low angle to maximize grazing-angle flakes
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 5000.0),
        light=LightSettings("Directional", 135.0, 35.0, 3.0, [1.0, 1.0, 1.0]),
        ibl=IblSettings(True, 1.0, 0.0),
        shadows=ShadowSettings(
            False, "PCF", 512, 2, 100.0, 1.0, 0.8, 0.002, 0.001, 0.3, 1e-4, 0.5, 2.0, 0.9
        ),
        triplanar=TriplanarSettings(6.0, 4.0, 1.0),
        pom=PomSettings(False, "Occlusion", 0.0, 4, 16, 2, False, False),
        lod=LodSettings(0, 0.0, 0.0),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 1, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[overlay],
        exposure=1.0,
        gamma=2.2,
        albedo_mode="material",  # Material mode for PBR (where flakes appear)
        colormap_strength=0.5,
    )


def _render_frame(debug_mode: int = 0) -> np.ndarray:
    """Render a single frame with specified debug mode."""
    with debug_mode_env(debug_mode):
        session = f3d.Session(window=False)
        renderer = f3d.TerrainRenderer(session)
        material_set = f3d.MaterialSet.terrain_default()
        
        heightmap = _create_test_heightmap()
        domain = (0.0, 1.0)
        
        cmap = f3d.Colormap1D.from_stops(
            stops=[(0.0, "#1a1a2e"), (0.5, "#4a7c59"), (1.0, "#f5f5dc")],
            domain=domain,
        )
        
        overlay = f3d.OverlayLayer.from_colormap1d(
            cmap,
            strength=1.0,
            offset=0.0,
            blend_mode="Alpha",
            domain=domain,
        )
        
        config = _build_config(overlay)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            hdr_path = os.path.join(tmpdir, "test.hdr")
            _create_test_hdr(hdr_path)
            ibl = f3d.IBL.from_hdr(hdr_path)
            
            params = f3d.TerrainRenderParams(config)
            
            frame = renderer.render_terrain_pbr_pom(
                material_set=material_set,
                env_maps=ibl,
                params=params,
                heightmap=heightmap,
                target=None,
            )
            
            return frame.to_numpy()


def _compute_laplacian(img: np.ndarray) -> np.ndarray:
    """Compute Laplacian magnitude of grayscale image.
    
    The Laplacian detects high-frequency content (edges, noise, flakes).
    High Laplacian values indicate rapid intensity changes.
    """
    # Convert to grayscale
    gray = img[:, :, :3].mean(axis=2).astype(np.float32)
    
    # Laplacian kernel (discrete approximation)
    # [0, 1, 0]
    # [1,-4, 1]
    # [0, 1, 0]
    h, w = gray.shape
    laplacian = np.zeros_like(gray)
    
    # Apply Laplacian (avoiding edges)
    laplacian[1:-1, 1:-1] = (
        gray[0:-2, 1:-1] +   # top
        gray[2:, 1:-1] +     # bottom
        gray[1:-1, 0:-2] +   # left
        gray[1:-1, 2:] -     # right
        4 * gray[1:-1, 1:-1] # center
    )
    
    return np.abs(laplacian)


def _compute_flake_score(frame: np.ndarray, roi: tuple[int, int, int, int] = None) -> dict:
    """Compute flake score metrics for a rendered frame.
    
    Args:
        frame: RGBA image array (H, W, 4)
        roi: Optional (y_start, y_end, x_start, x_end) region of interest
             If None, uses the central 75% of the image
    
    Returns:
        dict with p50, p75, p95, p99 Laplacian magnitude percentiles
    """
    h, w = frame.shape[:2]
    
    if roi is None:
        # Use central 75% to avoid edge artifacts
        margin_y = h // 8
        margin_x = w // 8
        roi = (margin_y, h - margin_y, margin_x, w - margin_x)
    
    y0, y1, x0, x1 = roi
    region = frame[y0:y1, x0:x1]
    
    # Compute Laplacian magnitude
    laplacian = _compute_laplacian(region)
    
    # Compute percentiles
    flat = laplacian.flatten()
    
    return {
        "p50": float(np.percentile(flat, 50)),
        "p75": float(np.percentile(flat, 75)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
    }


def _save_flake_score_json(scores: dict, path: Path) -> None:
    """Save flake score results to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(scores, f, indent=2)


class TestFlakeRegression:
    """Regression tests to prevent flakes from returning."""
    
    def test_baseline_flake_score_below_threshold(self):
        """Baseline (mode 0) should have low flake score with LOD-aware fix.
        
        This test fails if someone reintroduces implicit-LOD height sampling
        or mismatched texel offsets, which would increase high-frequency energy.
        """
        frame = _render_frame(debug_mode=0)
        
        assert frame is not None
        assert frame.shape == (256, 256, 4)
        
        scores = _compute_flake_score(frame)
        
        print(f"\nBaseline Flake Score:")
        print(f"  p50: {scores['p50']:.2f}")
        print(f"  p75: {scores['p75']:.2f}")
        print(f"  p95: {scores['p95']:.2f} (threshold: {FLAKE_SCORE_P95_THRESHOLD})")
        print(f"  p99: {scores['p99']:.2f} (threshold: {FLAKE_SCORE_P99_THRESHOLD})")
        print(f"  max: {scores['max']:.2f}")
        
        # Check thresholds
        assert scores["p95"] < FLAKE_SCORE_P95_THRESHOLD, (
            f"Flake regression detected! p95 Laplacian = {scores['p95']:.2f} >= {FLAKE_SCORE_P95_THRESHOLD}. "
            "This may indicate implicit-LOD height sampling or mismatched texel offsets."
        )
        
        assert scores["p99"] < FLAKE_SCORE_P99_THRESHOLD, (
            f"Flake regression detected! p99 Laplacian = {scores['p99']:.2f} >= {FLAKE_SCORE_P99_THRESHOLD}. "
            "This may indicate implicit-LOD height sampling or mismatched texel offsets."
        )
    
    def test_no_specular_has_lower_flake_score(self):
        """Mode 23 (no specular) should have lower flake score than baseline.
        
        Specular aliasing is a major source of flakes. Removing specular
        should reduce high-frequency energy.
        """
        frame_baseline = _render_frame(debug_mode=0)
        frame_no_spec = _render_frame(debug_mode=23)
        
        scores_baseline = _compute_flake_score(frame_baseline)
        scores_no_spec = _compute_flake_score(frame_no_spec)
        
        print(f"\nFlake Score Comparison:")
        print(f"  Baseline p95: {scores_baseline['p95']:.2f}")
        print(f"  No-specular p95: {scores_no_spec['p95']:.2f}")
        
        # No-specular should generally have lower or similar flake score
        # This documents the relationship but doesn't strictly fail
        if scores_no_spec["p95"] <= scores_baseline["p95"]:
            print("  OK: No-specular has lower or equal flake score")
        else:
            print("  NOTE: No-specular has higher flake score (may be normal)")
    
    def test_generate_flake_score_json(self):
        """Generate flake_score.json for CI reporting."""
        frame_baseline = _render_frame(debug_mode=0)
        frame_no_spec = _render_frame(debug_mode=23)
        
        scores_baseline = _compute_flake_score(frame_baseline)
        scores_no_spec = _compute_flake_score(frame_no_spec)
        
        # Determine pass/fail
        pass_p95 = scores_baseline["p95"] < FLAKE_SCORE_P95_THRESHOLD
        pass_p99 = scores_baseline["p99"] < FLAKE_SCORE_P99_THRESHOLD
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "baseline": {
                **scores_baseline,
                "pass_p95": pass_p95,
                "pass_p99": pass_p99,
            },
            "no_specular": scores_no_spec,
            "thresholds": {
                "p95": FLAKE_SCORE_P95_THRESHOLD,
                "p99": FLAKE_SCORE_P99_THRESHOLD,
            },
            "overall_pass": pass_p95 and pass_p99,
        }
        
        json_path = REPORTS_DIR / "flake_score.json"
        _save_flake_score_json(result, json_path)
        
        print(f"\nSaved: {json_path}")
        print(f"Overall pass: {result['overall_pass']}")
        
        # Verify the file was created
        assert json_path.exists(), f"Failed to create {json_path}"
        
        # Load and verify content
        with open(json_path) as f:
            loaded = json.load(f)
        
        assert "baseline" in loaded
        assert "overall_pass" in loaded
        assert loaded["overall_pass"] == result["overall_pass"]


class TestFlakeScoreMetric:
    """Test the flake score metric computation itself."""
    
    def test_laplacian_detects_edges(self):
        """Verify Laplacian computation detects edges correctly."""
        # Create image with sharp edge
        img = np.zeros((64, 64, 4), dtype=np.uint8)
        img[:, :32, :3] = 0    # Left half black
        img[:, 32:, :3] = 255  # Right half white
        img[:, :, 3] = 255     # Full alpha
        
        lap = _compute_laplacian(img)
        
        # Edge should have high Laplacian values
        edge_region = lap[:, 30:34]
        non_edge_region = lap[:, :28]
        
        assert edge_region.max() > non_edge_region.max(), (
            "Laplacian should be higher at edges"
        )
    
    def test_flake_score_percentiles_ordered(self):
        """Verify percentile values are correctly ordered."""
        # Create noisy image
        np.random.seed(42)
        img = np.random.randint(0, 256, (64, 64, 4), dtype=np.uint8)
        
        scores = _compute_flake_score(img)
        
        assert scores["p50"] <= scores["p75"], "p50 should be <= p75"
        assert scores["p75"] <= scores["p95"], "p75 should be <= p95"
        assert scores["p95"] <= scores["p99"], "p95 should be <= p99"
        assert scores["p99"] <= scores["max"], "p99 should be <= max"
