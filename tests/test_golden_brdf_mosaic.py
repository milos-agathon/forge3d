#!/usr/bin/env python3
"""
P7-09: Golden BRDF Mosaic Tests

Compares rendered BRDF gallery against reference golden images to validate
rendering consistency across GPU hardware. Uses SSIM/PSNR thresholds to
tolerate cross-GPU variance while catching regressions.

Environment Variables:
    FORGE3D_RECORD_GOLDENS=1  Re-record golden images locally
"""
import os
import sys
from pathlib import Path
import pytest
import numpy as np

try:
    import forge3d as f3d
    FORGE3D_AVAILABLE = True
except ImportError:
    FORGE3D_AVAILABLE = False

try:
    import forge3d._forge3d as f3d_native
    NATIVE_AVAILABLE = hasattr(f3d_native, 'render_brdf_tile') if FORGE3D_AVAILABLE else False
except (ImportError, AttributeError):
    NATIVE_AVAILABLE = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# Test configuration
GOLDEN_DIR = Path(__file__).parent / "golden" / "p7"
RECORD_GOLDENS = os.environ.get("FORGE3D_RECORD_GOLDENS", "0") == "1"

# SSIM/PSNR thresholds to tolerate cross-GPU variance
SSIM_THRESHOLD = 0.95  # Structural similarity (1.0 = identical)
PSNR_THRESHOLD = 35.0  # Peak signal-to-noise ratio (higher = more similar)

# Skip decorators
skip_if_no_forge3d = pytest.mark.skipif(
    not FORGE3D_AVAILABLE,
    reason="forge3d not available"
)
skip_if_no_native = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="Native module with GPU support not available (expected on CPU-only CI)"
)
skip_if_no_skimage = pytest.mark.skipif(
    not HAS_SKIMAGE,
    reason="scikit-image not available for SSIM/PSNR comparison"
)


def create_mosaic(tiles: list, gap: int = 4) -> np.ndarray:
    """Stitch tiles into a grid mosaic.
    
    Args:
        tiles: List of RGBA tiles (each is H×W×4 uint8)
        gap: Gap size between tiles in pixels
        
    Returns:
        Stitched RGBA mosaic image
    """
    if not tiles:
        raise ValueError("Empty tiles list")
    
    # Assume square grid
    grid_size = int(np.sqrt(len(tiles)))
    if grid_size * grid_size != len(tiles):
        raise ValueError(f"Tiles count {len(tiles)} is not a perfect square")
    
    tile_h, tile_w = tiles[0].shape[:2]
    
    # Calculate mosaic dimensions
    mosaic_h = grid_size * tile_h + (grid_size - 1) * gap
    mosaic_w = grid_size * tile_w + (grid_size - 1) * gap
    
    # Create mosaic with dark background
    mosaic = np.full((mosaic_h, mosaic_w, 4), [20, 20, 20, 255], dtype=np.uint8)
    
    # Place tiles
    for idx, tile in enumerate(tiles):
        row = idx // grid_size
        col = idx % grid_size
        y = row * (tile_h + gap)
        x = col * (tile_w + gap)
        mosaic[y:y+tile_h, x:x+tile_w] = tile
    
    return mosaic


def compute_similarity(img1: np.ndarray, img2: np.ndarray) -> tuple[float, float]:
    """Compute SSIM and PSNR between two images.
    
    Args:
        img1: First image (H×W×4 uint8)
        img2: Second image (H×W×4 uint8)
        
    Returns:
        (ssim_score, psnr_score)
    """
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image required for similarity metrics")
    
    # Convert to float [0, 1]
    img1_f = img1.astype(np.float32) / 255.0
    img2_f = img2.astype(np.float32) / 255.0
    
    # Compute SSIM (use all channels)
    ssim_score = ssim(img1_f, img2_f, channel_axis=2, data_range=1.0)
    
    # Compute PSNR
    psnr_score = psnr(img1_f, img2_f, data_range=1.0)
    
    return ssim_score, psnr_score


@skip_if_no_forge3d
@skip_if_no_native
@skip_if_no_skimage
class TestGoldenBrdfMosaic:
    """Golden reference tests for BRDF mosaic rendering."""
    
    def test_golden_3x3_ggx_disney_phong(self):
        """
        P7-09: Compare 3×3 BRDF mosaic against golden reference.
        
        Models: GGX, Disney, Phong (3 models)
        Roughness: 0.3, 0.5, 0.7 (3 values)
        Tile size: 128×128
        Total: 9 tiles in 3×3 grid
        """
        golden_path = GOLDEN_DIR / "mosaic_3x3_128.png"
        
        # Configuration
        models = ["ggx", "disney", "phong"]
        roughness_values = [0.3, 0.5, 0.7]
        tile_size = 128
        
        # Render tiles
        tiles = []
        for model in models:
            for roughness in roughness_values:
                tile = f3d.render_brdf_tile(model, roughness, tile_size, tile_size, False)
                tiles.append(tile)
        
        # Create mosaic
        mosaic = create_mosaic(tiles, gap=4)
        
        # Record mode: save golden and pass
        if RECORD_GOLDENS:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            if HAS_PIL:
                Image.fromarray(mosaic).save(golden_path)
                print(f"\n✓ Recorded golden: {golden_path}")
                print(f"  Mosaic size: {mosaic.shape[1]}×{mosaic.shape[0]}")
            else:
                # Fallback to numpy
                np.save(golden_path.with_suffix('.npy'), mosaic)
                print(f"\n✓ Recorded golden (numpy): {golden_path.with_suffix('.npy')}")
            pytest.skip("Golden recorded successfully")
            return
        
        # Comparison mode: load golden and compare
        if not golden_path.exists():
            pytest.fail(
                f"Golden reference not found: {golden_path}\n"
                f"Run with FORGE3D_RECORD_GOLDENS=1 to generate it."
            )
        
        # Load golden
        if HAS_PIL:
            golden = np.array(Image.open(golden_path))
        else:
            # Try numpy fallback
            npy_path = golden_path.with_suffix('.npy')
            if npy_path.exists():
                golden = np.load(npy_path)
            else:
                pytest.fail(f"Cannot load golden (PIL unavailable): {golden_path}")
        
        # Verify shapes match
        assert mosaic.shape == golden.shape, \
            f"Shape mismatch: rendered {mosaic.shape} vs golden {golden.shape}"
        
        # Compute similarity metrics
        ssim_score, psnr_score = compute_similarity(mosaic, golden)
        
        print(f"\nSimilarity metrics:")
        print(f"  SSIM: {ssim_score:.4f} (threshold: {SSIM_THRESHOLD})")
        print(f"  PSNR: {psnr_score:.2f} dB (threshold: {PSNR_THRESHOLD})")
        
        # Assert thresholds
        assert ssim_score >= SSIM_THRESHOLD, \
            f"SSIM {ssim_score:.4f} below threshold {SSIM_THRESHOLD} - visual regression detected"
        
        assert psnr_score >= PSNR_THRESHOLD, \
            f"PSNR {psnr_score:.2f} below threshold {PSNR_THRESHOLD} - significant pixel differences"
        
        print("✓ Golden comparison passed")
    
    def test_golden_2x3_ndf_only(self):
        """
        P7-09: Compare 2×3 NDF-only mosaic against golden reference.
        
        Models: GGX, Disney (2 models)
        Roughness: 0.2, 0.5, 0.8 (3 values)
        Tile size: 128×128
        NDF-only: True
        Total: 6 tiles in 2×3 grid
        """
        golden_path = GOLDEN_DIR / "mosaic_2x3_ndf_128.png"
        
        # Configuration
        models = ["ggx", "disney"]
        roughness_values = [0.2, 0.5, 0.8]
        tile_size = 128
        
        # Render tiles (2 models × 3 roughness = 6 tiles)
        tiles = []
        for model in models:
            for roughness in roughness_values:
                tile = f3d.render_brdf_tile(model, roughness, tile_size, tile_size, ndf_only=True)
                tiles.append(tile)
        
        # For 2×3 layout, need to pad to 3×3 (9 tiles) for square grid
        # Or implement rectangular grid - for simplicity, skip if not 9 tiles
        if len(tiles) != 9:
            # Pad with black tiles to make it square
            while len(tiles) < 9:
                black_tile = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
                black_tile[:, :, 3] = 255  # Opaque alpha
                tiles.append(black_tile)
        
        # Create mosaic
        mosaic = create_mosaic(tiles[:9], gap=4)
        
        # Record mode
        if RECORD_GOLDENS:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            if HAS_PIL:
                Image.fromarray(mosaic).save(golden_path)
                print(f"\n✓ Recorded NDF golden: {golden_path}")
            else:
                np.save(golden_path.with_suffix('.npy'), mosaic)
                print(f"\n✓ Recorded NDF golden (numpy): {golden_path.with_suffix('.npy')}")
            pytest.skip("NDF golden recorded successfully")
            return
        
        # Comparison mode
        if not golden_path.exists():
            pytest.skip(f"Optional golden not found: {golden_path}")
        
        # Load and compare
        if HAS_PIL:
            golden = np.array(Image.open(golden_path))
        else:
            npy_path = golden_path.with_suffix('.npy')
            if npy_path.exists():
                golden = np.load(npy_path)
            else:
                pytest.skip("Cannot load NDF golden (PIL unavailable)")
        
        assert mosaic.shape == golden.shape
        
        ssim_score, psnr_score = compute_similarity(mosaic, golden)
        
        print(f"\nNDF similarity metrics:")
        print(f"  SSIM: {ssim_score:.4f} (threshold: {SSIM_THRESHOLD})")
        print(f"  PSNR: {psnr_score:.2f} dB (threshold: {PSNR_THRESHOLD})")
        
        assert ssim_score >= SSIM_THRESHOLD, \
            f"NDF SSIM {ssim_score:.4f} below threshold"
        assert psnr_score >= PSNR_THRESHOLD, \
            f"NDF PSNR {psnr_score:.2f} below threshold"
        
        print("✓ NDF golden comparison passed")


def test_skip_gracefully_without_gpu():
    """Verify golden tests skip gracefully on CPU-only CI."""
    if not FORGE3D_AVAILABLE:
        pytest.skip("forge3d not available (expected on CPU-only CI)")
    elif not NATIVE_AVAILABLE:
        pytest.skip("Native module with GPU support not available (expected on CPU-only CI)")
    elif not HAS_SKIMAGE:
        pytest.skip("scikit-image not available (install with: pip install scikit-image)")
    else:
        print("\n✓ GPU and native module available - golden tests will run")


if __name__ == "__main__":
    # Run with: python tests/test_golden_brdf_mosaic.py
    # Or: FORGE3D_RECORD_GOLDENS=1 pytest tests/test_golden_brdf_mosaic.py -v
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
