# P7 Golden Reference Images

Reference images for BRDF mosaic regression testing.

## Overview

Golden images validate that BRDF tile rendering remains consistent across code changes and GPU hardware. Tests use SSIM/PSNR thresholds to tolerate minor GPU variance while catching visual regressions.

## Golden Images

### `mosaic_3x3_128.png` (Primary)
- **Models**: GGX, Disney, Phong
- **Roughness**: 0.3, 0.5, 0.7
- **Tile size**: 128×128 pixels
- **Layout**: 3×3 grid (9 tiles)
- **Size**: ~412×412 pixels
- **Purpose**: Primary regression test

### `mosaic_2x3_ndf_128.png` (Optional)
- **Models**: GGX, Disney
- **Roughness**: 0.2, 0.5, 0.8
- **Tile size**: 128×128 pixels
- **Layout**: 2×3 grid padded to 3×3 (6 active tiles)
- **NDF-only**: True (grayscale)
- **Purpose**: NDF-only debug mode validation

## Recording Goldens

### First Time Setup

```bash
# From repository root
cd /path/to/forge3d

# Ensure native module is built
maturin develop --release

# Record golden images
FORGE3D_RECORD_GOLDENS=1 pytest tests/test_golden_brdf_mosaic.py -v
```

### After Code Changes

Re-record goldens if BRDF shader or renderer changes:

```bash
# Backup old goldens
cp tests/golden/p7/mosaic_3x3_128.png tests/golden/p7/mosaic_3x3_128.png.backup

# Record new goldens
FORGE3D_RECORD_GOLDENS=1 pytest tests/test_golden_brdf_mosaic.py::TestGoldenBrdfMosaic::test_golden_3x3_ggx_disney_phong -v

# Compare visually before committing
```

### GPU-Specific Recording

Record on a representative GPU for your CI environment:

```bash
# Example: NVIDIA RTX 3080
FORGE3D_RECORD_GOLDENS=1 pytest tests/test_golden_brdf_mosaic.py -v
# Creates: tests/golden/p7/mosaic_3x3_128.png

# Commit with GPU info
git add tests/golden/p7/mosaic_3x3_128.png
git commit -m "Update P7 goldens (NVIDIA RTX 3080)"
```

## Running Golden Tests

### Standard Run (Comparison Mode)

```bash
# Run all golden tests
pytest tests/test_golden_brdf_mosaic.py -v

# Run specific test
pytest tests/test_golden_brdf_mosaic.py::TestGoldenBrdfMosaic::test_golden_3x3_ggx_disney_phong -v
```

### Expected Output (GPU Machine)

```
tests/test_golden_brdf_mosaic.py::TestGoldenBrdfMosaic::test_golden_3x3_ggx_disney_phong PASSED

Similarity metrics:
  SSIM: 0.9823 (threshold: 0.95)
  PSNR: 42.15 dB (threshold: 35.0)
✓ Golden comparison passed
```

### Expected Output (CPU-only CI)

```
tests/test_golden_brdf_mosaic.py::test_skip_gracefully_without_gpu SKIPPED
  Native module with GPU support not available (expected on CPU-only CI)
```

## Similarity Thresholds

### SSIM (Structural Similarity Index)
- **Range**: 0.0 to 1.0 (1.0 = identical)
- **Threshold**: 0.95
- **Purpose**: Detects structural/perceptual changes
- **Tolerance**: Allows minor GPU variance in texture filtering, rounding

### PSNR (Peak Signal-to-Noise Ratio)
- **Range**: 0 to ∞ dB (higher = more similar)
- **Threshold**: 35.0 dB
- **Purpose**: Quantifies pixel-level differences
- **Tolerance**: Allows ~1-2 unit per-pixel variations

### Why Both Metrics?

- **SSIM**: Better for detecting visual quality changes
- **PSNR**: Better for detecting subtle numerical drift
- Combined: Robust against false positives while catching real regressions

## Handling Test Failures

### SSIM Below Threshold

```
AssertionError: SSIM 0.8912 below threshold 0.95 - visual regression detected
```

**Causes:**
- BRDF shader changes
- Camera/lighting changes
- Mesh generation changes
- Major GPU driver update

**Actions:**
1. Visually inspect rendered vs golden
2. If intentional: re-record golden
3. If bug: fix shader/renderer

### PSNR Below Threshold

```
AssertionError: PSNR 28.34 below threshold 35.0 - significant pixel differences
```

**Causes:**
- Precision changes in calculations
- Tone mapping modifications
- Exposure changes

**Actions:**
1. Check for unintended numerical changes
2. Verify exposure=1.0 and tone mapping disabled
3. Re-record if shader improvements are intentional

### Both Pass but Visual Diff

Thresholds allow minor variance. If images look identical but metrics are near threshold:
- GPU variance is within tolerance
- Test is working correctly

## Cross-GPU Variance

### Expected Differences

Different GPUs may produce slightly different results due to:
- Floating point precision differences
- Texture filtering implementations
- Driver optimizations
- Rounding modes

### Threshold Selection

Current thresholds (SSIM ≥ 0.95, PSNR ≥ 35.0) are chosen to:
- **Pass** on different GPU families (NVIDIA, AMD, Intel)
- **Fail** on visual regressions
- **Tolerate** 1-2 units per-pixel variance

### Testing on Multiple GPUs

Record goldens on primary CI GPU:
```bash
# NVIDIA RTX 3080 (primary CI)
FORGE3D_RECORD_GOLDENS=1 pytest tests/test_golden_brdf_mosaic.py -v
```

Validate on other GPUs (comparison mode):
```bash
# AMD RX 6800 XT (validation)
pytest tests/test_golden_brdf_mosaic.py -v
# Should pass with SSIM ≥ 0.95
```

## File Size Considerations

Keep golden images small to avoid large repository:
- **128×128 tiles**: ~412×412 mosaic (~200KB PNG)
- **3×3 grid**: Only 9 tiles, fast to render
- **No 4K tiles**: Would create multi-MB files

## CI Integration

### GitHub Actions Example

```yaml
- name: Run golden BRDF tests
  run: |
    pytest tests/test_golden_brdf_mosaic.py -v
  continue-on-error: false
  if: runner.gpu == 'available'
```

### Skip on CPU-only

Tests automatically skip when GPU unavailable:
```python
@skip_if_no_native
class TestGoldenBrdfMosaic:
    # Tests only run with GPU
```

## Troubleshooting

### "Golden reference not found"

```
pytest.Failed: Golden reference not found: tests/golden/p7/mosaic_3x3_128.png
Run with FORGE3D_RECORD_GOLDENS=1 to generate it.
```

**Solution:**
```bash
FORGE3D_RECORD_GOLDENS=1 pytest tests/test_golden_brdf_mosaic.py -v
```

### "scikit-image not available"

```
SKIPPED: scikit-image not available (install with: pip install scikit-image)
```

**Solution:**
```bash
pip install scikit-image
```

### Shape Mismatch

```
AssertionError: Shape mismatch: rendered (412, 412, 4) vs golden (256, 256, 4)
```

**Cause:** Golden was recorded with different tile size

**Solution:** Re-record golden with correct configuration

## Related Files

- `tests/test_golden_brdf_mosaic.py`: Golden test implementation
- `tests/test_brdf_tile.py`: Unit tests without golden comparison
- `examples/brdf_gallery.py`: Gallery generator
- `docs/P7_ACCEPTANCE.md`: Acceptance documentation

## Maintenance

### When to Update Goldens

✅ **Required:**
- BRDF shader algorithm changes
- NDF formula modifications
- Rendering pipeline improvements

⚠️ **Optional:**
- Minor GPU driver updates
- Precision improvements
- Code refactoring (no visual change)

❌ **Never:**
- Random test failures
- Cross-GPU variance within thresholds
- Temporary debugging changes

### Version Control

```bash
# Check golden status
git status tests/golden/p7/

# View golden diff (if binary diff tool configured)
git diff tests/golden/p7/mosaic_3x3_128.png

# Commit updated goldens
git add tests/golden/p7/*.png
git commit -m "Update P7 goldens after BRDF shader fix"
```
