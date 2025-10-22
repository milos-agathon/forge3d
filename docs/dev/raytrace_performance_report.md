# Ray Tracing Performance Report: Switzerland 4K @ 50 SPP

## Summary

Implemented optimizations for terrain ray tracing but hit **critical bottleneck**: CPU BVH construction is pathologically slow for large terrain meshes.

## Current State (After Optimizations)

### Achieved ✅
1. **Mesh decimation**: 7.9M → 500k triangles (configurable via `--max-rt-triangles`)
2. **Degenerate triangle filtering**: Filters near-zero area triangles (0 filtered in test)
3. **Detailed instrumentation**: Per-stage timing (prep, BVH, upload, render)
4. **GPU LBVH integration attempt**: Falls back to CPU SAH when GPU fails

### Bottleneck ⚠️
**BVH Construction**: 500k triangles takes **60+ seconds** on CPU SAH builder

Breakdown of 4K @ 50 SPP render:
- ✅ Mesh prep: 20-50ms (fast)
- ⛔ **BVH build: 60,000+ ms (STUCK HERE)**
- ❓ GPU upload: ~100ms (not reached)
- ❓ Ray tracing: ~30,000-60,000ms estimated
- ❓ Denoise: ~5,000ms estimated

**Total**: Cannot complete (hangs at BVH)

## Root Cause

The CPU SAH (Surface Area Heuristic) BVH builder in `src/accel/sah_cpu.rs` is O(N log N) but with a very high constant factor. For 500k triangles:
- Expected: ~1-2 seconds
- Actual: 60+ seconds (30-60x slower)

This suggests the SAH builder is either:
1. Using inefficient algorithms (full O(N²) scan per split?)
2. Missing spatial hashing optimizations
3. Spending excessive time on degenerate nearly-flat triangles (terrain zscale=0.05)

## Attempted Solutions

### 1. GPU LBVH (Failed) ❌
```
[PT-RUST] GPU LBVH failed (wgpu error: Validation Error - Incompatible bind group), falling back to CPU SAH...
```

The GPU LBVH builder has a **bind group layout compatibility bug** in the radix sort pipeline. This needs separate debugging but would provide 100-1000x speedup (milliseconds vs minutes).

### 2. Aggressive Decimation (Partial) ⚙️
Reducing to 100k triangles (--max-rt-triangles 100000) makes BVH tractable (~5-10s) but loses significant terrain detail.

## Recommendations

### Short-term (hours)
1. **Reduce default triangle budget** to 100k-200k for interactive use
2. **Document realistic expectations**: 1-2 min for 1080p @ 32 SPP, 5-10 min for 4K @ 50 SPP
3. **Add progress callback** to BVH construction so users know it's working

### Medium-term (days)
1. **Fix GPU LBVH bind groups**: Debug radix sort pipeline compatibility
2. **Optimize CPU SAH**: Add spatial binning, parallel splits, early termination
3. **Implement progressive rendering**: Render in 4x4 or 8x8 tiles with progress updates

### Long-term (weeks)
1. **Heightfield accelerator**: Replace triangle mesh with 2D DDA traversal (optimal for DEMs)
2. **Adaptive LOD**: Generate multiple triangle density levels, select based on screen-space size
3. **Streaming BVH**: Build incrementally, start rendering as soon as root is ready

## Test Results

### Command
```bash
python3 examples/switzerland_landcover_drape.py \
  --render-mode raytrace --rt-spp 50 --rt-seed 111 \
  --zscale 0.05 --phi 80 --theta 0 --gamma 45 \
  --width 3840 --height 2160 \
  --max-rt-triangles 500000 \
  --output test_rt_4k.png
```

### Output
```
[RAYTRACE] Decimating mesh: 1315x3000 -> 331x755 (0.252x)
[RAYTRACE] Triangle count: 7,881,372 -> ~497,640
[RAYTRACE] Mesh: 249,905 vertices, 497,640 triangles
[RAYTRACE] Prep time: 0.02s
[PT-RUST] Triangle prep: 2ms (497640 valid, 0 degenerate filtered)
[PT-RUST] Starting BVH construction for 497640 triangles...
[HANGS HERE - NO FURTHER OUTPUT AFTER 60+ SECONDS]
```

### Working Configuration (Reduced Quality)
```bash
--max-rt-triangles 100000  # ~100k triangles
# Expected time: ~2-3 minutes for 4K @ 50 SPP
```

## Code Changes Made

### Python (`python/forge3d/terrain.py`)
- ✅ Added `max_rt_triangles` parameter (default: 2M)
- ✅ Intelligent mesh decimation based on triangle budget
- ✅ Progress logging at each stage

### Rust (`src/lib.rs`)
- ✅ Degenerate triangle filtering (normal length < 1e-24)
- ✅ Per-stage timing instrumentation
- ✅ GPU LBVH fallback to CPU SAH
- ✅ Detailed debug output

### CLI (`examples/switzerland_landcover_drape.py`)
- ✅ Added `--max-rt-triangles` flag
- ✅ Added `--rt-batch-spp` flag (not yet used in backend)

## Next Steps

The **immediate priority** is fixing the GPU LBVH bind group issue or optimizing the CPU SAH builder. Until then, users should:

1. Use `--max-rt-triangles 100000` for interactive previews
2. Use raster mode for quick iterations
3. Reserve raytrace mode for final renders with patience

## Performance Target Status

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| 4K @ 50 SPP | 2-5 min | ∞ (hangs) | ❌ FAIL |
| 1080p @ 32 SPP | <1 min | ~30-60s* | ⚠️ ESTIMATED |
| Triangle limit | 2M | 100k practical | ⚠️ REDUCED |
| Progress updates | Every 0.5-1s | None during BVH | ❌ MISSING |

*With `--max-rt-triangles 100000`
