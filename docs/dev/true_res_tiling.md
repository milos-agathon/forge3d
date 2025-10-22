# True-Resolution Tiled GPU Path Tracing

## Overview

The true-resolution tiled rendering system allows rendering large images (>640px) without downscaling by breaking the render into tiles. Each tile is rendered at full resolution and assembled into the final image.

## Architecture

### Coordinate Systems

The tiling system uses two coordinate systems:

1. **Global (Image) Coordinates**: Full image pixel coordinates `(gx, gy)` in range `[0, img_w) × [0, img_h)`
2. **Local (Tile) Coordinates**: Per-tile pixel coordinates `(lx, ly)` in range `[0, tile_w) × [0, tile_h)`

### WGSL Shader (hybrid_kernel.wgsl)

The shader receives:
- `TileParams.tile_origin_size`: `[tile_x, tile_y, tile_w, tile_h]` - tile position and size
- `TileParams.img_spp_counts`: `[img_w, img_h, spp_batch, spp_done]` - full image dimensions and sampling info
- `Uniforms.width/height`: **Must be full image dimensions** for correct camera ray generation

**Critical**: The shader computes global coordinates as:
```wgsl
let gx = tile_params.tile_origin_size.x + lx;
let gy = tile_params.tile_origin_size.y + ly;
```

Camera rays use `(gx, gy)` for seamless perspective across tiles.

Texture writes use local coordinates `(lx, ly)` to write into the tile-sized output texture.

### Rust Backend (hybrid_compute.rs)

The `render_tile_with_params` function must:
1. Extract full image dimensions from `tile_params.img_spp_counts[0..1]`
2. Set `base_uniforms.width/height` to **full image dimensions**, not tile dimensions
3. Create output texture with tile dimensions
4. Pass `TileParams` to shader for coordinate mapping

**Bug Fixed**: Previously, tile dimensions were incorrectly written to `base_uniforms.width/height`, causing the camera to use wrong aspect ratio and field of view, resulting in black tiles for certain positions.

## Debug Modes

The shader supports several diagnostic modes via `TileParams.seeds_misc[1]` (debug_mode):

| Mode | Purpose | Output |
|------|---------|--------|
| 0 | Normal rendering | Path-traced scene |
| 1 | Tile grid diagnostic | Colored grid with tile IDs |
| 2 | Green sentinel | RGB bands (for testing) |
| 3 | Environment only | HDRI/sky without geometry |
| 4 | **XY gradient** | Continuous gradient to verify tile addressing |

### Debug Mode 4: XY Gradient

This mode renders a pure XY gradient:
- Red channel = horizontal position (X/img_w)
- Green channel = vertical position (Y/img_h)
- Blue channel = 0.5 (constant)
- Tile borders every 64px appear darker

**Usage**:
```bash
python examples/switzerland_landcover_drape.py --debug-tiles --width 1280 --height 720
```

**Expected result**: Smooth gradient from red (left) to yellow (right), dark (bottom) to bright (top), with faint grid lines.

**Failure modes**:
- **Solid black tiles**: Shader not writing (check bind groups, texture size)
- **Seams at tile boundaries**: Global coordinate computation wrong
- **Solid color tiles**: TileParams not updating per dispatch

## Validation

The Rust backend validates each tile after rendering:

```rust
// Check for solid black or single-color tiles
let is_solid_color = uniq.len() == 1;
let is_nearly_black = mean_r < 1.0 && mean_g < 1.0 && mean_b < 1.0;

if is_solid_color || is_nearly_black {
    eprintln!("[HYBRID-RENDER] WARNING: Tile {} is solid color! ...");
}
```

This warning indicates a critical rendering failure that must be addressed.

## Common Issues and Solutions

### Issue: Top tiles are solid black

**Root Cause**: `base_uniforms.width/height` set to tile dimensions instead of full image dimensions.

**Solution**: In `render_tile_with_params`, always use:
```rust
modified_params.base_uniforms.width = tile_params.img_spp_counts[0];
modified_params.base_uniforms.height = tile_params.img_spp_counts[1];
```

### Issue: Seams at tile boundaries

**Root Cause**: Camera ray generation uses local coordinates instead of global.

**Solution**: Ensure shader computes `global_px/py` by adding tile origin:
```wgsl
let global_px = f32(gx) + 0.5 + jx;
let global_py = f32(gy) + 0.5 + jy;
```

### Issue: Wrong colors in some tiles

**Root Cause**: Bind groups (HDRI, mesh buffers) not set for every dispatch.

**Solution**: Set all bind groups before each tile dispatch in the compute pass.

## Memory Management

Tiles are rendered sequentially with the same output texture size, minimizing memory usage:
- Single tile-sized output texture (e.g., 512×512 RGBA16F = 2 MB)
- Shared BVH and mesh buffers across all tiles
- TileParams buffer updated per dispatch (256 bytes)

Total overhead: ~2-4 MB per tile + shared scene geometry.

## Testing

Run regression tests:
```bash
pytest tests/test_tile_black_regression.py -v
```

Run gradient diagnostic on real data:
```bash
python examples/switzerland_landcover_drape.py --debug-tiles --width 1280 --height 720 --output debug_tiles.png
```

Visually inspect `debug_tiles.png` for:
- ✅ Smooth gradients with no seams
- ✅ All tiles have color
- ✅ Grid lines visible every 64px
- ❌ Black tiles indicate addressing bug
- ❌ Seams indicate coordinate mismatch

## Performance

Typical render times for Switzerland 1280×720 with 512×512 tiles:
- 6 tiles total (3×2 grid)
- ~2-5 seconds per tile @ 1 SPP
- ~12-30 seconds total (without OIDN)
- OIDN denoising: +5-10 seconds

Memory usage stays within 512 MB budget with auto-tiling enabled.
