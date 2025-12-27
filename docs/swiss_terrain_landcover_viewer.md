# Swiss Terrain with Land Cover Overlay - User Guide

## File Location
`examples/swiss_terrain_landcover_viewer.py`

## Overview

Interactive 3D viewer for Swiss terrain (DEM elevation) with land cover classification draped as a lit overlay. The land cover is automatically resampled to match DEM resolution and reprojected to EPSG:3035 (LAEA Europe).

**Key Features:**
- **Elevation:** Switzerland DEM from `assets/tif/switzerland_dem.tif`
- **Land Cover Overlay:** Classification from `assets/tif/switzerland_land_cover.tif`
- **Automatic Resampling:** Land cover matched to DEM resolution and CRS
- **Lit Overlays:** Land cover blended into albedo before lighting → receives sun, shadows, AO
- **4 High-Quality Presets:** Optimized rendering configurations for different use cases
- **Interactive Controls:** Real-time camera orbit, overlay opacity adjustment, snapshots

## Quick Start

### Basic Usage

```bash
# Interactive viewer with default settings
python examples/swiss_terrain_landcover_viewer.py

# With high-quality preset 1 (standard quality)
python examples/swiss_terrain_landcover_viewer.py --preset hq1

# Take snapshot with maximum quality preset
python examples/swiss_terrain_landcover_viewer.py --preset hq4 --snapshot swiss_render.png

# Custom overlay opacity
python examples/swiss_terrain_landcover_viewer.py --overlay-opacity 0.7 --preset hq2
```

### Prerequisites

**Required:**
- `rasterio` for GeoTIFF handling and reprojection
  ```bash
  pip install rasterio
  ```

**Optional:**
- `Pillow` for PNG export (fallback uses NumPy)
  ```bash
  pip install pillow
  ```

**Build Requirements:**
- Interactive viewer binary must be built:
  ```bash
  cargo build --release --bin interactive_viewer
  ```

## High-Quality Presets

The script provides 4 optimized presets for different rendering scenarios:

### Preset HQ1: Standard High Quality
```bash
python examples/swiss_terrain_landcover_viewer.py --preset hq1
```

**Configuration:**
- Resolution: 3840×2160 (4K)
- MSAA: 8x
- Heightfield AO: Enabled (8 directions, 24 steps)
- Sun Visibility: Enabled (soft shadows, 6 samples, 32 steps)
- Exposure: 1.2
- Temperature: 6500K (neutral D65)

**Use Case:** General-purpose high-quality renders with balanced performance.

### Preset HQ2: Alpine
```bash
python examples/swiss_terrain_landcover_viewer.py --preset hq2
```

**Configuration:**
- Resolution: 3840×2160 (4K)
- MSAA: 4x
- Heightfield AO: Enabled
- Sun Visibility: Enabled
- Snow Layer: Enabled (altitude ≥2200m, slope ≤50°)
- Rock Layer: Enabled (slope ≥40°)
- Exposure: 1.3
- Temperature: 7000K (cool, for alpine snow)

**Use Case:** Alpine mountain scenes with realistic snow/rock material layers.

### Preset HQ3: Cinematic
```bash
python examples/swiss_terrain_landcover_viewer.py --preset hq3
```

**Configuration:**
- Resolution: 3840×2160 (4K)
- MSAA: 4x
- Heightfield AO: Enabled
- Depth of Field: Enabled (f/8.0, 85mm focal length, 3000m focus)
- Lens Effects: Enabled (chromatic aberration, vignette)
- Exposure: 1.4
- Temperature: 5500K (warm golden hour)

**Use Case:** Cinematic renders with camera effects and warm color grading.

### Preset HQ4: Maximum Quality
```bash
python examples/swiss_terrain_landcover_viewer.py --preset hq4
```

**Configuration:**
- Resolution: 3840×2160 (4K)
- MSAA: 8x
- Heightfield AO: Enabled
- Sun Visibility: Enabled
- Snow Layer: Enabled
- Rock Layer: Enabled
- Depth of Field: Enabled
- Lens Effects: Enabled
- Exposure: 1.2
- Temperature: 6500K (neutral)

**Use Case:** Maximum quality renders with all effects enabled (slower performance).

## Command-Line Options

### Basic Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dem` | Path | `assets/tif/switzerland_dem.tif` | Path to DEM GeoTIFF |
| `--landcover` | Path | `assets/tif/switzerland_land_cover.tif` | Path to land cover GeoTIFF |
| `--width` | int | 1920 | Window width in pixels |
| `--height` | int | 1080 | Window height in pixels |
| `--snapshot` | Path | None | Take snapshot and exit |
| `--crs` | str | `EPSG:3035` | Target CRS for reprojection |
| `--preset` | choice | None | High-quality preset: `hq1`, `hq2`, `hq3`, `hq4` |

### Overlay Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--overlay-opacity` | float | 0.6 | Overlay opacity 0.0-1.0 |
| `--no-overlay` | flag | False | Disable land cover overlay (DEM only) |

### PBR Rendering Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--exposure` | float | 1.0 | ACES exposure multiplier |
| `--msaa` | choice | 1 | MSAA samples: 1, 4, or 8 |
| `--shadows` | choice | `pcss` | Shadow technique: `none`, `hard`, `pcf`, `pcss` |

### Sun Lighting Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--sun-azimuth` | float | 135.0 | Sun azimuth angle in degrees |
| `--sun-elevation` | float | 35.0 | Sun elevation angle in degrees |

## Interactive Commands

Once the viewer is running, you can use these terminal commands:

### Overlay Control
```
overlay on              # Enable land cover overlay
overlay off             # Disable land cover overlay
overlay opacity=0.5     # Set overlay opacity (0.0-1.0)
```

### Camera Control
```
set phi=45 theta=60 radius=2000    # Set camera position
set fov=35                         # Set field of view
```

### Sun Control
```
set sun_az=135 sun_el=45           # Set sun direction
set intensity=1.5                  # Set sun intensity
```

### Terrain Control
```
set zscale=0.2                     # Set height exaggeration
```

### Snapshot
```
snap output.png                    # Take screenshot at current resolution
snap output.png 3840x2160          # Take screenshot at specific resolution
```

### Other
```
params                             # Show current parameters
quit                               # Exit viewer
```

## Technical Details

### Land Cover Resampling

The script automatically resamples the land cover to match the DEM:

1. **Read DEM metadata:** Extract CRS, transform, dimensions
2. **Reproject land cover:** Transform to target CRS (EPSG:3035)
3. **Resample:** Use nearest-neighbor resampling (preserves class values)
4. **Colormap:** Apply Corine-style land cover colors
5. **Export:** Save as RGBA PNG for overlay system

**Resampling Method:**
- Uses `rasterio.warp.reproject()` with `Resampling.nearest`
- Nearest-neighbor preserves categorical class values
- Output dimensions match DEM exactly for pixel-perfect alignment

### Land Cover Colormap

The script applies a Corine Land Cover-style colormap:

| Class Range | Color | Description |
|-------------|-------|-------------|
| 0 | Transparent | NoData |
| 1-9 | Red/Pink | Artificial surfaces (urban, industrial) |
| 10-19 | Yellow/Green | Agricultural areas |
| 20-29 | Green | Forest and semi-natural vegetation |
| 30-39 | Cyan | Wetlands |
| 40-49 | Blue | Water bodies |

**Specific Classes:**
- **1:** Urban fabric (red)
- **10:** Arable land (yellow)
- **20:** Broad-leaved forest (dark green)
- **21:** Coniferous forest (bright green)
- **29:** Glaciers/perpetual snow (light blue)
- **40:** Water courses (cyan)
- **41:** Water bodies (light cyan)

### Overlay Integration

The land cover overlay uses the **draped texture overlay system** (Option A):

1. **Texture Binding:** Overlay texture bound at `@binding(5)` in PBR shader
2. **UV Sampling:** Sampled in terrain UV space [0,1]×[0,1]
3. **Albedo Blending:** Blended into terrain albedo **before lighting**
4. **Lighting:** Receives full PBR lighting (sun, shadows, AO, IBL)

**Shader Integration:**
```wgsl
// Sample overlay texture
let overlay_color = textureSample(overlay_tex, overlay_sampler, uv);

// Blend into albedo (before lighting)
let albedo = mix(terrain_albedo, overlay_color.rgb, overlay_color.a * overlay_opacity);

// Apply PBR lighting to blended albedo
let lit_color = calculate_pbr_lighting(albedo, normal, roughness, metallic, ...);
```

This means overlays:
- ✅ Are lit by sun (diffuse term includes overlay color)
- ✅ Are shadowed (shadow_term multiplies diffuse result)
- ✅ Receive ambient occlusion (height_ao multiplies ambient term)
- ❌ Do NOT affect specular (specular depends on roughness, not albedo)

### Camera Configuration

Default camera settings optimized for Swiss Alps:

```python
{
    "phi": 45.0,        # Azimuth angle (degrees)
    "theta": 40.0,      # Elevation angle (degrees)
    "radius": 4500.0,   # Distance from terrain center (meters)
    "fov": 35.0,        # Field of view (degrees)
    "zscale": 0.15,     # Height exaggeration (15% of terrain width)
}
```

### Material Settings

**Snow Layer (Alpine Preset):**
- Altitude threshold: 2200m (Swiss Alps snow line)
- Altitude blend: 300m transition zone
- Slope threshold: 50° (snow doesn't accumulate on steep slopes)

**Rock Layer (Alpine Preset):**
- Slope threshold: 40° (exposed rock on steep faces)

## Practical Examples

### Example 1: Quick Preview
```bash
# Fast preview with default settings
python examples/swiss_terrain_landcover_viewer.py
```

**Result:** 1920×1080 interactive viewer, land cover overlay at 60% opacity, basic PBR rendering.

### Example 2: High-Quality Snapshot
```bash
# 4K snapshot with standard high quality
python examples/swiss_terrain_landcover_viewer.py \
  --preset hq1 \
  --snapshot swiss_hq.png
```

**Result:** 3840×2160 PNG with MSAA 8x, heightfield AO, sun visibility, exposure 1.2.

### Example 3: Alpine Scene with Snow
```bash
# Alpine preset with custom sun angle
python examples/swiss_terrain_landcover_viewer.py \
  --preset hq2 \
  --sun-azimuth 120 \
  --sun-elevation 25 \
  --snapshot swiss_alpine.png
```

**Result:** 4K render with snow/rock layers, cool temperature (7000K), low sun angle for dramatic shadows.

### Example 4: Cinematic Golden Hour
```bash
# Cinematic preset with warm tones
python examples/swiss_terrain_landcover_viewer.py \
  --preset hq3 \
  --sun-elevation 15 \
  --overlay-opacity 0.5 \
  --snapshot swiss_cinematic.png
```

**Result:** 4K render with DoF, lens effects, warm temperature (5500K), low sun for golden hour lighting.

### Example 5: Maximum Quality Comparison
```bash
# Maximum quality with overlay
python examples/swiss_terrain_landcover_viewer.py \
  --preset hq4 \
  --snapshot swiss_with_overlay.png

# Maximum quality without overlay (DEM only)
python examples/swiss_terrain_landcover_viewer.py \
  --preset hq4 \
  --no-overlay \
  --snapshot swiss_no_overlay.png
```

**Result:** Two 4K renders for side-by-side comparison of overlay impact.

### Example 6: Custom Resolution and Opacity
```bash
# Custom settings for specific output
python examples/swiss_terrain_landcover_viewer.py \
  --width 2560 \
  --height 1440 \
  --msaa 4 \
  --overlay-opacity 0.8 \
  --exposure 1.3 \
  --snapshot swiss_custom.png
```

**Result:** 2560×1440 render with high overlay opacity and increased exposure.

### Example 7: Interactive Exploration
```bash
# Start interactive viewer with alpine preset
python examples/swiss_terrain_landcover_viewer.py --preset hq2
```

**Interactive Session:**
```
> set phi=60 theta=35 radius=3000    # Adjust camera
> set sun_az=90 sun_el=40            # Adjust sun
> overlay opacity=0.7                # Adjust overlay
> snap view1.png                     # Take snapshot
> set phi=120                        # Rotate camera
> snap view2.png                     # Take another snapshot
> quit                               # Exit
```

### Example 8: DEM-Only Rendering
```bash
# Render DEM without land cover overlay
python examples/swiss_terrain_landcover_viewer.py \
  --no-overlay \
  --preset hq1 \
  --snapshot swiss_dem_only.png
```

**Result:** Pure elevation rendering with PBR materials (no land cover classification).

## Performance Considerations

### Resolution vs. Performance

| Resolution | MSAA | Preset | Approx. FPS (RTX 3080) |
|------------|------|--------|------------------------|
| 1920×1080 | 1x | Default | 120+ |
| 1920×1080 | 4x | HQ1 | 80-100 |
| 3840×2160 | 4x | HQ2 | 40-60 |
| 3840×2160 | 8x | HQ4 | 25-35 |

### Optimization Tips

**For Real-Time Interaction:**
- Use default settings (1920×1080, MSAA 1x)
- Disable DoF and lens effects
- Lower heightfield AO/sun visibility resolution

**For High-Quality Snapshots:**
- Use presets HQ1-HQ4
- Take snapshot and exit (no real-time requirement)
- Consider rendering at 4K and downsampling for anti-aliasing

**For Batch Rendering:**
- Use `--snapshot` mode (no window, faster)
- Script multiple camera angles
- Parallelize with different output paths

## Troubleshooting

### Issue: "rasterio not installed"
**Solution:** Install rasterio:
```bash
pip install rasterio
```

### Issue: "interactive_viewer binary not found"
**Solution:** Build the viewer:
```bash
cargo build --release --bin interactive_viewer
```

### Issue: Land cover not appearing
**Possible Causes:**
1. Overlay opacity too low → increase with `--overlay-opacity 0.8`
2. Overlay disabled → remove `--no-overlay` flag
3. Resampling failed → check rasterio installation and file paths

**Debug:**
```bash
# Check if overlay system is enabled
python examples/swiss_terrain_landcover_viewer.py
> overlay on
```

### Issue: Resampling takes too long
**Solution:** Pre-generate the overlay:
```python
# Run once to generate persistent overlay
python examples/swiss_terrain_landcover_viewer.py
# Overlay saved to: assets/tif/switzerland_landcover_overlay.png
# Subsequent runs will use cached version
```

### Issue: Colors look wrong
**Possible Causes:**
1. Exposure too high/low → adjust with `--exposure`
2. Temperature incorrect → presets set appropriate values
3. Overlay opacity masking terrain → reduce with `--overlay-opacity`

**Debug:**
```bash
# Try without overlay to check base terrain
python examples/swiss_terrain_landcover_viewer.py --no-overlay

# Try different exposures
python examples/swiss_terrain_landcover_viewer.py --exposure 0.8
python examples/swiss_terrain_landcover_viewer.py --exposure 1.5
```

## Integration with Other Tools

### Using Custom DEMs
```bash
python examples/swiss_terrain_landcover_viewer.py \
  --dem /path/to/custom_dem.tif \
  --landcover /path/to/custom_landcover.tif \
  --crs EPSG:32633
```

### Scripting Camera Paths
```python
import subprocess
import json

camera_positions = [
    {"phi": 30, "theta": 40, "radius": 4000},
    {"phi": 60, "theta": 35, "radius": 3500},
    {"phi": 90, "theta": 45, "radius": 5000},
]

for i, cam in enumerate(camera_positions):
    # Start viewer and set camera via IPC
    # (See terrain_viewer_interactive.py for IPC implementation)
    pass
```

### Batch Rendering
```bash
#!/bin/bash
# Render multiple presets
for preset in hq1 hq2 hq3 hq4; do
  python examples/swiss_terrain_landcover_viewer.py \
    --preset $preset \
    --snapshot swiss_${preset}.png
done
```

## Comparison with Other Viewers

### vs. `terrain_viewer_interactive.py`
- **Swiss Viewer:** Specialized for Swiss terrain with land cover overlay
- **Generic Viewer:** Works with any DEM, no overlay support
- **Use Swiss Viewer When:** You need land cover classification visualization
- **Use Generic Viewer When:** You need maximum flexibility and custom settings

### vs. `terrain_demo.py`
- **Swiss Viewer:** Interactive with real-time controls
- **Terrain Demo:** Batch rendering with Python API
- **Use Swiss Viewer When:** You need interactive exploration
- **Use Terrain Demo When:** You need programmatic rendering

## References

- **Overlay System Documentation:** `docs/terrain_overlays.rst`
- **Overlay Implementation Plan:** `docs/plan_overlays_option_a_draped_textures.md`
- **PBR Shader Reference:** `docs/terrain_pbr_pom_shader_reference.md`
- **Generic Interactive Viewer:** `examples/terrain_viewer_interactive.py`
- **Python API Tests:** `tests/test_terrain_overlay_stack.py`

## Coordinate Systems

### EPSG:3035 (LAEA Europe)
- **Name:** Lambert Azimuthal Equal Area (Europe)
- **Units:** Meters
- **Coverage:** Europe-wide
- **Use Case:** Area-preserving projection for statistical analysis
- **Swiss Coverage:** Full coverage with minimal distortion

### Alternative CRS Options
```bash
# Swiss national grid (LV95)
--crs EPSG:2056

# UTM Zone 32N (Central Europe)
--crs EPSG:32632

# WGS84 Geographic
--crs EPSG:4326
```

## File Outputs

### Snapshot Files
- **Format:** PNG (RGBA8)
- **Color Space:** sRGB
- **Gamma:** 2.2 (applied in shader)
- **Tonemapping:** ACES (applied in shader)

### Temporary Files
- **Resampled Overlay:** `assets/tif/switzerland_landcover_overlay.png`
  - Cached for subsequent runs
  - Delete to force regeneration

## Advanced Usage

### Custom Land Cover Colormap

To use a custom colormap, modify the `apply_landcover_colormap()` function:

```python
# In swiss_terrain_landcover_viewer.py
colormap = {
    0: (0, 0, 0, 0),          # NoData - transparent
    1: (255, 0, 0, 200),      # Class 1 - red
    2: (0, 255, 0, 200),      # Class 2 - green
    # ... add your classes
}
```

### Programmatic Control

```python
# Example: Automated rendering script
import subprocess
import time

viewer = subprocess.Popen([
    "python", "examples/swiss_terrain_landcover_viewer.py",
    "--preset", "hq1"
])

time.sleep(5)  # Wait for viewer to start

# Send IPC commands (requires socket connection)
# See terrain_viewer_interactive.py for IPC implementation
```

## Changelog

### Version 1.3 (2024-12-24)

**Bug Fixes:**

1. **Python CRS Mismatch Fix** - Fixed critical bug in `resample_landcover_to_dem()` where
   `dst_crs` was set to EPSG:3035 but `dst_transform` was from DEM in EPSG:4326. This caused
   reprojection to produce empty/zeros. Now uses DEM's CRS consistently.

2. **Float32 RGB Conversion Fix** - Fixed conversion of land cover RGB data from float32
   (values 0-255) to uint8. Also added NaN handling for robust data processing.

**Files Modified:**
- `examples/swiss_terrain_landcover_viewer.py` - Fixed CRS mismatch and float-to-uint8 conversion

### Version 1.2 (2024-12-24)

**Bug Fixes:**

1. **Overlay Texture Format Fix** - Fixed critical bug where overlay fallback used R32Float
   (AO texture format) instead of Rgba8UnormSrgb. Now correctly uses RGBA fallback from
   overlay stack, ensuring overlays are properly sampled in the shader.

2. **Overlay Composite Rebuild** - Added automatic rebuild of overlay composite before
   rendering if the stack is dirty. Ensures overlay changes are immediately visible.

3. **NoData Edge Artifacts Fix** - Fixed terrain edge artifacts (spikes/stretching) by
   replacing NoData values (-9999, NaN, etc.) with minimum elevation before GPU upload.

**Files Modified:**
- `src/viewer/terrain/render.rs` - Fixed overlay texture format and added composite rebuild
- `src/viewer/terrain/scene.rs` - Added NoData value replacement for clean terrain edges

### Version 1.1 (2024-12-24)

**Bug Fixes:**

1. **IPC Overlay Commands Added** - Added missing overlay IPC commands to `protocol.rs`:
   - `load_overlay` - Load overlay from image file
   - `remove_overlay` - Remove overlay by ID
   - `set_overlay_visible` - Toggle overlay visibility
   - `set_overlay_opacity` - Set overlay opacity
   - `set_global_overlay_opacity` - Set global opacity multiplier
   - `set_overlays_enabled` - Enable/disable overlay system
   - `list_overlays` - List all overlay IDs

2. **CPU Compositing Fix** - Fixed critical bug in `overlay.rs` where image-based overlays
   were skipped during CPU compositing. Now stores RGBA data directly instead of path
   reference for proper compositing.

3. **Render Fallback Fix** - Fixed incorrect fallback logic in `render.rs` for overlay
   texture binding when no overlays are present.

4. **PIL Deprecation Warning** - Removed deprecated `mode` parameter from `Image.fromarray()`.

**Files Modified:**
- `src/viewer/ipc/protocol.rs` - Added overlay IPC request variants
- `src/viewer/terrain/overlay.rs` - Fixed RGBA data storage for image overlays
- `src/viewer/terrain/render.rs` - Fixed overlay view fallback logic
- `examples/swiss_terrain_landcover_viewer.py` - Fixed PIL deprecation warning

---

**Last Updated:** 2024-12-24  
**Script Version:** 1.3  
**Dependencies:** rasterio, Pillow (optional), forge3d ≥1.7.0  
**Tested On:** Windows 10/11, Python 3.8+
