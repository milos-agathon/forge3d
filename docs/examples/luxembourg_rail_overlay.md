# Luxembourg Terrain with Rail Network Overlay - User Guide

## File Location
`examples/luxembourg_rail_overlay.py`

## Overview

This example demonstrates the **Vector Overlay (Option B)** feature of the Forge3D engine. Unlike raster overlays (which are draped textures), vector overlays are rendered as actual 3D geometry. This allows for crisp, resolution-independent lines that can be draped onto the terrain surface with depth testing and halo effects.

The viewer renders the Luxembourg terrain with a vector overlay of the national rail network, utilizing high-quality PBR lighting, shadows, and material layering.

**Key Features:**
- **Vector-to-Mesh Conversion:** Converts GeoPackage line strings into 3D triangle strips (quads) for variable-width rendering.
- **Terrain Draping:** Automatically adjusts vector vertex height to match the underlying terrain DEM.
- **Depth-Correct Rendering:** Vector lines are properly occluded by terrain features (hills, mountains).
- **Halo Effects:** Adds outline halos to vector lines for better visibility against complex backgrounds.
- **Coordinate Reprojection:** Automatically reprojects vector data to EPSG:3035 to match the terrain system.
- **Material Layering:** Procedural snow, rock, and wetness layers based on slope and altitude.

## Quick Start

### Basic Usage

```bash
# Basic usage with default settings
python examples/luxembourg_rail_overlay.py --dem assets/tif/luxembourg_dem.tif \
    --gpkg assets/gpkg/luxembourg_rail.gpkg --snapshot output.png

# Custom line color (bright red) and width
python examples/luxembourg_rail_overlay.py \
    --line-color "#FF4444" \
    --line-width 25 \
    --snapshot red_rails.png
```

### Prerequisites

**Required Python Packages:**
- `geopandas` & `fiona`: For loading `.gpkg` vector files.
- `pyproj`: For coordinate reprojection.
- `rasterio`: For reading DEM metadata.
- `numpy`: For numerical operations.
- `Pillow` (PIL): For generating masking overlays.

```bash
pip install geopandas fiona pyproj rasterio numpy Pillow
```

**Build Requirements:**
- The `interactive_viewer` binary must be built:
  ```bash
  cargo build --release --bin interactive_viewer
  ```

## Vector Overlay System

This example uses "Option B" from the overlay implementation plan: **Vector Geometry Overlays**.

### How It Works
1.  **Load:** Reads vector data (LineStrings) from a GeoPackage.
2.  **Reproject:** Transforms coordinates to the target CRS (EPSG:3035).
3.  **Tessellate:** Generates a triangle strip (quads) for each line segment.
    *   Since WebGPU doesn't natively support wide lines, we generate geometry with a specific world-space width.
4.  **Drape:** The vertex shader (or CPU pre-process) samples the terrain height and offsets the vector geometry to sit just above the surface.

### Advantages over Raster Overlays
*   **Infinite Resolution:** Zooming in keeps edges crisp; no pixelation.
*   **Depth Testing:** Lines can disappear behind terrain ridges naturally.
*   **Styling:** Width and color can be changed dynamically without resampling textures.
*   **Halo/Outline:** Shader-based halo effects improve readability.

## Presets

The script includes presets optimized for different rendering styles:

### Default
```bash
python examples/luxembourg_rail_overlay.py --preset default
```
Balanced settings for interactive viewing.

### Alpine
```bash
python examples/luxembourg_rail_overlay.py --preset alpine
```
Adds snow (above 2500m) and rock layers on steep slopes. Good for emphasizing terrain morphology.

### Cinematic
```bash
python examples/luxembourg_rail_overlay.py --preset cinematic
```
Enables Depth of Field (DoF), motion blur, and lens effects (vignette, chromatic aberration) for a "filmic" look.

### High Quality
```bash
python examples/luxembourg_rail_overlay.py --preset high_quality
```
Maximizes all quality settings: MSAA 8x, PCSS soft shadows, high-res AO and sun visibility rays.

## Command-Line Options

### Vector Overlay Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--gpkg` | Path | `assets/gpkg/luxembourg_rail.gpkg` | Path to vector data file |
| `--line-width` | float | 15.0 | Line width in world units (meters) |
| `--line-color` | hex | `#E64A19` | Line color (e.g., `#FF0000`) |
| `--drape-offset` | float | 50.0 | Height offset above terrain to prevent z-fighting |
| `--overlay-depth` | flag | False | Enable depth testing (lines occluded by terrain) |
| `--overlay-halo` | flag | False | Enable halo outline around lines |
| `--overlay-halo-width` | float | 2.0 | Width of the halo in pixels |

### Terrain & PBR Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dem` | Path | `assets/tif/luxembourg_dem.tif` | Path to DEM GeoTIFF |
| `--z-scale` | float | 0.15 | Vertical exaggeration factor |
| `--pbr` / `--no-pbr` | flag | True | Enable Physically Based Rendering |
| `--shadows` | choice | `pcss` | Shadow mode: `none`, `hard`, `pcf`, `pcss` |
| `--height-ao` | flag | True | Enable Ray-Traced Ambient Occlusion |
| `--sun-vis` | flag | True | Enable Ray-Traced Sun Visibility (Shadows) |

### Material Layering (M4)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--snow` | flag | False | Enable procedural snow layer |
| `--snow-altitude` | float | 2500.0 | Min altitude for snow |
| `--rock` | flag | False | Enable rock layer on steep slopes |
| `--wetness` | flag | False | Enable wetness accumulation in valleys |

### Camera & Post-Processing

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dof` | flag | False | Enable Depth of Field |
| `--lens-effects` | flag | False | Enable Vignette, CA, Distortion |
| `--tonemap` | choice | `aces` | Tonemapping operator |
| `--exposure` | float | 1.2 | Exposure multiplier |

## Interactive Controls

When running interactively, the terminal accepts these commands:

```text
# Overlay Control
overlay on/off           # Toggle rail overlay visibility
overlay opacity=0.8      # Set overlay opacity

# Camera
set phi=135 theta=35 radius=3000   # Orbit camera

# Lighting
set sun_az=180 sun_el=45           # Move sun position

# Snapshot
snap output.png                    # Save screenshot
snap output.png 3840x2160          # Save 4K screenshot
```

## Technical Details

### Coordinate System
The viewer uses **EPSG:3035 (ETRS89-LAEA)**.
- If the DEM or Vector data is in a different CRS (like EPSG:4326), the script uses `pyproj` to reproject bounds and vertices automatically.
- The terrain is centered at `(0,0,0)` in world space.

### Geometry Generation
The function `load_gpkg_lines` reads the GeoPackage and calls `_add_linestring_as_quads`.
1.  **Tangent Calculation:** For every point in the line string, the tangent direction is calculated.
2.  **Extrusion:** Vertices are extruded perpendicular to the tangent by `line_width / 2`.
3.  **Strip Construction:** A Triangle Strip is created connecting the left and right extruded vertices.

## Troubleshooting

### "geopandas is required"
The script falls back to a demo grid pattern if `geopandas` is missing. Install it via pip:
```bash
pip install geopandas fiona
```
*Note: On Windows, installing `fiona` and `gdal` can be tricky; using `conda` or pre-built wheels is recommended.*

### "No feature table found in GPKG"
Ensure your `.gpkg` file is valid and contains a vector layer. You can inspect it with `fiona.listlayers('file.gpkg')`.

### Lines look broken or z-fighting
If lines flicker or disappear into the terrain:
1.  Increase `--drape-offset` (e.g., `--drape-offset 100`).
2.  Enable `--overlay-depth-bias` to nudge depth values.
3.  If using `--overlay-depth`, lines behind mountains will correctly disappear. If you want them always visible, remove this flag (default is off).

### Performance is low
- **Reduce Shadow Quality:** Use `--shadows pcf` instead of `pcss`.
- **Disable AO:** `--no-height-ao`.
- **Reduce Line Count:** The script has a `--max-vertices` limit (default 20000). For massive networks, this may need adjustment or decimation.
