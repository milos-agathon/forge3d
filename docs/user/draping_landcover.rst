Draping Categorical Rasters (Land-Cover)
==========================================

Overview
--------

The terrain draping pipeline allows you to overlay categorical raster data (like land-cover classifications) onto a DEM-displaced 3D terrain mesh. This is ideal for visualizing datasets like ESA WorldCover, NLCD, or other classification maps with proper 3D relief.

**Key Features:**

- DEM-only vertex displacement (no color data in height channel)
- Categorical texture sampling with nearest filtering (preserves class boundaries)
- Proper row-pitch alignment for WebGPU (256-byte padding)
- Y-flip handling for different coordinate systems
- Memory-efficient streaming for large datasets

Architecture
------------

The draping system consists of three main components:

1. **DEM Texture (R32Float/R16Float)**: Single-channel height data in meters, sampled with linear filtering in the vertex shader
2. **Land-Cover Texture (RGBA8)**: Categorical color data, sampled with nearest filtering in the fragment shader
3. **UV Transform**: Handles alignment, scaling, and Y-flip between different coordinate systems

Pipeline Flow
~~~~~~~~~~~~~

::

    [DEM Heights (f32)] ──> R32Float Texture ──> Vertex Shader (displacement)
                                                        │
                                                        ↓
    [Land-Cover (u8x4)] ──> RGBA8 Texture ────> Fragment Shader (color)
                                                        │
                                                        ↓
                                                  [Final RGBA8]

Common Pitfalls
---------------

1. **Row-Pitch Alignment**
   
   WebGPU requires texture row data to be aligned to 256 bytes. For odd widths, padding is automatically added:
   
   ::
   
      width = 10001 pixels
      bytes_per_row = 10001 × 4 = 40004 bytes
      padded_bytes_per_row = 40192 bytes (next 256-byte boundary)
   
   **Solution**: Use ``upload_rgba8_texture()`` or ``upload_r32f_texture()`` which handle padding automatically.

2. **Y-Flip Coordinate Systems**
   
   Image formats use different coordinate origins:
   
   - **Top-left origin** (PNG, TIFF): Y increases downward
   - **Bottom-left origin** (OpenGL, some GIS): Y increases upward
   
   **Solution**: Use ``UVTransform`` with ``y_flip=true`` when needed::
   
      from forge3d.core import UVTransform
      xform = UVTransform.with_y_flip()

3. **Categorical Sampling**
   
   Linear filtering blends adjacent pixels, creating intermediate colors not in your palette:
   
   ::
   
      Water (blue) + Forest (green) = Cyan (invalid class)
   
   **Solution**: Use ``SamplerBindingType::NonFiltering`` (nearest neighbor) for land-cover textures.

4. **DEM/Land-Cover Misalignment**
   
   If DEM and land-cover are at different resolutions or projections, features won't line up:
   
   ::
   
      DEM: 30m resolution, EPSG:4326
      Land-Cover: 10m resolution, EPSG:3857
      Result: Rivers don't follow valleys ❌
   
   **Solution**: Reproject land-cover to match DEM grid with nearest resampling::
   
      from rasterio.warp import reproject, Resampling
      reproject(
          source=lc_band,
          destination=lc_aligned,
          src_transform=lc_transform,
          dst_transform=dem_transform,
          resampling=Resampling.nearest,  # Preserve categories
      )

5. **Vertical Spikes**
   
   If RGBA land-cover data is uploaded to the height texture slot, vertices will be displaced by color values (0-255) instead of elevation:
   
   ::
   
      Red pixel (255, 0, 0, 255) → height = 255 meters → spike
   
   **Solution**: Ensure height texture is R32Float with actual elevation values.

Usage Example
-------------

Python API
~~~~~~~~~~

::

    import numpy as np
    from forge3d.io import read_dem, read_raster, resample_to
    from forge3d.renderer import drape_landcover
    
    # Load DEM (meters)
    dem = read_dem("swiss_dem.tif")  # (H, W) float32
    
    # Load land-cover and align to DEM grid
    lc = read_raster("swiss_landcover.tif")  # (H, W, 4) uint8
    lc_aligned = resample_to(lc, dem, method="nearest")
    
    # Render with draping
    img = drape_landcover(
        heightmap=dem,
        landcover=lc_aligned,
        height_scale=2.0,      # Vertical exaggeration
        y_flip=True,           # Handle coordinate flip
        size=(1920, 1080),
        camera_distance=None,  # Auto-calculate
    )
    
    # Save output
    from PIL import Image
    Image.fromarray(img).save("draped_terrain.png")

Performance Tips
----------------

**Memory Budget**

The GPU memory budget is **512 MiB** for host-visible allocations. For large datasets:

1. **Downsample** to target resolution::

      target_max_dim = 2048  # Limits max texture dimension
      scale = min(1.0, target_max_dim / max(height, width))
      
2. **Use R16Float** instead of R32Float for DEMs if precision allows::

      R32Float: 4 bytes/pixel
      R16Float: 2 bytes/pixel (50% memory savings)

3. **Tile streaming** for datasets > 8k×8k::

      from forge3d.terrain import enable_tile_streaming
      enable_tile_streaming(tile_size=512, max_in_flight=4)

**Rendering Speed**

- Reduce mesh resolution for preview: ``mesh_subsample=4``
- Use smaller output size: ``size=(1280, 720)``
- Disable anti-aliasing: ``sample_count=1``

Validation Checklist
--------------------

Before rendering, verify:

☑ DEM and land-cover have **same shape** ``(H, W)``
☑ DEM dtype is ``float32`` (meters or relative heights)
☑ Land-cover dtype is ``uint8`` with 4 channels (RGBA)
☑ Elevation range is reasonable: ``0 < (max - min) < 10000`` meters
☑ Memory estimate: ``H × W × 4 × 2 < 512 MiB`` (DEM + land-cover)
☑ Y-flip setting matches your coordinate system

Troubleshooting
---------------

**Black output**

- Check camera position and target are valid
- Verify heightmap contains valid values (not all NaN/Inf)
- Ensure land-cover alpha channel is not all zeros

**Banding/striping**

- Row-pitch padding issue - verify using correct upload functions
- Check texture width is not being truncated

**Wrong colors**

- Land-cover texture may be in wrong format (BGR vs RGB)
- Check if sRGB conversion is applied correctly
- Verify categorical palette matches data values

**Out of memory**

- Reduce texture resolution: downsample before upload
- Check for memory leaks: ensure textures are released
- Monitor: ``forge3d.renderer.get_memory_usage()``

Reference
---------

**Coordinate Systems**

- Mesh: Y-up ([X, Y, Z]), Y = height
- Triangle winding: Counter-clockwise (CCW) for front faces
- UV origin: (0, 0) = bottom-left or top-left (use y_flip)

**Texture Formats**

- DEM: ``R32Float`` (preferred) or ``R16Float``
- Land-Cover: ``Rgba8UnormSrgb`` (categorical)
- Output: ``Rgba8UnormSrgb``

**Sampling Modes**

- DEM: Linear (bilinear) for smooth terrain
- Land-Cover: Nearest (point) to preserve categories

See Also
--------

- :doc:`terrain_rendering` - General terrain rendering guide
- :doc:`../api/terrain` - Python API reference
- :doc:`../examples/switzerland_landcover` - Complete example
