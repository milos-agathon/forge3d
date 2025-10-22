"""
Terrain rendering utilities for forge3d.

Provides high-level functions for terrain visualization with DEM displacement
and categorical land-cover texture draping.
"""

from typing import Optional, Tuple, Sequence
import numpy as np
from PIL import Image

from . import _forge3d


def drape_landcover(
    heightmap: np.ndarray,
    landcover: np.ndarray,
    *,
    aoi=None,
    render_mode: str = "raster",
    rt_spp: int = 64,
    rt_max_bounces: int = 2,
    rt_seed: int = 0,
    rt_batch_spp: int = 8,
    rt_sampling_mode: str = "sobol",
    rt_debug_mode: int = 0,
    max_rt_triangles: int = 2000000,
    max_dim: int = 3000,
    rt_mem_limit_mib: int = 512,
    rt_auto_tile: bool = True,
    rt_tile_min: Tuple[int, int] = (96, 96),
    rt_tile_max: Tuple[int, int] = (512, 512),
    rt_accum_format: str = "rgba16f",
    oidn_mode: str = "final",
    width: int = 1280,
    height: int = 720,
    z_dir: float = 1.0,
    zscale: float = 1.0,
    pixel_spacing: Optional[Tuple[float, float]] = None,
    camera_distance: Optional[float] = None,
    camera_theta: float = 45.0,
    camera_phi: float = 25.0,
    camera_gamma: float = 0.0,
    camera_fov: float = 35.0,
    light_type: str = "directional",
    light_elevation: float = 45.0,
    light_azimuth: float = 315.0,
    light_intensity: float = 1.0,
    ambient: float = 0.25,
    shadow_intensity: float = 0.6,
    lighting_model: str = "blinn_phong",
    shininess: float = 32.0,
    specular_strength: float = 0.3,
    shadow_softness: float = 2.0,
    shadow_map_res: int = 2048,
    shadow_bias: float = 0.0015,
    enable_shadows: bool = True,
    hdri: Optional[str] = None,
    hdri_intensity: float = 1.0,
    hdri_rotation_deg: float = 0.0,
    background: str = "#FFFFFF",
    y_flip: bool = False,
    denoiser: str = "oidn",
    denoise_strength: float = 0.8,
    # AOV controls
    save_aovs: bool = False,
    aovs: Sequence[str] = ("albedo", "normal", "depth", "visibility"),
    aov_dir: Optional[str] = None,
    basename: Optional[str] = None,
) -> np.ndarray:
    """
    Render terrain with DEM displacement and categorical land-cover texture draping.
    
    This function creates a 3D visualization where:
    - Vertices are displaced by the DEM heightmap (vertex shader)
    - Categorical colors are sampled from the land-cover texture (fragment shader)
    - Configurable lighting provides 3D relief
    - Pixels outside valid data (NoData) are transparent
    
    Parameters
    ----------
    heightmap : np.ndarray
        DEM elevation data, shape (H, W), dtype float32, in meters.
        Elevations can be absolute or relative (recommend subtracting minimum).
    landcover : np.ndarray
        Categorical land-cover colors, shape (H, W, 4), dtype uint8 (RGBA).
        Must have the same H, W dimensions as heightmap.
    aoi : optional
        Area of interest mask or polygon (not yet implemented).
    render_mode : str, default="raster"
        Rendering mode: "raster" (GPU rasterization, fast) or "raytrace" (GPU path tracing, high quality).
    rt_spp : int, default=64
        Ray tracing samples per pixel (only used when render_mode="raytrace"). Higher values reduce noise.
    rt_max_bounces : int, default=2
        Maximum light bounces for ray tracing (only used when render_mode="raytrace").
    rt_seed : int, default=0
        Random seed for ray tracing (only used when render_mode="raytrace").
    rt_sampling_mode : str, default="sobol"
        Sampling mode for ray tracing: "rng" (pseudo-random), "sobol" (low-discrepancy, best quality),
        or "cmj" (correlated multi-jittered). Only used when render_mode="raytrace".
    max_dim : int, default=3000
        Maximum dimension for input rasters. If max(H, W) > max_dim,
        both rasters will be resampled to preserve aspect ratio.
    width : int, default=1280
        Output image width in pixels.
    height : int, default=720
        Output image height in pixels.
    z_dir : float, default=1.0
        Height displacement direction: +1.0 = outward/up, -1.0 = inward/down.
    zscale : float, default=1.0
        Vertical scale factor (vertical exaggeration).
    camera_distance : Optional[float], default=None
        Camera distance from terrain center. If None, auto-calculated.
    camera_theta : float, default=45.0
        Camera azimuth angle in degrees (0=North, 90=East, 180=South, 270=West).
    camera_phi : float, default=25.0
        Camera elevation angle in degrees (0=horizon, 90=directly above).
    camera_gamma : float, default=0.0
        Camera roll in degrees (rotation around view axis).
    camera_fov : float, default=35.0
        Camera field of view in degrees.
    light_type : str, default="directional"
        Lighting type: "none", "directional", or "hemisphere".
    light_elevation : float, default=45.0
        Light elevation angle in degrees above horizon.
    light_azimuth : float, default=315.0
        Light azimuth in degrees from +X axis (clockwise).
    light_intensity : float, default=1.0
        Key light strength multiplier.
    ambient : float, default=0.25
        Ambient lighting term (minimum brightness).
    shadow_intensity : float, default=0.6
        Shadow mix factor (0=no shadows, 1=full shadows).
    shadow_softness : float, default=2.0
        PCF kernel radius for soft shadows (1.0-5.0). Higher = softer shadows.
    shadow_map_res : int, default=2048
        Shadow map resolution in pixels (512-4096). Higher = better quality.
    shadow_bias : float, default=0.0015
        Depth bias to prevent shadow acne (0.0001-0.01).
    enable_shadows : bool, default=True
        Enable or disable shadow rendering.
    hdri : Optional[str], default=None
        Path to HDRI environment map file (.hdr or .exr). If None, uses constant ambient lighting.
    hdri_intensity : float, default=1.0
        HDRI environment intensity multiplier (0-2+). Higher values = brighter ambient.
    hdri_rotation_deg : float, default=0.0
        Y-axis rotation of HDRI environment in degrees (0-360). Rotates lighting direction.
    lighting_model : str, default="blinn_phong"
        Lighting model: "lambert" (simple), "phong" (view-dependent), or "blinn_phong" (HQ).
    shininess : float, default=32.0
        Phong/Blinn-Phong specular exponent (1-256). Higher = tighter highlights.
    specular_strength : float, default=0.3
        Specular reflection strength (0-1). Controls glossiness.
    background : str, default="#FFFFFF"
        Background color as hex string (e.g., "#FFFFFF" for white, "#00000000" for transparent).
    y_flip : bool, default=False
        If True, flip texture coordinates in Y (for inverted coordinate systems).
    denoiser : str, default="oidn"
        Denoising method: "none" (no denoising), "oidn" (Intel Open Image Denoise),
        or "bilateral" (fast edge-preserving filter). OIDN provides best quality but
        requires OIDN library; falls back to bilateral if unavailable.
    denoise_strength : float, default=0.8
        Denoising intensity (0.0-1.0). Higher values = more aggressive denoising.
        0.0 = passthrough, 1.0 = maximum denoising.
    pixel_spacing : Optional[Tuple[float, float]]
        Horizontal spacing (sx, sy) in meters per pixel for the heightmap grid. If provided,
        the mesh will be generated in metric units for X/Z, and vertical scale will be applied
        directly to elevation meters for consistent proportions. When the heightmap is decimated
        to respect the triangle budget, the spacing will be scaled accordingly so the physical
        extent remains unchanged.
    
    Returns
    -------
    np.ndarray
        Rendered RGBA image, shape (height, width, 4), dtype uint8.
    
    Raises
    ------
    ValueError
        If heightmap is not 2D, landcover is not 3D RGBA, or dimensions don't match.
    RuntimeError
        If GPU rendering fails.
    
    Examples
    --------
    >>> import numpy as np
    >>> from forge3d.terrain import drape_landcover
    >>> 
    >>> # Load DEM (example: 512x512 terrain)
    >>> dem = np.random.rand(512, 512).astype(np.float32) * 1000  # Heights in meters
    >>> 
    >>> # Create land-cover (example: green terrain with blue water)
    >>> landcover = np.zeros((512, 512, 4), dtype=np.uint8)
    >>> landcover[:, :, 1] = 180  # Green channel
    >>> landcover[:, :, 3] = 255  # Alpha
    >>> landcover[:256, :, :] = [30, 144, 255, 255]  # Blue water in top half
    >>> 
    >>> # Render with 2x vertical exaggeration
    >>> img = drape_landcover(dem, landcover, height_scale=2.0, width=1920, height=1080)
    >>> 
    >>> # Save result
    >>> from PIL import Image
    >>> Image.fromarray(img.reshape(1080, 1920, 4)).save("terrain.png")
    
    Notes
    -----
    - DEM and land-cover must be aligned to the same grid (use resampling if needed)
    - Use nearest-neighbor resampling for land-cover to preserve categorical values
    - GPU memory usage is approximately: (H * W * 4 * 2) bytes for textures
    - For datasets > 4k x 4k, consider downsampling to improve performance
    
    See Also
    --------
    forge3d.io.read_dem : Load DEM from GeoTIFF
    forge3d.io.resample_to : Align rasters to the same grid
    """
    # Validate inputs
    if not isinstance(heightmap, np.ndarray):
        raise TypeError(f"heightmap must be np.ndarray, got {type(heightmap)}")
    if not isinstance(landcover, np.ndarray):
        raise TypeError(f"landcover must be np.ndarray, got {type(landcover)}")
    
    if heightmap.ndim != 2:
        raise ValueError(f"heightmap must be 2D (H, W), got shape {heightmap.shape}")
    if landcover.ndim != 3 or landcover.shape[2] != 4:
        raise ValueError(f"landcover must be 3D (H, W, 4) RGBA, got shape {landcover.shape}")
    
    if heightmap.shape != landcover.shape[:2]:
        raise ValueError(
            f"heightmap and landcover must have same H, W dimensions: "
            f"heightmap={heightmap.shape}, landcover={landcover.shape[:2]}"
        )
    
    # Validate render_mode
    render_mode_lower = render_mode.lower()
    if render_mode_lower not in ["raster", "raytrace"]:
        raise ValueError(f"render_mode must be 'raster' or 'raytrace', got '{render_mode}'")
    
    # Ensure correct dtypes
    if heightmap.dtype != np.float32:
        heightmap = heightmap.astype(np.float32)
    if landcover.dtype != np.uint8:
        landcover = landcover.astype(np.uint8)
    
    # Automatic resampling to max_dim
    h, w = heightmap.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        import warnings
        warnings.warn(
            f"Resampling from {h}x{w} to {new_h}x{new_w} (max_dim={max_dim}, scale={scale:.3f})",
            UserWarning
        )
        
        # Resample DEM with bilinear interpolation
        from scipy.ndimage import zoom
        heightmap = zoom(heightmap, (new_h / h, new_w / w), order=1)
        
        # Resample landcover with nearest-neighbor to preserve categorical values
        # Process each channel separately
        landcover_resampled = np.zeros((new_h, new_w, 4), dtype=np.uint8)
        for c in range(4):
            landcover_resampled[:, :, c] = zoom(landcover[:, :, c], (new_h / h, new_w / w), order=0)
        landcover = landcover_resampled
    
    # Ray tracing mode: convert terrain to mesh and use path tracer
    if render_mode_lower == "raytrace":
        from .render import heightmap_to_mesh, render_raytrace_mesh
        from pathlib import Path
        import time
        
        print(f"[RAYTRACE] Preparing terrain mesh for ray tracing...")
        t_start = time.time()
        
        # OPTIMIZATION: Decimate heightmap for ray tracing to reduce triangle count
        # Each grid cell produces 2 triangles, so H×W grid = 2×(H-1)×(W-1) triangles
        h_rt, w_rt = heightmap.shape
        # Track decimation scaling so we can preserve physical pixel spacing
        scale_decim_y = 1.0
        scale_decim_x = 1.0
        current_tris = 2 * (h_rt - 1) * (w_rt - 1)
        
        if current_tris > max_rt_triangles:
            # Calculate scale factor needed to hit target triangle count
            # tris = 2 * (H-1) * (W-1), solve for scale: new_tris = 2 * (scale*H-1) * (scale*W-1)
            target_tris = max_rt_triangles
            scale_factor = np.sqrt(target_tris / current_tris)
            new_h = max(64, int(h_rt * scale_factor))  # Minimum 64 to avoid over-decimation
            new_w = max(64, int(w_rt * scale_factor))
            
            print(f"[RAYTRACE] Decimating mesh: {h_rt}x{w_rt} -> {new_h}x{new_w} ({scale_factor:.3f}x)")
            print(f"[RAYTRACE] Triangle count: {current_tris:,} -> ~{2*(new_h-1)*(new_w-1):,}")
            
            from scipy.ndimage import zoom
            heightmap_rt = zoom(heightmap, (new_h / h_rt, new_w / w_rt), order=1)  # Bilinear
            
            # Also decimate landcover with nearest-neighbor
            landcover_rt = np.zeros((new_h, new_w, 4), dtype=np.uint8)
            for c in range(4):
                landcover_rt[:, :, c] = zoom(landcover[:, :, c], (new_h / h_rt, new_w / w_rt), order=0)
            # Update spacing scale to preserve physical size
            scale_decim_y = h_rt / new_h
            scale_decim_x = w_rt / new_w
        else:
            print(f"[RAYTRACE] Mesh within triangle budget: {current_tris:,} triangles")
            heightmap_rt = heightmap
            landcover_rt = landcover
        
        # Convert heightmap to mesh with landcover vertex colors
        # Determine horizontal spacing (meters per pixel) and adjust for decimation
        # Default to unit spacing if not provided
        sx = float(pixel_spacing[0]) if (pixel_spacing is not None) else 1.0
        sy = float(pixel_spacing[1]) if (pixel_spacing is not None) else 1.0
        # Apply decimation scaling so the mesh retains the correct physical extent
        sx *= float(scale_decim_x)
        sy *= float(scale_decim_y)
        spacing = (sx, sy)
        vertices, indices = heightmap_to_mesh(heightmap_rt, spacing, vertical_scale=float(zscale * z_dir))
        
        n_tris = len(indices)
        print(f"[RAYTRACE] Mesh: {len(vertices):,} vertices, {n_tris:,} triangles")
        print(f"[RAYTRACE] Prep time: {time.time() - t_start:.2f}s")
        
        # Map landcover colors to vertices
        # Vertices are in row-major order, matching the heightmap grid
        h_mesh, w_mesh = heightmap_rt.shape
        landcover_flat = landcover_rt.reshape(-1, 4)  # Flatten to (H*W, 4)
        vertex_colors = landcover_flat.astype(np.float32) / 255.0  # Convert to float [0,1]
        
        # Combine positions with colors: (N, 3) + (N, 4) -> (N, 8) [x,y,z,pad,r,g,b,pad2]
        n_verts = vertices.shape[0]
        vertices_colored = np.zeros((n_verts, 8), dtype=np.float32)
        vertices_colored[:, :3] = vertices  # positions
        vertices_colored[:, 4:7] = vertex_colors[:, :3]  # RGB from landcover
        
        # Parse background color for ray tracer
        def parse_hex_color(hex_str: str) -> Tuple[float, float, float]:
            hex_str = hex_str.lstrip('#')
            if len(hex_str) == 6:
                r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
                return r/255.0, g/255.0, b/255.0
            elif len(hex_str) == 8:
                r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
                return r/255.0, g/255.0, b/255.0
            else:
                return 1.0, 1.0, 1.0
        
        bg_color = parse_hex_color(background)
        
        # Call ray tracer with terrain mesh
        # Note: render_raytrace_mesh expects orbit angles and handles camera setup
        print(f"[RAYTRACE] Starting ray tracing: {width}x{height} @ {rt_spp} spp...")
        t_render = time.time()
        
        result, meta = render_raytrace_mesh(
            (vertices_colored, indices),
            size=(width, height),
            frames=rt_spp,
            seed=rt_seed,
            sampling_mode=rt_sampling_mode,
            debug_mode=rt_debug_mode,
            fov_y=camera_fov,
            orbit_theta=camera_theta,
            orbit_phi=camera_phi,
            radius=camera_distance,
            prefer_gpu=True,
            normalize=True,
            normalize_scale=1.6,
            denoiser=denoiser if denoiser != "none" else "off",
            lighting_type="lambertian",
            lighting_intensity=light_intensity,
            lighting_azimuth=light_azimuth,
            lighting_elevation=light_elevation,
            shadows=enable_shadows,
            shadow_intensity=shadow_intensity,
            hdri_path=Path(hdri) if hdri else None,
            hdri_rotation_deg=hdri_rotation_deg,
            hdri_intensity=hdri_intensity,
            background_color=bg_color,
            save_aovs=bool(save_aovs),
            aovs=tuple(aovs) if aovs is not None else (),
            aov_dir=aov_dir,
            basename=basename,
            verbose=True,
        )
        
        print(f"[RAYTRACE] Ray tracing complete: {time.time() - t_render:.2f}s")
        print(f"[RAYTRACE] Total time: {time.time() - t_start:.2f}s")
        
        return result
    
    # Validate ranges
    if not np.isfinite(heightmap).all():
        raise ValueError("heightmap contains non-finite values (NaN or Inf)")
    
    h_min, h_max = heightmap.min(), heightmap.max()
    if h_max - h_min > 20000:
        import warnings
        warnings.warn(
            f"Heightmap elevation range is {h_max - h_min:.1f} meters. "
            "Consider using relative heights (subtract minimum) to avoid scale issues.",
            UserWarning
        )
    
    # Convert light_type string to integer
    light_type_map = {"none": 0, "directional": 1, "hemisphere": 2}
    light_type_int = light_type_map.get(light_type.lower(), 1)
    
    # Convert lighting_model string to integer
    lighting_model_map = {"lambert": 0, "phong": 1, "blinn_phong": 2}
    lighting_model_int = lighting_model_map.get(lighting_model.lower(), 2)
    
    # Parse background color from hex string
    def parse_hex_color(hex_str: str) -> Tuple[float, float, float, float]:
        """Parse hex color string to RGBA floats (0-1 range)."""
        hex_str = hex_str.lstrip('#')
        if len(hex_str) == 6:
            # RGB, assume opaque
            r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
            return r/255.0, g/255.0, b/255.0, 1.0
        elif len(hex_str) == 8:
            # RGBA
            r, g, b, a = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16), int(hex_str[6:8], 16)
            return r/255.0, g/255.0, b/255.0, a/255.0
        else:
            raise ValueError(f"Invalid hex color format: {hex_str}. Expected 6 or 8 hex digits.")
    
    bg_r, bg_g, bg_b, bg_a = parse_hex_color(background)
    
    # Call Rust implementation
    result_flat = _forge3d.terrain_drape_render(
        heightmap,
        landcover,
        width=width,
        height=height,
        z_dir=z_dir,
        zscale=zscale,
        camera_distance=camera_distance,
        camera_theta=camera_theta,
        camera_phi=camera_phi,
        camera_gamma=camera_gamma,
        camera_fov=camera_fov,
        light_type=light_type_int,
        light_elevation=light_elevation,
        light_azimuth=light_azimuth,
        light_intensity=light_intensity,
        ambient=ambient,
        shadow_intensity=shadow_intensity,
        lighting_model=lighting_model_int,
        shininess=shininess,
        specular_strength=specular_strength,
        shadow_softness=shadow_softness,
        background_r=bg_r,
        background_g=bg_g,
        background_b=bg_b,
        background_a=bg_a,
        shadow_map_res=shadow_map_res,
        shadow_bias=shadow_bias,
        enable_shadows=enable_shadows,
        hdri_path=hdri,
        hdri_intensity=hdri_intensity,
        hdri_rotation_deg=hdri_rotation_deg,
        y_flip=y_flip,
        denoiser=denoiser,
        denoise_strength=denoise_strength,
    )
    
    # Reshape to (H, W, 4)
    result = result_flat.reshape(height, width, 4)
    
    return result


def render_terrain_preview(
    heightmap: np.ndarray,
    landcover: np.ndarray,
    output_path: str,
    *,
    width: int = 1280,
    height: int = 720,
    height_scale: float = 1.0,
    camera_theta: float = 45.0,
    camera_phi: float = 25.0,
) -> None:
    """
    Render terrain and save directly to file (convenience wrapper).
    
    Parameters
    ----------
    heightmap : np.ndarray
        DEM elevation data, shape (H, W), dtype float32.
    landcover : np.ndarray
        Categorical land-cover colors, shape (H, W, 4), dtype uint8.
    output_path : str
        Path to save output image (PNG recommended).
    width : int, default=1280
        Output width in pixels.
    height : int, default=720
        Output height in pixels.
    height_scale : float, default=1.0
        Vertical exaggeration.
    camera_theta : float, default=45.0
        Camera azimuth in degrees.
    camera_phi : float, default=25.0
        Camera elevation in degrees.
    
    Examples
    --------
    >>> from forge3d.terrain import render_terrain_preview
    >>> render_terrain_preview(dem, landcover, "preview.png", width=1920, height=1080)
    """
    img = drape_landcover(
        heightmap,
        landcover,
        width=width,
        height=height,
        height_scale=height_scale,
        camera_theta=camera_theta,
        camera_phi=camera_phi,
    )
    
    Image.fromarray(img).save(output_path)
    print(f"✓ Saved terrain preview to {output_path}")


def estimate_memory_usage(heightmap_shape: Tuple[int, int]) -> dict:
    """
    Estimate GPU memory usage for terrain draping.
    
    Parameters
    ----------
    heightmap_shape : Tuple[int, int]
        Shape of heightmap (H, W).
    
    Returns
    -------
    dict
        Memory estimates in bytes for different components.
    
    Examples
    --------
    >>> from forge3d.terrain import estimate_memory_usage
    >>> estimate_memory_usage((4096, 4096))
    {'heightmap_r32f': 67108864, 'landcover_rgba8': 67108864, 'total_textures': 134217728, 'total_mb': 128.0}
    """
    h, w = heightmap_shape
    pixels = h * w
    
    heightmap_bytes = pixels * 4  # R32Float
    landcover_bytes = pixels * 4  # RGBA8
    total_bytes = heightmap_bytes + landcover_bytes
    
    return {
        "heightmap_r32f": heightmap_bytes,
        "landcover_rgba8": landcover_bytes,
        "total_textures": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "exceeds_512mb_budget": total_bytes > 512 * 1024 * 1024,
    }


__all__ = [
    "drape_landcover",
    "render_terrain_preview",
    "estimate_memory_usage",
]
