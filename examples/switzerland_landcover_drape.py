#!/usr/bin/env python3
"""
Switzerland 3D Land Cover Terrain Draping Example

Demonstrates the terrain draping pipeline:
1. Fetch DEM and ESA WorldCover land-cover data for Switzerland
2. Align both rasters to the same grid
3. Render using GPU terrain draping (DEM displacement + categorical texture sampling)

Usage:
    python switzerland_landcover_drape.py --width 1920 --height 1080 --output terrain_draped.png
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask as rio_mask
import geopandas as gpd
from PIL import Image

try:
    import forge3d
    from forge3d.terrain import drape_landcover, estimate_memory_usage
    import planetary_computer
    import pystac_client
    import rioxarray
    import xarray as xr
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install forge3d pystac-client planetary-computer rioxarray geopandas xarray")
    sys.exit(1)


# ESA WorldCover 2021 color mapping (categorical)
ESA_COLORS = {
    10: "#006400",  # Tree cover
    20: "#ffbb22",  # Shrubland
    30: "#ffff4c",  # Grassland
    40: "#f096ff",  # Cropland
    50: "#fa0000",  # Built-up
    60: "#b4b4b4",  # Bare / sparse vegetation
    70: "#f0f0f0",  # Snow and ice
    80: "#0064c8",  # Permanent water bodies
    90: "#0096a0",  # Herbaceous wetland
    95: "#00cf75",  # Mangroves
    100: "#fae6a0", # Moss and lichen
    0: "#00000000", # NoData (transparent)
}


def hex_to_rgba(hex_color):
    """Convert hex color to RGBA tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 8:  # RGBA
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4, 6))
    else:  # RGB, add full alpha
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) + (255,)


def get_country_boundary(country_code: str) -> gpd.GeoDataFrame:
    """Fetch country boundary from EU GISCO service."""
    import ssl
    import urllib.request
    
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
    urllib.request.install_opener(opener)
    
    url = "https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson/CNTR_RG_01M_2020_4326.geojson"
    gdf = gpd.read_file(url)
    country = gdf[gdf['CNTR_ID'] == country_code]
    if country.empty:
        raise ValueError(f"Country code '{country_code}' not found")
    return country


def fetch_worldcover(geometry, max_dim=1200):
    """Fetch ESA WorldCover 2021 land-cover data."""
    print("📡 Fetching ESA WorldCover 2021 land-cover...")
    
    # Get bounding box from geometry
    bounds = geometry.bounds
    minx, miny, maxx, maxy = bounds
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=[minx, miny, maxx, maxy],
    )
    items = list(search.items())
    
    if not items:
        raise RuntimeError("No WorldCover data found for this region")
    
    print(f"   Found {len(items)} WorldCover tile(s)")
    
    # Load all items using rioxarray and mosaic
    import xarray as xr
    from rioxarray.merge import merge_arrays
    
    datasets = []
    for item in items:
        signed_href = planetary_computer.sign(item.assets["map"].href)
        ds = rioxarray.open_rasterio(signed_href, masked=True)
        ds = ds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        datasets.append(ds)
    
    # Merge all datasets
    if len(datasets) == 1:
        data = datasets[0]
    else:
        try:
            data = merge_arrays(datasets)
        except Exception:
            # Fallback: align to first dataset grid
            base = datasets[0]
            filled = base.copy()
            for ds in datasets[1:]:
                ds_r = ds.rio.reproject_match(base)
                mask = filled.isnull()
                filled = xr.where(mask, ds_r, filled)
            data = filled
    
    # Clip to country boundary
    data = data.rio.clip([geometry], all_touched=True)
    
    # Extract array
    landcover = data.values[0] if data.values.ndim == 3 else data.values
    landcover = landcover.astype(np.uint8)
    transform = data.rio.transform()
    crs = data.rio.crs
    
    print(f"   Land-cover shape: {landcover.shape}, dtype: {landcover.dtype}")
    
    # Validate shape
    if landcover.size == 0 or landcover.ndim != 2:
        raise RuntimeError(f"Land-cover loading failed: got shape {landcover.shape}")
    
    if landcover.shape[0] < 10 or landcover.shape[1] < 10:
        raise RuntimeError(f"Land-cover too small: {landcover.shape}. Check geometry and data availability.")
    
    # Downsample if too large
    h, w = landcover.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        from scipy.ndimage import zoom
        landcover = zoom(landcover, (new_h / h, new_w / w), order=0)  # Nearest neighbor
        
        # Adjust transform
        transform = transform * rasterio.Affine.scale(w / new_w, h / new_h)
        print(f"   Downsampled from {h}x{w} to {landcover.shape[0]}x{landcover.shape[1]}")
    
    return landcover, transform, crs


def fetch_dem(geometry, max_dim=1200):
    """Fetch DEM from Planetary Computer (Copernicus DEM 30m)."""
    print("📡 Fetching Copernicus DEM 30m...")
    
    # Get bounding box from geometry
    bounds = geometry.bounds
    minx, miny, maxx, maxy = bounds
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    search = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=[minx, miny, maxx, maxy],
    )
    items = list(search.items())
    
    if not items:
        raise RuntimeError("No DEM data found for this region")
    
    print(f"   Found {len(items)} DEM tile(s)")
    
    # Load all items using rioxarray and mosaic
    import xarray as xr
    from rioxarray.merge import merge_arrays
    
    datasets = []
    for item in items:
        signed_href = planetary_computer.sign(item.assets["data"].href)
        ds = rioxarray.open_rasterio(signed_href, masked=True)
        ds = ds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        datasets.append(ds)
    
    # Merge all datasets
    if len(datasets) == 1:
        data = datasets[0]
    else:
        try:
            data = merge_arrays(datasets)
        except Exception:
            # Fallback: reproject to first dataset grid
            base = datasets[0]
            filled = base.copy()
            for ds in datasets[1:]:
                ds_r = ds.rio.reproject_match(base)
                mask = filled.isnull()
                filled = xr.where(mask, ds_r, filled)
            data = filled
    
    # Clip to country boundary
    data = data.rio.clip([geometry], all_touched=True)
    
    # Extract array
    dem = data.values[0] if data.values.ndim == 3 else data.values
    dem = dem.astype(np.float32)
    transform = data.rio.transform()
    crs = data.rio.crs
    
    print(f"   DEM shape: {dem.shape}, dtype: {dem.dtype}")
    print(f"   Elevation range: {np.nanmin(dem):.1f}m to {np.nanmax(dem):.1f}m")
    
    # Validate shape
    if dem.size == 0 or dem.ndim != 2:
        raise RuntimeError(f"DEM loading failed: got shape {dem.shape}")
    
    if dem.shape[0] < 10 or dem.shape[1] < 10:
        raise RuntimeError(f"DEM too small: {dem.shape}. Check geometry and data availability.")
    
    # Fill nodata (ocean = 0)
    dem = np.nan_to_num(dem, nan=0.0)
    
    # Downsample if too large
    h, w = dem.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        from scipy.ndimage import zoom
        dem = zoom(dem, (new_h / h, new_w / w), order=1)  # Bilinear for DEM
        
        transform = transform * rasterio.Affine.scale(w / new_w, h / new_h)
        print(f"   Downsampled from {h}x{w} to {dem.shape[0]}x{dem.shape[1]}")
    
    return dem, transform, crs


def align_rasters(dem, dem_transform, dem_crs, landcover, lc_transform, lc_crs):
    """Align land-cover to DEM grid using nearest-neighbor resampling."""
    print("🔄 Aligning land-cover to DEM grid...")
    
    h, w = dem.shape
    lc_h, lc_w = landcover.shape
    
    print(f"   DEM shape: {dem.shape}, Land-cover shape: {landcover.shape}")
    print(f"   DEM CRS: {dem_crs}, Land-cover CRS: {lc_crs}")
    
    # If already same dimensions and CRS, just return
    if (h, w) == (lc_h, lc_w) and dem_crs == lc_crs:
        print(f"   Already aligned (same CRS and dimensions)")
        return landcover
    
    # If same CRS but different resolution, use scipy zoom
    if dem_crs == lc_crs:
        print(f"   Resizing from {landcover.shape} to {dem.shape}...")
        from scipy.ndimage import zoom
        scale_y = h / lc_h
        scale_x = w / lc_w
        lc_aligned = zoom(landcover, (scale_y, scale_x), order=0)  # Nearest neighbor
    else:
        # Different CRS - need reprojection
        print(f"   Reprojecting from {lc_crs} to {dem_crs}...")
        lc_aligned = np.zeros((h, w), dtype=landcover.dtype)
        
        reproject(
            source=landcover,
            destination=lc_aligned,
            src_transform=lc_transform,
            src_crs=lc_crs,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=Resampling.nearest,
        )
    
    print(f"   Aligned shape: {lc_aligned.shape}")
    return lc_aligned


def classes_to_rgba(landcover_classes):
    """Convert land-cover class IDs to RGBA colors."""
    h, w = landcover_classes.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Map each class to its color
    for class_id, hex_color in ESA_COLORS.items():
        mask = landcover_classes == class_id
        rgba[mask] = hex_to_rgba(hex_color)
    
    # Default color for unmapped classes
    unmapped = np.all(rgba == 0, axis=2)
    rgba[unmapped] = [128, 128, 128, 255]  # Gray
    
    return rgba


def main():
    parser = argparse.ArgumentParser(description="Switzerland 3D Land Cover Terrain Draping")
    parser.add_argument("--country", default="CH", help="Country code (default: CH for Switzerland)")
    parser.add_argument("--output", default="terrain_draped.png", help="Output image path")
    parser.add_argument("--width", type=int, default=1920, help="Output width")
    parser.add_argument("--height", type=int, default=1080, help="Output height")
    parser.add_argument("--render-mode", default="raster", choices=["raster", "raytrace"], 
                        help="Rendering mode: raster (fast GPU rasterization) or raytrace (high quality path tracing)")
    parser.add_argument("--rt-spp", type=int, default=64, help="Ray tracing samples per pixel (raytrace mode only)")
    parser.add_argument("--rt-seed", type=int, default=0, help="Random seed for ray tracing (raytrace mode only)")
    parser.add_argument("--rt-batch-spp", type=int, default=8, help="Ray tracing batch size for progressive rendering (raytrace mode only)")
    parser.add_argument("--rt-sampling-mode", default="sobol", choices=["rng", "sobol", "cmj"], 
                        help="Sampling mode: rng (fast), sobol (best quality, recommended), cmj (alternative)")
    parser.add_argument("--max-rt-triangles", type=int, default=2000000, help="Max triangles for ray tracing mesh (will decimate if needed)")
    parser.add_argument("--rt-debug-mode", type=int, default=0, help="Ray tracing debug mode (0=normal, 1=grid, 2=green sentinel)")
    parser.add_argument("--zscale", type=float, default=None, help="Vertical exaggeration (auto if not specified)")
    parser.add_argument("--z-dir", type=float, default=1.0, help="Height direction: +1.0=up, -1.0=down")
    parser.add_argument("--theta", type=float, default=45.0, help="Camera azimuth (degrees)")
    parser.add_argument("--phi", type=float, default=25.0, help="Camera elevation (degrees)")
    parser.add_argument("--gamma", type=float, default=0.0, help="Camera roll (degrees)")
    parser.add_argument("--fov", type=float, default=35.0, help="Field of view (degrees)")
    parser.add_argument("--light-type", default="directional", help="Lighting: none, directional, hemisphere")
    parser.add_argument("--light-elevation", type=float, default=45.0, help="Light elevation (degrees)")
    parser.add_argument("--light-azimuth", type=float, default=315.0, help="Light azimuth (degrees)")
    parser.add_argument("--light-intensity", type=float, default=1.0, help="Light intensity")
    parser.add_argument("--ambient", type=float, default=0.25, help="Ambient light term")
    # Shadow mapping parameters
    parser.add_argument("--shadow-intensity", type=float, default=0.6, help="Shadow darkness (0=no shadows, 1=full shadows)")
    parser.add_argument("--shadow-softness", type=float, default=2.0, help="Shadow softness/PCF kernel radius (1.0-5.0)")
    parser.add_argument("--shadow-map-res", type=int, default=2048, help="Shadow map resolution (512-4096)")
    parser.add_argument("--shadow-bias", type=float, default=0.0015, help="Depth bias for shadow acne prevention")
    parser.add_argument("--no-shadows", action="store_true", help="Disable shadow rendering")
    
    # HDRI environment lighting
    parser.add_argument("--hdri", type=str, default=None, help="Path to HDRI environment map (.hdr, .exr)")
    parser.add_argument("--hdri-intensity", type=float, default=1.0, help="HDRI intensity multiplier (0-2+)")
    parser.add_argument("--hdri-rotation", type=float, default=0.0, help="HDRI rotation in degrees (0-360)")
    
    # Denoising
    parser.add_argument("--denoiser", default="oidn", choices=["none", "oidn", "bilateral"], help="Denoising method (default: oidn with fallback)")
    parser.add_argument("--denoise-strength", type=float, default=0.8, help="Denoising intensity 0.0-1.0 (default: 0.8)")
    
    # Memory management and tiling (ray tracing only)
    parser.add_argument("--rt-mem-limit-mib", type=int, default=512, help="Memory budget in MiB for ray tracing (default: 512)")
    parser.add_argument("--rt-auto-tile", action="store_true", default=True, help="Auto-compute tile size to fit memory budget")
    parser.add_argument("--rt-tile-min", type=int, nargs=2, default=[96, 96], metavar=("W", "H"), help="Minimum tile size (default: 96 96)")
    parser.add_argument("--rt-tile-max", type=int, nargs=2, default=[512, 512], metavar=("W", "H"), help="Maximum tile size (default: 512 512)")
    parser.add_argument("--rt-accum-format", default="rgba16f", choices=["rgba16f", "rgba32f"], help="Accumulation buffer format (default: rgba16f for memory efficiency)")
    parser.add_argument("--oidn-mode", default="final", choices=["final", "tiled", "off"], help="OIDN mode: final (full-frame), tiled (per-tile with overlap), off (no denoising)")

    # AOV saving
    parser.add_argument("--save-aovs", action="store_true", help="Save AOVs (albedo, normal, depth, visibility) alongside output")
    parser.add_argument("--aovs", nargs="+", default=["normal", "depth", "visibility"], help="Which AOVs to save")
    parser.add_argument("--aov-dir", default=None, help="Directory to save AOVs (default: alongside output)")
        
    parser.add_argument("--max-dim", type=int, default=3000, help="Max dimension for DEM/landcover (will resample)")
    parser.add_argument("--cache-dir", default="./cache", help="Cache directory for fetched data")
    # P0 diagnostic helpers
    parser.add_argument("--p0-green-sentinel", action="store_true", help="Enable P0 diagnostic: Green Sentinel (debug_mode=2) and force raytrace path")
    parser.add_argument("--res-4k", action="store_true", help="Convenience flag to set 4K resolution (3840x2160)")
    parser.add_argument("--p1-miss-sky", action="store_true", help="Enable P1 diagnostic: Miss path sky/HDRI only (debug_mode=3) and force raytrace path")
    parser.add_argument("--debug-tiles", action="store_true", help="Enable tile addressing diagnostic: XY gradient (debug_mode=4) to verify seamless tile writes")
        
    args = parser.parse_args()
        
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    # Apply P0 diagnostic defaults when requested
    if args.res_4k:
        args.width, args.height = 3840, 2160
    if args.p0_green_sentinel:
        # Force raytrace path and Green Sentinel mode
        args.render_mode = "raytrace"
        args.rt_debug_mode = 2
        # Prefer 4K unless user explicitly set a different size
        if (args.width, args.height) in [(1920, 1080), (1280, 720)]:
            args.width, args.height = 3840, 2160
        # Use 10 spp for P0 unless explicitly overridden
        if args.rt_spp == 64:
            args.rt_spp = 10
        print("\n[P0] Green Sentinel diagnostic enabled: debug_mode=2, raytrace, {}x{}, {} spp".format(args.width, args.height, args.rt_spp))
        print("[P0] Expect three vertical bands (red, green, blue) with ramps and tile borders."
              " If the final image is still predominantly green, the copy/write pipeline is at fault."
              " Otherwise, investigate ray/palette path.")

    # Apply P1 diagnostic defaults when requested
    if args.p1_miss_sky:
        # Force raytrace path and Miss-Sky diagnostic mode
        args.render_mode = "raytrace"
        args.rt_debug_mode = 3
        # Prefer 4K unless user explicitly set a different size
        if args.res_4k or (args.width, args.height) in [(1920, 1080), (1280, 720)]:
            args.width, args.height = 3840, 2160
        # Use 1 spp for P1 unless explicitly overridden
        if args.rt_spp == 64:
            args.rt_spp = 1
        # Denoiser not needed for sky diagnostic
        if str(args.denoiser).lower() != "none":
            args.denoiser = "none"
        mode_msg = "HDRI" if args.hdri else "procedural sky"
    
    # Apply debug-tiles diagnostic when requested
    if args.debug_tiles:
        # Force raytrace path and XY gradient mode
        args.render_mode = "raytrace"
        args.rt_debug_mode = 4
        # Use 1 spp for gradient diagnostic
        if args.rt_spp == 64:
            args.rt_spp = 1
        # Denoiser not needed for gradient diagnostic
        if str(args.denoiser).lower() != "none":
            args.denoiser = "none"
        print("\n[DEBUG-TILES] XY gradient diagnostic enabled: debug_mode=4, raytrace, {}x{}, {} spp".format(args.width, args.height, args.rt_spp))
        print("[DEBUG-TILES] Expect continuous horizontal/vertical gradient (red=X, green=Y) across all tiles.")
        print("[DEBUG-TILES] Tile borders should be visible as darker lines every 64px.")
        print("[DEBUG-TILES] If any tiles are solid black, the tile addressing or write path is broken.")
        
    
    print(f"🌍 Switzerland 3D Land Cover Terrain Draping")
    print(f"=" * 60)
    
    # Step 1: Get country boundary
    print(f"\n1️⃣  Fetching boundary for {args.country}...")
    boundary = get_country_boundary(args.country)
    geometry = boundary.geometry.iloc[0]
    
    # Step 2: Fetch data
    dem_cache = cache_dir / f"dem_{args.country}_{args.max_dim}.npz"
    lc_cache = cache_dir / f"landcover_{args.country}_{args.max_dim}.npz"
    
    if dem_cache.exists() and lc_cache.exists():
        print(f"\n2️⃣  Loading cached data...")
        dem_data = np.load(dem_cache)
        dem = dem_data['dem']
        # We saved transform using Affine.to_gdal(); reconstruct with from_gdal
        dem_transform = rasterio.Affine.from_gdal(*dem_data['transform'])
        dem_crs = rasterio.CRS.from_string(str(dem_data['crs']))
        
        lc_data = np.load(lc_cache)
        landcover = lc_data['landcover']
        lc_transform = rasterio.Affine.from_gdal(*lc_data['transform'])
        lc_crs = rasterio.CRS.from_string(str(lc_data['crs']))
    else:
        print(f"\n2️⃣  Fetching data from Planetary Computer...")
        dem, dem_transform, dem_crs = fetch_dem(geometry, max_dim=args.max_dim)
        landcover, lc_transform, lc_crs = fetch_worldcover(geometry, max_dim=args.max_dim)
        
        # Cache results
        np.savez_compressed(
            dem_cache,
            dem=dem,
            transform=dem_transform.to_gdal(),
            crs=str(dem_crs),
        )
        np.savez_compressed(
            lc_cache,
            landcover=landcover,
            transform=lc_transform.to_gdal(),
            crs=str(lc_crs),
        )
        print(f"   💾 Cached data to {cache_dir}")
    
    # Step 3: Align rasters
    print(f"\n3️⃣  Processing data...")
    lc_aligned = align_rasters(dem, dem_transform, dem_crs, landcover, lc_transform, lc_crs)
    
    # Convert to relative heights (subtract minimum)
    dem_min = dem.min()
    dem_relative = dem - dem_min
    elev_span = dem.max() - dem_min
    print(f"   Elevation range: {dem_min:.1f} to {dem.max():.1f} m (span: {elev_span:.1f} m)")
    
    # Auto-calculate zscale and pixel spacing from geometry bounds to avoid cached-transform issues
    if args.zscale is None:
        # Degrees per pixel from geometry bounds
        minx, miny, maxx, maxy = geometry.bounds
        width_deg = max(1e-12, float(maxx - minx))
        height_deg = max(1e-12, float(maxy - miny))
        pixel_size_x_deg = width_deg / float(dem.shape[1])
        pixel_size_y_deg = height_deg / float(dem.shape[0])
        
        # Convert to meters-per-degree at mid-latitude of AOI
        lat_mid = float(0.5 * (miny + maxy))
        meters_per_deg_x = 111000.0 * float(np.cos(np.radians(lat_mid)))
        meters_per_deg_y = 111000.0
        
        sx_m = pixel_size_x_deg * meters_per_deg_x
        sy_m = pixel_size_y_deg * meters_per_deg_y
        terrain_width_m = dem.shape[1] * sx_m
        terrain_height_m = dem.shape[0] * sy_m
        terrain_extent_m = max(terrain_width_m, terrain_height_m)
        
        # Vertical exaggeration target based on true horizontal extent
        args.zscale = elev_span / terrain_extent_m if terrain_extent_m > 0 else 1.0
        args.zscale = max(0.5, min(args.zscale, 3.0))
        print(f"   Auto zscale: {args.zscale:.3f} (terrain={terrain_extent_m/1000:.1f}km, elev_span={elev_span:.1f}m)")
        print(f"   Pixel spacing: sx={sx_m:.2f} m, sy={sy_m:.2f} m (lat≈{lat_mid:.2f}°)")
    else:
        # If user specified zscale, still compute pixel spacing from bounds
        minx, miny, maxx, maxy = geometry.bounds
        width_deg = max(1e-12, float(maxx - minx))
        height_deg = max(1e-12, float(maxy - miny))
        pixel_size_x_deg = width_deg / float(dem.shape[1])
        pixel_size_y_deg = height_deg / float(dem.shape[0])
        lat_mid = float(0.5 * (miny + maxy))
        meters_per_deg_x = 111000.0 * float(np.cos(np.radians(lat_mid)))
        meters_per_deg_y = 111000.0
        sx_m = pixel_size_x_deg * meters_per_deg_x
        sy_m = pixel_size_y_deg * meters_per_deg_y
    
    # Convert land-cover classes to RGBA
    lc_rgba = classes_to_rgba(lc_aligned)
    print(f"   Unique land-cover classes: {len(np.unique(lc_aligned))}")
    
    # Step 4: Estimate memory usage
    print(f"\n4️⃣  Estimating GPU memory usage...")
    mem_info = estimate_memory_usage(dem.shape)
    print(f"   DEM texture (R32F): {mem_info['heightmap_r32f'] / (1024**2):.1f} MB")
    print(f"   Land-cover texture (RGBA8): {mem_info['landcover_rgba8'] / (1024**2):.1f} MB")
    print(f"   Total textures: {mem_info['total_mb']:.1f} MB")
    
    if mem_info['exceeds_512mb_budget']:
        print(f"   ⚠️  Warning: Exceeds 512 MB budget. Consider reducing --max-dim")
    
    # Step 5: Render with terrain draping
    print(f"\n5️⃣  Rendering terrain with GPU {'path tracing' if args.render_mode == 'raytrace' else 'rasterization'}...")
    print(f"   Render mode: {args.render_mode}")
    if args.render_mode == "raytrace":
        print(f"   Ray tracing SPP: {args.rt_spp}, seed: {args.rt_seed}")
    print(f"   Output size: {args.width}x{args.height}")
    print(f"   Vertical exaggeration: {args.zscale}x, z_dir={args.z_dir}")
    print(f"   Camera: theta={args.theta}°, phi={args.phi}°, gamma={args.gamma}°, fov={args.fov}°")
    print(f"   Lighting: {args.light_type}, elevation={args.light_elevation}°, azimuth={args.light_azimuth}°")
    print(f"   Shadows: {'disabled' if args.no_shadows else f'enabled (intensity={args.shadow_intensity}, softness={args.shadow_softness})'}")
    print(f"   HDRI: {args.hdri if args.hdri else 'none (using constant ambient)'}")
    if args.hdri:
        print(f"   HDRI intensity: {args.hdri_intensity}, rotation: {args.hdri_rotation}°")
    print(f"   Denoiser: {args.denoiser} (strength={args.denoise_strength})")
    print(f"   Max dimension cap: {args.max_dim}")

    # If denoiser is 'none', disable OIDN at GPU stage as well
    if str(args.denoiser).lower() == "none" and getattr(args, "oidn_mode", "final") != "off":
        args.oidn_mode = "off"
        print("   OIDN disabled (oidn_mode=off) due to denoiser=none")

    # Inform about AOVs
    if args.save_aovs:
        print(f"   AOVs: will save {', '.join(args.aovs)}")

    result = drape_landcover(
        dem_relative,
        lc_rgba,
        render_mode=args.render_mode,
        rt_spp=args.rt_spp,
        rt_seed=args.rt_seed,
        rt_batch_spp=args.rt_batch_spp,
        rt_sampling_mode=args.rt_sampling_mode,
        rt_debug_mode=args.rt_debug_mode,
        max_rt_triangles=args.max_rt_triangles,
        max_dim=args.max_dim,
        # Memory management parameters
        rt_mem_limit_mib=args.rt_mem_limit_mib,
        rt_auto_tile=args.rt_auto_tile,
        rt_tile_min=tuple(args.rt_tile_min),
        rt_tile_max=tuple(args.rt_tile_max),
        rt_accum_format=args.rt_accum_format,
        oidn_mode=args.oidn_mode,
        width=args.width,
        height=args.height,
        z_dir=args.z_dir,
        zscale=args.zscale,
        pixel_spacing=(sx_m, sy_m),
        save_aovs=bool(args.save_aovs),
        aovs=tuple(args.aovs),
        aov_dir=(args.aov_dir if args.aov_dir else str(Path(args.output).parent)),
        basename=str(Path(args.output).stem),
        camera_theta=args.theta,
        camera_phi=args.phi,
        camera_gamma=args.gamma,
        camera_fov=args.fov,
        light_type=args.light_type,
        light_elevation=args.light_elevation,
        light_azimuth=args.light_azimuth,
        light_intensity=args.light_intensity,
        ambient=args.ambient,
        # Shadow mapping
        shadow_intensity=args.shadow_intensity,
        shadow_softness=args.shadow_softness,
        shadow_map_res=args.shadow_map_res,
        shadow_bias=args.shadow_bias,
        enable_shadows=not args.no_shadows,
        # HDRI environment
        hdri=args.hdri,
        hdri_intensity=args.hdri_intensity,
        hdri_rotation_deg=args.hdri_rotation,
        # Denoising
        denoiser=args.denoiser,
        denoise_strength=args.denoise_strength,
        background="#00000000" if True else "#FFFFFF",  # Transparent background
        y_flip=False,
    )
    
    # Step 6: Save result
    print(f"\n6️⃣  Saving result...")
    Image.fromarray(result).save(args.output)
    print(f"   ✅ Saved to {args.output}")
    
    # Print statistics
    unique_colors = len(np.unique(result.reshape(-1, 4), axis=0))
    print(f"\n📊 Statistics:")
    print(f"   Input DEM shape: {dem.shape}")
    print(f"   Input land-cover shape: {lc_rgba.shape}")
    print(f"   Output image shape: {result.shape}")
    print(f"   Unique colors in output: {unique_colors:,}")
    # Additional color channel diagnostics
    rgb = result[..., :3].astype(np.uint8)
    mean_r = float(np.mean(rgb[..., 0])) if rgb.size else 0.0
    mean_g = float(np.mean(rgb[..., 1])) if rgb.size else 0.0
    mean_b = float(np.mean(rgb[..., 2])) if rgb.size else 0.0
    total_px = float(rgb.shape[0] * rgb.shape[1]) if rgb.ndim == 3 else 0.0
    pure_green = float(np.sum((rgb[..., 0] == 0) & (rgb[..., 2] == 0) & (rgb[..., 1] > 0))) if total_px > 0 else 0.0
    g_dom = float(np.sum((rgb[..., 1].astype(np.int32) > rgb[..., 0].astype(np.int32) + 10) &
                         (rgb[..., 1].astype(np.int32) > rgb[..., 2].astype(np.int32) + 10))) if total_px > 0 else 0.0
    print(f"   Mean RGB: ({mean_r:.1f}, {mean_g:.1f}, {mean_b:.1f})")
    if total_px > 0:
        print(f"   Pure-green pixels: {100.0 * pure_green / total_px:.2f}% | G-dominant pixels: {100.0 * g_dom / total_px:.2f}%")
    
    print(f"\n✨ Done! View the result: {args.output}")


if __name__ == "__main__":
    main()
