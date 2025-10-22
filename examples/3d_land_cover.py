#!/usr/bin/env python3
"""
3D Land Cover Visualization using forge3d

Uses ESA WorldCover 2021 10m land cover data and py3dep for elevation.
Creates a 3D terrain visualization with forge3d.

Usage:
    python 3d_land_cover.py --country CH --output-dir ./output
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import rasterio
import multiprocessing as mp
import threading
import time
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask as rio_mask
import geopandas as gpd
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from affine import Affine

try:
    import forge3d
    import planetary_computer
    import pystac_client
    import odc.stac
    import rioxarray
    # Interactive single-terminal CLI and raytrace helper
    from forge3d.helpers.interactive_cli import (
        _command_reader,
        interactive_control_loop,
        demo_snapshots,
    )
    from forge3d.render import (
        RaytraceMeshCache,
        load_dem as _pkg_load_dem,
        render_raytrace_mesh,
        heightmap_to_mesh,
    )
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install forge3d pystac-client odc-stac planetary-computer rioxarray")
    sys.exit(1)


# ESA WorldCover 2021 color mapping
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
    111: "#b040b0", # Unknown
    0: "#000000",   # NoData (will be transparent in overlay)
}

ESA_LABELS = {
    10: "Tree cover",
    20: "Shrubland", 
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse",
    70: "Snow/ice",
    80: "Water",
    90: "Wetland",
    95: "Mangroves",
    100: "Moss/lichen",
    111: "Unknown",
    0: "NoData",
}


def get_country_boundary(country_code: str) -> gpd.GeoDataFrame:
    """Fetch country boundary."""
    import ssl
    import urllib.request
    
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
    urllib.request.install_opener(opener)
    
    url = "https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson/CNTR_RG_01M_2020_4326.geojson"
    gdf = gpd.read_file(url)
    return gdf[gdf['CNTR_ID'] == country_code]


def fetch_esa_worldcover(bbox: tuple, output_path: Path, country_geom) -> Path:
    """Fetch ESA WorldCover 2021 data from Planetary Computer."""
    minx, miny, maxx, maxy = bbox
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=[minx, miny, maxx, maxy],
    )
    
    items = list(search.items())
    print(f"  Found {len(items)} ESA WorldCover items")
    
    # Load all items that intersect bbox and mosaic
    import xarray as xr
    from rioxarray.merge import merge_arrays
    if len(items) == 0:
        raise RuntimeError("No ESA WorldCover items for requested bounds")
    datasets = []
    for it in items:
        signed_href = planetary_computer.sign(it.assets["map"].href)
        ds = rioxarray.open_rasterio(signed_href, masked=True)
        ds = ds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        datasets.append(ds)

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
    data = data.rio.clip([country_geom], all_touched=True)

    lc_array = data.values[0].astype(np.uint8)
    print(f"  Land cover mosaic shape: {lc_array.shape}, dtype: {lc_array.dtype}")
    print(f"  Value range: {lc_array.min()} to {lc_array.max()}")

    # Save to GeoTIFF
    data.rio.to_raster(output_path, compress='lzw')
    
    return output_path


def fetch_elevation(bbox: tuple, output_dir: Path, country_geom) -> np.ndarray:
    """Fetch Copernicus DEM from Planetary Computer."""
    minx, miny, maxx, maxy = bbox
    dem_path = output_dir / "elevation.tif"
    
    if dem_path.exists() and dem_path.stat().st_size > 1000:
        with rasterio.open(dem_path) as src:
            return src.read(1)
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    search = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=[minx, miny, maxx, maxy],
    )
    
    items = list(search.items())
    print(f"  Found {len(items)} Copernicus DEM items")
    
    # Merge all DEM tiles using rioxarray
    import xarray as xr
    datasets = []
    for item in items:
        signed_href = planetary_computer.sign(item.assets["data"].href)
        ds = rioxarray.open_rasterio(signed_href, masked=True)
        ds = ds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        datasets.append(ds)
    
    # Merge all datasets into a single mosaic (union of extents)
    if len(datasets) == 0:
        raise RuntimeError("No DEM tiles found for requested bounds")
    if len(datasets) == 1:
        data = datasets[0]
    else:
        try:
            from rioxarray.merge import merge_arrays
            data = merge_arrays(datasets)
        except Exception:
            # Fallback: reproject to the first dataset grid then take first non-NaN per-pixel
            base = datasets[0]
            filled = base.copy()
            for ds in datasets[1:]:
                ds_r = ds.rio.reproject_match(base)
                mask = filled.isnull()
                filled = xr.where(mask, ds_r, filled)
            data = filled
    
    # Clip to country boundary
    data = data.rio.clip([country_geom], all_touched=True)
    
    elev_data = data.values[0] if data.values.ndim == 3 else data.values
    elev_data = elev_data.astype(np.float32)
    print(f"  Elevation shape: {elev_data.shape}, dtype: {elev_data.dtype}")
    print(f"  Elevation range: {np.nanmin(elev_data):.1f}m to {np.nanmax(elev_data):.1f}m")
    
    # Save for caching
    data.rio.to_raster(dem_path, compress='lzw')
    
    return elev_data


def raster_to_rgb(raster_path: Path, color_map: dict) -> np.ndarray:
    """Convert land cover raster to RGB using color mapping."""
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        return lc_array_to_rgb(data, color_map)


def lc_array_to_rgb(lc_array: np.ndarray, color_map: dict) -> np.ndarray:
    """Map a land cover array (categorical) to RGB using a LUT; safe for NaNs."""
    # Fast LUT-based mapping instead of per-class masking
    lut = np.zeros((256, 3), dtype=np.uint8)
    for value, hex_color in color_map.items():
        vi = int(value)
        if 0 <= vi <= 255:
            lut[vi, 0] = int(hex_color[1:3], 16)
            lut[vi, 1] = int(hex_color[3:5], 16)
            lut[vi, 2] = int(hex_color[5:7], 16)
    data = np.nan_to_num(lc_array, nan=0)
    data_clamped = np.clip(data, 0, 255).astype(np.uint8)
    rgb = lut[data_clamped]
    return rgb


def align_landcover_to_dem(
    lc_path: Path,
    dem_path: Path,
    *,
    target_max_dim: int = 1200,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray, tuple[float, float]]:
    """Reproject ESA WorldCover to DEM grid so they align perfectly.

    Returns (elevation_float32, lc_rgb_uint8, present_classes)
    """
    # Optional on-disk cache to skip expensive reprojection on subsequent runs
    cache_root = cache_dir or dem_path.parent
    with rasterio.open(dem_path) as dem_ds:
        src_h, src_w = dem_ds.height, dem_ds.width
        # Compute downscale factor to cap the largest dimension
        cap = max(1, int(target_max_dim))
        scale = min(1.0, float(cap) / float(max(src_h, src_w)))
        dst_h = max(1, int(round(src_h * scale)))
        dst_w = max(1, int(round(src_w * scale)))
        cache_path = cache_root / f"aligned_{dem_path.stem}_{Path(lc_path).stem}_{dst_w}x{dst_h}.npz"
        if (not refresh) and cache_path.exists():
            try:
                data = np.load(cache_path)
                elev = data["elev"].astype(np.float32)
                lc_reproj = data["lc"].astype(np.uint8)
                sx = float(data["sx"]) if "sx" in data else abs(float(dem_ds.transform.a))
                sy = float(data["sy"]) if "sy" in data else abs(float(dem_ds.transform.e))
                classes = sorted(int(v) for v in np.unique(lc_reproj) if int(v) in ESA_COLORS and int(v) != 0)
                lc_rgb = lc_array_to_rgb(lc_reproj, ESA_COLORS)
                print(f"  Using aligned cache: {cache_path.name} -> {elev.shape}")
                return elev, lc_rgb, classes, lc_reproj, (sx, sy)
            except Exception:
                pass
        # Increase pixel size to match downsample using Affine scale
        if scale < 1.0:
            dst_transform = dem_ds.transform * Affine.scale(1.0 / scale, 1.0 / scale)
        else:
            dst_transform = dem_ds.transform
        dst_crs = dem_ds.crs

        # Reproject DEM directly to the target resolution (bilinear)
        elev = np.zeros((dst_h, dst_w), dtype=np.float32)
        reproject(
            source=rasterio.band(dem_ds, 1),
            destination=elev,
            src_transform=dem_ds.transform,
            src_crs=dem_ds.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=dem_ds.nodata,
            dst_nodata=np.nan,
        )
        try:
            sx = abs(float(dst_transform.a)) or 1.0
            sy = abs(float(dst_transform.e)) or 1.0
        except Exception:
            sx, sy = 1.0, 1.0

    # Reproject land cover to the same grid (nearest)
    with rasterio.open(lc_path) as src:
        lc_dtype = np.uint8
        lc_reproj = np.zeros((dst_h, dst_w), dtype=lc_dtype)
        reproject(
            source=rasterio.band(src, 1),
            destination=lc_reproj,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=0,
            dst_nodata=0,
        )
    classes = sorted(int(v) for v in np.unique(lc_reproj) if int(v) in ESA_COLORS and int(v) != 0)
    lc_rgb = lc_array_to_rgb(lc_reproj, ESA_COLORS)
    # Save cache for next runs
    try:
        np.savez_compressed(cache_path, elev=elev.astype(np.float32), lc=lc_reproj.astype(np.uint8), sx=float(sx), sy=float(sy))
        print(f"  Wrote aligned cache: {cache_path.name}")
    except Exception:
        pass
    return elev, lc_rgb, classes, lc_reproj, (sx, sy)


def create_legend(output_path: Path, color_map: dict, label_map: dict, selected: list = None):
    """Create legend for land cover classes with adaptive layout."""
    if selected is None:
        selected = [10, 40, 50, 80]  # Tree, Crop, Built, Water
    selected = [c for c in selected if c in color_map]
    n = len(selected)
    ncols = 1 if n <= 6 else (2 if n <= 12 else 3)
    nrows = (n + ncols - 1) // ncols
    fig_w = 4.0 * ncols
    fig_h = max(2.0, 0.6 * nrows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='none')
    ax.axis('off')

    handles = [mpatches.Patch(color=color_map[c], label=label_map.get(c, str(c))) for c in selected]
    ax.legend(
        handles=handles,
        loc='center',
        fontsize=14,
        frameon=False,
        ncol=ncols,
        columnspacing=1.0,
        handlelength=1.2,
        handletextpad=0.6,
        borderaxespad=0.2,
    )
    plt.savefig(output_path, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()


def render_3d_terrain(
    elevation: np.ndarray,
    texture_rgb: np.ndarray,
    texture_classes: np.ndarray | None,
    output_path: Path,
    width: int = 2000,
    height: int = 2000,
    z_scale: float = 30.0,
    renderer: str = "auto",
    frames: int = 2,
    pt_scale: float = 0.66,
    pt_mesh_sub: int = 2,
    pt_backend: str = "auto",
    pt_orbit_theta: float = 45.0,
    pt_orbit_phi: float = 25.0,
    pt_denoiser: str = "svgf",
    pt_svgf_iters: int = 5,
    pt_exposure: float = 0.85,
    pt_auto_exposure: bool = True,
    pt_gamma: float = 1.1,
    pt_supersample: float = 1.0,
    pt_luma_clamp: float | None = 1.8,
    pt_detail_boost: float = 0.3,
    pt_hillshade_mix: float = 0.5,
    pt_max_dim: int | None = 2048,
    max_elev_size: int | None = None,
    spacing: tuple[float, float] | None = None,
    lighting_type: str = "blinn-phong",
    lighting_intensity: float = 1.15,
    lighting_azimuth: float = 315.0,
    lighting_elevation: float = 20.0,
    # Tone/contrast controls
    gamma: float = 1.0,
    equalize: bool = False,
    contrast_pct: float = 1.25,
    exaggeration: float = 2.2,
    # Final output tone controls
    final_gamma: float = 1.1,
    final_gain: float = 1.0,
    final_sharpen: float = 0.0,
    # Shadow controls
    shadow_enabled: bool = True,
    shadow_intensity: float = 1.2,
    # Overlay controls
    overlay_alpha: int = 110,
    overlay_blend: str = "overlay",
    # Water overlay controls (default OFF for speed)
    water_mode: str = "off",  # one of: off|percentile|level
    water_percentile: float | None = None,
    water_level: float | None = None,
    water_morph_iter: int = 1,
    water_min_area_pct: float = 0.01,
    water_keep_components: int = 2,
    water_max_slope_deg: float = 6.0,
    water_min_depth: float = 0.1,
    water_debug: bool = False,
):
    """Render 3D terrain using forge3d with render_overlay."""
    from scipy.ndimage import zoom, gaussian_filter
    
    # Fill NaNs in elevation to avoid hillshade NaNs
    if not np.isfinite(elevation).all():
        finite = elevation[np.isfinite(elevation)]
        if finite.size:
            elevation = np.nan_to_num(elevation, nan=float(np.nanmin(finite)))
        else:
            elevation = np.nan_to_num(elevation, nan=0.0)

    # Optional: cap elevation resolution for speed if requested
    cap = int(max_elev_size) if (max_elev_size is not None and int(max_elev_size) > 0) else None
    if cap is not None and (elevation.shape[0] > cap or elevation.shape[1] > cap):
        scale = min(cap / elevation.shape[0], cap / elevation.shape[1])
        elevation = zoom(elevation, scale, order=1)
    
    # Match texture to elevation shape
    if texture_rgb.shape[:2] != elevation.shape:
        zoom_factors = (
            elevation.shape[0] / texture_rgb.shape[0],
            elevation.shape[1] / texture_rgb.shape[1],
            1
        )
        texture_rgb = zoom(texture_rgb, zoom_factors, order=0)
        if texture_classes is not None and texture_classes.shape[:2] != elevation.shape:
            texture_classes = zoom(texture_classes, (
                elevation.shape[0] / texture_classes.shape[0],
                elevation.shape[1] / texture_classes.shape[1]
            ), order=0)
    
    # Select renderer: auto -> prefer GPU path tracer (no fallback logic here)
    selected_renderer = str(renderer).lower()
    if selected_renderer == "auto":
        selected_renderer = "path_tracer"

    # Minimal logging; avoid verbose prints
    if selected_renderer == "path_tracer":
        prefer_gpu = True if str(pt_backend).lower() in ("auto", "gpu") else False
        # Render the 3D base at a reduced resolution for speed, then upscale
        ss = max(1.0, float(pt_supersample))
        # Desired internal size before clamping to device limits
        want_w = max(64, int(width * float(pt_scale) * ss))
        want_h = max(64, int(height * float(pt_scale) * ss))
        # Clamp to device-safe max texture size (heuristic default=2048; can be overridden)
        rt_w, rt_h = want_w, want_h
        if pt_max_dim is not None:
            limit = max(256, int(pt_max_dim))
            mx = max(rt_w, rt_h)
            if mx > limit:
                s = float(limit) / float(mx)
                rt_w = max(64, int(rt_w * s))
                rt_h = max(64, int(rt_h * s))
        # Auto-subsample the heightmap to keep triangle count reasonable for BVH construction
        # Target: ~100k triangles max (BVH builds in <3 seconds)
        sub = int(pt_mesh_sub)
        if sub <= 0:  # Auto mode
            h, w = elevation.shape
            estimated_tris = h * w * 2
            target_tris = 100_000  # Aggressive target for fast BVH (empirically tested)
            if estimated_tris > target_tris:
                # Calculate subsampling factor needed, round UP to ensure we stay under target
                ratio = estimated_tris / target_tris
                sub = max(1, int(ratio ** 0.5) + 1)
                actual_tris = estimated_tris // sub // sub
                print(f"[PT] Auto mesh subsample: {sub}x (reducing {estimated_tris:,} → {actual_tris:,} triangles)")
            else:
                sub = 1
        sub = max(1, sub)
        
        if sub > 1:
            elev_src = elevation[::sub, ::sub]
        else:
            elev_src = elevation
        
        # Spacing from cache is already in the downsampled grid's units
        # Subsample AGAIN for mesh generation, so scale spacing accordingly
        spacing_src = (spacing[0] * sub, spacing[1] * sub) if spacing is not None else (1.0, 1.0)
        
        # Convert spacing from degrees to meters (geospatial data uses degrees)
        # Approximate: 1 degree ≈ 111km at equator, with cosine correction for latitude
        # For Switzerland (lat ~47°), use ~74km per degree for longitude
        lat_approx = 47.0  # Switzerland central latitude
        meters_per_deg_lon = 111320.0 * np.cos(np.radians(lat_approx))
        meters_per_deg_lat = 111320.0
        spacing_meters = (spacing_src[0] * meters_per_deg_lon, spacing_src[1] * meters_per_deg_lat)
        print(f"[PT-DEBUG] Spacing conversion: {spacing_src[0]:.6f}° × {spacing_src[1]:.6f}° → {spacing_meters[0]:.1f}m × {spacing_meters[1]:.1f}m")
        spacing_src = spacing_meters
        
        # Convert absolute elevations to relative heights (crucial for coordinate scale matching)
        elev_min = float(np.min(elev_src))
        elev_relative = elev_src - elev_min
        elev_max_rel = float(np.max(elev_relative))
        
        # Apply exaggeration to z_scale for better 3D relief appearance
        effective_z_scale = z_scale * float(exaggeration) / 2.2  # Normalize by default exaggeration
        
        print(f"[PT-DEBUG] Mesh generation: shape={elev_src.shape}, spacing={spacing_src}, z_scale={z_scale}, exaggeration={exaggeration:.2f}, effective_z_scale={effective_z_scale:.1f}")
        print(f"[PT-DEBUG] Elevation range: {elev_min:.1f}m to {elev_min + elev_max_rel:.1f}m (relative: 0 to {elev_max_rel:.1f}m)")
        
        # Build mesh and render via raytracer using RELATIVE elevations
        V, F = heightmap_to_mesh(elev_relative, spacing_src, vertical_scale=float(effective_z_scale))
        
        # Attach vertex colors from land cover texture
        # Subsample texture to match mesh resolution
        if sub > 1:
            texture_src = texture_rgb[::sub, ::sub]
        else:
            texture_src = texture_rgb
        
        # Create colored vertices (N, 8): [x, y, z, pad, r, g, b, pad2]
        H, W = elev_src.shape
        V_colored = np.zeros((V.shape[0], 8), dtype=np.float32)
        V_colored[:, :3] = V  # positions
        
        # Map vertex colors from texture (texture is RGB uint8, convert to float32 0-1)
        for i in range(H):
            for j in range(W):
                v_idx = i * W + j
                if v_idx < V.shape[0]:
                    # Get color from texture (already aligned to elevation grid)
                    color_rgb = texture_src[i, j].astype(np.float32) / 255.0
                    V_colored[v_idx, 4:7] = color_rgb
        
        print(f"[PT-DEBUG] Mesh created: {V.shape[0]} verts with colors, bounds X=[{V[:,0].min():.3f}, {V[:,0].max():.3f}], Y=[{V[:,1].min():.3f}, {V[:,1].max():.3f}], Z=[{V[:,2].min():.3f}, {V[:,2].max():.3f}]")
        # Triangle winding is now correct from heightmap_to_mesh (counter-clockwise for Y-up)
        # Verify winding for first triangle
        try:
            if F.size >= 3:
                v0, v1, v2 = V[F[0, 0]], V[F[0, 1]], V[F[0, 2]]
                n = np.cross(v1 - v0, v2 - v0)
                upv = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                dot_up = float(np.dot(n, upv))
                print(f"[PT-DEBUG] First triangle normal dot(up): {dot_up:.4f} (should be positive for upward-facing)")
        except Exception:
            pass
        img, _meta = render_raytrace_mesh(
            (V_colored, F),
            size=(rt_w, rt_h),
            frames=max(1, int(frames)),
            # Prefer GPU path tracer; robust CPU fallback remains if unavailable
            prefer_gpu=True,
            up=(0.0, 1.0, 0.0),
            orbit_theta=float(pt_orbit_theta),
            orbit_phi=float(pt_orbit_phi),
            # Use normalization to scale mesh to GPU-friendly coordinates
            # Scale 2.0 gives ~2 unit max dimension, preserving exaggeration ratios
            normalize=True,
            normalize_scale=2.0,
            # Enable GPU ambient lighting for 3D appearance
            # Disable GPU shadow rays (use CPU hillshade instead for clean shadows)
            lighting_type=str(lighting_type),
            lighting_intensity=float(lighting_intensity),
            lighting_azimuth=float(lighting_azimuth),
            lighting_elevation=float(lighting_elevation),
            shadows=False,  # GPU shadows disabled - use CPU hillshade
            shadow_intensity=0.0,
            denoiser=("svgf" if str(pt_denoiser).lower() == "svgf" else "off"),
            svgf_iters=int(max(1, pt_svgf_iters)),
            luminance_clamp=(float(pt_luma_clamp) if pt_luma_clamp is not None else None),
            background_color=(1.0, 1.0, 1.0),
        )
        # Ensure output is uint8 in 0..255 regardless of backend return dtype/range
        if img.dtype != np.uint8:
            imgf = img.astype(np.float32)
            maxv = float(np.max(imgf)) if imgf.size else 0.0
            if np.isfinite(maxv) and maxv <= 1.0 + 1e-4:
                img = np.clip(imgf * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                img = np.clip(imgf, 0.0, 255.0).astype(np.uint8)
        terrain_rgba = img
        if (rt_w, rt_h) != (width, height):
            # Downsample when supersampling was used for extra crispness
            resample_filter = Image.LANCZOS if (rt_w >= width and rt_h >= height) else Image.BICUBIC
            terrain_rgba = np.array(Image.fromarray(terrain_rgba).resize((width, height), resample_filter))
        # Optional: inject high-frequency detail from DEM normals to restore acutance
        try:
            boost = float(np.clip(pt_detail_boost, 0.0, 1.0))
            if boost > 1e-6:
                # Compute lambertian shading from full-res elevation to capture micro-relief
                hm_full = elevation.astype(np.float32)
                sx0 = float(spacing[0] if spacing is not None else 1.0)
                sy0 = float(spacing[1] if spacing is not None else 1.0)
                gy, gx = np.gradient(hm_full, sy0, sx0)
                nx = -gx; ny = np.ones_like(hm_full, dtype=np.float32); nz = -gy
                norm = np.maximum(np.sqrt(nx*nx + ny*ny + nz*nz), 1e-6)
                nx /= norm; ny /= norm; nz /= norm
                az = np.radians(float(lighting_azimuth)); el = np.radians(float(lighting_elevation))
                lx = np.cos(el) * np.sin(az); ly = np.sin(el); lz = np.cos(el) * np.cos(az)
                lam = np.clip(nx*lx + ny*ly + nz*lz, 0.0, 1.0).astype(np.float32)
                # High-pass filter (DoG) to isolate fine structure (adaptive sigmas)
                s_small = 0.6 if max(width, height) < 1800 else 0.7
                s_large = 1.4 if max(width, height) < 1800 else 1.8
                lam_s = gaussian_filter(lam, sigma=s_large)
                lam_f = gaussian_filter(lam, sigma=s_small)
                hf = np.clip(lam_f - lam_s, -0.18, 0.18)
                # Resize to output (high quality)
                hf_img = np.array(Image.fromarray((hf * 255.0 + 128.0).astype(np.uint8)).resize((width, height), Image.LANCZOS)).astype(np.float32)
                hf01 = (hf_img - 128.0) / 255.0
                # Apply as multiplicative detail map
                rgbf = terrain_rgba[..., :3].astype(np.float32) / 255.0
                rgbf = np.clip(rgbf * (1.0 + boost * hf01[..., None]), 0.0, 1.0)
                terrain_rgba[..., :3] = (rgbf * 255.0 + 0.5).astype(np.uint8)
        except Exception:
            pass

        # Optional: blend a crisp grayscale hillshade computed at output resolution to emphasize relief
        try:
            mix = float(np.clip(pt_hillshade_mix, 0.0, 1.0))
            if mix > 1e-6:
                # Compute lambertian at full res from DEM normals (grayscale)
                hm_full = elevation.astype(np.float32)
                sx0 = float(spacing[0] if spacing is not None else 1.0)
                sy0 = float(spacing[1] if spacing is not None else 1.0)
                gy, gx = np.gradient(hm_full, sy0, sx0)
                nx = -gx; ny = np.ones_like(hm_full, dtype=np.float32); nz = -gy
                norm = np.maximum(np.sqrt(nx*nx + ny*ny + nz*nz), 1e-6)
                nx /= norm; ny /= norm; nz /= norm
                az = np.radians(float(lighting_azimuth)); el = np.radians(float(lighting_elevation))
                lx = np.cos(el) * np.sin(az); ly = np.sin(el); lz = np.cos(el) * np.cos(az)
                lam = np.clip(nx*lx + ny*ly + nz*lz, 0.0, 1.0).astype(np.float32)
                # Moderate shadow floor for natural relief without harsh darkness
                # Min 0.35 (35% brightness) for gentle shadows, max 1.0 for highlights
                lam_lift = np.clip(0.35 + 0.65 * lam, 0.0, 1.0)
                lam_img = np.array(Image.fromarray((lam_lift * 255.0 + 0.5).astype(np.uint8)).resize((width, height), Image.LANCZOS)).astype(np.float32) / 255.0
                hs_f = np.stack([lam_img, lam_img, lam_img], axis=-1)
                base_f = terrain_rgba[..., :3].astype(np.float32) / 255.0
                # Apply only where we have data (avoid tinting solid background)
                if texture_classes is not None:
                    mask = (texture_classes != 0).astype(np.uint8)
                    mask_resized = np.array(Image.fromarray(mask).resize((width, height), Image.NEAREST))
                    mask_f = (mask_resized > 0)[..., None].astype(np.float32)
                else:
                    mask_f = np.ones_like(base_f[..., :1])
                # Simple multiplicative shadows: hillshade acts as shadow mask
                # This creates clear, dark shadows without complex soft-light math
                # Result = base_color * (1.0 - mix * (1.0 - hillshade))
                # When hillshade=0 (dark) and mix=1: result = base_color * 0
                # When hillshade=1 (light) and mix=1: result = base_color * 1
                shadow_factor = 1.0 - mix * (1.0 - hs_f)
                rgbf = np.clip(base_f * shadow_factor, 0.0, 1.0)
                rgbf = mask_f * rgbf + (1.0 - mask_f) * base_f
                terrain_rgba[..., :3] = (rgbf * 255.0 + 0.5).astype(np.uint8)
        except Exception:
            pass

        # If the path-traced base is effectively black, inject a minimal ambient
        # grayscale derived from DEM normals to ensure the overlay blends visibly.
        try:
            rgb = terrain_rgba[..., :3]
            # Detect near-black frame
            if float(np.percentile(rgb, 99.0)) <= 3.0 and float(np.mean(rgb)) <= 2.0:
                # Compute simple lambert from DEM gradients (using the traced source DEM)
                hm = elev_src.astype(np.float32)
                sx, sy = float(spacing_src[0] or 1.0), float(spacing_src[1] or 1.0)
                gy, gx = np.gradient(hm, sy, sx)
                nx = -gx; ny = np.ones_like(hm, dtype=np.float32); nz = -gy
                norm = np.maximum(np.sqrt(nx*nx + ny*ny + nz*nz), 1e-6)
                nx /= norm; ny /= norm; nz /= norm
                az = np.radians(float(lighting_azimuth)); el = np.radians(float(lighting_elevation))
                lx = np.cos(el) * np.sin(az); ly = np.sin(el); lz = np.cos(el) * np.cos(az)
                lambert = np.clip(nx*lx + ny*ly + nz*lz, 0.0, 1.0).astype(np.float32)
                # Resize to final output and scale to a soft lift (about 20% gray)
                shade = np.array(Image.fromarray((lambert * 255.0 + 0.5).astype(np.uint8)).resize((width, height), Image.BILINEAR))
                lift = (np.clip(0.12 + 0.45*lambert, 0.12, 0.75) * 255.0 + 0.5).astype(np.uint8)
                lift_img = np.array(Image.fromarray(lift).resize((width, height), Image.BILINEAR))
                lift_rgb = np.stack([lift_img, lift_img, lift_img], axis=-1)
                terrain_rgba[..., :3] = np.maximum(terrain_rgba[..., :3], lift_rgb)
        except Exception:
            pass
        # Apply exposure/gain to PT base before overlay
        try:
            ex = float(pt_exposure)
            if not np.isfinite(ex):
                ex = 0.85
            ex = float(np.clip(ex, 0.5, 1.1))
            if abs(ex - 1.0) > 1e-6:
                scaled = np.clip(terrain_rgba[..., :3].astype(np.float32) * ex, 0.0, 255.0).astype(np.uint8)
                terrain_rgba[..., :3] = scaled
        except Exception:
            pass
        # Optional auto exposure (only darken): match 95th percentile to target ~0.78
        try:
            if bool(pt_auto_exposure):
                rgbf = terrain_rgba[..., :3].astype(np.float32) / 255.0
                lum = 0.2126 * rgbf[..., 0] + 0.7152 * rgbf[..., 1] + 0.0722 * rgbf[..., 2]
                p95 = float(np.percentile(lum, 95.0)) if lum.size else 0.0
                target = 0.78
                if np.isfinite(p95) and p95 > 1e-6 and p95 > target:
                    k = max(0.55, min(1.0, target / p95))
                    rgbf *= k
                    terrain_rgba[..., :3] = np.clip(rgbf * 255.0 + 0.5, 0, 255).astype(np.uint8)
        except Exception:
            pass
        # Apply gamma to further tame highlights (gamma>1 darkens)
        try:
            g = float(pt_gamma)
            if np.isfinite(g) and abs(g - 1.0) > 1e-6:
                rgbf = terrain_rgba[..., :3].astype(np.float32) / 255.0
                rgbf = np.clip(rgbf, 0.0, 1.0) ** float(np.clip(g, 0.7, 1.6))
                terrain_rgba[..., :3] = np.clip(rgbf * 255.0 + 0.5, 0, 255).astype(np.uint8)
        except Exception:
            pass
    else:
        # Decide water overlay parameters
        wl = None
        wp = None
        mode = str(water_mode).lower()
        if mode == "percentile":
            wp = float(water_percentile) if (water_percentile is not None) else 30.0
        elif mode == "level":
            wl = float(water_level) if (water_level is not None) else None

        t0 = time.perf_counter()
        terrain_rgba = forge3d.render_raster(
            elevation,
            size=(width, height),
            renderer="hillshade",
            spacing=spacing,
            # Camera-style controls for oblique view
            camera_phi=65.0,
            camera_theta=45.0,
            camera_distance=2.4,
            # Lighting controls
            lighting_type=str(lighting_type),
            lighting_intensity=float(lighting_intensity),
            lighting_azimuth=float(lighting_azimuth),
            lighting_elevation=float(lighting_elevation),
            shadow_enabled=bool(shadow_enabled),
            shadow_intensity=float(shadow_intensity),
            # Tone controls
            gamma=float(gamma),
            equalize=bool(equalize),
            contrast_pct=float(contrast_pct),
            exaggeration=float(exaggeration),
            # Water overlay (off unless explicitly enabled)
            water_level=wl,
            water_level_percentile=wp,
            water_keep_components=int(water_keep_components),
            water_min_area_pct=float(water_min_area_pct),
            water_morph_iter=int(water_morph_iter),
            water_max_slope_deg=float(water_max_slope_deg),
            water_min_depth=float(water_min_depth),
            water_debug=bool(water_debug),
        )
        t1 = time.perf_counter()
        _ = t1  # suppress timing print in quiet mode
    
    # Use render_overlay to composite land cover texture
    # Apply land cover overlay
    
    # Convert texture to RGBA
    texture_rgba = np.zeros((texture_rgb.shape[0], texture_rgb.shape[1], 4), dtype=np.uint8)
    texture_rgba[:, :, :3] = texture_rgb
    if texture_classes is not None:
        # Transparent outside data (class 0), semi-transparent otherwise
        oa = int(np.clip(overlay_alpha, 0, 255))
        alpha = np.where(texture_classes == 0, 0, oa).astype(np.uint8)
        texture_rgba[:, :, 3] = alpha
    else:
        texture_rgba[:, :, 3] = int(np.clip(overlay_alpha, 0, 255))  # Lower alpha to reveal relief
    
    # Resize texture to match output size
    texture_img = Image.fromarray(texture_rgba).resize((width, height), Image.NEAREST)
    texture_resized = np.array(texture_img)
    
    # Improved blending: multiply land cover colors by shading to preserve 3D relief
    # This maintains both the categorical color identity AND the lighting gradients
    alpha = texture_resized[:, :, 3:4].astype(np.float32) / 255.0
    base = terrain_rgba[:, :, :3].astype(np.float32) / 255.0
    over = texture_resized[:, :, :3].astype(np.float32) / 255.0
    mode = str(overlay_blend).lower()
    
    # Extract shading from base (grayscale 3D relief with lighting)
    # Use luminance to get the lighting intensity independent of any base color
    base_lum = 0.2126 * base[..., 0] + 0.7152 * base[..., 1] + 0.0722 * base[..., 2]
    base_lum = np.clip(base_lum, 0.01, 10.0)  # Prevent division by zero, allow HDR
    
    # Tonemap the base HDR lighting first to get proper contrast
    # Simple Reinhard with slight toe to preserve shadow detail
    base_lum_tm = base_lum / (1.0 + base_lum * 0.3)
    
    # Scale to [0.15, 1.0] range to preserve shadow depth while avoiding pure black
    base_lum_norm = np.clip((base_lum_tm - base_lum_tm.min()) / (base_lum_tm.max() - base_lum_tm.min() + 1e-6), 0.0, 1.0)
    shading = 0.15 + 0.85 * base_lum_norm
    
    # If the terrain base is effectively black (failed render), use fallback
    try:
        b_p99 = float(np.percentile(base, 99.0))
        if b_p99 <= 0.02:
            shading = np.clip(shading + 0.25, 0.3, 1.0)  # Boost visibility
    except Exception:
        pass
    
    # Apply the shading to land cover colors (multiply blend preserves both)
    comp = over * shading[..., None]
    
    # Full alpha application - no more weak 39% that washes everything out
    mixed = alpha * comp + (1.0 - alpha) * base
    # Apply final tone controls: gamma>1 darkens, gain scales
    try:
        g = float(final_gamma)
        k = float(final_gain)
        g = float(np.clip(g, 0.7, 1.7))
        k = float(np.clip(k, 0.5, 1.2))
        if abs(g - 1.0) > 1e-6:
            mixed = np.clip(mixed, 0.0, 1.0) ** g
        if abs(k - 1.0) > 1e-6:
            mixed = np.clip(mixed * k, 0.0, 1.0)
    except Exception:
        pass
    final_rgb = np.clip(np.rint(mixed * 255.0), 0, 255).astype(np.uint8)
    final_rgba = terrain_rgba.copy()
    final_rgba[:, :, :3] = final_rgb
    # Normalize background based on overlay mask (class==0 => alpha=0)
    try:
        bg_color = np.array([242, 242, 242], dtype=np.uint8)  # light neutral gray
        # Rebuild the same texture_resized alpha used for blending
        if texture_classes is not None:
            oa = int(np.clip(overlay_alpha, 0, 255))
            alpha_mask = np.where(texture_classes == 0, 0, oa).astype(np.uint8)
            alpha_mask = np.array(Image.fromarray(alpha_mask).resize((width, height), Image.NEAREST))
            bg_mask = alpha_mask == 0
            final_rgba[bg_mask, 0] = bg_color[0]
            final_rgba[bg_mask, 1] = bg_color[1]
            final_rgba[bg_mask, 2] = bg_color[2]
    except Exception:
        pass

    # Optional unsharp mask to restore acutance after denoising/up/down-sampling
    if float(final_sharpen) > 1e-6:
        amount = float(np.clip(final_sharpen, 0.0, 2.5))
        # radius in pixels proportional to output size for stable look
        radius = max(0.5, min(2.0, 0.0015 * max(width, height)))
        try:
            img_pil = Image.fromarray(final_rgba)
            img_pil = img_pil.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(150*amount), threshold=1))
            img_pil.save(output_path)
        except Exception:
            Image.fromarray(final_rgba).save(output_path)
    else:
        Image.fromarray(final_rgba).save(output_path)


def composite_final(render_path: Path, legend_path: Path, output_path: Path):
    """Composite render with legend."""
    render = Image.open(render_path)
    legend = Image.open(legend_path)
    
    # Scale legend relative to render size; avoid upscaling the render to keep things fast
    target_legend_w = max(300, int(render.width * 0.22))
    legend = legend.resize((target_legend_w, int(target_legend_w * legend.height / legend.width)), Image.LANCZOS)
    
    # Composite
    x_offset, y_offset = 100, render.height - legend.height
    if legend.mode == 'RGBA':
        render.paste(legend, (x_offset, y_offset), legend)
    else:
        render.paste(legend, (x_offset, y_offset))
    
    render.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="3D Land Cover Visualization")
    parser.add_argument("--country", default="CH", help="Country code (e.g., CH, UA, HR)")
    parser.add_argument("--country-name", default="switzerland", help="Country name for files")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--width", type=int, default=2000, help="Output width")
    parser.add_argument("--height", type=int, default=2000, help="Output height")
    parser.add_argument("--refresh", action="store_true", help="Refetch remote rasters and rebuild mosaics")
    parser.add_argument("--renderer", default="auto", choices=["auto","hillshade","path_tracer"], help="Rendering backend")
    parser.add_argument("--frames", type=int, default=2, help="Path tracer accumulation frames when renderer=path_tracer or auto+GPU")
    parser.add_argument("--pt-scale", type=float, default=0.66, help="Internal scale for path tracer (speed vs quality)")
    parser.add_argument("--pt-mesh-sub", type=int, default=0, help="Subsample factor for heightmap before path tracer (0=auto based on align-max-dim, reduces triangles)")
    parser.add_argument("--pt-backend", type=str, default="auto", choices=["auto", "gpu", "cpu"], help="Path tracer backend preference")
    parser.add_argument("--pt-theta", type=float, default=45.0, help="Orbit theta (deg) around vertical axis for PT camera")
    parser.add_argument("--pt-phi", type=float, default=25.0, help="Orbit phi (deg) elevation for PT camera")
    parser.add_argument("--pt-denoiser", type=str, default="svgf", choices=["off","svgf"], help="Path tracer denoiser")
    parser.add_argument("--pt-svgf-iters", type=int, default=5, help="SVGF iterations when --pt-denoiser=svgf")
    parser.add_argument("--pt-exposure", type=float, default=0.85, help="Exposure for PT base (0.5..1.1, lower=darker)")
    ae = parser.add_mutually_exclusive_group()
    ae.add_argument("--pt-auto-exposure", dest="pt_auto_exposure", action="store_true", help="Enable auto exposure (darken only)")
    ae.add_argument("--no-pt-auto-exposure", dest="pt_auto_exposure", action="store_false", help="Disable auto exposure")
    parser.set_defaults(pt_auto_exposure=True)
    parser.add_argument("--pt-gamma", type=float, default=1.1, help="Gamma for PT base (gamma>1 darkens)")
    # PT crispness controls
    parser.add_argument("--pt-supersample", type=float, default=1.0, help="Supersampling factor for PT internal resolution (>=1)")
    parser.add_argument("--pt-luma-clamp", type=float, default=1.8, help="Luminance clamp for PT to prevent fireflies (None to disable)")
    parser.add_argument("--pt-detail-boost", type=float, default=0.3, help="Boost high-frequency hillshade detail into PT base (0..1)")
    parser.add_argument("--pt-hillshade-mix", type=float, default=0.5, help="Blend amount of crisp hillshade into PT base (0..1)")
    parser.add_argument("--pt-max-dim", type=int, default=0, help="Clamp PT internal texture size to this device-safe max (0 = unlimited)")
    # Lighting model & parameters (hillshade renderer)
    parser.add_argument(
        "--lighting-type",
        type=str,
        default="blinn-phong",
        choices=["lambertian", "flat", "phong", "blinn-phong"],
        help="Lighting model for hillshade renderer",
    )
    parser.add_argument("--lighting-intensity", type=float, default=0.9, help="Light intensity multiplier")
    parser.add_argument("--lighting-azimuth", type=float, default=315.0, help="Light azimuth (degrees, 0=N)")
    parser.add_argument("--lighting-elevation", type=float, default=20.0, help="Light elevation (degrees)")

    # Tone mapping and relief controls
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction")
    parser.add_argument("--contrast-pct", type=float, default=1.25, help="Percentile clip for normalization")
    parser.add_argument("--exaggeration", type=float, default=2.2, help="Vertical exaggeration (hillshade)")
    eq_group = parser.add_mutually_exclusive_group()
    eq_group.add_argument("--equalize", dest="equalize", action="store_true", help="Enable histogram equalization")
    eq_group.add_argument("--no-equalize", dest="equalize", action="store_false", help="Disable histogram equalization")
    parser.set_defaults(equalize=False)

    # Shadow controls
    shadow_group = parser.add_mutually_exclusive_group()
    shadow_group.add_argument("--shadows", dest="shadow_enabled", action="store_true", help="Enable shadows")
    shadow_group.add_argument("--no-shadows", dest="shadow_enabled", action="store_false", help="Disable shadows")
    parser.set_defaults(shadow_enabled=True)
    parser.add_argument("--shadow-intensity", type=float, default=1.2, help="Shadow strength [0..2]")

    # Overlay compositing controls
    parser.add_argument("--overlay-alpha", type=int, default=110, help="Overlay alpha (0..255)")
    parser.add_argument(
        "--overlay-blend",
        type=str,
        default="overlay",
        choices=["overlay", "screen", "multiply"],
        help="Blend mode for land cover over terrain",
    )
    # Final output tone controls
    parser.add_argument("--final-gamma", type=float, default=1.12, help="Final gamma on composite (gamma>1 darkens)")
    parser.add_argument("--final-gain", type=float, default=1.0, help="Final gain on composite (0.5..1.2)")
    parser.add_argument("--final-sharpen", type=float, default=0.0, help="Unsharp mask amount after compositing (0..2.5)")
    parser.add_argument("--align-max-dim", type=int, default=1200, help="Max dimension for alignment/resampling (speeds up pipeline)")
    # Water overlay controls (default OFF)
    parser.add_argument("--water", type=str, default="off", choices=["off", "percentile", "level"], help="Water overlay detection mode")
    parser.add_argument("--water-percentile", type=float, default=30.0, help="Percentile for water level when --water=percentile")
    parser.add_argument("--water-level", type=float, default=None, help="Absolute elevation for water when --water=level")
    parser.add_argument("--water-morph-iter", type=int, default=1, help="Morphological smoothing iterations")
    parser.add_argument("--water-min-area-pct", type=float, default=0.01, help="Min area percentage to keep as water")
    parser.add_argument("--water-keep-components", type=int, default=2, help="Keep N largest water components")
    parser.add_argument("--water-max-slope-deg", type=float, default=6.0, help="Discard water on slopes steeper than this")
    parser.add_argument("--water-min-depth", type=float, default=0.1, help="Minimum depth below level to keep")
    parser.add_argument("--water-debug", action="store_true", help="Print water overlay diagnostics")

    # Viewer/CLI options (ported from examples/geopandas_demo.py)
    parser.add_argument("--viewer", action="store_true", help="Open 2D image viewer for the final composite instead of saving only")
    parser.add_argument("--viewer-3d", action="store_true", help="Open interactive 3D terrain viewer with land cover texture and CLI")
    parser.add_argument("--viewer-subsample", type=int, default=4, help="Subsample factor for 3D mesh (1=full, 2=half, 4=quarter)")
    parser.add_argument("--viewer-vscale", type=float, default=1.2, help="Vertical exaggeration for 3D viewer")
    parser.add_argument("--demo-snapshots", action="store_true", help="Automatically capture snapshots from multiple camera angles in viewer-3d")
    parser.add_argument("--demo-snapshot-dir", type=Path, default=Path("snapshots"), help="Directory for demo snapshots")
    parser.add_argument("--cli-mode", type=str, default="process", choices=["process","thread"], help="CLI mode for viewer-3d")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"3D Land Cover: {args.country_name.title()}")
    
    # 1. Get country boundary
    print("Loading boundary...")
    country = get_country_boundary(args.country)
    bounds = country.total_bounds
    
    # 2. Fetch ESA WorldCover
    print("Fetching ESA WorldCover 2021...")
    lc_path = output_dir / f"landcover_{args.country_name}.tif"
    if args.refresh and lc_path.exists():
        lc_path.unlink(missing_ok=True)
    if not lc_path.exists():
        fetch_esa_worldcover(tuple(bounds), lc_path, country.geometry.iloc[0])
    
    # 3. Fetch elevation
    print("Fetching elevation (DEM)...")
    dem_path = output_dir / "elevation.tif"
    if args.refresh and dem_path.exists():
        dem_path.unlink(missing_ok=True)
    _ = fetch_elevation(tuple(bounds), output_dir, country.geometry.iloc[0])
    
    # 4. Align and convert land cover to RGB
    print("Aligning land cover to DEM grid...")
    elevation_data, texture_rgb, present_classes, texture_classes, spacing = align_landcover_to_dem(
        lc_path, dem_path, target_max_dim=int(args.align_max_dim), refresh=bool(args.refresh), cache_dir=output_dir
    )
    
    # 5. Create legend (include only classes present; skip 0)
    print("Creating legend...")
    legend_path = output_dir / "legend.png"
    if not present_classes:
        present_classes = [10, 40, 50, 80]
    create_legend(legend_path, ESA_COLORS, ESA_LABELS, selected=present_classes)
    
    # If interactive 3D viewer requested, launch with CLI support
    if args.viewer_3d:
        # Prepare CLI and raytrace helper similar to geopandas_demo
        cli_started = False
        stop_event = threading.Event()
        command_queue = None  # type: ignore
        response_queue = None  # type: ignore
        command_proc = None  # type: ignore
        command_reader_thread = None  # type: ignore
        command_stop = None  # type: ignore

        try:
            raytrace_helper = RaytraceMeshCache(
                elevation_data,
                spacing,
                subsample=max(1, int(args.viewer_subsample)),
                vertical_scale=float(args.viewer_vscale),
            )
        except Exception as exc:
            print(f"[WARN] Raytrace mesh unavailable: {exc}")
            raytrace_helper = None  # type: ignore

        # Start CLI before heavy viewer init for single-terminal control
        print(f"Interactive CLI ({'thread' if args.cli_mode=='thread' else 'process'}) started. Type 'help' for commands; 'quit' to exit.", flush=True)
        if args.cli_mode == "thread":
            import queue as threading_queue
            command_queue = threading_queue.Queue()
            response_queue = threading_queue.Queue()
            command_stop = threading.Event()

            command_reader_thread = threading.Thread(
                target=_command_reader,
                args=(command_queue, response_queue, command_stop),
                daemon=True,
            )
            command_reader_thread.start()

            control_thread = threading.Thread(
                target=interactive_control_loop,
                kwargs=dict(
                    dem_path=str(dem_path),
                    dem_data=elevation_data,
                    spacing=spacing,
                    raytrace_helper=raytrace_helper,
                    default_size=(args.width, args.height),
                    stop_event=stop_event,
                    command_queue=command_queue,
                    response_queue=response_queue,
                    command_stop=command_stop,
                ),
                daemon=True,
            )
            control_thread.start()
        else:
            command_queue = mp.Queue()
            response_queue = mp.Queue()
            command_stop = mp.Event()

            command_proc = mp.Process(
                target=_command_reader,
                args=(command_queue, response_queue, command_stop),
                daemon=True,
            )
            command_proc.start()

            control_thread = threading.Thread(
                target=interactive_control_loop,
                kwargs=dict(
                    dem_path=str(dem_path),
                    dem_data=elevation_data,
                    spacing=spacing,
                    raytrace_helper=raytrace_helper,
                    default_size=(args.width, args.height),
                    stop_event=stop_event,
                    command_queue=command_queue,
                    response_queue=response_queue,
                    command_stop=command_stop,
                ),
                daemon=True,
            )
            control_thread.start()
        cli_started = True

        # Build viewer texture (RGBA with NoData transparent)
        texture_rgba_view = np.zeros((texture_rgb.shape[0], texture_rgb.shape[1], 4), dtype=np.uint8)
        texture_rgba_view[:, :, :3] = texture_rgb
        if texture_classes is not None:
            texture_rgba_view[:, :, 3] = np.where(texture_classes == 0, 0, 255).astype(np.uint8)
        else:
            texture_rgba_view[:, :, 3] = 255

        # Optional: schedule demo snapshots shortly after viewer opens
        if args.demo_snapshots:
            print(f"[DEMO] Capturing snapshots to {args.demo_snapshot_dir}...", flush=True)
            def run_demo_after_delay():
                time.sleep(3)
                try:
                    demo_snapshots(args.demo_snapshot_dir, width=args.width, height=args.height)
                except Exception as e:
                    print(f"[DEMO] Error: {e}")
            demo_thread = threading.Thread(target=run_demo_after_delay, daemon=True)
            demo_thread.start()

        # Open the interactive 3D viewer with texture
        try:
            forge3d.open_terrain_viewer_3d(
                elevation_data,
                texture_rgba=texture_rgba_view,
                spacing=spacing,
                vertical_scale=args.viewer_vscale,
                subsample=args.viewer_subsample,
                width=args.width,
                height=args.height,
                title=f"3D Land Cover - {args.country_name.title()}",
            )
        except Exception as exc:
            print(f"Failed to open 3D viewer: {exc}")
            return 1
        finally:
            stop_event.set()
            if command_stop is not None:
                try:
                    command_stop.set()
                except Exception:
                    pass
            if command_queue is not None:
                try:
                    command_queue.put(None, timeout=0.1)
                except Exception:
                    pass
            if response_queue is not None:
                try:
                    response_queue.put(None)
                except Exception:
                    pass
            try:
                control_thread.join(timeout=1.0)  # type: ignore
            except Exception:
                pass
            if command_proc is not None:
                command_proc.join(timeout=1.0)
            if command_reader_thread is not None:
                command_reader_thread.join(timeout=1.0)
        return 0

    # Offline render path
    print("Rendering 3D terrain...")
    render_path = output_dir / f"3d_{args.country_name}.png"
    render_3d_terrain(
        elevation_data,
        texture_rgb,
        texture_classes,
        render_path,
        args.width,
        args.height,
        max_elev_size=int(args.align_max_dim),
        renderer=args.renderer,
        frames=args.frames,
        pt_scale=args.pt_scale,
        pt_mesh_sub=args.pt_mesh_sub,
        pt_backend=args.pt_backend,
        pt_orbit_theta=args.pt_theta,
        pt_orbit_phi=args.pt_phi,
        pt_denoiser=args.pt_denoiser,
        pt_svgf_iters=args.pt_svgf_iters,
        pt_exposure=args.pt_exposure,
        pt_auto_exposure=args.pt_auto_exposure,
        pt_gamma=args.pt_gamma,
        pt_supersample=args.pt_supersample,
        pt_luma_clamp=(None if args.pt_luma_clamp is None else args.pt_luma_clamp),
        pt_hillshade_mix=args.pt_hillshade_mix,
        pt_max_dim=(None if (args.pt_max_dim is None or int(args.pt_max_dim) <= 0) else int(args.pt_max_dim)),
        spacing=spacing,
        lighting_type=args.lighting_type,
        lighting_intensity=args.lighting_intensity,
        lighting_azimuth=args.lighting_azimuth,
        lighting_elevation=args.lighting_elevation,
        gamma=args.gamma,
        equalize=args.equalize,
        contrast_pct=args.contrast_pct,
        exaggeration=args.exaggeration,
        shadow_enabled=args.shadow_enabled,
        shadow_intensity=args.shadow_intensity,
        overlay_alpha=args.overlay_alpha,
        overlay_blend=args.overlay_blend,
        final_gamma=args.final_gamma,
        final_gain=args.final_gain,
        final_sharpen=args.final_sharpen,
        # detail boost for crispness without heavy PT cost
        pt_detail_boost=args.pt_detail_boost,
        water_mode=args.water,
        water_percentile=args.water_percentile,
        water_level=args.water_level,
        water_morph_iter=args.water_morph_iter,
        water_min_area_pct=args.water_min_area_pct,
        water_keep_components=args.water_keep_components,
        water_max_slope_deg=args.water_max_slope_deg,
        water_min_depth=args.water_min_depth,
        water_debug=args.water_debug,
    )

    # Composite final
    print("Creating final composite...")
    final_path = output_dir / f"3d_{args.country_name}_final.png"
    composite_final(render_path, legend_path, final_path)

    if args.viewer:
        try:
            forge3d.open_viewer_image(
                np.array(Image.open(final_path).convert('RGBA')),
                width=args.width,
                height=args.height,
                title=f"Land Cover - {args.country_name.title()}",
                vsync=True,
            )
        except Exception as exc:
            print(f"Failed to open image viewer: {exc}")

    print(f"✓ Complete: {final_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
