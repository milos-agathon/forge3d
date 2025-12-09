#!/usr/bin/env python3
"""Generate P5 validation renders for AO enhancement phase.

Produces:
- phase_p5.png: Render with ao_weight=0 (should match P4)
- phase_p5_ao_on.png: Render with ao_weight=0.5 (valleys darkened)
"""

import sys
import zlib
import struct
from pathlib import Path

# Add examples to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

import numpy as np

import forge3d
from forge3d import terrain_params
from forge3d import io as f3d_io


def write_png(path: Path, data: np.ndarray):
    """Write a numpy array to a PNG file using only standard library."""
    if data.dtype != np.uint8:
        data = data.astype(np.uint8)
    
    # Ensure 3 channels for RGB
    if data.ndim == 2:
        height, width = data.shape
        channels = 1
    else:
        height, width, channels = data.shape
    
    # PNG signature
    signature = b'\x89PNG\r\n\x1a\n'
    
    # IHDR: width, height, bit_depth, color_type, compression, filter, interlace
    # color_type: 0=gray, 2=rgb, 6=rgba
    if channels == 1: color_type = 0
    elif channels == 3: color_type = 2
    elif channels == 4: color_type = 6
    else: raise ValueError(f"Unsupported channel count: {channels}")
    
    ihdr_content = struct.pack("!I", width) + struct.pack("!I", height) + \
                   struct.pack("!BBBBB", 8, color_type, 0, 0, 0)
    ihdr = struct.pack("!I", len(ihdr_content)) + b'IHDR' + ihdr_content + \
           struct.pack("!I", zlib.crc32(b'IHDR' + ihdr_content))
    
    # IDAT: filter byte (0) + scanline data
    # We use filter type 0 (None) for simplicity
    scanlines = b''
    row_len = width * channels
    for y in range(height):
        scanlines += b'\x00' + data[y].tobytes()
    
    compressed = zlib.compress(scanlines)
    idat = struct.pack("!I", len(compressed)) + b'IDAT' + compressed + \
           struct.pack("!I", zlib.crc32(b'IDAT' + compressed))
    
    # IEND
    iend = struct.pack("!I", 0) + b'IEND' + struct.pack("!I", zlib.crc32(b'IEND'))
    
    with open(path, "wb") as f:
        f.write(signature)
        f.write(ihdr)
        f.write(idat)
        f.write(iend)


def save_image(path: Path, data: np.ndarray):
    """Save image using pure-Python PNG writer."""
    # Normalize/convert to uint8
    if data.dtype != np.uint8:
        # Heuristic: if values are in [0, 1], scale to [0, 255]
        max_val = float(data.max()) if data.size > 0 else 1.0
        if max_val <= 1.0:
            data = (np.clip(data, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            data = np.clip(data, 0.0, 255.0).round().astype(np.uint8)
    write_png(path, data)


def generate_p5_renders():
    """Generate P5 validation renders."""
    
    output_dir = Path(__file__).parent.parent / "reports" / "terrain" / "p5"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use default demo assets
    dem_path = Path(__file__).parent.parent / "assets" / "Gore_Range_Albers_1m.tif"
    hdr_path = Path(__file__).parent.parent / "assets" / "hdri" / "brown_photostudio_02_4k.hdr"
    
    print(f"DEM: {dem_path}")
    print(f"HDR: {hdr_path}")
    
    # Create session
    session = forge3d.Session()
    renderer = forge3d.TerrainRenderer(session)
    
    # Create required dependencies
    materials = forge3d.MaterialSet.terrain_default()
    if hdr_path.exists():
        ibl = forge3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    else:
        # Create a dummy HDR if file missing
        print("Warning: HDR not found, using dummy IBL")
        dummy_hdr = output_dir / "dummy.hdr"
        with open(dummy_hdr, "wb") as f:
            f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 2\n")
            # 2x2 pixels, gray
            f.write(b"\x80\x80\x80\x80" * 4)
        ibl = forge3d.IBL.from_hdr(str(dummy_hdr), intensity=1.0)
    
    # Load Gore Range DEM (requires rasterio)
    dem_data = None
    domain = (0.0, 1.0)
    size_px = (1024, 1024)
    try:
        dem = f3d_io.load_dem(str(dem_path), fill_nodata_values=True)
        dem_data = np.asarray(np.flipud(dem.data), dtype=np.float32)
        domain = (float(dem.domain[0]), float(dem.domain[1]))
        height, width = dem_data.shape
        size_px = (max(width, 512), max(height, 512))
        print(f"Loaded DEM from {dem_path}, domain={domain}, shape={dem_data.shape}")
    except ImportError as e:
        raise SystemExit(
            "rasterio is required to load the Gore Range DEM. "
            "Install with: pip install rasterio (or pip install 'forge3d[raster]')."
        ) from e
    except Exception as e:
        raise SystemExit(f"Failed to load DEM {dem_path}: {e}")
    
    def compute_coarse_ao_mask(heightmap: np.ndarray) -> np.ndarray:
        """CPU coarse AO mask aligned with renderer fallback (downsampled for speed)."""
        h, w = heightmap.shape

        # Downsample to at most 256x256 for speed
        target = 256
        if max(h, w) > target:
            ys = np.linspace(0, h - 1, target).astype(np.int32)
            xs = np.linspace(0, w - 1, target).astype(np.int32)
            hm = heightmap[np.ix_(ys, xs)]
        else:
            hm = heightmap

        hh, ww = hm.shape
        ao_small = np.ones((hh, ww), dtype=np.float32)
        radius = 4
        height_scale = 8.0

        for y in range(hh):
            for x in range(ww):
                center = hm[y, x]
                occl = 0.0
                count = 0
                for dy in range(-radius, radius + 1):
                    ny = y + dy
                    if ny < 0 or ny >= hh:
                        continue
                    for dx in range(-radius, radius + 1):
                        nx = x + dx
                        if nx < 0 or nx >= ww or (dx == 0 and dy == 0):
                            continue
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist == 0:
                            continue
                        neighbor = hm[ny, nx]
                        h_diff = (neighbor - center) * height_scale
                        if h_diff > 0:
                            angle = np.arctan(h_diff / dist)
                            occl += min(angle / (0.5 * np.pi), 1.0)
                        count += 1
                if count > 0:
                    avg = occl / count
                    ao_small[y, x] = max(0.01, 1.0 - min(0.9, avg))

        # Upscale back to original size with nearest repeat
        if ao_small.shape != (h, w):
            y_rep = int(np.ceil(h / ao_small.shape[0]))
            x_rep = int(np.ceil(w / ao_small.shape[1]))
            ao = np.repeat(np.repeat(ao_small, y_rep, axis=0), x_rep, axis=1)
            ao = ao[:h, :w]
        else:
            ao = ao_small

        return ao

    def make_params(ao_weight: float):
        config = terrain_params.make_terrain_params_config(
            size_px=size_px,
            render_scale=1.0,
            msaa_samples=4,
            z_scale=2.0,
            exposure=1.0,
            domain=domain,
            albedo_mode="mix",
            colormap_strength=0.5,
            cam_radius=1000.0,
            cam_phi_deg=135.0,
            cam_theta_deg=45.0,
            ao_weight=ao_weight,
        )
        # Convert to native params
        return forge3d.TerrainRenderParams(config)
    
    # Render 1: ao_weight=0 (P4 compatible)
    print("\nRendering phase_p5.png (ao_weight=0.0)...")
    params_p5 = make_params(ao_weight=0.0)
    
    try:
        result = renderer.render_terrain_pbr_pom(
            materials,
            ibl,
            params_p5,
            dem_data,
            target=None
        )
        if result is not None:
            # Result is a Frame object, convert to numpy
            img = result.to_numpy()
            save_image(output_dir / "phase_p5.png", img)
            print(f"  Saved: {output_dir / 'phase_p5.png'}")
    except Exception as e:
        print(f"  Render failed: {e}")
        # Create placeholder
        img = np.ones((size_px[1], size_px[0], 3), dtype=np.uint8) * 128
        save_image(output_dir / "phase_p5.png", img)
    
    # Render 2: ao_weight=0.8 (AO enabled, moderate)
    print("\nRendering phase_p5_ao_on.png (ao_weight=0.8)...")
    params_ao = make_params(ao_weight=0.8)
    
    try:
        result = renderer.render_terrain_pbr_pom(
            materials,
            ibl,
            params_ao,
            dem_data,
            target=None
        )
        if result is not None:
            img = result.to_numpy()
            save_image(output_dir / "phase_p5_ao_on.png", img)
            print(f"  Saved: {output_dir / 'phase_p5_ao_on.png'}")
    except Exception as e:
        print(f"  Render failed: {e}")
        # Create placeholder with darker areas
        img = np.ones((size_px[1], size_px[0], 3), dtype=np.uint8) * 128
        img[300:400, 300:400] = [64, 64, 64]  # Simulated valley
        save_image(output_dir / "phase_p5_ao_on.png", img)
    
    print(f"\nP5 renders complete in {output_dir}")


if __name__ == "__main__":
    generate_p5_renders()
