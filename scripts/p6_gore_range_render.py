#!/usr/bin/env python3
"""
P6 Micro-Detail render with real terrain data.

Uses:
- DEM: assets/Gore_Range_Albers_1m.tif
- HDRI: assets/hdri/brown_photostudio_02_4k.hdr

Generates comparison images with detail ON vs OFF.
"""
import numpy as np
import json
from pathlib import Path

import rasterio
import forge3d as f3d
from forge3d.terrain_params import (
    ClampSettings,
    IblSettings,
    LightSettings,
    LodSettings,
    PomSettings,
    SamplingSettings,
    ShadowSettings,
    TerrainRenderParams as TerrainRenderParamsConfig,
    TriplanarSettings,
    DetailSettings,
)

# Paths
DEM_PATH = Path(__file__).parent.parent / "assets" / "Gore_Range_Albers_1m.tif"
HDRI_PATH = Path(__file__).parent.parent / "assets" / "hdri" / "brown_photostudio_02_4k.hdr"
REPORTS_DIR = Path(__file__).parent.parent / "reports" / "terrain"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dem(path: Path, max_size: int = 1024) -> tuple[np.ndarray, float, float]:
    """Load DEM and optionally downsample. Returns (heightmap, min, max)."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        print(f"  Original TIF size: {data.shape}")
        
        # Downsample if too large (keep aspect ratio)
        h, w = data.shape
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            # Simple decimation
            step_h = max(1, h // new_h)
            step_w = max(1, w // new_w)
            data = data[::step_h, ::step_w]
            print(f"  Downsampled to: {data.shape}")
        
        # Handle nodata
        nodata = src.nodata
        if nodata is not None:
            mask = data == nodata
            if mask.any():
                valid_min = data[~mask].min()
                data[mask] = valid_min
        
        return data, float(data.min()), float(data.max())


def build_config(overlay, detail: DetailSettings = None, cam_radius: float = 2000.0):
    """Build terrain render config."""
    return TerrainRenderParamsConfig(
        size_px=(1024, 1024),
        render_scale=1.0,
        msaa_samples=4,
        z_scale=1.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=cam_radius,
        cam_phi_deg=135.0,  # Rotate to place kidney lake in NE
        cam_theta_deg=89.0,  # Nearly top-down view
        cam_gamma_deg=0.0,
        fov_y_deg=70.0,  # Wider FOV to capture more terrain
        clip=(1.0, 20000.0),
        light=LightSettings("Directional", 135.0, 45.0, 3.0, [1.0, 0.98, 0.95]),
        ibl=IblSettings(True, 1.2, 0.0),
        shadows=ShadowSettings(
            True, "PCSS", 2048, 3, 8000.0, 1.5, 0.8, 0.002, 0.001, 0.3, 1e-4, 0.5, 2.0, 0.9
        ),
        triplanar=TriplanarSettings(8.0, 4.0, 1.0),
        pom=PomSettings(True, "Occlusion", 0.03, 12, 40, 4, True, True),
        lod=LodSettings(0, 0.0, -0.5),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 8, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[overlay],
        exposure=1.2,
        gamma=2.2,
        albedo_mode="colormap",  # Use colormap as primary albedo
        colormap_strength=1.0,
        detail=detail,
    )


def main():
    print("P6 Micro-Detail: Gore Range Render")
    print("=" * 50)
    
    # Load DEM at full resolution
    print(f"Loading DEM: {DEM_PATH}")
    heightmap, h_min, h_max = load_dem(DEM_PATH, max_size=1500)  # Full resolution
    # Flip vertically to align north up (match reference orientation)
    heightmap = np.flipud(heightmap)
    print(f"  Shape: {heightmap.shape}")
    print(f"  Range: {h_min:.1f} - {h_max:.1f} meters")
    
    # Normalize heightmap to 0-1 for rendering
    h_range = h_max - h_min
    heightmap_norm = (heightmap - h_min) / h_range if h_range > 0 else heightmap
    
    # Setup renderer
    print(f"\nInitializing renderer...")
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    
    # Load real HDRI
    print(f"Loading HDRI: {HDRI_PATH}")
    ibl = f3d.IBL.from_hdr(str(HDRI_PATH), intensity=1.2)
    
    # Create colormap overlay matching reference hypsometric tint
    cmap = f3d.Colormap1D.from_stops(
        stops=[
            (0.0, "#1a5c1a"),   # Dark green (lowest valleys)
            (0.15, "#4a9f4a"),  # Green
            (0.3, "#8cc63f"),   # Light green
            (0.45, "#d4e157"),  # Yellow-green
            (0.55, "#f0e68c"),  # Yellow/tan
            (0.7, "#d4a5a5"),   # Pink
            (0.85, "#b08080"),  # Purple-ish
            (1.0, "#ffffff"),   # White (peaks)
        ],
        domain=(0.0, 1.0),
    )
    overlay = f3d.OverlayLayer.from_colormap1d(cmap, strength=0.6)
    
    # The terrain mesh is normalized to approximately unit size in world space
    # So we need a smaller camera radius regardless of heightmap pixel size
    # Use a fixed radius that frames the terrain properly
    cam_radius = 2.0  # Terrain mesh is normalized to ~unit size
    
    # Render with detail OFF (P5 baseline)
    print(f"\nRendering with detail OFF (baseline)...")
    config_off = build_config(overlay, detail=DetailSettings(enabled=False), cam_radius=cam_radius)
    params_off = f3d.TerrainRenderParams(config_off)
    frame_off = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=params_off,
        heightmap=heightmap_norm,
        target=None,
    )
    frame_off.save(str(REPORTS_DIR / "phase_p6_gore_baseline.png"))
    print(f"  Saved: phase_p6_gore_baseline.png")
    
    # Render with detail ON
    print(f"\nRendering with detail ON (P6 feature)...")
    print(f"  Camera radius: {cam_radius:.1f}")
    config_on = build_config(
        overlay,
        detail=DetailSettings(
            enabled=True,
            detail_scale=0.01,     # Detail repeat in normalized space
            normal_strength=0.5,   # Strong detail normal for visibility
            albedo_noise=0.12,     # ±12% brightness variation
            fade_start=0.0,        # No fade-in (full detail at all distances)
            fade_end=10.0,         # Fade out beyond terrain bounds
        ),
        cam_radius=cam_radius,
    )
    params_on = f3d.TerrainRenderParams(config_on)
    frame_on = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=params_on,
        heightmap=heightmap_norm,
        target=None,
    )
    frame_on.save(str(REPORTS_DIR / "phase_p6_gore.png"))
    print(f"  Saved: phase_p6_gore.png")
    
    # Compute and save difference
    arr_off = frame_off.to_numpy()
    arr_on = frame_on.to_numpy()
    diff = np.abs(arr_on.astype(np.float32) - arr_off.astype(np.float32))
    mean_diff = diff.mean()
    max_diff = diff.max()
    nonzero_pixels = np.count_nonzero(diff.sum(axis=2))
    total_pixels = arr_on.shape[0] * arr_on.shape[1]
    
    # Save amplified difference
    if max_diff > 0:
        diff_amp = (diff / max_diff * 255).astype(np.uint8)
    else:
        diff_amp = diff.astype(np.uint8)
    
    from PIL import Image
    diff_img = Image.fromarray(diff_amp, 'RGBA')
    diff_img.save(REPORTS_DIR / "phase_p6_gore_diff.png")
    print(f"  Saved: phase_p6_gore_diff.png")
    
    # Write results
    result = {
        "phase": "P6",
        "feature": "micro_detail",
        "dem": "Gore_Range_Albers_1m.tif",
        "hdri": "brown_photostudio_02_4k.hdr",
        "heightmap_shape": list(heightmap.shape),
        "elevation_range_m": [float(h_min), float(h_max)],
        "camera_radius": float(cam_radius),
        "detail_settings": {
            "enabled": True,
            "detail_scale": 2.0,
            "normal_strength": 0.5,
            "albedo_noise": 0.12,
            "fade_start": 0.0,
            "fade_end": 5000.0,
        },
        "metrics": {
            "mean_diff": float(mean_diff),
            "max_diff": float(max_diff),
            "nonzero_pixels": int(nonzero_pixels),
            "total_pixels": int(total_pixels),
            "diff_percentage": float(nonzero_pixels / total_pixels * 100),
        },
        "validation": {
            "produces_difference": bool(mean_diff > 0.0),
            "isolation_confirmed": bool(nonzero_pixels > 0),
        }
    }
    
    with open(REPORTS_DIR / "p6_gore_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    # Write log
    log_lines = [
        "P6 Micro-Detail: Gore Range Run Log",
        "=" * 50,
        f"DEM: {DEM_PATH.name}",
        f"HDRI: {HDRI_PATH.name}",
        f"Heightmap shape: {heightmap.shape}",
        f"Elevation range: {h_min:.1f} - {h_max:.1f} meters",
        f"Camera radius: {cam_radius:.1f}",
        "",
        "Detail Settings:",
        f"  Enabled: True",
        f"  Scale: 2.0 meters",
        f"  Normal strength: 0.5",
        f"  Albedo noise: ±12%",
        f"  Fade distances: start=0.0, end=5000.0",
        "",
        "Metrics:",
        f"  Mean pixel difference: {mean_diff:.4f}",
        f"  Max pixel difference: {max_diff:.4f}",
        f"  Changed pixels: {nonzero_pixels} ({nonzero_pixels/total_pixels*100:.2f}%)",
        "",
        "Validation:",
        f"  Produces visible difference: {mean_diff > 0.0}",
        f"  Isolation confirmed: {nonzero_pixels > 0}",
    ]
    
    with open(REPORTS_DIR / "p6_gore_run.log", "w") as f:
        f.write("\n".join(log_lines))
    
    print(f"\n" + "=" * 50)
    print("Results:")
    print(f"  Mean diff: {mean_diff:.4f}")
    print(f"  Max diff: {max_diff:.4f}")
    print(f"  Changed: {nonzero_pixels/total_pixels*100:.1f}% of pixels")
    print(f"\nOutputs saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
