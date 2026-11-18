#!/usr/bin/env python3
"""Reproduce the Swiss land-cover gallery scene through Studio's offscreen path.

Run from studio-tauri:  ../.venv/bin/python experiments/swiss-land-cover.py

Findings (2026-11-03):
- Raster overlays work through this path: the 3-band land-cover raster drapes
  with its colours preserved (lakes, forest, urban).
- `radius = 18000` in the gallery snippet is a ~18 km close-up; the engine clamps
  view coordinates to +/-65,536 units, so a 353 km country cannot be framed in
  one shot. Country-scale maps need either a scaled scene (units != metres) or an
  engine change to normalise the view.
- Cost at 100 m data, 60 km view: ~1.7 s (terrain upload dominates).


Equivalent of:
    with f3d.open_viewer_async(terrain_path=f3d.fetch_dem("swiss")) as viewer:
        viewer.load_overlay("land-cover", f3d.fetch_dataset("swiss-land-cover"),
                            opacity=0.82, preserve_colors=True)
        viewer.set_orbit_camera(phi_deg=90, theta_deg=10, radius=18000)
        viewer.set_sun(azimuth_deg=315, elevation_deg=17)
        viewer.snapshot("swiss-land-cover.png")

Differences forced by being windowless: the overlay is passed to the render
params as an RGBA layer (what load_overlay does internally), and the radius is
given in metres for the reprojected DEM rather than viewer units.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

import forge3d as f3d
from forge3d.terrain_params import (
    AovSettings,
    DetailSettings,
    HeightAoSettings,
    PomSettings,
    SkySettings,
    make_terrain_params_config,
)

ROOT = Path("/home/adeerkhan/github/forge3d")
DEM = ROOT / "assets/tif/switzerland_dem.tif"
COVER = ROOT / "assets/tif/switzerland_land_cover.tif"
OUT = Path("/tmp/opencode/swiss-land-cover-offscreen.png")
SKY = Path("/tmp/opencode/studio_sky.hdr")

TARGET_CRS = "EPSG:2056"
RESOLUTION_M = 100.0
OPACITY = 0.82


def ensure_sky() -> None:
    if SKY.exists():
        return
    height, width = 64, 128
    v = np.linspace(0.0, 1.0, height)[:, None]
    zenith = np.array([0.25, 0.40, 0.75])
    horizon = np.array([0.85, 0.72, 0.55])
    ground = np.array([0.10, 0.09, 0.08])
    sky = horizon[None, None, :] * (1.0 - v[..., None]) + zenith[None, None, :] * v[..., None]
    image = np.where(v[..., None] > 0.5, sky, np.broadcast_to(ground, sky.shape))
    image = np.broadcast_to((image.astype(np.float32) * 1.6), (height, width, 3)).copy()
    peak = image.max(axis=2)
    exponent = np.zeros_like(peak)
    nonzero = peak > 1e-32
    _, exp = np.frexp(peak)
    exponent[nonzero] = exp[nonzero]
    scale = np.where(nonzero, 256.0 / np.exp2(exponent), 0.0)
    rgb = np.clip(image * scale[..., None], 0, 255).astype(np.uint8)
    rgbe = np.concatenate([rgb, exponent.astype(np.uint8)[..., None]], axis=2)
    SKY.write_bytes(
        b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n"
        + f"-Y {height} +X {width}\n".encode()
        + rgbe.tobytes()
    )


def to_metric(path: Path) -> tuple[np.ndarray, rasterio.Affine, tuple[float, float]]:
    """Reproject a raster to LV95 metres at RESOLUTION_M."""
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    with rasterio.open(path) as src:
        left, bottom, right, top = transform_bounds(src.crs, TARGET_CRS, *src.bounds)
        width = max(1, int(np.ceil((right - left) / RESOLUTION_M)))
        height = max(1, int(np.ceil((top - bottom) / RESOLUTION_M)))
        transform = from_origin(left, top, RESOLUTION_M, RESOLUTION_M)

        data = np.zeros((src.count, height, width), dtype=np.float32)
        for band in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band),
                destination=data[band - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.bilinear,
            )
        return data, transform, (width * RESOLUTION_M, height * RESOLUTION_M)


def main() -> None:
    ensure_sky()

    started = time.perf_counter()
    dem_bands, transform, (span_x, span_y) = to_metric(DEM)
    heights = dem_bands[0]
    nodata = heights < -1000.0
    if nodata.any():
        heights[nodata] = float(np.nanmin(heights[~nodata]))
    print(
        f"DEM {heights.shape[1]}x{heights.shape[0]} span {span_x/1000:.0f}x{span_y/1000:.0f} km "
        f"range {heights.min():.0f}-{heights.max():.0f} m"
    )

    cover_bands, _, _ = to_metric(COVER)
    # 3-band float raster whose values are 0-255; stretch/clip and drape it.
    rgb = np.moveaxis(cover_bands[:3], 0, -1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    alpha = np.where(rgb.max(axis=2) > 0, 255, 0).astype(np.uint8)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=2)
    print(f"land cover {rgba.shape[1]}x{rgba.shape[0]}")

    session = f3d.Session(window=False, backend="vulkan")
    renderer = f3d.TerrainRenderer(session)
    ibl = f3d.IBL.from_hdr(str(SKY), intensity=1.0)

    width, height = 1600, 1000
    radius = 18_000.0
    settings = make_terrain_params_config(
        size_px=(width, height),
        render_scale=1.0,
        msaa_samples=1,
        terrain_span=span_x,
        z_scale=1.0,
        camera_mode="mesh:zup:north:dem",
        cam_radius=radius,
        cam_phi_deg=90.0,
        cam_theta_deg=10.0,
        # The renderer's uniform ranges require terrain centred near the origin.
        cam_target=(0.0, 0.0, float(heights.mean())),
        fov_y_deg=45.0,
        clip=(10.0, 900_000.0),
        domain=(float(heights.min()), float(heights.max())),
        overlays=[f3d.OverlayLayer.from_raster_rgba(rgba, strength=OPACITY)],
        albedo_mode="colormap",
        hue_variation_strength=0.0,
        colormap_strength=1.0,
        light_azimuth_deg=315.0,
        light_elevation_deg=17.0,
        sun_intensity=3.0,
        ibl_enabled=True,
        exposure=1.35,
        ao_weight=0.0,
        pom=PomSettings(True, "Occlusion", 0.3, 8, 32, 4, False, False),
        detail=DetailSettings(enabled=False),
        height_ao=HeightAoSettings(enabled=False),
        aov=AovSettings(enabled=False),
        sky=SkySettings(enabled=True),
    )
    settings.shadows.enabled = True
    settings.shadows.technique = "PCF"

    from PIL import Image

    def render(radius_m: float, out: Path) -> None:
        s2 = make_terrain_params_config(
            size_px=(width, height), render_scale=1.0, msaa_samples=1,
            terrain_span=span_x, z_scale=1.0, camera_mode="mesh:zup:north:dem",
            cam_radius=radius_m, cam_phi_deg=90.0, cam_theta_deg=10.0,
            cam_target=(0.0, 0.0, float(heights.mean())), fov_y_deg=45.0,
            clip=(10.0, 2_500_000.0),
            domain=(float(heights.min()), float(heights.max())),
            overlays=[f3d.OverlayLayer.from_raster_rgba(rgba, strength=OPACITY)],
            albedo_mode="colormap", hue_variation_strength=0.0, colormap_strength=1.0,
            light_azimuth_deg=315.0, light_elevation_deg=17.0, sun_intensity=3.0,
            ibl_enabled=True, exposure=1.35, ao_weight=0.0,
            pom=PomSettings(True, "Occlusion", 0.3, 8, 32, 4, False, False),
            detail=DetailSettings(enabled=False), height_ao=HeightAoSettings(enabled=False),
            aov=AovSettings(enabled=False), sky=SkySettings(enabled=True))
        s2.shadows.enabled = True
        s2.shadows.technique = "PCF"
        t0 = time.perf_counter()
        frame = renderer.render_terrain_pbr_pom(
            f3d.MaterialSet.terrain_default(), ibl, f3d.TerrainRenderParams(s2), heights
        )
        image = np.asarray(frame.to_numpy())[:, :, :3].astype(np.uint8)
        Image.fromarray(image).save(out)
        print(f"{out.name}: radius {radius_m/1000:.1f} km, {(time.perf_counter()-t0)*1000:.0f} ms")

    render(60_000.0, OUT.with_name("swiss-60km.png"))
    render(18_000.0, OUT)
    print(f"total {(time.perf_counter()-started):.1f} s")


if __name__ == "__main__":
    main()
