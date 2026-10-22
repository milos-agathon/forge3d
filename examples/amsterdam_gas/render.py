"""Render one frame of the Amsterdam-Noord gas map as a building heightfield.

Encoding (redundant on purpose, so the map survives colour-vision deficiency
and greyscale printing):
    height  = gas burned per home (m3/year) of the building's postcode cluster
    colour  = the same value on a lightness ramp (cream = little, deep red = a lot)
    blue    = gas-free building (flat plinth), a separate category
    grey    = building with no small-connection data (context only)

Uses camera_mode="mesh:zup:north:dem" so every 1 m height sample is a mesh
vertex and building walls stay sharp.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import forge3d as f3d
from forge3d.terrain_params import (
    AovSettings, DetailSettings, HeightAoSettings, PomSettings, make_terrain_params_config,
)

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "amsterdam_gas"
HDR = HERE.parents[1] / "assets" / "hdri" / "forge3d_neutral_daylight_v1.hdr"

# --- encoding -----------------------------------------------------------------
GROUND = (46, 49, 55)
WATER = (14, 22, 32)
CONTEXT = (100, 104, 112)
GAS_FREE = (59, 139, 212)
# m3 per home -> sRGB. Monotone lightness from cream to deep red.
RAMP = [(100, (251, 230, 197)), (600, (245, 185, 113)), (1200, (234, 123, 60)),
        (1800, (208, 76, 48)), (2500, (160, 40, 38))]
GAS_CAP = 2500.0
METRES_PER_M3 = 0.03     # 1000 m3/home -> 30 m tall
PLINTH = 2.0             # gas-free buildings and the base of every data building
CONTEXT_HEIGHT = 5.0
WATER_LEVEL = -1.5


def ramp(values: np.ndarray) -> np.ndarray:
    xs = np.array([s for s, _ in RAMP], dtype=np.float32)
    cs = np.array([c for _, c in RAMP], dtype=np.float32)
    v = np.clip(values, xs[0], xs[-1])
    return np.stack([np.interp(v, xs, cs[:, k]) for k in range(3)], axis=-1)


def layers(year: int, scene: str = "noord", metres_per_m3: float = METRES_PER_M3):
    d = np.load(DATA / "prep" / f"{scene}_{year}.npz")
    gas, status, water = d["gas"], d["status"], d["water"].astype(bool)
    h = np.zeros(gas.shape, dtype=np.float32)
    rgb = np.empty(gas.shape + (3,), dtype=np.float32)
    rgb[:] = GROUND
    rgb[water] = WATER
    h[water] = WATER_LEVEL
    rgb[status == 1] = CONTEXT
    h[status == 1] = CONTEXT_HEIGHT
    data = status == 2
    g = np.clip(gas[data], 0, GAS_CAP)
    rgb[data] = ramp(g)
    h[data] = PLINTH + g * metres_per_m3
    rgb[status == 3] = GAS_FREE
    h[status == 3] = PLINTH
    return h, rgb, status, float(d["res"])


def wall_colours(rgb: np.ndarray, height: np.ndarray, status: np.ndarray) -> np.ndarray:
    """Wall facets span roof and lower vertices; their UVs sample both texels.
    Give the lower texel the tallest higher building neighbour's original colour
    so the wall keeps that building's colour across the facet.
    """
    out = rgb.copy()
    top = height.copy()
    rows, cols = height.shape
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)):
        y0, y1 = max(0, -dy), min(rows, rows - dy)
        x0, x1 = max(0, -dx), min(cols, cols - dx)
        dst = (slice(y0, y1), slice(x0, x1))
        src = (slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx))
        take = height[src] > top[dst]
        take &= status[src] > 0
        np.copyto(out[dst], rgb[src], where=take[..., None])
        np.copyto(top[dst], height[src], where=take)
    return out


def render(year: int, out: Path, *, size=(1920, 1080), theta=58.0, phi=180.0, radius=2300.0,
           fov=32.0, target=(0.0, -150.0, 0.0), samples=16, sun_az=225.0, sun_el=32.0,
           scene="noord", metres_per_m3=METRES_PER_M3) -> dict:
    h, rgb, status, res = layers(year, scene, metres_per_m3)
    rgb = wall_colours(rgb, h, status)
    rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.float32)], axis=-1)
    rgba = np.ascontiguousarray(rgba.round().astype(np.uint8))
    span = float(h.shape[1]) * res  # square box
    settings = make_terrain_params_config(
        size_px=size, render_scale=1.0, msaa_samples=1, terrain_span=span, z_scale=1.0,
        camera_mode="mesh:zup:north:dem", cam_radius=radius, cam_phi_deg=phi, cam_theta_deg=theta,
        cam_target=target, fov_y_deg=fov, clip=(10.0, 200000.0),
        domain=(WATER_LEVEL, float(h.max())), overlays=[f3d.OverlayLayer.from_raster_rgba(rgba)],
        albedo_mode="colormap", hue_variation_strength=0.0, colormap_strength=1.0,
        light_azimuth_deg=sun_az, light_elevation_deg=sun_el, sun_intensity=2.2,
        ibl_enabled=True, exposure=1.0, ao_weight=0.0,
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        detail=DetailSettings(enabled=False), height_ao=HeightAoSettings(enabled=False),
        aov=AovSettings(enabled=True, albedo=True, normal=True, depth=True),
    )
    settings.shadows.enabled = True
    settings.shadows.technique = "PCF"
    session = f3d.Session(window=False, backend="vulkan")
    renderer = f3d.TerrainRenderer(session)
    ibl = f3d.IBL.from_hdr(str(HDR), intensity=0.9)
    params = f3d.TerrainRenderParams(settings)
    heights = np.ascontiguousarray(h)
    out.parent.mkdir(parents=True, exist_ok=True)
    if samples > 1:
        result = f3d.render_offline(renderer, f3d.MaterialSet.terrain_default(), ibl, params, heights,
                                    settings=f3d.OfflineQualitySettings(
                                        enabled=True, adaptive=False, batch_size=4,
                                        max_samples=samples, min_samples=samples))
        image = result.frame.to_numpy()
        np.save(out.with_suffix(".depth.npy"), result.aov_frame.depth())
    else:
        image = renderer.render_terrain_pbr_pom(f3d.MaterialSet.terrain_default(), ibl, params, heights).to_numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    f3d.numpy_to_png(str(out), image)
    return {"out": str(out), "year": year, "size": size, "theta": theta, "phi": phi, "radius": radius,
            "fov": fov, "target": target, "samples": samples, "adapter": session.adapter_name,
            "backend": session.backend, "scene": scene, "metres_per_m3": metres_per_m3, "native": f3d._forge3d.__file__}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, nargs=2, default=(1920, 1080))
    ap.add_argument("--theta", type=float, default=58.0)
    ap.add_argument("--phi", type=float, default=180.0)
    ap.add_argument("--radius", type=float, default=2300.0)
    ap.add_argument("--fov", type=float, default=32.0)
    ap.add_argument("--target", type=float, nargs=3, default=(0.0, -150.0, 0.0))
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--scene", default="noord")
    ap.add_argument("--scale", type=float, default=METRES_PER_M3)
    a = ap.parse_args()
    meta = render(a.year, a.out, size=tuple(a.size), theta=a.theta, phi=a.phi, radius=a.radius,
                  fov=a.fov, target=tuple(a.target), samples=a.samples,
                  scene=a.scene, metres_per_m3=a.scale)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
