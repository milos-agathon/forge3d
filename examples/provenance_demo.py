#!/usr/bin/env python3
"""VERITAS per-pixel provenance demo: which texture painted each pixel, and can you prove it?

Renders Switzerland in the Swiss national grid (EPSG:2056 / LV95) from
assets/tif/switzerland_dem.tif (the swiss_terrain_landcover_viewer.py input), looking
straight down. Four texture sources carry the land half of Crameri's *bukavu* palette;
the terrain shader blends them by elevation into a continuous bukavu relief ramp, so every
colour on screen really comes from an attributed source. With ``emit_provenance=True``
the render also writes, next to image.png:

* image.source_map.npy  - for every pixel, the id of the texture source that painted it
                          (the highest-resolution resident tile it sampled; equal
                          resolutions go to the most heavily blended layer)
* image.provenance.json - SHA256 Merkle root over those ids, the contributing texture
                          tiles and the PNG's own bytes, signed with Ed25519

tools/verify_provenance.py re-checks the triple offline. Changing one pixel's source id,
or one pixel of the PNG, breaks the signed root. provenance_demo.png puts it side by side.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d
from forge3d.alignment import reproject_dem_to_target
from forge3d.map_scene import _terrain_renderer_runtime_available

ROOT = Path(__file__).resolve().parents[1]
DEM_PATH = ROOT / "assets" / "tif" / "switzerland_dem.tif"
VERIFIER = ROOT / "tools" / "verify_provenance.py"
OUT_DIR = ROOT / "out" / "provenance_demo"
TARGET_CRS = "EPSG:2056"  # Swiss LV95
# Demo-only signing seed. Real deployments keep their 32-byte Ed25519 seed secret.
SIGNING_KEY = hashlib.sha256(b"forge3d-provenance-demo-key").digest()
# The terrain shader weights its 4 material layers by elevation around evenly spaced
# centres (0, 1/3, 2/3, 1 of the height range); source id = layer index + 1.
LAYERS = ["lowland", "foothill", "alpine", "summit"]
LEGEND = [(230, 97, 1), (27, 158, 119), (117, 112, 179), (102, 166, 230)]
GRID_CELLS = 3600          # heightmap width in cells (~100 m in LV95)
VERTICAL_EXAGGERATION = 2.0
SIZE = 4800                # render is SIZE x SIZE pixels
LIGHTING = {
    "albedo_mode": "material",
    "colormap_strength": 0.0,
    "hue_variation_strength": 0.0,
    "material_slope_bias": 0.0,
    # The mesh camera already displaces real geometry; parallax occlusion on top of it
    # ray-marches height in discrete steps, which shows up as contour-like bands.
    "pom": False,
    "camera": {"camera_mode": "mesh"},
    "renderer_config": {"lighting": {"exposure": 1.5}},
}
SUN_INTENSITY = 3.5

GPU_AVAILABLE = f3d.has_gpu() and _terrain_renderer_runtime_available()


def bukavu_land(positions: np.ndarray) -> list[tuple[int, int, int]]:
    """Sample the land half (0.5-1.0) of the bukavu colormap."""
    from cmcrameri import cm

    rgb = cm.bukavu(0.5 + 0.5 * np.asarray(positions))[:, :3]
    return [tuple(int(round(v * 255)) for v in c) for c in rgb]


def load_heightmap() -> tuple[np.ndarray, np.ndarray]:
    """Square LV95 heightmap in grid units plus the matching inside-Switzerland mask."""
    import rasterio
    from rasterio.enums import Resampling

    lv95 = OUT_DIR / "switzerland_dem_epsg2056.tif"
    if not lv95.exists():
        reproject_dem_to_target(DEM_PATH, TARGET_CRS, output_path=lv95, resampling="bilinear")
    with rasterio.open(lv95) as src:
        scale = src.width / GRID_CELLS
        dem = src.read(1, out_shape=(round(src.height / scale), GRID_CELLS), resampling=Resampling.average)
        cell_m = src.res[0] * scale
    inside = dem > -1000.0  # nodata is -3.4e38 outside the border
    dem = np.where(inside, dem, dem[inside].min()) / cell_m * VERTICAL_EXAGGERATION  # metres -> cells
    # MapScene lays any heightmap on a square footprint: pad to square so the country keeps
    # its proportions, and flip rows so north ends up at the top of a straight-down view.
    n, top = max(dem.shape), (max(dem.shape) - dem.shape[0]) // 2
    square, mask = np.full((n, n), dem.min(), np.float32), np.zeros((n, n), bool)
    square[top:top + dem.shape[0]], mask[top:top + dem.shape[0]] = dem, inside
    return np.ascontiguousarray(np.flipud(square)), mask


def write_texture_sources(folder: Path) -> list[dict]:
    folder.mkdir(parents=True, exist_ok=True)
    specs = []
    for index, (name, rgb) in enumerate(zip(LAYERS, bukavu_land(np.linspace(0.0, 1.0, 4)))):
        np.save(folder / f"{name}.npy", np.full((512, 512, 4), (*rgb, 255), np.uint8))
        specs.append({"material_index": index, "family": "albedo", "path": str(folder / f"{name}.npy"),
                      "virtual_size_px": [512, 512]})
    return specs


def build_scene(heightmap: np.ndarray, image_path: Path) -> f3d.MapScene:
    vt = {"enabled": True, "families": [{"family": "albedo", "virtual_size_px": [512, 512]}],
          "sources": write_texture_sources(OUT_DIR / "sources")}
    return f3d.MapScene(
        terrain=f3d.TerrainSource(data=heightmap, crs=TARGET_CRS,
                                  metadata={"source_id": "switzerland-dem-lv95", "virtual_texture": vt}),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=1700.0 * heightmap.shape[0] / 1215,
                               azimuth_deg=90.0, elevation_deg=89.0, fov_deg=40.0),
        lighting=f3d.LightingPreset(intensity=SUN_INTENSITY, settings=LIGHTING),
        # Provenance needs the one-shot path: 1 sample, no denoise, no HDR.
        output=f3d.OutputSpec(width=SIZE, height=SIZE, format="png", samples=1, path=str(image_path)),
    )


def verify(image: Path, source_map: Path, manifest: Path) -> bool:
    result = subprocess.run([sys.executable, str(VERIFIER), str(image), str(source_map), str(manifest)],
                            capture_output=True, text=True, cwd=str(ROOT))
    print(result.stdout.rstrip() or result.stderr.rstrip())
    return result.returncode == 0


def compose(image: Path, source_map: np.ndarray, mask: np.ndarray, ok: bool, map_tampered_ok: bool,
            png_tampered_ok: bool, out: Path) -> None:
    """Display only (the signed files are untouched): hide the padding outside the border."""
    ys, xs = np.nonzero(source_map)  # the terrain square on screen
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    screen_mask = np.zeros(source_map.shape, bool)
    border = Image.fromarray(mask).resize((box[2] - box[0], box[3] - box[1]), Image.NEAREST)
    # Trim the few pixels where the terrain drops to the padding, so no dark wall frames the border.
    border = border.convert("L").filter(ImageFilter.MinFilter(2 * (SIZE // 1600) + 1))
    screen_mask[box[1]:box[3], box[0]:box[2]] = np.asarray(border) > 0
    rows, cols = np.nonzero(screen_mask)
    pad = SIZE // 80
    crop = (slice(max(rows.min() - pad, 0), rows.max() + pad), slice(max(cols.min() - pad, 0), cols.max() + pad))
    white = np.full(source_map.shape + (3,), 255.0)
    beauty = np.asarray(Image.open(image).convert("RGB"), np.float32)
    colors = np.array([(255, 255, 255)] + LEGEND, np.float32)[np.clip(source_map, 0, 4)]
    panels = [np.where(screen_mask[..., None], beauty, white)[crop],
              np.where(screen_mask[..., None], 0.35 * beauty + 0.65 * colors, white)[crop]]
    height, width = panels[0].shape[:2]
    unit = width // 60  # text scales with the render
    canvas = Image.new("RGB", (2 * width + 2 * unit, height + 14 * unit), "white")
    for i, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel.astype(np.uint8)), (i * (width + 2 * unit), 3 * unit))
    draw = ImageDraw.Draw(canvas)
    big, small = ImageFont.load_default(size=int(1.5 * unit)), ImageFont.load_default(size=unit)
    draw.text((0, unit // 2), "1. The rendered map (EPSG:2056, bukavu relief)", fill="black", font=big)
    draw.text((width + 2 * unit, unit // 2), "2. Which texture source painted each pixel", fill="black", font=big)
    inside = source_map[screen_mask]
    for row, (name, color) in enumerate(zip(LAYERS, LEGEND)):
        x, y = width + 2 * unit + 14 * unit * row, height + 4 * unit
        draw.rectangle((x, y, x + unit, y + unit), fill=color)
        draw.text((x + int(1.5 * unit), y - unit // 10), f"{name}  {100 * np.mean(inside == row + 1):.1f}%",
                  fill="black", font=small)
    lines = [("3. Signed record checked by tools/verify_provenance.py", "black"),
             (f"Original render:  {'VERIFIED' if ok else 'FAILED'}", "green" if ok else "red"),
             (f"1 pixel's recorded source changed:  {'VERIFIED' if map_tampered_ok else 'REJECTED'}",
              "red" if map_tampered_ok else "green"),
             (f"1 pixel of the PNG changed:  {'VERIFIED' if png_tampered_ok else 'REJECTED'}",
              "red" if png_tampered_ok else "green")]
    for i, (text, fill) in enumerate(lines):
        draw.text((0, height + 4 * unit + int(2.4 * unit) * i), text, fill=fill, font=big)
    canvas.save(out)


def main() -> int:
    if not GPU_AVAILABLE:
        print("Skipping: provenance demo requires a terrain-capable GPU adapter "
              "(no CPU fallback: a provenance seal must describe a real render).")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = OUT_DIR / "image.png"
    source_map_path, manifest = OUT_DIR / "image.source_map.npy", OUT_DIR / "image.provenance.json"
    heightmap, mask = load_heightmap()
    build_scene(heightmap, image).render(emit_provenance=True, provenance_signing_key=SIGNING_KEY)
    print(f"Rendered {image}\n  + {source_map_path.name}\n  + {manifest.name}\n")

    print("== Verifying the untouched render ==")
    ok = verify(image, source_map_path, manifest)

    print("\n== Tampering 1: claiming one pixel came from a different source ==")
    source_map = np.load(source_map_path)
    tampered = source_map.copy()
    y, x = np.argwhere(tampered != 0)[np.count_nonzero(tampered) // 2]
    tampered[y, x] = tampered[y, x] % len(LAYERS) + 1  # still a real, known source id
    np.save(OUT_DIR / "tampered.source_map.npy", tampered)
    print(f"pixel ({x}, {y}): source {source_map[y, x]} -> {tampered[y, x]}")
    map_tampered_ok = verify(image, OUT_DIR / "tampered.source_map.npy", manifest)

    print("\n== Tampering 2: editing one pixel of the published PNG ==")
    pixels = np.array(Image.open(image))
    pixels[y, x, :3] = 255 - pixels[y, x, :3]
    Image.fromarray(pixels).save(OUT_DIR / "tampered.png")
    print(f"pixel ({x}, {y}): colour inverted")
    png_tampered_ok = verify(OUT_DIR / "tampered.png", source_map_path, manifest)

    compose(image, source_map, mask, ok, map_tampered_ok, png_tampered_ok, OUT_DIR / "provenance_demo.png")
    print(f"\nOriginal verified: {ok}  |  source-map tamper verified: {map_tampered_ok}  |  "
          f"PNG tamper verified: {png_tampered_ok}")
    print(f"Summary image: {OUT_DIR / 'provenance_demo.png'}")
    return 0 if ok and not map_tampered_ok and not png_tampered_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
