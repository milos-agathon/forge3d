#!/usr/bin/env python3
"""Regenerate the documentation gallery images.

Each gallery entry is rendered by its dedicated example script via
``run_example()``.  After the base render, a title bar is composited on
top and the image is saved to ``docs/gallery/images/``.

This architecture avoids re-implementing camera/PBR/settle logic and
reuses the battle-tested snapshot pipelines in the example scripts.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

os.environ.setdefault("FORGE3D_REPO_ROOT", str(REPO_ROOT))
os.environ.setdefault("FORGE3D_LICENSE_KEY", "F3D-PRO-20991231-test-signature")

import forge3d as f3d
from forge3d._license import set_license_key
from forge3d.export import VectorScene, export_svg
from forge3d.legend import Legend, LegendConfig
from forge3d.map_plate import BBox, MapPlate, MapPlateConfig
from forge3d.north_arrow import NorthArrow, NorthArrowConfig
from forge3d.scale_bar import ScaleBar, ScaleBarConfig


IMAGES_DIR = REPO_ROOT / "docs" / "gallery" / "images"
WORK_DIR = REPO_ROOT / "logs" / "gallery-regen-20260315"

# Final output dimensions.
W, H = 1200, 720


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ["DejaVuSans-Bold.ttf", "Arial Bold.ttf"]
        if bold
        else ["DejaVuSans.ttf", "Arial.ttf"]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def rgba(color: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(int(max(0.0, min(1.0, c)) * 255) for c in color)


def wait_for_file(path: Path, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    last_size = -1
    stable_hits = 0
    while time.time() < deadline:
        if path.exists():
            size = path.stat().st_size
            if size > 0:
                if size == last_size:
                    stable_hits += 1
                else:
                    stable_hits = 0
                last_size = size
                if stable_hits >= 3:
                    return
        time.sleep(0.3)
    raise RuntimeError(f"Timed out waiting for output: {path}")


def run_example(name: str, args: list[str], output_path: Path) -> None:
    """Run an example script in a subprocess and wait for *output_path*.

    The viewer process sometimes hangs during shutdown even after a
    successful snapshot.  We therefore do NOT use ``check=True`` and
    instead verify success by checking whether the output file was
    actually written.
    """
    print(f"  [{name}] running …")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    env = os.environ.copy()
    env["FORGE3D_REPO_ROOT"] = str(REPO_ROOT)
    env["FORGE3D_LICENSE_KEY"] = env.get(
        "FORGE3D_LICENSE_KEY", "F3D-PRO-20991231-test-signature"
    )
    env["PYTHONPATH"] = str(PYTHON_DIR)

    try:
        subprocess.run(
            [sys.executable] + args,
            cwd=REPO_ROOT,
            env=env,
            check=False,         # viewer may exit non-zero on shutdown
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        print(f"  [{name}] subprocess timed out (180 s) — checking file …")

    wait_for_file(output_path)
    print(f"  [{name}] done → {output_path.name}")


def draw_halo_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    label_font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    halo: tuple[int, int, int, int],
    halo_px: int = 2,
    anchor: str = "mm",
) -> None:
    x, y = xy
    for dx in range(-halo_px, halo_px + 1):
        for dy in range(-halo_px, halo_px + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text(
                (x + dx, y + dy), text, font=label_font, fill=halo, anchor=anchor
            )
    draw.text((x, y), text, font=label_font, fill=fill, anchor=anchor)


# ---------------------------------------------------------------------------
# Composition: add a title bar on top of a rendered image
# ---------------------------------------------------------------------------

def compose_titled(
    base_path: Path,
    out_path: Path,
    title: str,
    subtitle: str,
    *,
    width: int = W,
    height: int = H,
    bar_height: int = 36,
    bg: tuple[int, int, int, int] = (24, 26, 30, 255),
) -> None:
    """Load *base_path*, resize to (width, height - bar_height), paste onto
    a dark canvas with a title bar, and save to *out_path*.
    """
    base = Image.open(base_path).convert("RGBA")
    img_h = height - bar_height
    base = base.resize((width, img_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (width, height), bg)
    draw = ImageDraw.Draw(canvas)

    # Title bar
    title_font = font(16, bold=True)
    sub_font = font(14)
    label = f"{title} \u00b7 {subtitle}"
    draw.text((14, bar_height // 2), label, font=title_font,
              fill=(230, 232, 236, 255), anchor="lm")

    # Paste terrain render below title bar
    canvas.paste(base, (0, bar_height))

    out_path.unlink(missing_ok=True)
    canvas.save(out_path, optimize=True)


# ---------------------------------------------------------------------------
# Gallery entries
# ---------------------------------------------------------------------------


def render_01_rainier() -> None:
    """Mount Rainier — hero shot via terrain_viewer_interactive.py."""
    print("[gallery] 01 Mount Rainier")
    tmp = WORK_DIR / "01-base.png"
    out = IMAGES_DIR / "01-mount-rainier.png"

    run_example("01 terrain", [
        "examples/terrain_viewer_interactive.py",
        "--dem", "assets/tif/dem_rainier.tif",
        "--width", str(W), "--height", str(H),
        "--pbr", "--msaa", "8",
        "--shadow-technique", "pcss",
        "--exposure", "1.1",
        "--sun-azimuth", "302", "--sun-elevation", "24",
        "--height-ao", "--height-ao-strength", "1.0",
        "--sun-vis", "--sun-vis-mode", "soft",
        "--tonemap", "aces",
        "--snapshot", str(tmp),
    ], tmp)

    compose_titled(tmp, out, "Mount Rainier", "dem_rainier.tif")


def render_02_fuji_labels() -> None:
    """Mount Fuji with GeoPackage labels via fuji_labels_demo.py."""
    print("[gallery] 02 Mount Fuji Labels")
    tmp = WORK_DIR / "02-base.png"
    out = IMAGES_DIR / "02-mount-fuji-labels.png"

    run_example("02 fuji", [
        "examples/fuji_labels_demo.py",
        "--width", str(W), "--height", str(H),
        "--pbr", "--msaa", "8",
        "--shadows", "pcss",
        "--exposure", "1.1",
        "--sun-azimuth", "296", "--sun-elevation", "22",
        "--height-ao",
        "--sun-vis",
        "--snapshot", str(tmp),
    ], tmp)

    compose_titled(tmp, out, "Mount Fuji", "labels from GeoPackage")


def render_03_swiss_landcover() -> None:
    """Swiss terrain with raster land-cover overlay."""
    print("[gallery] 03 Swiss Landcover")
    out = IMAGES_DIR / "03-swiss-landcover.png"

    run_example("03 swiss", [
        "examples/swiss_terrain_landcover_viewer.py",
        "--preset", "hq4",
        "--width", str(W), "--height", str(H),
        "--snapshot", str(out),
    ], out)


def render_04_luxembourg_rail() -> None:
    """Luxembourg rail network overlay via luxembourg_rail_overlay.py."""
    print("[gallery] 04 Luxembourg Rail Network")
    tmp = WORK_DIR / "04-base.png"
    out = IMAGES_DIR / "04-luxembourg-rail-network.png"

    run_example("04 rail", [
        "examples/luxembourg_rail_overlay.py",
        "--width", str(W), "--height", str(H),
        "--msaa", "8",
        "--shadow-technique", "pcss", "--shadow-map-res", "4096",
        "--exposure", "1.2",
        "--height-ao",
        "--sun-vis",
        "--normal-strength", "1.2",
        "--snapshot", str(tmp),
    ], tmp)

    compose_titled(tmp, out, "Luxembourg", "rail network overlay")


def render_05_buildings() -> None:
    """3D buildings (CityJSON) on Fuji terrain."""
    print("[gallery] 05 3D Buildings")
    out = IMAGES_DIR / "05-3d-buildings.png"

    run_example("05 buildings", [
        "examples/buildings_viewer_interactive.py",
        "--fuji",
        "--width", str(W), "--height", str(H),
        "--snapshot", str(out),
    ], out)


def render_06_point_cloud() -> None:
    """LiDAR point cloud (Mt St Helens)."""
    print("[gallery] 06 Point Cloud")
    out = IMAGES_DIR / "06-point-cloud.png"

    run_example("06 pointcloud", [
        "examples/pointcloud_viewer_interactive.py",
        "--input", "assets/lidar/MtStHelens.laz",
        "--width", str(W), "--height", str(H),
        "--point-size", "2.0",
        "--max-points", "500000",
        "--color-mode", "rgb",
        "--phi", "0.8", "--theta", "0.62", "--radius", "2.1",
        "--snapshot", str(out),
    ], out)


def render_07_flyover() -> None:
    """Camera flyover — 3 orbit frames composited side-by-side."""
    print("[gallery] 07 Camera Flyover")
    out = IMAGES_DIR / "07-camera-flyover.png"

    frame_w = W // 3
    angles = [0, 120, 240]
    frame_paths: list[Path] = []

    for i, phi in enumerate(angles):
        tmp = WORK_DIR / f"07-frame-{phi}.png"
        run_example(f"07 frame {phi}°", [
            "examples/terrain_viewer_interactive.py",
            "--dem", "assets/tif/dem_rainier.tif",
            "--width", str(frame_w), "--height", str(H - 72),
            "--pbr", "--msaa", "8",
            "--shadow-technique", "pcss",
            "--exposure", "1.1",
            "--sun-azimuth", str(302 + phi), "--sun-elevation", "24",
            "--height-ao",
            "--sun-vis",
            "--snapshot", str(tmp),
        ], tmp)
        frame_paths.append(tmp)

    # Compose 3 frames side-by-side with title + frame labels
    bar_h = 36
    label_h = 36
    canvas = Image.new("RGBA", (W, H), (24, 26, 30, 255))
    draw = ImageDraw.Draw(canvas)

    # Title bar
    title_font = font(16, bold=True)
    draw.text((14, bar_h // 2),
              "Camera flyover \u00b7 orbit animation frames",
              font=title_font, fill=(230, 232, 236, 255), anchor="lm")

    # Frame labels
    label_font = font(13)
    for i, phi in enumerate(angles):
        cx = i * frame_w + frame_w // 2
        draw.text((cx, bar_h + 14), f"Frame {phi}\u00b0",
                  font=label_font, fill=(180, 184, 190, 255), anchor="mm")

    # Paste frames
    top_y = bar_h + label_h
    img_h = H - bar_h - label_h
    for i, fp in enumerate(frame_paths):
        frame = Image.open(fp).convert("RGBA").resize(
            (frame_w, img_h), Image.LANCZOS
        )
        canvas.paste(frame, (i * frame_w, top_y))

    out.unlink(missing_ok=True)
    canvas.save(out, optimize=True)
    wait_for_file(out)


def render_08_vector_export() -> None:
    """Vector export SVG preview — pure 2-D composition."""
    print("[gallery] 08 Vector Export")
    out = IMAGES_DIR / "08-vector-export.png"

    scene = VectorScene()
    scene.add_polygon(
        [(0, 0), (100, 0), (80, 70), (10, 90)],
        fill_color=(0.20, 0.52, 0.83, 0.92),
        stroke_color=(0.08, 0.19, 0.34, 1.0),
        stroke_width=3.0,
    )
    scene.add_polygon(
        [(18, 18), (48, 12), (62, 32), (35, 52)],
        fill_color=(0.95, 0.73, 0.35, 0.82),
        stroke_color=(0.55, 0.33, 0.08, 1.0),
        stroke_width=2.0,
    )
    scene.add_polyline(
        [(5, 10), (40, 35), (90, 60)],
        stroke_color=(0.07, 0.08, 0.10, 1.0),
        stroke_width=4.0,
    )
    scene.add_label(
        "Ridge",
        position=(55, 52),
        font_size=20,
        color=(0.07, 0.08, 0.10, 1.0),
    )

    svg_path = WORK_DIR / "08-vector-export.svg"
    svg_path.unlink(missing_ok=True)
    export_svg(scene, svg_path, width=W, height=H)

    bounds = scene.compute_bounds(padding=12.0)
    pad = 80.0
    span_x = max(bounds.width, 1.0)
    span_y = max(bounds.height, 1.0)
    drawable_w = W - 2 * pad
    drawable_h = H - 2 * pad
    scale = min(drawable_w / span_x, drawable_h / span_y)

    def project(pt: tuple[float, float]) -> tuple[float, float]:
        return (pad + (pt[0] - bounds.min_x) * scale,
                H - pad - (pt[1] - bounds.min_y) * scale)

    image = Image.new("RGBA", (W, H), (248, 246, 239, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(
        (28, 28, W - 28, H - 28), radius=24,
        outline=(36, 43, 50, 32), width=2,
    )
    draw.rounded_rectangle(
        (54, 44, 356, 126), radius=18, fill=(255, 255, 255, 224)
    )
    draw.text((76, 70), "Vector Export", font=font(32, bold=True),
              fill=(24, 31, 36, 255), anchor="lm")
    draw.text((76, 102), "SVG scene preview", font=font(18),
              fill=(72, 82, 90, 255), anchor="lm")

    for polygon in scene.polygons:
        pts = [project(p) for p in polygon.exterior]
        draw.polygon(pts, fill=rgba(polygon.style.fill_color),
                     outline=rgba(polygon.style.stroke_color),
                     width=max(1, int(round(polygon.style.stroke_width))))
    for polyline in scene.polylines:
        pts = [project(p) for p in polyline.path]
        draw.line(pts, fill=rgba(polyline.style.stroke_color),
                  width=max(1, int(round(polyline.style.stroke_width))),
                  joint="curve")
    for label in scene.labels:
        pt = project(label.position)
        draw_halo_text(
            draw, pt, label.text,
            label_font=font(int(round(label.style.font_size)), bold=True),
            fill=rgba(label.style.color),
            halo=rgba(label.style.halo_color),
        )

    out.unlink(missing_ok=True)
    image.save(out, optimize=True)
    wait_for_file(out)


def render_09_shadow_comparison() -> None:
    """Side-by-side shadow comparison: morning vs evening sun."""
    print("[gallery] 09 Shadow Comparison")
    out = IMAGES_DIR / "09-shadow-comparison.png"
    left_tmp = WORK_DIR / "09-morning.png"
    right_tmp = WORK_DIR / "09-evening.png"

    panel_w = W // 2
    panel_h = H - 72  # room for title + labels

    # Morning: east sun
    run_example("09 morning", [
        "examples/terrain_viewer_interactive.py",
        "--dem", "assets/tif/dem_rainier.tif",
        "--width", str(panel_w), "--height", str(panel_h),
        "--pbr", "--msaa", "8",
        "--shadow-technique", "pcss",
        "--exposure", "1.1",
        "--sun-azimuth", "110", "--sun-elevation", "18",
        "--height-ao",
        "--sun-vis",
        "--snapshot", str(left_tmp),
    ], left_tmp)

    # Evening: west sun
    run_example("09 evening", [
        "examples/terrain_viewer_interactive.py",
        "--dem", "assets/tif/dem_rainier.tif",
        "--width", str(panel_w), "--height", str(panel_h),
        "--pbr", "--msaa", "8",
        "--shadow-technique", "pcss",
        "--exposure", "1.1",
        "--sun-azimuth", "285", "--sun-elevation", "42",
        "--height-ao",
        "--sun-vis",
        "--snapshot", str(right_tmp),
    ], right_tmp)

    # Compose side-by-side
    bar_h = 36
    label_h = 36
    canvas = Image.new("RGBA", (W, H), (24, 26, 30, 255))
    draw = ImageDraw.Draw(canvas)

    # Title bar
    title_font = font(16, bold=True)
    draw.text((14, bar_h // 2),
              "Shadow comparison \u00b7 sun position effect",
              font=title_font, fill=(230, 232, 236, 255), anchor="lm")

    # Panel labels
    label_font = font(13)
    draw.text((panel_w // 2, bar_h + 14), "Morning (east sun)",
              font=label_font, fill=(180, 184, 190, 255), anchor="mm")
    draw.text((panel_w + panel_w // 2, bar_h + 14), "Evening (west sun)",
              font=label_font, fill=(180, 184, 190, 255), anchor="mm")

    # Paste panels
    top_y = bar_h + label_h
    img_h = H - bar_h - label_h
    left = Image.open(left_tmp).convert("RGBA").resize(
        (panel_w, img_h), Image.LANCZOS
    )
    right = Image.open(right_tmp).convert("RGBA").resize(
        (panel_w, img_h), Image.LANCZOS
    )
    canvas.paste(left, (0, top_y))
    canvas.paste(right, (panel_w, top_y))

    out.unlink(missing_ok=True)
    canvas.save(out, optimize=True)
    wait_for_file(out)


def render_10_map_plate() -> None:
    """Cartographic map plate with legend, scale bar, north arrow, inset."""
    print("[gallery] 10 Map Plate")
    source_tmp = WORK_DIR / "10-plate-source.png"
    out = IMAGES_DIR / "10-map-plate.png"
    bbox = BBox(west=-122.0, south=46.7, east=-121.6, north=46.95)

    # Render base terrain
    run_example("10 terrain", [
        "examples/terrain_viewer_interactive.py",
        "--dem", "assets/tif/dem_rainier.tif",
        "--width", "900", "--height", "560",
        "--pbr", "--msaa", "8",
        "--shadow-technique", "pcss",
        "--exposure", "1.1",
        "--sun-azimuth", "315", "--sun-elevation", "32",
        "--height-ao",
        "--sun-vis",
        "--snapshot", str(source_tmp),
    ], source_tmp)

    source_rgba = np.asarray(
        Image.open(source_tmp).convert("RGBA"), dtype=np.uint8
    )
    plate = MapPlate(
        MapPlateConfig(width=W, height=H, margin=(70, 250, 70, 45))
    )
    plate.set_map_region(source_rgba, bbox)
    plate.add_title("Map Plate \u00b7 legend + scale bar + north arrow",
                    font_size=24)

    cmap = f3d.get_colormap("forge3d:viridis")
    legend = Legend.from_colormap(
        cmap,
        domain=(0.0, 4000.0),
        config=LegendConfig(title="Elevation", label_suffix=" m",
                            tick_count=5),
    )
    plate.add_legend(legend.render(), position="right")

    meters_per_pixel = ScaleBar.compute_meters_per_pixel(
        bbox, source_rgba.shape[1]
    )
    scale_bar = ScaleBar(
        meters_per_pixel, ScaleBarConfig(units="km", width_px=220)
    )
    plate.add_scale_bar(scale_bar.render(), position="bottom-left")

    north_arrow = NorthArrow(NorthArrowConfig(style="arrow", size=72))
    plate.add_north_arrow(north_arrow.render(), position="top-right")

    inset = np.asarray(
        Image.open(source_tmp)
        .resize((220, 136), Image.LANCZOS)
        .convert("RGBA"),
        dtype=np.uint8,
    )
    plate.add_inset(inset, position="bottom-right", border_width=2)

    out.unlink(missing_ok=True)
    plate.export_png(out)
    wait_for_file(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ensure_dir(IMAGES_DIR)
    ensure_dir(WORK_DIR)
    set_license_key(os.environ["FORGE3D_LICENSE_KEY"])

    steps = [
        render_01_rainier,
        render_02_fuji_labels,
        render_03_swiss_landcover,
        render_04_luxembourg_rail,
        render_05_buildings,
        render_06_point_cloud,
        render_07_flyover,
        render_08_vector_export,
        render_09_shadow_comparison,
        render_10_map_plate,
    ]

    started = time.time()
    for step in steps:
        step()

    elapsed = time.time() - started
    print(f"\n[gallery] complete in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
