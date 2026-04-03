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
from PIL import Image, ImageDraw, ImageFilter, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
TESTS_DIR = REPO_ROOT / "tests"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

os.environ.setdefault("FORGE3D_REPO_ROOT", str(REPO_ROOT))

from _license_test_keys import sign_test_key

os.environ.setdefault("FORGE3D_LICENSE_KEY", sign_test_key("PRO", "20991231"))

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


def build_natural_relief_colormap(domain: tuple[float, float]):
    """Build the earth-tone relief ramp used by the map plate legend."""
    from forge3d.colormaps.core import from_stops

    vmin, vmax = map(float, domain)
    span = max(vmax - vmin, 1.0)
    anchor_colors = [
        (0.00, "#427038"),
        (0.08, "#548a45"),
        (0.18, "#73994c"),
        (0.30, "#8c944f"),
        (0.42, "#9e854c"),
        (0.55, "#947047"),
        (0.68, "#856652"),
        (0.80, "#99918a"),
        (0.90, "#c7c1bc"),
        (1.00, "#f2f2f5"),
    ]
    stops = [(vmin + t * span, color) for t, color in anchor_colors]
    normalized_stops = [((value - vmin) / span, color) for value, color in stops]
    return from_stops("forge3d:natural_relief", normalized_stops, n=256)


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


def _step_log_path(name: str) -> Path:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    return WORK_DIR / f"{safe or 'render'}.log"


def _tail_log(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return "<no log captured>"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"<unable to read log: {exc}>"
    if not lines:
        return "<log empty>"
    return "\n".join(lines[-max_lines:])


def _example_env() -> dict[str, str]:
    env = os.environ.copy()
    env["FORGE3D_REPO_ROOT"] = str(REPO_ROOT)
    env["FORGE3D_LICENSE_KEY"] = env.get(
        "FORGE3D_LICENSE_KEY", sign_test_key("PRO", "20991231")
    )
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(PYTHON_DIR)
        if not existing_pythonpath
        else os.pathsep.join([str(PYTHON_DIR), existing_pythonpath])
    )
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    return env


def wait_for_file_from_process(
    path: Path,
    *,
    process: subprocess.Popen[str],
    timeout_s: float,
    name: str,
    log_path: Path,
) -> None:
    deadline = time.time() + timeout_s
    last_size = -1
    stable_hits = 0
    while time.time() < deadline:
        if path.exists():
            size = path.stat().st_size
            if size > 0:
                if process.poll() is not None:
                    return
                if size == last_size:
                    stable_hits += 1
                else:
                    stable_hits = 0
                last_size = size
                if stable_hits >= 3:
                    return

        returncode = process.poll()
        if returncode is not None:
            detail = (
                f"[{name}] subprocess exited with code {returncode} before writing "
                f"{path.name}\nLog: {log_path}\n{_tail_log(log_path)}"
            )
            raise RuntimeError(detail)

        time.sleep(0.3)

    detail = f"Timed out waiting for output: {path}\nLog: {log_path}\n{_tail_log(log_path)}"
    raise RuntimeError(detail)


def run_example(name: str, args: list[str], output_path: Path) -> None:
    """Run an example script in a subprocess and wait for *output_path*.

    The viewer process sometimes hangs during shutdown even after a
    successful snapshot.  We therefore do NOT use ``check=True`` and
    instead verify success by checking whether the output file was
    actually written.
    """
    print(f"  [{name}] running ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    env = _example_env()
    log_path = _step_log_path(name)
    log_path.unlink(missing_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable] + args,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            wait_for_file_from_process(
                output_path,
                process=process,
                timeout_s=240.0,
                name=name,
                log_path=log_path,
            )
        finally:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                print(f"  [{name}] subprocess hung on shutdown -- terminating ...")
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

    print(f"  [{name}] done -> {output_path.name}")


def run_example_interactive(
    name: str,
    args: list[str],
    commands: list[str],
    output_path: Path,
) -> None:
    """Run an interactive example, feed scripted stdin commands, wait for
    *output_path*, then close the viewer.
    """
    print(f"  [{name}] running ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    env = _example_env()
    log_path = _step_log_path(name)
    log_path.unlink(missing_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable] + args,
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            if process.stdin is None:
                raise RuntimeError("Failed to open stdin for interactive example")

            for command in commands:
                process.stdin.write(command + "\n")
                process.stdin.flush()
                time.sleep(0.3)

            wait_for_file_from_process(
                output_path,
                process=process,
                timeout_s=240.0,
                name=name,
                log_path=log_path,
            )

            process.stdin.write("quit\n")
            process.stdin.flush()
            process.stdin.close()

            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                print(f"  [{name}] interactive subprocess hung on shutdown -- terminating ...")
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        except BrokenPipeError:
            if not output_path.exists() or output_path.stat().st_size <= 0:
                raise RuntimeError(
                    f"[{name}] interactive subprocess exited early before snapshot\n"
                    f"Log: {log_path}\n{_tail_log(log_path)}"
                ) from None

    print(f"  [{name}] done -> {output_path.name}")


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


def normalize(vec: np.ndarray) -> np.ndarray:
    """Return a normalized 3-vector, preserving zeros."""
    length = float(np.linalg.norm(vec))
    if length <= 1e-8:
        return vec
    return vec / length


def render_cityjson_building_preview(
    layer,
    out_path: Path,
    *,
    width: int = 1600,
    height: int = 912,
    focus_subset: bool = True,
    zoom: float = 1.0,
) -> None:
    """Render a deterministic axonometric preview from CityJSON triangles."""
    if layer.building_count == 0 or layer.total_triangles == 0:
        raise RuntimeError("CityJSON layer did not produce any renderable building geometry")

    ui_scale = max(width, height) / 1600.0

    source_buildings = []
    for building in layer.buildings:
        positions = np.asarray(building.positions, dtype=np.float32).reshape(-1, 3)
        indices = np.asarray(building.indices, dtype=np.uint32).reshape(-1, 3)
        if len(positions) == 0 or len(indices) == 0:
            continue

        height_span = float(positions[:, 2].max() - positions[:, 2].min())
        if height_span <= 1.0:
            continue

        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        footprint_span = maxs[:2] - mins[:2]
        footprint_area = float(max(footprint_span[0], 1.0) * max(footprint_span[1], 1.0))
        source_buildings.append(
            {
                "building": building,
                "positions": positions,
                "indices": indices,
                "center_xy": positions[:, :2].mean(axis=0),
                "footprint_span": footprint_span,
                "footprint_area": footprint_area,
                "height_span": height_span,
            }
        )

    if not source_buildings:
        raise RuntimeError("CityJSON layer buildings did not contain any volumetric geometry")

    if focus_subset and len(source_buildings) > 80:
        anchor_index = max(
            range(len(source_buildings)),
            key=lambda i: source_buildings[i]["footprint_area"] * max(source_buildings[i]["height_span"], 6.0),
        )
        anchor_center = np.asarray(source_buildings[anchor_index]["center_xy"], dtype=np.float32)
        anchor_span = np.asarray(source_buildings[anchor_index]["footprint_span"], dtype=np.float32)
        focus_radius = float(np.clip(max(anchor_span) * 1.35, 90.0, 180.0))

        distances = [
            (
                float(np.linalg.norm(np.asarray(item["center_xy"], dtype=np.float32) - anchor_center)),
                item,
            )
            for item in source_buildings
        ]
        distances.sort(key=lambda entry: entry[0])
        selected = [item for distance, item in distances if distance <= focus_radius]
        if len(selected) < 8:
            selected = [item for _, item in distances[:8]]
        elif len(selected) > 24:
            selected = selected[:24]
        source_buildings = selected

    building_records: list[dict[str, np.ndarray | tuple[int, int, int, int]]] = []
    geo_points: list[np.ndarray] = []
    accent_palette = np.array(
        [
            (190, 136, 105),
            (156, 112, 90),
            (169, 142, 116),
            (143, 120, 102),
            (181, 154, 129),
        ],
        dtype=np.float32,
    ) / 255.0

    for index, item in enumerate(source_buildings):
        building = item["building"]
        positions = np.asarray(item["positions"], dtype=np.float32)
        indices = np.asarray(item["indices"], dtype=np.uint32)
        geo_points.append(positions)
        base = np.array(building.material.albedo, dtype=np.float32)
        accent = accent_palette[index % len(accent_palette)]
        wall_rgb = np.clip(base * 0.5 + accent * 0.5, 0.0, 1.0)
        roof_rgb = np.clip(
            wall_rgb * np.array([0.95, 0.88, 0.82], dtype=np.float32) + 0.04,
            0.0,
            1.0,
        )

        building_records.append(
            {
                "positions_geo": positions,
                "indices": indices,
                "wall_rgba": tuple(int(channel * 255) for channel in (*wall_rgb, 1.0)),
                "roof_rgba": tuple(int(channel * 255) for channel in (*roof_rgb, 1.0)),
            }
        )

    if not building_records:
        raise RuntimeError("CityJSON layer buildings did not contain any triangles")

    all_geo = np.vstack(geo_points)
    is_geographic = bool(
        np.all(np.abs(all_geo[:, 0]) <= 180.0 + 1e-3)
        and np.all(np.abs(all_geo[:, 1]) <= 90.0 + 1e-3)
    )

    center_x = float(all_geo[:, 0].mean())
    center_y = float(all_geo[:, 1].mean())
    if is_geographic:
        x_scale = 111_000.0 * np.cos(np.radians(center_y))
        y_scale = 111_000.0
    else:
        x_scale = 1.0
        y_scale = 1.0

    for record in building_records:
        positions_geo = np.asarray(record["positions_geo"], dtype=np.float32)
        record["positions_local"] = np.column_stack(
            [
                (positions_geo[:, 0] - center_x) * x_scale,
                positions_geo[:, 2],
                -(positions_geo[:, 1] - center_y) * y_scale,
            ]
        ).astype(np.float32)

    all_local = np.vstack([np.asarray(record["positions_local"], dtype=np.float32) for record in building_records])
    min_x, max_x = float(all_local[:, 0].min()), float(all_local[:, 0].max())
    min_z, max_z = float(all_local[:, 2].min()), float(all_local[:, 2].max())
    pad_x = max((max_x - min_x) * 0.24, 40.0)
    pad_z = max((max_z - min_z) * 0.28, 40.0)

    ground = np.array(
        [
            [min_x - pad_x, 0.0, min_z - pad_z],
            [max_x + pad_x, 0.0, min_z - pad_z * 0.7],
            [max_x + pad_x * 1.2, 0.0, max_z + pad_z],
            [min_x - pad_x * 0.8, 0.0, max_z + pad_z * 1.15],
        ],
        dtype=np.float32,
    )

    yaw = np.radians(-45.0)
    pitch = np.radians(46.0)
    rot_y = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float32,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float32,
    )
    rotation = rot_x @ rot_y

    light = normalize(np.array([-0.55, 1.15, -0.42], dtype=np.float32))

    def rotate(points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float32) @ rotation.T

    def shadow_on_ground(points: np.ndarray) -> np.ndarray:
        heights = points[:, 1:2]
        shadow = points.copy()
        shadow[:, 0:1] -= light[0] * (heights / light[1])
        shadow[:, 2:3] -= light[2] * (heights / light[1])
        shadow[:, 1] = 0.0
        return shadow

    shadow_points = np.vstack(
        [shadow_on_ground(np.asarray(record["positions_local"], dtype=np.float32)) for record in building_records]
    )
    fit_points = np.vstack([rotate(all_local), rotate(shadow_points), rotate(ground)])
    fit_x = fit_points[:, 0]
    fit_y = -fit_points[:, 1]

    span_x = max(float(fit_x.max() - fit_x.min()), 1.0)
    span_y = max(float(fit_y.max() - fit_y.min()), 1.0)
    margin_x = width * 0.08
    margin_y = height * 0.10
    scale = min((width - 2 * margin_x) / span_x, (height - 2 * margin_y) / span_y)
    scale *= zoom
    center_x = width * 0.5
    center_y = height * 0.58
    proj_center_x = float((fit_x.min() + fit_x.max()) * 0.5)
    proj_center_y = float((fit_y.min() + fit_y.max()) * 0.5)

    def project(points: np.ndarray) -> np.ndarray:
        rotated = rotate(points)
        return np.column_stack(
            [
                center_x + (rotated[:, 0] - proj_center_x) * scale,
                center_y + (-rotated[:, 1] - proj_center_y) * scale,
                rotated[:, 2],
            ]
        )

    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")

    ground_poly = [tuple(pt[:2]) for pt in project(ground)]
    draw.polygon(ground_poly, fill=(199, 206, 198, 255), outline=(124, 132, 123, 230))

    guide_color = (131, 139, 129, 82)
    guide_width = max(1, int(round(ui_scale)))
    for t in np.linspace(0.15, 0.85, 5):
        start = ground[0] * (1.0 - t) + ground[3] * t
        end = ground[1] * (1.0 - t) + ground[2] * t
        line_pts = [tuple(pt[:2]) for pt in project(np.vstack([start, end]))]
        draw.line(line_pts, fill=guide_color, width=guide_width)
    for t in np.linspace(0.16, 0.84, 5):
        start = ground[0] * (1.0 - t) + ground[1] * t
        end = ground[3] * (1.0 - t) + ground[2] * t
        line_pts = [tuple(pt[:2]) for pt in project(np.vstack([start, end]))]
        draw.line(line_pts, fill=guide_color, width=guide_width)

    pedestal = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pedestal_draw = ImageDraw.Draw(pedestal, "RGBA")
    pedestal_draw.ellipse(
        (
            width * 0.18,
            height * 0.69,
            width * 0.84,
            height * 0.94,
        ),
        fill=(86, 84, 76, 28),
    )
    canvas.alpha_composite(
        pedestal.filter(ImageFilter.GaussianBlur(radius=max(20, int(round(20 * ui_scale)))))
    )

    shadow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer, "RGBA")
    triangle_draw_list: list[tuple[float, list[tuple[float, float]], tuple[int, int, int, int]]] = []

    for record in building_records:
        positions = np.asarray(record["positions_local"], dtype=np.float32)
        for tri_indices in np.asarray(record["indices"], dtype=np.uint32):
            triangle = positions[tri_indices]
            shadow_poly = [tuple(pt[:2]) for pt in project(shadow_on_ground(triangle))]
            shadow_draw.polygon(shadow_poly, fill=(48, 42, 34, 34))

            edge_a = triangle[1] - triangle[0]
            edge_b = triangle[2] - triangle[0]
            normal = normalize(np.cross(edge_a, edge_b))
            is_roof = abs(float(normal[1])) > 0.72
            base_rgba = np.array(record["roof_rgba"] if is_roof else record["wall_rgba"], dtype=np.float32)
            if is_roof:
                brightness = 0.92
            else:
                brightness = float(np.clip(0.62 + np.dot(normal, light) * 0.24, 0.48, 0.92))
            shaded_rgb = np.clip(
                base_rgba[:3] * brightness + np.array([8.0, 6.0, 4.0], dtype=np.float32),
                0.0,
                255.0,
            )
            fill = tuple(int(channel) for channel in (*shaded_rgb, 255))
            projected = project(triangle)
            triangle_draw_list.append(
                (
                    float(projected[:, 2].mean()),
                    [tuple(pt[:2]) for pt in projected],
                    fill,
                )
            )

    shadow_layer = shadow_layer.filter(
        ImageFilter.GaussianBlur(radius=max(10, int(round(10 * ui_scale))))
    )
    canvas.alpha_composite(shadow_layer)

    for _, polygon, fill in sorted(triangle_draw_list, key=lambda item: item[0]):
        draw.polygon(polygon, fill=fill)

    out_path.unlink(missing_ok=True)
    canvas.save(out_path, optimize=True)


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
    if base.size != (width, img_h):
        base = base.resize((width, img_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (width, height), bg)
    draw = ImageDraw.Draw(canvas)

    # Title bar
    title_font = font(max(16, int(round(bar_height * 0.42))), bold=True)
    label = f"{title} \u00b7 {subtitle}"
    draw.text((max(14, int(round(bar_height * 0.35))), bar_height // 2), label, font=title_font,
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
        "--cam-radius", "5200",
        "--pbr", "--msaa", "8",
        "--shadow-technique", "pcss", "--shadow-map-res", "4096",
        "--exposure", "1.35",
        "--sun-azimuth", "305", "--sun-elevation", "24",
        "--height-ao", "--height-ao-strength", "1.2",
        "--height-ao-directions", "8", "--height-ao-steps", "24",
        "--sun-vis", "--sun-vis-mode", "soft",
        "--sun-vis-samples", "6", "--sun-vis-steps", "32",
        "--normal-strength", "1.1",
        "--snow", "--snow-altitude", "3200", "--snow-blend", "300",
        "--snow-slope", "50",
        "--rock", "--rock-slope", "42",
        "--tonemap", "aces",
        "--white-balance", "--temperature", "6000",
        "--lens-vignette", "0.25",
        "--sky", "--sky-turbidity", "2.5",
        "--snapshot", str(tmp),
    ], tmp)

    compose_titled(tmp, out, "Mount Rainier", "dem_rainier.tif")


def render_02_fuji_labels() -> None:
    """Mount Fuji with GeoPackage labels via fuji_labels_demo.py."""
    print("[gallery] 02 Mount Fuji Labels")
    tmp = WORK_DIR / "02-base.png"
    out = IMAGES_DIR / "02-mount-fuji-labels.png"

    run_example_interactive("02 fuji", [
        "examples/fuji_labels_demo.py",
        "--width", str(W), "--height", str(H),
        "--pbr", "--msaa", "8",
        "--shadows", "pcss", "--shadow-map-res", "4096",
        "--exposure", "1.3",
        "--sun-azimuth", "296", "--sun-elevation", "26",
        "--height-ao", "--height-ao-strength", "1.2",
        "--height-ao-directions", "8", "--height-ao-steps", "24",
        "--sun-vis", "--sun-vis-mode", "soft",
        "--sun-vis-samples", "6", "--sun-vis-steps", "32",
        "--normal-strength", "1.1",
        "--snow", "--snow-altitude", "2800", "--snow-blend", "400",
        "--snow-slope", "50",
        "--rock", "--rock-slope", "40",
        "--tonemap", "aces",
        "--white-balance", "--temperature", "6200",
        "--lens-vignette", "0.2",
        "--sky", "--sky-turbidity", "2.0",
    ], [
        "set phi=332 theta=18 radius=3650 fov=34",
        f"snap {tmp} {W}x{H}",
    ], tmp)

    compose_titled(tmp, out, "Mount Fuji", "labels from GeoPackage")


def render_03_swiss_landcover() -> None:
    """Swiss terrain with raster land-cover overlay."""
    print("[gallery] 03 Swiss Landcover")
    tmp = WORK_DIR / "03-base.png"
    out = IMAGES_DIR / "03-swiss-landcover.png"

    run_example("03 swiss", [
        "examples/swiss_terrain_landcover_viewer.py",
        "--width", "3840",
        "--height", "3840",
        "--snapshot", str(tmp),
    ], tmp)

    # The Swiss example now composes its own title/caption/legend layout.
    from shutil import copy2
    copy2(tmp, out)


def render_04_luxembourg_rail() -> None:
    """Luxembourg rail network overlay via luxembourg_rail_overlay.py."""
    print("[gallery] 04 Luxembourg Rail Network")
    tmp = WORK_DIR / "04-base.png"
    out = IMAGES_DIR / "04-luxembourg-rail-network.png"

    run_example("04 rail", [
        "examples/luxembourg_rail_overlay.py",
        "--width", "3840", "--height", "3840",
        "--phi", "90",
        "--radius", "3400",
        "--msaa", "8",
        "--shadow-technique", "pcss", "--shadow-map-res", "4096",
        "--exposure", "1.4",
        "--sun-azimuth", "215", "--sun-elevation", "32",
        "--sun-intensity", "1.5",
        "--height-ao", "--height-ao-strength", "0.5",
        "--sun-vis", "--sun-vis-mode", "soft",
        "--normal-strength", "0.9",
        "--wetness", "--wetness-strength", "0.25",
        "--tonemap", "aces",
        "--white-balance", "--temperature", "6200",
        "--lens-vignette", "0.2",
        "--sky", "--sky-turbidity", "3.0",
        "--no-solid",
        "--line-width", "22.0",
        "--line-color", "#D92626",
        "--snapshot", str(tmp),
    ], tmp)

    compose_titled(tmp, out, "Luxembourg", "rail network overlay",
                   width=3840, height=3840)



def render_05_buildings() -> None:
    """3D buildings rendered from CityJSON geometry."""
    print("[gallery] 05 3D Buildings")
    tmp = WORK_DIR / "05-base.png"
    out = IMAGES_DIR / "05-3d-buildings.png"
    canvas_size = 3840
    bar_height = 96
    scene_height = canvas_size - bar_height
    layer = f3d.add_buildings_cityjson(REPO_ROOT / "assets" / "geojson" / "10-270-592.city.json")
    render_cityjson_building_preview(
        layer,
        tmp,
        width=canvas_size,
        height=scene_height,
        focus_subset=False,
        zoom=1.18,
    )
    compose_titled(
        tmp,
        out,
        "3D Buildings",
        "10-270-592.city.json",
        width=canvas_size,
        height=canvas_size,
        bar_height=bar_height,
        bg=(22, 24, 27, 255),
    )


def render_06_point_cloud() -> None:
    """LiDAR point cloud (Mt St Helens)."""
    print("[gallery] 06 Point Cloud")
    tmp = WORK_DIR / "06-base.png"
    out = IMAGES_DIR / "06-point-cloud.png"

    run_example("06 pointcloud", [
        "examples/pointcloud_viewer_interactive.py",
        "--input", "assets/lidar/MtStHelens.laz",
        "--width", "1600", "--height", "912",
        "--point-size", "3.5",
        "--max-points", "750000",
        "--color-mode", "rgb",
        "--phi", "0.6", "--theta", "0.5", "--radius", "1.4",
        "--snapshot", str(tmp),
    ], tmp)

    compose_titled(tmp, out, "Mt St. Helens", "LiDAR point cloud")


def render_07_flyover() -> None:
    """Camera flyover — 3 orbit frames composited side-by-side."""
    print("[gallery] 07 Camera Flyover")
    out = IMAGES_DIR / "07-camera-flyover.png"

    frame_w = W // 3
    angles = [0, 120, 240]
    frame_paths: list[Path] = []

    for i, phi in enumerate(angles):
        tmp = WORK_DIR / f"07-frame-{phi}.png"
        run_example(f"07 frame {phi}deg", [
            "examples/terrain_viewer_interactive.py",
            "--dem", "assets/tif/dem_rainier.tif",
            "--width", str(frame_w), "--height", str(H - 72),
            "--pbr", "--msaa", "8",
            "--shadow-technique", "pcss", "--shadow-map-res", "4096",
            "--exposure", "1.25",
            "--sun-azimuth", str(305 + phi), "--sun-elevation", "28",
            "--height-ao", "--height-ao-strength", "1.2",
            "--sun-vis", "--sun-vis-mode", "soft",
            "--normal-strength", "1.1",
            "--snow", "--snow-altitude", "3200", "--snow-blend", "300",
            "--rock", "--rock-slope", "42",
            "--tonemap", "aces",
            "--white-balance", "--temperature", "6000",
            "--sky", "--sky-turbidity", "2.5",
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
        [(0, 6), (34, 0), (74, 10), (78, 44), (36, 58), (8, 42)],
        fill_color=(0.78, 0.86, 0.68, 0.96),
        stroke_color=(0.34, 0.46, 0.28, 1.0),
        stroke_width=2.0,
    )
    scene.add_polygon(
        [(26, 62), (44, 78), (70, 74), (86, 92), (74, 114), (42, 108)],
        fill_color=(0.36, 0.67, 0.86, 0.94),
        stroke_color=(0.11, 0.27, 0.41, 1.0),
        stroke_width=2.0,
    )
    scene.add_polygon(
        [(82, 12), (124, 26), (120, 74), (72, 68)],
        fill_color=(0.94, 0.82, 0.55, 0.9),
        stroke_color=(0.57, 0.40, 0.14, 1.0),
        stroke_width=2.0,
    )
    scene.add_polygon(
        [(100, 88), (150, 100), (184, 88), (212, 112), (194, 138),
         (148, 146), (110, 126)],
        fill_color=(0.72, 0.58, 0.43, 0.9),
        stroke_color=(0.34, 0.22, 0.14, 1.0),
        stroke_width=2.0,
    )
    scene.add_polygon(
        [(132, 18), (178, 20), (220, 8), (220, 54), (186, 76), (140, 64)],
        fill_color=(0.67, 0.82, 0.60, 0.92),
        stroke_color=(0.31, 0.45, 0.26, 1.0),
        stroke_width=2.0,
    )
    scene.add_polyline(
        [(8, 52), (42, 60), (84, 82), (118, 96), (166, 106), (214, 118)],
        stroke_color=(0.12, 0.32, 0.60, 1.0),
        stroke_width=5.0,
    )
    scene.add_label(
        "River",
        position=(124, 94),
        font_size=18,
        color=(0.08, 0.18, 0.30, 1.0),
    )
    scene.add_polyline(
        [(12, 18), (54, 34), (96, 48), (146, 42), (202, 36)],
        stroke_color=(0.15, 0.14, 0.14, 1.0),
        stroke_width=6.0,
    )
    scene.add_label(
        "Road",
        position=(162, 42),
        font_size=18,
        color=(0.07, 0.08, 0.10, 1.0),
    )
    scene.add_polyline(
        [(88, 108), (114, 120), (136, 124), (164, 132), (196, 124)],
        stroke_color=(0.55, 0.40, 0.20, 0.9),
        stroke_width=3.0,
    )
    scene.add_label(
        "Valley",
        position=(38, 34),
        font_size=18,
        color=(0.20, 0.26, 0.16, 1.0),
    )
    scene.add_label(
        "Peak",
        position=(164, 126),
        font_size=18,
        color=(0.19, 0.14, 0.12, 1.0),
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

    grid_left = 40
    grid_top = 36
    grid_right = W - 40
    grid_bottom = H - 36
    for x in range(grid_left, grid_right + 1, 48):
        alpha = 30 if (x - grid_left) % 144 == 0 else 14
        draw.line((x, grid_top, x, grid_bottom), fill=(60, 72, 82, alpha), width=1)
    for y in range(grid_top, grid_bottom + 1, 48):
        alpha = 30 if (y - grid_top) % 144 == 0 else 14
        draw.line((grid_left, y, grid_right, y), fill=(60, 72, 82, alpha), width=1)

    draw.rounded_rectangle(
        (28, 28, W - 28, H - 28), radius=24,
        outline=(36, 43, 50, 32), width=2,
    )
    draw.rounded_rectangle(
        (54, 44, 404, 138), radius=18, fill=(255, 255, 255, 228)
    )
    draw.text((76, 70), "Vector Export", font=font(32, bold=True),
              fill=(24, 31, 36, 255), anchor="lm")
    draw.text((76, 102), "SVG scene preview with map annotations", font=font(18),
              fill=(72, 82, 90, 255), anchor="lm")
    draw.text((76, 126), "graticule, labels, legend", font=font(16),
              fill=(98, 106, 114, 255), anchor="lm")

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
        draw.ellipse(
            (pt[0] - 3, pt[1] - 3, pt[0] + 3, pt[1] + 3),
            fill=(34, 39, 46, 220),
        )
        draw_halo_text(
            draw, (pt[0], pt[1] - 12), label.text,
            label_font=font(int(round(label.style.font_size)), bold=True),
            fill=rgba(label.style.color),
            halo=rgba(label.style.halo_color),
        )

    legend_box = (868, 66, 1140, 246)
    draw.rounded_rectangle(legend_box, radius=18, fill=(255, 255, 255, 226),
                           outline=(44, 52, 60, 42), width=2)
    draw.text((892, 92), "Legend", font=font(24, bold=True),
              fill=(28, 34, 40, 255), anchor="lm")
    draw.line((892, 128, 952, 128), fill=(18, 82, 153, 255), width=5)
    draw.text((970, 128), "River", font=font(17), fill=(54, 63, 72, 255), anchor="lm")
    draw.line((892, 158, 952, 158), fill=(38, 36, 36, 255), width=6)
    draw.text((970, 158), "Road", font=font(17), fill=(54, 63, 72, 255), anchor="lm")
    draw.rounded_rectangle((892, 182, 952, 206), radius=8,
                           fill=(239, 208, 140, 255), outline=(145, 102, 36, 255))
    draw.text((970, 194), "Valley floor", font=font(17),
              fill=(54, 63, 72, 255), anchor="lm")
    draw.polygon([(892, 232), (922, 208), (952, 232)], fill=(161, 118, 88, 255))
    draw.text((970, 226), "Peak zone", font=font(17),
              fill=(54, 63, 72, 255), anchor="lm")

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

    # Morning: east sun - dramatic low-angle lighting
    run_example("09 morning", [
        "examples/terrain_viewer_interactive.py",
        "--dem", "assets/tif/dem_rainier.tif",
        "--width", str(panel_w), "--height", str(panel_h),
        "--pbr", "--msaa", "8",
        "--shadow-technique", "pcss", "--shadow-map-res", "4096",
        "--exposure", "1.2",
        "--sun-azimuth", "85", "--sun-elevation", "12",
        "--height-ao", "--height-ao-strength", "1.2",
        "--sun-vis", "--sun-vis-mode", "soft",
        "--sun-vis-samples", "8", "--sun-vis-steps", "48",
        "--normal-strength", "1.1",
        "--snow", "--snow-altitude", "3200", "--snow-blend", "300",
        "--snow-slope", "50",
        "--rock", "--rock-slope", "42",
        "--tonemap", "aces",
        "--white-balance", "--temperature", "7500",
        "--sky", "--sky-turbidity", "2.5",
        "--snapshot", str(left_tmp),
    ], left_tmp)

    # Evening: west sun - warm golden hour
    run_example("09 evening", [
        "examples/terrain_viewer_interactive.py",
        "--dem", "assets/tif/dem_rainier.tif",
        "--width", str(panel_w), "--height", str(panel_h),
        "--pbr", "--msaa", "8",
        "--shadow-technique", "pcss", "--shadow-map-res", "4096",
        "--exposure", "1.2",
        "--sun-azimuth", "275", "--sun-elevation", "45",
        "--height-ao", "--height-ao-strength", "1.2",
        "--sun-vis", "--sun-vis-mode", "soft",
        "--sun-vis-samples", "8", "--sun-vis-steps", "48",
        "--normal-strength", "1.1",
        "--snow", "--snow-altitude", "3200", "--snow-blend", "300",
        "--snow-slope", "50",
        "--rock", "--rock-slope", "42",
        "--tonemap", "aces",
        "--white-balance", "--temperature", "5500",
        "--sky", "--sky-turbidity", "2.5",
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
    dem_path = REPO_ROOT / "assets" / "tif" / "dem_rainier.tif"
    legend_domain = (0.0, 4000.0)

    try:
        import rasterio

        with rasterio.open(dem_path) as src:
            band = src.read(1, masked=True)
            legend_domain = (float(band.min()), float(band.max()))
    except Exception as exc:
        print(f"  [10 map plate] warning: using fallback legend domain: {exc}")

    # Render base terrain
    run_example("10 terrain", [
        "examples/terrain_viewer_interactive.py",
        "--dem", str(dem_path),
        "--width", "900", "--height", "560",
        "--pbr", "--msaa", "8",
        "--shadow-technique", "pcss", "--shadow-map-res", "4096",
        "--exposure", "1.25",
        "--sun-azimuth", "315", "--sun-elevation", "30",
        "--height-ao", "--height-ao-strength", "1.2",
        "--sun-vis", "--sun-vis-mode", "soft",
        "--normal-strength", "1.1",
        "--snow", "--snow-altitude", "3200", "--snow-blend", "300",
        "--rock", "--rock-slope", "42",
        "--tonemap", "aces",
        "--white-balance", "--temperature", "6000",
        "--lens-vignette", "0.15",
        "--sky", "--sky-turbidity", "2.0",
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

    cmap = build_natural_relief_colormap(legend_domain)
    legend = Legend.from_colormap(
        cmap,
        domain=legend_domain,
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
