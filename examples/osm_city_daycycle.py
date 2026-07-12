#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from _import_shim import ensure_repo_import

ensure_repo_import()

import osm_city_demo as city  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "examples" / "out" / "osm_city_daycycle" / "copenhagen_daycycle.mp4"
LIGHT_SURFACE_COLORS = {
    city.COLORS["base"],
    city.COLORS["water"],
}
SHADOW_TINT_RGB = (58, 71, 90)
TIMER_FONT_CANDIDATES = (
    "/Users/mpopovic3/Library/Fonts/Inconsolata.ttf",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "DejaVuSansMono.ttf",
    "DejaVuSans.ttf",
)


@dataclass(frozen=True)
class SunState:
    t: float
    noon_weight: float
    azimuth_deg: float
    elevation_deg: float
    light_dir: np.ndarray
    shadow_strength: float
    sky_top_rgb: tuple[int, int, int]
    sky_bottom_rgb: tuple[int, int, int]


@dataclass(frozen=True)
class PreparedSurface:
    surface: city.SurfaceLayer
    mask: Image.Image


@dataclass(frozen=True)
class PreparedTriangle:
    depth: float
    polygon: list[tuple[float, float]]
    world: np.ndarray
    normal: np.ndarray
    mean_y: float
    layer_min_y: float
    layer_height: float
    rgba: tuple[int, int, int, int]
    specular: float
    shadow_alpha: int
    radial_t: float
    is_wall: bool


@dataclass(frozen=True)
class PreparedScene:
    width: int
    height: int
    render_width: int
    render_height: int
    supersample: int
    radius: float
    world_to_px: float
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_deg: float
    view_dir: np.ndarray
    far_dir_screen: np.ndarray
    scale: float
    offset_x: float
    offset_y: float
    surfaces: list[PreparedSurface]
    triangles: list[PreparedTriangle]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a lightweight OSM city day-cycle video with the sun moving from "
            "east to west and building shadows updating every frame."
        )
    )
    parser.add_argument("--lon", type=float, default=12.56553, help="AOI center longitude.")
    parser.add_argument("--lat", type=float, default=55.67594, help="AOI center latitude.")
    parser.add_argument("--radius", type=float, default=700.0, help="AOI radius in meters.")
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--frames", type=int, default=240, help="Number of animation frames to render.")
    parser.add_argument("--fps", type=int, default=24, help="Video frame rate.")
    parser.add_argument("--supersample", type=int, default=1, help="Internal render scale before downsampling.")
    parser.add_argument("--clock-start-hour", type=float, default=4.0, help="Clock overlay start hour.")
    parser.add_argument("--clock-end-hour", type=float, default=23.0, help="Clock overlay end hour.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output MP4 path.")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Optional directory for rendered PNG frames.")
    parser.add_argument("--keep-frames", action="store_true", help="Keep the PNG frame sequence after encoding.")
    parser.add_argument("--refresh-osm", action="store_true", help="Ignore cached Overpass responses.")
    return parser.parse_args()


def lerp_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(x + (y - x) * float(t))) for x, y in zip(a, b))


def load_timer_font(size_px: int) -> ImageFont.ImageFont:
    for font_name in TIMER_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_name, size=max(1, int(size_px)))
        except OSError:
            continue
    return ImageFont.load_default()


def fit_timer_font(text: str, *, target_size_px: int, max_width_px: int, min_size_px: int) -> ImageFont.ImageFont:
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
    for size_px in range(max(1, int(target_size_px)), max(0, int(min_size_px)) - 1, -4):
        font = load_timer_font(size_px)
        if hasattr(probe, "textbbox"):
            bbox = probe.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
        else:
            text_w = int(round(probe.textlength(text, font=font)))
        if text_w <= max_width_px:
            return font
    return load_timer_font(min_size_px)


def format_clock_text(frame_index: int, total_frames: int, start_hour: float, end_hour: float) -> str:
    t = frame_index / max(total_frames - 1, 1)
    hour_value = float(start_hour) + (float(end_hour) - float(start_hour)) * t
    total_minutes = int(round(hour_value * 60.0))
    wrapped_minutes = total_minutes % (24 * 60)
    hour24 = wrapped_minutes // 60
    minute = wrapped_minutes % 60
    suffix = "AM" if hour24 < 12 else "PM"
    hour12 = hour24 % 12
    if hour12 == 0:
        hour12 = 12
    return f"{hour12:02d}:{minute:02d} {suffix}"


def add_timer_overlay(
    image: Image.Image,
    *,
    frame_index: int,
    total_frames: int,
    start_hour: float,
    end_hour: float,
) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    scale = canvas.width / 1280.0
    text = format_clock_text(frame_index, total_frames, start_hour, end_hour)
    font = fit_timer_font(
        text,
        target_size_px=max(48, int(round(68 * scale))),
        max_width_px=max(220, int(round(canvas.width * 0.25))),
        min_size_px=max(42, int(round(60 * scale))),
    )
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    else:
        text_w = int(round(draw.textlength(text, font=font)))
        text_h = getattr(font, "size", 18)
        bbox = (0, 0, text_w, text_h)

    pad_x = max(18, int(round(24 * scale)))
    pad_y = max(12, int(round(16 * scale)))
    margin = max(14, int(round(18 * scale)))
    radius = max(12, int(round(18 * scale)))
    box_w = text_w + pad_x * 2
    box_h = text_h + pad_y * 2
    x1 = canvas.width - margin
    x0 = x1 - box_w
    y0 = margin
    y1 = y0 + box_h

    draw.rounded_rectangle(
        [(x0, y0), (x1, y1)],
        radius=radius,
        fill=(18, 24, 32, 188),
        outline=(255, 255, 255, 56),
        width=max(1, int(round(1.5 * scale))),
    )
    text_x = int(round(x0 + (box_w - text_w) * 0.5 - bbox[0]))
    text_y = int(round(y0 + (box_h - text_h) * 0.5 - bbox[1]))
    draw.text(
        (text_x, text_y),
        text,
        font=font,
        fill=(248, 246, 242, 255),
    )
    return canvas


def sun_direction_from_angles(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    horizontal = math.cos(elevation)
    return city.normalize(
        np.array(
            [
                horizontal * math.sin(azimuth),
                math.sin(elevation),
                -horizontal * math.cos(azimuth),
            ],
            dtype=np.float32,
        )
    )


def sun_state_for_frame(frame_index: int, total_frames: int) -> SunState:
    t = frame_index / max(total_frames - 1, 1)
    noon_weight = math.sin(math.pi * t) ** 0.9
    azimuth_deg = 82.0 + 196.0 * t
    elevation_deg = 11.0 + 55.0 * noon_weight
    light_dir = sun_direction_from_angles(azimuth_deg, elevation_deg)
    shadow_strength = 0.78 + 0.26 * (1.0 - noon_weight)

    if t <= 0.5:
        mix = t / 0.5
        sky_top = lerp_rgb((77, 113, 181), (115, 182, 233), mix)
        sky_bottom = lerp_rgb((248, 188, 121), (220, 238, 255), mix)
    else:
        mix = (t - 0.5) / 0.5
        sky_top = lerp_rgb((115, 182, 233), (83, 110, 175), mix)
        sky_bottom = lerp_rgb((220, 238, 255), (255, 196, 126), mix)

    return SunState(
        t=t,
        noon_weight=noon_weight,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        light_dir=light_dir,
        shadow_strength=shadow_strength,
        sky_top_rgb=sky_top,
        sky_bottom_rgb=sky_bottom,
    )


def project_points_quiet(
    points: np.ndarray,
    *,
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        return city.project_points(
            points,
            eye=eye,
            target=target,
            up=up,
            width=width,
            height=height,
            fov_deg=fov_deg,
        )


def project_surface_parts_quiet(
    surface: city.SurfaceLayer,
    *,
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        return city.project_surface_parts(
            surface,
            eye=eye,
            target=target,
            up=up,
            width=width,
            height=height,
            fov_deg=fov_deg,
        )


def build_lightweight_scene(lon: float, lat: float, radius: float, *, refresh_osm: bool) -> city.SceneLayers:
    scene = city.build_city_scene(lon, lat, radius, refresh_osm=refresh_osm)
    for mesh in scene.meshes:
        mesh.shadow_alpha = max(int(mesh.shadow_alpha), 40)
        mesh.specular = 0.0
    return city.SceneLayers(
        surfaces=[surface for surface in scene.surfaces if surface.rgba in LIGHT_SURFACE_COLORS],
        meshes=scene.meshes,
        roof_outlines=[],
        focus_landmarks=[],
        radius=scene.radius,
    )


def prepare_scene(
    scene: city.SceneLayers,
    *,
    width: int,
    height: int,
    supersample: int,
    eye_scale: tuple[float, float, float] = (-1.74, 1.18, -1.46),
    target_height_ratio: float = 0.06,
    fov_deg: float = 33.0,
    margin_ratio: float = 0.052,
) -> PreparedScene:
    supersample = max(1, int(supersample))
    render_width = width * supersample
    render_height = height * supersample
    radius = float(scene.radius)
    eye = np.array(
        [radius * float(eye_scale[0]), radius * float(eye_scale[1]), radius * float(eye_scale[2])],
        dtype=np.float32,
    )
    target = np.array([0.0, radius * float(target_height_ratio), 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fov_deg = float(fov_deg)
    view_dir = city.normalize(eye - target)
    world_to_px = min(render_width, render_height) / max(radius * 2.0, 1e-6)

    surface_batches: list[tuple[city.SurfaceLayer, list[tuple[np.ndarray, list[np.ndarray]]]]] = []
    fit_inputs: list[np.ndarray] = []
    for surface in scene.surfaces:
        parts = project_surface_parts_quiet(
            surface,
            eye=eye,
            target=target,
            up=up,
            width=render_width,
            height=render_height,
            fov_deg=fov_deg,
        )
        if parts:
            surface_batches.append((surface, parts))
            fit_inputs.extend(exterior[:, :2] for exterior, _ in parts)

    mesh_projections: list[tuple[city.MeshLayer, np.ndarray]] = []
    for layer in scene.meshes:
        projected = project_points_quiet(
            layer.positions,
            eye=eye,
            target=target,
            up=up,
            width=render_width,
            height=render_height,
            fov_deg=fov_deg,
        )
        mesh_projections.append((layer, projected))
        fit_inputs.append(projected[:, :2])

    scale, offset_x, offset_y = city.compute_fit_transform(
        fit_inputs,
        width=render_width,
        height=render_height,
        margin_ratio=float(margin_ratio),
    )

    far_probe = project_points_quiet(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, radius * 0.30]], dtype=np.float32),
        eye=eye,
        target=target,
        up=up,
        width=render_width,
        height=render_height,
        fov_deg=fov_deg,
    )
    far_dir_screen = city.normalize(far_probe[1, :2] - far_probe[0, :2])
    if float(np.linalg.norm(far_dir_screen)) <= 1e-6:
        far_dir_screen = np.array([0.0, -1.0], dtype=np.float32)

    prepared_surfaces = [
        PreparedSurface(
            surface=surface,
            mask=city.build_surface_mask(
                parts,
                width=render_width,
                height=render_height,
                scale=scale,
                offset_x=offset_x,
                offset_y=offset_y,
            ),
        )
        for surface, parts in surface_batches
    ]

    mesh_points = []
    for _, projected in mesh_projections:
        pts = np.asarray(projected[:, :2], dtype=np.float32).copy()
        pts[:, 0] = pts[:, 0] * scale + offset_x
        pts[:, 1] = pts[:, 1] * scale + offset_y
        mesh_points.append(pts)
    packed = np.vstack(mesh_points) if mesh_points else np.array([[render_width * 0.5, render_height * 0.5]], dtype=np.float32)
    center_x = float(0.5 * (packed[:, 0].min() + packed[:, 0].max()))
    center_y = float(0.5 * (packed[:, 1].min() + packed[:, 1].max()))
    radius_x = max(1.0, float(0.5 * (packed[:, 0].max() - packed[:, 0].min())))
    radius_y = max(1.0, float(0.5 * (packed[:, 1].max() - packed[:, 1].min())))

    prepared_triangles: list[PreparedTriangle] = []
    for layer, projected in mesh_projections:
        layer_min_y = float(layer.positions[:, 1].min())
        layer_max_y = float(layer.positions[:, 1].max())
        layer_height = max(layer_max_y - layer_min_y, 1e-3)
        for tri in np.asarray(layer.indices, dtype=np.uint32):
            world = layer.positions[tri]
            screen = projected[tri]
            if np.any(screen[:, 2] <= 1.0):
                continue
            raw_normal = np.cross(world[1] - world[0], world[2] - world[0])
            normal_length = float(np.linalg.norm(raw_normal))
            if normal_length <= 1e-6:
                continue
            raw_normal = raw_normal / normal_length
            mean_y = float(world[:, 1].mean())
            if mean_y <= layer_min_y + 0.05 and abs(float(raw_normal[1])) > 0.92:
                continue
            polygon = city.fit_polygon(screen[:, :2], scale=scale, offset_x=offset_x, offset_y=offset_y)
            centroid_x = sum(point[0] for point in polygon) / 3.0
            centroid_y = sum(point[1] for point in polygon) / 3.0
            radial_t = math.sqrt(
                ((centroid_x - center_x) / radius_x) ** 2
                + ((centroid_y - center_y) / radius_y) ** 2
            )
            prepared_triangles.append(
                PreparedTriangle(
                    depth=float(screen[:, 2].mean()),
                    polygon=polygon,
                    world=world.copy(),
                    normal=raw_normal.astype(np.float32),
                    mean_y=mean_y,
                    layer_min_y=layer_min_y,
                    layer_height=layer_height,
                    rgba=layer.rgba,
                    specular=layer.specular,
                    shadow_alpha=int(layer.shadow_alpha),
                    radial_t=float(np.clip(radial_t, 0.0, 1.0)),
                    is_wall=abs(float(raw_normal[1])) < 0.55,
                )
            )
    prepared_triangles.sort(key=lambda item: item.depth, reverse=True)

    return PreparedScene(
        width=width,
        height=height,
        render_width=render_width,
        render_height=render_height,
        supersample=supersample,
        radius=radius,
        world_to_px=world_to_px,
        eye=eye,
        target=target,
        up=up,
        fov_deg=fov_deg,
        view_dir=view_dir,
        far_dir_screen=far_dir_screen,
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
        surfaces=prepared_surfaces,
        triangles=prepared_triangles,
    )


def make_background(width: int, height: int, sun: SunState) -> Image.Image:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    top = np.asarray(sun.sky_top_rgb, dtype=np.float32).reshape(1, 1, 3)
    bottom = np.asarray(sun.sky_bottom_rgb, dtype=np.float32).reshape(1, 1, 3)
    rgb = top * (1.0 - y) + bottom * y

    sun_x = 0.16 + 0.68 * sun.t
    sun_y = 0.16 + (1.0 - sun.noon_weight) * 0.06
    glow = np.exp(-(((x - sun_x) ** 2) / 0.012 + ((y - sun_y) ** 2) / 0.005))
    haze = city.smoothstep(0.55, 1.0, y[:, :, 0]).astype(np.float32)[:, :, None]
    rgb = rgb + glow * np.asarray([34.0, 24.0, 10.0], dtype=np.float32).reshape(1, 1, 3)
    rgb = rgb + haze * np.asarray([10.0, 10.0, 14.0], dtype=np.float32).reshape(1, 1, 3)
    rgba = np.concatenate(
        [np.clip(rgb, 0.0, 255.0).astype(np.uint8), np.full((height, width, 1), 255, dtype=np.uint8)],
        axis=2,
    )
    return Image.fromarray(rgba)


def shade_triangle(
    triangle: PreparedTriangle,
    prepared: PreparedScene,
    sun: SunState,
    *,
    shadow_tint_rgb: tuple[int, int, int] = SHADOW_TINT_RGB,
) -> tuple[int, int, int, int]:
    rgba = city.shade_rgba(
        triangle.rgba,
        normal=triangle.normal,
        light_dir=sun.light_dir,
        view_dir=prepared.eye - triangle.world.mean(axis=0),
        specular=triangle.specular,
    )
    if triangle.is_wall:
        wall_t = float(city.smoothstep(0.0, 1.0, np.clip((triangle.mean_y - triangle.layer_min_y) / triangle.layer_height, 0.0, 1.0)))
        wall_gain = 0.86 + 0.28 * wall_t
        wall_mix = (1.0 - wall_t) * 0.10
        base_rgb = np.asarray(rgba[:3], dtype=np.float32) * wall_gain
        wall_rgb = np.clip(
            base_rgb * (1.0 - wall_mix) + np.asarray(shadow_tint_rgb, dtype=np.float32) * wall_mix,
            0.0,
            255.0,
        ).astype(np.uint8)
        rgba = (int(wall_rgb[0]), int(wall_rgb[1]), int(wall_rgb[2]), rgba[3])
    return city.apply_depth_grade_rgba(rgba, radial_t=triangle.radial_t)


def render_frame(
    prepared: PreparedScene,
    sun: SunState,
    *,
    frame_index: int,
    total_frames: int,
    clock_start_hour: float,
    clock_end_hour: float,
    show_timer: bool = True,
    shadow_tint_rgb: tuple[int, int, int] | None = None,
    shadow_opacity: float = 1.0,
) -> Image.Image:
    shadow_tint = SHADOW_TINT_RGB if shadow_tint_rgb is None else shadow_tint_rgb
    image = make_background(prepared.render_width, prepared.render_height, sun)

    for item in prepared.surfaces:
        shaded = city.shade_rgba(
            item.surface.rgba,
            normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            light_dir=sun.light_dir,
            view_dir=prepared.view_dir,
            specular=item.surface.specular,
        )
        if item.surface.reflectivity > 0.0:
            layer_image = city.render_water_surface_layer(
                width=prepared.render_width,
                height=prepared.render_height,
                mask=item.mask,
                shaded_rgb=shaded[:3],
                alpha_value=item.surface.rgba[3],
                light_dir=sun.light_dir,
                specular_intensity=max(0.15, item.surface.specular),
                far_dir_screen=prepared.far_dir_screen,
            )
        else:
            alpha = item.mask.point(lambda value, a=item.surface.rgba[3]: (value * a + 127) // 255)
            layer_image = Image.new("RGBA", (prepared.render_width, prepared.render_height), shaded[:3] + (0,))
            layer_image.putalpha(alpha)
        image = Image.alpha_composite(image, layer_image)

    shadow_entries: list[tuple[float, list[tuple[float, float]], float, float]] = []
    for triangle in prepared.triangles:
        if triangle.mean_y <= triangle.layer_min_y + 0.05 or triangle.shadow_alpha <= 0 or sun.light_dir[1] <= 1e-4:
            continue
        shadow_world = triangle.world.copy()
        shadow_scale = shadow_world[:, 1:2] / sun.light_dir[1]
        shadow_world[:, 0:1] -= sun.light_dir[0] * shadow_scale
        shadow_world[:, 2:3] -= sun.light_dir[2] * shadow_scale
        shadow_world[:, 1] = 0.0
        if not np.isfinite(shadow_world).all():
            continue
        if float(np.max(np.abs(shadow_world[:, [0, 2]]))) > prepared.radius * 6.0:
            continue
        shadow_proj = project_points_quiet(
            shadow_world,
            eye=prepared.eye,
            target=prepared.target,
            up=prepared.up,
            width=prepared.render_width,
            height=prepared.render_height,
            fov_deg=prepared.fov_deg,
        )
        if np.any(shadow_proj[:, 2] <= 1.0):
            continue
        caster_height = max(0.0, triangle.mean_y - triangle.layer_min_y + 0.04)
        shadow_vec = shadow_world[:, [0, 2]].mean(axis=0) - triangle.world[:, [0, 2]].mean(axis=0)
        receiver_distance = float(np.linalg.norm(shadow_vec))
        blur_px = float(np.clip(1.2 + receiver_distance * prepared.world_to_px * 0.012, 1.0, prepared.render_width / 170.0))
        alpha = float(np.clip(triangle.shadow_alpha * sun.shadow_strength * (0.92 + caster_height / 52.0), 0.0, 255.0))
        shadow_entries.append(
            (
                float(shadow_proj[:, 2].mean()),
                city.fit_polygon(shadow_proj[:, :2], scale=prepared.scale, offset_x=prepared.offset_x, offset_y=prepared.offset_y),
                alpha,
                blur_px,
            )
        )

    blur_bands = (1.5, 3.0, 5.0, 7.0, 9.0, 12.0)
    grouped: dict[float, list[tuple[float, list[tuple[float, float]], float, float]]] = {band: [] for band in blur_bands}
    for shadow_entry in shadow_entries:
        band = min(blur_bands, key=lambda candidate: abs(candidate - shadow_entry[3]))
        grouped[band].append(shadow_entry)
    for blur_radius, entries in grouped.items():
        if not entries:
            continue
        layer = Image.new("L", (prepared.render_width, prepared.render_height), 0)
        layer_draw = ImageDraw.Draw(layer)
        for depth_value, polygon, alpha, _ in sorted(entries, key=lambda entry: entry[0], reverse=True):
            _ = depth_value
            layer_draw.polygon(polygon, fill=int(alpha))
        layer = layer.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
        layer_alpha = np.asarray(layer, dtype=np.float32) / 255.0
        layer_alpha = 1.0 - np.power(1.0 - layer_alpha, 1.35)
        layer_alpha *= max(0.58, 1.0 - blur_radius / 20.0)
        image = city.apply_multiply_tint(
            image,
            tint_rgb=shadow_tint,
            mask_alpha=np.clip(layer_alpha, 0.0, 1.0),
            opacity=float(np.clip(shadow_opacity, 0.0, 1.0)),
        )

    draw = ImageDraw.Draw(image, "RGBA")
    for triangle in prepared.triangles:
        draw.polygon(
            triangle.polygon,
            fill=shade_triangle(triangle, prepared, sun, shadow_tint_rgb=shadow_tint),
        )

    if prepared.supersample > 1:
        resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        image = image.resize((prepared.width, prepared.height), resample=resampling)
    if not show_timer:
        return image
    return add_timer_overlay(
        image,
        frame_index=frame_index,
        total_frames=total_frames,
        start_hour=clock_start_hour,
        end_hour=clock_end_hour,
    )


def encode_video(frames_dir: Path, output_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required to assemble the MP4 output.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(int(fps)),
            "-i",
            str(frames_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output_path),
        ],
        check=True,
    )


def main() -> int:
    args = parse_args()
    width, height = map(int, args.size)
    frame_count = max(1, int(args.frames))
    fps = max(1, int(args.fps))
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Daycycle] center=({args.lon:.5f}, {args.lat:.5f}) radius={args.radius:.0f}m")
    scene = build_lightweight_scene(args.lon, args.lat, args.radius, refresh_osm=bool(args.refresh_osm))
    if not scene.meshes:
        raise SystemExit("No buildings were generated for the requested AOI.")
    prepared = prepare_scene(scene, width=width, height=height, supersample=int(args.supersample))

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.frames_dir is not None:
        frames_dir = args.frames_dir.resolve()
        frames_dir.mkdir(parents=True, exist_ok=True)
    elif args.keep_frames:
        frames_dir = output_path.with_suffix("")
        frames_dir = frames_dir.parent / f"{frames_dir.name}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="osm_city_daycycle_", dir=str(output_path.parent))
        frames_dir = Path(temp_dir.name)

    try:
        for frame_index in range(frame_count):
            sun = sun_state_for_frame(frame_index, frame_count)
            frame = render_frame(
                prepared,
                sun,
                frame_index=frame_index,
                total_frames=frame_count,
                clock_start_hour=float(args.clock_start_hour),
                clock_end_hour=float(args.clock_end_hour),
            )
            frame.save(frames_dir / f"frame_{frame_index:05d}.png", format="PNG", compress_level=0)
            if frame_index == 0 or (frame_index + 1) % 24 == 0 or frame_index + 1 == frame_count:
                print(
                    f"[Daycycle] frame {frame_index + 1}/{frame_count} "
                    f"| sun az={sun.azimuth_deg:.1f} el={sun.elevation_deg:.1f}"
                )
        encode_video(frames_dir, output_path, fps=fps)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    print(f"[Daycycle] Wrote {city.display_path(output_path)}")
    if args.keep_frames or args.frames_dir is not None:
        print(f"[Daycycle] Frames saved in {city.display_path(frames_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
