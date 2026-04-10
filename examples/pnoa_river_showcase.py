#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from matplotlib.patches import Rectangle
from matplotlib.colors import LightSource, LinearSegmentedColormap
from rasterio.enums import Resampling
from rasterio.warp import transform, transform_bounds

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEM = PROJECT_ROOT / "assets" / "tif" / "Bryce_Canyon.tif"
DEFAULT_OUTPUT = PROJECT_ROOT / "examples" / "out" / "pnoa_river_showcase" / "pnoa_river_showcase.png"
CACHE_DIR = PROJECT_ROOT / "examples" / ".cache" / "pnoa_river_showcase"
OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
WATERWAY_STYLES = {
    "river": (5.0, "#0d7d97", "#67e8f9"),
    "canal": (4.0, "#12839d", "#74efff"),
    "stream": (3.0, "#1a8ca6", "#86f4ff"),
}
IMHOF_PALETTE = [
    "#edf4e5",
    "#d8e2bf",
    "#c9d1a1",
    "#cfbf8d",
    "#c9a36f",
    "#b48258",
    "#9d6e4c",
    "#8f7e71",
    "#bfb8b0",
    "#f3f1ec",
]
BATLOW_FALLBACK = [
    "#011959",
    "#124984",
    "#1c6bb1",
    "#2489be",
    "#2fa7b0",
    "#56c48b",
    "#a2d64f",
    "#e4d645",
    "#fbbd2d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Bryce Canyon terrain with OSM waterways.")
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, nargs=2, default=(3840, 3840), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument(
        "--max-dem-size",
        "--downsample",
        dest="max_dem_size",
        type=int,
        default=None,
        help="Downsample the DEM so its longest edge is this many samples.",
    )
    parser.add_argument("--palette", choices=["natural", "vibrant"], default="natural")
    parser.add_argument("--refresh-osm", action="store_true")
    parser.add_argument("--save-waterways-preview", action="store_true")
    parser.add_argument("--save-river-mask", dest="save_waterways_preview", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_dem(path: Path, max_dim: int | None) -> tuple[np.ndarray, float, float, tuple[float, float, float, float], str]:
    with rasterio.open(path) as src:
        scale = 1.0 if not max_dim or max_dim <= 0 else min(1.0, float(max_dim) / float(max(src.width, src.height)))
        width = max(1, int(round(src.width * scale)))
        height = max(1, int(round(src.height * scale)))
        data = src.read(1, out_shape=(height, width), resampling=Resampling.bilinear, masked=True)
        if np.ma.isMaskedArray(data):
            finite = np.asarray(data.compressed(), dtype=np.float32)
            fill = float(finite.min()) if finite.size else 0.0
            heightmap = np.asarray(data.filled(fill), dtype=np.float32)
        else:
            heightmap = np.asarray(data, dtype=np.float32)
        if not np.isfinite(heightmap).all():
            finite = heightmap[np.isfinite(heightmap)]
            fill = float(finite.min()) if finite.size else 0.0
            heightmap = np.where(np.isfinite(heightmap), heightmap, fill).astype(np.float32)
        bounds = (float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top))
        cell_x = (bounds[2] - bounds[0]) / float(width)
        cell_y = (bounds[3] - bounds[1]) / float(height)
        crs = str(src.crs)
    heightmap -= float(heightmap.min())
    return np.ascontiguousarray(heightmap), cell_x, cell_y, bounds, crs


def smooth_heightmap(heightmap: np.ndarray, passes: int = 1) -> np.ndarray:
    out = np.asarray(heightmap, dtype=np.float32)
    h, w = out.shape
    for _ in range(max(0, int(passes))):
        padded = np.pad(out, 1, mode="edge")
        neighbors = [padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        out = (neighbors[4] * 4.0 + sum(neighbors[i] for i in range(9) if i != 4)) / 12.0
    return np.ascontiguousarray(out, dtype=np.float32)


def osm_cache_path(dem_path: Path) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{dem_path.stem}_waterways_osm.json"


def fetch_osm_waterways(bounds: tuple[float, float, float, float], crs: str, cache_path: Path, refresh: bool) -> list[dict]:
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8")).get("elements", [])
    west, south, east, north = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    query = (
        "[out:json][timeout:90];"
        f'(way["waterway"~"^(river|stream|canal)$"]({south},{west},{north},{east}););'
        "out geom;"
    )
    payload = None
    last_error: Exception | None = None
    for endpoint in OVERPASS_URLS:
        for attempt in range(2):
            request = Request(
                endpoint,
                data=query.encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8", "User-Agent": "forge3d-codex/1.0"},
            )
            try:
                with urlopen(request, timeout=120) as response:
                    payload = json.load(response)
                break
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                sleep(1.5 * (attempt + 1))
        if payload is not None:
            break
    if payload is None:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8")).get("elements", [])
        raise SystemExit(f"Failed to fetch OSM waterways: {last_error}") from last_error
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload.get("elements", [])


def waterway_priority(feature: dict) -> float:
    return feature["length"] * {"river": 3.0, "canal": 1.5}.get(feature["waterway"], 1.0)


def project_waterways(elements: list[dict], dem_crs: str) -> list[dict]:
    features: list[dict] = []
    for element in elements:
        geometry = element.get("geometry") or []
        if len(geometry) < 2:
            continue
        lon = [float(point["lon"]) for point in geometry]
        lat = [float(point["lat"]) for point in geometry]
        x_proj, y_proj = transform("EPSG:4326", dem_crs, lon, lat)
        x = np.asarray(x_proj, dtype=np.float32)
        y = np.asarray(y_proj, dtype=np.float32)
        if x.size < 2:
            continue
        length = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
        if length <= 0.0:
            continue
        features.append(
            {
                "waterway": str(element.get("tags", {}).get("waterway", "stream")),
                "length": length,
                "x": x,
                "y": y,
            }
        )
    features.sort(key=waterway_priority, reverse=True)
    return features


def localize_waterways(
    features: list[dict],
    shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
    cell_x: float,
    cell_y: float,
) -> list[dict]:
    height, width = shape
    left, bottom, right, top = bounds
    margin_x = cell_x * 2.5
    margin_y = cell_y * 2.5
    localized: list[dict] = []
    for feature in features:
        x = feature["x"]
        y = feature["y"]
        inside = (
            (x >= left - margin_x)
            & (x <= right + margin_x)
            & (y >= bottom - margin_y)
            & (y <= top + margin_y)
        )
        if int(inside.sum()) < 2:
            continue
        local_x = (x[inside] - left) / max(cell_x, 1e-6)
        local_y = (top - y[inside]) / max(cell_y, 1e-6)
        local_x = np.clip(local_x, 0.0, float(width - 1))
        local_y = np.clip(local_y, 0.0, float(height - 1))
        if local_x.size >= 2:
            localized.append({"waterway": feature["waterway"], "length": feature["length"], "x": local_x, "y": local_y})
    localized.sort(key=waterway_priority, reverse=True)
    return localized


def build_waterway_masks(features: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    scale = 6
    resample = getattr(getattr(Image, "Resampling", None), "BILINEAR", getattr(Image, "BILINEAR"))
    halo_image = Image.new("L", (width * scale, height * scale), 0)
    core_image = Image.new("L", (width * scale, height * scale), 0)
    halo_draw = ImageDraw.Draw(halo_image)
    core_draw = ImageDraw.Draw(core_image)
    for feature in features:
        line_width, _, _ = WATERWAY_STYLES.get(feature["waterway"], WATERWAY_STYLES["stream"])
        points = [(float(x) * scale, float(y) * scale) for x, y in zip(feature["x"], feature["y"])]
        if len(points) < 2:
            continue
        halo_draw.line(points, fill=255, width=max(1, int(round((line_width + 0.3) * scale * 0.40))))
        core_draw.line(points, fill=255, width=max(1, int(round((line_width + 0.1) * scale * 0.32))))
    halo = np.asarray(halo_image.resize((width, height), resample=resample), dtype=np.float32) / 255.0
    core = np.asarray(core_image.resize((width, height), resample=resample), dtype=np.float32) / 255.0
    return halo, core


def save_waterways_preview(features: list[dict], output_path: Path, shape: tuple[int, int]) -> None:
    scale = 4
    height, width = shape
    image = Image.new("RGBA", (width * scale, height * scale), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    for feature in features:
        line_width, under, over = WATERWAY_STYLES.get(feature["waterway"], WATERWAY_STYLES["stream"])
        points = [(float(x) * scale, float(y) * scale) for x, y in zip(feature["x"], feature["y"])]
        if len(points) < 2:
            continue
        draw.line(points, fill=under, width=max(1, int(round((line_width + 1.4) * scale * 0.35))))
        draw.line(points, fill=over, width=max(1, int(round(line_width * scale * 0.30))))
    image.save(output_path)


def _nice_scale_length(span_m: float) -> int:
    candidates = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000], dtype=np.int32)
    target = max(100.0, span_m * 0.22)
    valid = candidates[candidates <= target]
    return int(valid[-1] if valid.size else candidates[0])


def _draw_map_furniture(figure, title: str, caption: str, span_m: float) -> None:
    overlay = figure.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
    overlay.set_axis_off()
    overlay.set_xlim(0.0, 1.0)
    overlay.set_ylim(0.0, 1.0)

    overlay.text(0.055, 0.95, title, ha="left", va="top", fontsize=20, fontweight="bold", color="#1f1f1f")
    overlay.text(0.055, 0.915, caption, ha="left", va="top", fontsize=10.5, color="#4b4b4b")

    overlay.annotate(
        "",
        xy=(0.92, 0.92),
        xytext=(0.92, 0.82),
        arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#1d1d1d", shrinkA=0, shrinkB=0),
    )
    overlay.text(0.92, 0.935, "N", ha="center", va="bottom", fontsize=14, fontweight="bold", color="#1d1d1d")

    scale_m = _nice_scale_length(span_m)
    half_scale_m = scale_m / 2.0
    def fmt_dist(value_m: float) -> str:
        if value_m >= 1000:
            km = value_m / 1000.0
            return f"{km:.0f} km" if abs(km - round(km)) < 1e-6 else f"{km:.1f} km"
        return f"{value_m:.0f} m"
    total_width = 0.20
    left = 0.055
    bottom = 0.08
    seg_w = total_width / 2.0
    box_h = 0.012
    overlay.add_patch(Rectangle((left, bottom), seg_w, box_h, facecolor="#1a1a1a", edgecolor="#1a1a1a", lw=0.8))
    overlay.add_patch(Rectangle((left + seg_w, bottom), seg_w, box_h, facecolor="white", edgecolor="#1a1a1a", lw=0.8))
    overlay.text(left, bottom - 0.012, "0", ha="center", va="top", fontsize=9.5, color="#333333")
    overlay.text(left + seg_w, bottom - 0.012, fmt_dist(half_scale_m), ha="center", va="top", fontsize=9.5, color="#333333")
    overlay.text(left + total_width, bottom - 0.012, fmt_dist(scale_m), ha="center", va="top", fontsize=9.5, color="#333333")


def _terrain_cmap(name: str):
    if name == "vibrant":
        try:
            import scicolor

            cmap = scicolor.get_cmap("batlow")
            if cmap is not None:
                return cmap
        except Exception:
            pass
        return LinearSegmentedColormap.from_list("batlow_fallback", BATLOW_FALLBACK)
    return LinearSegmentedColormap.from_list("natural_palette", IMHOF_PALETTE)


def render_scene(
    output_path: Path,
    heightmap: np.ndarray,
    waterways: list[dict],
    cell_x: float,
    cell_y: float,
    size_px: tuple[int, int],
    title: str,
    caption: str,
    palette_name: str,
) -> None:
    width_px, height_px = map(int, size_px)
    dpi = 180
    figure = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    axis = figure.add_subplot(111, projection="3d")
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")

    height, width = heightmap.shape
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    x = x_grid.astype(np.float32) * cell_x
    y = y_grid.astype(np.float32) * cell_y
    z = heightmap * 1.5
    base_z = -float(np.percentile(z, 20.0)) * 0.18
    span = float(max(x.max() - x.min(), y.max() - y.min()))

    terrain_cmap = _terrain_cmap(palette_name)
    normalized = np.clip(heightmap / max(float(np.percentile(heightmap, 99.8)), 1e-6), 0.0, 1.0)
    shade = LightSource(azdeg=315.0, altdeg=36.0).hillshade(heightmap, vert_exag=0.5, dx=cell_x, dy=cell_y, fraction=1.15)
    terrain_rgb = terrain_cmap(np.power(normalized, 0.96))[..., :3]
    shade_strength = 0.32 if palette_name == "vibrant" else 0.42
    terrain_rgb = np.clip(terrain_rgb * ((1.0 - shade_strength) + shade_strength * shade)[..., None], 0.0, 1.0)
    halo_mask, core_mask = build_waterway_masks(waterways, heightmap.shape)
    halo_color = np.array([13.0, 125.0, 151.0], dtype=np.float32) / 255.0
    core_color = np.array([103.0, 232.0, 249.0], dtype=np.float32) / 255.0
    terrain_rgb = terrain_rgb * (1.0 - halo_mask[..., None] * 0.58) + halo_color * (halo_mask[..., None] * 0.58)
    terrain_rgb = terrain_rgb * (1.0 - core_mask[..., None] * 0.82) + core_color * (core_mask[..., None] * 0.82)
    terrain_rgb = np.clip(terrain_rgb, 0.0, 1.0)

    axis.set_proj_type("persp", focal_length=0.95)
    axis.plot_surface(x, y, z, facecolors=terrain_rgb, linewidth=0, antialiased=False, shade=False, rstride=1, cstride=1)
    axis.view_init(elev=41.0, azim=-58.0)
    axis.set_box_aspect((x.max() - x.min(), y.max() - y.min(), (z.max() - base_z) * 0.70))
    axis.set_axis_off()
    axis.set_xlim(x.min() - span * 0.05, x.max() + span * 0.09)
    axis.set_ylim(y.min() - span * 0.08, y.max() + span * 0.03)
    axis.set_zlim(base_z - 5.0, z.max() * 1.05)
    axis.set_position([0.0, 0.0, 1.0, 1.0])
    plt.subplots_adjust(0.0, 0.0, 1.0, 1.0)
    _draw_map_furniture(figure, title, caption, span)
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    dem_path = args.dem.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    heightmap, cell_x, cell_y, bounds, crs = read_dem(dem_path, args.max_dem_size)
    heightmap = smooth_heightmap(heightmap, passes=1)
    waterways = project_waterways(fetch_osm_waterways(bounds, crs, osm_cache_path(dem_path), bool(args.refresh_osm)), crs)
    if not waterways:
        raise SystemExit("No OSM waterways found for the DEM extent.")
    localized = localize_waterways(waterways, heightmap.shape, bounds, cell_x, cell_y)
    if not localized:
        raise SystemExit("OSM waterways were found, but none intersect the render extent.")

    print(f"[Terrain] DEM: {display_path(dem_path)}")
    print(f"[Terrain] OSM waterways: {len(waterways)} features in DEM extent")
    print(f"[Terrain] Render grid: {heightmap.shape[1]}x{heightmap.shape[0]} samples")
    print(f"[Terrain] Waterways used: {len(localized)} features in full extent")

    title = "Bryce Canyon Relief and Rivers"
    caption = f"DEM: {dem_path.stem.replace('_', ' ')} | OSM waterways overlay | {args.palette} palette"
    render_scene(output_path, heightmap, localized, cell_x, cell_y, tuple(args.size), title, caption, args.palette)
    print(f"[Terrain] Wrote {display_path(output_path)}")

    if args.save_waterways_preview:
        preview_path = output_path.with_name(f"{output_path.stem}_waterways_preview.png")
        save_waterways_preview(localized, preview_path, heightmap.shape)
        print(f"[Terrain] Wrote {display_path(preview_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
