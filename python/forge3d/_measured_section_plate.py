"""C02 encodings and C06 plate composition for a measured transect."""

from dataclasses import asdict
from math import ceil, floor, log10
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from . import gis
from ._canonical_json import canonical_json_bytes
from ._map_scene_render import _draw_text, _text_outline_metrics
from ._png import load_png_rgba, save_png
from .bundle import _compute_sha256
from .map_plate import BBox, MapPlate, MapPlateConfig
from .measured_sections import MeasuredSection, _read_grid
from .publication import _check_frame_palette, _drawn_map_rect, _publication_scene
from .raster_style import CategoricalRasterStyle

INK = (33, 44, 48, 255)
ACCENT = (142, 43, 25, 255)
GRID = (206, 216, 216, 255)
GAP = (229, 232, 231, 255)
PAPER = (249, 249, 245, 255)
WHITE = (255, 255, 255, 255)
BREAKS = (-2500, -1500, -750, -250, 0, 350, 800)
BANDS = ("Below -2500", "-2500 to -1500", "-1500 to -750", "-750 to -250",
         "-250 to 0", "0 to 350", "350 to 800", "800 and above")
PALETTE = ((221, 225, 222, 255), (24, 54, 83, 255), (42, 80, 109, 255),
           (64, 113, 140, 255), (95, 150, 169, 255), (161, 201, 209, 255),
           (228, 225, 186, 255), (184, 184, 130, 255), (135, 132, 86, 255))
PLATE_SIZE = (1500, 1130)
MAP_REGION = (45, 114, 981, 738)
PROFILE = (134, 842, 1454, 1010)


def _text(image, message, x, y, size, *, align="left", color=INK, halo=False):
    width, _, _ = _text_outline_metrics(message, size)
    if align == "center":
        x -= width / 2
    elif align == "right":
        x -= width
    if x < 0 or x + width > image.shape[1]:
        raise ValueError(f"Section text does not fit the plate: {message}")
    _draw_text(image, message, (round(x), round(y)), color=color,
               halo=WHITE if halo else (0, 0, 0, 0), halo_width_px=2, font_size=size)


def _nice_step(span):
    # About five grid divisions, with familiar decimal tick values.
    exponent = 10 ** floor(log10(span / 5))
    return next(multiplier * exponent for multiplier in (1, 2, 5, 10)
                if multiplier * exponent >= span / 5)


def _axes(section):
    elevations = [sample.elevation_m for sample in section.samples if sample.elevation_m is not None]
    low, high = min([0.0, *elevations]), max([0.0, *elevations])
    # A level, all-zero or wholly missing transect still needs visible axes in metres.
    step = _nice_step(high - low) if high != low else 1.0
    bottom, top = (floor(low / step) - 1) * step, (ceil(high / step) + 1) * step
    length = section.samples[-1].distance_m
    x_step = _nice_step(length)
    x_max = ceil(length / x_step) * x_step
    x0, y0, x1, y1 = PROFILE
    return {
        "rect": PROFILE, "distance_m": [0.0, x_max], "elevation_m": [bottom, top],
        "distance_ticks_m": [i * x_step for i in range(round(x_max / x_step) + 1)],
        "elevation_ticks_m": [bottom + i * step for i in range(round((top - bottom) / step) + 1)],
        "vertical_exaggeration": (y1 - y0) * x_max / ((x1 - x0) * (top - bottom)),
    }


def _marker(draw, x, y):
    draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=ACCENT, outline=WHITE, width=2)


def _map_trace(pixels, section, bbox, anchors):
    from PIL import Image, ImageDraw

    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    w, h = image.size
    map_points = [
        ((sample.longitude - bbox.west) / bbox.width * w - 0.5,
         (bbox.north - sample.latitude) / bbox.height * h - 0.5)
        for sample in section.samples
    ]
    draw.line(map_points, fill=WHITE, width=6)
    draw.line(map_points, fill=ACCENT, width=3)
    for anchor in anchors:
        x, y = map_points[anchor["sample_index"]]
        if -0.5 <= x < w - 0.5 and -0.5 <= y < h - 0.5:
            _marker(draw, x, y)
            anchor["map_pixel"] = [x, y]
    pixels = np.array(image)
    for anchor in anchors:
        if anchor["map_pixel"] is not None:
            x, y = anchor["map_pixel"]
            label_x = x + 10 if x < w - 38 else x - 25
            label_y = y - 30 if y >= 30 else y + 8
            _text(pixels, anchor["label"], label_x, label_y, 21, color=ACCENT, halo=True)
    return pixels


def _draw_profile(canvas, section, axes, anchors):
    from PIL import Image, ImageDraw

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = axes["rect"]
    _, distance_max = axes["distance_m"]
    low, high = axes["elevation_m"]

    def x(distance):
        return x0 + (x1 - x0) * distance / distance_max

    def y(elevation):
        return y1 - (y1 - y0) * (elevation - low) / (high - low)

    draw.rectangle(axes["rect"], fill=WHITE)
    rows = section.samples
    points = [(x(row.distance_m), y(row.elevation_m) if row.elevation_m is not None else None)
              for row in rows]
    for i, (px, py) in enumerate(points):
        if py is None:
            left = (points[i - 1][0] + px) / 2 if i else x0
            right = (px + points[i + 1][0]) / 2 if i + 1 < len(rows) else px
            draw.rectangle((left, y0, right, y1), fill=GAP)
    for tick in axes["elevation_ticks_m"]:
        draw.line((x0, y(tick), x1, y(tick)), fill=INK if tick == 0 else GRID)
    for tick in axes["distance_ticks_m"]:
        draw.line((x(tick), y0, x(tick), y1), fill=GRID)
    draw.line((x0, y0, x0, y1, x1, y1), fill=INK)
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        if ay is not None and by is not None:
            draw.line((ax, ay, bx, by), fill=ACCENT, width=3)
    for i, (px, py) in enumerate(points):
        if py is not None and (i == 0 or points[i - 1][1] is None):
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=ACCENT)
    for anchor in anchors:
        px, py = points[anchor["sample_index"]]
        anchor["profile_pixel"] = [px, py]
        draw.line((px, y0, px, y1 if py is None else py), fill=ACCENT, width=1)
        if py is not None:
            _marker(draw, px, py)
    canvas = np.array(image)
    for tick in axes["elevation_ticks_m"]:
        _text(canvas, f"{tick:g}", x0 - 12, y(tick) - 11, 18, align="right")
    for tick in axes["distance_ticks_m"]:
        _text(canvas, f"{tick / 1000:g}", x(tick), y1 + 8, 18, align="center")
    for anchor in anchors:
        px, _ = anchor["profile_pixel"]
        _text(canvas, anchor["label"], px, y0 - 27, 22, align="center", color=ACCENT)
    _text(canvas, "WGS 84 geodesic distance from A (km)", (x0 + x1) / 2, y1 + 47, 22, align="center")
    return canvas


def render_section_plate(section: MeasuredSection, path: Path, directory: Path, title: str, certificate, cache):
    grid, values = _read_grid(path, section.grid.source_id, section.grid.vertical_crs)
    if grid != section.grid:
        raise ValueError("The map GeoTIFF differs from the sampled source")
    if directory.exists():
        raise FileExistsError(f"Section output already exists: {directory}")
    if not title.strip() or "\n" in title:
        raise ValueError("Provide a single-line section title")

    a, _, west, _, e, north = grid.transform
    bbox = BBox(west, north + e * grid.height, west + a * grid.width, north)
    left, top, right, bottom = MAP_REGION
    width, height = PLATE_SIZE
    scale = min((right - left) / bbox.width, (bottom - top) / bbox.height)
    size = (max(1, round(bbox.width * scale)), max(1, round(bbox.height * scale)))
    cfg = MapPlateConfig(width=width, height=height, margin=(top, width - right, height - bottom, left),
                         background=PAPER)
    plate = MapPlate(cfg)
    classes = gis.classify_raster(values.data, bins=BREAKS, right=False,
                                  valid_mask=~values.mask)["array"][0]
    labels = ("No raster data", *BANDS)
    style = CategoricalRasterStyle(dict(enumerate(PALETTE)), dict(enumerate(labels)),
                                   units=f"Elevation (m) / {grid.vertical_reference}")
    scene = _publication_scene(classes, style, bbox, size, grid.source_id)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".measured-section-", dir=directory.parent) as staging:
        output = Path(staging) / "section"
        output.mkdir()
        map_path = output / "map.png"
        scene.render(str(map_path), certificate=certificate, cache=cache)
        pixels = load_png_rgba(map_path)
        deviation = _check_frame_palette(pixels, style.palette.values())
        map_rect = _drawn_map_rect(cfg, size)
        indices = [0, len(section.samples) - 1]
        names = ["A", "C"]
        if len(section.samples) > 2:
            indices.insert(1, len(section.samples) // 2)
            names.insert(1, "B")
        anchors = [{"label": label, "sample_index": index, "map_pixel": None}
                   for label, index in zip(names, indices)]
        plate.set_map_region(_map_trace(pixels, section, bbox, anchors), bbox)
        for anchor in anchors:
            if anchor["map_pixel"] is not None:
                x, y = anchor["map_pixel"]
                anchor["map_pixel"] = [x + map_rect[0], y + map_rect[1]]

        axes = _axes(section)
        canvas = _draw_profile(plate.compose(), section, axes, anchors)
        _text(canvas, title, 45, 21, 44)
        _text(canvas, f"{grid.source_id}  /  Regional terrain and bathymetry", 46, 77, 22)
        _text(canvas, "Elevation bands (m)", 1021, 123, 29)
        _text(canvas, grid.vertical_reference, 1021, 160, 20)
        for index, entry in enumerate(style.legend()["entries"]):
            y = 207 + index * 34
            canvas[y:y + 22, 1022:1057] = entry["rgba"]
            _text(canvas, entry["label"], 1069, y - 2, 20)
        _text(canvas, "North up / WGS 84 (lon, lat)", 1021, 531, 21)
        _text(canvas, "A to C  /  Requested transect", 1021, 567, 21, color=ACCENT)
        _text(canvas, "Matching letters mark the same sample", 1021, 603, 19)
        _text(canvas, f"Elevation (m)  /  {grid.vertical_reference}", 45, 765, 27)
        _text(canvas, f"Vertical exaggeration {axes['vertical_exaggeration']:.3g}x  /  "
                      "Gray spans = missing samples", 1454, 773, 21, align="right")

        save_png(output / "section.png", canvas)
        (output / "samples.csv").write_bytes(section.to_csv().encode("utf-8"))
        (output / "samples.json").write_bytes(section.to_json().encode("utf-8"))
        receipt = {
            "title": title, "map_rect": map_rect, "map_bounds": asdict(bbox), "profile": axes,
            "annotations": anchors, "size": PLATE_SIZE, "legend": style.legend(), "breaks_m": BREAKS,
            "map_backend": scene.last_render_backend, "map_render": scene.last_render_metadata,
            "map_palette_max_deviation_levels": deviation,
            "sha256": {file.name: _compute_sha256(file) for file in output.iterdir()},
        }
        (output / "layout.json").write_bytes(canonical_json_bytes(receipt, error_context="Section layout"))
        output.rename(directory)
    return receipt
