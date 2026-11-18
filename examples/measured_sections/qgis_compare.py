"""Independent PyQGIS sampler and matching 2D map/profile workload."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

# Keep the QGIS comparison from writing auxiliary files beside the pinned TIFF.
os.environ["GDAL_PAM_ENABLED"] = "NO"

import numpy as np
from osgeo import gdal
from qgis.PyQt.QtCore import QPointF, QRectF, QSize
from qgis.PyQt.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPen
from qgis.core import (
    Qgis, QgsApplication, QgsCoordinateReferenceSystem,
    QgsCoordinateTransformContext, QgsDistanceArea, QgsMapRendererParallelJob,
    QgsMapSettings, QgsPalettedRasterRenderer, QgsPointXY, QgsRasterLayer,
)

# Fixed display settings shared with the published workload by specification.
SIZE = QSize(936, 624)
INK, ACCENT = QColor("#212c30"), QColor("#8e2b19")
BREAKS = (-2500, -1500, -750, -250, 0, 350, 800)
COLORS = ("#183653", "#2a506d", "#40718c", "#5f96a9", "#a1c9d1",
          "#e4e1ba", "#b8b882", "#878456")
MISSING = QColor("#dde1de")
LABELS = ("Below -2500", "-2500 to -1500", "-1500 to -750", "-750 to -250",
          "-250 to 0", "0 to 350", "350 to 800", "800 and above", "No raster data")


def end_to_end_samples(layer, recipe, crs):
    from bisect import bisect_right

    distance = QgsDistanceArea()
    distance.setSourceCrs(crs, QgsCoordinateTransformContext())
    if not distance.setEllipsoid("EPSG:7030"):
        raise RuntimeError("QGIS could not select the WGS 84 ellipsoid")
    start, end = [QgsPointXY(*recipe[key]) for key in ("start", "end")]
    length = distance.measureLine(start, end)
    intervals = math.ceil(length / recipe["spacing_m"])
    azimuth = distance.bearing(start, end)
    provider = layer.dataProvider()
    raster = gdal.Open(layer.source())
    west, dx, _, north, _, dy = raster.GetGeoTransform()
    # Follow the declared GDAL pixel-edge affine in binary64, even when a
    # provider's normalized-extent arithmetic rounds an exact edge differently.
    east_edges = [west + dx * col for col in range(raster.RasterXSize + 1)]
    south_edges = [-north - dy * row for row in range(raster.RasterYSize + 1)]
    rows = []
    for i in range(intervals + 1):
        d = length * i / intervals
        point = start if i == 0 else end if i == intervals else distance.computeSpheroidProject(start, d, azimuth)
        x, y = point.x(), point.y()
        col = bisect_right(east_edges, x) - 1
        row = bisect_right(south_edges, -y) - 1
        inside = 0 <= col < raster.RasterXSize and 0 <= row < raster.RasterYSize
        elevation = direct = None
        valid = False
        if inside:
            center = QgsPointXY(west + dx * (col + 0.5), north + dy * (row + 0.5))
            elevation, valid = provider.sample(center, 1)
            direct, direct_valid = provider.sample(point, 1)
            direct = direct if direct_valid else None
        rows.append(dict(index=i, longitude=x, latitude=y, distance_m=d,
                         elevation_m=elevation if valid else None,
                         status="valid" if valid else "nodata" if inside else "out_of_bounds",
                         row=row if inside else None, column=col if inside else None,
                         provider_direct_elevation_m=direct))
    return rows


def map_image(path, crs, classified_path):
    # Classify at source resolution before QGIS renders the display layer.
    # Retain QGIS's own viewport resampling for the comparison.
    raster = gdal.Open(str(path))
    heights = raster.ReadAsArray()
    valid = raster.GetRasterBand(1).GetMaskBand().ReadAsArray() != 0
    classes = np.where(valid, np.digitize(heights, BREAKS) + 1, 0).astype(np.uint8)
    classified = gdal.GetDriverByName("GTiff").Create(
        str(classified_path), raster.RasterXSize, raster.RasterYSize, 1, gdal.GDT_Byte)
    classified.SetGeoTransform(raster.GetGeoTransform())
    classified.SetProjection(crs.toWkt())
    classified.GetRasterBand(1).WriteArray(classes)
    classified = None
    layer = QgsRasterLayer(str(classified_path.resolve()), "ETOPO elevation bands", "gdal")
    if not layer.isValid():
        raise RuntimeError("QGIS could not open the classified map grid")
    palette = [QgsPalettedRasterRenderer.Class(i, QColor(color), label)
               for i, (color, label) in enumerate(zip([MISSING, *COLORS], [LABELS[-1], *LABELS[:-1]]))]
    layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, palette))
    settings = QgsMapSettings()
    settings.setLayers([layer])
    settings.setDestinationCrs(crs)
    settings.setExtent(layer.extent())
    settings.setOutputSize(SIZE)
    settings.setBackgroundColor(MISSING)
    job = QgsMapRendererParallelJob(settings)
    job.start()
    job.waitForFinished()
    if job.errors() or job.renderedImage().isNull():
        raise RuntimeError(f"QGIS raster render failed: {job.errors()}")
    return job.renderedImage(), settings


def tick_step(span):
    exponent = 10 ** math.floor(math.log10(span / 5))
    return next(v * exponent for v in (1, 2, 5, 10) if v * exponent >= span / 5)


def plate_image(background, settings, rows, title, font_family):
    plate = QImage(1500, 1130, QImage.Format_ARGB32)
    plate.fill(QColor("#f9f9f5"))
    p = QPainter(plate)
    p.setRenderHint(QPainter.Antialiasing)

    def label(x, y, text, size=20, color=INK, center=False):
        font = QFont(font_family)
        font.setPixelSize(size)
        p.setFont(font)
        p.setPen(color)
        if center:
            x -= p.fontMetrics().horizontalAdvance(text) / 2
        p.drawText(QPointF(x, y), text)

    def line(a, b, color, width=1):
        p.setPen(QPen(color, width))
        p.drawLine(QPointF(*a), QPointF(*b))

    def marker(x, y):
        p.setPen(QPen(QColor("white"), 2))
        p.setBrush(ACCENT)
        p.drawEllipse(QPointF(x, y), 6, 6)

    p.drawImage(45, 114, background)
    label(45, 66, title, 44)
    label(46, 100, "NOAA ETOPO 2022 v1 / 15 arc-sec surface  /  Regional terrain and bathymetry", 22)
    label(1021, 153, "Elevation bands (m)", 29)
    label(1021, 180, "EGM2008 height")
    for i, (name, color) in enumerate(zip(LABELS, [*COLORS, MISSING])):
        top = 207 + i * 34
        p.fillRect(1022, top, 35, 22, QColor(color))
        label(1069, top + 19, name)
    label(1021, 552, "North up / WGS 84 (lon, lat)", 21)
    label(1021, 588, "A to C  /  Requested transect", 21, ACCENT)
    label(1021, 624, "Matching letters mark the same sample", 19)
    label(45, 792, "Elevation (m)  /  EGM2008 height", 27)

    transform = settings.mapToPixel()
    map_points = []
    for row in rows:
        pt = transform.transform(row["longitude"], row["latitude"])
        map_points.append((45 + pt.x() - 0.5, 114 + pt.y() - 0.5))
    p.save()
    p.setClipRect(45, 114, SIZE.width(), SIZE.height())
    for color, width in [(QColor("white"), 6), (ACCENT, 3)]:
        for first, last in zip(map_points, map_points[1:]):
            line(first, last, color, width)
    anchors = dict(zip((0, len(rows) // 2, len(rows) - 1), "ABC")) if len(rows) > 2 else {0: "A", 1: "C"}
    for index, name in anchors.items():
        x, y = map_points[index]
        marker(x, y)
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            label(x + 10 + dx, y - 8 + dy, name, 21, QColor("white"))
        label(x + 10, y - 8, name, 21, ACCENT)
    p.restore()

    e = [row["elevation_m"] for row in rows if row["elevation_m"] is not None]
    low, high = min([0.0, *e]), max([0.0, *e])
    vertical_step = tick_step(high - low) if high != low else 1.0
    low = (math.floor(low / vertical_step) - 1) * vertical_step
    high = (math.ceil(high / vertical_step) + 1) * vertical_step
    distance_step = tick_step(rows[-1]["distance_m"])
    length = math.ceil(rows[-1]["distance_m"] / distance_step) * distance_step
    left, top, right, bottom = (134, 842, 1454, 1010)

    def px(d):
        return left + (right - left) * d / length

    def py(z):
        return bottom - (bottom - top) * (z - low) / (high - low)

    exaggeration = (bottom - top) * length / ((right - left) * (high - low))
    label(916, 794, f"Vertical exaggeration {exaggeration:.3g}x  /  Gray spans = missing samples", 19)
    p.fillRect(left, top, right - left, bottom - top, QColor("white"))
    points = [(px(row["distance_m"]), py(row["elevation_m"]) if row["elevation_m"] is not None else None)
              for row in rows]
    for i, (x, y) in enumerate(points):
        if y is None:
            x0 = (x + points[i - 1][0]) / 2 if i else left
            x1 = (x + points[i + 1][0]) / 2 if i + 1 < len(rows) else x
            p.fillRect(QRectF(x0, top, x1 - x0, bottom - top), QColor("#e5e8e7"))
    for i in range(round((high - low) / vertical_step) + 1):
        z = low + i * vertical_step
        y = py(z)
        line((left, y), (right, y), INK if z == 0 else QColor("#ced8d8"))
        label(left - 42, y + 6, f"{z:g}", 18, center=True)
    for i in range(round(length / distance_step) + 1):
        d = i * distance_step
        x = px(d)
        line((x, top), (x, bottom), QColor("#ced8d8"))
        label(x, bottom + 26, f"{d / 1000:g}", 18, center=True)
    line((left, top), (left, bottom), INK)
    line((left, bottom), (right, bottom), INK)
    for a, b in zip(points, points[1:]):
        if a[1] is not None and b[1] is not None:
            line(a, b, ACCENT, 3)
    for index, name in anchors.items():
        x, y = points[index]
        label(x, top - 7, name, 22, ACCENT, center=True)
        line((x, top), (x, bottom if y is None else y), ACCENT)
        if y is not None:
            marker(x, y)
    label((left + right) / 2, bottom + 68, "WGS 84 geodesic distance from A (km)", 22, center=True)
    p.end()
    return plate, {"distance_m": [0, length], "elevation_m": [low, high],
                   "vertical_exaggeration": exaggeration}


def main():
    from monterey import peak_memory, sha, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("subset", "recipe", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    setup_start = time.perf_counter()
    app = QgsApplication([], False, str((args.output / "qgis-profile").resolve()))
    app.initQgis()
    gdal.UseExceptions()
    font_path = Path(__file__).resolve().parents[2] / "python/forge3d/data/fonts/NotoSansLatin-subset.ttf"
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    families = QFontDatabase.applicationFontFamilies(font_id)
    if not families:
        raise RuntimeError("QGIS could not load the comparison font")
    source = args.subset / "surface.tif"
    layer = QgsRasterLayer(str(source.resolve()), "ETOPO 2022 surface", "gdal")
    if not layer.isValid():
        raise RuntimeError("QGIS GDAL could not open the source TIFF")
    crs = QgsCoordinateReferenceSystem("EPSG:4326")
    sampling_start = time.perf_counter()
    rows = end_to_end_samples(layer, recipe, crs)
    drawing_start = time.perf_counter()
    background, settings = map_image(source, crs, args.output / "display.tif")
    plate, axes = plate_image(background, settings, rows, recipe["title"], families[0])
    if not (background.save(str(args.output / "map.png")) and plate.save(str(args.output / "section.png"))):
        raise RuntimeError("QGIS PNG export failed")
    end = time.perf_counter()
    with (args.output / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / "recipe.json", recipe)
    write_json(args.output / "qgis.json", {
        "qgis": Qgis.QGIS_VERSION, "gdal": gdal.VersionInfo("--version"), "source_sha256": sha(source),
        "setup_seconds": sampling_start - setup_start, "sample_seconds": drawing_start - sampling_start,
        "render_and_export_seconds": end - drawing_start, "peak_process_working_set_bytes": peak_memory(),
        "axes": axes, "plate_size": [1500, 1130], "map_size": [936, 624],
    })
    del layer
    app.exitQgis()


if __name__ == "__main__":
    main()
