"""Render the identical original/revised observations independently with PyQGIS."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from qgis.PyQt.QtCore import QSize, Qt
from qgis.PyQt.QtGui import QColor, QFont, QFontDatabase, QFontMetrics, QImage, QPainter
from qgis.core import (
    Qgis, QgsApplication, QgsCoordinateReferenceSystem, QgsFeature, QgsGeometry,
    QgsMapRendererSequentialJob, QgsMapSettings, QgsMarkerSymbol, QgsPointXY,
    QgsRasterLayer, QgsRectangle, QgsSingleSymbolRenderer, QgsVectorLayer,
)

def clock(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def choose(data, variant, presentation):
    case = data["variants"][variant]
    available = [r for r in data["rasters"]
                 if r["payload"]["source"]["provenance"]["catalog_id"] in case["rasters"]
                 and max(clock(r["observed_at"]), clock(r["valid_from"])) <= presentation]
    valid = [r for r in available if presentation < clock(r["valid_until"])]
    selected = max(valid or available, key=lambda r: clock(r["observed_at"]), default=None)
    state = "observed" if valid else ("held" if selected else "missing")
    start = presentation - timedelta(seconds=data["lookback_seconds"])
    events = [r for r in data["events"] if start < clock(r["observed_at"]) <= presentation
              and clock(r["valid_from"]) <= presentation < clock(r["valid_until"])]
    return selected, state, events


def captions(image, title, lines, legend, state, family):
    # Match the fixed map viewport and furniture margins. Qt handles its own
    # resampling and text rasterization.
    header = 24 + 18 * (len(lines) + 1)
    footer = len(legend) * 18 + 40
    map_height = image.height() - header - footer
    map_width = round(image.width() * map_height / image.height())
    frame = QImage(image.size(), QImage.Format_RGB32)
    frame.fill(QColor("#1b2021"))
    painter = QPainter(frame)
    painter.drawImage((image.width() - map_width) // 2, header,
                      image.scaled(map_width, map_height, Qt.IgnoreAspectRatio, Qt.FastTransformation))
    painter.setRenderHint(QPainter.TextAntialiasing)
    font = QFont(family)
    font.setPixelSize(14)
    painter.setFont(font)
    for i, text in enumerate([title, *lines]):
        color = {"observed": "#a0dfc9", "held": "#ffd58c", "missing": "#ffb2a1"}[state]
        painter.setPen(QColor(color if i == 2 else "#f4f5f1"))
        painter.drawText(12, 25 + i * 18, text)
    font.setPixelSize(12)
    painter.setFont(font)
    metrics = QFontMetrics(font)
    labels = [(row["label"], row["color"]) if isinstance(row, dict) else (row, None)
              for row in legend]
    width = max(metrics.horizontalAdvance(label) for label, _ in labels) + 36
    x, y = image.width() - width - 12, image.height() - (len(labels) * 18 + 16) - 12
    painter.fillRect(x, y, width, len(labels) * 18 + 16, QColor(255, 255, 255, 205))
    painter.setPen(QColor(20, 20, 20))
    for i, (label, color) in enumerate(labels):
        if color:
            painter.fillRect(x + 7, y + 9 + i * 18, 7, 7, QColor(color))
        painter.drawText(x + 18, y + 19 + i * 18, label)
    painter.end()
    return frame


def display_raster(source, destination):
    """Present the shared linear stretch with the same sRGB transfer as MapScene."""
    from osgeo import gdal

    dataset = gdal.Open(str(source))
    alpha = dataset.GetRasterBand(1).GetMaskBand().ReadAsArray()
    linear = dataset.ReadAsArray().astype(np.float64) / 255
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    rgba = gdal.GetDriverByName("GTiff").Create(
        str(destination), dataset.RasterXSize, dataset.RasterYSize, 4, gdal.GDT_Byte,
        options=["COMPRESS=DEFLATE", "PHOTOMETRIC=RGB", "ALPHA=YES"])
    rgba.SetGeoTransform(dataset.GetGeoTransform())
    rgba.SetProjection(dataset.GetProjection())
    for index, band in enumerate(np.rint(srgb * 255).astype(np.uint8), 1):
        rgba.GetRasterBand(index).WriteArray(band)
    rgba.GetRasterBand(4).WriteArray(alpha)
    rgba = None
    layer = QgsRasterLayer(str(destination), destination.stem)
    if not layer.isValid():
        raise RuntimeError(f"QGIS did not load {destination}")
    renderer = layer.renderer()
    renderer.setAlphaBand(4)
    return layer


def render(prepared, forge_dir, output, variant):
    from osgeo import gdal

    gdal.UseExceptions()
    data = json.loads((prepared / "input.json").read_text(encoding="utf-8"))
    reference = json.loads((forge_dir / "time-selection.json").read_text(encoding="utf-8"))["frames"]
    case = data["variants"][variant]
    if len(case["times"]) != len(reference):
        raise ValueError("Unmatched presentation sequence")
    rasters = {}
    app = QgsApplication([], False)
    app.setPrefixPath(os.environ["QGIS_PREFIX_PATH"], True)
    app.initQgis()
    font_path = Path(__file__).resolve().parents[1] / "assets/fonts/NotoSansLatin-subset.ttf"
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    families = QFontDatabase.applicationFontFamilies(font_id)
    if not families:
        raise RuntimeError(f"Qt could not load the shared map font {font_path}")
    out_crs = QgsCoordinateReferenceSystem(data["crs"])
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    context = display_raster(data["context"], output / "context-display.tif")
    summaries = []
    for index, (time_string, expected) in enumerate(zip(case["times"], reference)):
        at = clock(time_string)
        record, state, events = choose(data, variant, at)
        record_id = None if record is None else record["record_id"]
        expected_record = expected["raster_selection"]["record"]
        expected_id = None if expected_record is None else expected_record["record_id"]
        ids = sorted(r["record_id"] for r in events)
        expected_ids = sorted(r["record_id"] for r in expected["vector_window"]["records"])
        if (record_id, state, ids) != (expected_id, expected["raster_selection"]["state"], expected_ids):
            raise ValueError(f"QGIS selection differs at frame {index}")
        layers = [context]
        if record is not None and expected["raster"]["valid_pixels"]:
            if record_id not in rasters:
                path = output / f"{record_id}-rgba.tif"
                rasters[record_id] = display_raster(expected["raster"]["snapshot_path"], path)
            layers.insert(0, rasters[record_id])
        points = QgsVectorLayer(f"Point?crs={data['crs']}", "VIIRS active-fire reports", "memory")
        features = []
        for detection in events:
            xy = detection["payload"]["feature"]["geometry"]["coordinates"]
            feature = QgsFeature()
            feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(*xy)))
            features.append(feature)
        if features:
            provider = points.dataProvider()
            if not provider.addFeatures(features)[0] or points.featureCount() != len(features):
                raise RuntimeError("QGIS did not preserve every selected event")
            points.updateExtents()
            marker = QgsMarkerSymbol.createSimple({"name": "circle", "color": data["point_color"],
                                                   "outline_style": "no"})
            marker.setSizeUnit(Qgis.RenderUnit.Pixels)
            marker.setSize(data["point_width"])
            points.setRenderer(QgsSingleSymbolRenderer(marker))
            layers.insert(0, points)
        settings = QgsMapSettings()
        settings.setDestinationCrs(out_crs)
        settings.setExtent(QgsRectangle(*data["bounds"]))
        settings.setOutputSize(QSize(*data["size"]))
        settings.setLayers(layers)
        settings.setBackgroundColor(QColor("#1b2021"))
        job = QgsMapRendererSequentialJob(settings)
        job.start()
        job.waitForFinished()
        if job.errors():
            raise RuntimeError(str(job.errors()))
        image = job.renderedImage()
        if image.isNull():
            raise RuntimeError("QGIS did not produce an image")
        image.save(str(output / f"map_{index:08d}.png"))
        # The comparator authors selection independently. Shared caption strings
        # keep the final-size comparison on the same timing and wording.
        image = captions(image, data["title"], expected["captions"], data["legend"], state, families[0])
        if not image.save(str(output / f"frame_{index:08d}.png")):
            raise RuntimeError("QGIS did not save the rendered frame")
        summaries.append(dict(index=index, presentation=time_string, record_id=record_id, state=state, events=ids))
    receipt = dict(qgis=Qgis.QGIS_VERSION, gdal=gdal.VersionInfo(), render_seconds=time.perf_counter() - started,
                   frames=summaries)
    encode_started = time.perf_counter()
    subprocess.run([os.environ.get("DIXIE_FFMPEG", "ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-framerate", f"1/{data['screen_seconds_per_frame']}", "-i", str(output / "frame_%08d.png"),
                    "-frames:v", str(len(summaries)), "-c:v", "libx264rgb", "-crf", "0",
                    "-pix_fmt", "rgb24", "-movflags", "+faststart", str(output / "observations.mp4")], check=True)
    receipt["encode_seconds"] = time.perf_counter() - encode_started
    (output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("forge_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("variant", choices=("original", "revised"))
    args = parser.parse_args()
    render(args.prepared, args.forge_dir, args.output, args.variant)
