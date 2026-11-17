"""Compare the same prepared Rhodes data with the QGIS raster calculator."""

import json
import os
import sys
import time
from pathlib import Path
from osgeo import gdal
from qgis.core import (
    QgsApplication,
    Qgis,
    QgsRasterLayer,
    QgsProject,
    QgsPalettedRasterRenderer,
    QgsMapSettings,
    QgsMapRendererSequentialJob,
)
from qgis.analysis import QgsRasterCalculator, QgsRasterCalculatorEntry
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtCore import QSize
import numpy as np

gdal.UseExceptions()
root = Path(sys.argv[1]).resolve()
source = root / "prepared"
output = root / "qgis"
output.mkdir(exist_ok=True)
app = QgsApplication([], False)
app.initQgis()
print(Qgis.QGIS_VERSION, gdal.VersionInfo("--version"), flush=True)
palette = ["#962f24", "#e69757", "#ece0c0", "#60ab92", "#1b6165"]
runs = []
for date in ("2023-08-07", "2023-08-17"):
    for revision, edges in enumerate(
        (
            [-0.5, -0.125, 0.125, 0.5],
            [-0.375, -0.0625, 0.0625, 0.375],
            [-0.25, -0.0625, 0.125, 0.25],
        )
    ):
        started = time.perf_counter()
        project = QgsProject.instance()
        project.clear()
        layers = [
            QgsRasterLayer(str(source / (n + ".tif")), role)
            for n, role in (
                ("2023-07-18", "before"),
                (date, "after"),
                (date + "-mask", "mask"),
            )
        ]
        assert all(layer.isValid() for layer in layers)
        entries = []
        for layer in layers:
            entry = QgsRasterCalculatorEntry()
            entry.ref = layer.name() + "@1"
            entry.raster = layer
            entry.bandNumber = 1
            entries.append(entry)
        before, after, mask = ('"before@1"', '"after@1"', '"mask@1"')
        first = layers[0]
        project.setCrs(first.crs())
        first_expression = f"(({after}-{before})/({after}+{before})) * ({mask}/{mask})"
        stem = f"{date}-{revision}"
        value_file = output / (stem + "-values.tif")
        cls_file = output / (stem + "-classes.tif")

        def calculate(expression, target, inputs):
            calculator = QgsRasterCalculator(
                expression,
                str(target),
                "GTiff",
                first.extent(),
                first.crs(),
                first.width(),
                first.height(),
                inputs,
                project.transformContext(),
            )
            code = calculator.processCalculation()
            assert code == 0, (code, calculator.lastError())

        calculate(first_expression, value_file, entries)
        value = QgsRasterLayer(str(value_file), "value")
        assert value.isValid()
        entry = QgsRasterCalculatorEntry()
        entry.ref = "value@1"
        entry.raster = value
        entry.bandNumber = 1
        expression = "1" + "".join(f' + ("value@1" >= {edge})' for edge in edges)
        calculate(expression, cls_file, [entry])
        classes = QgsRasterLayer(str(cls_file), "Relative NDVI change")
        assert classes.isValid()
        labels = [
            f"< {edges[0]:g}",
            *(f"[{a:g}, {b:g})" for a, b in zip(edges, edges[1:])),
            f">= {edges[-1]:g}",
        ]
        classes.setRenderer(
            QgsPalettedRasterRenderer(
                classes.dataProvider(),
                1,
                [
                    QgsPalettedRasterRenderer.Class(i, QColor(rgb), label)
                    for i, (rgb, label) in enumerate(zip(palette, labels), 1)
                ],
            )
        )
        for layer in [*layers, value, classes]:
            project.addMapLayer(layer)
        project.setFileName(str(output / (stem + ".qgz")))
        assert project.write()
        settings = QgsMapSettings()
        settings.setLayers([classes])
        settings.setDestinationCrs(classes.crs())
        settings.setOutputSize(QSize(1024, 768))
        settings.setExtent(first.extent())
        job = QgsMapRendererSequentialJob(settings)
        job.start()
        job.waitForFinished()
        assert job.renderedImage().save(str(output / (stem + ".png")))
        ds = gdal.Open(str(cls_file))
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray()
        match = array != band.GetNoDataValue()
        arr = np.where(match, array, 0).astype(np.uint8)
        counts = np.bincount(arr.ravel(), minlength=6).tolist()
        np.save(output / (stem + "-classes.npy"), arr)
        runs.append(
            {
                "date": date,
                "revision": revision,
                "bins": edges,
                "seconds": time.perf_counter() - started,
                "counts": counts,
            }
        )
        print(runs[-1], flush=True)
(output / "metrics.json").write_text(
    json.dumps(
        {
            "qgis": Qgis.QGIS_VERSION,
            "gdal": gdal.VersionInfo("--version"),
            "runs": runs,
        },
        indent=2,
    )
)

sys.stdout.flush()
os._exit(0)
