"""Independent QGIS layout comparator for an executed C06 page receipt.

Run in QGIS Python: c06_qgis_publication.py RECEIPT_JSON OUTPUT_DIRECTORY.
Reads and independently classifies the original GeoTIFF, never a Forge3D PNG.
The saved project contains the layout and its native 2D/3D map items.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from osgeo import gdal
from qgis.core import (Qgis, QgsApplication, QgsProject, QgsRasterLayer,
                       QgsPalettedRasterRenderer, QgsCoordinateReferenceSystem,
                       QgsPrintLayout, QgsLayoutItemMap, QgsLayoutItemLabel,
                       QgsLayoutPoint, QgsLayoutSize, QgsRectangle, QgsLayoutExporter,
                       QgsLayoutItemMapOverview, QgsFillSymbol, QgsLayoutItemScaleBar,
                       QgsUnitTypes, QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY, QgsVector3D)
from qgis._3d import QgsLayoutItem3DMap, Qgs3DMapSettings, QgsFlatTerrainSettings, QgsCameraPose
from qgis.PyQt.QtGui import QColor, QFont


def peak_process_memory_bytes():
    """Windows process high-water working set, including QGIS/native allocations."""
    if platform.system() != "Windows":
        return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize


def run(receipt_path, output):
    start = time.perf_counter()
    receipt = json.loads(receipt_path.read_text())
    perspective = receipt["footprint"]["kind"] == "flat_terrain_perspective_ground_intersection"
    if receipt["footprint"]["kind"] not in {"flat_screen_viewport", "flat_terrain_perspective_ground_intersection"}:
        raise ValueError("Unsupported comparison camera")
    source = Path(receipt["source"]["path"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != receipt["source"]["sha256"]:
        raise ValueError("Source hash differs from the Forge3D workload")
    output.mkdir(parents=True, exist_ok=True)
    dataset = gdal.Open(str(source))
    band = dataset.GetRasterBand(1)
    values = band.ReadAsArray()
    valid = (band.GetMaskBand().ReadAsArray() != 0) & np.isfinite(values)
    # Independent NumPy classification from the retained actual interval table.
    table = receipt["classification"][1:]
    breaks = [row["right"] for row in table if row["right"] is not None]
    right = bool(table[0]["right_inclusive"])
    classes = np.where(valid, np.searchsorted(breaks, values, side="left" if right else "right") + 1, 0)
    raster_path = output / "classes.tif"
    classified = gdal.GetDriverByName("GTiff").Create(str(raster_path), dataset.RasterXSize,
                                                    dataset.RasterYSize, 1, gdal.GDT_UInt32)
    classified.SetGeoTransform(dataset.GetGeoTransform())
    classified.SetProjection(dataset.GetProjection())
    classified.GetRasterBand(1).WriteArray(classes)
    classified.FlushCache()
    classified = None
    layer = QgsRasterLayer(str(raster_path), receipt["source"]["name"])
    if not layer.isValid():
        raise RuntimeError("QGIS could not load the independently classified raster")
    entries = receipt["legend"]["entries"]
    layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, [
        QgsPalettedRasterRenderer.Class(e["class_id"], QColor(*e["rgba"]), e["label"]) for e in entries]))
    project = QgsProject.instance()
    project.clear()
    project.addMapLayer(layer)
    project.setCrs(layer.crs())
    project.setEllipsoid("WGS84")
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName("C06 matched plan publication")
    width, height = receipt["output"]["size"]
    dpi = 150  # Existing MapPlateConfig default, declared for this comparison.
    mm = 25.4 / dpi
    layout.renderContext().setDpi(dpi)
    layout.pageCollection().pages()[0].setPageSize(QgsLayoutSize(width * mm, height * mm))

    def label(text, rect, pixels):
        item = QgsLayoutItemLabel(layout)
        item.setText(text)
        font = QFont("Noto Sans")
        font.setPointSizeF(pixels * 72 / dpi)
        item.setFont(font)
        layout.addLayoutItem(item)
        x, y, w, h = rect
        item.attemptMove(QgsLayoutPoint(x * mm, y * mm))
        item.attemptResize(QgsLayoutSize(w * mm, h * mm))
        return item

    def map_item(rect, bounds):
        x0, y0, x1, y1 = rect
        item = QgsLayoutItemMap(layout)
        layout.addLayoutItem(item)
        item.attemptMove(QgsLayoutPoint(x0 * mm, y0 * mm))
        item.attemptResize(QgsLayoutSize((x1-x0) * mm, (y1-y0) * mm))
        item.setCrs(QgsCoordinateReferenceSystem(receipt["source"]["grid"]["crs"]))
        item.setLayers([layer])
        item.setKeepLayerSet(True)
        item.zoomToExtent(QgsRectangle(*bounds))
        return item

    footprint = np.asarray(receipt["footprint"]["coordinates"])
    if perspective:
        b = receipt["map_bounds"]
        extent = QgsRectangle(b["west"], b["south"], b["east"], b["north"])
        settings3d = Qgs3DMapSettings()
        settings3d.setCrs(layer.crs())
        settings3d.setExtent(extent)
        settings3d.setLayers([layer])
        terrain = QgsFlatTerrainSettings()
        terrain.setElevationOffset(-0.5)  # Same native flat-plane [0,1] height domain.
        settings3d.setTerrainSettings(terrain)
        settings3d.setTerrainShadingEnabled(False)
        settings3d.setBackgroundColor(QColor(255,255,255))
        camera = receipt["camera"]
        settings3d.setFieldOfView(camera["fov_deg"])
        pose = QgsCameraPose()
        center = np.array([(b["west"]+b["east"])/2,(b["south"]+b["north"])/2,0])
        target = center+np.array(camera["target"])
        pose.setCenterPoint(settings3d.mapToWorldCoordinates(QgsVector3D(*target)))
        pose.setDistanceFromCenterPoint(camera["distance"])
        pose.setPitchAngle(camera["elevation_deg"])
        # QGIS 3.44 uses Z-up: Rz(heading) * Rx(pitch) applied to +Z.
        # https://github.com/qgis/QGIS/blob/final-3_44_0/src/3d/qgs3dutils.cpp
        pose.setHeadingAngle(90+camera["azimuth_deg"])
        main = QgsLayoutItem3DMap(layout)
        layout.addLayoutItem(main)
        x0,y0,x1,y1 = receipt["map_rect"]
        main.attemptMove(QgsLayoutPoint(x0*mm,y0*mm))
        main.attemptResize(QgsLayoutSize((x1-x0)*mm,(y1-y0)*mm))
        main.setMapSettings(settings3d)
        main.setCameraPose(pose)
        # Independent pinhole intersection in the GIS geographic coordinate frame.
        theta,phi = np.deg2rad([camera["elevation_deg"],camera["azimuth_deg"]])
        eye = target+camera["distance"]*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
        forward=target-eye;forward/=np.linalg.norm(forward)
        side=np.cross(forward,[0,0,1]);side/=np.linalg.norm(side)
        up=np.cross(side,forward)
        corners=[]
        for x,y in [(-1,1),(1,1),(1,-1),(-1,-1)]:
            ray=forward+np.tan(np.deg2rad(camera["fov_deg"]/2))*(x*(x1-x0)/(y1-y0)*side+y*up)
            if ray[2]>=0: raise ValueError("QGIS comparator camera ray misses ground")
            point=eye+(-0.5-eye[2])/ray[2]*ray
            corners.append(QgsPointXY(float(point[0]),float(point[1])))
        outline=QgsVectorLayer(f"Polygon?crs={b['crs']}","Independent camera footprint","memory")
        feature=QgsFeature();feature.setGeometry(QgsGeometry.fromPolygonXY([corners+corners[:1]]))
        outline.dataProvider().addFeatures([feature])
        outline.renderer().setSymbol(QgsFillSymbol.createSimple({"color":"0,0,0,0","outline_color":"18,33,43,255","outline_width":str(3*mm)}))
        project.addMapLayer(outline)
    else:
        main = map_item(receipt["map_rect"], [footprint[:, 0].min(), footprint[:, 1].min(),
                                            footprint[:, 0].max(), footprint[:, 1].max()])
    grid = receipt["source"]["grid"]
    locator = map_item(receipt["locator_rect"], [grid[k] for k in ("west", "south", "east", "north")])
    if perspective:
        locator.setLayers([outline,layer])
    else:
        overview = QgsLayoutItemMapOverview("Actual linked main extent", locator)
        locator.overviews().addOverview(overview)
        overview.setLinkedMap(main)
        overview.setFrameSymbol(QgsFillSymbol.createSimple({"color": "0,0,0,0", "outline_color": "18,33,43,255",
                                                           "outline_width": str(3 * mm)}))
        bar=QgsLayoutItemScaleBar(layout)
        layout.addLayoutItem(bar)
        bar.setStyle("Single Box")
        bar.setLinkedMap(main)
        bar.setUnits(QgsUnitTypes.DistanceMeters)
        number,unit=receipt["scale"]["bar_label"].split()
        bar.setUnitsPerSegment(float(number)*{"m":1,"km":1000}[unit])
        bar.setNumberOfSegments(1);bar.setNumberOfSegmentsLeft(0)
        bar.setUnitLabel("m");bar.setHeight(10*mm)
        bx,by,_,_=receipt["scale"]["bar_rect"]
        bar.attemptMove(QgsLayoutPoint(bx*mm,by*mm))
    label(f"{receipt['title']} / {receipt['source']['year']}", (30, 10, width - 60, 70), 38)
    lx, ly, _, _ = receipt["locator_rect"]
    label(receipt["legend"]["units"], (lx, 100, width-lx-20, 45), 25)
    # Same information hierarchy, explicit keys, units, locator and source caption.
    for index, entry in enumerate(entries):
        y = 155 + index * 43
        key = label("", (lx, y, 30, 24), 22)
        key.setBackgroundEnabled(True)
        key.setBackgroundColor(QColor(*entry["rgba"]))
        label(entry["label"], (lx + 45, y - 3, width-lx-65, 40), 22)
    label("Locator / grid north up", (lx, ly-40, width-lx-20, 35), 22)
    label(receipt["caption"], (30, height-150, width-60, 145), 20)
    for inset in receipt.get("insets", []):
        b = inset["bounds"]
        map_item(inset["map_rect"], [b[k] for k in ("west", "south", "east", "north")])
        x0, y0, x1, _ = inset["rect"]
        label(inset["title"], (x0, y0, x1-x0, 35), 22)
    project.layoutManager().addLayout(layout)
    project.setFileName(str(output / "publication.qgz"))
    if not project.write():
        raise RuntimeError("QGIS project save failed")
    settings = QgsLayoutExporter.ImageExportSettings()
    settings.dpi = dpi
    result = QgsLayoutExporter(layout).exportToImage(str(output / "publication.png"), settings)
    if result != QgsLayoutExporter.Success:
        raise RuntimeError(f"QGIS layout export failed: {result}")
    evidence = {"tool": "QGIS " + Qgis.QGIS_VERSION, "python": platform.python_version(),
                "source_sha256": receipt["source"]["sha256"], "input_receipt": str(receipt_path.resolve()),
                "requested_pixels": [width, height], "elapsed_seconds": time.perf_counter()-start,
                "peak_process_working_set_bytes": peak_process_memory_bytes(),
                "classification_counts": np.bincount(classes.ravel()).tolist(),
                "renderer": "QGIS native 3D layout item" if perspective else "QGIS 2D map renderer",
                "camera_ground_footprint": [[p.x(),p.y()] for p in corners] if perspective else None,
                "differences": ["Independent QGIS terrain raster sampling and font rendering",
                                "QGIS font shaping differs; no human setup-effort comparison",
                                "No competitive superiority claim"],
                "final_size_inspection": "not performed by this script"}
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf8")
    return evidence


def run_sequence(manifest_path, output):
    """Render each declared camera independently in QGIS with editable layouts."""
    manifest = json.loads(manifest_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError("QGIS sequence comparison requires a new or empty directory")
    records = []
    start = time.perf_counter()
    for frame in manifest["frames"]:
        receipt = manifest_path.parent / "frames" / f"{frame['page_id']}.json"
        record = run(receipt, output / frame["page_id"])
        records.append({"page_id":frame["page_id"], "camera":frame["camera"], **record})
    result = {"frames":records,"elapsed_seconds":time.perf_counter()-start,
              "manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "human_effort":None,"human_expertise":None,
              "human_comparison_status":"Requires a separately observed human workload"}
    (output / "sequence-evidence.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sequence", action="store_true", help="Input is publication-flyover.json")
    args = parser.parse_args()
    app = QgsApplication([], False)
    app.initQgis()
    if args.sequence:
        run_sequence(args.receipt, args.output)
    else:
        run(args.receipt, args.output)
    app.exitQgis()
