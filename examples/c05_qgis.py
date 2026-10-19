"""C05 independent QGIS/GDAL preparation and saved-project comparison.

Run with QGIS's Python, not Forge3D's interpreter. Both tools consume the shared
raw subsets and the same explicit recipe, grid and display mapping. This file
does not import Forge3D or reuse its prepared arrays to create QGIS outputs.
"""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
import json
import hashlib
import os
from pathlib import Path
import time

import numpy as np
# Set before the first GDAL dataset is opened: comparisons must not write PAM
# sidecars into shared source directories or change their recorded identities.
os.environ["GDAL_PAM_ENABLED"] = "NO"
from osgeo import gdal
gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")
from qgis.core import (QgsApplication, Qgis, QgsProject, QgsRasterLayer,
    QgsCoordinateReferenceSystem, QgsRectangle, QgsMapSettings,
    QgsMapRendererParallelJob, QgsMultiBandColorRenderer)
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QColor


def record(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def memory():
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),
            ('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),
            ('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),
            ('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),
            ('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
    c=Counters(); c.cb=ctypes.sizeof(c)
    kernel=ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype=wintypes.HANDLE
    get_memory=ctypes.WinDLL('psapi').GetProcessMemoryInfo
    get_memory.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    get_memory.restype=wintypes.BOOL
    if not get_memory(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb):
        raise ctypes.WinError()
    return c.PeakWorkingSetSize


def warp(source, grid, method, output):
    ds=gdal.Open(source)
    values=ds.ReadAsArray().astype(np.float64)
    valid=ds.GetRasterBand(1).GetMaskBand().ReadAsArray()!=0
    nodata=ds.GetRasterBand(1).GetNoDataValue()
    valid &= np.isfinite(values)
    if nodata is not None: valid &= values != nodata
    values[~valid]=np.nan
    scratch=gdal.GetDriverByName('MEM').Create('',ds.RasterXSize,ds.RasterYSize,1,gdal.GDT_Float64)
    scratch.SetGeoTransform(ds.GetGeoTransform()); scratch.SetProjection(ds.GetProjection())
    scratch.GetRasterBand(1).WriteArray(values); scratch.GetRasterBand(1).SetNoDataValue(float('nan'))
    gt=grid['geotransform']; w,h=grid['width'],grid['height']
    target=gdal.Warp(str(output),scratch,format='GTiff',outputBounds=[gt[0],gt[3]+h*gt[5],gt[0]+w*gt[1],gt[3]],
        width=w,height=h,dstSRS=grid['crs'],srcNodata=float('nan'),dstNodata=float('nan'),
        outputType=gdal.GDT_Float64,resampleAlg=method,creationOptions=['COMPRESS=DEFLATE'])
    if target is None: raise RuntimeError('GDAL warp failed')
    result=target.ReadAsArray(); target=None
    return result,np.isfinite(result)


def compare(directory):
    started=time.perf_counter()
    recipe=json.loads((directory/'recipe.json').read_text(encoding='utf-8'))
    grid=recipe['target_grid']; out=directory/'qgis'
    out.mkdir(exist_ok=False)
    result={}; arrays={}; masks={}
    for source in recipe['sources']:
        sid=source['source_id']
        if source['kind']=='count':
            # Native count cells remain a separate analysis layer.
            native=gdal.Translate(str(out/f'{sid}-native.tif'),str(source['path']),format='GTiff')
            native_data=native.ReadAsArray(); native_valid=native.GetRasterBand(1).GetMaskBand().ReadAsArray()!=0
            source_ds=gdal.Open(source['path'])
            if not (np.array_equal(native_data,source_ds.ReadAsArray()) and np.array_equal(native_valid,source_ds.GetRasterBand(1).GetMaskBand().ReadAsArray()!=0)):
                raise AssertionError('QGIS count values or masks changed')
            native=None
            result['native_count_values_and_masks_exact']=True
            data,valid=warp(source['path'],grid,'near',out/f'{sid}-display.tif')
        else:
            data,valid=warp(source['path'],grid,'bilinear' if source['kind']=='elevation' else 'near',out/f'{sid}.tif')
        arrays[sid],masks[sid]=data,valid
    preparation_seconds=time.perf_counter()-started
    texture=np.full((grid['height'],grid['width'],4),[193,192,179,255],np.uint8)
    display_valid=np.zeros(texture.shape[:2],dtype=bool)
    # JSON canonicalization sorts style keys; layer order is the recipe source
    # order, as in PreparedScene.viewer_assets, not dictionary key order.
    for source in recipe['sources']:
        sid=source['source_id']
        if sid not in recipe['display_styles']: continue
        style=recipe['display_styles'][sid]
        values,valid=arrays[sid],masks[sid]
        if 'classes' in style:
            for key,color in style['classes'].items():
                selected=valid & (values==int(key)) & (color[3]!=0)
                texture[selected]=color
                display_valid |= selected
        else:
            low,high=style['range']; colors=np.asarray(style['colors'],dtype=np.float64)
            mapped=np.where(valid,values,low)
            scale=style.get('scale','linear')
            if scale=='log1p':
                if low<=-1 or np.any(mapped<=-1): raise ValueError('Invalid log1p domain')
                mapped,low,high=np.log1p(mapped),np.log1p(low),np.log1p(high)
            elif scale!='linear': raise ValueError('Unknown display scale')
            t=np.clip((mapped-low)/(high-low),0,1)[...,None]
            rgba=np.rint(colors[0]+t*(colors[1]-colors[0])).astype(np.uint8)
            selected=valid & (rgba[...,3]!=0)
            texture[selected]=rgba[selected]
            display_valid |= selected
    rr,cc=np.indices(texture.shape[:2]); unknown=~display_valid
    texture[unknown & ((rr+cc)%8==0)]=[145,145,138,255]
    terrain_valid=masks['terrain']; texture[~terrain_valid]=[72,111,137,255]
    texture[(~terrain_valid) & ((rr-cc)%8==0)]=[192,212,220,255]
    ds=gdal.GetDriverByName('GTiff').Create(str(out/'display.tif'),grid['width'],grid['height'],4,gdal.GDT_Byte)
    ds.SetGeoTransform(grid['geotransform']); ds.SetProjection(QgsCoordinateReferenceSystem(grid['crs']).toWkt())
    for b in range(4): ds.GetRasterBand(b+1).WriteArray(texture[...,b])
    ds=None
    # Compare only after both outputs have been independently produced.
    manifest=json.loads((directory/'prepared/preparation.json').read_text(encoding='utf-8'))
    differences={}
    for layer in manifest['layers']:
        sid=layer['source_id']
        if layer['kind']=='count': continue
        expected=np.load(directory/'prepared'/layer['files']['data']['path'],allow_pickle=False)
        expected_valid=np.load(directory/'prepared'/layer['files']['valid_mask']['path'],allow_pickle=False)
        actual=arrays[sid]; common=expected_valid & masks[sid]
        differences[sid]={'valid_masks_exact':bool(np.array_equal(expected_valid,masks[sid])),
            'common_valid_cells':int(common.sum()),'max_absolute_value_difference':float(np.abs(expected[common]-actual[common]).max()),
            'values_exact':bool(np.array_equal(expected[common],actual[common]))}
        if not differences[sid]['valid_masks_exact']: raise AssertionError('Tool coverage masks differ')
        if layer['kind']=='categorical' and not differences[sid]['values_exact']: raise AssertionError('Tool class IDs differ')
    f3d_display=gdal.Open(str(directory/'display.tif')).ReadAsArray()
    result['display_pixels_exact']=bool(np.array_equal(np.moveaxis(texture,-1,0),f3d_display))
    if not result['display_pixels_exact']: raise AssertionError('Display mappings differ')
    # A usable, editable QGIS project preserves the raw analysis layers as well
    # as the explicitly styled display. No external account/profile is used.
    project=QgsProject.instance(); project.clear()
    project.setCrs(QgsCoordinateReferenceSystem(grid['crs']))
    layers=[]
    for name,file in [('Prepared display','display.tif'),('Measured terrain','terrain.tif')]:
        layer=QgsRasterLayer(str(out/file),name)
        if not layer.isValid(): raise RuntimeError(f'Invalid QGIS provider: {file}')
        project.addMapLayer(layer); layers.append(layer)
    layers[0].setRenderer(QgsMultiBandColorRenderer(layers[0].dataProvider(),1,2,3))
    project.layerTreeRoot().findLayer(layers[1].id()).setItemVisibilityChecked(False)
    gt=grid['geotransform']; extent=QgsRectangle(gt[0],gt[3]+grid['height']*gt[5],gt[0]+grid['width']*gt[1],gt[3])
    settings=QgsMapSettings(); settings.setLayers([layers[0]]); settings.setDestinationCrs(project.crs())
    settings.setExtent(extent); settings.setOutputSize(QSize(1280,720)); settings.setBackgroundColor(QColor('#f6f7f7'))
    if any(s['kind']=='count' for s in recipe['sources']):
        layer=QgsRasterLayer(str(out/'population-native.tif'),'CBS published values - original 100 m cells')
        if not layer.isValid(): raise RuntimeError('Invalid QGIS native count layer')
        project.addMapLayer(layer); project.layerTreeRoot().findLayer(layer.id()).setItemVisibilityChecked(False)
        state=QgsRasterLayer(str(out/'population_state.tif'),'CBS state: 0 absent, 1 suppressed/unknown, 2 published')
        if not state.isValid(): raise RuntimeError('Invalid QGIS population state layer')
        project.addMapLayer(state); project.layerTreeRoot().findLayer(state.id()).setItemVisibilityChecked(False)
    project.viewSettings().setDefaultViewExtent(__import__('qgis.core',fromlist=['QgsReferencedRectangle']).QgsReferencedRectangle(extent,project.crs()))
    project.setFileName(str(out/'scene.qgz'))
    if not project.write(): raise RuntimeError('QGIS project save failed')
    render_start=time.perf_counter()
    job=QgsMapRendererParallelJob(settings); job.start(); job.waitForFinished()
    errors=job.errors()
    if errors: raise RuntimeError(str(errors))
    if not job.renderedImage().save(str(out/'registered-plan.png')): raise RuntimeError('QGIS image save failed')
    result.update({'tool':'QGIS '+Qgis.QGIS_VERSION,'gdal':gdal.VersionInfo('--version'),
        'grid':grid,'preparation_seconds':preparation_seconds,'plan_render_seconds':time.perf_counter()-render_start,
        'total_seconds':time.perf_counter()-started,'peak_working_set_bytes':memory(),
        'differences_from_forge3d':differences,'renderer':'QGIS 2D map renderer; no substituted 3D result',
        'full_camera_comparison':'Separate execution record: ../qgis-3d/evidence.json and exit-state.json; this preparation process does not certify it.',
        'implementation_sha256':{str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'gdal_pam_enabled':gdal.GetConfigOption('GDAL_PAM_ENABLED'),
        'independence':'Separately implemented pipeline using GDAL, as does Forge3D/rasterio. Equality is agreement, not independent geographic ground truth.',
        'performance_comparability':'descriptive local runs, no isolation, different renderer scope; no speed or quality superiority claim'})
    record(out/'evidence.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    profile=Path('artifacts/c05-real-data/qgis-profile/standalone').resolve(); profile.mkdir(parents=True,exist_ok=True)
    app=QgsApplication([],False,str(profile)); app.initQgis(); gdal.UseExceptions()
    compare(args.directory.resolve())
    QgsProject.instance().clear(); app.exitQgis()
