"""Actual QGIS 3.44 3D capture using its native Save as Image action.

Run from QGIS --code (see the real-data guide), with C05_SCENE_DIR set.
No optional PyQt3D bindings, screen grabs, replacement mesh or CPU images.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
os.environ['GDAL_PAM_ENABLED']='NO'
from osgeo import gdal
gdal.SetConfigOption('GDAL_PAM_ENABLED','NO')
from qgis.core import (QgsProject,QgsRasterLayer,QgsCoordinateReferenceSystem,
    QgsRectangle,QgsVector3D,QgsMultiBandColorRenderer,Qgis)
from qgis._3d import QgsDemTerrainSettings,QgsCameraController,Qgs3DMapScene,QgsCameraPose
from qgis.PyQt import sip
from qgis.PyQt.QtCore import QObject,QTimer,QEvent,Qt,QFileSystemWatcher,QSize
from qgis.PyQt.QtGui import QImage,QColor
from qgis.PyQt.QtWidgets import QApplication,QAction,QFileDialog
from qgis.utils import iface
from c05_qgis import memory


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


class Capture(QObject):
    def __init__(self,directory,probe=False):
        super().__init__()
        self.directory=directory
        self.output=directory/os.environ.get('C05_QGIS_OUTPUT_NAME','qgis-3d-probe' if probe else 'qgis-3d')
        if not self.output.resolve().is_relative_to(directory):
            raise ValueError('Capture output must stay inside the demonstration directory')
        self.output.mkdir(exist_ok=False)
        self.frames=self.output/'frames'; self.frames.mkdir()
        self.path=json.loads((directory/'camera-path.json').read_text(encoding='utf-8'))
        self.samples=(self.path['frames'][:2] if os.environ.get('C05_QGIS_PROBE')=='transition'
                      else self.path['frames'][:1] if probe else self.path['frames'])
        self.index=0; self.pending=None; self.capture_scheduled=False
        self.camera_changed=False; self.hashes=[]; self.started=time.perf_counter()
        self.camera_records=[]
        self.watcher=QFileSystemWatcher([str(self.frames)])
        self.watcher.directoryChanged.connect(self.file_changed)
        self.watcher.fileChanged.connect(self.file_changed)
        QApplication.setAttribute(Qt.AA_DontUseNativeDialogs,True)
        QApplication.instance().installEventFilter(self)
        self.setup()

    def journal(self,message):
        with (self.output/'progress.log').open('a',encoding='utf-8') as stream:
            stream.write(message+'\n')

    def setup(self):
        project=QgsProject.instance()
        if not project.read(str(self.directory/'qgis/scene.qgz')):
            raise RuntimeError('QGIS project did not load')
        grid=self.path['grid']; gt=grid['geotransform']
        extent=QgsRectangle(gt[0],gt[3]+grid['height']*gt[5],gt[0]+grid['width']*gt[1],gt[3])
        self.layers=list(project.mapLayers().values())
        self.display=QgsRasterLayer(str(self.directory/'qgis/display.tif'),'Measured display')
        self.display.setRenderer(QgsMultiBandColorRenderer(self.display.dataProvider(),1,2,3))
        project.addMapLayer(self.display)
        ds=gdal.Open(str(self.directory/'qgis/terrain.tif'))
        raw=ds.ReadAsArray(); values=np.where(np.isfinite(raw),raw,0).astype(np.float32)
        low,high=values.min(),values.max()
        self.target=QgsVector3D(gt[0]+grid['width']*gt[1]/2,gt[3]+grid['height']*gt[5]/2,
                              float(low)+float(np.float32((high-low)*np.float32(.5))))
        render_dem=self.output/'render-terrain.tif'
        target=gdal.GetDriverByName('GTiff').Create(str(render_dem),grid['width'],grid['height'],1,gdal.GDT_Float32)
        target.SetGeoTransform(gt); target.SetProjection(ds.GetProjection()); target.GetRasterBand(1).WriteArray(values); target=None
        self.dem=QgsRasterLayer(str(render_dem),'Terrain - declared display void fill 0 m')
        project.addMapLayer(self.dem)
        for layer in project.mapLayers().values():
            project.layerTreeRoot().findLayer(layer.id()).setItemVisibilityChecked(layer.id()==self.display.id())
        iface.mapCanvas().setDestinationCrs(QgsCoordinateReferenceSystem(grid['crs']))
        iface.mapCanvas().setExtent(extent)
        self.canvas=iface.createNewMapCanvas3D('C05 measured scene')
        self.settings=self.canvas.mapSettings()
        self.settings.setCrs(QgsCoordinateReferenceSystem(grid['crs']))
        self.settings.setExtent(extent)
        self.settings.setLayers([self.display])
        self.settings.setBackgroundColor(QColor('#f6f7f7'))
        self.terrain=QgsDemTerrainSettings(); self.terrain.setLayer(self.dem)
        self.terrain.setResolution(grid['width'])
        self.settings.setTerrainSettings(self.terrain)
        self.settings.setTerrainVerticalScale(1)
        self.settings.setTerrainShadingEnabled(True)
        self.settings.setMaxTerrainGroundError(gt[1])
        self.settings.setLayers([self.display])
        self.controller=sip.cast(self.canvas.cameraController(),QgsCameraController)
        self.scene=sip.cast(self.canvas.scene(),Qgs3DMapScene)
        actions={a for w in QApplication.topLevelWidgets() for a in w.findChildren(QAction)}
        self.save_action=next(a for a in actions if a.text().startswith('Save as Image'))
        self.widget=self.save_action.parent().parent()
        # QGIS 3.44's canvas widget layout: toolbar, editing toolbar, message
        # bar, then the horizontal canvas/navigation/debug layout. Fix the
        # embedding QWidget, whose layout otherwise overrides QWindow.resize.
        self.container=self.widget.layout().itemAt(3).layout().itemAt(0).widget()
        self.container.setFixedSize(QSize(1280,720))
        self.window=self.widget.window()
        self.window.resize(self.window.width()+1280-self.canvas.width(),self.window.height()+720-self.canvas.height())
        self.window.show()
        self.canvas.show()
        self.controller.cameraChanged.connect(self.on_camera)
        self.scene.sceneStateChanged.connect(self.maybe_capture)
        self.scene.totalPendingJobsCountChanged.connect(self.maybe_capture)
        self.render_start=time.perf_counter()
        self.journal('3D canvas, DEM, display and native Save as Image action configured')
        QTimer.singleShot(0,self.next_frame)

    def eventFilter(self,obj,event):
        if event.type()==QEvent.Show and isinstance(obj,QFileDialog) and self.pending is not None:
            def accept():
                try:
                    obj.setOption(QFileDialog.DontConfirmOverwrite,True)
                    filters=[f for f in obj.nameFilters() if '*.png' in f.lower()]
                    if not filters: raise RuntimeError('No PNG export filter')
                    obj.selectNameFilter(filters[0]); obj.selectFile(str(self.pending)); obj.accept()
                except Exception: self.fail()
            QTimer.singleShot(0,accept)
        return False

    def next_frame(self):
        try:
            if self.index==len(self.samples):
                self.finish(); return
            sample=self.samples[self.index]
            self.pending=self.frames/f"frame_{sample['frame']:05d}.png"
            self.camera_changed=False; self.capture_scheduled=False
            self.settings.setFieldOfView(sample['fov_deg'])
            # QGIS Z-up heading rotates a southward eye vector about +Z.
            # heading=90-phi and pitch=theta reproduce the recorded F3D eye.
            self.controller.setLookingAtMapPoint(self.target,sample['radius'],sample['theta_deg'],90-sample['phi_deg'])
            pose=self.controller.cameraPose()
            for actual,expected in [(pose.pitchAngle(),sample['theta_deg']),
                (pose.headingAngle(),90-sample['phi_deg']),(pose.distanceFromCenterPoint(),sample['radius'])]:
                if np.float32(actual)!=np.float32(expected):
                    raise AssertionError('QGIS camera did not apply the recorded float32 pose')
            target=self.controller.lookingAtMapPoint()
            self.camera_records.append({'frame':sample['frame'],'map_target':[target.x(),target.y(),target.z()],
                'pitch':pose.pitchAngle(),'heading':pose.headingAngle(),'distance':pose.distanceFromCenterPoint(),
                'fov':self.settings.fieldOfView()})
            self.journal(f'Camera {self.index} applied')
            QTimer.singleShot(0,self.maybe_capture)
            if self.index==0:
                self.journal(f'First camera applied; exposed={self.canvas.isExposed()}, visible={self.canvas.isVisible()}, state={int(self.scene.sceneState())}, jobs={self.scene.totalPendingJobsCount()}')
        except Exception: self.fail()

    def on_camera(self):
        self.camera_changed=True
        if self.index==0: self.journal('First cameraChanged signal received')
        self.maybe_capture()

    def maybe_capture(self,*args):
        if self.pending is None or not self.camera_changed or self.capture_scheduled:
            return
        if self.scene.totalPendingJobsCount()!=0 or self.scene.sceneState()!=Qgs3DMapScene.Ready:
            return
        self.capture_scheduled=True
        QTimer.singleShot(0,self.export)

    def export(self):
        try:
            if self.scene.totalPendingJobsCount()!=0 or self.scene.sceneState()!=Qgs3DMapScene.Ready:
                self.capture_scheduled=False; return
            self.journal(f'Exporting frame {self.index}')
            self.save_action.trigger()
        except Exception: self.fail()

    def file_changed(self,*args):
        try:
            if self.pending is None or not self.pending.exists(): return
            path=self.pending
            if str(path) not in self.watcher.files(): self.watcher.addPath(str(path))
            raw=path.read_bytes()
            # Format-defined PNG end marker: do not accept a partially written image.
            if not raw.endswith(b'\x00\x00\x00\x00IEND\xaeB`\x82'): return
            image=QImage(str(path))
            if image.isNull(): raise RuntimeError('QGIS exported an undecodable frame')
            if (image.width(),image.height())!=(1280,720):
                raise RuntimeError(f'QGIS frame is {image.width()}x{image.height()}, expected 1280x720')
            self.hashes.append(hashlib.sha256(raw).hexdigest())
            self.pending=None; self.index+=1
            self.journal(f'Captured {self.index}/{len(self.samples)}')
            QTimer.singleShot(0,self.next_frame)
        except Exception: self.fail()

    def finish(self):
        if len(set(self.hashes))!=len(self.hashes): raise AssertionError('Repeated QGIS camera image')
        write(self.output/'evidence.json',{'tool':'QGIS '+Qgis.QGIS_VERSION,'frames':len(self.hashes),
            'implementation_sha256':{str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'input_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                            [self.directory/'recipe.json',self.directory/'camera-path.json',self.directory/'qgis/display.tif',self.directory/'qgis/terrain.tif']},
            'gdal_pam_enabled':gdal.GetConfigOption('GDAL_PAM_ENABLED'),
            'resolution':[1280,720],'fps':self.path['fps'],'frame_sha256':self.hashes,
            'camera_poses':self.camera_records,
            'render_seconds':time.perf_counter()-self.render_start,'total_seconds':time.perf_counter()-self.started,
            'peak_working_set_bytes':memory(),'terrain_vertical_scale':1,'target_map_xyz':[self.target.x(),self.target.y(),self.target.z()],
            'camera_mapping':'QGIS pitch=theta; heading=90-phi; target height includes source minimum omitted by F3D local height origin',
            'viewport':[self.canvas.width(),self.canvas.height()],'device_pixel_ratio':self.canvas.devicePixelRatio(),
            'capture':'QGIS native Save as Image; pending scene jobs=0 and scene Ready for every request',
            'lighting':'QGIS terrain shading enabled with default lights; no exact material/shader equivalence claim',
            'terrain_mesh_resolution':self.terrain.resolution(),'max_ground_error_m':self.settings.maxTerrainGroundError(),
            'stability':'Application exit status is recorded separately; frame completion does not certify shutdown.'})
        metadata=self.directory.parent.parent/'qgis-opengl.json'
        if not metadata.exists():
            # Diagnostic of the app's actual global share context, separate
            # from the already observed native frame exports.
            try:
                import ctypes
                from qgis.PyQt.QtGui import QOpenGLContext,QOffscreenSurface
                context=QOpenGLContext.globalShareContext()
                if context is None: raise RuntimeError('No global share context exposed')
                surface=QOffscreenSurface(context.screen()); surface.setFormat(context.format()); surface.create()
                if not context.makeCurrent(surface): raise RuntimeError('Share context could not be made current')
                gl=ctypes.WinDLL('opengl32'); gl.glGetString.restype=ctypes.c_char_p
                values={name:gl.glGetString(key).decode('ascii') for name,key in [('vendor',0x1F00),('renderer',0x1F01),('version',0x1F02)]}
                context.doneCurrent(); surface.destroy()
                write(metadata,{'status':'observed','scope':'QGIS global OpenGL share context','values':values})
            except Exception as exc:
                write(metadata,{'status':'unavailable','scope':'metadata probe only; no render substituted','reason':str(exc)})
        self.journal('All requested native frames verified')
        QgsProject.instance().setDirty(False)
        QTimer.singleShot(0,iface.mainWindow().close)

    def fail(self):
        (self.output/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
        self.journal('Capture failed; see failure.log')
        QgsProject.instance().setDirty(False)
        QTimer.singleShot(0,iface.mainWindow().close)


RUNNER=None


def start():
    global RUNNER
    directory=Path(os.environ['C05_SCENE_DIR']).resolve()
    try:
        # Keep the Python objects alive while their C++ scene uses them.
        RUNNER=Capture(directory,os.environ.get('C05_QGIS_PROBE') in {'1','transition'})
    except Exception:
        (directory/'qgis-3d-startup-error.log').write_text(traceback.format_exc(),encoding='utf-8')
        QgsProject.instance().setDirty(False); QTimer.singleShot(0,iface.mainWindow().close)


QTimer.singleShot(0,start)
