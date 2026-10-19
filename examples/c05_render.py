"""Render the recorded C05 scenes and complete existing eight-second flyover.

Uses real ViewerHandle snapshots. Raw frames, camera samples, executable hashes,
timings and process peak memory are recorded; no visual pass is inferred.
"""
from __future__ import annotations
import argparse
from importlib.metadata import distributions
import json
from pathlib import Path
import time

import rasterio  # preload before example shim
import numpy as np
from PIL import Image, ImageDraw
import psutil
from _import_shim import ensure_repo_import
ensure_repo_import()
import forge3d
from forge3d.animation import CameraAnimation
from forge3d.viewer import _find_viewer_binary
from c05_real_data import record, peak_memory, implementation_identity
from forge3d.preparation import _sha


def camera_path(directory: Path) -> dict:
    recipe = json.loads((directory / "recipe.json").read_text(encoding="utf-8"))
    grid = recipe["target_grid"]
    span = grid["width"] * grid["geotransform"][1]
    # Existing camera_animation_demo flyover: same eight seconds, five authored
    # views; radius scale fits this real extent. 30 fps is render_animation's
    # existing default. This is a workload definition, not an acceptance cap.
    animation = CameraAnimation()
    keys = [(0,45,60,4000,45),(2,60,45,2000,55),(4,90,30,1500,60),(6,135,40,2500,55),(8,180,55,3500,50)]
    # Authored radius ratios (1.6, .8, .6, 1, 1.4 times the scene width).
    # Record every physical radius and orientation explicitly for both tools.
    for t, phi, theta, radius, fov in keys:
        animation.add_keyframe(time=t,phi=phi,theta=theta,radius=radius*span/2500,fov=fov)
    frames=[]
    for i in range(animation.get_frame_count(30)):
        state=animation.evaluate(i/30)
        if state is None:
            raise AssertionError("Missing interpolated camera sample")
        frames.append({"frame":i,"time":i/30,"phi_deg":state.phi_deg,"theta_deg":state.theta_deg,
                       "radius":state.radius,"fov_deg":state.fov_deg})
    return {"fps":30,"duration_seconds":animation.duration,"grid":grid,"frames":frames,
            "theta_convention":"polar angle down from vertical; 0 overhead, 90 horizon",
            "phi_convention":"eye offset east = r*sin(theta)*cos(phi), south = r*sin(theta)*sin(phi)",
            "source":"examples/camera_animation_demo.py:create_flyover_animation"}


def render(directory: Path, still_only: bool) -> None:
    started=time.perf_counter()
    output=directory/'forge3d-render'
    output.mkdir(exist_ok=True)
    assets=json.loads((directory/'viewer-assets/viewer-assets.json').read_text(encoding='utf-8'))
    path=camera_path(directory)
    record(directory/'camera-path.json',path)
    frame_dir=output/'frames'; frame_dir.mkdir(exist_ok=True)
    binary=Path(_find_viewer_binary())
    input_paths=[Path(assets['terrain_path']),directory/'display.png',directory/'recipe.json',directory/'camera-path.json']
    input_hashes={str(p):_sha(p) for p in input_paths}
    with forge3d.open_viewer_async(terrain_path=assets['terrain_path'],width=1280,height=720) as viewer:
        viewer.send_ipc({'cmd':'set_terrain_pbr','enabled':True})
        viewer.load_overlay('C05 measured display',directory/'display.png',extent=[0,0,1,1],preserve_colors=False)
        viewer.set_z_scale(1.0)
        viewer.set_sun(azimuth_deg=300,elevation_deg=35)
        first=path['frames'][0]
        viewer.set_orbit_camera(**{k:first[k] for k in ('phi_deg','theta_deg','radius','fov_deg')})
        viewer.snapshot(output/'still.png',width=1280,height=720)
        record(output/'still-stats.json',viewer.get_stats())
        # Observe the modules in this actual frame-producing process tree,
        # rather than borrowing identity from a later diagnostic launch.
        loaded_modules = {}
        launcher = psutil.Process(viewer._process.pid)
        for process in [launcher, *launcher.children(recursive=True)]:
            for mapping in process.memory_maps(grouped=True):
                module = Path(mapping.path)
                if 'forge3d' in str(module).lower() and module.suffix.lower() in {'.pyd', '.dll'} and module.is_file():
                    loaded_modules[str(module)] = {'sha256': _sha(module), 'process_id': process.pid}
        if not loaded_modules: raise AssertionError('Actual viewer native module identity was not observed')
        viewer_versions = {str(dist.locate_file('')): dist.version
                           for site in {str(Path(module).parent.parent) for module in loaded_modules}
                           for dist in distributions(path=[site]) if dist.metadata['Name'].lower() == 'forge3d'}
        if still_only:
            print('Rendered still',output/'still.png',flush=True)
            return
        render_start=time.perf_counter()
        hashes=[]; viewer_peak=0; process_peaks={}
        for sample in path['frames']:
            viewer.set_orbit_camera(**{k:sample[k] for k in ('phi_deg','theta_deg','radius','fov_deg')})
            frame=frame_dir/f"frame_{sample['frame']:05d}.png"
            viewer.snapshot(frame,width=1280,height=720)
            with Image.open(frame) as image:
                image.load()
                if image.size != (1280,720):
                    raise AssertionError('Unexpected frame size')
            hashes.append(_sha(frame))
            # Kernel-reported peak, no time-based sampling or invented cap.
            launcher=psutil.Process(viewer._process.pid)
            for process in [launcher, *launcher.children(recursive=True)]:
                try:
                    value=process.memory_info().peak_wset
                    if process.pid not in process_peaks:
                        exe=Path(process.exe())
                        process_peaks[process.pid]={'executable':str(exe),'executable_sha256':_sha(exe)}
                    process_peaks[process.pid]['peak_working_set_bytes']=value
                except psutil.NoSuchProcess:
                    continue
            viewer_peak=max(viewer_peak,sum(p['peak_working_set_bytes'] for p in process_peaks.values()))
        render_seconds=time.perf_counter()-render_start
        stats=viewer.get_stats()
    if len(set(hashes)) != len(hashes):
        raise AssertionError('A camera-path frame did not change')
    if any(_sha(p)!=input_hashes[str(p)] for p in input_paths):
        raise AssertionError('Render input changed during the camera sequence')
    encode_start=time.perf_counter()
    images=[Image.open(frame_dir/f"frame_{s['frame']:05d}.png").convert('RGB') for s in path['frames']]
    images[0].save(output/'flyover.webp',save_all=True,append_images=images[1:],duration=1000/path['fps'],loop=0,lossless=True)
    for image in images: image.close()
    sheet=Image.new('RGB',(1280,720*3),(245,244,240))
    indices=[0,len(path['frames'])//2,len(path['frames'])-1]
    for n,index in enumerate(indices):
        with Image.open(frame_dir/f'frame_{index:05d}.png') as image: sheet.paste(image,(0,n*720))
    sheet.save(output/'contact-sheet.png')
    record(output/'evidence.json',{'tool':'Forge3D existing local viewer','binary':str(binary),'binary_sha256':_sha(binary),
        'implementation':implementation_identity(),'loaded_viewer_native_modules':loaded_modules,
        'workspace_package_version':forge3d.__version__,'loaded_viewer_distribution_versions':viewer_versions,
        'version_scope':'Prebuilt installed viewer, not a build of the modified checkout; paths, modules and distribution versions observed during this frame run.',
        'preparation_binding_sha256':_sha(Path(forge3d.__file__).parent/'_forge3d.pyd'),
        'resolution':[1280,720],'frames':len(hashes),'fps':path['fps'],'frame_sha256':hashes,
        'input_sha256':input_hashes,
        'render_seconds':render_seconds,'encode_seconds':time.perf_counter()-encode_start,
        'total_seconds':time.perf_counter()-started,'driver':stats,
        'python_peak_working_set_bytes':peak_memory(),'viewer_process_sum_of_kernel_peaks_bytes':viewer_peak,
        'viewer_processes':process_peaks,
        'certificate_scope':'ViewerHandle.snapshot is excluded from offscreen CENSOR certification.',
        'vertical_scale':1,'review':'Raw images require inspection. Distinct frames prove movement, not visual quality.'})
    print('Rendered full camera path',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--still-only',action='store_true')
    args=parser.parse_args()
    render(args.directory.resolve(),args.still_only)
