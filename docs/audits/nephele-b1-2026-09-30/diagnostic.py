"""T4/B1 diagnostic only. Never produces physical-lane acceptance markers."""
from pathlib import Path
import hashlib
import json
import os
import sys
import subprocess
import time

REPO = Path('D:/forge3d/.worktrees/nephele-172-reference')
FIXTURE = REPO / 'tests/nephele/fixture'
OUT = Path('D:/forge3d/cache/nephele172-6a4ae50a-b1')
SOURCE = '6a4ae50a4c3c01b390b1222468632e88d1db6ca0'
os.environ.update(FORGE3D_NO_BOOTSTRAP='1', FORGE3D_TEST_INSTALLED_WHEEL='1',
                  WGPU_BACKEND='vulkan', WGPU_BACKENDS='vulkan')
sys.path.insert(0, str(REPO))
import numpy as np
import forge3d as f3d
from forge3d.media import Medium, _native_module
from forge3d.terrain_params import AovSettings, TonemapSettings, make_terrain_params_config
from scripts.generate_media_fixture import _aces_srgb8
from scripts.run_media_physical_capture import _write_hdr, _camera_contract, _crop
from tests._deltae import srgb_to_lab, delta_e_2000


def read(name):
    return json.loads((FIXTURE / name).read_text(encoding='utf-8'))


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def metrics(a, b, mask):
    values = delta_e_2000(srgb_to_lab(a), srgb_to_lab(b))[mask]
    return {'pixels': int(values.size), 'mean_delta_e': float(values.mean()),
            'maximum_delta_e': float(values.max()), 'p95_delta_e': float(np.quantile(values, .95)),
            'fraction_delta_e_below_2': float(np.mean(values < 2.0))}


def render_vacuum(count):
    start = time.monotonic()
    result = _native_module()._render_volumetric_reference(
        vacuum._native, np.ascontiguousarray(terrain, dtype=np.float32),
        crop['width'], crop['height'], native_camera,
        spacing=tuple(terrain_data['spacing']), exaggeration=terrain_data['exaggeration'],
        albedo=tuple(material_input['albedo']), sun_azimuth_deg=sun['azimuth_deg'],
        sun_elevation_deg=sun['elevation_deg'], sun_intensity=sun['intensity'], sun_color=tuple(sun['color']),
        environment_intensity=atmosphere['environment_intensity'], exposure=exposure['value'],
        samples_per_pixel=count, homogeneous_medium_reach=120.0, seed=provenance['seed'],
        full_viewport=tuple(crop['full_viewport']), crop=(crop['x'], crop['y'], crop['width'], crop['height']),
    )
    rgb = _aces_srgb8(np.asarray(result['beauty'], dtype=np.float32))
    np.save(OUT / f'vacuum-reference-spp{count}.npy', rgb, allow_pickle=False)
    if (result['diagnostics']['source_revision'] != SOURCE or
        any(result['diagnostics'][key] != provenance['diagnostics'][key]
            for key in ('adapter', 'backend', 'driver'))):
        raise RuntimeError('Unexpected vacuum reference source/adapter/backend/driver')
    record = {'spp': count, 'wall_seconds': time.monotonic() - start, 'diagnostics': dict(result['diagnostics'])}
    save(f'vacuum-reference-spp{count}.json', record)
    return record


manifest = json.loads((REPO / 'tests/nephele/fixture-manifest.json').read_text(encoding='utf-8'))
if manifest['status'] != 'APPROVED':
    raise RuntimeError('T4/B1 runs only after reference convergence approval')
provenance = read('reference-provenance.json')
if provenance['source_revision'] != SOURCE:
    raise RuntimeError('Reference source identity changed')
for item in manifest['scene_inputs'].values():
    path = REPO / item['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        raise RuntimeError(f'Scene input hash mismatch: {path.name}')
camera_input, terrain_data, medium_data = read('camera.json'), read('terrain.json'), read('medium.json')
material_input, sun, atmosphere = read('material.json'), read('sun.json'), read('atmosphere.json')
exposure, crop = read('exposure.json'), read('crop.json')
camera = camera_input['terrain_camera']
terrain = np.load(FIXTURE / terrain_data['dem'], allow_pickle=False)
shape = medium_data['domain']['grid_shape']
density = np.asarray(medium_data['density_r16'], dtype=np.float32).reshape(shape[2], shape[1], shape[0]) / 65535.0
bounds = (medium_data['domain']['bounds_min'], medium_data['domain']['bounds_max'])
medium = Medium.grid3d(medium_data['sigma_a'], medium_data['sigma_s'], density, bounds,
                       phase='henyey_greenstein', g=medium_data['phase']['g'],
                       density_scale=medium_data['density_scale'], version=1)
vacuum = Medium.grid3d([0.0] * 3, [0.0] * 3, density, bounds,
                       phase='henyey_greenstein', g=medium_data['phase']['g'],
                       density_scale=medium_data['density_scale'], version=1)
spans = [terrain_data['spacing'][i] * (terrain_data['dimensions'][i] - 1) for i in range(2)]
if spans[0] != spans[1]:
    raise RuntimeError('Expected the tracked square terrain span')
native_camera = dict(camera_input)
native_camera['terrain_camera'] = {**camera, 'target': tuple(camera['target'])}
if len(sys.argv) == 3 and sys.argv[1] == '--vacuum':
    if not OUT.is_dir():
        raise RuntimeError('Vacuum worker requires its diagnostic parent directory')
    render_vacuum(int(sys.argv[2]))
    raise SystemExit(0)
OUT.mkdir(parents=True, exist_ok=False)
config = make_terrain_params_config(
    size_px=tuple(crop['full_viewport']), render_scale=1.0, terrain_span=spans[0], msaa_samples=1,
    z_scale=float(terrain_data['exaggeration']), exposure=float(exposure['value']),
    domain=(float(terrain.min()), float(terrain.max())), light_azimuth_deg=sun['azimuth_deg'],
    light_elevation_deg=sun['elevation_deg'], sun_intensity=sun['intensity'], sun_color=sun['color'],
    albedo_mode=material_input['albedo_mode'], colormap_strength=float(material_input['colormap_strength']),
    cam_radius=camera['radius'], cam_phi_deg=camera['phi_deg'], cam_theta_deg=camera['theta_deg'],
    cam_target=camera['target'], fov_y_deg=float(camera_input['fov_y']), camera_mode=camera['mode'],
    aa_samples=1, aa_seed=0x4E455048, tonemap=TonemapSettings(operator='aces'),
    aov=AovSettings(enabled=True, transmittance=True, in_scatter=True, cloud_shadow=True, optical_depth=True),
    media=medium,
)
params = f3d.TerrainRenderParams(config)
renderer = f3d.TerrainRenderer(f3d.Session(window=False))
material = f3d.MaterialSet.custom(tuple(material_input['albedo']), float(material_input['metallic']),
    float(material_input['roughness']), triplanar_scale=float(material_input['triplanar_scale']),
    normal_strength=float(material_input['normal_strength']), blend_sharpness=float(material_input['blend_sharpness']))
_write_hdr(OUT / 'fixture.hdr')
ibl = f3d.IBL.from_hdr(str(OUT / 'fixture.hdr'), intensity=float(atmosphere['environment_intensity']))
# Use the exact no-medium branch used by the physical ablation. Do not compare
# the enabled image to the reference or use it to alter the frozen fixture.
capture = renderer._capture_nephele_acceptance(material, ibl, params, terrain,
                                              terrain_occlusion_in_media=True, include_no_medium=True)
diag = dict(capture['diagnostics'])
if (diag['source_revision'] != SOURCE or
    any(diag[key] != provenance['diagnostics'][key]
        for key in ('adapter', 'backend', 'driver'))):
    raise RuntimeError('Unexpected candidate diagnostic source/adapter/backend/driver')
actual_camera, expected_camera = _camera_contract(capture['camera_contract']), _camera_contract(provenance['camera_contract'])
if any(not np.array_equal(np.asarray(actual_camera[k], np.float32), np.asarray(expected_camera[k], np.float32))
       for k in expected_camera):
    raise RuntimeError('Candidate/reference camera contracts differ')
candidate = _crop(np.asarray(capture['no_medium_beauty'], dtype=np.uint8), crop)
np.save(OUT / 'medium-disabled-rgb.npy', candidate, allow_pickle=False)
save('candidate-diagnostics.json', diag)
save('camera-contract.json', actual_camera)
mask = np.load(FIXTURE / 'terrain-mask.npy', allow_pickle=False)
native_camera = dict(camera_input)
native_camera['terrain_camera'] = {**camera, 'target': tuple(camera['target'])}
# Reuse the fixture's measured exact-doubling sequence from its first prefix.
# The vacuum control must itself meet the same terrain convergence criterion;
# the medium-filled scene's eventual sample count is not a vacuum requirement.
prefix_events = [json.loads(line) for line in
    Path('D:/forge3d/cache/nephele172-6a4ae50a-prefixes/progress.jsonl').read_text(encoding='utf-8').splitlines()]
count = min(event['spp'] for event in prefix_events if event['event'] == 'generation_finished')
previous = None
records = []
while True:
    print(json.dumps({'event': 'vacuum_reference_start', 'spp': count}), flush=True)
    with (OUT / f'vacuum-reference-spp{count}.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, __file__, '--vacuum', str(count)], check=True,
                       stdout=log, stderr=subprocess.STDOUT, timeout=12 * 60 * 60)
    rgb = np.load(OUT / f'vacuum-reference-spp{count}.npy', allow_pickle=False)
    record = json.loads((OUT / f'vacuum-reference-spp{count}.json').read_text(encoding='utf-8'))
    if previous is not None:
        record['self_convergence'] = metrics(previous, rgb, mask)
    records.append(record)
    print(json.dumps(record), flush=True)
    if previous is not None and record['self_convergence']['fraction_delta_e_below_2'] >= .95:
        break
    # Owner's 12-hour per-generation budget also bounds this diagnostic.
    if record['wall_seconds'] * 2 > 12 * 60 * 60:
        raise RuntimeError('Vacuum doubling projects beyond the owner generation budget')
    previous, count = rgb, count * 2
comparison = metrics(candidate, rgb, mask)
record = {'schema': 'forge3d.nephele.b1_diagnostic/1', 'physical_acceptance': False,
          'diagnostic_tool': {'path': str(Path(__file__).resolve()), 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
          'native_source_revision': SOURCE, 'fixture_manifest_sha256': hashlib.sha256((REPO / 'tests/nephele/fixture-manifest.json').read_bytes()).hexdigest(),
          'scene_inputs': manifest['scene_inputs'], 'changed_control_input': 'sigma_a = sigma_s = RGB zero',
          'vacuum_reference_prefixes': records, 'terrain_comparison': comparison,
          'owner_decision_OD9_required': comparison['fraction_delta_e_below_2'] < .95}
save('b1-report.json', record)
print(json.dumps(record), flush=True)
