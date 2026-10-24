"""CPU-only audit of the two phase CDF coordinates and an actual CI capture."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from scripts import nephele_acceptance as acceptance

    asset = root / 'assets/media/nephele_blue_noise_8x8.txt'
    ranks = np.loadtxt(asset).astype(np.uint32)
    height, width = ranks.shape
    count = ranks.size
    if sorted(ranks.ravel().tolist()) != list(range(count)):
        raise ValueError('Expected the existing complete rank permutation')
    y, x = np.indices(ranks.shape)
    first = ranks.ravel()
    shifted = ranks[(y+7) % height, (x+1) % width].ravel()
    reversed_u32 = np.array([int(f'{int(i):032b}'[::-1], 2) for i in first], dtype=np.uint32)
    u1 = (first.astype(np.float32)+np.float32(.5))/np.float32(count)
    corrected = reversed_u32.astype(np.float32)/np.float32(4294967296.)+np.float32(.5/count)
    medium = json.loads((root/'tests/nephele/fixture/medium.json').read_text(encoding='utf-8'))
    g = medium['phase']['g']
    ct = (1+g*g-((1-g*g)/(1-g+2*g*u1.astype(np.float64)))**2)/(2*g)
    sine = np.sqrt(1-ct*ct)
    result = {}
    for name, u2 in [('shifted_tile', (shifted.astype(np.float32)+np.float32(.5))/np.float32(count)),
                     ('rank_radical_inverse', corrected)]:
        cells = np.zeros((height, width), int)
        np.add.at(cells, (np.floor(u1*height).astype(int), np.floor(u2*width).astype(int)), 1)
        phi = 2*np.pi*u2.astype(np.float64)
        result[name] = {
            'occupied_joint_cdf_cells': int(np.count_nonzero(cells)),
            'empty_joint_cdf_cells': int((cells == 0).sum()),
            'maximum_samples_per_cell': int(cells.max()),
            'coordinate_correlation': float(np.corrcoef(u1, u2)[0, 1]),
            'mean_axial_component': float(ct.mean()),
            'expected_axial_component': g,
            'mean_transverse_components': [float((np.cos(phi)*sine).mean()), float((np.sin(phi)*sine).mean())],
        }
    capture = json.loads((args.artifact_dir/'run-a.json').read_text(encoding='utf-8'))
    paths = ['src/shaders/nephele_froxel.wgsl', 'src/shaders/nephele_realtime_terrain_trace.wgsl']
    report = {
        'measurement_kind': 'CPU CDF-coordinate audit; not GPU execution of the corrected sampler',
        'candidate_source_sha256': {name: hashlib.sha256((root/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for name in paths},
        'rank_asset_sha256': hashlib.sha256(asset.read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'rank_count': count,
        'phase_cdf': result,
        'captured_source_revision': capture['diagnostics']['source_revision'],
        'captured_gate3': acceptance.gate3(args.artifact_dir),
        'captured_gate4': acceptance.gate4(args.artifact_dir),
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
