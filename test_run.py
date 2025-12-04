from pathlib import Path
p = Path('reports/flake/test_run')
p.mkdir(parents=True, exist_ok=True)
(p / 'marker.txt').write_text('proofpack test')
print('marker created')
