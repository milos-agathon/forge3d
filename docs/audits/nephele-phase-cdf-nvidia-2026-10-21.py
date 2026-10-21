"""Record the existing NVIDIA capture without rendering or changing gates."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import nephele_acceptance as acceptance

artifact = ROOT / '.release-evidence/2026-10-21/ci-39580065706'
candidate = '68cc04e9ab87de3d90e92e76a8690e016e5d759b'
previous = 'ac7390968bcba2b393ad4fbea58716f6dc1acd59'
runs = [json.loads((artifact / f'run-{run}.json').read_text(encoding='utf-8')) for run in ['a', 'b']]
assert all(run['diagnostics']['source_revision'] == candidate for run in runs)
assert (artifact/'checked-out-head.txt').read_text().strip() == candidate
cases = ET.parse(artifact/'junit.xml').getroot().findall('.//testcase')
tests = [{'name':case.attrib['name'], 'outcome':'failed' if case.find('failure') is not None else 'skipped' if case.find('skipped') is not None else 'passed'} for case in cases]
fixtures = {}
for path in sorted(acceptance.FIXTURE.glob('*.npy')):
    relative = path.relative_to(ROOT).as_posix()
    old = subprocess.check_output(['git','show',previous+':'+relative],cwd=ROOT)
    data = path.read_bytes()
    assert old == data, relative
    fixtures[relative] = hashlib.sha256(data).hexdigest()
report = {
    'measurement_kind':'Observed NVIDIA CI capture, read-only CPU evaluation of existing gate functions',
    'candidate_source_revision':candidate,
    'ci_run_id':39580065706,
    'ci_run_url':'https://github.com/milos-agathon/forge3d/actions/runs/39580065706',
    'ci_conclusion':'failure',
    'full_acceptance_summary':'failure',
    'adapter':runs[0]['diagnostics']['adapter'],
    'backend':runs[0]['diagnostics']['backend'],
    'driver':runs[0]['diagnostics']['driver'],
    'physical_test_counts':{key:sum(case['outcome']==key for case in tests) for key in ['passed','failed','skipped']},
    'physical_tests':tests,
    'gate3':acceptance.gate3(artifact),
    'gate4':acceptance.gate4(artifact),
    'unchanged_fixture_base_revision':previous,
    'unchanged_fixture_sha256':fixtures,
    'evidence_sha256':{name:hashlib.sha256((artifact/name).read_bytes()).hexdigest() for name in ['run-a.json','run-b.json','junit.xml','realtime-rgb.npy','terrain-occlusion-disabled-rgb.npy']},
    'owner_scope_decision':{
        'date':'2026-10-21',
        'answer':'Keep the current bounded scope',
        'effect':'No coupled environment/multiple-scattering expansion; existing physical failures block release.'
    },
    'release_status':'Not merged, tagged or published in this task; no full CI dispatched on the failing candidate.',
    'local_gpu_tests':'NOT_RUN',
}
output = ROOT/'docs/audits/nephele-phase-cdf-nvidia-2026-10-21.json'
output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'counts':report['physical_test_counts'],'gate3':report['gate3'],'gate4':report['gate4'],'unchanged_npy_files':len(fixtures)},indent=2))
