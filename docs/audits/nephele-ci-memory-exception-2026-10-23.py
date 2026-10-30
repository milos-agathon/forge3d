"""Reproduce the proposed assertion check without editing tests or using a GPU."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
RECORD = Path(__file__).with_suffix('.json')


def main():
    record = json.loads(RECORD.read_text(encoding='utf-8'))
    raw = subprocess.check_output(
        ['git', 'show', record['head_sha'] + ':' + record['path']], cwd=ROOT
    )
    assert hashlib.sha256(raw).hexdigest() == record['old_git_sha256']
    text = raw.decode('utf-8')
    assert text.count(record['exact_old']) == 1
    text = text.replace(record['exact_old'], record['exact_new'], 1)
    assert hashlib.sha256(text.encode()).hexdigest() == record['new_git_sha256']
    os.environ['FORGE3D_NO_BOOTSTRAP'] = '1'
    sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
    module = types.ModuleType('tests._memory_contract_reproducer')
    module.__file__ = str(ROOT / record['path'])
    sys.modules[module.__name__] = module
    exec(compile(text, module.__file__, 'exec'), module.__dict__)
    test = module.test_nephele_physical_exclusion_is_one_hosted_site_and_six_required_gates
    test()
    original_read = Path.read_text
    workflow_path = ROOT / '.github/workflows/ci.yml'
    workflow = workflow_path.read_text(encoding='utf-8')
    option = '--deselect=tests/test_budget_enforce.py::test_600mib_host_visible_raises_named_budget_error'
    mutations = {
        'arbitrary extra deselection': workflow + '\n# --deselect=tests/unauthorized.py\n',
        'duplicate memory exception': workflow + '\n# ' + option + '\n',
        'exception outside fast step': workflow.replace(option, '', 1) + '\n# ' + option + '\n',
        'different memory case': workflow.replace(option, '--deselect=tests/test_budget_enforce.py::other_case', 1),
    }
    for name, mutated in mutations.items():
        def read(path, *args, **kwargs):
            if path.resolve() == workflow_path.resolve():
                return mutated
            return original_read(path, *args, **kwargs)
        Path.read_text = read
        try:
            try:
                test()
            except (AssertionError, IndexError):
                continue
            raise AssertionError(name + ' was accepted')
        finally:
            Path.read_text = original_read
    print(json.dumps({'proposed_assertion_pass': True, 'rejected_mutations': list(mutations), 'GPU_executed': False}))


if __name__ == '__main__':
    main()
