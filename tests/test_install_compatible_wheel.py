import importlib.util
import itertools
from pathlib import Path
import re

from packaging.tags import sys_tags


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "install_compatible_wheel.py"
PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("install_compatible_wheel", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _package_version() -> str:
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"(.+?)"', text, re.MULTILINE)
    assert match, "No version entry found in pyproject.toml"
    return match.group(1)


def _wheel_name(tag) -> str:
    return f"forge3d-{_package_version()}-{tag.interpreter}-{tag.abi}-{tag.platform}.whl"


def _unsupported_platform_tag():
    supported_platforms = {tag.platform for tag in sys_tags()}
    for platform in ("win_amd64", "manylinux_2_17_x86_64", "macosx_11_0_arm64"):
        if platform not in supported_platforms:
            return platform
    return "forge3d_test_platform"


def test_wheel_score_rejects_unsupported_platform():
    module = _load_script_module()
    tags = list(itertools.islice(sys_tags(), 8))
    assert tags

    supported_path = Path(_wheel_name(tags[0]))
    unsupported_path = Path(
        f"forge3d-{_package_version()}-{tags[0].interpreter}-{tags[0].abi}-{_unsupported_platform_tag()}.whl"
    )

    ranks = module._supported_tag_ranks()
    assert module._wheel_score(supported_path, ranks) is not None
    assert module._wheel_score(unsupported_path, ranks) is None


def test_wheel_score_prefers_more_specific_supported_tag():
    module = _load_script_module()
    tags = list(itertools.islice(sys_tags(), 8))
    assert len(tags) >= 2

    ranks = module._supported_tag_ranks()
    best = Path(_wheel_name(tags[0]))
    fallback = Path(_wheel_name(tags[-1]))

    assert module._wheel_score(best, ranks) < module._wheel_score(fallback, ranks)


def test_main_installs_best_matching_wheel(tmp_path, monkeypatch):
    module = _load_script_module()
    tags = list(itertools.islice(sys_tags(), 8))
    assert len(tags) >= 2

    best = tmp_path / _wheel_name(tags[0])
    fallback = tmp_path / _wheel_name(tags[-1])
    wrong_platform = tmp_path / (
        f"forge3d-{_package_version()}-{tags[0].interpreter}-{tags[0].abi}-{_unsupported_platform_tag()}.whl"
    )
    for wheel in (best, fallback, wrong_platform):
        wheel.touch()

    calls = []

    def fake_run(args, check):
        calls.append((args, check))

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.sys, "argv", ["install_compatible_wheel.py", str(tmp_path)])

    assert module.main() == 0
    assert calls == [([module.sys.executable, "-m", "pip", "install", str(best)], True)]
