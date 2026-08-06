"""Smoke tests for package installation metadata and public API."""

from importlib.metadata import distribution, metadata
from pathlib import Path
import hashlib
import platform
import re
import struct
import subprocess
import sys

import pytest

from tests.conftest import _validate_installed_wheel_paths


# One-off provenance lock evaluated by the installed release wheel from exact
# pre-SELENE source commit 7fa1b98464a44b672ebf9077503ad354a2a9285a.
_EGM96_BYTE_LOCK_POINTS = [
    (-89.5, 0.5),
    (-75.25, 42.75),
    (-60.0, -120.0),
    (-45.5, 179.5),
    (-30.25, -179.75),
    (-15.0, 90.0),
    (0.0, 0.0),
    (0.5, 179.5),
    (12.345, 67.89),
    (23.5, -45.5),
    (35.0, 120.0),
    (46.87, 102.45),
    (51.5074, -0.1278),
    (60.0, 10.0),
    (70.25, -135.0),
    (80.0, 179.0),
    (89.5, 359.5),
    (-33.8688, 151.2093),
    (27.9881, 86.925),
    (-22.9068, -43.1729),
]

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None


def _load_project_urls(pyproject: Path) -> dict[str, str]:
    """Return the ``[project.urls]`` table without requiring Python 3.11+."""
    if tomllib is not None:
        with pyproject.open("rb") as fh:
            return tomllib.load(fh)["project"]["urls"]

    urls: dict[str, str] = {}
    in_urls = False
    for raw_line in pyproject.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "[project.urls]":
            in_urls = True
            continue
        if in_urls and line.startswith("["):
            break
        if not in_urls:
            continue

        match = re.match(r'^"?(.+?)"?\s*=\s*"(.+)"$', line)
        if match:
            urls[match.group(1)] = match.group(2)

    if not urls:
        raise AssertionError("No [project.urls] table found in pyproject.toml")
    return urls


def test_load_project_urls_falls_back_without_tomllib(monkeypatch, tmp_path):
    """Python 3.10 still loads project URLs without stdlib tomllib."""

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "forge3d"

[project.urls]
Homepage = "https://example.com"
"Bug Tracker" = "https://example.com/issues"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys.modules[__name__], "tomllib", None)

    assert _load_project_urls(pyproject) == {
        "Homepage": "https://example.com",
        "Bug Tracker": "https://example.com/issues",
    }


def test_python_version_floor():
    """forge3d requires Python 3.10+."""
    assert sys.version_info >= (3, 10), "forge3d requires Python 3.10+"


def test_import_forge3d():
    """Package imports without error and exposes a version."""
    import forge3d

    assert forge3d.__version__


@pytest.mark.skipif(
    sys.platform != "win32" or platform.machine().lower() != "amd64",
    reason="one-off exact pre-SELENE Windows AMD64 release provenance",
)
def test_pre_selene_egm96_windows_release_byte_lock():
    """Establish the Windows baseline from unchanged pre-refactor sources."""
    import forge3d

    root = Path(__file__).resolve().parent.parent
    commit_object = subprocess.check_output(
        ["git", "cat-file", "-p", "HEAD"], cwd=root, text=True
    )
    parent = next(
        line.removeprefix("parent ")
        for line in commit_object.splitlines()
        if line.startswith("parent ")
    )
    assert parent == "7fa1b98464a44b672ebf9077503ad354a2a9285a"
    source_hashes = {
        "src/geo/geoid.rs": "fd9410f851ce7433102d6e40b052f8c76c0c9406be09b2833c819598e7b314c1",
        "assets/geoid/egm96_n120.bin": "b640e9dcefd1040f0b184a101e1eab2740486a85680a560080ec091eab796fe4",
        "Cargo.lock": "152339c2ddae5195920068029940c163119bcd19e5cc9f3bc82107f7f43b2313",
        "pyproject.toml": "b9013b02c47808c345f12320c7c7ec77e3f14d89a4b90ddb66c5cef3bb0888a4",
    }
    for relative, expected in source_hashes.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected

    payload = b"".join(
        struct.pack("<d", forge3d.geoid_undulation(lat, lon))
        for lat, lon in _EGM96_BYTE_LOCK_POINTS
    )
    actual = hashlib.sha256(payload).hexdigest()
    print(
        {
            "pre_refactor_sha": parent,
            "target": (sys.platform, platform.machine().lower()),
            "egm96_release_payload_sha256": actual,
        }
    )
    assert actual == "ab9469d5e078dbfaa5df9b02219733c482ee21ed1f872fa251ed7056da27a639"


def test_public_api_surface():
    """Key public symbols are accessible from the package root."""
    import forge3d

    required = [
        "open_viewer",
        "open_viewer_async",
        "Renderer",
        "RendererConfig",
        "MapPlate",
        "Legend",
        "ScaleBar",
        "has_gpu",
        "enumerate_adapters",
        "fetch_dataset",
        "set_license_key",
        "LicenseError",
        "__version__",
    ]
    for name in required:
        assert hasattr(forge3d, name), f"Missing public symbol: {name}"
    assert not hasattr(forge3d, "RenderView"), "RenderView should not be exported from package root"


def test_fetch_dataset_alias_matches_datasets_module():
    """The package root keeps the documented fetch_dataset alias."""
    import forge3d

    assert callable(forge3d.fetch_dataset)
    assert forge3d.fetch_dataset is forge3d.datasets.fetch
    assert not hasattr(forge3d, "fetch"), "Root package should expose fetch_dataset, not fetch"


def test_version_consistency():
    """Package version stays in sync with pyproject.toml."""
    import forge3d

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject.toml not available in this environment")

    match = re.search(
        r'^version\s*=\s*"(.+?)"',
        pyproject.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, "No version entry found in pyproject.toml"
    assert forge3d.__version__ == match.group(1)


def test_enumerate_adapters_smoke():
    """Adapter enumeration should not crash, even on GPU-less CI."""
    import forge3d

    adapters = forge3d.enumerate_adapters()
    assert isinstance(adapters, list)


def test_legacy_render_api_removed():
    """The legacy render_raster/render_polygons/render_raytrace_mesh API is gone."""
    import forge3d

    for name in ("render_raster", "render_polygons", "render_raytrace_mesh"):
        assert not hasattr(forge3d, name), f"Legacy API should be removed: {name}"


def test_installed_project_urls_match_public_metadata():
    """Installed metadata should point at the live repository and docs."""

    meta = metadata("forge3d")
    project_urls = meta.get_all("Project-URL") or meta.get_all("Project-Url") or []
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject.toml not available in this environment")

    expected_urls = _load_project_urls(pyproject)

    for label, url in expected_urls.items():
        assert f"{label}, {url}" in project_urls
    assert all("github.com/forge3d/forge3d" not in value for value in project_urls)


def test_installs_interactive_viewer_console_script():
    """The wheel exposes the interactive viewer command via console_scripts."""
    entry_points = distribution("forge3d").entry_points
    assert any(
        ep.group == "console_scripts"
        and ep.name == "interactive_viewer"
        and ep.value == "forge3d._viewer_entry:main"
        for ep in entry_points
    )


def test_installed_wheel_path_gate_rejects_repo_local_package(tmp_path):
    repo_package = Path(__file__).resolve().parent.parent / "python" / "forge3d"
    native = tmp_path / "site-packages" / "forge3d" / "_forge3d.abi3.so"
    native.parent.mkdir(parents=True)
    native.touch()

    with pytest.raises(RuntimeError, match="repo-local Python package"):
        _validate_installed_wheel_paths(repo_package / "__init__.py", native)


def test_installed_wheel_path_gate_rejects_repo_local_native(tmp_path):
    package = tmp_path / "site-packages" / "forge3d" / "__init__.py"
    package.parent.mkdir(parents=True)
    package.touch()
    repo_native = (
        Path(__file__).resolve().parent.parent
        / "python"
        / "forge3d"
        / "_forge3d.abi3.so"
    )

    with pytest.raises(RuntimeError, match="repo-local native extension"):
        _validate_installed_wheel_paths(package, repo_native)


def test_installed_wheel_path_gate_accepts_external_package_and_native(tmp_path):
    package = tmp_path / "site-packages" / "forge3d" / "__init__.py"
    native = tmp_path / "site-packages" / "forge3d" / "_forge3d.abi3.so"
    package.parent.mkdir(parents=True)
    package.touch()
    native.touch()

    _validate_installed_wheel_paths(package, native)
