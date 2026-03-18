"""Smoke tests for package installation metadata and public API."""

from importlib.metadata import metadata
from pathlib import Path
import re
import sys
import tomllib

import pytest


def test_python_version_floor():
    """forge3d requires Python 3.10+."""
    assert sys.version_info >= (3, 10), "forge3d requires Python 3.10+"


def test_import_forge3d():
    """Package imports without error and exposes a version."""
    import forge3d

    assert forge3d.__version__


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

    with pyproject.open("rb") as fh:
        expected_urls = tomllib.load(fh)["project"]["urls"]

    for label, url in expected_urls.items():
        assert f"{label}, {url}" in project_urls
    assert all("github.com/forge3d/forge3d" not in value for value in project_urls)
