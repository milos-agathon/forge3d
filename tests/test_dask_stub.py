from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PYTHON = REPO_ROOT / "python"


def _load_stub_package(alias: str):
    package_dir = REPO_PYTHON / "dask"
    init_path = package_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        alias,
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _has_real_xarray() -> bool:
    repo_python_norm = str(REPO_PYTHON.resolve()).casefold()
    filtered = [
        entry
        for entry in sys.path
        if str(Path(entry or ".").resolve()).casefold() != repo_python_norm
    ]
    spec = importlib.machinery.PathFinder.find_spec("xarray", filtered)
    return spec is not None and spec.origin is not None


def test_local_dask_stub_exports_xarray_compat_surface():
    stub = _load_stub_package("forge3d_test_dask_stub")
    base = importlib.import_module("forge3d_test_dask_stub.base")

    assert stub.__forge3d_stub__ is True
    assert stub.is_dask_collection is base.is_dask_collection
    assert stub.is_dask_collection([1, 2, 3]) is False
    assert base.tokenize({"a": 1}, mode="r") == base.tokenize({"a": 1}, mode="r")
    assert list(base.flatten([[("a", 1)], [("b", 2)]])) == [("a", 1), ("b", 2)]
    assert base.replace_name_in_key(("old", 1, 2), {"old": "new"}) == ("new", 1, 2)
    assert base.get_scheduler() is None


@pytest.mark.skipif(not _has_real_xarray(), reason="real xarray not installed")
def test_real_xarray_dataset_creation_works_with_local_dask_stub():
    code = """
import sys
repo_python = r"{repo_python}"
import dask
assert getattr(dask, "__forge3d_stub__", False)
repo_python_norm = repo_python.lower()
sys.path = [
    p for p in sys.path
    if p.lower() != repo_python_norm
]
import xarray as xr
ds = xr.Dataset({{"a": ("x", [1, 2, 3])}}, coords={{"x": [0, 1, 2]}})
assert ds.sizes["x"] == 3
""".format(repo_python=str(REPO_PYTHON))
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_PYTHON)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
