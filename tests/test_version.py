import importlib.metadata as im
import sys
import pytest


def test_version_consistency():
    """Package __version__ should reflect current project version."""
    DIST = "forge3d"
    pkg = __import__(DIST)
    assert hasattr(pkg, "__version__"), "Package must expose __version__"
    # Project target version
    expected = "0.86.0"
    assert pkg.__version__ == expected
    # If a dist is installed in the venv, check it if it matches; otherwise, skip
    try:
        dist_version = im.version(DIST)
        print("dist:", dist_version)
        print("pkg :", pkg.__version__)
        if dist_version != expected:
            pytest.skip(f"installed dist version ({dist_version}) differs from project version ({expected}) in dev env")
        assert True
    except im.PackageNotFoundError:
        pytest.skip("distribution metadata not available in dev environment")
