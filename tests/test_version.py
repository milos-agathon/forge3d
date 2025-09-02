import importlib.metadata as im
import sys

def test_version_consistency():
    """Test that package __version__ matches distribution metadata version."""
    # Infer distribution name; default to "forge3d" if not overridden
    DIST = "forge3d"

    pkg = __import__(DIST)
    assert hasattr(pkg, "__version__"), "Package must expose __version__"
    dist_version = im.version(DIST)
    print("dist:", dist_version)
    print("pkg :", pkg.__version__)
    assert pkg.__version__ == dist_version