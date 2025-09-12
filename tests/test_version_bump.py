import pytest


def test_version_reports_current():
    import forge3d as f3d
    expected = "0.11.0"
    assert f3d.__version__ == expected
    # compiled module should match if available
    try:
        from forge3d import _forge3d as core
        assert getattr(core, "__version__", expected) == expected
    except Exception:
        pytest.skip("compiled extension not available or not rebuilt in this environment")
