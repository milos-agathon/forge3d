def test_version_reports_050():
    import forge3d as f3d
    assert f3d.__version__ == "0.5.0"
    # compiled module should match
    from forge3d import _forge3d as core
    assert core.__version__ == "0.5.0"