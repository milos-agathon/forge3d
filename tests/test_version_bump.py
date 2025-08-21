def test_version_reports_010():
    import forge3d as f3d
    assert f3d.__version__ == "0.1.0"
    # compiled module should match
    from forge3d import _vulkan_forge as core
    assert core.__version__ == "0.1.0"