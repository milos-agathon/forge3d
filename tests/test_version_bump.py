def test_version_reports_010():
    import vulkan_forge as vf
    assert vf.__version__ == "0.1.0"
    # compiled module should match
    from vulkan_forge import _vulkan_forge as core
    assert core.__version__ == "0.1.0"