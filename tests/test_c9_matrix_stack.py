import pytest

def test_c9_roundtrip_identity():
    try:
        import forge3d as f3d
    except ImportError as e:
        pytest.skip(f"forge3d unavailable: {e}")
    assert f3d.c9_push_pop_roundtrip(200) is True