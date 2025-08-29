import pytest

def test_c6_multithread_metrics():
    try:
        import forge3d as f3d
    except ImportError as e:
        pytest.skip(f"forge3d unavailable: {e}")
    m = f3d.c6_parallel_record_metrics(None)
    assert isinstance(m, dict)
    assert m.get("threads_used", 0) >= 2
    assert m.get("checksum_parallel") == m.get("checksum_single")