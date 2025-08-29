import pytest
import forge3d

def test_c5_framegraph_booleans_true():
    report = forge3d._forge3d.c5_build_framegraph_report()
    assert isinstance(report, dict)
    assert "alias_reuse" in report and "barrier_ok" in report
    assert isinstance(report["alias_reuse"], bool)
    assert isinstance(report["barrier_ok"], bool)
    # Acceptance: must be True
    assert report["alias_reuse"] is True
    assert report["barrier_ok"] is True