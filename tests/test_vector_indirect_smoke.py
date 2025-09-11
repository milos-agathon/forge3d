"""Minimal smoke test for indirect draw path.

This test does not execute GPU draws. It ensures that the code paths for
vector indirect culling are wired and discoverable from the Python surface
via the GPU metrics mapping.
"""

from forge3d import gpu_metrics


def test_indirect_vector_metric_present():
    metrics = gpu_metrics.get_available_metrics()
    # The mapping includes a 'vector_indirect_culling' entry for the indirect path
    assert "vector_indirect_culling" in metrics

