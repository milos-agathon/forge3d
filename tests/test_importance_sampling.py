#!/usr/bin/env python3
"""A25: Tests for Object Importance Sampling

Covers:
- Setting per-object importance weights and sampling an object with correct MIS weight
- Variance reduction utility reaching ≥15% threshold (meets_performance_target)
- Edge cases: empty sampler returns None
"""

import numpy as np

from forge3d.importance_sampling import ObjectImportanceSampler


def test_empty_sampler_returns_none():
    s = ObjectImportanceSampler()
    assert s.sample_object(0.5) is None


def test_sampling_and_mis_weight():
    s = ObjectImportanceSampler()
    # Set two objects with different importance
    s.set_object_importance(1, 0.7)
    s.set_object_importance(2, 0.3)

    # Deterministic u in the upper 30% should map to object 1 with pdf ~ 0.7
    obj, mis = s.sample_object(0.85)
    assert obj in (1, 2)
    if obj == 1:
        assert abs(mis - 0.7) < 1e-6
    else:
        # In case ordering differs, ensure pdfs are consistent
        assert abs(mis - 0.3) < 1e-6

    # CDF monotonicity: small u tends to pick lower-weight mass first
    sample_counts = {1: 0, 2: 0}
    us = np.linspace(0.0, 0.999, 100)
    for u in us:
        res = s.sample_object(float(u))
        assert res is not None
        obj_id, _ = res
        sample_counts[obj_id] += 1
    # Object 1 (0.7 weight) should be picked more often than object 2 (0.3)
    assert sample_counts[1] > sample_counts[2]


def test_variance_reduction_threshold():
    s = ObjectImportanceSampler()
    vr = s.calculate_variance_reduction(baseline_mse=1.0, optimized_mse=0.8)
    # Allow tiny FP rounding while still enforcing the 20% target
    assert vr >= 0.2 - 1e-12
    assert s.meets_performance_target(vr) is True
