"""AEQUITAS scoring on the DIFFERENTIA Python recovery surface."""

import numpy as np

from _deltae import median_delta_e2000_linear
from forge3d import inverse


def test_recover_scene_reference_score_uses_aequitas(monkeypatch):
    recovered = np.array([[[0.25, 0.35, 0.45], [0.65, 0.55, 0.45]]], dtype=np.float32)
    reference = np.array([[[0.20, 0.30, 0.40], [0.60, 0.50, 0.40]]], dtype=np.float32)

    def fake_native(*args, **kwargs):
        return {
            "albedo": recovered,
            "sun_dir": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "sun_intensity": 2.5,
            "turbidity": 2.5,
            "loss_history": [0.1],
            "rgba": np.zeros((1, 2, 4), dtype=np.uint8),
            "iterations_run": 1,
            "peak_host_visible_bytes": 0,
        }

    monkeypatch.setattr(inverse, "_native_inverse", lambda name: fake_native)
    target = np.zeros((1, 2, 4), dtype=np.uint8)
    dem = np.zeros((1, 2), dtype=np.float32)

    scored = inverse.recover_scene(target, dem, reference_albedo=reference)
    assert scored.albedo_delta_e2000_median == median_delta_e2000_linear(
        recovered, reference
    )

    unscored = inverse.recover_scene(target, dem)
    assert unscored.albedo_delta_e2000_median is None
