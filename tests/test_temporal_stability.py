"""
Tests for Milestone D3: Temporal stability metrics.

Verifies orbit sweep has acceptable frame-to-frame deltas.

RELEVANT FILES: docs/flake_debug_contract.md, scripts/run_flake_proofpack.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPORTS_DIR = Path(__file__).parent.parent / "reports" / "flake"

# D3 thresholds (relaxed for initial baseline)
SYNTH_DELTA_MEAN_MAX = 5.0
SYNTH_DELTA_P99_MAX = 50.0
SYNTH_DELTA_MAX_MAX = 150.0

REAL_DELTA_MEAN_MAX = 5.0
REAL_DELTA_P99_MAX = 50.0
REAL_DELTA_MAX_MAX = 150.0


def _load_temporal_metrics(scene: str) -> dict:
    """Load temporal metrics from proof pack output."""
    path = REPORTS_DIR / "milestone_d" / f"temporal_metrics_{scene}.json"
    if not path.exists():
        pytest.skip(f"Temporal metrics not found: {path}. Run proof pack first.")
    
    with open(path) as f:
        return json.load(f)


class TestTemporalStabilitySynthetic:
    """Milestone D3: Synthetic scene temporal stability."""
    
    @pytest.fixture
    def metrics(self):
        return _load_temporal_metrics("synth")
    
    def test_delta_mean_below_threshold(self, metrics):
        """Mean frame-to-frame delta should be below threshold."""
        delta_mean = metrics["metrics"]["delta_mean"]
        assert delta_mean <= SYNTH_DELTA_MEAN_MAX, (
            f"Temporal delta mean {delta_mean:.2f} > {SYNTH_DELTA_MEAN_MAX}"
        )
    
    def test_delta_p99_below_threshold(self, metrics):
        """99th percentile frame-to-frame delta should be below threshold."""
        delta_p99 = metrics["metrics"]["delta_p99"]
        assert delta_p99 <= SYNTH_DELTA_P99_MAX, (
            f"Temporal delta p99 {delta_p99:.2f} > {SYNTH_DELTA_P99_MAX}"
        )
    
    def test_delta_max_below_threshold(self, metrics):
        """Maximum frame-to-frame delta should be below threshold."""
        delta_max = metrics["metrics"]["delta_max"]
        assert delta_max <= SYNTH_DELTA_MAX_MAX, (
            f"Temporal delta max {delta_max:.2f} > {SYNTH_DELTA_MAX_MAX}"
        )
    
    def test_sufficient_frame_count(self, metrics):
        """Orbit should have at least 36 frames."""
        frame_count = metrics["metrics"]["frame_count"]
        assert frame_count >= 36, f"Only {frame_count} frames, need at least 36"


@pytest.mark.skip(reason="Real terrain orbit not yet implemented")
class TestTemporalStabilityReal:
    """Milestone D3: Real terrain scene temporal stability."""
    
    @pytest.fixture
    def metrics(self):
        return _load_temporal_metrics("real")
    
    def test_delta_mean_below_threshold(self, metrics):
        delta_mean = metrics["metrics"]["delta_mean"]
        assert delta_mean <= REAL_DELTA_MEAN_MAX
    
    def test_delta_p99_below_threshold(self, metrics):
        delta_p99 = metrics["metrics"]["delta_p99"]
        assert delta_p99 <= REAL_DELTA_P99_MAX
    
    def test_delta_max_below_threshold(self, metrics):
        delta_max = metrics["metrics"]["delta_max"]
        assert delta_max <= REAL_DELTA_MAX_MAX
