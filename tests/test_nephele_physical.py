"""Exact six-case, zero-skip NEPHELE physical acceptance suite.

Collect with ``-o python_functions=gate*`` so JUnit names are the six binding
contract names without a synthetic ``test_`` prefix. The evidence comes from
``scripts/nephele_acceptance.py produce`` and ``render`` on the physical lane.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts import nephele_acceptance as acceptance

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def physical() -> Path:
    raw = os.environ.get("FORGE3D_NEPHELE_ARTIFACT_DIR")
    if not raw:
        raise RuntimeError("FORGE3D_NEPHELE_ARTIFACT_DIR is required; physical acceptance never skips")
    return Path(raw).resolve()


def gate1_estimator_majorant_rr(physical: Path) -> None:
    result = acceptance.gate1(physical)
    assert not result["failures"], result


def gate2_energy(physical: Path) -> None:
    result = acceptance.gate2(physical)
    assert not result["failures"], result


def gate3_realtime_reference(physical: Path) -> None:
    result = acceptance.gate3(physical)
    assert not result["failures"], result


def gate4_terrain_coupling(physical: Path) -> None:
    result = acceptance.gate4(physical)
    assert not result["failures"], result


def gate5_compute_shadow_ridgeline(physical: Path) -> None:
    result = acceptance.gate5(physical)
    assert not result["failures"], result


def gate6_determinism_memory(physical: Path) -> None:
    result = acceptance.gate6(physical)
    assert not result["failures"], result
