# tests/test_f15_validate.py
# Mesh validation smoke tests for geometry diagnostics pipeline
# Exists to ensure F15 detects duplicates, degenerates, and non-manifold edges
# RELEVANT FILES:src/geometry/validate.rs,python/forge3d/geometry.py,tests/test_f8_weld.py,tests/test_f1_extrude.py

import numpy as np
import pytest

geometry = pytest.importorskip("forge3d.geometry")


def test_validation_clean_mesh() -> None:
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    report = geometry.validate_mesh(positions, indices)
    assert report["ok"] is True
    assert report["stats"]["vertex_count"] == 3


def test_validation_detects_duplicate() -> None:
    positions = np.array(
        [[0.0, 0.0, 0.0], [1e-7, 1e-7, 1e-7], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    indices = np.array([[0, 2, 1]], dtype=np.uint32)
    report = geometry.validate_mesh(positions, indices)
    kinds = {issue["type"] for issue in report["issues"]}
    assert "duplicate_vertex" in kinds or "degenerate_triangle" in kinds


def test_validation_detects_non_manifold() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [[0, 1, 2], [2, 1, 3], [0, 2, 1]],
        dtype=np.uint32,
    )
    report = geometry.validate_mesh(positions, indices)
    kinds = {issue["type"] for issue in report["issues"]}
    assert "non_manifold_edge" in kinds
