"""Guards for the Paris-Eiffel acceptance fixture.

These run without a GPU. They protect the oracle itself: if the reference
frame, the class mask, or the derived targets drift, the colour gate silently
changes meaning, and every later "it matches the reference" claim becomes
unfalsifiable.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "paris_eiffel_day"
REFERENCE = FIXTURES / "reference_day.png"
MASK = FIXTURES / "class_mask.png"
TARGETS = FIXTURES / "class_targets.json"
METRIC = REPO_ROOT / "tools" / "paris_eiffel_day" / "colour_metric.py"

REFERENCE_SHA256 = "CA13E48F90F8C3ACDA40BC086C76E3F1C0CF0D7B456228D7F3D2EA92ED22A067"

# Design spec section 9. The mask must never silently gain or lose a class.
EXPECTED_CLASS_IDS = set(range(15))
EXPECTED_MEASURED = {1, 3, 4, 5, 8, 9, 10, 12, 13}
EXPECTED_ABSTAINED = {0, 2, 6, 7, 11, 14}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_reference_frame_hash_is_pinned():
    assert REFERENCE.exists(), "reference frame fixture is missing"
    assert _sha256(REFERENCE) == REFERENCE_SHA256


def test_mask_matches_reference_geometry():
    reference = Image.open(REFERENCE)
    mask = Image.open(MASK)
    assert mask.size == reference.size == (960, 720)
    assert mask.mode == "L"


def test_targets_were_derived_from_this_reference():
    payload = json.loads(TARGETS.read_text(encoding="utf-8"))
    assert payload["reference_sha256"] == REFERENCE_SHA256


def test_class_table_is_complete_and_partitioned():
    payload = json.loads(TARGETS.read_text(encoding="utf-8"))
    ids = {entry["class_id"] for entry in payload["classes"].values()}
    assert ids == EXPECTED_CLASS_IDS

    measured = {e["class_id"] for e in payload["classes"].values() if e["status"] == "measured"}
    abstained = {e["class_id"] for e in payload["classes"].values() if e["status"] == "abstained"}
    assert measured == EXPECTED_MEASURED
    assert abstained == EXPECTED_ABSTAINED
    assert measured.isdisjoint(abstained)


def test_every_abstention_records_a_reason():
    payload = json.loads(TARGETS.read_text(encoding="utf-8"))
    for name, entry in payload["classes"].items():
        if entry["status"] == "abstained":
            assert entry.get("reason"), f"{name} abstains without a recorded reason"


def test_mask_only_contains_declared_classes():
    mask = np.asarray(Image.open(MASK), dtype=np.uint8)
    present = set(np.unique(mask).tolist()) - {255}
    assert present <= EXPECTED_CLASS_IDS
    assert present == EXPECTED_MEASURED, "mask pixels disagree with the measured class set"


def test_measured_targets_reproduce_from_the_mask():
    """The stored targets must be what the mask actually yields.

    This is the load-bearing check: it catches a targets file edited by hand,
    or a mask regenerated without refreshing the targets.
    """
    reference = np.asarray(Image.open(REFERENCE).convert("RGB"), dtype=np.float32)
    mask = np.asarray(Image.open(MASK), dtype=np.uint8)
    payload = json.loads(TARGETS.read_text(encoding="utf-8"))

    for name, entry in payload["classes"].items():
        if entry["status"] != "measured":
            continue
        sel = mask == entry["class_id"]
        assert sel.sum() == entry["assigned_px"], f"{name}: pixel count drifted"
        median = np.median(reference[sel], axis=0)
        stored = np.asarray(entry["target_rgb"], dtype=np.float32)
        assert np.allclose(median, stored, atol=0.51), f"{name}: target is not the mask median"


def test_tolerances_respect_the_derivation_rule():
    """Spec section 12.2: tolerance = max(6, IQR). Never hand-picked."""
    payload = json.loads(TARGETS.read_text(encoding="utf-8"))
    for name, entry in payload["classes"].items():
        if entry["status"] != "measured":
            continue
        expected = max(6.0, entry["iqr_max"])
        assert entry["tolerance"] == pytest.approx(expected, abs=0.05), (
            f"{name}: tolerance {entry['tolerance']} is not max(6, IQR={entry['iqr_max']})"
        )


def test_metric_self_test_passes():
    """The reference must score a perfect pass against itself."""
    result = subprocess.run(
        [sys.executable, str(METRIC), "--self-test"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VERDICT: PASS" in result.stdout


def test_metric_rejects_a_wrong_image(tmp_path):
    """A metric that cannot fail is not a gate.

    Feed it a flat grey frame: every class must fail and the exit code must be
    non-zero.
    """
    wrong = tmp_path / "flat.png"
    Image.new("RGB", (960, 720), (128, 128, 128)).save(wrong)
    result = subprocess.run(
        [sys.executable, str(METRIC), str(wrong)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "VERDICT: FAIL" in result.stdout


def test_metric_refuses_a_mismatched_resolution(tmp_path):
    """Comparison must happen at the reference's own size, unresampled."""
    wrong = tmp_path / "small.png"
    Image.new("RGB", (480, 360), (128, 128, 128)).save(wrong)
    result = subprocess.run(
        [sys.executable, str(METRIC), str(wrong)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "resolution" in (result.stdout + result.stderr).lower() or "reference is" in (
        result.stdout + result.stderr
    )
