# tests/test_provenance_offline_verify.py
# VERITAS: proves the standalone verifier re-verifies a natively-sealed
# provenance triple WITHOUT the compiled _forge3d extension. Runs the CLI in
# a subprocess with an import hook that masks forge3d._forge3d, forcing the
# pure-Python _ed25519 fallback, against the committed fixture triple.
# RELEVANT FILES: tools/verify_provenance.py, python/forge3d/provenance.py,
# python/forge3d/_ed25519.py, tests/test_provenance_veritas.py

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER = REPO_ROOT / "tools" / "verify_provenance.py"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "provenance"
IMAGE = FIXTURE_DIR / "image.png"
SOURCE_MAP = FIXTURE_DIR / "source_map.npy"
MANIFEST = FIXTURE_DIR / "provenance.json"

pytestmark = pytest.mark.skipif(
    not (IMAGE.exists() and SOURCE_MAP.exists() and MANIFEST.exists()),
    reason=(
        "committed provenance fixture missing; regenerate on a GPU host with "
        "FORGE3D_UPDATE_PROVENANCE_FIXTURE=1 python -m pytest "
        "tests/test_provenance_veritas.py -k measurable_win"
    ),
)

# Bootstrap that masks the compiled extension before anything imports it,
# then runs the standalone verifier exactly as a third party would.
_MASKED_BOOTSTRAP = textwrap.dedent(
    """
    import importlib.abc
    import runpy
    import sys

    class _MaskNative(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "forge3d._forge3d":
                raise ImportError("forge3d._forge3d masked for offline verification test")
            return None

    sys.meta_path.insert(0, _MaskNative())
    tool, *argv = sys.argv[1:]
    sys.argv = [tool] + argv
    runpy.run_path(tool, run_name="__main__")
    """
)


def _run_verifier_without_native(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _MASKED_BOOTSTRAP, str(VERIFIER), *map(str, args)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def test_offline_verifier_verifies_native_manifest_without_extension() -> None:
    result = _run_verifier_without_native(IMAGE, SOURCE_MAP, MANIFEST)
    assert result.returncode == 0, f"verifier failed:\n{result.stdout}\n{result.stderr}"
    assert "merkle_root_match: True" in result.stdout
    assert "signature_valid: True" in result.stdout
    assert "image_dims_match: True" in result.stdout
    assert "tamper_probe_single_texel_detected: True" in result.stdout
    assert "verified: True" in result.stdout
    # Per-source pixel coverage is reported for at least two real sources.
    coverage_lines = [
        line for line in result.stdout.splitlines() if line.startswith("coverage source_id=")
    ]
    assert len(coverage_lines) >= 2, result.stdout


def test_offline_verifier_detects_tampered_source_map(tmp_path) -> None:
    source_map = np.load(SOURCE_MAP)
    source_map = np.asarray(source_map, dtype=np.uint32).copy()
    source_map[0, 0] ^= 1
    tampered_path = tmp_path / "source_map_tampered.npy"
    np.save(tampered_path, source_map)

    result = _run_verifier_without_native(IMAGE, tampered_path, MANIFEST)
    assert result.returncode != 0
    assert "merkle_root_match: False" in result.stdout
    assert "verified: False" in result.stdout


def test_pure_python_report_on_fixture() -> None:
    """In-process check of the pure-Python path (no native calls involved)."""
    from forge3d import provenance as prov

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == prov.SCHEMA_VERSION
    source_map = np.asarray(np.load(SOURCE_MAP), dtype=np.uint32)

    report = prov.verify_provenance_offline(source_map, manifest)
    assert report["ok"] is True
    assert report["root_match"] is True
    assert report["signature_valid"] is True
    assert len(report["coverage"]) >= 2
    assert prov.SOURCE_ID_NONE not in report["coverage"]
    assert not report["unknown_source_ids"]

    # The signature must be pinned to the committed public key + root.
    from forge3d import _ed25519

    assert _ed25519.verify(
        bytes.fromhex(manifest["public_key"]),
        prov.SIGN_CONTEXT + bytes.fromhex(manifest["merkle_root"]),
        bytes.fromhex(manifest["signature"]),
    )
