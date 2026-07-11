# tests/test_determinism_hash.py
# TERRA-DETERMINATA: hash-diff harness for the deterministic reference render.
# Renders the canonical terrain+CSM+IBL scene, asserts intra-backend
# bit-identity (fast local proxy for the cross-vendor claim) and equality with
# the committed golden SHA-256.
# RELEVANT FILES: python/forge3d/determinism.py, src/core/gpu.rs,
# tests/goldens/determinism/terra_determinata_v1.sha256,
# .github/workflows/determinism-matrix.yml
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from _terrain_runtime import terrain_rendering_available
from forge3d import determinism
from forge3d.determinism import CANONICAL_SCENE, render_reference

if not terrain_rendering_available():
    pytest.skip(
        "determinism hash tests require a terrain-capable hardware-backed forge3d runtime",
        allow_module_level=True,
    )

GOLDEN_PATH = Path(__file__).parent / "goldens" / "determinism" / f"{CANONICAL_SCENE}.sha256"


def _local_backend() -> str:
    """Single backend to pin for the local leg of the determinism proof."""
    explicit = os.environ.get("FORGE3D_DETERMINISM_TEST_BACKEND")
    if explicit:
        return explicit
    if sys.platform == "win32":
        return "dx12"
    if sys.platform == "darwin":
        return "metal"
    return "vulkan"


def test_intra_backend_bit_identity(tmp_path):
    """Two renders of the canonical scene on the same backend must be byte-identical.

    Zero-byte tolerance: the SHA-256 of the PNGs must match exactly. This is
    the local, always-runnable proxy for the cross-vendor CI matrix.
    """
    backend = _local_backend()
    first = render_reference(
        CANONICAL_SCENE,
        width=512,
        height=512,
        backend=backend,
        out_png=tmp_path / "render_a.png",
    )
    second = render_reference(
        CANONICAL_SCENE,
        width=512,
        height=512,
        backend=backend,
        out_png=tmp_path / "render_b.png",
    )
    assert (tmp_path / "render_a.png").stat().st_size > 0
    assert first == second, (
        f"intra-backend nondeterminism on '{backend}': re-render changed bytes\n"
        f"  first:  {first}\n  second: {second}"
    )


def test_matches_committed_golden(tmp_path):
    """The canonical render must equal the committed golden hash, byte-exact."""
    assert GOLDEN_PATH.exists(), (
        f"missing golden hash file {GOLDEN_PATH}; generate it with\n"
        f"  python -m forge3d.determinism --out-png ref.png\n"
        f"under FORGE3D_DETERMINISTIC=1 + a pinned WGPU_BACKENDS and commit the hash"
    )
    golden = GOLDEN_PATH.read_text().split()[0].strip()
    actual = render_reference(
        CANONICAL_SCENE,
        width=512,
        height=512,
        backend=_local_backend(),
        out_png=tmp_path / "render_golden_check.png",
    )
    assert actual == golden, (
        f"determinism hash mismatch against committed golden\n"
        f"  golden: {golden}\n  actual: {actual}\n"
        f"Zero-byte tolerance: if this diverges the pipeline picked up a "
        f"nondeterminism source (or the scene changed; regenerate the golden "
        f"deliberately and document why)."
    )


def test_deterministic_mode_requires_backend_pin(tmp_path):
    """render_reference refuses to guess a backend (loud failure, no fallback)."""
    env_backends = os.environ.pop("WGPU_BACKENDS", None)
    env_backend = os.environ.pop("WGPU_BACKEND", None)
    try:
        with pytest.raises(ValueError, match="explicit backend"):
            render_reference(
                CANONICAL_SCENE,
                width=64,
                height=64,
                backend=None,
                out_png=tmp_path / "never_written.png",
            )
    finally:
        if env_backends is not None:
            os.environ["WGPU_BACKENDS"] = env_backends
        if env_backend is not None:
            os.environ["WGPU_BACKEND"] = env_backend


def test_cli_attributes_hash_to_requested_backend(monkeypatch, tmp_path, capsys):
    """Adapter reporting must probe the backend that rendered the hash."""
    requested = []

    monkeypatch.setenv("WGPU_BACKENDS", "dx12")
    monkeypatch.setenv("FORGE3D_DETERMINISTIC", "1")
    monkeypatch.setattr(
        determinism,
        "_render_reference_inprocess",
        lambda *_args: "0" * 64,
    )

    import forge3d as f3d

    def probe(backend=None):
        requested.append(backend)
        return {
            "name": "test adapter",
            "backend": "Dx12",
            "device_type": "DiscreteGpu",
            "software_fallback": False,
        }

    monkeypatch.setattr(f3d, "device_probe", probe)

    assert determinism._main(["--out-png", str(tmp_path / "unused.png")]) == 0
    assert requested == ["dx12"]
    assert '"backend": "Dx12"' in capsys.readouterr().out


@pytest.mark.skipif(sys.platform != "win32", reason="requires local DX12 and Vulkan")
def test_device_probe_reports_initialized_render_adapter(monkeypatch, tmp_path):
    """A post-render probe must not enumerate a different same-backend adapter."""
    monkeypatch.setenv("FORGE3D_DETERMINISTIC", "1")
    monkeypatch.setenv("WGPU_BACKENDS", "dx12")
    determinism._render_reference_inprocess(
        CANONICAL_SCENE, 64, 64, tmp_path / "active-adapter.png"
    )

    import forge3d as f3d

    assert f3d.device_probe("vulkan")["backend"] == "Dx12"
