"""SIDERA night-sky golden — DoD 6.

The committed frame must be byte-identical across two in-process renders, across
two *separate processes* on a pinned backend, and equal to the committed PNG and
its SHA-256 sidecar in ``tests/goldens/determinism/`` — the same inventory and
the same zero-byte tolerance the TERRA-DETERMINATA harness
(``tests/test_determinism_hash.py``) applies to the canonical terrain render.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import forge3d as f3d
from forge3d import _forge3d as native
from forge3d.diagnostics import render_certificate

if not f3d.has_gpu():
    pytest.skip(
        "the SIDERA night golden is a GPU render; no adapter present",
        allow_module_level=True,
    )


def _adapter_is_hardware() -> bool:
    """True only for a non-software adapter.

    ``has_gpu()`` is satisfied by WARP/lavapipe, which rasterise the blended
    sub-pixel star quads differently from real hardware. Following the
    ANAMNESIS rule in ``AGENTS.md``, a byte-exact comparison against a golden
    produced on one physical device is only claimed where the adapter proves it
    is not a software rasteriser; elsewhere the two-run and cross-process
    reproducibility claims still apply, but the committed-bytes claim is ABSENT.
    """
    try:
        probe = f3d.device_probe(os.environ.get("WGPU_BACKEND"))
    except Exception:  # pragma: no cover - probe failure means "cannot prove"
        return False
    if not isinstance(probe, dict):
        return False
    if bool(probe.get("software_fallback", False)):
        return False
    device_type = str(probe.get("device_type", "")).lower()
    name = str(probe.get("name", "")).lower()
    software_markers = ("cpu", "software", "warp", "lavapipe", "llvmpipe", "swiftshader")
    if any(marker in name for marker in software_markers):
        return False
    return device_type in {"discretegpu", "integratedgpu", "discrete_gpu", "integrated_gpu"}


HARDWARE_ADAPTER = _adapter_is_hardware()
requires_hardware = pytest.mark.skipif(
    not HARDWARE_ADAPTER,
    reason="byte-exact golden equality is only claimed on a proven non-software adapter",
)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "sidera_night.png"
#: Membership in the determinism harness' golden inventory.
GOLDEN_SHA = ROOT / "tests" / "goldens" / "determinism" / "sidera_night.sha256"
UPDATE_ENV = "FORGE3D_UPDATE_SIDERA_GOLDEN"


def _update_goldens_enabled() -> bool:
    """Read the refresh flag at *call* time.

    Binding this at import defeats per-test ``monkeypatch.delenv`` and has
    burned this repo before — see the CENSOR F-02 note in ``AGENTS.md``.
    """
    return os.environ.get(UPDATE_ENV) == "1"


def _determinism_backend() -> str:
    explicit = os.environ.get("FORGE3D_DETERMINISM_TEST_BACKEND")
    if explicit:
        return explicit
    if sys.platform == "win32":
        return "dx12"
    if sys.platform == "darwin":
        return "metal"
    return "vulkan"


def _render_in_subprocess(destination: Path) -> str:
    """Render the night golden in a fresh, backend-pinned process; return its SHA."""
    env = dict(os.environ)
    env.update(FORGE3D_DETERMINISTIC="1", WGPU_BACKENDS=_determinism_backend())
    env.pop(UPDATE_ENV, None)
    script = (
        "from forge3d import _forge3d as native;"
        f"native._astro_night_golden_frame().save({str(destination)!r})"
    )
    subprocess.run(
        [sys.executable, "-c", script], env=env, check=True, capture_output=True, text=True
    )
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def _components(mask: np.ndarray) -> list[int]:
    pending = set(map(tuple, np.argwhere(mask)))
    sizes = []
    while pending:
        stack = [pending.pop()]
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in pending:
                    pending.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return sizes


def test_night_sky_render_is_repeatable_and_certified(tmp_path):
    """Two in-process renders, zero byte tolerance, plus the certificate shape.

    Deliberately does NOT compare against the committed golden: the exact bytes
    depend on which backend wgpu selected for this process, and a test cannot
    pin that after the GPU context has been created. The committed-bytes claim
    therefore lives in the backend-pinned subprocess test below, which is how
    ``tests/test_determinism_hash.py`` makes the same kind of claim.
    """
    first = native._astro_night_golden_frame(certificate=True)
    second = native._astro_night_golden_frame(certificate=True)
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first.save(str(first_path))
    second.save(str(second_path))
    assert first_path.read_bytes() == second_path.read_bytes()

    rgba = np.asarray(first.to_numpy())
    assert rgba.shape == (512, 768, 4)
    # Moon + planets + stars: at least three lit, multi-pixel components.
    assert len([size for size in _components(rgba[..., :3].max(axis=2) > 24) if size >= 2]) >= 3

    certificate = render_certificate(sign=False)
    labels = [entry["label"] for entry in certificate["passes"]]
    assert labels == [
        "astro.twilight.sidera_civil_to_astronomical_smoothstep",
        "astro.moonlight.krisciunas_schaefer_1991",
        "astro.night.overlay",
    ]
    assert set(certificate["engine"]["wgsl_module_hashes"]) == {"astro.night.shader"}
    # `finish_render_capture` re-derives a `capability_absent` degradation for
    # every negotiated feature the device did not grant, and records
    # `timing_unavailable` when a timestamp query resolves invalid. Neither is
    # a SIDERA defect, so gate on the *kinds* the way the rest of the repo does
    # rather than demanding an empty list only an RTX-class device produces.
    granted = set(certificate["capabilities"]["granted"])
    for degradation in certificate["degradations"]:
        assert degradation["kind"] in {"capability_absent", "timing_unavailable"}, degradation
        if degradation["kind"] == "capability_absent":
            assert degradation["name"] not in granted, degradation
        else:
            assert degradation["name"] == "astro.night.overlay", degradation


@requires_hardware
def test_night_golden_matches_committed_bytes_on_a_pinned_backend(tmp_path):
    """DoD 6: two *processes* on a pinned backend, byte-identical to the golden.

    This is the SIDERA member of the ``tests/test_determinism_hash.py`` family:
    same zero-byte tolerance, same backend pin, same committed SHA-256 sidecar.
    """
    first_png = tmp_path / "process_a.png"
    first = _render_in_subprocess(first_png)
    second = _render_in_subprocess(tmp_path / "process_b.png")
    assert first == second, (
        "cross-process nondeterminism in the SIDERA night render on "
        f"{_determinism_backend()}\n  first:  {first}\n  second: {second}"
    )

    if _update_goldens_enabled():
        # Refreshing happens here, from the pinned-backend render, so the
        # committed bytes can never come from whatever backend the ambient
        # test process happened to pick.
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_bytes(first_png.read_bytes())
        GOLDEN_SHA.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_SHA.write_text(f"{first}\n", encoding="ascii")

    committed = hashlib.sha256(GOLDEN.read_bytes()).hexdigest()
    assert first == committed, (
        "night golden hash mismatch against the committed PNG\n"
        f"  golden: {committed}\n  actual: {first}\n"
        f"Zero-byte tolerance: regenerate deliberately with {UPDATE_ENV}=1 and "
        f"say why in the commit message."
    )
    assert GOLDEN_SHA.exists(), f"missing determinism sidecar {GOLDEN_SHA}"
    assert GOLDEN_SHA.read_text(encoding="ascii").split()[0].strip() == committed
    assert np.array_equal(f3d.png_to_numpy(first_png), f3d.png_to_numpy(GOLDEN))


@requires_hardware
def test_golden_refresh_does_not_rewrite_the_committed_file_when_disabled(
    tmp_path, monkeypatch
):
    """Negative control for the refresh flag.

    Simulates a refresh run and then removes the flag; the committed golden and
    its sidecar must be untouched. If ``_update_goldens_enabled`` were bound at
    import, this ``delenv`` would be dead code and a corrupted frame could be
    copied over the golden during a refresh sweep.
    """
    if not HARDWARE_ADAPTER:
        pytest.skip("byte-exact golden equality requires a proven hardware adapter")
    before_png = GOLDEN.read_bytes()
    before_sha = GOLDEN_SHA.read_bytes()
    monkeypatch.setenv(UPDATE_ENV, "1")
    assert _update_goldens_enabled() is True
    monkeypatch.delenv(UPDATE_ENV)
    assert _update_goldens_enabled() is False
    test_night_golden_matches_committed_bytes_on_a_pinned_backend(tmp_path)
    assert GOLDEN.read_bytes() == before_png
    assert GOLDEN_SHA.read_bytes() == before_sha
