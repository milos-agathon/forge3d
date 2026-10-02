from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from forge3d._png import load_png_rgba
from forge3d.helpers.offscreen import save_png_deterministic


RESULT_KEYS = (
    "normal_lighting_ssim",
    "family_residency_budget",
    "missing_family_fatal",
    "partial_normal_residency",
)
RENDER_MEMORY_SOURCE = "forge3d.diagnostics.render_certificate.allocations"
READBACK_STAGING_LABEL = "forge3d-readback-staging"


@contextmanager
def captured_render_memory(samples: list[dict[str, int]]) -> Iterator[None]:
    """Record one render *and its readbacks* as a single certificate capture.

    The native render capture closes before ``Frame.to_numpy()`` allocates its
    host-visible staging buffer, so the body (render + readback) runs inside an
    outer capture. The nested renderer capture joins it with the renderer's
    owned allocations; every allocation made in the body is added, while
    unrelated live allocations from other tests are not.
    """
    from forge3d._native import get_native_module
    from forge3d.certificate import _render_capture

    native = get_native_module()
    if native is None or not hasattr(native, "begin_render_execution_capture"):
        raise AssertionError("render memory evidence requires native certificate capture")
    with _render_capture("substratia.render_and_readback", "substratia.readback"):
        yield
    samples.append(render_memory_sample())


def render_memory_sample() -> dict[str, int]:
    """Return tracked-allocation peaks of the LAST completed certificate capture.

    ``total_tracked_bytes`` is host-visible peak + device-local peak: device-
    local memory never appears in the host-visible number, and the certificate
    does not expose a simultaneous total, so this sum bounds it from above.
    ``readback_staging_bytes`` proves the capture spanned the frame readback.
    """
    from forge3d.diagnostics import render_certificate

    allocations = render_certificate(sign=False).get("allocations")
    if not isinstance(allocations, dict):
        raise AssertionError("render certificate has no allocations ledger")
    sample = {}
    for key in ("peak_host_visible_bytes", "peak_device_local_bytes"):
        value = allocations.get(key)
        if type(value) is not int or value < 0:
            raise AssertionError(f"render certificate {key} is not a byte count: {value!r}")
        sample[key] = value
    sample["total_tracked_bytes"] = (
        sample["peak_host_visible_bytes"] + sample["peak_device_local_bytes"]
    )
    by_label = allocations.get("by_label")
    readback = by_label.get(READBACK_STAGING_LABEL) if isinstance(by_label, dict) else None
    if type(readback) is not int or readback <= 0:
        raise AssertionError(
            f"render certificate does not include the {READBACK_STAGING_LABEL!r} readback"
        )
    sample["readback_staging_bytes"] = readback
    return sample


def summarize_render_memory(
    samples: list[dict[str, int]], limit_bytes: int
) -> dict[str, Any]:
    """Assert the per-render host-visible peak ceiling and build gate evidence."""
    if not samples:
        raise AssertionError("no completed SUBSTRATIA render recorded allocation peaks")
    renders = [dict(sample) for sample in samples]
    peak = max(render["peak_host_visible_bytes"] for render in renders)
    total = max(render["total_tracked_bytes"] for render in renders)
    assert peak <= limit_bytes, (
        f"per-render host-visible peak {peak} B exceeds the {limit_bytes} B ceiling"
    )
    return {
        "source": RENDER_MEMORY_SOURCE,
        "renders": renders,
        "peak_host_visible_bytes": peak,
        "total_tracked_bytes": total,
        "memory_limit_bytes": limit_bytes,
    }


def _artifact_dir() -> Path | None:
    value = os.environ.get("FORGE3D_SUBSTRATIA_ARTIFACT_DIR")
    if not value:
        return None
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_substratia_result(name: str, values: Mapping[str, Any]) -> None:
    """Print and optionally persist one deterministic SUBSTRATIA gate result."""
    payload = {"gate": name, **dict(values)}
    print(f"SUBSTRATIA_RESULT {json.dumps(payload, sort_keys=True)}")
    artifact_dir = _artifact_dir()
    if artifact_dir is None:
        return
    ledger_path = artifact_dir / "results.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not isinstance(ledger, dict):
            raise ValueError("SUBSTRATIA results ledger must be a JSON object")
    else:
        ledger = {
            "schema": "forge3d.substratia.results.v1",
            "candidate_sha": os.environ.get("FORGE3D_SUBSTRATIA_CANDIDATE_SHA", ""),
            "gates": {},
        }
    if ledger.get("schema") != "forge3d.substratia.results.v1":
        raise ValueError("SUBSTRATIA results ledger has an unexpected schema")
    gates = ledger.setdefault("gates", {})
    if not isinstance(gates, dict):
        raise ValueError("SUBSTRATIA results ledger gates must be a JSON object")
    gates[name] = dict(values)
    ledger["gates"] = {key: gates[key] for key in sorted(gates)}
    temporary = ledger_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(ledger_path)


def record_substratia_image(filename: str, image: np.ndarray) -> Path | None:
    artifact_dir = _artifact_dir()
    if artifact_dir is None:
        return None
    path = artifact_dir / filename
    rgba = np.asarray(image)
    if rgba.dtype != np.uint8:
        rgba = np.clip(np.rint(rgba), 0, 255).astype(np.uint8)
    save_png_deterministic(path, np.ascontiguousarray(rgba))
    return path


def load_golden(path: Path) -> np.ndarray:
    if not path.is_file():
        raise AssertionError(
            f"required committed SUBSTRATIA golden is missing: {path}"
        )
    return np.asarray(load_png_rgba(path), dtype=np.uint8)


def image_sha256(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
