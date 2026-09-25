"""Per-backend TERRA-DETERMINATA golden record (stdlib only).

The committed golden stores one SHA-256 per wgpu backend together with the
attributable adapter and the build that produced it. A render is compared only
against the golden of its own backend; a backend without a committed golden is
ABSENT (reported, never silently passed). Cross-backend byte identity is a
separate claim recorded under ``cross_backend_identity``; while its status is
ABSENT, dx12 and vulkan hashes are allowed to differ.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "forge3d.determinism.golden/2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_STATES = {"ABSENT", "PROVEN"}


def backend_key(backend: Any) -> str:
    """Normalize a wgpu backend label ("Vulkan", "Dx12", "vulkan") to a key."""
    return str(backend or "").strip().lower()


def load_golden(path: str | Path, scene: str | None = None) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} determinism golden")
    if scene is not None and data.get("scene") != scene:
        raise ValueError(f"{path}: golden is for scene {data.get('scene')!r}, not {scene!r}")
    backends = data.get("backends")
    if not isinstance(backends, Mapping) or not backends:
        raise ValueError(f"{path}: golden has no per-backend entries")
    for key, entry in backends.items():
        if key != backend_key(key):
            raise ValueError(f"{path}: backend key {key!r} must be lowercase")
        if not isinstance(entry, Mapping) or not _SHA256.match(str(entry.get("sha256", ""))):
            raise ValueError(f"{path}: backend {key!r} lacks a sha256")
        adapter = entry.get("adapter")
        if not isinstance(adapter, Mapping) or not adapter.get("name"):
            raise ValueError(f"{path}: backend {key!r} lacks attributable adapter metadata")
        if backend_key(adapter.get("backend")) != key:
            raise ValueError(f"{path}: backend {key!r} adapter reports {adapter.get('backend')!r}")
        if not entry.get("source"):
            raise ValueError(f"{path}: backend {key!r} lacks its provenance source")
    identity = data.get("cross_backend_identity")
    if not isinstance(identity, Mapping) or identity.get("status") not in _IDENTITY_STATES:
        raise ValueError(f"{path}: cross_backend_identity.status must be ABSENT or PROVEN")
    if identity["status"] == "PROVEN" and len({e["sha256"] for e in backends.values()}) != 1:
        raise ValueError(f"{path}: PROVEN cross-backend identity with differing hashes")
    return dict(data)


def golden_sha256(golden: Mapping[str, Any], backend: Any) -> str | None:
    """The committed hash for ``backend``, or None when that backend is ABSENT."""
    entry = golden["backends"].get(backend_key(backend))
    return None if entry is None else str(entry["sha256"])
