# python/forge3d/_degradation.py
# CENSOR Task 10: pure-Python degradation sink, mirroring the Rust one in
# src/core/degradation.rs. Every Python-side fallback / placeholder /
# absent-capability path records a structured entry here; the RenderCertificate
# (forge3d.diagnostics.render_certificate) drains it and merges it with the
# native sink.
# RELEVANT FILES: src/core/degradation.rs, python/forge3d/diagnostics.py,
# python/forge3d/certificate.py

"""Process-global degradation sink (Python side).

Byte-for-byte behavioural twin of the Rust sink in
``src/core/degradation.rs``: entries are deduplicated on ``(kind, name)`` so
per-frame fallbacks do not flood the certificate, and :func:`snapshot` returns
them sorted by ``(kind, name)``.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Tuple

# (kind, name) -> consequence. First writer for a given (kind, name) wins, so
# the recorded consequence is stable regardless of how many times a per-frame
# fallback fires.
_SINK: Dict[Tuple[str, str], str] = {}
_LOCK = threading.Lock()


def record(kind: str, name: str, consequence: str) -> None:
    """Record a degradation, deduplicated on ``(kind, name)``.

    The first recorded ``consequence`` for a given ``(kind, name)`` is kept;
    later calls with the same key are ignored (mirrors the Rust sink).
    """
    key = (str(kind), str(name))
    with _LOCK:
        if key not in _SINK:
            _SINK[key] = str(consequence)


def snapshot() -> List[Dict[str, str]]:
    """Return all recorded degradations as dicts, sorted by ``(kind, name)``."""
    with _LOCK:
        items = sorted(_SINK.items())
    return [
        {"kind": kind, "name": name, "consequence": consequence}
        for (kind, name), consequence in items
    ]


def clear() -> None:
    """Reset the sink. Exposed so tests and callers can isolate renders."""
    with _LOCK:
        _SINK.clear()
