# python/forge3d/_degradation.py
# CENSOR Task 10: pure-Python degradation sink, mirroring the Rust one in
# src/core/degradation.rs. Every Python-side fallback / placeholder /
# absent-capability path records a structured entry here; the RenderCertificate
# captures it with the native render's records.
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
_CAPTURE = threading.local()
_LOCK = threading.Lock()


def record(kind: str, name: str, consequence: str) -> None:
    """Record a degradation, deduplicated on ``(kind, name)``.

    The first recorded ``consequence`` for a given ``(kind, name)`` is kept;
    later calls with the same key are ignored (mirrors the Rust sink).
    """
    key = (str(kind), str(name))
    value = str(consequence)
    with _LOCK:
        _SINK.setdefault(key, value)
    active = getattr(_CAPTURE, "active", None)
    if active is not None:
        active.setdefault(key, value)


def snapshot() -> List[Dict[str, str]]:
    """Return all recorded degradations as dicts, sorted by ``(kind, name)``."""
    with _LOCK:
        items = sorted(_SINK.items())
    return [
        {"kind": kind, "name": name, "consequence": consequence}
        for (kind, name), consequence in items
    ]


def begin_capture() -> None:
    """Start a fresh capture on the rendering thread."""
    _CAPTURE.active = {}


def finish_capture() -> List[Tuple[str, str, str]]:
    """Return this render's Python degradations to native certificate capture."""
    active = _suspend_capture()
    return [
        (kind, name, consequence)
        for (kind, name), consequence in (active or {}).items()
    ]


def abort_capture() -> None:
    """Discard the rendering thread's unfinished capture."""
    _CAPTURE.active = None


def _suspend_capture():
    previous = getattr(_CAPTURE, "active", None)
    _CAPTURE.active = None
    return previous


def _restore_capture(previous):
    _CAPTURE.active = previous


def clear() -> None:
    """Reset the process sink and the calling thread's capture."""
    with _LOCK:
        _SINK.clear()
    abort_capture()
