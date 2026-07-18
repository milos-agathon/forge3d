"""Loopback-server helpers for tests that need local HTTP fixtures."""

from __future__ import annotations

import errno
from typing import TypeVar

import pytest

ServerT = TypeVar("ServerT")


def bind_loopback_or_skip(server_cls: type[ServerT], handler_cls: type) -> ServerT:
    """Create a loopback server, or skip when the host forbids local sockets."""
    try:
        return server_cls(("127.0.0.1", 0), handler_cls)
    except OSError as exc:
        if isinstance(exc, PermissionError) or exc.errno in {errno.EACCES, errno.EPERM}:
            pytest.skip(f"loopback TCP bind unavailable in this environment: {exc}")
        raise
