"""Shared socket transport for the two viewer command APIs.

Each call sends one NDJSON command and consumes one reply. Unread bytes belong
to the socket, so callers can switch APIs without losing buffered reply lines.
Only one request may be outstanding on a socket; this is not a batching API.
"""

from __future__ import annotations

import json
import socket
from typing import Any, Dict, Optional, Type
from weakref import WeakKeyDictionary


# Values must not retain their socket: unused connections and their buffers
# should be collectible, including sockets closed directly by raw callers.
_receive_buffers: WeakKeyDictionary[socket.socket, bytearray] = WeakKeyDictionary()


def send_command(
    sock: socket.socket,
    cmd: Dict[str, Any],
    *,
    error_type: Optional[Type[Exception]] = None,
) -> Dict[str, Any]:
    """Exchange one command using the caller's existing failure convention.

    With ``error_type``, EOF and invalid JSON raise that exception, while socket
    failures propagate unchanged. Without it, connection and JSON failures return
    error dictionaries and partial replies at EOF/timeout are parsed, as in the
    legacy raw helper. Invalid UTF-8 still raises on both paths. The connected
    socket's timeout is never changed.
    """
    payload = (json.dumps(cmd) + "\n").encode("utf-8")
    if error_type is None and len(payload) > 100000:
        print(f"  Sending large IPC message: {len(payload) / 1024 / 1024:.1f} MB")

    try:
        sock.sendall(payload)
    except Exception as exc:
        if error_type is not None:
            raise
        return {"ok": False, "error": f"Send failed: {exc}"}

    data = _receive_buffers.setdefault(sock, bytearray())
    while b"\n" not in data:
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            if error_type is not None:
                data.clear()
                raise
            if not data:
                return {"ok": False, "error": "Timeout waiting for response"}
            break
        except Exception as exc:
            data.clear()
            if error_type is not None:
                raise
            return {"ok": False, "error": f"Receive failed: {exc}"}

        if not chunk:
            if error_type is not None:
                data.clear()
                raise error_type("Connection closed by viewer")
            break
        data.extend(chunk)

    while True:
        line, _, remaining = data.partition(b"\n")
        data[:] = remaining
        decoded = line.decode("utf-8")
        if error_type is None:
            decoded = decoded.strip()
            # The raw helper historically stripped leading whitespace from the
            # whole received buffer. Skip buffered blank lines without reading
            # ahead or consuming another non-empty reply.
            if not decoded and data:
                continue
            if not decoded:
                return {"ok": False, "error": "Empty response from viewer"}
        break

    try:
        return json.loads(decoded)
    except json.JSONDecodeError as exc:
        message = f"Invalid JSON response: {exc}"
        if error_type is not None:
            raise error_type(message) from exc
        return {"ok": False, "error": message}


def forget_connection(sock: socket.socket) -> None:
    """Release unread data when an API explicitly closes its connection."""
    _receive_buffers.pop(sock, None)
