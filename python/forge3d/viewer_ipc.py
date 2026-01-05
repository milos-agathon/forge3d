"""Standalone IPC utilities for terrain viewer examples.

This module provides low-level IPC functions for scripts that need direct
socket control rather than the higher-level ViewerHandle class.

For most use cases, prefer `forge3d.viewer.open_viewer_async()` which provides
a cleaner API. This module is for terrain viewer examples that need custom
interactive loops and direct socket access.
"""

from __future__ import annotations

import json
import platform
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_READY_PATTERN = re.compile(r"FORGE3D_VIEWER_READY\s+port=(\d+)")


def find_viewer_binary() -> str:
    """Find the interactive_viewer binary.
    
    Searches in cargo target/release and target/debug directories.
    
    Returns:
        Path to the interactive_viewer binary
        
    Raises:
        FileNotFoundError: if binary not found
    """
    ext = ".exe" if platform.system() == "Windows" else ""
    
    # Try relative to this file (python/forge3d/viewer_ipc.py -> repo root)
    repo_root = Path(__file__).parent.parent.parent
    candidates = [
        repo_root / "target" / "release" / f"interactive_viewer{ext}",
        repo_root / "target" / "debug" / f"interactive_viewer{ext}",
    ]
    
    for c in candidates:
        if c.exists():
            return str(c)
    
    raise FileNotFoundError(
        "interactive_viewer binary not found. "
        "Build with: cargo build --release --bin interactive_viewer"
    )


def send_ipc(sock: socket.socket, cmd: Dict[str, Any]) -> Dict[str, Any]:
    """Send an IPC command and receive response.
    
    This is a robust implementation that handles:
    - Large message payloads (logs size for debugging)
    - Timeout handling
    - JSON parsing errors
    
    Args:
        sock: Connected socket to viewer
        cmd: Command dictionary to send
        
    Returns:
        Response dictionary from viewer
    """
    msg = json.dumps(cmd) + "\n"
    msg_bytes = msg.encode()
    
    # Log message size for debugging large payloads
    msg_size = len(msg_bytes)
    if msg_size > 100000:
        print(f"  Sending large IPC message: {msg_size / 1024 / 1024:.1f} MB")
    
    try:
        sock.sendall(msg_bytes)
    except Exception as e:
        return {"ok": False, "error": f"Send failed: {e}"}
    
    data = b""
    while True:
        try:
            chunk = sock.recv(8192)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        except socket.timeout:
            if not data:
                return {"ok": False, "error": "Timeout waiting for response"}
            break
        except Exception as e:
            return {"ok": False, "error": f"Receive failed: {e}"}
    
    line = data.decode().strip()
    if not line:
        return {"ok": False, "error": "Empty response from viewer"}
    
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"Invalid JSON response: {e}"}


def launch_viewer(
    width: int = 1280,
    height: int = 720,
    timeout: float = 30.0,
    print_output: bool = True,
) -> Tuple[subprocess.Popen, int, socket.socket]:
    """Launch viewer subprocess and connect via IPC.
    
    This is a convenience function that combines:
    1. Finding the viewer binary
    2. Launching with IPC enabled
    3. Waiting for READY message
    4. Connecting socket
    
    Args:
        width: Window width
        height: Window height
        timeout: Timeout for viewer startup in seconds
        print_output: Whether to print viewer output lines
        
    Returns:
        Tuple of (process, port, connected_socket)
        
    Raises:
        FileNotFoundError: if viewer binary not found
        RuntimeError: if viewer fails to start or times out
    """
    binary = find_viewer_binary()
    
    cmd = [binary, "--ipc-port", "0", "--size", f"{width}x{height}"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Wait for READY message
    port = None
    start = time.time()
    
    while time.time() - start < timeout:
        if process.poll() is not None:
            raise RuntimeError("Viewer exited unexpectedly")
        
        line = process.stdout.readline()
        if line:
            if print_output:
                print(f"  {line.rstrip()}")
            match = _READY_PATTERN.search(line)
            if match:
                port = int(match.group(1))
                break
    
    if port is None:
        process.terminate()
        raise RuntimeError(f"Timeout waiting for viewer after {timeout}s")
    
    # Connect socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", port))
    sock.settimeout(30.0)
    
    return process, port, sock


def close_viewer(
    sock: socket.socket,
    process: subprocess.Popen,
    timeout: float = 5.0,
) -> None:
    """Gracefully close viewer connection and process.
    
    Args:
        sock: Connected socket
        process: Viewer subprocess
        timeout: Timeout for process termination
    """
    try:
        send_ipc(sock, {"cmd": "close"})
    except Exception:
        pass
    
    try:
        sock.close()
    except Exception:
        pass
    
    try:
        process.terminate()
        process.wait(timeout=timeout)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


# === LABEL API ===

def add_label(
    sock: socket.socket,
    text: str,
    world_pos: Tuple[float, float, float],
    size: Optional[float] = None,
    color: Optional[Tuple[float, float, float, float]] = None,
    halo_color: Optional[Tuple[float, float, float, float]] = None,
    halo_width: Optional[float] = None,
    priority: Optional[int] = None,
) -> Dict[str, Any]:
    """Add a text label at a world position.
    
    Args:
        sock: Connected socket to viewer
        text: Label text content
        world_pos: World position (x, y, z)
        size: Font size in pixels (default: 14)
        color: Text color as RGBA tuple (0-1 range)
        halo_color: Halo/outline color as RGBA tuple
        halo_width: Halo width in pixels (0 = no halo)
        priority: Priority for collision (higher = more important)
        
    Returns:
        Response dict with 'ok' and optionally 'id' of the created label
    """
    cmd: Dict[str, Any] = {
        "cmd": "add_label",
        "text": text,
        "world_pos": list(world_pos),
    }
    if size is not None:
        cmd["size"] = size
    if color is not None:
        cmd["color"] = list(color)
    if halo_color is not None:
        cmd["halo_color"] = list(halo_color)
    if halo_width is not None:
        cmd["halo_width"] = halo_width
    if priority is not None:
        cmd["priority"] = priority
    
    return send_ipc(sock, cmd)


def remove_label(sock: socket.socket, label_id: int) -> Dict[str, Any]:
    """Remove a label by ID.
    
    Args:
        sock: Connected socket to viewer
        label_id: ID of the label to remove
        
    Returns:
        Response dict with 'ok' status
    """
    return send_ipc(sock, {"cmd": "remove_label", "id": label_id})


def clear_labels(sock: socket.socket) -> Dict[str, Any]:
    """Clear all labels.
    
    Args:
        sock: Connected socket to viewer
        
    Returns:
        Response dict with 'ok' status
    """
    return send_ipc(sock, {"cmd": "clear_labels"})


def set_labels_enabled(sock: socket.socket, enabled: bool) -> Dict[str, Any]:
    """Enable or disable label rendering.
    
    Args:
        sock: Connected socket to viewer
        enabled: Whether labels should be rendered
        
    Returns:
        Response dict with 'ok' status
    """
    return send_ipc(sock, {"cmd": "set_labels_enabled", "enabled": enabled})


def load_label_atlas(
    sock: socket.socket,
    atlas_png_path: str,
    metrics_json_path: str,
) -> Dict[str, Any]:
    """Load a font atlas for label rendering.
    
    Args:
        sock: Connected socket to viewer
        atlas_png_path: Path to the MSDF atlas PNG image
        metrics_json_path: Path to the glyph metrics JSON file
        
    Returns:
        Response dict with 'ok' status
    """
    return send_ipc(sock, {
        "cmd": "load_label_atlas",
        "atlas_png_path": atlas_png_path,
        "metrics_json_path": metrics_json_path,
    })
