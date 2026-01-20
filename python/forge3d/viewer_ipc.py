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
import threading
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
    
    # Start background thread to keep draining stdout so the viewer doesn't block
    def _drain_stdout():
        try:
            for line in process.stdout:
                if print_output:
                    print(f"  {line.rstrip()}")
        except Exception:
            pass

    thread = threading.Thread(target=_drain_stdout, daemon=True)
    thread.start()
    
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
    min_zoom: Optional[float] = None,
    max_zoom: Optional[float] = None,
    offset: Optional[Tuple[float, float]] = None,
    rotation: Optional[float] = None,
    underline: Optional[bool] = None,
    small_caps: Optional[bool] = None,
    leader: Optional[bool] = None,
    horizon_fade_angle: Optional[float] = None,
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
        min_zoom: Minimum zoom level for visibility (default: 0)
        max_zoom: Maximum zoom level for visibility (default: inf)
        offset: Screen offset from anchor in pixels (x, y)
        rotation: Rotation angle in radians
        underline: Enable underline style
        small_caps: Enable small caps style
        leader: Show leader line to anchor when offset
        horizon_fade_angle: Angle in degrees for horizon fade (default: 5)
        
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
    if min_zoom is not None:
        cmd["min_zoom"] = min_zoom
    if max_zoom is not None:
        cmd["max_zoom"] = max_zoom
    if offset is not None:
        cmd["offset"] = list(offset)
    if rotation is not None:
        cmd["rotation"] = rotation
    if underline is not None:
        cmd["underline"] = underline
    if small_caps is not None:
        cmd["small_caps"] = small_caps
    if leader is not None:
        cmd["leader"] = leader
    if horizon_fade_angle is not None:
        cmd["horizon_fade_angle"] = horizon_fade_angle
    
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


def add_line_label(
    sock: socket.socket,
    text: str,
    polyline: list,
    size: Optional[float] = None,
    color: Optional[Tuple[float, float, float, float]] = None,
    halo_color: Optional[Tuple[float, float, float, float]] = None,
    halo_width: Optional[float] = None,
    priority: Optional[int] = None,
    placement: str = "center",
    repeat_distance: Optional[float] = None,
    min_zoom: Optional[float] = None,
    max_zoom: Optional[float] = None,
) -> Dict[str, Any]:
    """Add a line label along a polyline.
    
    Args:
        sock: Connected socket to viewer
        text: Label text content
        polyline: List of world positions [(x, y, z), ...]
        size: Font size in pixels (default: 14)
        color: Text color as RGBA tuple (0-1 range)
        halo_color: Halo/outline color as RGBA tuple
        halo_width: Halo width in pixels (0 = no halo)
        priority: Priority for collision (higher = more important)
        placement: "center" or "along" (default: "center")
        repeat_distance: Distance in pixels between repeated labels (0 = no repeat)
        min_zoom: Minimum zoom level for visibility
        max_zoom: Maximum zoom level for visibility
        
    Returns:
        Response dict with 'ok' and optionally 'id' of the created label
    """
    cmd: Dict[str, Any] = {
        "cmd": "add_line_label",
        "text": text,
        "polyline": [list(p) for p in polyline],
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
    if placement:
        cmd["placement"] = placement
    if repeat_distance is not None:
        cmd["repeat_distance"] = repeat_distance
    if min_zoom is not None:
        cmd["min_zoom"] = min_zoom
    if max_zoom is not None:
        cmd["max_zoom"] = max_zoom
    
    return send_ipc(sock, cmd)


def set_label_zoom(sock: socket.socket, zoom: float) -> Dict[str, Any]:
    """Set the current zoom level for scale-dependent label visibility.
    
    Args:
        sock: Connected socket to viewer
        zoom: Current zoom level
        
    Returns:
        Response dict with 'ok' status
    """
    return send_ipc(sock, {"cmd": "set_label_zoom", "zoom": zoom})


def set_max_visible_labels(sock: socket.socket, max_labels: int) -> Dict[str, Any]:
    """Set the maximum number of visible labels.
    
    Args:
        sock: Connected socket to viewer
        max_labels: Maximum number of labels to display
        
    Returns:
        Response dict with 'ok' status
    """
    return send_ipc(sock, {"cmd": "set_max_visible_labels", "max": max_labels})


# === Plan 3: Premium Label Features ===

def add_curved_label(
    sock: socket.socket,
    text: str,
    polyline: list,
    size: Optional[float] = None,
    color: Optional[Tuple[float, float, float, float]] = None,
    halo_color: Optional[Tuple[float, float, float, float]] = None,
    halo_width: Optional[float] = None,
    priority: Optional[int] = None,
    tracking: Optional[float] = None,
    center_on_path: bool = True,
) -> Dict[str, Any]:
    """Add a curved label along a polyline path (Plan 3 feature).
    
    Each glyph is positioned along the path with rotation following
    the path tangent for atlas-style curved text.
    
    Args:
        sock: Connected socket to viewer
        text: Label text content
        polyline: List of world positions [(x, y, z), ...] defining the path
        size: Font size in pixels (default: 14)
        color: Text color as RGBA tuple (0-1 range)
        halo_color: Halo/outline color as RGBA tuple
        halo_width: Halo width in pixels (0 = no halo)
        priority: Priority for collision (higher = more important)
        tracking: Letter-spacing as multiple of font size (0.0 = normal)
        center_on_path: Whether to center text on the path (default: True)
        
    Returns:
        Response dict with 'ok' and optionally 'id' of the created label
    """
    cmd: Dict[str, Any] = {
        "cmd": "add_curved_label",
        "text": text,
        "polyline": [list(p) for p in polyline],
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
    if tracking is not None:
        cmd["tracking"] = tracking
    cmd["center_on_path"] = center_on_path
    
    return send_ipc(sock, cmd)


def add_callout(
    sock: socket.socket,
    text: str,
    anchor: Tuple[float, float, float],
    offset: Tuple[float, float] = (0.0, -50.0),
    background_color: Optional[Tuple[float, float, float, float]] = None,
    border_color: Optional[Tuple[float, float, float, float]] = None,
    border_width: Optional[float] = None,
    corner_radius: Optional[float] = None,
    padding: Optional[float] = None,
    text_size: Optional[float] = None,
    text_color: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    """Add a callout box with pointer at a world position (Plan 3 feature).
    
    Creates a labeled box with rounded corners and a pointer connecting
    to the anchor position.
    
    Args:
        sock: Connected socket to viewer
        text: Callout text content (may be multi-line)
        anchor: World position (x, y, z) for the pointer tip
        offset: Screen offset from anchor (x, y) in pixels
        background_color: Box background color as RGBA
        border_color: Box border color as RGBA
        border_width: Border width in pixels (0 = no border)
        corner_radius: Rounded corner radius in pixels
        padding: Padding inside the box in pixels
        text_size: Font size for text in pixels
        text_color: Text color as RGBA
        
    Returns:
        Response dict with 'ok' and optionally 'id' of the created callout
    """
    cmd: Dict[str, Any] = {
        "cmd": "add_callout",
        "text": text,
        "anchor": list(anchor),
        "offset": list(offset),
    }
    if background_color is not None:
        cmd["background_color"] = list(background_color)
    if border_color is not None:
        cmd["border_color"] = list(border_color)
    if border_width is not None:
        cmd["border_width"] = border_width
    if corner_radius is not None:
        cmd["corner_radius"] = corner_radius
    if padding is not None:
        cmd["padding"] = padding
    if text_size is not None:
        cmd["text_size"] = text_size
    if text_color is not None:
        cmd["text_color"] = list(text_color)
    
    return send_ipc(sock, cmd)


def remove_callout(sock: socket.socket, callout_id: int) -> Dict[str, Any]:
    """Remove a callout by ID.
    
    Args:
        sock: Connected socket to viewer
        callout_id: ID of the callout to remove
        
    Returns:
        Response dict with 'ok' status
    """
    return send_ipc(sock, {"cmd": "remove_callout", "id": callout_id})


def set_label_typography(
    sock: socket.socket,
    tracking: Optional[float] = None,
    kerning: Optional[bool] = None,
    line_height: Optional[float] = None,
    word_spacing: Optional[float] = None,
) -> Dict[str, Any]:
    """Set global typography settings for labels (Plan 3 feature).
    
    Args:
        sock: Connected socket to viewer
        tracking: Letter-spacing as multiple of font size (0.0 = normal)
        kerning: Enable/disable kerning adjustments
        line_height: Line height as multiple of font size (1.0 = single)
        word_spacing: Word spacing as multiple of space width
        
    Returns:
        Response dict with 'ok' status
    """
    cmd: Dict[str, Any] = {"cmd": "set_label_typography"}
    if tracking is not None:
        cmd["tracking"] = tracking
    if kerning is not None:
        cmd["kerning"] = kerning
    if line_height is not None:
        cmd["line_height"] = line_height
    if word_spacing is not None:
        cmd["word_spacing"] = word_spacing
    
    return send_ipc(sock, cmd)


def set_declutter_algorithm(
    sock: socket.socket,
    algorithm: str = "greedy",
    seed: Optional[int] = None,
    max_iterations: Optional[int] = None,
) -> Dict[str, Any]:
    """Set the label declutter algorithm (Plan 3 feature).
    
    Args:
        sock: Connected socket to viewer
        algorithm: "greedy" (fast) or "annealing" (better results)
        seed: Random seed for reproducibility (annealing only)
        max_iterations: Maximum iterations for annealing
        
    Returns:
        Response dict with 'ok' status
    """
    cmd: Dict[str, Any] = {
        "cmd": "set_declutter_algorithm",
        "algorithm": algorithm,
    }
    if seed is not None:
        cmd["seed"] = seed
    if max_iterations is not None:
        cmd["max_iterations"] = max_iterations
    
    return send_ipc(sock, cmd)


# === VECTOR OVERLAY API ===

def add_vector_overlay(
    sock: socket.socket,
    name: str,
    vertices: list[list[float]],  # list of [x, y, z, r, g, b, a]
    indices: list[int],
    primitive: str = "triangles",
    drape: bool = True,
    drape_offset: float = 0.5,
    opacity: float = 1.0,
    depth_bias: float = 0.001,
    line_width: float = 2.0,
    point_size: float = 5.0,
    z_order: int = 0,
) -> Dict[str, Any]:
    """Add a vector overlay with geometry.
    
    Args:
        sock: Connected socket
        name: Unique name for the layer
        vertices: List of vertices, each is [x, y, z, r, g, b, a]
        indices: List of vertex indices
        primitive: "triangles", "lines", "points", etc.
        drape: Whether to drape onto terrain
        drape_offset: Height offset when draped
        opacity: Layer opacity
        depth_bias: Depth bias
        line_width: Width for lines
        point_size: Size for points
        z_order: Draw order
    """
    return send_ipc(sock, {
        "cmd": "add_vector_overlay",
        "name": name,
        "vertices": vertices,
        "indices": indices,
        "primitive": primitive,
        "drape": drape,
        "drape_offset": drape_offset,
        "opacity": opacity,
        "depth_bias": depth_bias,
        "line_width": line_width,
        "point_size": point_size,
        "z_order": z_order,
    })

def set_vector_overlays_enabled(sock: socket.socket, enabled: bool) -> Dict[str, Any]:
    return send_ipc(sock, {"cmd": "set_vector_overlays_enabled", "enabled": enabled})


# === PICKING API (Plan 3) ===

def poll_pick_events(sock: socket.socket) -> Dict[str, Any]:
    """Poll for pending pick events.
    
    Returns a dict with 'pick_events': list of events.
    """
    return send_ipc(sock, {"cmd": "poll_pick_events"})

def set_lasso_mode(sock: socket.socket, enabled: bool) -> Dict[str, Any]:
    """Enable or disable lasso selection mode."""
    return send_ipc(sock, {"cmd": "set_lasso_mode", "enabled": enabled})

def get_lasso_state(sock: socket.socket) -> Dict[str, Any]:
    """Get current lasso state ('inactive', 'drawing', 'complete')."""
    return send_ipc(sock, {"cmd": "get_lasso_state"})

def clear_selection(sock: socket.socket) -> Dict[str, Any]:
    """Clear current selection."""
    return send_ipc(sock, {"cmd": "clear_selection"})


# === OIT (Order-Independent Transparency) API (P0.1/M1) ===

def set_oit_enabled(sock: socket.socket, enabled: bool, mode: str = "auto") -> Dict[str, Any]:
    """Enable or disable Order-Independent Transparency.
    
    OIT provides correct rendering of overlapping transparent surfaces
    (water, volumetrics, vector overlays) without sorting artifacts.
    
    Args:
        sock: Connected socket to viewer
        enabled: Whether to enable OIT
        mode: OIT mode - 'auto' (default), 'wboit', 'dual_source', or 'standard'
              'auto' selects dual-source if hardware supports it, else WBOIT
              'wboit' forces weighted-blended OIT
              'dual_source' forces dual-source blending (may fail on some hardware)
              'standard' disables OIT (uses regular alpha blending)
        
    Returns:
        Response dict with 'ok' status
    """
    return send_ipc(sock, {
        "cmd": "set_oit_enabled",
        "enabled": enabled,
        "mode": mode,
    })


def get_oit_mode(sock: socket.socket) -> Dict[str, Any]:
    """Get current OIT mode.
    
    Returns:
        Response dict with 'mode' key containing current OIT mode string
    """
    return send_ipc(sock, {"cmd": "get_oit_mode"})


# === BUNDLE API ===

def save_bundle(
    sock: socket.socket,
    path: str,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Save current scene state to a bundle.
    
    Args:
        sock: Connected socket to viewer
        path: Output bundle path (will create directory with .forge3d suffix)
        name: Optional bundle name (defaults to filename)
        
    Returns:
        Response dict with 'ok' status and 'path' to created bundle
        
    Example:
        >>> response = save_bundle(sock, "output/my_scene.forge3d", name="Mountain View")
        >>> print(response['path'])
        'output/my_scene.forge3d'
    """
    cmd: Dict[str, Any] = {
        "cmd": "SaveBundle",
        "path": str(path),
    }
    if name is not None:
        cmd["name"] = name
    return send_ipc(sock, cmd)


def load_bundle(
    sock: socket.socket,
    path: str,
) -> Dict[str, Any]:
    """Load a bundle into the viewer.
    
    Args:
        sock: Connected socket to viewer
        path: Path to bundle directory (.forge3d)
        
    Returns:
        Response dict with 'ok' status
        
    Example:
        >>> response = load_bundle(sock, "scenes/my_scene.forge3d")
        >>> if response.get('ok'):
        ...     print("Bundle loaded successfully")
    """
    return send_ipc(sock, {
        "cmd": "LoadBundle",
        "path": str(path),
    })


def poll_pending_bundle_save(sock: socket.socket) -> Dict[str, Any]:
    """Poll for pending bundle save request.
    
    Used by scripts to check if user requested a bundle save via IPC.
    
    Returns:
        Response dict with 'pending' (bool) and optionally 'path' and 'name'
    """
    return send_ipc(sock, {"cmd": "poll_pending_bundle_save"})


def poll_pending_bundle_load(sock: socket.socket) -> Dict[str, Any]:
    """Poll for pending bundle load request.
    
    Used by scripts to check if user requested a bundle load via IPC.
    
    Returns:
        Response dict with 'pending' (bool) and optionally 'path'
    """
    return send_ipc(sock, {"cmd": "poll_pending_bundle_load"})


# === TAA API (P1.3) ===

def set_taa_enabled(sock: socket.socket, enabled: bool) -> Dict[str, Any]:
    """Enable or disable Temporal Anti-Aliasing (TAA).
    
    Args:
        sock: Connected socket to viewer
        enabled: Whether to enable TAA
        
    Returns:
        Response dict with 'ok' status
    """
    return send_ipc(sock, {"cmd": "set_taa_enabled", "enabled": enabled})


def get_taa_status(sock: socket.socket) -> Dict[str, Any]:
    """Get current TAA status.
    
    Returns:
        Response dict with status info
    """
    return send_ipc(sock, {"cmd": "get_taa_status"})


def set_taa_params(
    sock: socket.socket,
    history_weight: Optional[float] = None,
    jitter_scale: Optional[float] = None,
    enable_jitter: Optional[bool] = None,
) -> Dict[str, Any]:
    """Set TAA parameters.
    
    Args:
        sock: Connected socket to viewer
        history_weight: History blending weight (0.0-0.99)
        jitter_scale: Jitter scale factor (1.0 = standard)
        enable_jitter: Explicitly enable/disable jitter
        
    Returns:
        Response dict with 'ok' status
    """
    cmd: Dict[str, Any] = {"cmd": "set_taa_params"}
    if history_weight is not None:
        cmd["history_weight"] = history_weight
    if jitter_scale is not None:
        cmd["jitter_scale"] = jitter_scale
    if enable_jitter is not None:
        cmd["enable_jitter"] = enable_jitter
    return send_ipc(sock, cmd)


def set_terrain_pbr(
    sock: socket.socket,
    enabled: Optional[bool] = None,
    hdr_path: Optional[str] = None,
    ibl_intensity: Optional[float] = None,
    shadow_technique: Optional[str] = None,
    shadow_map_res: Optional[int] = None,
    exposure: Optional[float] = None,
    msaa: Optional[int] = None,
    normal_strength: Optional[float] = None,
    height_ao: Optional[Dict[str, Any]] = None,
    sun_visibility: Optional[Dict[str, Any]] = None,
    materials: Optional[Dict[str, Any]] = None,
    vector_overlay: Optional[Dict[str, Any]] = None,
    tonemap: Optional[Dict[str, Any]] = None,
    lens_effects: Optional[Dict[str, Any]] = None,
    dof: Optional[Dict[str, Any]] = None,
    motion_blur: Optional[Dict[str, Any]] = None,
    volumetrics: Optional[Dict[str, Any]] = None,
    denoise: Optional[Dict[str, Any]] = None,
    sky: Optional[Dict[str, Any]] = None,
    debug_mode: Optional[int] = None,
) -> Dict[str, Any]:
    """Configure terrain PBR/advanced rendering mode settings.
    
    Args:
        sock: Connected socket
        enabled: Enable PBR mode
        hdr_path: Path to HDR environment map
        ibl_intensity: Intensity of IBL lighting
        shadow_technique: Shadow mode ("none", "hard", "pcf", "pcss", "vsm", "evsm")
        shadow_map_res: Resolution of shadow maps
        exposure: Exposure value
        msaa: MSAA sample count (1, 4, 8)
        normal_strength: Normal map strength
        height_ao: Dict of height-based AO settings
        sun_visibility: Dict of sun visibility settings
        materials: Dict of material settings (snow, rock, wetness)
        vector_overlay: Dict of vector overlay settings
        tonemap: Dict of tonemapping settings
        lens_effects: Dict of lens effects settings (vignette, distortion)
        dof: Dict of Depth of Field settings
        motion_blur: Dict of motion blur settings
        volumetrics: Dict of volumetric lighting settings
        denoise: Dict of denoise settings
        sky: Dict of sky settings
        debug_mode: Debug visualization mode
    """
    cmd: Dict[str, Any] = {"cmd": "set_terrain_pbr"}
    if enabled is not None: cmd["enabled"] = enabled
    if hdr_path is not None: cmd["hdr_path"] = hdr_path
    if ibl_intensity is not None: cmd["ibl_intensity"] = ibl_intensity
    if shadow_technique is not None: cmd["shadow_technique"] = shadow_technique
    if shadow_map_res is not None: cmd["shadow_map_res"] = shadow_map_res
    if exposure is not None: cmd["exposure"] = exposure
    if msaa is not None: cmd["msaa"] = msaa
    if normal_strength is not None: cmd["normal_strength"] = normal_strength
    if height_ao is not None: cmd["height_ao"] = height_ao
    if sun_visibility is not None: cmd["sun_visibility"] = sun_visibility
    if materials is not None: cmd["materials"] = materials
    if vector_overlay is not None: cmd["vector_overlay"] = vector_overlay
    if tonemap is not None: cmd["tonemap"] = tonemap
    if lens_effects is not None: cmd["lens_effects"] = lens_effects
    if dof is not None: cmd["dof"] = dof
    if motion_blur is not None: cmd["motion_blur"] = motion_blur
    if volumetrics is not None: cmd["volumetrics"] = volumetrics
    if denoise is not None: cmd["denoise"] = denoise
    if sky is not None: cmd["sky"] = sky
    if debug_mode is not None: cmd["debug_mode"] = debug_mode
    
    return send_ipc(sock, cmd)
