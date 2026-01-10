# python/forge3d/viewer.py
# Viewer control utilities including non-blocking IPC-based viewer workflow.
# Supports Journey 1 (open populated -> interact -> Python updates -> snapshot)
# and Journey 2 (open blank -> build from Python -> snapshot).

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from . import _forge3d  # type: ignore[attr-defined]
except ImportError:
    _forge3d = None

# Import Renderer and MSAA config with fallbacks for standalone testing
try:
    from . import Renderer
    _SUPPORTED_MSAA = {1, 2, 4, 8}  # Standard MSAA sample counts
except ImportError:
    Renderer = None  # type: ignore
    _SUPPORTED_MSAA = {1, 2, 4, 8}


def set_msaa(samples: int) -> int:
    """Set the default MSAA sample count for newly created renderers."""
    if samples not in _SUPPORTED_MSAA:
        raise ValueError(f"Unsupported MSAA sample count: {samples} (allowed: {_SUPPORTED_MSAA})")

    if Renderer is not None:
        Renderer._set_default_msaa(samples)

    if _forge3d is not None:
        setter = getattr(_forge3d, "set_msaa_samples", None)
        if setter is not None:
            try:
                setter(samples)
            except Exception as exc:  # pragma: no cover - optional binding path
                warnings.warn(f"forge3d.set_msaa_samples failed: {exc}")

    return samples


# -----------------------------------------------------------------------------
# Non-blocking viewer via IPC (subprocess + TCP + NDJSON)
# -----------------------------------------------------------------------------

_READY_PATTERN = re.compile(r"FORGE3D_VIEWER_READY port=(\d+)")
_DEFAULT_TIMEOUT = 30.0


class ViewerError(Exception):
    """Error from viewer IPC communication."""
    pass


class ViewerHandle:
    """Handle to a non-blocking interactive viewer subprocess.
    
    The viewer runs as a separate process and communicates via TCP + NDJSON.
    Use `open_viewer_async()` to create a ViewerHandle.
    
    Example:
        >>> from forge3d.viewer import open_viewer_async
        >>> with open_viewer_async(obj_path="model.obj") as v:
        ...     v.set_camera_lookat(eye=[0, 5, 10], target=[0, 0, 0])
        ...     v.snapshot("output.png", width=1920, height=1080)
    """
    
    def __init__(
        self,
        process: subprocess.Popen,
        host: str,
        port: int,
        timeout: float = 10.0,
    ):
        self._process = process
        self._host = host
        self._port = port
        self._timeout = timeout
        self._socket: Optional[socket.socket] = None
        self._connect()
    
    def _connect(self) -> None:
        """Connect to the viewer's IPC server."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(self._timeout)
        self._socket.connect((self._host, self._port))
    
    def _send_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Send a command and wait for response."""
        if self._socket is None:
            raise ViewerError("Not connected to viewer")
        
        # Send NDJSON request
        request = json.dumps(cmd) + "\n"
        self._socket.sendall(request.encode("utf-8"))
        
        # Read response line
        response_data = b""
        while b"\n" not in response_data:
            chunk = self._socket.recv(4096)
            if not chunk:
                raise ViewerError("Connection closed by viewer")
            response_data += chunk
        
        line = response_data.split(b"\n")[0].decode("utf-8")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as e:
            raise ViewerError(f"Invalid JSON response: {e}")
        
        if not response.get("ok", False):
            error_msg = response.get("error", "Unknown error")
            raise ViewerError(f"Viewer command failed: {error_msg}")
        
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get viewer stats (geometry readiness, vertex/index counts).
        
        Returns:
            Dict with keys:
                - vb_ready: bool - whether vertex buffer is ready for drawing
                - vertex_count: int - number of vertices in current mesh
                - index_count: int - number of indices in current mesh
                - scene_has_mesh: bool - whether scene has any mesh loaded
        
        Raises:
            ViewerError: if stats unavailable or command fails
        """
        response = self._send_command({"cmd": "get_stats"})
        stats = response.get("stats")
        if stats is None:
            raise ViewerError("get_stats returned no stats data")
        return stats
    
    def load_obj(self, path: Union[str, Path]) -> None:
        """Load an OBJ file into the viewer."""
        self._send_command({"cmd": "load_obj", "path": str(path)})
    
    def load_gltf(self, path: Union[str, Path]) -> None:
        """Load a glTF/GLB file into the viewer."""
        self._send_command({"cmd": "load_gltf", "path": str(path)})
    
    def set_transform(
        self,
        translation: Optional[Tuple[float, float, float]] = None,
        rotation_quat: Optional[Tuple[float, float, float, float]] = None,
        scale: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Set object transform (translation, rotation quaternion, scale)."""
        cmd: Dict[str, Any] = {"cmd": "set_transform"}
        if translation is not None:
            cmd["translation"] = list(translation)
        if rotation_quat is not None:
            cmd["rotation_quat"] = list(rotation_quat)
        if scale is not None:
            cmd["scale"] = list(scale)
        self._send_command(cmd)
    
    def set_camera_lookat(
        self,
        eye: Tuple[float, float, float],
        target: Tuple[float, float, float],
        up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> None:
        """Set camera position using look-at parameters."""
        self._send_command({
            "cmd": "cam_lookat",
            "eye": list(eye),
            "target": list(target),
            "up": list(up),
        })
    
    def set_fov(self, deg: float) -> None:
        """Set camera field of view in degrees."""
        self._send_command({"cmd": "set_fov", "deg": float(deg)})
    
    def set_sun(self, azimuth_deg: float, elevation_deg: float) -> None:
        """Set sun direction (azimuth and elevation in degrees)."""
        self._send_command({
            "cmd": "lit_sun",
            "azimuth_deg": float(azimuth_deg),
            "elevation_deg": float(elevation_deg),
        })
    
    def set_ibl(self, path: Union[str, Path], intensity: float = 1.0) -> None:
        """Set IBL (environment map) from HDR file."""
        self._send_command({
            "cmd": "lit_ibl",
            "path": str(path),
            "intensity": float(intensity),
        })
    
    def set_z_scale(self, value: float) -> None:
        """Set terrain z-scale (height exaggeration). Only applies to terrain scenes."""
        self._send_command({"cmd": "set_z_scale", "value": float(value)})
    
    def set_orbit_camera(
        self,
        phi_deg: float,
        theta_deg: float,
        radius: float,
        fov_deg: Optional[float] = None,
    ) -> None:
        """Set orbit camera parameters (phi/theta/radius).
        
        Args:
            phi_deg: Azimuth angle in degrees (horizontal rotation)
            theta_deg: Elevation angle in degrees (vertical angle from horizon)
            radius: Distance from target/center
            fov_deg: Optional field of view in degrees
        """
        cmd: Dict[str, Any] = {
            "cmd": "set_terrain_camera",
            "phi_deg": float(phi_deg),
            "theta_deg": float(theta_deg),
            "radius": float(radius),
        }
        if fov_deg is not None:
            cmd["fov_deg"] = float(fov_deg)
        self._send_command(cmd)
    
    def snapshot(
        self,
        path: Union[str, Path],
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """Take a snapshot and save to file. Optionally override resolution."""
        cmd: Dict[str, Any] = {"cmd": "snapshot", "path": str(path)}
        if width is not None:
            cmd["width"] = int(width)
        if height is not None:
            cmd["height"] = int(height)
        self._send_command(cmd)
        # Give viewer time to write the file
        time.sleep(0.5)
    
    def render_animation(
        self,
        animation: "CameraAnimation",
        output_dir: Union[str, Path],
        fps: int = 30,
        width: Optional[int] = None,
        height: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> None:
        """Render animation frames to disk.
        
        Iterates through the animation, setting camera state for each frame,
        rendering, and saving to PNG files.
        
        Args:
            animation: CameraAnimation with keyframes
            output_dir: Directory to save frames (created if doesn't exist)
            fps: Frames per second (determines total frame count)
            width: Output width in pixels (optional, uses viewer size if None)
            height: Output height in pixels (optional, uses viewer size if None)
            progress_callback: Optional callback(frame, total_frames) for progress
        
        Example:
            >>> anim = CameraAnimation()
            >>> anim.add_keyframe(time=0.0, phi=0, theta=45, radius=5000, fov=60)
            >>> anim.add_keyframe(time=5.0, phi=180, theta=30, radius=3000, fov=60)
            >>> viewer.render_animation(anim, "./frames", fps=30)
        """
        from pathlib import Path as P
        
        output_path = P(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        total_frames = animation.get_frame_count(fps)
        
        for frame in range(total_frames):
            # Calculate time for this frame
            t = frame / fps
            
            # Get interpolated camera state
            state = animation.evaluate(t)
            if state is None:
                continue
            
            # Set camera state
            self.set_orbit_camera(
                phi_deg=state.phi_deg,
                theta_deg=state.theta_deg,
                radius=state.radius,
                fov_deg=state.fov_deg,
            )
            
            # Generate frame path
            frame_path = output_path / f"frame_{frame:04d}.png"
            
            # Render and save
            self.snapshot(str(frame_path), width=width, height=height)
            
            # Progress callback
            if progress_callback is not None:
                progress_callback(frame, total_frames)
    
    def close(self) -> None:
        """Close the viewer window and terminate the subprocess."""
        try:
            if self._socket is not None:
                self._send_command({"cmd": "close"})
        except Exception:
            pass
        finally:
            if self._socket is not None:
                try:
                    self._socket.close()
                except Exception:
                    pass
                self._socket = None
            if self._process is not None:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=5.0)
                except Exception:
                    try:
                        self._process.kill()
                    except Exception:
                        pass
    
    def __enter__(self) -> "ViewerHandle":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    @property
    def port(self) -> int:
        """The port the viewer IPC server is listening on."""
        return self._port
    
    @property
    def is_running(self) -> bool:
        """Check if the viewer process is still running."""
        return self._process is not None and self._process.poll() is None


def _find_viewer_binary() -> str:
    """Find the interactive_viewer binary."""
    # Try to find in cargo target directory
    # Path: python/forge3d/viewer.py -> python/forge3d -> python -> forge3d (repo root)
    cargo_target = Path(__file__).parent.parent.parent / "target"
    
    # Check release first, then debug
    for profile in ["release", "debug"]:
        if sys.platform == "win32":
            binary = cargo_target / profile / "interactive_viewer.exe"
        else:
            binary = cargo_target / profile / "interactive_viewer"
        if binary.exists():
            return str(binary)
    
    # Try PATH
    import shutil
    path_binary = shutil.which("interactive_viewer")
    if path_binary:
        return path_binary
    
    raise FileNotFoundError(
        "Could not find interactive_viewer binary. "
        "Build with: cargo build --release --bin interactive_viewer"
    )


def open_viewer_async(
    width: int = 1280,
    height: int = 720,
    title: str = "forge3d Interactive Viewer",
    obj_path: Optional[Union[str, Path]] = None,
    gltf_path: Optional[Union[str, Path]] = None,
    terrain_path: Optional[Union[str, Path]] = None,
    fov_deg: float = 60.0,
    timeout: float = _DEFAULT_TIMEOUT,
    ipc_host: str = "127.0.0.1",
    ipc_port: int = 0,
) -> ViewerHandle:
    """Open an interactive viewer in a subprocess with IPC control.
    
    Returns immediately with a ViewerHandle that can be used to send commands
    to the viewer while keeping the Python REPL available.
    
    Args:
        width: Window width in pixels
        height: Window height in pixels
        title: Window title
        obj_path: Optional OBJ file to load on startup
        gltf_path: Optional glTF file to load on startup (mutually exclusive)
        terrain_path: Optional DEM file to load as terrain (mutually exclusive)
        fov_deg: Initial field of view in degrees
        timeout: Timeout for startup and commands in seconds
        ipc_host: Host for IPC server (default: 127.0.0.1)
        ipc_port: Port for IPC server (0 = auto-select)
    
    Returns:
        ViewerHandle for controlling the viewer
    
    Example:
        >>> from forge3d.viewer import open_viewer_async
        >>> v = open_viewer_async(obj_path="model.obj")
        >>> v.set_camera_lookat(eye=[0, 5, 10], target=[0, 0, 0])
        >>> v.snapshot("output.png", width=3840, height=2160)
        >>> v.close()
    """
    if sum(x is not None for x in (obj_path, gltf_path, terrain_path)) > 1:
        raise ValueError("obj_path, gltf_path, and terrain_path are mutually exclusive")
    
    binary = _find_viewer_binary()
    
    # Build command line
    cmd = [
        binary,
        "--ipc-host", ipc_host,
        "--ipc-port", str(ipc_port),
        "--size", f"{width}x{height}",
        "--fov", str(fov_deg),
    ]
    
    if obj_path is not None:
        cmd.extend(["--obj", str(obj_path)])
    elif gltf_path is not None:
        cmd.extend(["--gltf", str(gltf_path)])
    elif terrain_path is not None:
        cmd.extend(["--terrain", str(terrain_path)])
    
    # Start subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Wait for READY line
    start_time = time.time()
    actual_port: Optional[int] = None
    
    while time.time() - start_time < timeout:
        if process.poll() is not None:
            # Process exited
            output = process.stdout.read() if process.stdout else ""
            raise ViewerError(f"Viewer process exited unexpectedly: {output}")
        
        line = process.stdout.readline() if process.stdout else ""
        if not line:
            time.sleep(0.01)
            continue
        
        match = _READY_PATTERN.search(line)
        if match:
            actual_port = int(match.group(1))
            break
    
    if actual_port is None:
        process.terminate()
        raise ViewerError(f"Timed out waiting for viewer READY line after {timeout}s")
    
    return ViewerHandle(process, ipc_host, actual_port, timeout=timeout)


def open_viewer(
    width: int = 1280,
    height: int = 720,
    title: str = "forge3d Interactive Viewer",
    obj_path: Optional[Union[str, Path]] = None,
    gltf_path: Optional[Union[str, Path]] = None,
    fov_deg: float = 60.0,
    snapshot_path: Optional[str] = None,
    snapshot_width: Optional[int] = None,
    snapshot_height: Optional[int] = None,
    initial_commands: Optional[list[str]] = None,
) -> int:
    """Open an interactive viewer and block until window is closed.
    
    This is a blocking function that launches the viewer as a subprocess
    and waits for it to exit. For non-blocking control, use open_viewer_async().
    
    Args:
        width: Window width in pixels
        height: Window height in pixels
        title: Window title
        obj_path: Optional OBJ file to load on startup
        gltf_path: Optional glTF file to load on startup
        fov_deg: Initial field of view in degrees
        snapshot_path: Optional path for automatic snapshot after startup
        snapshot_width: Snapshot width (requires snapshot_path)
        snapshot_height: Snapshot height (requires snapshot_path)
        initial_commands: List of initial commands to run (e.g., [":gi gtao on"])
    
    Returns:
        Exit code from viewer process
    
    Example:
        >>> import forge3d as f3d
        >>> f3d.open_viewer(obj_path="model.obj", width=1280, height=720)
    """
    if obj_path is not None and gltf_path is not None:
        raise ValueError("obj_path and gltf_path are mutually exclusive")
    
    binary = _find_viewer_binary()
    
    # Build command line
    cmd = [
        binary,
        "--size", f"{width}x{height}",
        "--fov", str(fov_deg),
    ]
    
    if obj_path is not None:
        cmd.extend(["--obj", str(obj_path)])
    if gltf_path is not None:
        cmd.extend(["--gltf", str(gltf_path)])
    if snapshot_path is not None:
        cmd.extend(["--snapshot", snapshot_path])
        if snapshot_width is not None and snapshot_height is not None:
            cmd.extend(["--snapshot-size", f"{snapshot_width}x{snapshot_height}"])
    
    # Start subprocess and wait for completion (blocking)
    process = subprocess.Popen(
        cmd,
        stdout=None,  # Inherit stdout
        stderr=None,  # Inherit stderr
    )
    
    # If initial commands are provided, we need IPC
    if initial_commands:
        # For initial commands, use IPC mode
        process.terminate()
        process.wait()
        
        # Restart with IPC
        handle = open_viewer_async(
            width=width,
            height=height,
            title=title,
            obj_path=obj_path,
            gltf_path=gltf_path,
            fov_deg=fov_deg,
        )
        
        # Send initial commands
        for cmd_str in initial_commands:
            # Parse command string (e.g., ":gi gtao on")
            if cmd_str.startswith(":"):
                cmd_str = cmd_str[1:]  # Remove leading colon
            # For now, just print the command - the viewer terminal handles these
            print(f"[init] {cmd_str}")
        
        # Take snapshot if requested
        if snapshot_path is not None:
            sw = snapshot_width or width
            sh = snapshot_height or height
            handle.snapshot(snapshot_path, width=sw, height=sh)
        
        # Wait for viewer to close
        return handle._process.wait()
    
    return process.wait()
