"""
Control functions for the interactive 3D viewer.
"""
from typing import Optional
from . import _forge3d as _native


def set_camera(distance: Optional[float] = None, theta: Optional[float] = None, phi: Optional[float] = None):
    """
    Adjust camera parameters in the active 3D viewer.
    
    Args:
        distance: Camera distance from target (zoom level)
        theta: Horizontal angle in degrees (yaw)
        phi: Vertical angle in degrees (pitch)
    
    Example:
        >>> import forge3d as f3d
        >>> # After opening viewer with --viewer-3d
        >>> f3d.set_camera(distance=2000, theta=45, phi=30)
    """
    _native.viewer_set_camera(distance, theta, phi)


def snapshot(path: str, width: Optional[int] = None, height: Optional[int] = None):
    """
    Capture a snapshot from the active 3D viewer.
    
    Args:
        path: Output PNG file path
        width: Image width (default: current viewer width)
        height: Image height (default: current viewer height)
    
    Example:
        >>> import forge3d as f3d
        >>> # After opening viewer with --viewer-3d
        >>> f3d.snapshot("terrain_view.png", width=1920, height=1080)
    """
    _native.viewer_snapshot(path, width, height)


def export(path: str):
    """
    Export current viewer content to PNG (convenience wrapper for snapshot).
    
    Args:
        path: Output PNG file path
    
    Example:
        >>> import forge3d as f3d
        >>> f3d.export("current_view.png")
    """
    _native.viewer_export(path)


def get_camera():
    """
    Get current camera state from the active 3D viewer.
    
    Returns:
        dict: Camera state with keys:
            - eye: [x, y, z] camera position
            - target: [x, y, z] camera target
            - distance: float, camera distance from target
            - theta: float, horizontal angle in degrees
            - phi: float, vertical angle in degrees
            - fov: float, field of view in degrees
    
    Example:
        >>> import forge3d as f3d
        >>> cam = f3d.get_camera()
        >>> print(f"Camera at {cam['eye']}, looking at {cam['target']}")
        >>> print(f"Angles: theta={cam['theta']:.1f}°, phi={cam['phi']:.1f}°")
    """
    eye, target, distance, theta, phi, fov = _native.viewer_get_camera()
    return {
        'eye': eye.tolist(),
        'target': target.tolist(),
        'distance': distance,
        'theta': theta,
        'phi': phi,
        'fov': fov,
    }
