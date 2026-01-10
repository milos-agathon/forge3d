# python/forge3d/animation.py
"""
Camera animation module for keyframe-based camera paths.

Provides:
    CameraAnimation     - Keyframe animation with cubic interpolation
    CameraState         - Interpolated camera state at a given time

Example:
    >>> from forge3d.animation import CameraAnimation
    >>> anim = CameraAnimation()
    >>> anim.add_keyframe(time=0.0, phi=0, theta=45, radius=5000, fov=60)
    >>> anim.add_keyframe(time=5.0, phi=180, theta=30, radius=3000, fov=60)
    >>> state = anim.evaluate(2.5)
    >>> print(f"phi={state.phi_deg}, theta={state.theta_deg}")
"""

from typing import Optional, Callable
from pathlib import Path

from ._native import get_native_module

_NATIVE = get_native_module()

# Re-export native classes if available
if _NATIVE is not None and hasattr(_NATIVE, "CameraAnimation"):
    CameraAnimation = _NATIVE.CameraAnimation
    CameraState = _NATIVE.CameraState
else:
    # Fallback pure-Python implementation for testing without native module
    class CameraState:
        """Interpolated camera state at a given time."""
        
        def __init__(self, phi_deg: float, theta_deg: float, radius: float, fov_deg: float):
            self.phi_deg = phi_deg
            self.theta_deg = theta_deg
            self.radius = radius
            self.fov_deg = fov_deg
        
        def __repr__(self) -> str:
            return (f"CameraState(phi={self.phi_deg:.2f}, theta={self.theta_deg:.2f}, "
                    f"radius={self.radius:.2f}, fov={self.fov_deg:.2f})")

    class CameraAnimation:
        """Camera animation with keyframe storage and cubic Hermite interpolation."""
        
        def __init__(self):
            self._keyframes = []
        
        def add_keyframe(self, time: float, phi: float, theta: float, 
                        radius: float, fov: float) -> None:
            """Add a keyframe. Keyframes are sorted by time automatically."""
            self._keyframes.append({
                'time': time,
                'phi': phi,
                'theta': theta,
                'radius': radius,
                'fov': fov,
            })
            self._keyframes.sort(key=lambda k: k['time'])
        
        # Alias for compatibility with native API
        def add_keyframe_py(self, time: float, phi: float, theta: float,
                           radius: float, fov: float) -> None:
            self.add_keyframe(time, phi, theta, radius, fov)
        
        @property
        def duration(self) -> float:
            """Get animation duration in seconds."""
            if not self._keyframes:
                return 0.0
            return self._keyframes[-1]['time']
        
        # Alias for native compatibility
        @property
        def duration_py(self) -> float:
            return self.duration
        
        @property
        def keyframe_count(self) -> int:
            """Get number of keyframes."""
            return len(self._keyframes)
        
        def get_frame_count(self, fps: int) -> int:
            """Get total frame count for given fps."""
            if self.duration <= 0 or fps <= 0:
                return 0
            import math
            return int(math.ceil(self.duration * fps)) + 1
        
        def evaluate(self, time: float) -> Optional[CameraState]:
            """Evaluate camera state at given time using cubic Hermite interpolation."""
            if not self._keyframes:
                return None
            
            # Clamp time
            time = max(0.0, min(time, self.duration))
            
            # Find surrounding keyframes
            n = len(self._keyframes)
            if n == 1:
                k = self._keyframes[0]
                return CameraState(k['phi'], k['theta'], k['radius'], k['fov'])
            
            # Find segment
            idx = 0
            for i, kf in enumerate(self._keyframes):
                if kf['time'] > time:
                    idx = max(0, i - 1)
                    break
                idx = i
            
            if idx >= n - 1:
                idx = n - 2
            
            k1 = self._keyframes[idx]
            k2 = self._keyframes[idx + 1]
            k0 = self._keyframes[idx - 1] if idx > 0 else k1
            k3 = self._keyframes[idx + 2] if idx + 2 < n else k2
            
            # Calculate t
            segment_duration = k2['time'] - k1['time']
            t = (time - k1['time']) / segment_duration if segment_duration > 0 else 0.0
            
            # Cubic Hermite interpolation
            def cubic_hermite(p0, p1, p2, p3, t):
                t2 = t * t
                t3 = t2 * t
                h1 = -0.5 * t3 + t2 - 0.5 * t
                h2 = 1.5 * t3 - 2.5 * t2 + 1.0
                h3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
                h4 = 0.5 * t3 - 0.5 * t2
                return h1 * p0 + h2 * p1 + h3 * p2 + h4 * p3
            
            return CameraState(
                phi_deg=cubic_hermite(k0['phi'], k1['phi'], k2['phi'], k3['phi'], t),
                theta_deg=cubic_hermite(k0['theta'], k1['theta'], k2['theta'], k3['theta'], t),
                radius=cubic_hermite(k0['radius'], k1['radius'], k2['radius'], k3['radius'], t),
                fov_deg=cubic_hermite(k0['fov'], k1['fov'], k2['fov'], k3['fov'], t),
            )
        
        # Alias for native compatibility
        def evaluate_py(self, time: float) -> Optional[CameraState]:
            return self.evaluate(time)
        
        def __repr__(self) -> str:
            return f"CameraAnimation(keyframes={self.keyframe_count}, duration={self.duration:.2f}s)"


class RenderConfig:
    """Configuration for offline animation rendering."""
    
    def __init__(
        self,
        output_dir: str = "./frames",
        fps: int = 30,
        width: int = 1920,
        height: int = 1080,
        filename_prefix: str = "frame",
        frame_digits: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.width = width
        self.height = height
        self.filename_prefix = filename_prefix
        self.frame_digits = frame_digits
    
    def frame_path(self, frame: int) -> Path:
        """Generate frame path for given frame number."""
        filename = f"{self.filename_prefix}_{frame:0{self.frame_digits}d}.png"
        return self.output_dir / filename
    
    def ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


class RenderProgress:
    """Progress information for render callbacks."""
    
    def __init__(self, frame: int, total_frames: int, time: float, output_path: Path):
        self.frame = frame
        self.total_frames = total_frames
        self.time = time
        self.output_path = output_path
    
    @property
    def percent(self) -> float:
        """Get progress as percentage (0.0 to 1.0)."""
        if self.total_frames == 0:
            return 0.0
        return self.frame / self.total_frames
    
    def __repr__(self) -> str:
        return f"RenderProgress({self.frame}/{self.total_frames}, {self.percent*100:.1f}%)"


__all__ = [
    "CameraAnimation",
    "CameraState",
    "RenderConfig",
    "RenderProgress",
]
