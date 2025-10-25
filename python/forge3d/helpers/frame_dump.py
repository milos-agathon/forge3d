# python/forge3d/helpers/frame_dump.py
# Workstream I3: Frame sequence dumper for animation recording
# - Captures sequences of frames to disk
# - Auto-numbered filenames with zero-padding
# - Configurable output directory and frame prefix

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np

from .offscreen import save_png_deterministic


class FrameDumper:
    """Records sequences of RGBA frames to disk with auto-numbered filenames.

    Example:
        dumper = FrameDumper(output_dir="frames", prefix="render")
        dumper.start_recording()
        for i in range(100):
            rgba = render_frame(i)
            dumper.capture_frame(rgba)
        dumper.stop_recording()
        # Produces: frames/render_0000.png, frames/render_0001.png, ...
    """

    def __init__(self, output_dir: str | Path = "frames", prefix: str = "frame"):
        """Initialize frame dumper.

        Args:
            output_dir: Directory to write frames (created if missing)
            prefix: Filename prefix for frames
        """
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.frame_count = 0
        self.recording = False

    def start_recording(self) -> None:
        """Start recording frames."""
        if self.recording:
            raise RuntimeError("Already recording; call stop_recording() first")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0
        self.recording = True

    def capture_frame(self, rgba: np.ndarray) -> Path:
        """Capture a single frame to disk.

        Args:
            rgba: RGBA array (H, W, 4) uint8 or float32

        Returns:
            Path to saved frame

        Raises:
            RuntimeError: If not currently recording
        """
        if not self.recording:
            raise RuntimeError("Not recording; call start_recording() first")

        filename = f"{self.prefix}_{self.frame_count:04d}.png"
        path = self.output_dir / filename

        save_png_deterministic(str(path), rgba)
        self.frame_count += 1

        return path

    def stop_recording(self) -> int:
        """Stop recording and return frame count.

        Returns:
            Number of frames captured
        """
        if not self.recording:
            raise RuntimeError("Not recording")

        self.recording = False
        return self.frame_count

    def get_frame_count(self) -> int:
        """Get current frame count."""
        return self.frame_count

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.recording


def dump_frame_sequence(
    frames: list[np.ndarray],
    output_dir: str | Path = "frames",
    prefix: str = "frame"
) -> int:
    """Convenience function to dump a list of frames.

    Args:
        frames: List of RGBA arrays
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        Number of frames written
    """
    dumper = FrameDumper(output_dir=output_dir, prefix=prefix)
    dumper.start_recording()

    for rgba in frames:
        dumper.capture_frame(rgba)

    return dumper.stop_recording()
