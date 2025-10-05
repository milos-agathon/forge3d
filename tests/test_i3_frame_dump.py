# tests/test_i3_frame_dump.py
# Workstream I3: Frame dumper tests
# - Validates sequential frame numbering
# - Validates frame count accuracy
# - Validates start/stop state management

import pytest
import numpy as np
from pathlib import Path

from forge3d.helpers.frame_dump import FrameDumper, dump_frame_sequence


@pytest.mark.offscreen
def test_frame_dumper_basic(tmp_path):
    """Test basic frame dumper functionality."""
    dumper = FrameDumper(output_dir=tmp_path / "frames", prefix="test")

    assert not dumper.is_recording()
    assert dumper.get_frame_count() == 0

    dumper.start_recording()
    assert dumper.is_recording()

    # Capture 5 frames
    for i in range(5):
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba[:, :, i % 3] = 255  # Vary color per frame
        rgba[:, :, 3] = 255

        path = dumper.capture_frame(rgba)
        assert path.exists()
        assert f"test_{i:04d}.png" in path.name

    assert dumper.get_frame_count() == 5

    count = dumper.stop_recording()
    assert count == 5
    assert not dumper.is_recording()

    # Verify files
    frames = sorted((tmp_path / "frames").glob("test_*.png"))
    assert len(frames) == 5
    assert frames[0].name == "test_0000.png"
    assert frames[4].name == "test_0004.png"


@pytest.mark.offscreen
def test_frame_dumper_sequential_numbering(tmp_path):
    """Test that frames are numbered sequentially with zero-padding."""
    dumper = FrameDumper(output_dir=tmp_path, prefix="frame")
    dumper.start_recording()

    # Capture 100 frames to test 4-digit padding
    for i in range(100):
        rgba = np.full((50, 50, 4), i % 256, dtype=np.uint8)
        dumper.capture_frame(rgba)

    dumper.stop_recording()

    # Verify sequential naming
    frames = sorted(tmp_path.glob("frame_*.png"))
    assert len(frames) == 100

    # Check padding
    assert frames[0].name == "frame_0000.png"
    assert frames[9].name == "frame_0009.png"
    assert frames[10].name == "frame_0010.png"
    assert frames[99].name == "frame_0099.png"


@pytest.mark.offscreen
def test_frame_dumper_state_errors():
    """Test that dumper raises errors for invalid state transitions."""
    dumper = FrameDumper()

    # Cannot capture before starting
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="Not recording"):
        dumper.capture_frame(rgba)

    # Cannot stop before starting
    with pytest.raises(RuntimeError, match="Not recording"):
        dumper.stop_recording()

    # Cannot start twice
    dumper.start_recording()
    with pytest.raises(RuntimeError, match="Already recording"):
        dumper.start_recording()

    dumper.stop_recording()


@pytest.mark.offscreen
def test_dump_frame_sequence_convenience(tmp_path):
    """Test convenience function for dumping list of frames."""
    frames = []
    for i in range(10):
        rgba = np.full((40, 40, 4), i * 25, dtype=np.uint8)
        frames.append(rgba)

    count = dump_frame_sequence(frames, output_dir=tmp_path / "seq", prefix="seq")

    assert count == 10

    # Verify files
    files = sorted((tmp_path / "seq").glob("seq_*.png"))
    assert len(files) == 10
    assert files[0].name == "seq_0000.png"
    assert files[9].name == "seq_0009.png"


@pytest.mark.offscreen
def test_frame_dumper_mixed_dtypes(tmp_path):
    """Test frame dumper with different input dtypes."""
    dumper = FrameDumper(output_dir=tmp_path, prefix="mixed")
    dumper.start_recording()

    # uint8
    rgba_uint8 = np.zeros((30, 30, 4), dtype=np.uint8)
    rgba_uint8[:, :, :3] = 100
    rgba_uint8[:, :, 3] = 255
    dumper.capture_frame(rgba_uint8)

    # float32
    rgba_float32 = np.ones((30, 30, 4), dtype=np.float32) * 0.5
    dumper.capture_frame(rgba_float32)

    # float64
    rgba_float64 = np.ones((30, 30, 4), dtype=np.float64) * 0.75
    dumper.capture_frame(rgba_float64)

    count = dumper.stop_recording()
    assert count == 3

    # All should be saved successfully
    files = sorted(tmp_path.glob("mixed_*.png"))
    assert len(files) == 3
