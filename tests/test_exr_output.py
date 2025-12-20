"""M2: Tests for EXR output and HDR save paths."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest


def _read_cstring(data: bytes, offset: int) -> tuple[str, int]:
    end = data.find(b"\x00", offset)
    if end < 0:
        raise ValueError("unterminated string in EXR header")
    return data[offset:end].decode("ascii"), end + 1


def _parse_channel_list(payload: bytes) -> list[str]:
    names: list[str] = []
    offset = 0
    while True:
        name, offset = _read_cstring(payload, offset)
        if name == "":
            break
        if offset + 16 > len(payload):
            raise ValueError("truncated EXR channel entry")
        offset += 16
        names.append(name)
    return names


def _parse_data_window(payload: bytes) -> tuple[int, int]:
    if len(payload) < 16:
        raise ValueError("truncated EXR dataWindow")
    min_x, min_y, max_x, max_y = struct.unpack("<iiii", payload[:16])
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    return width, height


def _read_exr_header(path: Path) -> tuple[list[str], int, int]:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError("invalid EXR header")
    magic, _version = struct.unpack("<II", data[:8])
    if magic != 20000630:
        raise ValueError("invalid EXR magic")

    offset = 8
    channels: list[str] | None = None
    width: int | None = None
    height: int | None = None

    while True:
        name, offset = _read_cstring(data, offset)
        if name == "":
            break
        _attr_type, offset = _read_cstring(data, offset)
        if offset + 4 > len(data):
            raise ValueError("truncated EXR attribute size")
        size = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        if offset + size > len(data):
            raise ValueError("truncated EXR attribute payload")
        payload = data[offset:offset + size]
        offset += size

        if name == "channels":
            channels = _parse_channel_list(payload)
        elif name == "dataWindow":
            width, height = _parse_data_window(payload)

    if channels is None or width is None or height is None:
        raise ValueError("missing EXR channels or dataWindow")
    return channels, width, height


def _require_native_exr(tmp_path: Path):
    try:
        import forge3d._forge3d as native
    except Exception:
        pytest.skip("native module not available")
    if not hasattr(native, "numpy_to_exr"):
        pytest.skip("numpy_to_exr not exported")

    probe = tmp_path / "_probe.exr"
    try:
        native.numpy_to_exr(str(probe), np.zeros((1, 1, 3), dtype=np.float32), channel_prefix="beauty")
    except Exception as exc:
        if "images" in str(exc).lower() and "feature" in str(exc).lower():
            pytest.skip("native EXR writer not enabled")
        raise
    return native


def test_numpy_to_exr_beauty_channels(tmp_path: Path) -> None:
    native = _require_native_exr(tmp_path)
    path = tmp_path / "beauty.exr"
    data = np.zeros((2, 3, 4), dtype=np.float32)
    native.numpy_to_exr(str(path), data, channel_prefix="beauty")

    channels, width, height = _read_exr_header(path)
    assert width == 3
    assert height == 2
    assert set(channels) == {"beauty.R", "beauty.G", "beauty.B", "beauty.A"}


@pytest.mark.parametrize(
    ("shape", "prefix", "expected_channels"),
    [
        ((2, 3, 3), "albedo", {"albedo.R", "albedo.G", "albedo.B"}),
        ((2, 3, 3), "normal", {"normal.X", "normal.Y", "normal.Z"}),
        ((2, 3), "depth", {"depth.Z"}),
        ((2, 3), "roughness", {"roughness"}),
        ((2, 3), "metallic", {"metallic"}),
        ((2, 3), "ao", {"ao"}),
        ((2, 3), "sun_vis", {"sun_vis"}),
        ((2, 3), "id", {"id"}),
        ((2, 3), "mask", {"mask"}),
    ],
)
def test_numpy_to_exr_channel_naming(
    tmp_path: Path,
    shape: tuple[int, ...],
    prefix: str,
    expected_channels: set[str],
) -> None:
    native = _require_native_exr(tmp_path)
    path = tmp_path / f"{prefix}.exr"
    data = np.zeros(shape, dtype=np.float32)
    native.numpy_to_exr(str(path), data, channel_prefix=prefix)

    channels, width, height = _read_exr_header(path)
    assert width == shape[1]
    assert height == shape[0]
    assert set(channels) == expected_channels


def test_save_aovs_exr_channels(tmp_path: Path) -> None:
    _require_native_exr(tmp_path)
    import forge3d.path_tracing as pt

    aovs = pt.render_aovs(
        4,
        3,
        scene=None,
        camera=None,
        aovs=("albedo", "normal", "depth"),
        seed=3,
        frames=1,
        use_gpu=False,
    )
    out_paths = pt.save_aovs(aovs, basename="frame0001", output_dir=str(tmp_path))

    albedo_path = Path(out_paths["albedo"])
    normal_path = Path(out_paths["normal"])
    depth_path = Path(out_paths["depth"])

    for path in (albedo_path, normal_path, depth_path):
        assert path.exists()

    channels, width, height = _read_exr_header(albedo_path)
    assert (width, height) == (4, 3)
    assert set(channels) == {"albedo.R", "albedo.G", "albedo.B"}

    channels, width, height = _read_exr_header(normal_path)
    assert (width, height) == (4, 3)
    assert set(channels) == {"normal.X", "normal.Y", "normal.Z"}

    channels, width, height = _read_exr_header(depth_path)
    assert (width, height) == (4, 3)
    assert set(channels) == {"depth.Z"}
