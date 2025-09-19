# python/forge3d/helpers/aov_io.py
# Utilities to load RGBA32F AOV dumps produced by examples (e.g., wavefront_instances)
# Format: raw bytes of float32, layout [H*W, 4] flattened, as written by std::fs::write

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


def load_rgba32f_raw(path: str, *, width: int | None = None, height: int | None = None) -> np.ndarray:
    """Load a raw RGBA32F dump into a NumPy array (no header expected).

    If width and height are provided, returns shape (H, W, 4). Otherwise returns shape (N, 4).
    """
    data = np.fromfile(path, dtype=np.float32)
    if data.size % 4 != 0:
        raise ValueError(f"File size is not multiple of 4 floats: {data.size}")
    arr = data.reshape((-1, 4))
    if width is not None and height is not None:
        expected = int(width) * int(height)
        if arr.shape[0] != expected:
            raise ValueError(f"Pixel count mismatch: file has {arr.shape[0]}, expected {expected}")
        return arr.reshape((int(height), int(width), 4))
    return arr


def depth_channel_x(aov_rgba: np.ndarray) -> np.ndarray:
    """Return the X channel as a 1D array (depth) from an RGBA32F array of shape (N,4) or (H,W,4)."""
    if aov_rgba.ndim == 3 and aov_rgba.shape[2] == 4:
        return aov_rgba[..., 0].reshape(-1)
    if aov_rgba.ndim == 2 and aov_rgba.shape[1] == 4:
        return aov_rgba[:, 0]
    raise ValueError("Invalid AOV array shape; expected (N,4) or (H,W,4)")


def parse_aov_header(path: str) -> Optional[Tuple[int, int, int, int]]:
    """Parse optional AOV header.

    Header format (16 bytes): b"AOV0" + width(u32 LE) + height(u32 LE) + channels(u32 LE)
    Returns (offset_bytes, width, height, channels) if present, else None.
    """
    try:
        with open(path, "rb") as f:
            hdr = f.read(16)
            if len(hdr) < 16:
                return None
            if hdr[:4] != b"AOV0":
                return None
            w = int.from_bytes(hdr[4:8], "little")
            h = int.from_bytes(hdr[8:12], "little")
            c = int.from_bytes(hdr[12:16], "little")
            return (16, w, h, c)
    except Exception:
        return None


def load_rgba32f_with_header(path: str) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Load an RGBA32F AOV dump with a 16-byte header. Returns (arr, (W,H,C)).

    The returned array has shape (H, W, 4). Raises if header is missing or channels != 4.
    """
    meta = parse_aov_header(path)
    if meta is None:
        raise ValueError("Missing AOV header 'AOV0'")
    offset, w, h, c = meta
    if c != 4:
        raise ValueError(f"Unsupported channels in AOV header: {c}")
    with open(path, "rb") as f:
        f.seek(offset)
        payload = f.read()
    data = np.frombuffer(payload, dtype=np.float32)
    if data.size != (w * h * 4):
        raise ValueError(f"AOV payload size mismatch: have {data.size} floats, expected {w*h*4}")
    return data.reshape((h, w, 4)), (w, h, c)


def load_rgba32f_auto(path: str) -> Tuple[np.ndarray, Optional[Tuple[int, int, int]]]:
    """Load an RGBA32F AOV dump, automatically detecting the optional header.

    Returns (arr, meta) where meta is (W,H,C) if header was present, else None.
    If no header is present, the array has shape (N,4).
    """
    meta = parse_aov_header(path)
    if meta is not None:
        arr, whc = load_rgba32f_with_header(path)
        return arr, whc
    # Fallback: raw blob without dims
    arr = load_rgba32f_raw(path)
    return arr, None
