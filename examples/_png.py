# examples/_png.py
# Tiny PNG writer for uint8 RGB/RGBA images without external deps.
# Exists to let demos save artifacts when forge3d/matplotlib are unavailable.
# RELEVANT FILES:examples/*.py

import struct
import zlib


def _crc(chunk_type: bytes, data: bytes) -> int:
    return zlib.crc32(chunk_type + data) & 0xFFFFFFFF


def write_png(path: str, image):
    import numpy as np
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] == 1:
        # gray -> RGB
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] not in (3, 4):
        raise ValueError("write_png expects RGB or RGBA last-dim")

    height, width, channels = arr.shape
    color_type = 6 if channels == 4 else 2

    # PNG signature
    signature = b"\x89PNG\r\n\x1a\n"
    chunks = []

    # IHDR
    ihdr = struct.pack(
        ">IIBBBBB",
        width,
        height,
        8,  # bit depth
        color_type,
        0,  # compression
        0,  # filter
        0,  # interlace
    )
    chunks.append(_chunk(b"IHDR", ihdr))

    # IDAT (with no filter per scanline)
    raw = b"".join(b"\x00" + arr[i].tobytes() for i in range(height))
    compressed = zlib.compress(raw)
    chunks.append(_chunk(b"IDAT", compressed))

    # IEND
    chunks.append(_chunk(b"IEND", b""))

    with open(path, "wb") as f:
        f.write(signature)
        for c in chunks:
            f.write(c)


def _chunk(typ: bytes, data: bytes) -> bytes:
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", _crc(typ, data))
    return length + typ + data + crc

