from __future__ import annotations

import struct
import zlib

import numpy as np

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_ADAM7_PASSES = (
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
)
_SAMPLES_PER_PIXEL = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", crc)
    )


def _as_png_writable_array(array: np.ndarray) -> tuple[np.ndarray, int]:
    arr = np.ascontiguousarray(array)
    if arr.dtype != np.uint8:
        raise ValueError("PNG encoder requires uint8 data")
    if arr.ndim == 2:
        return arr, 0
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr, 2
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr, 6
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def encode_png(array: np.ndarray, *, compress_level: int = 6) -> bytes:
    arr, color_type = _as_png_writable_array(array)
    height, width = arr.shape[:2]
    raw = bytearray()
    if arr.ndim == 2:
        for row in arr:
            raw.append(0)
            raw.extend(row.tobytes())
    else:
        flat = arr.reshape(height, -1)
        for row in flat:
            raw.append(0)
            raw.extend(row.tobytes())

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return b"".join(
        (
            _PNG_SIGNATURE,
            _chunk(b"IHDR", ihdr),
            _chunk(b"IDAT", zlib.compress(bytes(raw), level=compress_level)),
            _chunk(b"IEND", b""),
        )
    )


def save_png(path, array: np.ndarray, *, compress_level: int = 6) -> None:
    with open(path, "wb") as fh:
        fh.write(encode_png(array, compress_level=compress_level))


def _row_byte_count(width: int, bit_depth: int, color_type: int) -> int:
    samples = _SAMPLES_PER_PIXEL[color_type]
    return (width * samples * bit_depth + 7) // 8


def _filter_bpp(bit_depth: int, color_type: int) -> int:
    samples = _SAMPLES_PER_PIXEL[color_type]
    return max(1, (samples * bit_depth + 7) // 8)


def _paeth(left: int, up: int, up_left: int) -> int:
    p = left + up - up_left
    pa = abs(p - left)
    pb = abs(p - up)
    pc = abs(p - up_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return up
    return up_left


def _unfilter_rows(
    raw: bytes,
    width: int,
    height: int,
    bit_depth: int,
    color_type: int,
) -> list[bytes]:
    row_size = _row_byte_count(width, bit_depth, color_type)
    bpp = _filter_bpp(bit_depth, color_type)
    rows: list[bytes] = []
    pos = 0
    prev = bytearray(row_size)

    for _ in range(height):
        if pos >= len(raw):
            raise ValueError("PNG scanline data is truncated")
        filter_type = raw[pos]
        pos += 1
        row = bytearray(raw[pos : pos + row_size])
        pos += row_size
        if len(row) != row_size:
            raise ValueError("PNG scanline data is truncated")

        if filter_type == 1:
            for i in range(row_size):
                row[i] = (row[i] + (row[i - bpp] if i >= bpp else 0)) & 0xFF
        elif filter_type == 2:
            for i in range(row_size):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif filter_type == 3:
            for i in range(row_size):
                left = row[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + ((left + prev[i]) // 2)) & 0xFF
        elif filter_type == 4:
            for i in range(row_size):
                left = row[i - bpp] if i >= bpp else 0
                up = prev[i]
                up_left = prev[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + _paeth(left, up, up_left)) & 0xFF
        elif filter_type != 0:
            raise ValueError(f"Unsupported PNG filter type: {filter_type}")

        rows.append(bytes(row))
        prev = row

    if pos != len(raw):
        raise ValueError("PNG scanline data has trailing bytes")
    return rows


def _unpack_samples(row: bytes, width: int, bit_depth: int) -> np.ndarray:
    if bit_depth == 8:
        return np.frombuffer(row, dtype=np.uint8, count=width)
    if bit_depth == 16:
        return np.frombuffer(row, dtype=">u2", count=width).astype(np.uint16)

    mask = (1 << bit_depth) - 1
    shift_reset = 8 - bit_depth
    shift = shift_reset
    byte_index = 0
    values = np.empty(width, dtype=np.uint8)
    for i in range(width):
        values[i] = (row[byte_index] >> shift) & mask
        if shift == 0:
            byte_index += 1
            shift = shift_reset
        else:
            shift -= bit_depth
    return values


def _scale_to_u8(values: np.ndarray, bit_depth: int) -> np.ndarray:
    if bit_depth == 8:
        return values.astype(np.uint8, copy=False)
    if bit_depth == 16:
        scaled = (values.astype(np.uint32) * 255 + 32767) // 65535
        return scaled.astype(np.uint8)
    max_value = (1 << bit_depth) - 1
    scaled = (values.astype(np.uint16) * 255 + (max_value // 2)) // max_value
    return scaled.astype(np.uint8)


def _decode_row_to_rgba(
    row: bytes,
    width: int,
    bit_depth: int,
    color_type: int,
    palette_rgba: np.ndarray | None,
    trns: bytes | None,
) -> np.ndarray:
    if color_type == 0:
        raw = _unpack_samples(row, width, bit_depth)
        gray = _scale_to_u8(raw, bit_depth)
        rgba = np.empty((width, 4), dtype=np.uint8)
        rgba[:, :3] = gray[:, None]
        rgba[:, 3] = 255
        if trns is not None:
            transparent = int(np.frombuffer(trns, dtype=">u2", count=1)[0])
            rgba[raw == transparent, 3] = 0
        return rgba

    if color_type == 2:
        dtype = np.uint8 if bit_depth == 8 else ">u2"
        raw = np.frombuffer(row, dtype=dtype).reshape(width, 3)
        if bit_depth == 16:
            raw = raw.astype(np.uint16)
        rgb = _scale_to_u8(raw, bit_depth)
        rgba = np.empty((width, 4), dtype=np.uint8)
        rgba[:, :3] = rgb
        rgba[:, 3] = 255
        if trns is not None:
            transparent = np.frombuffer(trns, dtype=">u2", count=3).astype(raw.dtype)
            rgba[np.all(raw == transparent, axis=1), 3] = 0
        return rgba

    if color_type == 3:
        if palette_rgba is None:
            raise ValueError("Indexed PNG is missing a PLTE chunk")
        indices = _unpack_samples(row, width, bit_depth).astype(np.uint16)
        if np.any(indices >= palette_rgba.shape[0]):
            raise ValueError("PNG palette index is out of range")
        return palette_rgba[indices]

    if color_type == 4:
        dtype = np.uint8 if bit_depth == 8 else ">u2"
        raw = np.frombuffer(row, dtype=dtype).reshape(width, 2)
        if bit_depth == 16:
            raw = raw.astype(np.uint16)
        gray = _scale_to_u8(raw[:, 0], bit_depth)
        alpha = _scale_to_u8(raw[:, 1], bit_depth)
        rgba = np.empty((width, 4), dtype=np.uint8)
        rgba[:, :3] = gray[:, None]
        rgba[:, 3] = alpha
        return rgba

    if color_type == 6:
        dtype = np.uint8 if bit_depth == 8 else ">u2"
        raw = np.frombuffer(row, dtype=dtype).reshape(width, 4)
        if bit_depth == 16:
            raw = raw.astype(np.uint16)
        return _scale_to_u8(raw, bit_depth)

    raise ValueError(f"Unsupported PNG color type: {color_type}")


def decode_png(data: bytes) -> np.ndarray:
    if data[:8] != _PNG_SIGNATURE:
        raise ValueError("Not a PNG file")

    width = height = bit_depth = color_type = interlace = None
    palette_rgba = None
    trns = None
    idat = bytearray()
    pos = 8

    while pos < len(data):
        if pos + 12 > len(data):
            raise ValueError("PNG chunk header is truncated")
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        if pos + 12 + length > len(data):
            raise ValueError("PNG chunk data is truncated")
        chunk_type = data[pos + 4 : pos + 8]
        chunk_data = data[pos + 8 : pos + 8 + length]
        crc = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])[0]
        if zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF != crc:
            raise ValueError(f"PNG chunk CRC mismatch for {chunk_type.decode('ascii', 'replace')}")
        pos += length + 12

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            if compression != 0 or filter_method != 0:
                raise ValueError("Unsupported PNG compression or filter method")
            if color_type not in _SAMPLES_PER_PIXEL:
                raise ValueError(f"Unsupported PNG color type: {color_type}")
            if interlace not in (0, 1):
                raise ValueError(f"Unsupported PNG interlace method: {interlace}")
        elif chunk_type == b"PLTE":
            palette = np.frombuffer(chunk_data, dtype=np.uint8)
            if palette.size % 3 != 0:
                raise ValueError("Invalid PNG palette length")
            palette_rgba = np.empty((palette.size // 3, 4), dtype=np.uint8)
            palette_rgba[:, :3] = palette.reshape(-1, 3)
            palette_rgba[:, 3] = 255
        elif chunk_type == b"tRNS":
            trns = bytes(chunk_data)
            if palette_rgba is not None:
                alpha = np.frombuffer(trns, dtype=np.uint8, count=min(len(trns), palette_rgba.shape[0]))
                palette_rgba[: alpha.size, 3] = alpha
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
        elif chunk_type[:1].isupper():
            raise ValueError(f"Unsupported PNG critical chunk: {chunk_type.decode('ascii', 'replace')}")

    if width is None or height is None or bit_depth is None or color_type is None or interlace is None:
        raise ValueError("PNG is missing an IHDR chunk")

    raw = zlib.decompress(bytes(idat))
    output = np.empty((height, width, 4), dtype=np.uint8)
    passes = ((0, 0, 1, 1),) if interlace == 0 else _ADAM7_PASSES

    offset = 0
    for x0, y0, dx, dy in passes:
        pass_width = 0 if width <= x0 else (width - x0 + dx - 1) // dx
        pass_height = 0 if height <= y0 else (height - y0 + dy - 1) // dy
        if pass_width == 0 or pass_height == 0:
            continue

        row_size = _row_byte_count(pass_width, bit_depth, color_type)
        scan_size = pass_height * (1 + row_size)
        rows = _unfilter_rows(raw[offset : offset + scan_size], pass_width, pass_height, bit_depth, color_type)
        offset += scan_size

        decoded = np.empty((pass_height, pass_width, 4), dtype=np.uint8)
        for row_index, row in enumerate(rows):
            decoded[row_index] = _decode_row_to_rgba(
                row,
                pass_width,
                bit_depth,
                color_type,
                palette_rgba,
                trns,
            )
        output[y0:height:dy, x0:width:dx] = decoded

    if offset != len(raw):
        raise ValueError("PNG image data has trailing bytes")
    return output


def load_png_rgba(path) -> np.ndarray:
    try:
        from PIL import Image
    except Exception:
        with open(path, "rb") as fh:
            return decode_png(fh.read())

    with Image.open(path) as img:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        return np.array(img, dtype=np.uint8)
