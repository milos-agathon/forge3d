# tests/test_ibl_from_hdr_formats.py
# Slice 3 of the 2026-07-13 HDR terrain mood design: IBL.from_hdr must honour
# its documented .hdr / .rgbe / .exr contract by dispatching the loader on the
# file extension, and reject unsupported extensions with an I/O error.
#
# IBL.from_hdr decodes and stores the environment image WITHOUT touching the
# GPU (prefiltering is lazy), so these run without a rendering device. Value
# correctness of the EXR decode is locked by the Rust tests in
# src/formats/hdr.rs; here we lock the Python-visible extension dispatch.
#
# No EXR writer library ships in the test env and forge3d exposes no plain
# R/G/B EXR writer, so we author a minimal uncompressed OpenEXR (FLOAT samples,
# channels named exactly R/G/B) in pure Python.

import struct

import numpy as np
import pytest

import forge3d as f3d

pytestmark = pytest.mark.skipif(
    not hasattr(f3d, "IBL"), reason="native IBL not available"
)


def _write_radiance(path, width=8, height=4):
    with open(path, "wb") as fh:
        fh.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        fh.write(f"-Y {height} +X {width}\n".encode())
        # First byte 128 (!= 2) forces the uncompressed scanline branch.
        fh.write(bytes([128, 128, 128, 128]) * (width * height))


def _write_minimal_exr(path, width, height, channels):
    """Write an uncompressed single-part scanline OpenEXR with FLOAT samples.

    ``channels`` maps channel name -> flat row-major float sequence of length
    ``width * height``. Channels are stored alphabetically, as OpenEXR requires.
    """
    names = sorted(channels.keys())
    nch = len(names)

    def attr(name, type_, payload):
        return (
            name.encode() + b"\x00"
            + type_.encode() + b"\x00"
            + struct.pack("<i", len(payload))
            + payload
        )

    chlist = b""
    for nm in names:
        chlist += nm.encode() + b"\x00"
        chlist += struct.pack("<i", 2)   # pixel type: FLOAT
        chlist += struct.pack("<B", 0)   # pLinear
        chlist += b"\x00\x00\x00"        # reserved
        chlist += struct.pack("<i", 1)   # xSampling
        chlist += struct.pack("<i", 1)   # ySampling
    chlist += b"\x00"                    # end of channel list

    header = b""
    header += attr("channels", "chlist", chlist)
    header += attr("compression", "compression", struct.pack("<B", 0))  # none
    header += attr("dataWindow", "box2i", struct.pack("<iiii", 0, 0, width - 1, height - 1))
    header += attr("displayWindow", "box2i", struct.pack("<iiii", 0, 0, width - 1, height - 1))
    header += attr("lineOrder", "lineOrder", struct.pack("<B", 0))  # increasing Y
    header += attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    header += attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
    header += attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    header += b"\x00"                    # end of header

    magic = struct.pack("<i", 20000630)
    version = struct.pack("<i", 2)       # version 2, no flags (scanline)

    pix_size = nch * width * 4           # FLOAT = 4 bytes/sample
    block_size = 8 + pix_size            # y(4) + dataSize(4) + pixels
    offset_table_start = len(magic) + len(version) + len(header)
    scanlines_start = offset_table_start + height * 8

    offset_table = b"".join(
        struct.pack("<Q", scanlines_start + y * block_size) for y in range(height)
    )

    blocks = b""
    for y in range(height):
        blocks += struct.pack("<i", y)
        blocks += struct.pack("<i", pix_size)
        for nm in names:
            row = channels[nm][y * width:(y + 1) * width]
            blocks += struct.pack("<%df" % width, *row)

    with open(path, "wb") as fh:
        fh.write(magic + version + header + offset_table + blocks)


def test_hdr_extension_accepted(tmp_path):
    p = tmp_path / "env.hdr"
    _write_radiance(p)
    ibl = f3d.IBL.from_hdr(str(p))
    assert ibl.dimensions == (8, 4)


def test_rgbe_extension_accepted(tmp_path):
    p = tmp_path / "env.rgbe"
    _write_radiance(p)
    ibl = f3d.IBL.from_hdr(str(p))
    assert ibl.dimensions == (8, 4)


def test_exr_extension_accepted(tmp_path):
    p = tmp_path / "env.exr"
    w, h = 2, 2
    n = w * h
    _write_minimal_exr(
        p, w, h,
        {
            "R": [0.1, 0.2, 0.3, 0.4][:n],
            "G": [0.5, 0.6, 0.7, 0.8][:n],
            "B": [0.9, 1.0, 1.1, 1.2][:n],
        },
    )
    ibl = f3d.IBL.from_hdr(str(p))
    assert ibl.dimensions == (2, 2)


def test_exr_uppercase_extension_accepted(tmp_path):
    # Dispatch is case-insensitive.
    p = tmp_path / "ENV.EXR"
    _write_minimal_exr(p, 1, 1, {"R": [0.5], "G": [0.25], "B": [0.75]})
    ibl = f3d.IBL.from_hdr(str(p))
    assert ibl.dimensions == (1, 1)


def test_unsupported_extension_raises_ioerror(tmp_path):
    # The unsupported-format branch fires before the file is even opened.
    for name in ("env.png", "env.tif", "env.jpg", "env"):
        with pytest.raises(IOError):
            f3d.IBL.from_hdr(str(tmp_path / name))


def test_missing_exr_channel_raises_ioerror(tmp_path):
    # A decode failure (missing B channel) surfaces as IOError, not RuntimeError.
    p = tmp_path / "no_blue.exr"
    _write_minimal_exr(p, 2, 2, {"R": [0.1, 0.2, 0.3, 0.4], "G": [0.5, 0.6, 0.7, 0.8]})
    with pytest.raises(IOError):
        f3d.IBL.from_hdr(str(p))


def test_nonexistent_exr_raises_ioerror(tmp_path):
    with pytest.raises(IOError):
        f3d.IBL.from_hdr(str(tmp_path / "does_not_exist.exr"))
