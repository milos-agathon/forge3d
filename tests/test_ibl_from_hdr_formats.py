import struct

import pytest

import forge3d as f3d

pytestmark = pytest.mark.skipif(not hasattr(f3d, "IBL"), reason="native IBL not available")


def _write_radiance(path, width=8, height=4):
    with open(path, "wb") as fh:
        fh.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        fh.write(f"-Y {height} +X {width}\n".encode())
        fh.write(bytes([128, 128, 128, 128]) * (width * height))


def _write_minimal_exr(path, width, height, channels):
    names = sorted(channels.keys())

    def attr(name, type_, payload):
        return name.encode() + b"\x00" + type_.encode() + b"\x00" + struct.pack("<i", len(payload)) + payload

    chlist = b""
    for nm in names:
        chlist += nm.encode() + b"\x00"
        chlist += struct.pack("<i", 2)
        chlist += b"\x00\x00\x00\x00"
        chlist += struct.pack("<i", 1)
        chlist += struct.pack("<i", 1)
    chlist += b"\x00"

    header = b""
    header += attr("channels", "chlist", chlist)
    header += attr("compression", "compression", b"\x00")
    header += attr("dataWindow", "box2i", struct.pack("<iiii", 0, 0, width - 1, height - 1))
    header += attr("displayWindow", "box2i", struct.pack("<iiii", 0, 0, width - 1, height - 1))
    header += attr("lineOrder", "lineOrder", b"\x00")
    header += attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    header += attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
    header += attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    header += b"\x00"

    magic = struct.pack("<i", 20000630)
    version = struct.pack("<i", 2)
    pix_size = len(names) * width * 4
    block_size = 8 + pix_size
    offset_start = len(magic) + len(version) + len(header)
    scanlines_start = offset_start + height * 8
    offsets = b"".join(struct.pack("<Q", scanlines_start + y * block_size) for y in range(height))

    blocks = b""
    for y in range(height):
        blocks += struct.pack("<i", y)
        blocks += struct.pack("<i", pix_size)
        for nm in names:
            row = channels[nm][y * width:(y + 1) * width]
            blocks += struct.pack(f"<{width}f", *row)

    with open(path, "wb") as fh:
        fh.write(magic + version + header + offsets + blocks)


def test_hdr_and_rgbe_extensions_accepted(tmp_path):
    for suffix in ("hdr", "rgbe"):
        p = tmp_path / f"env.{suffix}"
        _write_radiance(p)
        assert f3d.IBL.from_hdr(str(p)).dimensions == (8, 4)


def test_exr_extensions_accepted_case_insensitively(tmp_path):
    for name in ("env.exr", "ENV.EXR"):
        p = tmp_path / name
        _write_minimal_exr(p, 2, 2, {"R": [0.1] * 4, "G": [0.2] * 4, "B": [0.3] * 4})
        assert f3d.IBL.from_hdr(str(p)).dimensions == (2, 2)


def test_unsupported_and_missing_files_raise_ioerror(tmp_path):
    for name in ("env.png", "env.tif", "env.jpg", "env"):
        with pytest.raises(IOError):
            f3d.IBL.from_hdr(str(tmp_path / name))
    with pytest.raises(IOError):
        f3d.IBL.from_hdr(str(tmp_path / "does_not_exist.exr"))


def test_missing_exr_channel_raises_ioerror(tmp_path):
    p = tmp_path / "no_blue.exr"
    _write_minimal_exr(p, 2, 2, {"R": [0.1] * 4, "G": [0.2] * 4})
    with pytest.raises(IOError):
        f3d.IBL.from_hdr(str(p))
