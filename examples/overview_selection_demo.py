# examples/overview_selection_demo.py
# Demo for S6: Overview selection and windowed read using mock dataset.
# Exists to generate an artifact image and show chosen overview factor.
# RELEVANT FILES:python/forge3d/adapters/rasterio_tiles.py,tests/test_overview_selection.py,docs/ingest/overviews.md

#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


class MockDataset:
    def __init__(self, width=1024, height=768, xres=2.0, yres=2.0, overviews=(2, 4, 8)):
        self.width = width
        self.height = height
        self.overviews = lambda band: list(overviews)
        class T: pass
        self.transform = T()
        self.transform.a = xres
        self.transform.e = -yres
        self.count = 3
        self.dtype = "uint8"


def main():
    parser = argparse.ArgumentParser(description="S6: Overview selection demo")
    parser.add_argument("--out", default="reports/s6_overviews.png")
    args = parser.parse_args()

    try:
        from forge3d.adapters.rasterio_tiles import select_overview_level
        ds = MockDataset()
        idx, info = select_overview_level(ds, 8.0, band=1)
        print("Selected overview:", info)
    except Exception as e:
        print(f"Overview selection unavailable: {e}")

    # Make a small checkerboard illustrating overview factor
    H, W = 128, 128
    tile = np.kron([[0, 1] * (W//2//2), [1, 0] * (W//2//2)] * (H//2//2), np.ones((2, 2), dtype=np.uint8))
    img = np.stack([tile*255, tile*64, 255 - tile*255], axis=2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        import forge3d
        forge3d.numpy_to_png(str(args.out), img)
        print(f"Saved {args.out}")
    except Exception:
        import struct, zlib
        def _chunk(t, d):
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
        sig = b"\x89PNG\r\n\x1a\n"
        h, w = img.shape[:2]
        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        raw = b"".join(b"\x00" + img[i].tobytes() for i in range(h))
        idat = zlib.compress(raw)
        with open(args.out, "wb") as f:
            f.write(sig)
            f.write(_chunk(b"IHDR", ihdr))
            f.write(_chunk(b"IDAT", idat))
            f.write(_chunk(b"IEND", b""))
        print(f"Saved {args.out} via fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
