# examples/raster_window_demo.py
# Demo for S1: rasterio-style windowed reads and block iteration using mock dataset.
# Exists to generate an artifact image and exercise windowed_read without real rasterio.
# RELEVANT FILES:python/forge3d/adapters/rasterio_tiles.py,tests/test_rasterio_adapter.py,docs/ingest/rasterio_tiles.md

#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


class MockDataset:
    def __init__(self, data: np.ndarray):
        # data shape: (bands, H, W)
        self.data = data
        self.count = data.shape[0]
        self.height = data.shape[1]
        self.width = data.shape[2]
        self.dtype = data.dtype

    def read(self, indexes=None, window=None, out_shape=None, resampling=None, dtype=None):
        if indexes is None:
            idxs = list(range(1, self.count + 1))
        elif isinstance(indexes, int):
            idxs = [indexes]
        else:
            idxs = list(indexes)

        if window is None:
            hslice = slice(0, self.height)
            wslice = slice(0, self.width)
        else:
            hslice = slice(int(window.row_off), int(window.row_end))
            wslice = slice(int(window.col_off), int(window.col_end))

        subset = self.data[np.array(idxs) - 1, hslice, wslice]
        if out_shape is not None:
            oh, ow = out_shape
            # naive nearest resample
            y = (np.linspace(0, subset.shape[1] - 1, oh)).astype(int)
            x = (np.linspace(0, subset.shape[2] - 1, ow)).astype(int)
            subset = subset[:, y][:, :, x]
        if dtype is not None:
            subset = subset.astype(dtype)
        return subset


def main():
    parser = argparse.ArgumentParser(description="S1: Raster windowed read demo")
    parser.add_argument("--out", default="reports/s1_windows.png")
    args = parser.parse_args()

    # synthetic 3-band gradient
    H, W = 256, 384
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    band1 = y @ np.ones((1, W), dtype=np.float32)
    band2 = np.ones((H, 1), dtype=np.float32) @ x
    band3 = 0.5 * (band1 + band2)
    data = np.stack([band1, band2, band3], axis=0)
    ds = MockDataset(data)

    try:
        from forge3d.adapters.rasterio_tiles import windowed_read
        from rasterio.windows import Window
        # read center window and upscale
        w = Window(W//4, H//4, W//2, H//2)
        rgb = windowed_read(ds, w, out_shape=(H, W), indexes=[1, 2, 3], dtype="uint8")
        rgb = np.transpose(rgb, (1, 2, 0)) if rgb.ndim == 3 else rgb
    except Exception:
        # fallback to full image
        rgb = (np.transpose((data[:3] * 255).astype(np.uint8), (1, 2, 0)))

    # save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        import forge3d
        forge3d.numpy_to_png(str(args.out), rgb)
        print(f"Saved {args.out}")
    except Exception:
        # Fallback tiny PNG writer
        import struct, zlib
        def _chunk(t, d):
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
        sig = b"\x89PNG\r\n\x1a\n"
        h, w = rgb.shape[:2]
        ctype = 2 if rgb.shape[2] == 3 else 6
        ihdr = struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0)
        raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))
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
