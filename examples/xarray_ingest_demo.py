# examples/xarray_ingest_demo.py
# Demo for S4: xarray/rioxarray DataArray ingestion with graceful degradation.
# Exists to generate a placeholder artifact and show ingestion path when available.
# RELEVANT FILES:python/forge3d/ingest/xarray_adapter.py,tests/test_xarray_ingestion.py,docs/ingest/xarray.md

#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="S4: xarray ingestion demo")
    parser.add_argument("--out", default="reports/s4_xarray.png")
    args = parser.parse_args()

    img = None
    try:
        from forge3d.ingest.xarray_adapter import create_synthetic_dataarray, ingest_dataarray, is_xarray_available
        if is_xarray_available():
            da = create_synthetic_dataarray((3, 128, 256))
            arr, meta = ingest_dataarray(da)
            print("Ingested dims:", meta.get("output_dims"))
            img = np.transpose(arr[:3], (1, 2, 0)) if arr.ndim == 3 else arr
    except Exception as e:
        print(f"xarray ingest unavailable: {e}")

    if img is None:
        # placeholder gradient
        H, W = 128, 256
        y = np.linspace(0, 255, H, dtype=np.uint8)[:, None]
        x = np.linspace(0, 255, W, dtype=np.uint8)[None, :]
        r = y @ np.ones((1, W), dtype=np.uint8)
        g = np.ones((H, 1), dtype=np.uint8) @ x
        b = (0.5 * (r.astype(np.float32) + g.astype(np.float32))).astype(np.uint8)
        img = np.stack([r, g, b], axis=2)

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
