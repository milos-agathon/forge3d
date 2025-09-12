# examples/reproject_window_demo.py
# Demo for S3: CRS normalization via WarpedVRT + pyproj using info inspection.
# Exists to produce an artifact image and exercise get_crs_info without heavy deps.
# RELEVANT FILES:python/forge3d/adapters/reproject.py,tests/test_reproject_window.py,docs/ingest/reprojection.md

#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="S3: Reprojection demo (info only)")
    parser.add_argument("--out", default="reports/s3_reproject.png")
    args = parser.parse_args()

    # small gradient placeholder image
    H, W = 128, 256
    img = (np.linspace(0, 255, W, dtype=np.uint8)[None, :]
           * np.ones((H, 1), dtype=np.uint8))
    img = np.repeat(img[:, :, None], 3, axis=2)

    try:
        from forge3d.adapters.reproject import get_crs_info
        info = get_crs_info("EPSG:4326")
        print("CRS info:", {k: info[k] for k in ("name", "authority")})
    except Exception as e:
        print(f"get_crs_info unavailable: {e}")

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
