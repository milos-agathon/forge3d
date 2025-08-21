# A2-BEGIN:terrain-cli
#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--grid", type=int, default=160)
    ap.add_argument("--out", default="terrain_spike.png")
    args = ap.parse_args(argv)

    try:
        import sys; exe={'exe': sys.executable}; from forge3d import TerrainSpike
    except Exception as e:
        raise SystemExit(
            "TerrainSpike not available. Rebuild in THIS venv with:\n  export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1  # if Python 3.13\n  maturin develop --release --features terrain_spike\n(Current python: %(exe)s)" % {'exe': sys.executable}
        ) from e

    r = TerrainSpike(args.width, args.height, args.grid)
    r.render_png(args.out)
    print(f"Wrote {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
# A2-END:terrain-cli
