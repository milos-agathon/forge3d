#!/usr/bin/env python3
"""
Bloom Demo

Enables bloom controls via the PostFX API and writes a PNG using the triangle
renderer to create an artifact for audits.
"""
import argparse
from pathlib import Path
import forge3d as f3d
import forge3d.postfx as postfx


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--strength", type=float, default=0.6)
    p.add_argument("--out", type=Path, default=Path("reports/bloom.png"))
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    postfx.set_chain_enabled(True)
    postfx.enable("bloom", threshold=args.threshold, strength=args.strength)
    postfx.enable("tonemap", exposure=1.0, gamma=2.2)
    print("Enabled:", postfx.list())

    r = f3d.Renderer(args.width, args.height)
    r.render_triangle_png(str(args.out))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
