#!/usr/bin/env python3
"""
PostFX Chain Demo

Enables a small post-processing chain and writes an output PNG using the
triangle renderer as a stand-in scene. Focuses on API wiring and artifact
creation for audits.
"""
import argparse
import time
from pathlib import Path
import forge3d as f3d
import forge3d.postfx as postfx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--effects", default="bloom,tonemap", help="Comma-separated effects to enable")
    parser.add_argument("--out", type=Path, default=Path("reports/postfx.png"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    postfx.set_chain_enabled(True)

    for name in [e.strip() for e in args.effects.split(",") if e.strip()]:
        if name == "bloom":
            postfx.enable("bloom", threshold=1.0, strength=0.6)
        elif name == "tonemap":
            postfx.enable("tonemap", exposure=1.1, gamma=2.2)
        elif name == "fxaa":
            postfx.enable("fxaa", quality=1.0)
        else:
            postfx.enable(name)

    print("Enabled effects (ordered):", postfx.list())
    print("Available effects:", postfx.list_available())
    print("Bloom info:", postfx.get_effect_info("bloom"))

    # Render a simple triangle PNG as artifact
    r = f3d.Renderer(args.width, args.height)
    r.render_triangle_png(str(args.out))

    # Illustrate timing stats (populated by native code when wired)
    time.sleep(0.01)
    print("Timing stats:", postfx.get_timing_stats())
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
