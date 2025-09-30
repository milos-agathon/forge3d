# examples/viewer_offscreen_demo.py
# Python-only example: Offscreen render and display helpers (Workstream I)
# - Renders an RGBA image offscreen via forge3d
# - Saves a deterministic PNG with a stable byte pattern for hashing
# - Optionally displays via Matplotlib if available

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import forge3d as f3d


def main() -> int:
    parser = argparse.ArgumentParser(description="forge3d offscreen viewer demo")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--denoiser", type=str, default="off", choices=["off", "svgf"]) 
    parser.add_argument("--out", type=Path, default=Path("reports/viewer_offscreen.png"))
    parser.add_argument("--show", action="store_true", help="Display via Matplotlib if available")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Render offscreen; if native module is present it will be used, otherwise CPU fallback
    rgba = f3d.render_offscreen_rgba(
        args.width,
        args.height,
        seed=args.seed,
        frames=args.frames,
        denoiser=args.denoiser,
    )

    assert isinstance(rgba, np.ndarray) and rgba.ndim == 3 and rgba.shape[2] == 4

    # Save deterministically
    f3d.save_png_deterministic(str(args.out), rgba)
    print(f"Wrote {args.out}")

    if args.show:
        try:
            from forge3d.helpers.mpl_display import quick_show

            quick_show(rgba, title="forge3d Offscreen Viewer Demo")
        except Exception as exc:
            print(f"Matplotlib display unavailable: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
