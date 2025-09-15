# examples/progressive_tiling.py
# Demonstrate progressive/tiling PathTracer with checkpoint callbacks.
# This exists to show A15 interactive preview behavior and write outputs to out/.
# RELEVANT FILES:python/forge3d/path_tracing.py,README.md,docs/api/path_tracing.md

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import forge3d.path_tracing as pt


def main() -> None:
    w, h = 512, 384
    tracer = pt.PathTracer(w, h, seed=5, tile=64)

    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)

    last_save = 0.0

    def on_update(info: dict) -> bool | None:
        nonlocal last_save
        img: np.ndarray = info["image"]
        ts: float = info["timestamp"]
        prog: float = info["progress"]
        if ts - last_save >= 0.5 or prog >= 0.9999:
            # Save a PNG if available
            try:
                from PIL import Image  # type: ignore

                Image.fromarray(img, mode="RGBA").save(out_dir / "progressive_demo.png")
            except Exception:
                pass
            last_save = ts
        return None

    _ = tracer.render_progressive(tile_size=64, min_updates_per_sec=2.0, callback=on_update)
    print("Wrote out/progressive_demo.png (if PIL available).")


if __name__ == "__main__":
    main()

