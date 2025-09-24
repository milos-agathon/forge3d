#!/usr/bin/env python3
"""
Demo: 3D Text Meshes (multiple instances with transforms and per-instance light)

Requires native module (forge3d._forge3d). Provide a font at assets/fonts/Roboto-Regular.ttf.
"""
from _import_shim import ensure_repo_import
ensure_repo_import()

import os
from pathlib import Path

try:
    from forge3d._forge3d import Scene  # native extension
except Exception:
    print("Native extension not available; 3D text mesh demo requires forge3d._forge3d")
    raise SystemExit(0)


def load_font_bytes() -> bytes:
    # Look for a bundled font first
    repo = Path(__file__).parent.parent
    font_path = repo / "assets" / "fonts" / "Roboto-Regular.ttf"
    if font_path.is_file():
        return font_path.read_bytes()
    # Try some common system fonts
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        str(Path.home() / "Library/Fonts/Arial.ttf"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return Path(p).read_bytes()
    raise FileNotFoundError("No font found. Place a font at assets/fonts/Roboto-Regular.ttf")


def main():
    font_bytes = load_font_bytes()

    s = Scene(1000, 700, grid=64, colormap="terrain")
    s.enable_text_meshes()

    # Instance 1: large gold text
    s.add_text_mesh(
        "Forge3D",
        font_bytes,
        size_px=96.0,
        depth=0.25,
        position=(-0.8, 0.1, 0.0),
        color=(1.0, 0.9, 0.2, 1.0),
        rotation_deg=(0.0, -10.0, 0.0),
        scale=1.0,
        light_dir=(0.5, 1.0, 0.3),
        light_intensity=1.0,
    )

    # Instance 2: smaller cyan text, rotated
    s.add_text_mesh(
        "3D TEXT",
        font_bytes,
        size_px=48.0,
        depth=0.18,
        position=(-0.75, -0.25, 0.0),
        color=(0.2, 0.9, 1.0, 1.0),
        rotation_deg=(0.0, 15.0, -10.0),
        scale=1.2,
        light_dir=(0.0, 1.0, 0.0),
        light_intensity=1.2,
    )

    # Instance 3: magenta, tilted
    s.add_text_mesh(
        "Meshes",
        font_bytes,
        size_px=64.0,
        depth=0.15,
        position=(0.2, -0.1, 0.0),
        color=(1.0, 0.2, 0.8, 1.0),
        rotation_deg=(15.0, 0.0, 8.0),
        scale=0.9,
        light_dir=(0.3, 0.8, 0.6),
        light_intensity=0.9,
    )

    # Adjust transforms post-hoc to showcase API
    s.update_text_mesh_transform(1, position=(-0.7, -0.3, 0.0), rotation_deg=(0.0, 10.0, -8.0), scale=1.1)
    s.update_text_mesh_light(2, dx=0.2, dy=1.0, dz=0.2, intensity=1.1)

    # Optional: overlays to see layering
    s.enable_altitude_overlay(alpha=0.2)

    out = Path("ex_text3d_demo.png")
    s.render_png(out)
    inst_count, verts, inds = s.get_text_mesh_stats()
    print(f"Wrote {out} | instances={inst_count} verts={verts} inds={inds}")


if __name__ == "__main__":
    main()
