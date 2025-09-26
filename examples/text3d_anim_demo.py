#!/usr/bin/env python3
"""
Demo: Animated 3D Text Meshes (scriptable rotations and light)

Requires native module (forge3d._forge3d). Provide a font at assets/fonts/Roboto-Regular.ttf or ensure a system font is available.
Writes frames to out/text3d_anim_####.png
"""
from _import_shim import ensure_repo_import
ensure_repo_import()

import math
import os
from pathlib import Path

try:
    from forge3d._forge3d import Scene  # native extension
except Exception:
    print("Native extension not available; 3D text mesh demo requires forge3d._forge3d")
    raise SystemExit(0)


def load_font_bytes() -> bytes:
    repo = Path(__file__).parent.parent
    font_path = repo / "assets" / "fonts" / "Roboto-Regular.ttf"
    if font_path.is_file():
        return font_path.read_bytes()
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        str(Path.home() / "Library/Fonts/Arial.ttf"),
    ):
        if os.path.isfile(p):
            return Path(p).read_bytes()
    raise FileNotFoundError("No font found. Place a font at assets/fonts/Roboto-Regular.ttf")


def main():
    font_bytes = load_font_bytes()
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    s = Scene(960, 540, grid=64, colormap="terrain")
    s.enable_text_meshes()

    # Add two animated text instances with different materials
    s.add_text_mesh(
        "Forge3D",
        font_bytes,
        size_px=96.0,
        depth=0.22,
        position=(-0.6, 0.05, 0.0),
        color=(1.0, 0.9, 0.2, 1.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        scale_xyz=(1.0, 1.0, 1.0),
        light_dir=(0.4, 1.0, 0.2),
        light_intensity=1.0,
        bevel_strength=0.06,
    )
    s.set_text_mesh_material(0, metallic=0.6, roughness=0.35)

    s.add_text_mesh(
        "Animation",
        font_bytes,
        size_px=72.0,
        depth=0.18,
        position=(0.1, -0.15, 0.0),
        color=(0.2, 0.8, 1.0, 1.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        scale_xyz=(1.0, 0.9, 1.0),
        light_dir=(0.0, 1.0, 0.0),
        light_intensity=1.0,
        bevel_strength=0.08,
    )
    s.set_text_mesh_material(1, metallic=0.1, roughness=0.8)

    # Animate 120 frames
    frames = 120
    for f in range(frames):
        t = f / frames
        # Rotate first text around Y
        s.update_text_mesh_transform(
            0,
            position=(-0.6, 0.05, 0.0),
            rotation_deg=(0.0, (t * 360.0) % 360.0, 0.0),
            scale=None,
            scale_xyz=None,
        )
        # Wobble scale and rotate second text
        scl = 1.0 + 0.1 * math.sin(2.0 * math.pi * t * 2.0)
        s.update_text_mesh_transform(
            1,
            position=(0.1, -0.15, 0.0),
            rotation_deg=(10.0 * math.sin(2.0 * math.pi * t), 0.0, 12.0 * math.cos(2.0 * math.pi * t)),
            scale=scl,
            scale_xyz=(1.0, 0.9, 1.0),
        )
        # Sweep light direction
        lx = math.sin(2.0 * math.pi * t)
        lz = math.cos(2.0 * math.pi * t)
        s.update_text_mesh_light(0, dx=0.4 + 0.2 * lx, dy=1.0, dz=0.2 + 0.2 * lz, intensity=1.0)
        s.update_text_mesh_light(1, dx=0.0 + 0.3 * lx, dy=1.0, dz=0.0 + 0.3 * lz, intensity=1.0)

        out = out_dir / f"text3d_anim_{f:04d}.png"
        s.render_png(out)

    print(f"Wrote {frames} frames to {out_dir}")


if __name__ == "__main__":
    main()
