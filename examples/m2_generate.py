#!/usr/bin/env python3
"""
M2 — BRDF hygiene and validation suite

Implements Work Items 1–5 from reqs.md using the offscreen GGX gallery tool.
Produces the required PNGs and prints concise diagnostics to the console.

Outputs:
  WI1 (Linear vs sRGB hygiene)
    - m2_flatness_check_linear.png
    - m2_flatness_check_srgb.png
  WI2 (Energy split kS + kD ≤ 1)
    - m2_energy_debug_nonmetal.png
    - m2_energy_debug_metal.png
  WI3 (D/G/spec verification)
    - m2_debug_D.png
    - m2_debug_G.png
    - m2_debug_spec_full.png
  WI4 (Angle-dependent sweep)
    - m2_angle_sweep_spec.png
    - m2_angle_sweep_diffuse.png
    - m2_angle_sweep_combined.png
  WI5 (Linear/sRGB hygiene, duplicated for clarity)
    - m2_linear_out.png
    - m2_srgb_out.png
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import forge3d as f3d

# -----------------------
# Image utilities
# -----------------------

def ensure_rgba_u8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected RGBA uint8 image, got {arr.shape} {arr.dtype}")
    return arr

def create_mosaic_row(tiles: List[np.ndarray], gap: int = 4, bg=(20,20,20,255)) -> np.ndarray:
    tiles = [ensure_rgba_u8(t) for t in tiles]
    h, w, _ = tiles[0].shape
    W = len(tiles)*w + (len(tiles)-1)*gap
    H = h
    out = np.full((H, W, 4), bg, dtype=np.uint8)
    for i, t in enumerate(tiles):
        x = i*(w+gap)
        out[:, x:x+w, :] = t
    return out

def draw_label_rgba(img: np.ndarray, text: str, xy=(6, 6), color=(255,255,255,255)) -> np.ndarray:
    im = Image.fromarray(ensure_rgba_u8(img).copy(), 'RGBA')
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    # black stroke for legibility
    draw.text(xy, text, fill=color, font=font, stroke_width=2, stroke_fill=(0,0,0,255))
    return np.asarray(im)

# sRGB helpers for diagnostics

def srgb_to_linear_u8(rgb_u8: np.ndarray) -> np.ndarray:
    x = (rgb_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)
    a = 0.055
    low = x <= 0.04045
    out = np.empty_like(x, dtype=np.float32)
    out[low] = x[low] / 12.92
    out[~low] = ((x[~low] + a) / (1 + a)) ** 2.4
    return out

# -----------------------
# Rendering wrappers
# -----------------------

def render_tile(
    model: str,
    r: float,
    w: int,
    h: int,
    *,
    mode: str | None = None,
    Li: float = 1.0,
    output_mode: int = 1,
    debug_no_srgb: bool = False,
    metallic_override: float = 0.0,
    base_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    spec_only: bool = False,
    debug_d: bool = False,
    debug_spec_no_nl: bool = False,
    debug_energy: bool = False,
    debug_angle_sweep: bool = False,
    debug_angle_component: int = 2,
) -> np.ndarray:
    return f3d.render_brdf_tile(
        model, float(r), int(w), int(h),
        ndf_only=False, g_only=False, dfg_only=False, spec_only=bool(spec_only),
        roughness_visualize=False,
        exposure=1.0, light_intensity=float(Li), base_color=base_color,
        clearcoat=0.0, clearcoat_roughness=0.0, sheen=0.0, sheen_tint=0.0, specular_tint=0.0,
        debug_dot_products=False,
        debug_lambert_only=(mode == "lambert" or mode == "flatness"),
        debug_d=bool(debug_d), debug_spec_no_nl=bool(debug_spec_no_nl), debug_energy=bool(debug_energy),
        debug_angle_sweep=bool(debug_angle_sweep), debug_angle_component=int(debug_angle_component),
        debug_no_srgb=bool(debug_no_srgb), output_mode=int(output_mode), metallic_override=float(metallic_override),
        mode=mode,
    )

# -----------------------
# WI1 + WI5: Linear vs sRGB hygiene
# -----------------------

def wi1_linear_vs_srgb(model="ggx", roughness=(0.1,0.3,0.5,0.7,0.9), size=(256,256), Li=3.0):
    w,h = size
    # Linear out (shader bypasses sRGB); lambert-only to eliminate specular for flatness
    tiles_lin = [render_tile(model, r, w, h, Li=Li, output_mode=0, debug_no_srgb=True, mode="flatness", base_color=(0.18,0.18,0.18)) for r in roughness]
    # sRGB out (shader applies sRGB once at end) from same pipeline/mode
    tiles_srgb = [render_tile(model, r, w, h, Li=Li, output_mode=1, debug_no_srgb=False, mode="flatness", base_color=(0.18,0.18,0.18)) for r in roughness]
    Image.fromarray(create_mosaic_row(tiles_lin), 'RGBA').save('m2_flatness_check_linear.png')
    Image.fromarray(create_mosaic_row(tiles_srgb), 'RGBA').save('m2_flatness_check_srgb.png')
    # Duplicate (WI5)
    Image.fromarray(create_mosaic_row(tiles_lin), 'RGBA').save('m2_linear_out.png')
    Image.fromarray(create_mosaic_row(tiles_srgb), 'RGBA').save('m2_srgb_out.png')

# -----------------------
# WI2: Energy split kS + kD ≤ 1
# -----------------------

def sample_points(img: np.ndarray) -> List[Tuple[int,int]]:
    h, w, _ = img.shape
    return [
        (w//2, h//2),                   # center
        (int(w*0.65), int(h*0.35)),     # near highlight
        (int(w*0.9), h//2),             # near terminator
    ]

def wi2_energy_split(size=(256,256)):
    w,h = size
    # Non-metal (metallic=0.0) — request linear output for numeric inspection
    nonmetal = render_tile('ggx', 0.5, w, h, debug_energy=True, output_mode=0, debug_no_srgb=True, metallic_override=0.0, base_color=(1.0,1.0,1.0))
    Image.fromarray(nonmetal, 'RGBA').save('m2_energy_debug_nonmetal.png')
    # Metal (metallic=1.0)
    metal = render_tile('ggx', 0.5, w, h, debug_energy=True, output_mode=0, debug_no_srgb=True, metallic_override=1.0, base_color=(1.0,0.0,0.0))
    Image.fromarray(metal, 'RGBA').save('m2_energy_debug_metal.png')
    # Print 3 sample pixels for A: kS (R), kD (G), sum (B) — already linear
    for name, img in [('nonmetal', nonmetal), ('metal', metal)]:
        rgba = img.astype(np.uint8)
        rgb = rgba[...,:3].astype(np.uint8)
        print(f"\n[WI2] {name} samples (kS, kD, kS+kD) at 3 points:")
        for (x,y) in sample_points(rgba):
            r,g,b = rgb[y,x].tolist()
            ks = r/255.0; kd = g/255.0; s = b/255.0
            print(f"  ({x:3d},{y:3d}) -> kS={ks:.3f}  kD={kd:.3f}  sum={s:.3f}")

# -----------------------
# WI3: Verify D, G (correlated), and spec full (with denominator)
# -----------------------

def wi3_debug_terms(model='ggx', roughness=(0.1,0.3,0.5,0.7,0.9), size=(256,256)):
    w,h = size
    tiles_D = [draw_label_rgba(render_tile(model, r, w, h, debug_d=True, output_mode=0, debug_no_srgb=True), f"D only  r={r:.1f}") for r in roughness]
    tiles_G = [draw_label_rgba(f3d.render_brdf_tile(model, float(r), w, h, g_only=True, output_mode=0, debug_no_srgb=True), f"G only  r={r:.1f}") for r in roughness]
    tiles_spec = [draw_label_rgba(render_tile(model, r, w, h, spec_only=True, debug_spec_no_nl=True, output_mode=0, debug_no_srgb=True), f"Spec BRDF  r={r:.1f}") for r in roughness]
    Image.fromarray(create_mosaic_row(tiles_D), 'RGBA').save('m2_debug_D.png')
    Image.fromarray(create_mosaic_row(tiles_G), 'RGBA').save('m2_debug_G.png')
    Image.fromarray(create_mosaic_row(tiles_spec), 'RGBA').save('m2_debug_spec_full.png')

def wi3_debug_D(model='ggx', roughness=(0.1,0.3,0.5,0.7,0.9), size=(256,256)):
    w,h = size
    tiles = [draw_label_rgba(render_tile(model, r, w, h, debug_d=True, output_mode=0, debug_no_srgb=True), f"D only  r={r:.1f}") for r in roughness]
    Image.fromarray(create_mosaic_row(tiles), 'RGBA').save('m2_debug_D.png')

def wi3_debug_G(model='ggx', roughness=(0.1,0.3,0.5,0.7,0.9), size=(256,256)):
    w,h = size
    tiles = [draw_label_rgba(f3d.render_brdf_tile(model, float(r), w, h, g_only=True, output_mode=0, debug_no_srgb=True), f"G only  r={r:.1f}") for r in roughness]
    Image.fromarray(create_mosaic_row(tiles), 'RGBA').save('m2_debug_G.png')

def wi3_debug_spec(model='ggx', roughness=(0.1,0.3,0.5,0.7,0.9), size=(256,256)):
    w,h = size
    tiles = [draw_label_rgba(render_tile(model, r, w, h, spec_only=True, debug_spec_no_nl=True, output_mode=0, debug_no_srgb=True), f"Spec BRDF  r={r:.1f}") for r in roughness]
    Image.fromarray(create_mosaic_row(tiles), 'RGBA').save('m2_debug_spec_full.png')

# -----------------------
# WI4: Angle-dependent validation (frontal → grazing sweep)
# -----------------------

def wi4_angle_sweep(size=(256,256)):
    w,h = size
    spec = render_tile('ggx', 0.5, w, h, debug_angle_sweep=True, debug_angle_component=0, output_mode=0, debug_no_srgb=True)
    diff = render_tile('ggx', 0.5, w, h, debug_angle_sweep=True, debug_angle_component=1, output_mode=0, debug_no_srgb=True)
    comb = render_tile('ggx', 0.5, w, h, debug_angle_sweep=True, debug_angle_component=2, output_mode=0, debug_no_srgb=True)
    Image.fromarray(spec, 'RGBA').save('m2_angle_sweep_spec.png')
    Image.fromarray(diff, 'RGBA').save('m2_angle_sweep_diffuse.png')
    Image.fromarray(comb, 'RGBA').save('m2_angle_sweep_combined.png')

# -----------------------
# Main
# -----------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="M2 generator: run WI1–WI4 individually or all")
    p.add_argument("task", nargs="?", default="all", choices=["all","wi1","wi2","wi3","wi3_d","wi3_g","wi3_spec","wi4"], help="Which work item to run")
    args = p.parse_args()
    print(f"[M2] Generating outputs for {args.task}...")
    if args.task in ("all", "wi1"):
        wi1_linear_vs_srgb()
    if args.task in ("all", "wi2"):
        wi2_energy_split()
    if args.task in ("all", "wi3"):
        wi3_debug_terms()
    if args.task == "wi3_d":
        wi3_debug_D()
    if args.task == "wi3_g":
        wi3_debug_G()
    if args.task == "wi3_spec":
        wi3_debug_spec()
    if args.task in ("all", "wi4"):
        wi4_angle_sweep()
    print("\n[M2] Done. Outputs written to current directory.")

if __name__ == '__main__':
    main()
