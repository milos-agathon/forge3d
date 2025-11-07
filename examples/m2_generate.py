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
import csv
import hashlib

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

WI3_ROUGHNESS = (0.1, 0.3, 0.5, 0.7, 0.9)

_LIGHT_DIR = np.array([0.5, 0.5, 1.0], dtype=np.float32)
_LIGHT_DIR /= np.linalg.norm(_LIGHT_DIR)

WI3_SAMPLE_TARGETS = {
    "center": (float(_LIGHT_DIR[2]), 1.0),
    "quarter_rim": (0.5, 0.6),
    "rim": (0.0, 0.4),
    "backface": (-0.2, 0.5),
    "offspec": (0.9530206138714227, 0.9530206138714227),
}

MAP_ORDER = ["center", "quarter_rim", "rim", "backface", "offspec"]

D_THRESHOLDS = {
    0.1: {"center": (0.55, 0.85), "quarter_rim": (0.10, 0.35), "rim": (0.0, 0.12), "backface": (0.0, 0.02), "offspec": (0.55, 0.85)},
    0.3: {"center": (0.35, 0.60), "quarter_rim": (0.10, 0.30), "rim": (0.0, 0.10), "backface": (0.0, 0.02), "offspec": (0.35, 0.60)},
    0.5: {"center": (0.22, 0.45), "quarter_rim": (0.08, 0.22), "rim": (0.0, 0.08), "backface": (0.0, 0.02), "offspec": (0.22, 0.45)},
    0.7: {"center": (0.12, 0.30), "quarter_rim": (0.06, 0.18), "rim": (0.0, 0.08), "backface": (0.0, 0.02), "offspec": (0.12, 0.30)},
    0.9: {"center": (0.07, 0.20), "quarter_rim": (0.05, 0.18), "rim": (0.0, 0.08), "backface": (0.0, 0.02), "offspec": (0.07, 0.20)},
}

G_THRESHOLDS = {
    0.1: {"center": (0.85, 1.00), "quarter_rim": (0.50, 0.75), "rim": (0.15, 0.35), "backface": (0.0, 0.02), "offspec": (0.85, 1.00)},
    0.3: {"center": (0.75, 0.95), "quarter_rim": (0.45, 0.70), "rim": (0.15, 0.35), "backface": (0.0, 0.02), "offspec": (0.75, 0.95)},
    0.5: {"center": (0.65, 0.90), "quarter_rim": (0.40, 0.65), "rim": (0.15, 0.35), "backface": (0.0, 0.02), "offspec": (0.65, 0.90)},
    0.7: {"center": (0.60, 0.85), "quarter_rim": (0.35, 0.60), "rim": (0.15, 0.35), "backface": (0.0, 0.02), "offspec": (0.60, 0.85)},
    0.9: {"center": (0.55, 0.80), "quarter_rim": (0.35, 0.60), "rim": (0.15, 0.35), "backface": (0.0, 0.02), "offspec": (0.55, 0.80)},
}

SPEC_FULL_BASELINE_SHA1 = "A34002ECBA9791043BE1D8398BF4F78A22749A26"

def _compute_wi3_metrics(img: np.ndarray) -> dict[str, float]:
    rgba = ensure_rgba_u8(img)
    h, w, _ = rgba.shape

    alpha = rgba[..., 3].astype(np.float32) / 255.0
    mask = alpha > 0.01
    if not np.any(mask):
        raise ValueError("Image mask empty for WI-3 sampling")

    ys, xs = np.nonzero(mask)
    cx = float(xs.mean())
    cy = float(ys.mean())
    radius = float(np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).max()))
    if radius <= 0.0:
        raise ValueError("Invalid radius computed for WI-3 sampling")

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    nx = (grid_x - cx) / radius
    ny = -(grid_y - cy) / radius
    r_sq = nx * nx + ny * ny
    inside = (r_sq <= 1.0) & mask

    nz = np.zeros_like(nx)
    nz[inside] = np.sqrt(np.clip(1.0 - r_sq[inside], 0.0, 1.0))

    raw_NoL = nx * _LIGHT_DIR[0] + ny * _LIGHT_DIR[1] + nz * _LIGHT_DIR[2]
    NoL = np.clip(raw_NoL, 0.0, 1.0)
    NoV = np.clip(nz, 0.0, 1.0)

    values: dict[str, float] = {}
    red = rgba[..., 0].astype(np.float32) / 255.0
    for name, (target_l, target_v) in WI3_SAMPLE_TARGETS.items():
        valid = inside.copy()
        if name == "backface":
            valid &= raw_NoL < 0.0
        else:
            valid &= raw_NoL >= 0.0
        if not np.any(valid):
            values[name] = 0.0
            continue
        metric = (NoL - target_l) ** 2 + (NoV - target_v) ** 2
        metric[~valid] = np.inf
        iy, ix = np.unravel_index(np.argmin(metric), metric.shape)
        values[name] = float(red[iy, ix])
    return values

def _write_wi3_csv(path: Path, metrics: List[tuple[float, dict[str, float]]]) -> None:
    header = ["roughness", "center", "quarter_rim", "rim", "backface", "offspec"]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r, samples in metrics:
            writer.writerow(
                [f"{r:.1f}"]
                + [f"{samples[name]:.6f}" for name in MAP_ORDER]
            )

def _evaluate_wi3_branch(name: str, metrics: List[tuple[float, dict[str, float]]], thresholds: dict[float, dict[str, tuple[float, float]]]) -> tuple[bool, List[str]]:
    failures: List[str] = []
    centers: List[tuple[float, float]] = []
    for roughness, samples in metrics:
        key = round(roughness, 1)
        branch_thr = thresholds.get(key)
        if branch_thr:
            for sample_name, (lo, hi) in branch_thr.items():
                value = samples[sample_name]
                if not (lo <= value <= hi):
                    failures.append(f"{name}: r={roughness:.1f} {sample_name}={value:.3f} expected in [{lo:.2f}, {hi:.2f}]")
        values = samples.values()
        if any(v > 1.0 + 1e-5 for v in values):
            failures.append(f"{name}: r={roughness:.1f} contains value > 1.0")
        if samples["backface"] > 0.02 + 1e-5:
            failures.append(f"{name}: r={roughness:.1f} backface={samples['backface']:.3f} expected <= 0.02")
        if samples["center"] <= samples["quarter_rim"] - 1e-3:
            failures.append(f"{name}: r={roughness:.1f} center should exceed quarter rim (center={samples['center']:.3f}, quarter={samples['quarter_rim']:.3f})")
        if name == "G-only" and samples["quarter_rim"] <= samples["rim"] - 1e-3:
            failures.append(f"{name}: r={roughness:.1f} quarter rim should exceed rim (quarter_rim={samples['quarter_rim']:.3f}, rim={samples['rim']:.3f})")
        centers.append((roughness, samples["center"]))

    centers.sort(key=lambda item: item[0])
    for i in range(len(centers) - 1):
        if centers[i][1] < centers[i + 1][1] - 0.02:
            failures.append(f"{name}: center luminance should decrease with roughness (r={centers[i][0]:.1f} -> {centers[i+1][0]:.1f})")

    return (len(failures) == 0, failures)

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
    wi3_mode: int | None = None,
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
        wi3_debug_mode=int(wi3_mode or 0),
        wi3_debug_roughness=float(r if wi3_mode else 0.0),
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

def wi3_debug_terms(model: str = "ggx", roughness: Tuple[float, ...] = WI3_ROUGHNESS, size: Tuple[int, int] = (256, 256)) -> bool:
    w, h = size
    d_metrics: List[tuple[float, dict[str, float]]] = []
    g_metrics: List[tuple[float, dict[str, float]]] = []
    tiles_D: List[np.ndarray] = []
    tiles_G: List[np.ndarray] = []
    tiles_spec: List[np.ndarray] = []

    for r in roughness:
        d_raw = np.array(render_tile(model, r, w, h, debug_d=False, output_mode=0, debug_no_srgb=True, wi3_mode=1), copy=True)
        d_metrics.append((r, _compute_wi3_metrics(d_raw)))
        tiles_D.append(draw_label_rgba(d_raw, f"D only  r={r:.1f}"))

        g_raw = np.array(render_tile(model, r, w, h, output_mode=0, debug_no_srgb=True, wi3_mode=2), copy=True)
        g_metrics.append((r, _compute_wi3_metrics(g_raw)))
        tiles_G.append(draw_label_rgba(g_raw, f"G only  r={r:.1f}"))

        spec_raw = np.array(render_tile(model, r, w, h, spec_only=False, debug_spec_no_nl=False, output_mode=0, debug_no_srgb=True, wi3_mode=3), copy=True)
        tiles_spec.append(draw_label_rgba(spec_raw, f"Spec BRDF  r={r:.1f}"))

    Image.fromarray(create_mosaic_row(tiles_D), "RGBA").save("m2_debug_D.png")
    Image.fromarray(create_mosaic_row(tiles_G), "RGBA").save("m2_debug_G.png")
    Image.fromarray(create_mosaic_row(tiles_spec), "RGBA").save("m2_debug_spec_full.png")

    _write_wi3_csv(Path("m2_debug_D.csv"), d_metrics)
    _write_wi3_csv(Path("m2_debug_G.csv"), g_metrics)

    d_pass, d_failures = _evaluate_wi3_branch("D-only", d_metrics, D_THRESHOLDS)
    g_pass, g_failures = _evaluate_wi3_branch("G-only", g_metrics, G_THRESHOLDS)

    spec_path = Path("m2_debug_spec_full.png")
    spec_hash = hashlib.sha1(spec_path.read_bytes()).hexdigest().upper() if spec_path.exists() else ""
    spec_pass = spec_hash == SPEC_FULL_BASELINE_SHA1

    for msg in d_failures:
        print(f"  [WI3][D-only] {msg}")
    for msg in g_failures:
        print(f"  [WI3][G-only] {msg}")
    if not spec_pass:
        print(f"  [WI3][Spec] SHA1 {spec_hash or 'N/A'} does not match baseline {SPEC_FULL_BASELINE_SHA1}")

    print(f"WI-3 D-only: {'PASS' if d_pass else 'FAIL'}")
    print(f"WI-3 G-only: {'PASS' if g_pass else 'FAIL'}")
    print(f"Spec-full unchanged checksum: {'PASS' if spec_pass else 'FAIL'}")

    return d_pass and g_pass and spec_pass

def wi3_debug_D(model='ggx', roughness=WI3_ROUGHNESS, size=(256,256)):
    w,h = size
    tiles = [draw_label_rgba(render_tile(model, r, w, h, debug_d=False, output_mode=0, debug_no_srgb=True, wi3_mode=1), f"D only  r={r:.1f}") for r in roughness]
    Image.fromarray(create_mosaic_row(tiles), 'RGBA').save('m2_debug_D.png')

def wi3_debug_G(model='ggx', roughness=WI3_ROUGHNESS, size=(256,256)):
    w,h = size
    tiles = [draw_label_rgba(render_tile(model, r, w, h, output_mode=0, debug_no_srgb=True, wi3_mode=2), f"G only  r={r:.1f}") for r in roughness]
    Image.fromarray(create_mosaic_row(tiles), 'RGBA').save('m2_debug_G.png')

def wi3_debug_spec(model='ggx', roughness=WI3_ROUGHNESS, size=(256,256)):
    w,h = size
    tiles = [draw_label_rgba(render_tile(model, r, w, h, spec_only=False, debug_spec_no_nl=False, output_mode=0, debug_no_srgb=True, wi3_mode=3), f"Spec BRDF  r={r:.1f}") for r in roughness]
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
    p.add_argument("--wi3", action="store_true", help="Run only WI-3 debug outputs with verification")
    p.add_argument("task", nargs="?", default=None, choices=["all","wi1","wi2","wi3","wi3_d","wi3_g","wi3_spec","wi4"], help="Which work item to run")
    args = p.parse_args()
    if args.wi3 and args.task not in (None, "wi3"):
        p.error("--wi3 cannot be combined with other task selections")
    task = "wi3" if args.wi3 else (args.task or "all")
    print(f"[M2] Generating outputs for {task}...")
    if task in ("all", "wi1"):
        wi1_linear_vs_srgb()
    if task in ("all", "wi2"):
        wi2_energy_split()
    if task in ("all", "wi3"):
        wi3_debug_terms()
    if task == "wi3_d":
        wi3_debug_D()
    if task == "wi3_g":
        wi3_debug_G()
    if task == "wi3_spec":
        wi3_debug_spec()
    if task in ("all", "wi4"):
        wi4_angle_sweep()


if __name__ == '__main__':
    main()
