"""Compare reference and real-time radiance against deterministic first-order transport.

All radiance is linear, pre-exposure (fixture exposure is 1). Reference and
real-time beauty are recovered by inverting sRGB8 + fitted ACES; the u8
quantisation bound is reported alongside the means.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
FO = Path(sys.argv[2])
RT_BEAUTY = Path(sys.argv[3])      # real-time beauty (u8) at the state under test
RT_MEDIA = Path(sys.argv[4])       # folder with primary-in_scatter / transmittance / multiple luminance
FIX = ROOT / "tests/nephele/fixture"
LUMA = np.asarray([0.2126, 0.7152, 0.0722])

xs = np.linspace(0.0, 64.0, 2_000_001)
aces = np.clip((xs * (2.51 * xs + 0.03)) / (xs * (2.43 * xs + 0.59) + 0.14), 0, 1)


def to_linear(u8):
    v = u8.astype(np.float64) / 255.0
    display = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    return np.interp(display, aces, xs)


fo = np.load(FO / "first-order.npz")
ref = to_linear(np.load(FIX / "reference-rgb.npy"))
rt = to_linear(np.load(RT_BEAUTY))
rt_in = np.load(RT_MEDIA / "primary-in_scatter.npy").astype(np.float64)
rt_T = np.load(RT_MEDIA / "primary-transmittance.npy").astype(np.float64)
rt_ms_lum = np.load(RT_MEDIA / "primary-in_scatter_multiple_luminance.npy").astype(np.float64)

terrain = np.load(FIX / "terrain-mask.npy").astype(bool)
masks = {
    "cloud-shadowed terrain": terrain & np.load(FIX / "cloud-shadow-mask.npy").astype(bool),
    "lit terrain": terrain & ~np.load(FIX / "cloud-shadow-mask.npy").astype(bool),
    "sky/cloud": np.load(FIX / "sky-cloud-mask.npy").astype(bool),
}


def lum(a, m):
    return float((a[m] @ LUMA).mean()) if a.ndim == 3 else float(a[m].mean())


report = {}
for name, m in masks.items():
    fo_ss = fo["ss_sun"] + fo["ss_env"]
    first = fo["first_order"]
    rt_surface = np.where(terrain[..., None], (rt - rt_in) / np.maximum(rt_T, 1e-6), 0.0)
    row = {
        "pixels": int(m.sum()),
        "reference_beauty": lum(ref, m),
        "first_order_total": lum(first, m),
        "higher_order_remainder": lum(ref - first, m),
        "fo_camera_T": lum(fo["T_cam"], m),
        "fo_single_scatter_sun": lum(fo["ss_sun"], m),
        "fo_single_scatter_env": lum(fo["ss_env"], m),
        "fo_surface_first_bounce": lum(fo["surface1"], m),
        "fo_surface_sun": lum(fo["surf_sun"], m),
        "fo_surface_env": lum(fo["surf_env"], m),
        "realtime_beauty": lum(rt, m),
        "realtime_camera_T": lum(rt_T, m),
        "realtime_in_scatter_total": lum(rt_in, m),
        "realtime_in_scatter_multiple": float(rt_ms_lum[m].mean()),
        "realtime_in_scatter_single": lum(rt_in, m) - float(rt_ms_lum[m].mean()),
        "realtime_surface": lum(rt_surface, m) if name != "sky/cloud" else None,
        "fo_single_scatter_total": lum(fo_ss, m),
    }
    report[name] = row

print(json.dumps(report, indent=2))
(FO / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
