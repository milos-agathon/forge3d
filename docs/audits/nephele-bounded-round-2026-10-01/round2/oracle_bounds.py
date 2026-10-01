"""Upper bounds on what any single correction of the real-time frame could reach on G3/G4.

Run from the repository root:
    python docs/audits/nephele-bounded-round-2026-10-01/round2/oracle_bounds.py

All candidates are mapped back through the fixture's ACES + sRGB8 presentation and
scored with the production `_delta_e` and the unchanged mask rules. The oracle rows use
the reference itself to choose gains, so they bound (and cannot be) a physical fix.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from scripts.nephele_evidence_report import _delta_e  # noqa: E402

HERE = Path(__file__).resolve().parent
FO = HERE / "first-order"
FIX = Path("tests/nephele/fixture")
RT = Path("docs/audits/nephele-bounded-round-2026-10-01/background-local/realtime-rgb.npy")
RT_T = Path("docs/audits/nephele-bounded-round-2026-10-01/shadow/primary-transmittance.npy")

xs = np.linspace(0.0, 64.0, 2_000_001)


def aces(x):
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


ACES = aces(xs)


def to_linear(u8):
    v = u8.astype(np.float64) / 255.0
    return np.interp(np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4), ACES, xs)


def to_u8(lin):
    d = aces(lin)
    s = np.where(d <= 0.0031308, d * 12.92, 1.055 * d ** (1 / 2.4) - 0.055)
    return np.clip(np.round(s * 255), 0, 255).astype(np.uint8)


ref_u8 = np.load(FIX / "reference-rgb.npy")
rt_u8 = np.load(RT)
ref, rt = to_linear(ref_u8), to_linear(rt_u8)
T = np.load(RT_T).astype(np.float64)[..., :3]
fo = np.load(FO / "first-order.npz")
S2 = np.load(FO / "surface-cloudlight.npy")
terrain = np.load(FIX / "terrain-mask.npy").astype(bool)
shadow = terrain & np.load(FIX / "cloud-shadow-mask.npy").astype(bool)
sky = np.load(FIX / "sky-cloud-mask.npy").astype(bool)


def score(img_u8):
    d = _delta_e(img_u8, ref_u8)
    return {"g4_shadow_fraction_below_2": float(np.mean(d[shadow] < 2.0)),
            "g3_sky_cloud_fraction_below_2_5": float(np.mean(d[sky] < 2.5))}


def tile_gain(base):
    img = base.copy()
    for by in range(0, 64, 8):
        for bx in range(0, 64, 8):
            for m in (terrain, sky):
                tile = np.zeros_like(m)
                tile[by:by + 8, bx:bx + 8] = True
                tile &= m
                if tile.any():
                    img[tile] = base[tile] * (ref[tile].mean(0) / np.maximum(base[tile].mean(0), 1e-9))
    return img


def global_gain(base, m):
    img = base.copy()
    img[m] = base[m] * (ref[m].mean(0) / base[m].mean(0))
    return img


assert np.array_equal(to_u8(rt), rt_u8), "presentation inverse must round-trip the captured frame"
rows = {
    "realtime_background_fix_state": score(rt_u8),
    "reference_320spp_vs_640spp_noise_floor": score(np.load(FIX / "reference-rgb-spp320.npy")),
    "predicted_realtime_plus_T_times_surface_sun_inscatter": score(to_u8(rt + T * S2)),
    "oracle_realtime_global_rgb_gain_on_terrain": score(to_u8(global_gain(rt, terrain))),
    "oracle_realtime_global_rgb_gain_on_sky": score(to_u8(global_gain(rt, sky))),
    "oracle_realtime_8x8_tile_rgb_gain": score(to_u8(tile_gain(rt))),
    "oracle_first_order_structure_8x8_tile_gain": score(to_u8(tile_gain(fo["first_order"]))),
    "oracle_first_order_plus_surface_sun_inscatter_8x8_tile_gain":
        score(to_u8(tile_gain(fo["first_order"] + fo["T_cam"] * S2))),
}
summary = {
    "terrain_mean_rgb_reference_minus_realtime": (ref - rt)[terrain].mean(0).tolist(),
    "terrain_mean_rgb_T_times_surface_sun_inscatter": (T * S2)[terrain].mean(0).tolist(),
    "rows": rows,
}
print(json.dumps(summary, indent=2))
(FO / "oracle-bounds.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
