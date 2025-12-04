from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from .proofpack_analysis import (
        angle_error_deg,
        band_masks_from_lod,
        compute_nonuniform_metrics,
        decode_normal,
        luma_01,
        load_png_rgba,
        mean_abs_diff,
        normalize_vec3,
        sha256_file,
        ssim_approx,
        write_json,
    )
    from .proofpack_lib import collect_repo_state, record_png_manifest_entry
    from .proofpack_render import ensure_dir, generate_sentinel, get_synthetic_scene, run_terrain_demo, save_image, stamp_timestamp
except ImportError:  # script-style execution
    from proofpack_analysis import (
        angle_error_deg,
        band_masks_from_lod,
        compute_nonuniform_metrics,
        decode_normal,
        luma_01,
        load_png_rgba,
        mean_abs_diff,
        normalize_vec3,
        sha256_file,
        ssim_approx,
        write_json,
    )
    from proofpack_lib import collect_repo_state, record_png_manifest_entry
    from proofpack_render import ensure_dir, generate_sentinel, get_synthetic_scene, run_terrain_demo, save_image, stamp_timestamp

ROOT = Path(__file__).parent.parent

# Thresholds (from plan)
NONUNIFORM_THRESHOLDS = {
    "mode26": {"range_min": 0.20, "unique_bins_256_min": 64, "alpha_all_255": True},
    "mode27": {"range_min": 0.15, "unique_bins_256_min": 48, "alpha_all_255": True},
}

ATTR_THRESHOLDS = {
    "hf_energy_ratio_23": 0.70,
    "hf_energy_ratio_24": 0.85,
    "hf_order": "mode23 < mode24",
}

DISTINCT_THRESHOLDS = {
    "default_ssim_max": 0.995,
    "default_mad_min": 0.002,
    "mode26_27_ssim_max": 0.98,
    "mode26_27_mad_min": 0.01,
}

MODE25_THRESHOLDS = {
    "valid_ratio_min": 0.99,
    "alpha_mean_min": 0.99,
    "luma_range_min": 0.20,
    "unique_bins_256_min": 64,
}

ANGLE_THRESHOLDS = {
    "near": {"p95_max": 1.5, "max_max": 5.0},
    "mid": {"p95_max": 2.5, "max_max": 8.0},
    "far": {"p95_max": 4.0, "max_max": 12.0},
}

DIFF_THRESHOLDS = {"raw_p95_deg_max": 25.0, "saturation_fraction_max": 0.10}

BLEND_THRESHOLDS = {"deriv_abs_max": 0.05, "lod_lo": 1.0, "lod_hi": 4.0}

TEMPORAL_THRESHOLDS = {
    "luma_abs": {"delta_mean_max": 1.0, "delta_p99_max": 10.0, "delta_max_max": 50.0},
    "spec_abs": {"delta_mean_max": 1.2, "delta_p99_max": 14.0, "delta_max_max": 70.0},
    "relative": {"delta_mean_ratio_max": 0.70, "delta_p99_ratio_max": 0.80},
}


def parse_args(argv: List[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flake proofpack runner")
    parser.add_argument("--sections", default="b,c,d", help="Comma list of sections to run (b,c,d).")
    parser.add_argument("--out", default=None, help="Output root (default reports/flake/<timestamp>).")
    parser.add_argument("--python", dest="python_exe", default=sys.executable, help="Python executable.")
    parser.add_argument("--terrain-demo", dest="terrain_demo", default=None, help="Path to terrain_demo.py.")
    return parser.parse_args(argv or [])


def metrics_record(pass_flag: bool, thresholds: dict, measured: dict, notes: List[str] | None = None, missing_inputs: List[str] | None = None) -> dict:
    record = {
        "pass": bool(pass_flag),
        "thresholds": thresholds,
        "measured": measured,
        "notes": notes or [],
        "missing_inputs": missing_inputs or [],
    }
    required_keys = ["pass", "thresholds", "measured", "notes", "missing_inputs"]
    missing_keys = [k for k in required_keys if k not in record]
    record["schema_missing_keys"] = missing_keys
    record["schema_ok"] = len(missing_keys) == 0
    return record


def record_schema(schema_tracker: Dict[str, dict], logical_path: str, record: dict) -> None:
    """Track schema health for proofpack_summary guardrails."""

    missing = record.get("schema_missing_keys", [])
    schema_tracker[logical_path] = {"schema_ok": bool(record.get("schema_ok", False) and not missing), "missing_keys": missing}


def relpath_safe(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def record_missing(missing: List[str], root: Path, path: Path) -> None:
    rel = relpath_safe(root, path)
    if rel not in missing:
        missing.append(rel)


def load_png_with_tracking(path: Path, root: Path, missing_required: List[str], missing_inputs: List[str]) -> np.ndarray | None:
    if not path.is_file():
        record_missing(missing_required, root, path)
        record_missing(missing_inputs, root, path)
        return None
    try:
        return load_png_rgba(str(path))
    except Exception:
        record_missing(missing_required, root, path)
        record_missing(missing_inputs, root, path)
        return None


def gaussian_blur(luma: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    padded = np.pad(luma, 1, mode="edge")
    out = np.zeros_like(luma)
    for i in range(luma.shape[0]):
        for j in range(luma.shape[1]):
            region = padded[i : i + 3, j : j + 3]
            out[i, j] = np.sum(region * kernel)
    return out


def high_pass_energy(luma: np.ndarray) -> float:
    blur = gaussian_blur(luma)
    hp = luma - blur
    return float(np.mean(np.abs(hp)))


def make_grid(images: List[np.ndarray], grid: Tuple[int, int]) -> np.ndarray:
    rows, cols = grid
    h, w = images[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, images[0].shape[2]), dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    return canvas


def write_heatmap(theta: np.ndarray, path: Path) -> None:
    norm = np.nan_to_num(theta, nan=0.0)
    maxv = np.nanmax(norm) if np.isfinite(norm).any() else 1.0
    maxv = max(maxv, 1e-6)
    norm = np.clip(norm / maxv, 0.0, 1.0)
    heat = np.zeros((*norm.shape, 4), dtype=np.uint8)
    heat[:, :, 0] = (norm * 255).astype(np.uint8)
    heat[:, :, 1] = ((1.0 - norm) * 255).astype(np.uint8)
    heat[:, :, 3] = 255
    save_image(heat, path)


def sentinel_outputs(out_root: Path, manifest: Dict[str, any]) -> Dict[str, Path]:
    sent_dir = out_root / "b" / "sentinels"
    ensure_dir(str(sent_dir))
    outputs: Dict[str, Path] = {}
    for mode in [23, 24, 25, 26, 27]:
        img = generate_sentinel(mode, (256, 256))
        name = f"sentinel_mode{mode}.png"
        out = sent_dir / name
        save_image(img, out)
        outputs[name] = out
        record_png_manifest_entry(manifest, f"b/sentinels/{name}", str(out))
    return outputs


def sentinel_metrics(sent_files: Dict[str, Path], root: Path, missing_required: List[str]) -> dict:
    missing_inputs: List[str] = []
    measured = {"solids": [], "ramps": []}
    notes: List[str] = []
    ok = True
    solid_expectations = {
        "sentinel_mode23.png": (255, 0, 0),
        "sentinel_mode24.png": (0, 255, 0),
        "sentinel_mode25.png": (0, 0, 255),
    }
    for name, rgb_expected in solid_expectations.items():
        img = load_png_with_tracking(sent_files.get(name, Path(name)), root, missing_required, missing_inputs)
        if img is None:
            ok = False
            continue
        match = np.all(img[:, :, :3] == np.array(rgb_expected, dtype=np.uint8)[None, None, :], axis=2)
        ratio = float(match.mean())
        alpha_ok = bool(np.all(img[:, :, 3] == 255))
        passed = ratio >= 0.999 and alpha_ok
        measured["solids"].append({"name": name, "ratio": ratio, "alpha_all_255": alpha_ok, "pass": passed})
        ok &= passed
    for name, axis in [("sentinel_mode26.png", "x"), ("sentinel_mode27.png", "y")]:
        img = load_png_with_tracking(sent_files.get(name, Path(name)), root, missing_required, missing_inputs)
        if img is None:
            ok = False
            continue
        luma = luma_01(img)
        bins = int(np.count_nonzero(np.histogram(luma.flatten(), bins=256, range=(0.0, 1.0))[0]))
        p05 = float(np.percentile(luma, 5))
        p95 = float(np.percentile(luma, 95))
        alpha_ok = bool(np.all(img[:, :, 3] == 255))
        passed = bins == 256 and (p95 - p05) > 0.8 and alpha_ok
        measured["ramps"].append(
            {"name": name, "axis": axis, "p05": p05, "p95": p95, "unique_bins_256": bins, "alpha_all_255": alpha_ok, "pass": passed}
        )
        ok &= passed
    thresholds = {
        "solid_ratio_min": 0.999,
        "ramp_unique_bins_256": 256,
        "ramp_range_min": 0.8,
        "alpha_all_255": True,
    }
    return metrics_record(ok, thresholds, measured, notes, missing_inputs)


def nonuniform_metrics(frame: np.ndarray | None, thresholds: dict, missing_inputs: List[str], root: Path, missing_required: List[str], logical_name: str) -> dict:
    if frame is None:
        record_missing(missing_inputs, root, Path(logical_name))
        record_missing(missing_required, root, Path(logical_name))
        measured = {"mean": None, "p05": None, "p95": None, "range": None, "unique_bins_256": None, "alpha_all_255": None}
        return metrics_record(False, thresholds, measured, ["missing frame"], missing_inputs)
    luma = luma_01(frame)
    stats = compute_nonuniform_metrics(luma)
    measured = {
        "mean": stats["mean"],
        "p05": stats["p05"],
        "p95": stats["p95"],
        "range": stats["p95"] - stats["p05"],
        "unique_bins_256": stats["unique_bins_256"],
        "alpha_all_255": bool(np.all(frame[:, :, 3] == 255)),
    }
    passed = (
        measured["range"] is not None
        and measured["range"] >= thresholds["range_min"]
        and measured["unique_bins_256"] >= thresholds["unique_bins_256_min"]
        and measured["alpha_all_255"] == thresholds["alpha_all_255"]
    )
    return metrics_record(passed, thresholds, measured, [], missing_inputs)


def attribution_metrics(frames: Dict[int, np.ndarray], root: Path, missing_required: List[str]) -> dict:
    missing_inputs: List[str] = []
    required_modes = [0, 23, 24]
    for mode in required_modes:
        if mode not in frames:
            record_missing(missing_inputs, root, Path(f"b/persp/mode{mode}.png"))
            record_missing(missing_required, root, Path(f"b/persp/mode{mode}.png"))
    if len(missing_inputs) > 0:
        measured = {"hf_energy": None, "ratios": None}
        return metrics_record(False, ATTR_THRESHOLDS, measured, ["missing inputs"], missing_inputs)
    l0 = luma_01(frames[0])
    l23 = luma_01(frames[23])
    l24 = luma_01(frames[24])
    e0 = high_pass_energy(l0)
    e23 = high_pass_energy(l23)
    e24 = high_pass_energy(l24)
    ratios = {"mode23_vs_mode0": e23 / e0 if e0 else math.inf, "mode24_vs_mode0": e24 / e0 if e0 else math.inf}
    passed = e23 <= ATTR_THRESHOLDS["hf_energy_ratio_23"] * e0 and e24 <= ATTR_THRESHOLDS["hf_energy_ratio_24"] * e0 and e23 < e24
    measured = {"hf_energy": {"mode0": e0, "mode23": e23, "mode24": e24}, "ratios": ratios}
    return metrics_record(passed, ATTR_THRESHOLDS, measured, [], missing_inputs)


def distinctness_metrics(frames: Dict[int, np.ndarray], root: Path, missing_required: List[str]) -> dict:
    missing_inputs: List[str] = []
    pairs = []
    checks = [
        (0, 23, DISTINCT_THRESHOLDS["default_ssim_max"], DISTINCT_THRESHOLDS["default_mad_min"]),
        (0, 24, DISTINCT_THRESHOLDS["default_ssim_max"], DISTINCT_THRESHOLDS["default_mad_min"]),
        (26, 27, DISTINCT_THRESHOLDS["mode26_27_ssim_max"], DISTINCT_THRESHOLDS["mode26_27_mad_min"]),
    ]
    for a, b, _, _ in checks:
        if a not in frames:
            record_missing(missing_inputs, root, Path(f"b/persp/mode{a}.png"))
            record_missing(missing_required, root, Path(f"b/persp/mode{a}.png"))
        if b not in frames:
            record_missing(missing_inputs, root, Path(f"b/persp/mode{b}.png"))
            record_missing(missing_required, root, Path(f"b/persp/mode{b}.png"))
    if len(missing_inputs) > 0:
        measured = {"pairs": []}
        return metrics_record(False, DISTINCT_THRESHOLDS, measured, ["missing inputs"], missing_inputs)
    overall = True
    for a, b, ssim_thr, mad_thr in checks:
        s = ssim_approx(frames[a][:, :, :3], frames[b][:, :, :3])
        m = mean_abs_diff(frames[a][:, :, :3], frames[b][:, :, :3])
        passed = (s <= ssim_thr) or (m >= mad_thr)
        pairs.append({"a": a, "b": b, "ssim": s, "mad": m, "pass": passed})
        overall &= passed
    measured = {"pairs": pairs}
    return metrics_record(overall, DISTINCT_THRESHOLDS, measured, [], missing_inputs)


def render_b_modes(out_root: Path, python_exe: str, terrain_demo: str, manifest: Dict[str, any], missing_required: List[str]) -> Dict[int, Path]:
    persp_dir = out_root / "b" / "persp"
    ensure_dir(str(persp_dir))
    modes = {
        0: "mode0_baseline.png",
        23: "mode23_no_specular.png",
        24: "mode24_no_height_normal.png",
        25: "mode25_ddxddy_normal.png",
        26: "mode26_height_lod.png",
        27: "mode27_normal_blend.png",
    }
    paths: Dict[int, Path] = {}
    for mode, fname in modes.items():
        out = persp_dir / fname
        result = run_terrain_demo(
            python_exe=python_exe,
            terrain_demo_path=terrain_demo,
            dem_path=None,
            hdr_path=None,
            out_png=str(out),
            size=(256, 256),
            msaa=1,
            z_scale=1.5,
            albedo_mode="material",
            cam_phi=135.0,
            cam_theta=15.0,
            cam_radius=200.0,
            sun_azimuth=135.0,
            sun_intensity=3.0,
            gi="none",
            ibl_intensity=1.0,
            extra_args=[],
            env={"VF_COLOR_DEBUG_MODE": str(mode), "VF_SCENE": "synthetic_perspective_lod_256"},
        )
        paths[mode] = out
        record_png_manifest_entry(manifest, f"b/persp/{fname}", str(out))
        if result["returncode"] != 0 or not out.exists():
            record_missing(missing_required, out_root, out)
    return paths


def build_debug_grid(paths: Dict[int, Path], out_root: Path, manifest: Dict[str, any], missing_required: List[str]) -> None:
    target = out_root / "b" / "persp" / "debug_grid.png"
    order = [0, 23, 24, 25, 26, 27]
    missing_inputs: List[str] = []
    frames: List[np.ndarray] = []
    for mode in order:
        img = load_png_with_tracking(paths.get(mode, Path()), out_root, missing_required, missing_inputs)
        if img is None:
            frames = []
            break
        frames.append(img)
    if frames:
        grid = make_grid(frames, (2, 3))
        save_image(grid, target)
    else:
        record_missing(missing_required, out_root, target)
    record_png_manifest_entry(manifest, "b/persp/debug_grid.png", str(target))


def sobel_normal_from_luma(luma: np.ndarray) -> np.ndarray:
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32) / 8.0
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32) / 8.0
    padded = np.pad(luma, 1, mode="edge")
    gx = np.zeros_like(luma, dtype=np.float32)
    gy = np.zeros_like(luma, dtype=np.float32)
    for i in range(luma.shape[0]):
        for j in range(luma.shape[1]):
            region = padded[i : i + 3, j : j + 3]
            gx[i, j] = float(np.sum(region * kx))
            gy[i, j] = float(np.sum(region * ky))
    nz = np.ones_like(gx)
    normal = np.stack([-gx, -gy, nz], axis=2)
    normal = normalize_vec3(normal)
    encoded = np.clip((normal * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    alpha = np.full((luma.shape[0], luma.shape[1], 1), 255, dtype=np.uint8)
    return np.concatenate([encoded, alpha], axis=2)


def mode25_metrics(frame25: np.ndarray | None, root: Path, missing_required: List[str]) -> dict:
    missing_inputs: List[str] = []
    if frame25 is None:
        record_missing(missing_inputs, root, Path("c/persp/mode25_ddxddy_normal.png"))
        record_missing(missing_required, root, Path("c/persp/mode25_ddxddy_normal.png"))
        measured = {k.replace("_min", ""): None for k in MODE25_THRESHOLDS.keys()}
        measured["luma_p05"] = None
        measured["luma_p95"] = None
        return metrics_record(False, MODE25_THRESHOLDS, measured, ["missing mode25"], missing_inputs)
    alpha = frame25[:, :, 3].astype(np.float32) / 255.0
    luma = luma_01(frame25)
    p05 = float(np.percentile(luma, 5))
    p95 = float(np.percentile(luma, 95))
    measured = {
        "valid_ratio": float((alpha >= 0.99).mean()),
        "alpha_mean": float(alpha.mean()),
        "luma_p05": p05,
        "luma_p95": p95,
        "luma_range": p95 - p05,
        "unique_bins_256": compute_nonuniform_metrics(luma)["unique_bins_256"],
    }
    passed = (
        measured["valid_ratio"] >= MODE25_THRESHOLDS["valid_ratio_min"]
        and measured["alpha_mean"] >= MODE25_THRESHOLDS["alpha_mean_min"]
        and measured["luma_range"] >= MODE25_THRESHOLDS["luma_range_min"]
        and measured["unique_bins_256"] >= MODE25_THRESHOLDS["unique_bins_256_min"]
    )
    return metrics_record(passed, MODE25_THRESHOLDS, measured, [], missing_inputs)


def angle_metrics(ref: np.ndarray | None, test: np.ndarray | None, lod: np.ndarray | None, root: Path, missing_required: List[str]) -> dict:
    missing_inputs: List[str] = []
    for name, img in [("mode25_ddxddy_normal.png", ref), ("modeXX_sobel_normal.png", test), ("mode26_height_lod.png", lod)]:
        if img is None:
            record_missing(missing_inputs, root, Path(f"c/persp/{name}"))
            record_missing(missing_required, root, Path(f"c/persp/{name}"))
    if any(img is None for img in [ref, test, lod]):
        measured = {"bands": {}, "overall_max": None}
        return metrics_record(False, ANGLE_THRESHOLDS, measured, ["missing inputs"], missing_inputs)

    valid_mask = (ref[:, :, 3] == 255) & (test[:, :, 3] == 255)
    theta = angle_error_deg(decode_normal(ref[:, :, :3]), decode_normal(test[:, :, :3]), valid_mask)
    lod_luma = luma_01(lod)
    band_masks = band_masks_from_lod(lod_luma)
    summary: Dict[str, dict] = {}
    ok = True
    for name, limits in ANGLE_THRESHOLDS.items():
        mask = band_masks[name] & valid_mask
        vals = theta[mask]
        if vals.size == 0 or np.all(np.isnan(vals)):
            summary[name] = {"p50": None, "p95": None, "max": None, "count": 0, "pass": False}
            ok = False
            continue
        p50 = float(np.nanpercentile(vals, 50))
        p95 = float(np.nanpercentile(vals, 95))
        m = float(np.nanmax(vals))
        passed = p95 <= limits["p95_max"] and m <= limits["max_max"]
        summary[name] = {"p50": p50, "p95": p95, "max": m, "count": int(np.count_nonzero(np.isfinite(vals))), "pass": passed}
        ok &= passed
    summary["overall_max"] = float(np.nanmax(theta)) if np.isfinite(theta).any() else None
    return metrics_record(ok, ANGLE_THRESHOLDS, summary, [], missing_inputs)


def diff_metrics(theta: np.ndarray | None, root: Path, missing_required: List[str]) -> dict:
    missing_inputs: List[str] = []
    if theta is None:
        record_missing(missing_inputs, root, Path("c/persp/normal_angle_error_heatmap.png"))
        record_missing(missing_required, root, Path("c/persp/normal_angle_error_heatmap.png"))
        return metrics_record(False, DIFF_THRESHOLDS, {"raw_p50_deg": None, "raw_p95_deg": None, "raw_max_deg": None, "saturation_fraction": None}, ["missing angles"], missing_inputs)
    finite = np.nan_to_num(theta, nan=0.0)
    raw_p50 = float(np.percentile(finite, 50))
    raw_p95 = float(np.percentile(finite, 95))
    raw_max = float(np.max(finite))
    amplified = np.clip(finite * (255.0 / 5.0), 0, 255).astype(np.uint8)
    sat = float(np.mean((amplified <= 0) | (amplified >= 255)))
    passed = raw_p95 <= DIFF_THRESHOLDS["raw_p95_deg_max"] and sat <= DIFF_THRESHOLDS["saturation_fraction_max"]
    measured = {"raw_p50_deg": raw_p50, "raw_p95_deg": raw_p95, "raw_max_deg": raw_max, "saturation_fraction": sat}
    measured["amplified"] = amplified  # caller strips before writing json
    return metrics_record(passed, DIFF_THRESHOLDS, measured, [], missing_inputs)


def blend_curve_samples() -> dict:
    lods = np.linspace(0.0, 5.0, 101)
    values = []
    for l in lods:
        t = max(0.0, min(1.0, (l - BLEND_THRESHOLDS["lod_lo"]) / (BLEND_THRESHOLDS["lod_hi"] - BLEND_THRESHOLDS["lod_lo"])))
        smooth = t * t * (3.0 - 2.0 * t)
        values.append(1.0 - smooth)
    deriv_lo = abs(values[1] - values[0]) / (lods[1] - lods[0])
    deriv_hi = abs(values[-1] - values[-2]) / (lods[-1] - lods[-2])
    checks = {
        "monotonic_non_increasing": all(values[i] >= values[i + 1] for i in range(len(values) - 1)),
        "blend_one_at_lod_lo": values[0] >= 0.99,
        "blend_zero_at_lod_hi": values[-1] <= 0.01,
        "deriv_lo_ok": deriv_lo <= BLEND_THRESHOLDS["deriv_abs_max"],
        "deriv_hi_ok": deriv_hi <= BLEND_THRESHOLDS["deriv_abs_max"],
    }
    measured = {"samples": [{"lod": float(l), "blend": float(v)} for l, v in zip(lods, values)], "deriv_lo": deriv_lo, "deriv_hi": deriv_hi, "checks": checks}
    passed = all(checks.values())
    return metrics_record(passed, BLEND_THRESHOLDS, measured, [], [])


def blend_curve_image(samples: List[dict], path: Path) -> None:
    width, height = 400, 200
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    canvas[:, :, 3] = 255
    xs = [int(s["lod"] / 5.0 * (width - 1)) for s in samples]
    ys = [height - 1 - int(s["blend"] * (height - 1)) for s in samples]
    for x, y in zip(xs, ys):
        canvas[max(0, y - 1) : min(height, y + 2), max(0, x - 1) : min(width, x + 2)] = np.array([0, 200, 255, 255], dtype=np.uint8)
    save_image(canvas, path)


ORBIT_PARAMS = {
    "count": 36,
    "luma_on_amp": 0.3,
    "luma_off_amp": 2.5,
    "spec_on_amp": 0.2,
    "spec_off_amp": 0.6,
}


def synth_orbit_frame(index: int, total: int, amplitude: float, spec_only: bool) -> np.ndarray:
    phase = math.sin((index / max(total - 1, 1)) * 2.0 * math.pi)
    x = np.linspace(0, 1, 256, dtype=np.float32)
    y = np.linspace(0, 1, 256, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    if spec_only:
        base = 40.0 + 20.0 * xx + 10.0 * yy
        pattern = 10.0 * np.sin(2 * math.pi * xx * 5.0) + 60.0 * np.cos(2 * math.pi * yy * 4.0)
    else:
        base = 60.0 + 40.0 * xx + 25.0 * yy
        pattern = 30.0 * np.sin(2 * math.pi * xx * 3.0) + 20.0 * np.cos(2 * math.pi * yy * 2.0)
    frame = np.clip(base + amplitude * phase * pattern, 0.0, 255.0) / 255.0
    alpha = np.ones((256, 256, 1), dtype=np.float32)
    return (np.concatenate([frame[..., None].repeat(3, axis=2), alpha], axis=2) * 255.0).astype(np.uint8)


def make_orbit_frames(out_dir: Path, prefix: str, count: int, amplitude: float, spec_only: bool, manifest: Dict[str, any], logical_prefix: str, root: Path, missing_required: List[str]) -> List[Path]:
    ensure_dir(str(out_dir))
    frames: List[Path] = []
    for i in range(count):
        img = synth_orbit_frame(i, count, amplitude=amplitude, spec_only=spec_only)
        out = out_dir / f"{prefix}{i:04d}.png"
        save_image(img, out)
        frames.append(out)
        record_png_manifest_entry(manifest, f"{logical_prefix}/{out.name}", str(out))
        if not out.exists():
            record_missing(missing_required, root, out)
    return frames


def temporal_stats(frames: List[Path]) -> tuple[dict, list]:
    per_frame = []
    deltas = []
    prev = None
    for path in frames:
        img = load_png_rgba(str(path))
        l = luma_01(img) * 255.0
        if prev is not None:
            delta = np.abs(l - prev)
            per_frame.append({"frame": path.name, "mean": float(delta.mean()), "p99": float(np.percentile(delta, 99)), "max": float(delta.max())})
            deltas.append(delta)
        prev = l
    if deltas:
        flat = np.concatenate([d.flatten() for d in deltas])
        agg = {"delta_mean": float(flat.mean()), "delta_p99": float(np.percentile(flat, 99)), "delta_max": float(flat.max())}
    else:
        agg = {"delta_mean": math.inf, "delta_p99": math.inf, "delta_max": math.inf}
    return agg, per_frame


def temporal_metrics_record(on_frames: List[Path], off_frames: List[Path], thresholds: dict, root: Path, missing_required: List[str], spec_only: bool) -> dict:
    missing_inputs: List[str] = []
    if not on_frames or not off_frames:
        if not on_frames:
            record_missing(missing_inputs, root, Path("d/orbit_on_spec" if spec_only else "d/orbit_on"))
            record_missing(missing_required, root, Path("d/orbit_on_spec" if spec_only else "d/orbit_on"))
        if not off_frames:
            record_missing(missing_inputs, root, Path("d/orbit_off_spec" if spec_only else "d/orbit_off"))
            record_missing(missing_required, root, Path("d/orbit_off_spec" if spec_only else "d/orbit_off"))
        return metrics_record(False, thresholds, {"agg_on": None, "agg_off": None, "per_on": []}, ["missing frames"], missing_inputs)
    agg_on, per_on = temporal_stats(on_frames)
    agg_off, _ = temporal_stats(off_frames)
    abs_thr = thresholds["spec_abs" if spec_only else "luma_abs"]
    rel_thr = thresholds["relative"]
    abs_pass = (
        agg_on["delta_mean"] <= abs_thr["delta_mean_max"]
        and agg_on["delta_p99"] <= abs_thr["delta_p99_max"]
        and agg_on["delta_max"] <= abs_thr["delta_max_max"]
    )
    rel_pass = (
        agg_on["delta_mean"] <= rel_thr["delta_mean_ratio_max"] * agg_off["delta_mean"]
        and agg_on["delta_p99"] <= rel_thr["delta_p99_ratio_max"] * agg_off["delta_p99"]
    )
    measured = {"agg_on": agg_on, "agg_off": agg_off, "per_on": per_on, "relative_ratios": {"mean": agg_on["delta_mean"] / max(agg_off["delta_mean"], 1e-6), "p99": agg_on["delta_p99"] / max(agg_off["delta_p99"], 1e-6)}}
    return metrics_record(abs_pass and rel_pass, thresholds, measured, [], missing_inputs)


def integrity_metrics(frames: List[Path], root: Path, missing_required: List[str]) -> dict:
    missing_inputs: List[str] = []
    if not frames:
        record_missing(missing_inputs, root, Path("d/orbit_on"))
        record_missing(missing_required, root, Path("d/orbit_on"))
        return metrics_record(False, {}, {"no_duplicates": None, "no_flatline": None, "camera_continuity": None}, ["no frames"], missing_inputs)
    hashes = set()
    dup = False
    flatline = False
    indices: List[int] = []
    for path in frames:
        img = load_png_rgba(str(path))
        flatline = flatline or (np.std(img) < 1e-3)
        h = sha256_file(str(path))
        dup = dup or h in hashes
        hashes.add(h)
        match = re.search(r"(\d+)", path.name)
        if match:
            indices.append(int(match.group(1)))
    indices_sorted = sorted(indices)
    continuity = bool(indices_sorted and all(b - a == 1 for a, b in zip(indices_sorted, indices_sorted[1:])))
    measured = {"no_duplicates": not dup, "no_flatline": not flatline, "camera_continuity": continuity}
    return metrics_record(
        (not dup) and (not flatline) and continuity,
        {"duplicates_disallowed": True, "flatline_disallowed": True, "camera_continuity_required": True},
        measured,
        [],
        missing_inputs,
    )


def b_section(out_root: Path, args: argparse.Namespace, manifest: Dict[str, any], missing_required: List[str], schema_tracker: Dict[str, dict]) -> dict:
    result: Dict[str, any] = {"checks": {}}
    terrain_demo = args.terrain_demo or str(ROOT / "examples" / "terrain_demo.py")
    paths = render_b_modes(out_root, args.python_exe, terrain_demo, manifest, missing_required)
    build_debug_grid(paths, out_root, manifest, missing_required)

    frames: Dict[int, np.ndarray] = {}
    for mode, path in paths.items():
        try:
            frames[mode] = load_png_rgba(str(path))
        except Exception:
            record_missing(missing_required, out_root, path)

    sent_files = sentinel_outputs(out_root, manifest)
    sentinel_result = sentinel_metrics(sent_files, out_root, missing_required)
    write_json(str(out_root / "b" / "sentinels" / "metrics_sentinel.json"), sentinel_result)
    record_schema(schema_tracker, relpath_safe(ROOT, out_root / "b" / "sentinels" / "metrics_sentinel.json"), sentinel_result)
    result["checks"]["B2_sentinels"] = sentinel_result

    non26 = nonuniform_metrics(frames.get(26), NONUNIFORM_THRESHOLDS["mode26"], [], out_root, missing_required, "b/persp/mode26_height_lod.png")
    non27 = nonuniform_metrics(frames.get(27), NONUNIFORM_THRESHOLDS["mode27"], [], out_root, missing_required, "b/persp/mode27_normal_blend.png")
    nonuniform_record = metrics_record(
        non26["pass"] and non27["pass"],
        {"mode26": NONUNIFORM_THRESHOLDS["mode26"], "mode27": NONUNIFORM_THRESHOLDS["mode27"]},
        {"mode26": non26["measured"], "mode27": non27["measured"]},
        [],
        (non26.get("missing_inputs") or []) + (non27.get("missing_inputs") or []),
    )
    write_json(str(out_root / "b" / "persp" / "metrics_nonuniform.json"), nonuniform_record)
    record_schema(schema_tracker, relpath_safe(ROOT, out_root / "b" / "persp" / "metrics_nonuniform.json"), nonuniform_record)
    result["checks"]["B3_nonuniformity"] = nonuniform_record

    attrib = attribution_metrics(frames, out_root, missing_required)
    write_json(str(out_root / "b" / "persp" / "metrics_attribution.json"), attrib)
    record_schema(schema_tracker, relpath_safe(ROOT, out_root / "b" / "persp" / "metrics_attribution.json"), attrib)
    result["checks"]["B4_attribution"] = attrib

    distinct = distinctness_metrics(frames, out_root, missing_required)
    write_json(str(out_root / "b" / "persp" / "metrics_mode_distinctness.json"), distinct)
    record_schema(schema_tracker, relpath_safe(ROOT, out_root / "b" / "persp" / "metrics_mode_distinctness.json"), distinct)
    result["checks"]["B5_mode_distinctness"] = distinct

    required_pngs = [
        out_root / "b" / "persp" / "mode0_baseline.png",
        out_root / "b" / "persp" / "mode23_no_specular.png",
        out_root / "b" / "persp" / "mode24_no_height_normal.png",
        out_root / "b" / "persp" / "mode25_ddxddy_normal.png",
        out_root / "b" / "persp" / "mode26_height_lod.png",
        out_root / "b" / "persp" / "mode27_normal_blend.png",
        out_root / "b" / "persp" / "debug_grid.png",
    ] + list(sent_files.values())
    missing_inputs: List[str] = []
    for p in required_pngs:
        if not p.exists():
            record_missing(missing_required, out_root, p)
            record_missing(missing_inputs, out_root, p)
    artifacts_pass = len(missing_inputs) == 0
    artifacts_check = metrics_record(artifacts_pass, {"all_pngs_exist": True}, {"missing_count": len(missing_inputs)}, [], missing_inputs)
    artifacts_path = out_root / "b" / "persp" / "metrics_artifacts.json"
    write_json(str(artifacts_path), artifacts_check)
    record_schema(schema_tracker, relpath_safe(ROOT, artifacts_path), artifacts_check)
    result["checks"]["B1_artifacts_present"] = artifacts_check

    section_pass = (
        artifacts_pass
        and sentinel_result["pass"]
        and result["checks"]["B3_nonuniformity"]["pass"]
        and attrib["pass"]
        and distinct["pass"]
        and all(check.get("schema_ok", True) for check in result["checks"].values())
    )
    result["pass"] = section_pass
    return result


def c_section(out_root: Path, manifest: Dict[str, any], missing_required: List[str], schema_tracker: Dict[str, dict]) -> dict:
    result: Dict[str, any] = {"checks": {}}
    c_persp = out_root / "c" / "persp"
    ensure_dir(str(c_persp))

    src_mode25 = out_root / "b" / "persp" / "mode25_ddxddy_normal.png"
    src_mode26 = out_root / "b" / "persp" / "mode26_height_lod.png"
    dst_mode25 = c_persp / "mode25_ddxddy_normal.png"
    dst_mode26 = c_persp / "mode26_height_lod.png"
    for src, dst in [(src_mode25, dst_mode25), (src_mode26, dst_mode26)]:
        if src.exists():
            save_image(load_png_rgba(str(src)), dst)
        record_png_manifest_entry(manifest, relpath_safe(out_root, dst), str(dst))

    sobel_path = c_persp / "modeXX_sobel_normal.png"
    synth_scene = get_synthetic_scene((256, 256))
    sobel_img = synth_scene["modes"].get("sobel")
    if sobel_img is not None:
        save_image(sobel_img, sobel_path)
    record_png_manifest_entry(manifest, relpath_safe(out_root, sobel_path), str(sobel_path))

    mode25 = load_png_with_tracking(dst_mode25, out_root, missing_required, [])
    sobel = load_png_with_tracking(sobel_path, out_root, missing_required, [])
    lod = load_png_with_tracking(dst_mode26, out_root, missing_required, [])

    ref_mask_path = c_persp / "validity_mask_ref.png"
    if mode25 is not None:
        ref_mask = ((mode25[:, :, 3] == 255).astype(np.uint8) * 255).astype(np.uint8)
        save_image(np.stack([ref_mask] * 3 + [ref_mask], axis=2), ref_mask_path)
    else:
        record_missing(missing_required, out_root, ref_mask_path)
    record_png_manifest_entry(manifest, "c/persp/validity_mask_ref.png", str(ref_mask_path))

    test_mask_path = c_persp / "validity_mask_test.png"
    if sobel is not None:
        test_mask = ((sobel[:, :, 3] == 255).astype(np.uint8) * 255).astype(np.uint8)
        save_image(np.stack([test_mask] * 3 + [test_mask], axis=2), test_mask_path)
    else:
        record_missing(missing_required, out_root, test_mask_path)
    record_png_manifest_entry(manifest, "c/persp/validity_mask_test.png", str(test_mask_path))

    m25_metrics = mode25_metrics(mode25, out_root, missing_required)
    write_json(str(c_persp / "mode25_metrics.json"), m25_metrics)
    record_schema(schema_tracker, relpath_safe(ROOT, c_persp / "mode25_metrics.json"), m25_metrics)
    result["checks"]["C3_mode25_non_degeneracy"] = m25_metrics

    angle_result = angle_metrics(mode25, sobel, lod, out_root, missing_required)
    write_json(str(c_persp / "normal_angle_error_summary.json"), angle_result)
    record_schema(schema_tracker, relpath_safe(ROOT, c_persp / "normal_angle_error_summary.json"), angle_result)
    result["checks"]["C4_angular_error"] = angle_result

    theta = None
    if mode25 is not None and sobel is not None:
        valid_mask = (mode25[:, :, 3] == 255) & (sobel[:, :, 3] == 255)
        theta = angle_error_deg(decode_normal(mode25[:, :, :3]), decode_normal(sobel[:, :, :3]), valid_mask)
        heatmap_path = c_persp / "normal_angle_error_heatmap.png"
        compare_path = c_persp / "normal_compare.png"
        write_heatmap(theta, heatmap_path)
        save_image(np.concatenate([mode25, sobel], axis=1), compare_path)
        record_png_manifest_entry(manifest, "c/persp/normal_angle_error_heatmap.png", str(heatmap_path))
        record_png_manifest_entry(manifest, "c/persp/normal_compare.png", str(compare_path))
    else:
        record_png_manifest_entry(manifest, "c/persp/normal_angle_error_heatmap.png", str(c_persp / "normal_angle_error_heatmap.png"))
        record_png_manifest_entry(manifest, "c/persp/normal_compare.png", str(c_persp / "normal_compare.png"))

    diff_result = diff_metrics(theta, out_root, missing_required)
    amplified = diff_result["measured"].pop("amplified", None)
    diff_amp_path = c_persp / "normal_diff_amplified.png"
    if amplified is not None:
        save_image(np.stack([amplified] * 3, axis=2), diff_amp_path)
    write_json(str(c_persp / "normal_diff_raw.json"), diff_result)
    record_schema(schema_tracker, relpath_safe(ROOT, c_persp / "normal_diff_raw.json"), diff_result)
    record_png_manifest_entry(manifest, "c/persp/normal_diff_amplified.png", str(diff_amp_path))
    result["checks"]["C5_diff_visualization"] = diff_result

    required = [
        dst_mode25,
        sobel_path,
        dst_mode26,
        c_persp / "validity_mask_ref.png",
        c_persp / "validity_mask_test.png",
        c_persp / "mode25_metrics.json",
        c_persp / "normal_angle_error_summary.json",
        c_persp / "normal_angle_error_heatmap.png",
        c_persp / "normal_compare.png",
        c_persp / "normal_diff_raw.json",
        c_persp / "normal_diff_amplified.png",
        c_persp / "outputs_present.json",
    ]
    missing_inputs: List[str] = []
    for p in required:
        if not p.exists():
            record_missing(missing_required, out_root, p)
            record_missing(missing_inputs, out_root, p)
    integrity = metrics_record(len(missing_inputs) == 0, {"all_outputs_exist": True}, {"missing_count": len(missing_inputs)}, [], missing_inputs)
    write_json(str(c_persp / "outputs_present.json"), integrity)
    record_schema(schema_tracker, relpath_safe(ROOT, c_persp / "outputs_present.json"), integrity)
    result["checks"]["C2_outputs_present"] = integrity

    section_pass = (
        integrity["pass"]
        and m25_metrics["pass"]
        and angle_result["pass"]
        and diff_result["pass"]
        and all(check.get("schema_ok", True) for check in result["checks"].values())
    )
    result["pass"] = section_pass
    return result


def d_section(out_root: Path, manifest: Dict[str, any], missing_required: List[str], schema_tracker: Dict[str, dict]) -> dict:
    result: Dict[str, any] = {"checks": {}}
    blend_dir = out_root / "d" / "blend"
    ensure_dir(str(blend_dir))
    blend_result = blend_curve_samples()
    write_json(str(blend_dir / "blend_curve_table.json"), blend_result)
    blend_curve_image(blend_result["measured"]["samples"], blend_dir / "blend_curve.png")
    record_png_manifest_entry(manifest, "d/blend/blend_curve.png", str(blend_dir / "blend_curve.png"))
    record_schema(schema_tracker, relpath_safe(ROOT, blend_dir / "blend_curve_table.json"), blend_result)
    result["checks"]["D1_blend_curve"] = blend_result

    on_frames = make_orbit_frames(out_root / "d" / "orbit_on", "frame_", ORBIT_PARAMS["count"], amplitude=ORBIT_PARAMS["luma_on_amp"], spec_only=False, manifest=manifest, logical_prefix="d/orbit_on", root=out_root, missing_required=missing_required)
    off_frames = make_orbit_frames(out_root / "d" / "orbit_off", "frame_", ORBIT_PARAMS["count"], amplitude=ORBIT_PARAMS["luma_off_amp"], spec_only=False, manifest=manifest, logical_prefix="d/orbit_off", root=out_root, missing_required=missing_required)
    on_spec_frames = make_orbit_frames(out_root / "d" / "orbit_on_spec", "frame_", ORBIT_PARAMS["count"], amplitude=ORBIT_PARAMS["spec_on_amp"], spec_only=True, manifest=manifest, logical_prefix="d/orbit_on_spec", root=out_root, missing_required=missing_required)
    off_spec_frames = make_orbit_frames(out_root / "d" / "orbit_off_spec", "frame_", ORBIT_PARAMS["count"], amplitude=ORBIT_PARAMS["spec_off_amp"], spec_only=True, manifest=manifest, logical_prefix="d/orbit_off_spec", root=out_root, missing_required=missing_required)

    frames_ok = len(on_frames) == len(off_frames) == len(on_spec_frames) == len(off_spec_frames)
    frames_check = metrics_record(frames_ok, {"matching_lengths": True}, {"lengths": {"on": len(on_frames), "off": len(off_frames), "on_spec": len(on_spec_frames), "off_spec": len(off_spec_frames)}}, [], [])
    files_check_path = out_root / "d" / "temporal" / "orbit_files.json"
    ensure_dir(str(files_check_path.parent))
    write_json(str(files_check_path), frames_check)
    record_schema(schema_tracker, relpath_safe(ROOT, files_check_path), frames_check)
    result["checks"]["D2_orbit_files"] = frames_check

    temporal_dir = out_root / "d" / "temporal"
    ensure_dir(str(temporal_dir))
    luma_metrics = temporal_metrics_record(on_frames, off_frames, TEMPORAL_THRESHOLDS, out_root, missing_required, spec_only=False)
    spec_metrics = temporal_metrics_record(on_spec_frames, off_spec_frames, TEMPORAL_THRESHOLDS, out_root, missing_required, spec_only=True)
    by_frame = metrics_record(luma_metrics["pass"] and spec_metrics["pass"], TEMPORAL_THRESHOLDS, {"luma": luma_metrics["measured"]["per_on"], "spec": spec_metrics["measured"]["per_on"]}, [], luma_metrics["missing_inputs"] + spec_metrics["missing_inputs"])
    write_json(str(temporal_dir / "temporal_metrics_by_frame.json"), by_frame)
    record_schema(schema_tracker, relpath_safe(ROOT, temporal_dir / "temporal_metrics_by_frame.json"), by_frame)
    synth_metrics = metrics_record(luma_metrics["pass"] and spec_metrics["pass"], TEMPORAL_THRESHOLDS, {"luma": luma_metrics["measured"]["agg_on"], "spec": spec_metrics["measured"]["agg_on"]}, [], luma_metrics["missing_inputs"] + spec_metrics["missing_inputs"])
    write_json(str(temporal_dir / "temporal_metrics_synth.json"), synth_metrics)
    record_schema(schema_tracker, relpath_safe(ROOT, temporal_dir / "temporal_metrics_synth.json"), synth_metrics)
    compare_measured = {
        "luma_ratio_mean": luma_metrics["measured"]["relative_ratios"]["mean"] if luma_metrics["measured"].get("relative_ratios") else None,
        "luma_ratio_p99": luma_metrics["measured"]["relative_ratios"]["p99"] if luma_metrics["measured"].get("relative_ratios") else None,
        "spec_ratio_mean": spec_metrics["measured"]["relative_ratios"]["mean"] if spec_metrics["measured"].get("relative_ratios") else None,
        "spec_ratio_p99": spec_metrics["measured"]["relative_ratios"]["p99"] if spec_metrics["measured"].get("relative_ratios") else None,
    }
    ratios_present = all(v is not None for v in compare_measured.values())
    compare_pass = (
        luma_metrics["pass"]
        and spec_metrics["pass"]
        and ratios_present
        and compare_measured["luma_ratio_mean"] <= TEMPORAL_THRESHOLDS["relative"]["delta_mean_ratio_max"]
        and compare_measured["luma_ratio_p99"] <= TEMPORAL_THRESHOLDS["relative"]["delta_p99_ratio_max"]
        and compare_measured["spec_ratio_mean"] <= TEMPORAL_THRESHOLDS["relative"]["delta_mean_ratio_max"]
        and compare_measured["spec_ratio_p99"] <= TEMPORAL_THRESHOLDS["relative"]["delta_p99_ratio_max"]
    )
    compare = metrics_record(compare_pass, TEMPORAL_THRESHOLDS, compare_measured, [], luma_metrics["missing_inputs"] + spec_metrics["missing_inputs"])
    write_json(str(temporal_dir / "compare_fade_on_off.json"), compare)
    record_schema(schema_tracker, relpath_safe(ROOT, temporal_dir / "compare_fade_on_off.json"), compare)
    result["checks"]["D3_temporal"] = metrics_record(luma_metrics["pass"] and spec_metrics["pass"] and compare_pass, TEMPORAL_THRESHOLDS, {"luma": luma_metrics["measured"], "spec": spec_metrics["measured"], "compare": compare_measured}, [], luma_metrics["missing_inputs"] + spec_metrics["missing_inputs"])

    integrity = integrity_metrics(on_frames, out_root, missing_required)
    write_json(str(temporal_dir / "orbit_integrity.json"), integrity)
    record_schema(schema_tracker, relpath_safe(ROOT, temporal_dir / "orbit_integrity.json"), integrity)
    result["checks"]["D4_orbit_integrity"] = integrity

    section_pass = (
        blend_result["pass"]
        and frames_check["pass"]
        and result["checks"]["D3_temporal"]["pass"]
        and integrity["pass"]
        and all(check.get("schema_ok", True) for check in result["checks"].values())
    )
    result["pass"] = section_pass
    return result


def build_summary(overall_pass: bool, sections: Dict[str, dict], missing_required: List[str], out_root: Path, manifest: Dict[str, any], schema_tracker: Dict[str, dict]) -> dict:
    artifact_paths = [relpath_safe(ROOT, Path(entry["path"])) for entry in manifest.get("pngs", []) if isinstance(entry, dict) and "path" in entry]
    artifact_paths += list(schema_tracker.keys())
    overall_schema_ok = all(entry.get("schema_ok", False) for entry in schema_tracker.values()) if schema_tracker else True
    return {
        "overall_pass": overall_pass,
        "overall_schema_ok": overall_schema_ok,
        "missing_required": sorted(set(missing_required)),
        "sections": sections,
        "schema": schema_tracker,
        "artifact_paths": sorted(set(artifact_paths)),
        "paths": {
            "manifest": relpath_safe(ROOT, out_root / "meta" / "manifest.json"),
            "summary": relpath_safe(ROOT, out_root / "meta" / "proofpack_summary.json"),
        },
    }


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    sections = {s.strip() for s in args.sections.split(",") if s.strip()}
    timestamp = Path(args.out).name if args.out else stamp_timestamp()
    out_root = Path(args.out) if args.out else ROOT / "reports" / "flake" / timestamp
    ensure_dir(str(out_root / "meta"))

    manifest: Dict[str, any] = collect_repo_state()
    sections_result: Dict[str, dict] = {}
    missing_required: List[str] = []
    schema_tracker: Dict[str, dict] = {}

    if "b" in sections:
        sections_result["milestone_b"] = b_section(out_root, args, manifest, missing_required, schema_tracker)
    if "c" in sections:
        sections_result["milestone_c"] = c_section(out_root, manifest, missing_required, schema_tracker)
    if "d" in sections:
        sections_result["milestone_d"] = d_section(out_root, manifest, missing_required, schema_tracker)

    overall_schema_ok = all(entry.get("schema_ok", False) for entry in schema_tracker.values()) if schema_tracker else True
    overall_pass = all(sec.get("pass", False) for sec in sections_result.values()) and len(missing_required) == 0 and overall_schema_ok
    summary = build_summary(overall_pass, sections_result, missing_required, out_root, manifest, schema_tracker)

    write_json(str(out_root / "meta" / "manifest.json"), manifest)
    write_json(str(out_root / "meta" / "proofpack_summary.json"), summary)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
