#!/usr/bin/env python3
"""Capture and judge NEPHELE physical acceptance (Gates 1-6) on one GPU lane.

``produce`` and ``render`` write the raw evidence for the tracked fixture scene
into an artifact directory; ``gate1`` .. ``gate6`` turn that evidence into the
six acceptance verdicts asserted by ``tests/test_nephele_physical.py``; and
``report`` writes the combined verdict the CI lane publishes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/nephele/fixture"
sys.path.insert(0, str(ROOT / "tests"))
from _deltae import delta_e_2000, srgb_to_lab  # noqa: E402
from _ssim import ssim  # noqa: E402

GATE1_SAMPLES = 1_000_000
GATE2_SAMPLES = 10_000_000
COMPARATOR_SAMPLES = 100_000_000
COMPARATOR_SEED = 0x4E455048454C4501
MAJORANT_PROBES = 1_000_000
HOST_VISIBLE_LIMIT = 512 * 1024**2
SUN_TRANSMITTANCE_METHOD = "bounded_nested_midpoint"
SUN_TRANSMITTANCE_BIAS = "fine_midpoint_with_coarse_fine_abs_rgb_error"


def _load(name: str, directory: Path = FIXTURE) -> dict[str, Any]:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Transported medium: the f16 density and N^3 majorant grid the renderer uses.
# ---------------------------------------------------------------------------


def represented_density(medium: dict[str, Any]) -> np.ndarray:
    """Return the unorm16 -> f32 -> f16 density as f32, indexed ``[z, y, x]``."""
    nx, ny, nz = medium["domain"]["grid_shape"]
    decoded = np.asarray(medium["density_r16"], dtype=np.float32) / np.float32(65535.0)
    return decoded.astype(np.float16).astype(np.float32).reshape((nz, ny, nx))


def _runtime_sigma_t(medium: dict[str, Any]) -> np.float32:
    return np.float32(max(float(np.float32(a) + np.float32(s)) for a, s in zip(medium["sigma_a"], medium["sigma_s"])))


def _next_up(value: np.float32) -> np.float32:
    if value == np.float32(0.0) or not np.isfinite(value):
        return value
    return np.asarray(np.asarray(value, dtype=np.float32).view(np.uint32) + np.uint32(1), dtype=np.uint32).view(np.float32)[()]


def _outward_mul(a: np.float32, b: np.float32) -> np.float32:
    rounded = np.float32(a * b)
    return _next_up(rounded) if float(rounded) < float(a) * float(b) else rounded


def _trilinear_upper(maximum: np.float32) -> np.float32:
    if maximum == np.float32(0.0):
        return maximum
    operations, unit_roundoff = 13.0, 1.0 / float(1 << 24)
    half_min_subnormal = float(np.nextafter(np.float32(0.0), np.float32(1.0))) * 0.5
    denominator = 1.0 - operations * unit_roundoff
    exact = float(maximum) * (1.0 + operations * unit_roundoff / denominator) + operations * half_min_subnormal / denominator
    rounded = np.float32(exact)
    return _next_up(rounded) if float(rounded) < exact else rounded


def canonical_majorant_cells(medium: dict[str, Any]) -> list[float]:
    """Reproduce the renderer's conservative per-cell extinction majorants."""
    represented = represented_density(medium)
    nz, ny, nx = represented.shape
    scale, sigma_t = np.float32(medium["density_scale"]), _runtime_sigma_t(medium)
    cells: list[float] = []
    for z, y, x in itertools.product(range(nz), range(ny), range(nx)):
        maximum = np.float32(represented[max(z - 1, 0):z + 2, max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].max())
        cells.append(float(_outward_mul(_outward_mul(_trilinear_upper(maximum), scale), sigma_t)))
    return cells


def _majorant_probe_points(shape: tuple[int, int, int], count: int) -> np.ndarray:
    """Domain corners, face centres, texel centres and both sides of every cell
    boundary first, then an irrational lattice fills the remaining probes."""
    points: list[tuple[float, ...]] = list(itertools.product((0.0, 1.0), repeat=3))
    for axis in range(3):
        a, b = (d for d in range(3) if d != axis)
        for i, j in itertools.product(range(shape[a]), range(shape[b])):
            for side in (0.0, 1.0):
                point = [0.0, 0.0, 0.0]
                point[axis], point[a], point[b] = side, (i + 0.5) / shape[a], (j + 0.5) / shape[b]
                points.append(tuple(point))
    points += [((x + 0.5) / shape[0], (y + 0.5) / shape[1], (z + 0.5) / shape[2])
               for z, y, x in itertools.product(range(shape[2]), range(shape[1]), range(shape[0]))]
    for axis in range(3):
        a, b = (d for d in range(3) if d != axis)
        for index in range(1, shape[axis]):
            boundary = np.float32(index / shape[axis])
            for i, j in itertools.product(range(shape[a]), range(shape[b])):
                for side in (np.nextafter(boundary, np.float32(0.0)), np.nextafter(boundary, np.float32(1.0))):
                    point = [0.0, 0.0, 0.0]
                    point[axis], point[a], point[b] = float(side), (i + 0.5) / shape[a], (j + 0.5) / shape[b]
                    points.append(tuple(point))
    prefix = np.asarray(points, dtype=np.float64)
    index = np.arange(len(prefix), count, dtype=np.float64)[:, None]
    lattice = np.mod(index * np.asarray([0.7548776662466927, 0.5698402909980532, 0.4385790210124317]), 1.0)
    return np.concatenate([prefix, lattice])


def majorant_probe(medium: dict[str, Any], count: int = MAJORANT_PROBES) -> np.ndarray:
    """Return ``(count, 2)`` f32 pairs of represented extinction and its cell majorant."""
    shape = tuple(medium["domain"]["grid_shape"])
    grid = represented_density(medium)
    unit = np.clip(_majorant_probe_points(shape, count).astype(np.float32), np.float32(0.0), np.float32(1.0))
    sizes = np.asarray(shape, dtype=np.float32)
    coordinate = unit * sizes - np.float32(0.5)
    floor = np.floor(coordinate)
    lower = np.clip(floor, 0, sizes - 1).astype(np.int64)
    upper = np.clip(floor + 1, 0, sizes - 1).astype(np.int64)
    fraction = (coordinate - floor).astype(np.float32)
    one = np.float32(1.0)
    density = np.zeros(len(unit), dtype=np.float32)
    for z, y, x in itertools.product(range(2), repeat=3):
        ix = upper[:, 0] if x else lower[:, 0]
        iy = upper[:, 1] if y else lower[:, 1]
        iz = upper[:, 2] if z else lower[:, 2]
        wx = fraction[:, 0] if x else one - fraction[:, 0]
        wy = fraction[:, 1] if y else one - fraction[:, 1]
        wz = fraction[:, 2] if z else one - fraction[:, 2]
        density = density + grid[iz, iy, ix] * ((wx * wy) * wz)
    extinction = (density * np.float32(medium["density_scale"])) * _runtime_sigma_t(medium)
    cell = np.minimum(np.floor(unit * sizes).astype(np.int64), np.asarray(shape) - 1)
    majorants = np.asarray(medium["majorant_cells"], dtype=np.float32)
    bound = majorants[(cell[:, 2] * shape[1] + cell[:, 1]) * shape[0] + cell[:, 0]]
    return np.column_stack([extinction, bound])


def comparator_survival(medium: dict[str, Any], samples: int = COMPARATOR_SAMPLES) -> float:
    """Independent Bernoulli estimate of transmittance through the slab along z.

    Each sample picks a transverse point, interpolates the exact texel-centre
    column optical depth, and survives with probability ``exp(-tau)``.
    """
    represented = represented_density(medium).astype(np.float64)
    shape = medium["domain"]["grid_shape"]
    span = float(medium["domain"]["bounds_max"][2] - medium["domain"]["bounds_min"][2])
    sigma_t = max(float(a) + float(s) for a, s in zip(medium["sigma_a"], medium["sigma_s"]))
    columns = represented.mean(axis=0) * span * float(medium["density_scale"])  # [y, x]

    def uniform(start: int, stop: int, dimension: int) -> np.ndarray:
        with np.errstate(over="ignore"):
            value = np.uint64(COMPARATOR_SEED) ^ (np.arange(start, stop, dtype=np.uint64) * np.uint64(3) + np.uint64(dimension))
            value += np.uint64(0x9E3779B97F4A7C15)
            value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            value ^= value >> np.uint64(31)
        return ((value >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)

    def axis(coordinate: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scaled = coordinate * size - 0.5
        base = np.floor(scaled).astype(np.int64)
        return np.clip(base, 0, size - 1), np.clip(base + 1, 0, size - 1), scaled - np.floor(scaled)

    survived = 0
    for start in range(0, samples, 1_000_000):
        stop = min(start + 1_000_000, samples)
        x0, x1, fx = axis(uniform(start, stop, 0), shape[0])
        y0, y1, fy = axis(uniform(start, stop, 1), shape[1])
        tau = ((1 - fx) * (1 - fy) * columns[y0, x0] + fx * (1 - fy) * columns[y0, x1]
               + (1 - fx) * fy * columns[y1, x0] + fx * fy * columns[y1, x1])
        survived += int(np.count_nonzero(uniform(start, stop, 2) < np.exp(-sigma_t * tau)))
    return survived / samples


# ---------------------------------------------------------------------------
# Producers: write raw evidence into the artifact directory.
# ---------------------------------------------------------------------------


def produce(artifact: Path) -> None:
    """Run the CPU estimator, comparator, majorant and static-shader producers."""
    import forge3d._forge3d as native
    from forge3d.media import Medium

    from scripts.nephele_shader_analyzer import analyze_shaders

    artifact.mkdir(parents=True, exist_ok=True)
    slab = _load("homogeneous-slab.json", ROOT / "tests/nephele")
    medium = _load("medium.json")
    homogeneous = Medium.homogeneous(slab["sigma_a"], slab["sigma_s"], slab["density"], version=1)
    samples = dict(native._nephele_physical_samples(
        homogeneous._native, fixture_medium()._native, float(slab["distance"]), GATE1_SAMPLES, GATE2_SAMPLES,
    ))
    np.save(artifact / "homogeneous-ratio-samples.npy", np.asarray(samples["homogeneous_ratio"], dtype=np.float64))
    np.save(artifact / "heterogeneous-ratio-samples.npy", np.asarray(samples["heterogeneous_ratio"], dtype=np.float64))
    np.save(artifact / "rr-contributions.npy", np.column_stack([
        np.asarray(samples["rr_on"], dtype=np.float64), np.asarray(samples["rr_off"], dtype=np.float64),
    ]))
    np.save(artifact / "gate2-energy-outcomes.npy", np.column_stack([
        np.asarray(samples[name], dtype=np.uint8) for name in ("transmitted", "scattered_out", "absorbed")
    ]))
    np.save(artifact / "majorant-probe-pairs.npy", majorant_probe(medium))
    _save_json(artifact / "heterogeneous-comparator.json", {
        "samples": COMPARATOR_SAMPLES, "mean": comparator_survival(medium),
    })
    _save_json(artifact / "gate5-static.json", analyze_shaders(ROOT))


def fixture_medium(phase: str = "henyey_greenstein"):
    from forge3d.media import Medium

    medium = _load("medium.json")
    extra = {"g": medium["phase"]["g"]} if phase == "henyey_greenstein" else {}
    nx, ny, nz = medium["domain"]["grid_shape"]
    density = np.asarray(medium["density_r16"], dtype=np.float32).reshape(nz, ny, nx) / 65535.0
    return Medium.grid3d(
        medium["sigma_a"], medium["sigma_s"], density,
        (medium["domain"]["bounds_min"], medium["domain"]["bounds_max"]),
        phase=phase, density_scale=medium["density_scale"], version=1, **extra,
    )


def _write_unit_hdr(path: Path) -> None:
    # RGBE (128,128,128,129) decodes to unit linear radiance; IBL intensity
    # therefore maps the tracked atmosphere input without another scale.
    path.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n" + bytes([128, 128, 128, 129]))


class FixtureScene:
    """The tracked cloud-over-terrain scene, rendered by the real-time path."""

    def __init__(self, phase: str = "henyey_greenstein") -> None:
        import forge3d as f3d
        from forge3d.terrain_params import AovSettings, TonemapSettings, make_terrain_params_config

        camera_input, terrain_data, sun = _load("camera.json"), _load("terrain.json"), _load("sun.json")
        atmosphere, exposure, material = _load("atmosphere.json"), _load("exposure.json"), _load("material.json")
        self.crop = _load("crop.json")
        camera = camera_input["terrain_camera"]
        self.terrain = np.load(FIXTURE / terrain_data["dem"], allow_pickle=False)
        spacing, dimensions = terrain_data["spacing"], terrain_data["dimensions"]
        self.params = f3d.TerrainRenderParams(make_terrain_params_config(
            terrain_shading_model="lambert_physical", size_px=tuple(self.crop["full_viewport"]), render_scale=1.0,
            terrain_span=float(spacing[0]) * (int(dimensions[0]) - 1), msaa_samples=1,
            z_scale=float(terrain_data["exaggeration"]), exposure=float(exposure["value"]),
            domain=(float(self.terrain.min()), float(self.terrain.max())),
            light_azimuth_deg=sun["azimuth_deg"], light_elevation_deg=sun["elevation_deg"],
            sun_intensity=sun["intensity"], sun_color=sun["color"], albedo_mode=material["albedo_mode"],
            colormap_strength=float(material["colormap_strength"]), cam_radius=camera["radius"],
            cam_phi_deg=camera["phi_deg"], cam_theta_deg=camera["theta_deg"], cam_target=camera["target"],
            fov_y_deg=float(camera_input["fov_y"]), camera_mode=camera["mode"], aa_samples=1, aa_seed=0x4E455048,
            tonemap=TonemapSettings(operator="aces"),
            aov=AovSettings(enabled=True, transmittance=True, in_scatter=True, cloud_shadow=True, optical_depth=True),
            media=fixture_medium(phase),
        ))
        self.renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        self.material = f3d.MaterialSet.custom(
            tuple(material["albedo"]), float(material["metallic"]), float(material["roughness"]),
            triplanar_scale=float(material["triplanar_scale"]), normal_strength=float(material["normal_strength"]),
            blend_sharpness=float(material["blend_sharpness"]),
        )
        with tempfile.TemporaryDirectory(prefix="nephele-ibl-", ignore_cleanup_errors=True) as temporary:
            hdr = Path(temporary) / "unit.hdr"
            _write_unit_hdr(hdr)
            self.ibl = f3d.IBL.from_hdr(str(hdr), intensity=float(atmosphere["environment_intensity"]))

    def capture(self, **flags: bool) -> dict[str, Any]:
        """Capture one frame; array outputs are cropped to the tracked fixture crop.

        Raises if the camera differs bitwise from the reference fixture camera.
        """
        flags.setdefault("terrain_occlusion_in_media", True)
        raw = dict(self.renderer._capture_nephele_acceptance(
            self.material, self.ibl, self.params, self.terrain, **flags,
        ))
        expected = _load("reference-provenance.json")["camera_contract"]
        for key, value in expected.items():
            if not np.array_equal(np.asarray(raw["camera_contract"][key], np.float32), np.asarray(value, np.float32)):
                raise RuntimeError(f"capture camera {key} differs from the reference fixture camera")
        x, y, w, h = (int(self.crop[key]) for key in ("x", "y", "width", "height"))
        for key, value in raw.items():
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                raw[key] = value[y:y + h, x:x + w]
        return raw


def render(artifact: Path, label: str, primary: bool) -> None:
    """Capture one clean run; the primary run also writes Gates 3-6 inputs."""
    artifact.mkdir(parents=True, exist_ok=True)
    scene = FixtureScene()
    capture = scene.capture(include_no_medium=primary)
    np.save(artifact / f"froxel-frame-run-{label}.npy", np.asarray(capture["beauty"], dtype=np.uint8))
    _save_json(artifact / f"run-{label}.json", {"process_id": os.getpid(), "diagnostics": dict(capture["diagnostics"])})
    if not primary:
        return
    for filename, key in (
        ("realtime-rgb.npy", "beauty"), ("medium-disabled-rgb.npy", "no_medium_beauty"),
        ("transmittance.npy", "transmittance"), ("in-scatter.npy", "in_scatter"),
        ("cloud-shadow-aov.npy", "cloud_shadow"), ("optical-depth.npy", "optical_depth"),
        ("realtime-termination-slice.npy", "termination_slice"),
    ):
        np.save(artifact / filename, np.asarray(capture[key]))
    ablation = scene.capture(terrain_occlusion_in_media=False)
    np.save(artifact / "terrain-occlusion-disabled-rgb.npy", np.asarray(ablation["beauty"], dtype=np.uint8))
    height, width = np.asarray(capture["beauty"]).shape[:2]
    _save_json(artifact / "memory.json", {
        "output_dimensions": [width, height], "froxel_depth": capture["froxel_depth"], **dict(capture["memory"]),
    })


# ---------------------------------------------------------------------------
# Gates: each returns its metrics and the list of failed criteria.
# ---------------------------------------------------------------------------


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(values.size))


def _result(metrics: dict[str, Any], checks: dict[str, bool]) -> dict[str, Any]:
    return {"metrics": metrics, "failures": [name for name, passed in checks.items() if not passed]}


def gate1(artifact: Path) -> dict[str, Any]:
    """Estimator correctness, majorant conservativeness and Russian roulette."""
    slab = _load("homogeneous-slab.json", ROOT / "tests/nephele")
    tau = slab["sigma_t_max_channel"] * slab["density"] * slab["distance"]
    homogeneous = np.load(artifact / "homogeneous-ratio-samples.npy")
    mean, se = _mean_se(homogeneous)
    analytic = math.exp(-tau)
    heterogeneous_mean, _ = _mean_se(np.load(artifact / "heterogeneous-ratio-samples.npy"))
    comparator = _load("heterogeneous-comparator.json", artifact)
    relative_error = abs(heterogeneous_mean - comparator["mean"]) / comparator["mean"]
    pairs = np.load(artifact / "majorant-probe-pairs.npy")
    violations = int(np.count_nonzero(pairs[:, 0] > pairs[:, 1]))
    rr = np.load(artifact / "rr-contributions.npy")
    (on_mean, on_se), (off_mean, off_se) = _mean_se(rr[:, 0]), _mean_se(rr[:, 1])
    oracle = (1.0 - math.exp(-tau)) * slab["sigma_s"][0] / slab["sigma_t_max_channel"]
    rr_difference = abs(on_mean - off_mean) / off_mean
    metrics = {
        "homogeneous": {"mean": mean, "standard_error": se, "analytic": analytic, "samples": homogeneous.size},
        "heterogeneous": {"mean": heterogeneous_mean, "comparator": comparator, "relative_error": relative_error},
        "majorant": {"probe_count": len(pairs), "violation_count": violations,
                     "max_excess": float((pairs[:, 0].astype(np.float64) - pairs[:, 1]).max())},
        "russian_roulette": {"on": [on_mean, on_se], "off": [off_mean, off_se], "analytic_oracle": oracle,
                             "relative_difference": rr_difference, "pairs": len(rr)},
    }
    return _result(metrics, {
        "homogeneous sample count is 1e6": homogeneous.size == GATE1_SAMPLES,
        "homogeneous analytic within mean +/- 3 SE": abs(mean - analytic) <= 3 * se,
        "heterogeneous relative error < 0.5%": relative_error < 0.005,
        "comparator ran 1e8 samples": comparator["samples"] == COMPARATOR_SAMPLES,
        "1e6 majorant probes": len(pairs) == MAJORANT_PROBES,
        "zero majorant violations": violations == 0,
        "1e6 RR pairs": len(rr) == GATE1_SAMPLES,
        "RR-on/off contain the analytic oracle within 3 SE":
            abs(on_mean - oracle) <= 3 * on_se and abs(off_mean - oracle) <= 3 * off_se,
        "RR relative difference < 0.2%": rr_difference < 0.002,
    })


def gate2(artifact: Path) -> dict[str, Any]:
    """Closed-sphere energy conservation from three independent ensembles."""
    outcomes = np.load(artifact / "gate2-energy-outcomes.npy")
    means = outcomes.mean(axis=0, dtype=np.float64)
    residual = abs(float(means.sum()) - 1.0)
    standard_error = math.sqrt(float(outcomes.var(axis=0, ddof=1, dtype=np.float64).sum()) / len(outcomes))
    metrics = {"sample_count": len(outcomes), "transmitted": means[0], "scattered_out": means[1],
               "absorbed": means[2], "relative_residual": residual, "residual_standard_error": standard_error}
    return _result(metrics, {
        "1e7 samples per ensemble": len(outcomes) == GATE2_SAMPLES,
        "binary outcomes": bool(np.isin(outcomes, (0, 1)).all()),
        # One shared walk would close exactly by construction.
        "ensembles are independent": bool(np.any(outcomes.sum(axis=1) != 1)),
        "energy residual <= 1e-3": residual <= 1.0e-3,
    })


def _rgb(artifact: Path, name: str) -> np.ndarray:
    return np.load(artifact / name).astype(np.uint8)


def _mask(name: str) -> np.ndarray:
    return np.load(FIXTURE / name).astype(bool)


def _roi(*images: np.ndarray) -> list[np.ndarray]:
    ys, xs = np.nonzero(_mask("godray-roi-mask.npy"))
    return [image[ys.min():ys.max() + 1, xs.min():xs.max() + 1] for image in images]


def gate3(artifact: Path) -> dict[str, Any]:
    """Real-time frame against the converged reference."""
    reference, realtime = np.load(FIXTURE / "reference-rgb.npy"), _rgb(artifact, "realtime-rgb.npy")
    delta_e = delta_e_2000(srgb_to_lab(realtime), srgb_to_lab(reference))
    fraction = float(np.mean(delta_e[_mask("sky-cloud-mask.npy")] < 2.5))
    shaft_ssim = ssim(*_roi(reference, realtime), data_range=255.0)
    return _result({"sky_cloud_delta_e_pass_fraction": fraction, "godray_roi_ssim": shaft_ssim}, {
        "sky/cloud DeltaE2000 < 2.5 on >= 95% of pixels": fraction >= 0.95,
        "god-ray ROI SSIM > 0.95": shaft_ssim > 0.95,
    })


def _inverse_display_aces(rgb: np.ndarray) -> np.ndarray:
    """Use the unchanged numerical inverse used for the accepted reference."""
    aces_x = np.linspace(0.0, 64.0, 2_000_001)
    aces_y = np.clip((aces_x * (2.51 * aces_x + 0.03)) /
                     (aces_x * (2.43 * aces_x + 0.59) + 0.14), 0, 1)
    value = rgb.astype(np.float64) / 255.0
    display = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
    return np.interp(display, aces_y, aces_x)


def _occlusion_response(realtime: np.ndarray, disabled: np.ndarray) -> float:
    delta = _inverse_display_aces(disabled) - _inverse_display_aces(realtime)
    return float((delta @ np.array([0.2126, 0.7152, 0.0722])).mean())


def gate4(artifact: Path) -> dict[str, Any]:
    """Cloud shadows on terrain, plus medium and terrain-occlusion ablations."""
    policy = _load("gate4-policy.json", ROOT / "tests/nephele")
    minimum = policy["aggregation"]["minimum_fraction"]
    reference, realtime = np.load(FIXTURE / "reference-rgb.npy"), _rgb(artifact, "realtime-rgb.npy")
    terrain = _mask("terrain-mask.npy")
    shadow = delta_e_2000(srgb_to_lab(realtime), srgb_to_lab(reference))[_mask("cloud-shadow-mask.npy") & terrain]
    shadow_fraction = float(np.mean(shadow < 2.0))
    no_medium = delta_e_2000(srgb_to_lab(_rgb(artifact, "medium-disabled-rgb.npy")), srgb_to_lab(realtime))
    changed = float(np.mean(no_medium[terrain] > 5.0))
    disabled = _rgb(artifact, "terrain-occlusion-disabled-rgb.npy")
    occlusion_ssim = ssim(*_roi(realtime, disabled), data_range=255.0)
    response_policy = policy["occlusion_response"]
    expected = response_policy["reference"]["mean_linear_luminance_change"]
    tolerance = response_policy["maximum_relative_error"]
    observed = _occlusion_response(*_roi(realtime, disabled))
    relative_error = abs(observed - expected) / expected
    metrics = {"shadow_delta_e_pass_fraction": shadow_fraction, "medium_ablation_changed_fraction": changed,
               "terrain_occlusion_ablation_ssim": occlusion_ssim, "shadow_pixels": int(shadow.size),
               "terrain_occlusion_mean_linear_luminance_change": observed,
               "terrain_occlusion_reference_mean_linear_luminance_change": expected,
               "terrain_occlusion_response_relative_error": relative_error,
               "terrain_occlusion_maximum_relative_error": tolerance}
    return _result(metrics, {
        f"owner-approved policy ({policy['approved_by']}, {policy['owner_record']['date']})":
            policy["status"] == "APPROVED" and policy["aggregation"]["kind"] == "per_pixel_pass_fraction",
        f"cloud-shadow terrain DeltaE2000 < 2 on >= {minimum:.0%} of pixels": shadow.size > 0 and shadow_fraction >= minimum,
        "removing the medium changes >= 10% of terrain pixels by DeltaE > 5": changed >= 0.10,
        "owner-approved physical occlusion-response policy":
            response_policy["status"] == "APPROVED" and bool(response_policy["approved_by"])
            and expected > 0 and math.isfinite(expected) and tolerance >= 0 and math.isfinite(tolerance),
        f"terrain occlusion response matches accepted reference within {tolerance:.0%}":
            math.isfinite(relative_error) and relative_error <= tolerance,
    })


def gate5(artifact: Path) -> dict[str, Any]:
    """No compute-stage comparison sampling; shafts end at the reference ridgeline."""
    static = _load("gate5-static.json", artifact)
    shaft = _mask("shaft-mask.npy") & _mask("godray-roi-mask.npy")
    reference = np.load(FIXTURE / "reference-terrain-slice.npy")
    realtime = np.load(artifact / "realtime-termination-slice.npy")
    agreement = float(np.mean(np.abs(reference[shaft] - realtime[shaft]) <= 1.0))
    metrics = {
        "compute_entry_texture_sample_compare_calls": static["compute_entry_texture_sample_compare_calls"],
        "stale_disabled_shadow_comments": static["stale_disabled_shadow_comments"],
        "naga_validated_assemblies": static["naga_validated_assemblies"],
        "ridgeline_within_one_slice_fraction": agreement,
    }
    return _result(metrics, {
        "every production shader assembly resolved": not static["unresolved_source_expressions"],
        "every resolved assembly validated by naga":
            static["naga_validated_assemblies"] == len(static["resolved_source_sha256"])
            == len(static["executed_assembled_naga_contracts"]),
        "zero textureSampleCompare calls from compute entry points":
            static["compute_entry_texture_sample_compare_calls"] == 0,
        "no stale disabled-shadow comments": static["stale_disabled_shadow_comments"] == 0,
        "shaft ROI is nonempty": bool(shaft.any()),
        "ridgeline within one slice on >= 99% of shaft pixels": agreement >= 0.99,
    })


def gate6(artifact: Path) -> dict[str, Any]:
    """Bitwise determinism across two processes and bounded memory."""
    frame_a = np.load(artifact / "froxel-frame-run-a.npy")
    frame_b = np.load(artifact / "froxel-frame-run-b.npy")
    run_a, run_b = _load("run-a.json", artifact), _load("run-b.json", artifact)
    diagnostics = run_a["diagnostics"]
    memory = _load("memory.json", artifact)
    termination = np.load(artifact / "realtime-termination-slice.npy")
    integration_steps = int(termination.astype(np.uint64).sum() + termination.size)
    width, height = memory["output_dimensions"]
    expected_readback = 40 * width * height + 24 * width * height  # beauty+AOVs+no-medium, internal buffers
    aovs = ("transmittance.npy", "in-scatter.npy", "cloud-shadow-aov.npy", "optical-depth.npy")
    metrics = {
        "frame_hashes": [hashlib.sha256(frame.tobytes()).hexdigest() for frame in (frame_a, frame_b)],
        "process_ids": [run_a["process_id"], run_b["process_id"]],
        "memory": memory, "termination_integration_steps": integration_steps,
        "sun_transmittance": {key: diagnostics[f"sun_transmittance_{key}"]
                              for key in ("method", "bias", "max_segment_length", "executed_steps", "max_abs_error")},
    }
    return _result(metrics, {
        "two distinct processes": run_a["process_id"] != run_b["process_id"],
        "frames bitwise identical": frame_a.dtype == frame_b.dtype and np.array_equal(frame_a, frame_b),
        "diagnostics identical across runs": run_a["diagnostics"] == run_b["diagnostics"],
        "Vulkan backend": str(diagnostics["backend"]).lower() == "vulkan",
        "physical terrain shading": diagnostics["terrain_shading_model"] == "lambert_physical",
        "fresh history rejected": diagnostics["temporal_history_decision"] == "rejected",
        "valid majorant and executed multiple scattering":
            diagnostics["majorant_valid"] is True and diagnostics["executed_multi_scatter"] is True,
        "f16 flush diagnostics consistent":
            0.0 <= diagnostics["f16_flushed_max_value"] < 2.0**-24
            and (diagnostics["f16_flushed_sample_count"] == 0) == (diagnostics["f16_flushed_max_value"] == 0.0),
        "termination slices are whole froxel indices":
            bool(np.all((termination >= 0) & (termination <= memory["froxel_depth"] - 1) & (termination == np.floor(termination)))),
        "step_count includes the termination integration work": diagnostics["step_count"] >= integration_steps,
        "readback bytes match the output dimensions": memory["readback_bytes"] == expected_readback,
        "peak host-visible bytes < 512 MiB": memory["peak_host_visible_bytes"] < HOST_VISIBLE_LIMIT,
        f"sun transmittance is {SUN_TRANSMITTANCE_METHOD}":
            diagnostics["sun_transmittance_method"] == SUN_TRANSMITTANCE_METHOD
            and diagnostics["sun_transmittance_bias"] == SUN_TRANSMITTANCE_BIAS,
        "sun transmittance executed with bounded error":
            diagnostics["sun_transmittance_max_segment_length"] > 0.0
            and diagnostics["sun_transmittance_executed_steps"] > 0
            and 0.0 <= diagnostics["sun_transmittance_max_abs_error"] <= 1.0,
        "AOVs are finite": all(np.isfinite(np.load(artifact / name)).all() for name in aovs),
    })


GATES = {"gate1": gate1, "gate2": gate2, "gate3": gate3, "gate4": gate4, "gate5": gate5, "gate6": gate6}


def report(artifact: Path, head_sha: str) -> bool:
    """Write ``verification-report.json``; on PASS also write ``lane-ran.json``."""
    # A rerun must never leave a previous PASS marker beside failed evidence.
    for name in ("verification-report.json", "verification-error.json", "lane-ran.json"):
        (artifact / name).unlink(missing_ok=True)
    gates = {name: gate(artifact) for name, gate in GATES.items()}
    passed = not any(gate["failures"] for gate in gates.values())
    verdict = {"status": "PASS" if passed else "FAIL", "head_sha": head_sha, "gates": gates}
    _save_json(artifact / ("verification-report.json" if passed else "verification-error.json"), verdict)
    if passed:
        _save_json(artifact / "lane-ran.json", {"status": "RAN", "head_sha": head_sha})
    print(json.dumps(verdict, sort_keys=True, default=float))
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("produce", "render", "report"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-label", choices=("a", "b"))
    parser.add_argument("--capture-primary", action="store_true")
    parser.add_argument("--head-sha")
    args = parser.parse_args(argv)
    if args.action == "produce":
        produce(args.artifact_dir)
    elif args.action == "render":
        if args.run_label is None:
            parser.error("render requires --run-label")
        render(args.artifact_dir, args.run_label, args.capture_primary)
    else:
        if not args.head_sha:
            parser.error("report requires --head-sha")
        return 0 if report(args.artifact_dir, args.head_sha) else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
