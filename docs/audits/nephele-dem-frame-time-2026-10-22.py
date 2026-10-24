"""Measure real-DEM first/repeated frames on the physical NEPHELE CI lane.

The input is the terrain demo's original 1500-square Gore Range DEM, at its
native metre spacing, and the demo's 1920x1080 viewport. Two observations cover
initial transport construction and reuse of an unchanged converged field;
they are individual frame latencies, not a statistical performance estimate.
--describe-only reads CPU input data and never imports the native renderer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FIXTURE = ROOT / "tests/nephele/fixture"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_clean_certificate(certificate, state):
    if certificate["degradations"]:
        raise RuntimeError(f"{state} render certificate has degradations: {certificate['degradations']}")


def describe():
    provenance = load_json(ROOT / "tests/data/prometheus/gore_range_dem.json")
    path = ROOT / provenance["source_path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != provenance["source_sha256"]:
        raise ValueError("Gore Range source differs from the preserved provenance")
    with Image.open(path) as image:
        heights = np.array(image, dtype=np.float32)
    if list(heights.shape) != provenance["source_shape"] or min(heights.shape) < 1024:
        raise ValueError("Real DEM does not meet the owner's >=1024-square requirement")
    if not np.isfinite(heights).all() or np.any(heights == provenance["source_nodata"]):
        raise ValueError("DEM has invalid elevations; no substitution is permitted")
    return heights, {
        "path": provenance["source_path"], "sha256": digest,
        "shape": list(heights.shape), "spacing_metres": abs(provenance["source_affine"][0]),
        "elevation_range_metres": [float(heights.min()), float(heights.max())],
        "provenance": "tests/data/prometheus/gore_range_dem.json",
        "dataset_doi": provenance["dataset_doi"],
    }


def run(heights, source, head):
    # Local GPU execution is explicitly prohibited for this release.
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Physical profiling must run in GPU CI; use --describe-only locally")
    import forge3d as f3d
    from forge3d.diagnostics import render_certificate
    from forge3d.media import Medium
    from forge3d.terrain_demo import DEFAULT_SIZE
    from forge3d.terrain_params import AovSettings, TonemapSettings, make_terrain_params_config
    from scripts.nephele_acceptance import HOST_VISIBLE_LIMIT, _write_unit_hdr

    terrain = load_json(FIXTURE / "terrain.json")
    medium = load_json(FIXTURE / "medium.json")
    camera_input = load_json(FIXTURE / "camera.json")
    camera = camera_input["terrain_camera"]
    material = load_json(FIXTURE / "material.json")
    sun = load_json(FIXTURE / "sun.json")
    atmosphere = load_json(FIXTURE / "atmosphere.json")
    exposure = load_json(FIXTURE / "exposure.json")
    span = (heights.shape[1] - 1) * source["spacing_metres"]
    scale = span / ((terrain["dimensions"][0] - 1) * terrain["spacing"][0])
    # Geometric similarity of the authored cloud: lengths scale by s and
    # extinction/scattering by 1/s, preserving optical thickness and albedo.
    # Elevation translation preserves the real DEM's full relief in metres.
    heights = np.ascontiguousarray(heights - heights.min())
    nx, ny, nz = medium["domain"]["grid_shape"]
    density = np.array(medium["density_r16"], np.float32).reshape(nz, ny, nx) / 65535.0
    cloud = Medium.grid3d(
        (np.array(medium["sigma_a"]) / scale).tolist(),
        (np.array(medium["sigma_s"]) / scale).tolist(), density,
        tuple((np.array(medium["domain"][key]) * scale).tolist() for key in ("bounds_min", "bounds_max")),
        phase=medium["phase"]["kind"], g=medium["phase"]["g"],
        density_scale=medium["density_scale"], version=1,
    )
    params = f3d.TerrainRenderParams(make_terrain_params_config(
        terrain_shading_model="lambert_physical", size_px=DEFAULT_SIZE, render_scale=1.0,
        terrain_span=span, msaa_samples=1, z_scale=1.0,
        exposure=exposure["value"], domain=(float(heights.min()), float(heights.max())),
        light_azimuth_deg=sun["azimuth_deg"], light_elevation_deg=sun["elevation_deg"],
        sun_intensity=sun["intensity"], sun_color=sun["color"],
        albedo_mode=material["albedo_mode"], colormap_strength=material["colormap_strength"],
        cam_radius=camera["radius"] * scale, cam_phi_deg=camera["phi_deg"],
        cam_theta_deg=camera["theta_deg"], cam_target=(np.array(camera["target"]) * scale).tolist(),
        fov_y_deg=camera_input["fov_y"], camera_mode=camera["mode"],
        aa_samples=1, aa_seed=0x4E455048, tonemap=TonemapSettings(operator="aces"),
        aov=AovSettings(enabled=True, transmittance=True, in_scatter=True, cloud_shadow=True, optical_depth=True),
        media=cloud,
    ))
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    materials = f3d.MaterialSet.custom(
        tuple(material["albedo"]), material["metallic"], material["roughness"],
        triplanar_scale=material["triplanar_scale"], normal_strength=material["normal_strength"],
        blend_sharpness=material["blend_sharpness"],
    )
    with tempfile.TemporaryDirectory(prefix="nephele-dem-ibl-", ignore_cleanup_errors=True) as temporary:
        hdr = Path(temporary) / "unit.hdr"
        _write_unit_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=atmosphere["environment_intensity"])
    observations = []
    for state in ("first_frame", "unchanged_frame"):
        started = time.perf_counter()
        frame = renderer.render_terrain_pbr_pom(materials, ibl, params, heights, certificate=True)
        submitted = time.perf_counter()
        rgba = np.asarray(frame.to_numpy())
        completed = time.perf_counter()
        if rgba.shape[:2] != (DEFAULT_SIZE[1], DEFAULT_SIZE[0]):
            raise RuntimeError("Measured frame has an unexpected viewport")
        certificate = render_certificate(sign=False)
        require_clean_certificate(certificate, state)
        observations.append({
            "state": state, "render_call_ms": (submitted-started)*1000,
            "completed_frame_wall_ms": (completed-started)*1000,
            "output_readback_ms": (completed-submitted)*1000,
            "rgba_sha256": hashlib.sha256(rgba.tobytes()).hexdigest(),
            "certificate": certificate,
        })
    # This explicit, untimed capture includes readbacks. Ordinary frame
    # diagnostics use CPU metadata; the timed calls above request no AOVs.
    capture = renderer._capture_nephele_acceptance(materials, ibl, params, heights)
    diagnostics = dict(capture["diagnostics"])
    if diagnostics["source_revision"] != head:
        raise RuntimeError("Profiler wheel revision differs from checkout")
    if "NVIDIA" not in diagnostics["adapter"].upper() or diagnostics["backend"].lower() != "vulkan":
        raise RuntimeError("Profiler requires a physical NVIDIA Vulkan adapter")
    convergence = diagnostics["directional_transport"]
    if not convergence["converged"] or not convergence["cache_reused"]:
        raise RuntimeError("Unchanged static transport was not reused after convergence")
    if dict(capture["memory"])["peak_host_visible_bytes"] >= HOST_VISIBLE_LIMIT:
        raise RuntimeError("Real-DEM capture exceeds the existing host-visible memory contract")
    return {
        "status": "measured", "head_sha": head, "source": source,
        "viewport": list(DEFAULT_SIZE), "viewport_source": "forge3d.terrain_demo.DEFAULT_SIZE",
        "terrain_span_metres": span, "medium_similarity_length_scale": scale,
        "observations": observations, "diagnostics": diagnostics,
        "memory": dict(capture["memory"]),
        "limitations": [
            "Individual first/unchanged frame observations; no average or percentile claim.",
            "Completed-frame wall time includes the product Frame.to_numpy readback.",
            "Cloudlight timestamp interval includes GPU idle gaps while convergence is read back.",
            "No measured before-change speedup claim; the old DEM-sized candidate was never physically captured.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--describe-only", action="store_true")
    args = parser.parse_args()
    heights, source = describe()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, encoding="utf-8").strip()
    if args.describe_only:
        result = {"status": "CPU_INPUT_ONLY", "head_sha": head, "source": source, "GPU_executed": False}
    else:
        result = run(heights, source, head)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "observations"}, indent=2))


if __name__ == "__main__":
    main()
