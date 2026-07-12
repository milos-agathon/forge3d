#!/usr/bin/env python3
"""Render a one-sample vs offline-accumulated MapScene comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if PYTHON_DIR.exists():
    sys.path.insert(0, str(PYTHON_DIR))

import forge3d as f3d


def _heightmap(size: int = 128) -> np.ndarray:
    coords = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    ridge = np.exp(-((xx * 1.8) ** 2 + (yy * 0.55) ** 2))
    ripples = 0.08 * np.sin(22.0 * xx + 7.0 * yy)
    return np.ascontiguousarray((ridge + ripples).astype(np.float32))


def build_scene(
    output_dir: str | Path,
    *,
    samples: int = 4,
    denoiser: str = "atrous",
    seed: int = 20260703,
    size: tuple[int, int] = (160, 96),
) -> f3d.MapScene:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=_heightmap(),
            crs="EPSG:32610",
            metadata={
                "source_id": "inline-offline-demo-dem",
                "width": 128,
                "height": 128,
                "resolution": [10.0, 10.0],
            },
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=3.4, azimuth_deg=138.0, elevation_deg=42.0),
        lighting=f3d.LightingPreset(name="rainier_showcase"),
        output=f3d.OutputSpec(
            width=int(size[0]),
            height=int(size[1]),
            path=str(output_dir / "offline_quality.png"),
            samples=samples,
            denoiser=denoiser,
            aovs=("albedo", "normal", "depth"),
            hdr=True,
            bit_depth=16,
        ),
        reproducibility_profile=f3d.ReproducibilityProfile(seed=seed),
    )


def run_example(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    scene = build_scene(output_dir)
    validation = scene.validate()
    render = scene.render()
    bundle_path = output_dir / "offline_quality.forge3d"
    bundle = scene.save_bundle(bundle_path)
    metadata = scene.last_render_metadata or {}
    png_path = Path(scene.last_render_path or scene.recipe.output.path)
    hdr_path = png_path.with_suffix(".exr")
    return {
        "validation_status": validation.status,
        "render_status": render.status,
        "render_backend": scene.last_render_backend,
        "bundle_status": bundle.status,
        "samples_used": int(metadata.get("samples_used", 0)),
        "denoiser_used": str(metadata.get("denoiser_used", "")),
        "aa_seed": metadata.get("aa_seed"),
        "aov_paths": dict(metadata.get("aov_paths", {})),
        "png_path": str(png_path),
        "hdr_path": str(hdr_path),
        "bundle_path": str(bundle_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("examples/out/mapscene_offline_quality"))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--denoiser", choices=["none", "atrous", "oidn"], default="atrous")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    one = build_scene(args.out_dir / "one_sample", samples=1, denoiser="none", seed=int(args.seed), size=(960, 540))
    accumulated = build_scene(
        args.out_dir / "offline",
        samples=max(2, int(args.samples)),
        denoiser=args.denoiser,
        seed=int(args.seed),
        size=(960, 540),
    )

    try:
        one_report = one.render()
        offline_report = accumulated.render()
    except RuntimeError as exc:
        raise SystemExit(
            "MapScene offline quality example requires the native terrain backend; "
            f"backend failed with: {exc}"
        ) from exc

    payload = {
        "one_sample": one.last_render_path,
        "offline": accumulated.last_render_path,
        "one_sample_status": one_report.status,
        "offline_status": offline_report.status,
        "offline_metadata": accumulated.last_render_metadata,
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"one_sample={payload['one_sample']} backend={one.last_render_backend} status={one_report.status}")
        print(
            "offline="
            f"{accumulated.last_render_path} backend={accumulated.last_render_backend} "
            f"samples={accumulated.last_render_metadata.get('samples_used') if accumulated.last_render_metadata else None} "
            f"denoiser={accumulated.last_render_metadata.get('denoiser_used') if accumulated.last_render_metadata else None} "
            f"status={offline_report.status}"
        )
        if accumulated.last_render_metadata:
            for name, path in sorted(accumulated.last_render_metadata.get("aov_paths", {}).items()):
                print(f"aov_{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
