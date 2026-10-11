"""Engine-backed MapScene water and cloud-shadow preset example."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import forge3d as f3d


def demo_heightmap(size: int = 48) -> np.ndarray:
    """Return a small DEM with a low, flat lake basin for auto water masking."""
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    hills = 0.45 + 0.18 * xx + 0.12 * yy + 0.09 * np.sin(xx * np.pi * 2.0)
    lake = (xx * xx / 0.50 + yy * yy / 0.18) < 1.0
    hills[lake] = 0.02
    return hills.astype(np.float32)


def build_scene(
    output_path: str | Path | None = None,
    *,
    frame_index: int = 0,
    total_frames: int = 1,
    width: int = 640,
    height: int = 360,
) -> f3d.MapScene:
    """Build a MapScene using GPU terrain water masks and engine cloud shadows."""
    frame_count = max(1, int(total_frames))
    phase = (int(frame_index) % frame_count) / float(frame_count)
    cloud_settings = {
        "enabled": True,
        "shadows_enabled": True,
        "coverage": 0.68,
        "density": 0.50,
        "shadow_strength": 0.36,
        "quality": "high",
        "shadow_offset_x": phase * 0.75,
        "shadow_offset_y": phase * 0.28,
    }
    water_settings = {
        "enabled": True,
        "auto_mask": True,
        "level": 0.08,
        "slope_threshold": 0.18,
    }
    heightmap = demo_heightmap()
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=heightmap,
            crs="EPSG:32610",
            metadata={
                "source_id": "mapscene-water-clouds-demo",
                "width": int(heightmap.shape[1]),
                "height": int(heightmap.shape[0]),
                "bounds": (-122.3, 46.8, -122.0, 47.0),
                "water": water_settings,
                "clouds": cloud_settings,
            },
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(
            target=(0.0, 0.0, 0.0),
            distance=900.0,
            azimuth_deg=38.0,
            elevation_deg=46.0,
        ),
        lighting=f3d.LightingPreset(
            name="water_cloud_daycycle",
            intensity=1.12,
            settings={"water": water_settings, "clouds": cloud_settings},
        ),
        output=f3d.OutputSpec(
            width=int(width),
            height=int(height),
            format="png",
            path=str(output_path) if output_path is not None else None,
        ),
        reproducibility_profile=f3d.ReproducibilityProfile(seed=2026),
    )


def render_frames(output_dir: str | Path, *, frames: int = 1, width: int = 640, height: int = 360) -> list[Path]:
    """Render one or more deterministic frames with moving cloud shadows."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    frame_count = max(1, int(frames))
    for frame_index in range(frame_count):
        path = target_dir / f"mapscene_water_clouds_{frame_index:03d}.png"
        scene = build_scene(path, frame_index=frame_index, total_frames=frame_count, width=width, height=height)
        scene.render(path)
        rendered.append(path)
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("out/mapscene_water_clouds"))
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args(argv)
    for path in render_frames(args.output_dir, frames=args.frames, width=args.width, height=args.height):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
