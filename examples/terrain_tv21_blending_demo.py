from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _import_forge3d():
    try:
        import forge3d as f3d

        return f3d
    except ModuleNotFoundError:
        from _import_shim import ensure_repo_import

        ensure_repo_import()
        import forge3d as f3d

        return f3d


f3d = _import_forge3d()

from forge3d.geometry import MeshBuffers
from forge3d.terrain_params import PomSettings, make_terrain_params_config
from forge3d.terrain_scatter import (
    TerrainContactSettings,
    TerrainMeshBlendSettings,
    TerrainScatterBatch,
    TerrainScatterLevel,
    TerrainScatterSource,
    apply_to_renderer,
    make_transform_row_major,
)


DEFAULT_DEM = Path(__file__).resolve().parent.parent / "assets" / "tif" / "dem_rainier.tif"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "out" / "terrain_tv21_blending_demo"
DEFAULT_CROP_SIZE = 160
DEFAULT_TERRAIN_SPAN = 220.0


@dataclass(frozen=True)
class CropSpec:
    name: str
    prefer_uv: tuple[float, float]
    target_slope_deg: float
    slope_sigma_deg: float
    target_elevation_norm: float
    elevation_sigma: float


@dataclass(frozen=True)
class Tv21Case:
    name: str
    crop_origin_rc: tuple[int, int]
    heightmap: np.ndarray
    batch: TerrainScatterBatch
    z_scale: float
    cam_radius: float
    cam_phi_deg: float
    cam_theta_deg: float


def _write_preview_hdr(path: Path, width: int = 8, height: int = 4) -> None:
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                handle.write(bytes([r, g, 180, 128]))


def _downsample_heightmap(heightmap: np.ndarray, max_dim: int) -> np.ndarray:
    if max_dim <= 0:
        return np.ascontiguousarray(heightmap)
    longest = max(int(heightmap.shape[0]), int(heightmap.shape[1]))
    if longest <= max_dim:
        return np.ascontiguousarray(heightmap)
    step = int(math.ceil(longest / max_dim))
    return np.ascontiguousarray(heightmap[::step, ::step])


def _load_dem(path: Path, max_dim: int) -> tuple[object, np.ndarray]:
    try:
        dem = f3d.io.load_dem(str(path), fill_nodata_values=True)
    except ImportError:
        try:
            from PIL import Image
        except ImportError as exc:
            raise SystemExit(
                "DEM loading requires rasterio or Pillow. "
                "Install with `pip install rasterio` or `pip install pillow`."
            ) from exc

        with Image.open(path) as image:
            data = np.asarray(image, dtype=np.float32)
        if data.ndim == 3:
            data = data[..., 0]
        dem = f3d.io.load_dem_from_array(data)

    heightmap = np.asarray(dem.data, dtype=np.float32).copy()
    return dem, _downsample_heightmap(heightmap, max_dim)


def _make_overlay(domain: tuple[float, float]):
    lo, hi = map(float, domain)
    span = max(hi - lo, 1e-6)
    cmap = f3d.Colormap1D.from_stops(
        stops=[
            (lo + 0.00 * span, "#132d19"),
            (lo + 0.18 * span, "#3c6a2f"),
            (lo + 0.42 * span, "#728046"),
            (lo + 0.64 * span, "#957957"),
            (lo + 0.82 * span, "#c7b7a0"),
            (lo + 1.00 * span, "#f3f5fb"),
        ],
        domain=(lo, hi),
    )
    return f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)


def _scaled_grounded_mesh(mesh: MeshBuffers, scale_xyz: tuple[float, float, float]) -> MeshBuffers:
    positions = np.asarray(mesh.positions, dtype=np.float32).copy()
    positions *= np.asarray(scale_xyz, dtype=np.float32)
    positions[:, 1] -= float(np.min(positions[:, 1]))
    return MeshBuffers(
        positions=positions,
        normals=np.asarray(mesh.normals, dtype=np.float32).copy(),
        uvs=np.asarray(mesh.uvs, dtype=np.float32).copy(),
        indices=np.asarray(mesh.indices, dtype=np.uint32).copy(),
        tangents=None if mesh.tangents is None else np.asarray(mesh.tangents, dtype=np.float32).copy(),
    )


def _relief_scale(domain: tuple[float, float], terrain_span: float) -> float:
    relief = max(float(domain[1]) - float(domain[0]), 1e-6)
    return float(np.clip((terrain_span / relief) * 0.18, 0.12, 10.0))


def _surface_translation(
    source: TerrainScatterSource,
    x: float,
    z: float,
    *,
    bury: float,
) -> tuple[float, float, float]:
    row, col = source.contract_to_pixel(x, z)
    y = source.sample_scaled_height(row, col) - float(bury)
    return (float(x), float(y), float(z))


def _select_crop_center(heightmap: np.ndarray, spec: CropSpec, crop_size: int) -> tuple[int, int]:
    source = TerrainScatterSource(heightmap, z_scale=1.0)
    rows = np.linspace(0.0, 1.0, source.height, dtype=np.float32)[:, None]
    cols = np.linspace(0.0, 1.0, source.width, dtype=np.float32)[None, :]
    slope = source.slope_degrees
    elevation = source.normalized_elevation

    position_term = ((rows - float(spec.prefer_uv[1])) / 0.22) ** 2 + (
        (cols - float(spec.prefer_uv[0])) / 0.22
    ) ** 2
    slope_term = ((slope - float(spec.target_slope_deg)) / max(float(spec.slope_sigma_deg), 1e-3)) ** 2
    elevation_term = (
        (elevation - float(spec.target_elevation_norm)) / max(float(spec.elevation_sigma), 1e-3)
    ) ** 2
    score = -(position_term + slope_term + elevation_term)

    half = max(int(crop_size // 2) + 2, 0)
    if source.height > (half * 2) and source.width > (half * 2):
        valid = np.zeros_like(score, dtype=bool)
        valid[half:-half, half:-half] = True
        score = np.where(valid, score, -np.inf)

    best_row, best_col = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(best_row), int(best_col)


def _crop_heightmap(heightmap: np.ndarray, center_row: int, center_col: int, crop_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    crop_size = max(32, min(int(crop_size), int(heightmap.shape[0]), int(heightmap.shape[1])))
    row0 = int(np.clip(center_row - crop_size // 2, 0, heightmap.shape[0] - crop_size))
    col0 = int(np.clip(center_col - crop_size // 2, 0, heightmap.shape[1] - crop_size))
    row1 = row0 + crop_size
    col1 = col0 + crop_size
    return np.ascontiguousarray(heightmap[row0:row1, col0:col1]), (row0, col0)


def _normalize_crop_heightmap(
    heightmap: np.ndarray,
    *,
    relief_target: float = 42.0,
    base_height: float = 18.0,
) -> np.ndarray:
    arr = np.asarray(heightmap, dtype=np.float32)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = max(hi - lo, 1e-6)
    normalized = (arr - lo) / span
    return np.ascontiguousarray(base_height + normalized * float(relief_target))


def _build_case(case_name: str, heightmap: np.ndarray, crop_origin_rc: tuple[int, int]) -> Tv21Case:
    domain = (float(np.min(heightmap)), float(np.max(heightmap)))
    z_scale = _relief_scale(domain, DEFAULT_TERRAIN_SPAN)
    source = TerrainScatterSource(heightmap, z_scale=z_scale)
    center = source.terrain_width * 0.5

    if case_name == "rock_cluster":
        mesh = _scaled_grounded_mesh(
            f3d.geometry.primitive_mesh("cylinder", radial_segments=10),
            (4.0, 6.2, 4.0),
        )
        transforms = np.asarray(
            [
                make_transform_row_major(
                    _surface_translation(source, center - 10.0, center - 5.0, bury=1.0),
                    yaw_deg=22.0,
                    scale=0.82,
                ),
                make_transform_row_major(
                    _surface_translation(source, center, center, bury=1.1),
                    yaw_deg=41.0,
                    scale=1.08,
                ),
                make_transform_row_major(
                    _surface_translation(source, center + 9.0, center + 6.0, bury=0.9),
                    yaw_deg=14.0,
                    scale=0.96,
                ),
                make_transform_row_major(
                    _surface_translation(source, center + 5.0, center - 11.0, bury=0.8),
                    yaw_deg=61.0,
                    scale=0.74,
                ),
            ],
            dtype=np.float32,
        )
        batch = TerrainScatterBatch(
            name="rocks",
            color=(0.55, 0.43, 0.32, 1.0),
            transforms=transforms,
            terrain_blend=TerrainMeshBlendSettings(enabled=True, bury_depth=1.4, fade_distance=3.0),
            terrain_contact=TerrainContactSettings(
                enabled=True,
                distance=2.7,
                strength=0.38,
                vertical_weight=0.55,
            ),
            levels=[TerrainScatterLevel(mesh=mesh)],
        )
        return Tv21Case(case_name, crop_origin_rc, heightmap, batch, z_scale, 84.0, 140.0, 58.0)

    if case_name == "road_edge":
        mesh = _scaled_grounded_mesh(f3d.geometry.primitive_mesh("box"), (40.0, 3.5, 8.0))
        batch = TerrainScatterBatch(
            name="road_edge",
            color=(0.55, 0.50, 0.44, 1.0),
            transforms=np.asarray(
                [
                    make_transform_row_major(
                        _surface_translation(source, center, center, bury=1.8),
                        yaw_deg=28.0,
                        scale=1.0,
                    )
                ],
                dtype=np.float32,
            ),
            terrain_blend=TerrainMeshBlendSettings(enabled=True, bury_depth=2.8, fade_distance=6.0),
            terrain_contact=TerrainContactSettings(
                enabled=True,
                distance=7.0,
                strength=1.0,
                vertical_weight=0.0,
            ),
            levels=[TerrainScatterLevel(mesh=mesh)],
        )
        return Tv21Case(case_name, crop_origin_rc, heightmap, batch, z_scale, 74.0, 134.0, 58.0)

    if case_name == "building_foundation":
        mesh = _scaled_grounded_mesh(f3d.geometry.primitive_mesh("box"), (34.0, 3.2, 34.0))
        batch = TerrainScatterBatch(
            name="foundation",
            color=(0.82, 0.82, 0.78, 1.0),
            transforms=np.asarray(
                [
                    make_transform_row_major(
                        _surface_translation(source, center, center, bury=1.8),
                        yaw_deg=12.0,
                        scale=1.0,
                    )
                ],
                dtype=np.float32,
            ),
            terrain_blend=TerrainMeshBlendSettings(enabled=True, bury_depth=2.6, fade_distance=5.4),
            terrain_contact=TerrainContactSettings(
                enabled=True,
                distance=5.5,
                strength=1.0,
                vertical_weight=0.2,
            ),
            levels=[TerrainScatterLevel(mesh=mesh)],
        )
        return Tv21Case(case_name, crop_origin_rc, heightmap, batch, z_scale, 80.0, 144.0, 62.0)

    raise AssertionError(f"Unknown TV21 case: {case_name}")


def _render_case(
    case: Tv21Case,
    batch: TerrainScatterBatch,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    domain = (float(np.min(case.heightmap)), float(np.max(case.heightmap)))

    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()

    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        hdr_path = Path(tmp.name)
    try:
        _write_preview_hdr(hdr_path)
        ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    finally:
        hdr_path.unlink(missing_ok=True)

    config = make_terrain_params_config(
        size_px=(width, height),
        render_scale=1.0,
        terrain_span=DEFAULT_TERRAIN_SPAN,
        msaa_samples=4,
        z_scale=case.z_scale,
        exposure=1.0,
        domain=domain,
        albedo_mode="mix",
        colormap_strength=0.3,
        overlays=[_make_overlay(domain)],
        cam_radius=case.cam_radius,
        cam_phi_deg=case.cam_phi_deg,
        cam_theta_deg=case.cam_theta_deg,
        fov_y_deg=46.0,
        camera_mode="mesh",
        clip=(0.1, DEFAULT_TERRAIN_SPAN * 4.0),
        light_azimuth_deg=136.0,
        light_elevation_deg=26.0,
        sun_intensity=2.6,
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
    )
    config.cam_target = [0.0, 0.0, 0.0]

    apply_to_renderer(renderer, [batch])
    frame = renderer.render_terrain_pbr_pom(
        material_set,
        ibl,
        f3d.TerrainRenderParams(config),
        case.heightmap,
    )
    return frame.to_numpy()


def _baseline_batch(case: Tv21Case) -> TerrainScatterBatch:
    return TerrainScatterBatch(
        name=case.batch.name,
        color=case.batch.color,
        transforms=case.batch.transforms.copy(),
        levels=case.batch.levels,
    )


def _changed_pixels(before: np.ndarray, after: np.ndarray) -> int:
    return int(np.count_nonzero(np.any(before != after, axis=-1)))


def _mean_delta(before: np.ndarray, after: np.ndarray) -> float:
    return float(np.mean(np.abs(after.astype(np.int16) - before.astype(np.int16))))


def _make_diff_image(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    delta = np.max(np.abs(after.astype(np.int16) - before.astype(np.int16)), axis=-1)
    delta_u8 = np.clip(delta * 3, 0, 255).astype(np.uint8)
    rgb = np.stack(
        [
            delta_u8,
            np.clip(delta_u8 // 2 + 24, 0, 255).astype(np.uint8),
            np.clip(255 - (delta_u8 // 2), 0, 255).astype(np.uint8),
        ],
        axis=-1,
    )
    alpha = np.full(delta_u8.shape, 255, dtype=np.uint8)
    return np.ascontiguousarray(np.dstack([rgb, alpha]))


def _compose_contact_sheet(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    cell_h, cell_w, _ = rows[0][0].shape
    gutter = 12
    sheet_h = len(rows) * cell_h + (len(rows) + 1) * gutter
    sheet_w = 3 * cell_w + 4 * gutter
    sheet = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)
    sheet[..., 0] = 16
    sheet[..., 1] = 18
    sheet[..., 2] = 22
    sheet[..., 3] = 255

    for row_index, row in enumerate(rows):
        y0 = gutter + row_index * (cell_h + gutter)
        for col_index, image in enumerate(row):
            x0 = gutter + col_index * (cell_w + gutter)
            sheet[y0 : y0 + cell_h, x0 : x0 + cell_w] = image
    return np.ascontiguousarray(sheet)


def _write_png(path: Path, image: np.ndarray) -> None:
    f3d.numpy_to_png(str(path), np.ascontiguousarray(image, dtype=np.uint8))


def render_tv21_demo(
    *,
    dem_path: Path = DEFAULT_DEM,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    width: int = 640,
    height: int = 420,
    max_dem_size: int = 1024,
    crop_size: int = DEFAULT_CROP_SIZE,
) -> dict[str, object]:
    dem, heightmap = _load_dem(Path(dem_path), int(max_dem_size))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_specs = [
        CropSpec(
            name="rock_cluster",
            prefer_uv=(0.42, 0.33),
            target_slope_deg=24.0,
            slope_sigma_deg=12.0,
            target_elevation_norm=0.72,
            elevation_sigma=0.14,
        ),
        CropSpec(
            name="road_edge",
            prefer_uv=(0.58, 0.52),
            target_slope_deg=14.0,
            slope_sigma_deg=8.0,
            target_elevation_norm=0.46,
            elevation_sigma=0.14,
        ),
        CropSpec(
            name="building_foundation",
            prefer_uv=(0.52, 0.70),
            target_slope_deg=8.0,
            slope_sigma_deg=6.0,
            target_elevation_norm=0.34,
            elevation_sigma=0.12,
        ),
    ]

    summary_cases: list[dict[str, object]] = []
    contact_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for spec in crop_specs:
        center_row, center_col = _select_crop_center(heightmap, spec, int(crop_size))
        crop, origin_rc = _crop_heightmap(heightmap, center_row, center_col, int(crop_size))
        crop = _normalize_crop_heightmap(crop)
        case = _build_case(spec.name, crop, origin_rc)

        baseline = _render_case(case, _baseline_batch(case), width=int(width), height=int(height))
        enabled = _render_case(case, case.batch, width=int(width), height=int(height))
        diff = _make_diff_image(baseline, enabled)

        baseline_path = output_dir / f"terrain_tv21_{case.name}_baseline.png"
        enabled_path = output_dir / f"terrain_tv21_{case.name}_tv21.png"
        diff_path = output_dir / f"terrain_tv21_{case.name}_diff.png"
        _write_png(baseline_path, baseline)
        _write_png(enabled_path, enabled)
        _write_png(diff_path, diff)

        changed = _changed_pixels(baseline, enabled)
        mean_delta = _mean_delta(baseline, enabled)
        summary_cases.append(
            {
                "name": case.name,
                "crop_origin_rc": [int(origin_rc[0]), int(origin_rc[1])],
                "crop_size": int(crop.shape[0]),
                "changed_pixels": changed,
                "mean_delta": mean_delta,
                "baseline_path": str(baseline_path),
                "tv21_path": str(enabled_path),
                "diff_path": str(diff_path),
            }
        )
        contact_rows.append((baseline, enabled, diff))

    contact_sheet_path = output_dir / "terrain_tv21_contact_sheet.png"
    summary_path = output_dir / "terrain_tv21_summary.json"
    _write_png(contact_sheet_path, _compose_contact_sheet(contact_rows))

    summary: dict[str, object] = {
        "dem": str(Path(dem_path).resolve()),
        "effective_dem_size": [int(heightmap.shape[1]), int(heightmap.shape[0])],
        "output_dir": str(output_dir),
        "contact_sheet_path": str(contact_sheet_path),
        "cases": summary_cases,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Epic TV21 demo: render real-DEM terrain crops showing baseline scatter, "
            "terrain-aware mesh blending, and contact shading."
        )
    )
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM, help="Path to a DEM GeoTIFF")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--width", type=int, default=640, help="Per-panel output width in pixels")
    parser.add_argument("--height", type=int, default=420, help="Per-panel output height in pixels")
    parser.add_argument(
        "--max-dem-size",
        type=int,
        default=1024,
        help="Clamp the longest DEM dimension before crop selection and rendering (0 disables downsampling)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=DEFAULT_CROP_SIZE,
        help="Square crop size in DEM samples for each TV21 case",
    )
    args = parser.parse_args()

    dem_path = args.dem.resolve()
    if not dem_path.exists():
        raise SystemExit(f"DEM not found: {dem_path}")

    summary = render_tv21_demo(
        dem_path=dem_path,
        output_dir=args.output_dir.resolve(),
        width=int(args.width),
        height=int(args.height),
        max_dem_size=int(args.max_dem_size),
        crop_size=int(args.crop_size),
    )

    print(f"DEM source: {summary['dem']}")
    print(f"Effective DEM size: {summary['effective_dem_size'][0]}x{summary['effective_dem_size'][1]}")
    print(f"Contact sheet: {summary['contact_sheet_path']}")
    for case in summary["cases"]:
        print(
            f"{case['name']}: changed_pixels={case['changed_pixels']} "
            f"mean_delta={case['mean_delta']:.3f} crop_origin_rc={tuple(case['crop_origin_rc'])}"
        )
        print(f"  baseline={case['baseline_path']}")
        print(f"  tv21={case['tv21_path']}")
        print(f"  diff={case['diff_path']}")
    print(f"Summary JSON: {Path(summary['output_dir']) / 'terrain_tv21_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
