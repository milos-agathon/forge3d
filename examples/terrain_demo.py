# examples/terrain_demo.py
# Render a high quality PBR terrain from a local DEM using forge3d.
# Exists to demonstrate Session → DEM → PBR render pipeline with custom LUT.
# RELEVANT FILES:examples/_import_shim.py,python/forge3d/__init__.py,python/forge3d/io.py,task.md

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from _import_shim import ensure_repo_import

ensure_repo_import()

try:
    import forge3d as f3d
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "forge3d import failed. Build the PyO3 extension with `maturin develop --release`."
    ) from exc


DEFAULT_DEM = Path("assets/Gore_Range_Albers_1m.tif")
DEFAULT_HDR = Path("assets/snow_field_4k.hdr")
DEFAULT_OUTPUT = Path("examples/out/terrain_demo.png")
DEFAULT_SIZE = (2560, 1440)
DEFAULT_DOMAIN = (200.0, 2200.0)
DEFAULT_COLORMAP_STOPS: Sequence[tuple[float, str]] = (
    (200.0, "#e7d8a2"),
    (800.0, "#c5a06e"),
    (1500.0, "#995f57"),
    (2200.0, "#4a3c37"),
)


def _require_attributes(attr_names: Iterable[str]) -> None:
    missing = [name for name in attr_names if not hasattr(f3d, name)]
    if missing:
        raise SystemExit(
            "Required forge3d attributes are missing in this build: "
            + ", ".join(missing)
        )


def _load_dem(path: Path):
    io_mod = getattr(f3d, "io", None)
    if io_mod is None:
        try:
            from forge3d import io as io_mod  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise SystemExit("forge3d.io module is unavailable.") from exc

    load_dem_fn = getattr(io_mod, "load_dem", None)
    if load_dem_fn is None:
        raise SystemExit(
            "forge3d.io.load_dem is not available. Update the forge3d build to a version "
            "that exposes the DEM loader."
        )

    return load_dem_fn(
        str(path),
        fill_nodata=True,
        to_local_metric=True,
    )


def _infer_domain(dem, fallback: tuple[float, float]) -> tuple[float, float]:
    candidates: Sequence[str] = (
        "domain",
        "elevation_range",
        "height_range",
        "bounds",
        "range",
    )
    for name in candidates:
        value = getattr(dem, name, None)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            try:
                lo = float(value[0])
                hi = float(value[1])
            except Exception:
                continue
            if hi > lo:
                return (lo, hi)

    min_keys = ("min_elevation", "min_height", "minimum")
    max_keys = ("max_elevation", "max_height", "maximum")
    for min_name in min_keys:
        for max_name in max_keys:
            lo = getattr(dem, min_name, None)
            hi = getattr(dem, max_name, None)
            if lo is None or hi is None:
                continue
            try:
                lo_f = float(lo)
                hi_f = float(hi)
            except Exception:
                continue
            if hi_f > lo_f:
                return (lo_f, hi_f)

    stats = getattr(dem, "stats", None)
    if callable(stats):
        try:
            result = stats()
        except Exception:
            result = None
        if isinstance(result, dict):
            lo = result.get("min")
            hi = result.get("max")
            try:
                lo_f = float(lo)
                hi_f = float(hi)
            except Exception:
                lo_f = None
                hi_f = None
            if lo_f is not None and hi_f is not None and hi_f > lo_f:
                return (lo_f, hi_f)

    return fallback


def _build_colormap(domain: tuple[float, float]):
    stops = []
    for value, color in DEFAULT_COLORMAP_STOPS:
        mapped_value = float(value)
        stops.append((mapped_value, color))

    stops[0] = (domain[0], stops[0][1])
    stops[-1] = (domain[1], stops[-1][1])

    return f3d.Colormap1D.from_stops(
        stops=stops,
        domain=domain,
    )


def _get_height_texture(dem):
    for attr in ("texture_view", "texture", "height_texture"):
        if hasattr(dem, attr):
            return getattr(dem, attr)
    raise SystemExit("DEM handle does not expose a GPU texture view.")


def _build_params(
    size: tuple[int, int],
    render_scale: float,
    msaa: int,
    z_scale: float,
    exposure: float,
    domain: tuple[float, float],
    colormap,
):
    return f3d.TerrainRenderParams(
        size_px=size,
        render_scale=render_scale,
        msaa_samples=msaa,
        z_scale=z_scale,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=1200.0,
        cam_phi_deg=135.0,
        cam_theta_deg=45.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 6000.0),
        light=f3d.LightSettings(
            light_type="Directional",
            azimuth_deg=135.0,
            elevation_deg=35.0,
            intensity=3.0,
            color=[1.0, 1.0, 1.0],
        ),
        ibl=f3d.IblSettings(
            enabled=True,
            intensity=1.0,
            rotation_deg=0.0,
        ),
        shadows=f3d.ShadowSettings(
            enabled=True,
            technique="PCSS",
            resolution=4096,
            cascades=3,
            max_distance=4000.0,
            softness=1.5,
            intensity=0.8,
            slope_scale_bias=0.5,
            depth_bias=0.002,
            normal_bias=0.5,
            min_variance=1e-4,
            light_bleed_reduction=0.5,
            evsm_exponent=0.0,
            fade_start=1.0,
        ),
        triplanar=f3d.TriplanarSettings(
            scale=6.0,
            blend_sharpness=4.0,
            normal_strength=1.0,
        ),
        pom=f3d.PomSettings(
            enabled=True,
            mode="Occlusion",
            scale=0.04,
            min_steps=12,
            max_steps=40,
            refine_steps=4,
            shadow=True,
            occlusion=True,
        ),
        lod=f3d.LodSettings(level=0, bias=0.0, lod0_bias=-0.5),
        sampling=f3d.SamplingSettings(
            mag_filter="Linear",
            min_filter="Linear",
            mip_filter="Linear",
            anisotropy=8,
            address_u="Repeat",
            address_v="Repeat",
            address_w="Repeat",
        ),
        clamp=f3d.ClampSettings(
            height_range=(domain[0], domain[1]),
            slope_range=(0.04, 1.0),
            ambient_range=(0.0, 1.0),
            shadow_range=(0.0, 1.0),
            occlusion_range=(0.0, 1.0),
        ),
        overlays=[
            f3d.OverlayLayer.from_colormap1d(
                colormap,
                strength=1.0,
                offset=0.0,
                blend_mode="Alpha",
                domain=domain,
            )
        ],
        exposure=exposure,
        gamma=2.2,
        albedo_mode="mix",
        colormap_strength=0.5,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a forge3d PBR terrain from a GeoTIFF DEM."
    )
    parser.add_argument(
        "--dem",
        type=Path,
        default=DEFAULT_DEM,
        help="Path to a GeoTIFF DEM.",
    )
    parser.add_argument(
        "--hdr",
        type=Path,
        default=DEFAULT_HDR,
        help="Path to an environment HDR image.",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=DEFAULT_SIZE,
        metavar=("WIDTH", "HEIGHT"),
        help="Output resolution in pixels.",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=1.0,
        help="Internal render scale multiplier.",
    )
    parser.add_argument(
        "--msaa",
        type=int,
        default=4,
        help="MSAA sample count.",
    )
    parser.add_argument(
        "--z-scale",
        dest="z_scale",
        type=float,
        default=1.5,
        help="Vertical exaggeration factor.",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=1.0,
        help="ACES exposure multiplier.",
    )
    parser.add_argument(
        "--colormap-domain",
        type=float,
        nargs=2,
        metavar=("MIN_ELEV", "MAX_ELEV"),
        help="Override the colormap elevation domain.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination image path (PNG/EXR/TIFF).",
    )
    parser.add_argument(
        "--window",
        action="store_true",
        help="Open a display window instead of headless rendering.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch the interactive viewer after rendering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    _require_attributes(
        (
            "Session",
            "TerrainRenderer",
            "MaterialSet",
            "IBL",
            "Colormap1D",
            "TerrainRenderParams",
            "LightSettings",
            "IblSettings",
            "ShadowSettings",
            "TriplanarSettings",
            "PomSettings",
            "LodSettings",
            "SamplingSettings",
            "ClampSettings",
            "OverlayLayer",
        )
    )

    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"Output file already exists: {args.output}. Use --overwrite to replace it."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    sess = f3d.Session(window=bool(args.window))

    dem = _load_dem(args.dem)
    height_texture = _get_height_texture(dem)

    domain = (
        tuple(args.colormap_domain)  # type: ignore[arg-type]
        if args.colormap_domain is not None
        else _infer_domain(dem, DEFAULT_DOMAIN)
    )
    domain = (float(domain[0]), float(domain[1]))

    colormap = _build_colormap(domain)

    materials = f3d.MaterialSet.terrain_default(
        triplanar_scale=6.0,
        normal_strength=1.0,
        blend_sharpness=4.0,
    )

    ibl = f3d.IBL.from_hdr(
        str(args.hdr),
        intensity=1.0,
        rotate_deg=0.0,
    )

    params = _build_params(
        size=(int(args.size[0]), int(args.size[1])),
        render_scale=float(args.render_scale),
        msaa=int(args.msaa),
        z_scale=float(args.z_scale),
        exposure=float(args.exposure),
        domain=domain,
        colormap=colormap,
    )

    renderer = f3d.TerrainRenderer(sess.device)

    frame = renderer.render_terrain_pbr_pom(
        target=None,
        heightmap=height_texture,
        material_set=materials,
        env_maps=ibl,
        params=params,
    )

    frame.save(str(args.output))
    print(f"Wrote {args.output}")

    if args.viewer:
        viewer_cls = getattr(f3d, "Viewer", None)
        if viewer_cls is None:
            print("forge3d.Viewer is not available in this build. Skipping interactive viewer.")
        else:
            with viewer_cls(sess, renderer, height_texture, materials, ibl, params) as view:
                view.run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
