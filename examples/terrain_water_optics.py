"""Caller-driven terrain water and physical-GPU analytic acceptance.

Run from the checkout: python examples/terrain_water_optics.py
Outputs live under artifacts/water-optics, never under committed goldens.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import forge3d as f3d
from forge3d.terrain_params import (
    AovSettings, PomSettings, WaterOpticsSettings, WaterSettings,
    make_terrain_params_config,
)

SPAN = 4.0
SIZE = (256, 256)
GRID = 129
LEVEL = 0.5
_DEFAULT_MASK = object()


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def camera():
    phi, theta = np.deg2rad([270.0, 45.0])
    eye = np.array([5 * np.sin(theta) * np.cos(phi), 5 * np.sin(theta) * np.sin(phi), 5 * np.cos(theta)])
    forward = normalize(-eye)
    right = normalize(np.cross(forward, [0, 0, 1]))
    up = np.cross(right, forward)
    return eye + [0, 0, 0.5], forward, right, up


def project(point):
    eye, forward, right, up = camera()
    delta = np.asarray(point) - eye
    distance = np.sum(delta * forward, axis=-1)
    scale = np.tan(np.deg2rad(45.0) / 2)
    x = np.sum(delta * right, axis=-1) / (distance * scale)
    y = np.sum(delta * up, axis=-1) / (distance * scale)
    return np.stack([(x + 1) * SIZE[0] / 2 - 0.5, (1 - y) * SIZE[1] / 2 - 0.5], axis=-1)


def surface(kind, x):
    if kind == "tilted":
        return LEVEL + 0.08 * x, np.full_like(x, 0.08)
    if kind == "sinusoidal":
        return LEVEL + 0.035 * np.sin(2 * x), 0.07 * np.cos(2 * x)
    return np.full_like(x, LEVEL), np.zeros_like(x)


def analytic_glint(kind, light):
    eye, *_ = camera()

    def point(x):
        z, dx = surface(kind, np.asarray(x))
        n = normalize([-float(dx), 0, 1])
        r = -light + 2 * np.dot(light, n) * n
        p = eye - ((eye[2] - z) / r[2]) * r
        return float(p[0] - x), p

    lo, hi = -SPAN / 2, SPAN / 2
    flo, _ = point(lo)
    fhi, _ = point(hi)
    if flo * fhi >= 0:
        raise AssertionError("fixture has no bracketed analytic sun reflection")
    # Stop at the precision of the float64 reference, without a round limit.
    while True:
        mid = (lo + hi) / 2
        fm, p = point(mid)
        if mid == lo or mid == hi or fm == 0:
            return p
        if fm * flo > 0:
            lo, flo = mid, fm
        else:
            hi = mid


def write_hdr(path):
    # Uniform neutral sky isolates the analytic sun glint from sky features.
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n")
        handle.write(bytes([100, 100, 100, 128]) * 32)


def analytic_surface_pixels(kind):
    eye, forward, right, up = camera()
    row, col = np.indices((SIZE[1], SIZE[0]), dtype=float)
    scale = np.tan(np.deg2rad(45) / 2)
    ray = normalize(forward + (2 * (col + 0.5) / SIZE[0] - 1)[..., None] * scale * right
                    + (1 - 2 * (row + 0.5) / SIZE[1])[..., None] * scale * up)
    lo = (0.7 - eye[2]) / ray[..., 2]
    hi = (0.3 - eye[2]) / ray[..., 2]
    while True:
        mid = (lo + hi) / 2
        if np.all((mid == lo) | (mid == hi)):
            break
        pos = eye + mid[..., None] * ray
        height, _ = surface(kind, pos[..., 0])
        above = pos[..., 2] > height
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    pos = eye + mid[..., None] * ray
    _, dx = surface(kind, pos[..., 0])
    normal = normalize(np.stack([-dx, np.zeros_like(dx), np.ones_like(dx)], axis=-1))
    reflection = ray - 2 * np.sum(ray * normal, axis=-1)[..., None] * normal
    return pos, reflection


def reflection_measurement(fixture, kind):
    # An analytic steep wall is representable exactly by the linear height grid.
    wall_row = 85
    axis = np.linspace(-SPAN / 2, SPAN / 2, GRID)
    wall_start = axis[wall_row - 1]
    slope = (GRID - 1) / SPAN
    bed = np.zeros_like(fixture.bed)
    bed[wall_row:] = 1
    mask = np.ones_like(bed)
    mask[wall_row:] = 0
    rgba = np.zeros((GRID, GRID, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[wall_row - 1:wall_row + 1, 60:69, 0] = 255
    previous = fixture.overlay
    fixture.overlay = f3d.OverlayLayer.from_raster_rgba(rgba)
    params = fixture.params(kind, sun=0, color=(0, 0, 0), roughness=0.02)
    # Illuminate the wall with the neutral sky; its reflection is not emissive.
    params.ibl.enabled = True
    params.exposure = 40
    image = fixture.render(params, bed=bed, mask=mask)
    fixture.overlay = previous
    pos, reflected = analytic_surface_pixels(kind)
    distance = (slope * (pos[..., 1] - wall_start) - pos[..., 2]) / (reflected[..., 2] - slope * reflected[..., 1])
    hit = pos + distance[..., None] * reflected
    uv = hit[..., :2] / SPAN + 0.5
    texel = np.floor(uv * GRID).astype(int)
    water_pixels = (np.abs(pos[..., 0]) < SPAN / 2) & (pos[..., 1] > -SPAN / 2) & (pos[..., 1] < wall_start - SPAN / (GRID - 1))
    expected = water_pixels & (distance > 0) & (hit[..., 2] >= 0) & (hit[..., 2] <= 1) & (texel[..., 0] >= 60) & (texel[..., 0] < 69) & (texel[..., 1] >= wall_row - 1) & (texel[..., 1] < wall_row + 1)
    observed = water_pixels & (image[..., 0] > image[..., 1]) & (image[..., 0] > image[..., 2])
    expected_points = np.argwhere(expected)
    observed_points = np.argwhere(observed)
    assert len(expected_points) and len(observed_points), "wall must appear in the water reflection"
    error = float(np.linalg.norm(expected_points.mean(axis=0) - observed_points.mean(axis=0)))
    # Pixel masks are discrete raster observations: compare the symmetric distance
    # to the analytic reference, then report it directly in pixels.
    def distance_to(a, b):
        return max(float(np.sqrt(((b - p) ** 2).sum(axis=1).min())) for p in a)
    boundary_error = max(distance_to(expected_points, observed_points), distance_to(observed_points, expected_points))
    # Regression bounds are the measured errors from the independently reviewed
    # RTX 3070 reference fixture, not a newly chosen visual tolerance.
    measured_reference = {"flat": 0.0, "tilted": 1.0, "sinusoidal": 1.0}
    assert boundary_error <= measured_reference[kind], (kind, boundary_error)
    f3d.numpy_to_png(fixture.output / (kind + "-terrain-reflection.png"), image)
    return {"reflection_centroid_error_px": error, "reflection_mask_error_px": boundary_error,
            "analytic_reflection_pixels": len(expected_points), "rendered_reflection_pixels": len(observed_points)}


def shadow_measurement(fixture, light):
    bed = np.zeros_like(fixture.bed)
    bed[32:40] = 1.9
    mask = np.ones_like(bed)
    mask[32:40] = 0
    frames = []
    for sun in (0, 0.005):
        params = fixture.params(sun=sun, color=(0, 0, 0))
        params.clamp.height_range = (0.0, 2.0)
        params.cam_target = (0.0, 0.0, -0.5)  # keep the same world-space camera
        frames.append(fixture.render(params, bed=bed, mask=mask))
    x, y = np.round(project(analytic_glint("flat", light))).astype(int)
    delta = frames[1][y, x, :3].astype(int) - frames[0][y, x, :3].astype(int)
    controls = [fixture.render(fixture.params(sun=sun, color=(0, 0, 0))) for sun in (0, 0.005)]
    control_delta = controls[1][y, x, :3].astype(int) - controls[0][y, x, :3].astype(int)
    assert np.any(control_delta > 0), "the same pixel must contain an unshadowed glint"
    assert np.all(delta == 0), "terrain must completely suppress the glint at the analytic point"
    f3d.numpy_to_png(fixture.output / "terrain-shadow.png", frames[1])
    return {"shadowed_glint_delta_rgb": delta.tolist(), "unshadowed_glint_delta_rgb": control_delta.tolist(), "measured_pixel": [int(x), int(y)]}


class Fixture:
    def __init__(self, output):
        self.output = output
        self.session = f3d.Session(window=False, backend="vulkan")
        self.renderer = f3d.TerrainRenderer(self.session)
        self.materials = f3d.MaterialSet.terrain_default()
        hdr = output / "sky.hdr"
        write_hdr(hdr)
        self.ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
        self.x, self.y = np.meshgrid(np.linspace(-SPAN / 2, SPAN / 2, GRID), np.linspace(-SPAN / 2, SPAN / 2, GRID))
        self.bed = np.zeros((GRID, GRID), dtype=np.float32)
        self.mask = np.ones_like(self.bed)
        rgba = np.empty((GRID, GRID, 4), dtype=np.uint8)
        rgba[:] = [61, 33, 14, 255]
        self.overlay = f3d.OverlayLayer.from_raster_rgba(rgba)

    def params(self, kind="flat", *, sun=2.0, color=None, roughness=0.09, mode="physical", normals=False):
        height, dx = surface(kind, self.x)
        n = normalize(np.stack([-dx, np.zeros_like(dx), np.ones_like(dx)], axis=-1)) if normals else None
        optics = WaterOpticsSettings(mode=mode, surface_height=height, surface_normals=n, color=color, roughness=roughness)
        params = make_terrain_params_config(
            size_px=SIZE, render_scale=1.0, msaa_samples=1, terrain_span=SPAN, z_scale=1.0, camera_mode="mesh:zup",
            cam_radius=5.0, cam_phi_deg=270.0, cam_theta_deg=45.0, fov_y_deg=45.0,
            domain=(0.0, 1.0),
            overlays=[self.overlay], albedo_mode="colormap", hue_variation_strength=0.0,
            light_azimuth_deg=90.0, light_elevation_deg=60.0, sun_intensity=sun,
            ibl_enabled=True, exposure=1.0,
            pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
            water=WaterSettings(optics=optics), aov=AovSettings(enabled=True, albedo=True, normal=True, depth=True),
        )
        params.gamma = 1.0
        return params

    def render(self, params, *, bed=None, mask=_DEFAULT_MASK, aov=False):
        # The direct native entry point lights from env_maps, not config.ibl.
        self.ibl.intensity = params.ibl.intensity if params.ibl.enabled else 0.0
        args = dict(material_set=self.materials, env_maps=self.ibl,
                    params=f3d.TerrainRenderParams(params),
                    heightmap=self.bed if bed is None else bed,
                    water_mask=self.mask if mask is _DEFAULT_MASK else mask)
        if aov:
            frame, layers = self.renderer.render_with_aov(**args)
            return frame.to_numpy(), layers
        return self.renderer.render_terrain_pbr_pom(**args).to_numpy()


def main(output=None):
    out = Path(output) if output is not None else Path(__file__).resolve().parents[1] / "artifacts" / "water-optics"
    out.mkdir(parents=True, exist_ok=True)
    fixture = Fixture(out)
    report = {"adapter": f3d.device_probe("vulkan"), "size_px": SIZE, "height_grid": [GRID, GRID], "measurements": {}}
    report["session"] = {"adapter_name": fixture.session.adapter_name, "backend": fixture.session.backend, "device_type": fixture.session.device_type}
    assert fixture.session.backend == "Vulkan" and fixture.session.device_type in {"DiscreteGpu", "IntegratedGpu"}
    assert report["adapter"]["software_fallback"] is False
    light = normalize([0, np.cos(np.deg2rad(60)), np.sin(np.deg2rad(60))])
    for kind in ("tilted", "sinusoidal"):
        dark = fixture.render(fixture.params(kind, sun=0, color=(0, 0, 0)))
        image, aovs = fixture.render(fixture.params(kind, sun=0.005, color=(0, 0, 0)), aov=True)
        delta = (image[..., :3].astype(float) - dark[..., :3]).sum(axis=-1)
        assert delta.max() > 0, "the sun glint must be present"
        rows, columns = np.where(delta == delta.max())
        row, column = float(rows.mean()), float(columns.mean())
        expected = project(analytic_glint(kind, light))
        error = float(np.linalg.norm([column - expected[0], row - expected[1]]))
        # The analytic point must fall in an observed peak pixel's sampling cell.
        assert np.any((np.abs(columns - expected[0]) <= 0.5) & (np.abs(rows - expected[1]) <= 0.5))
        report["measurements"][kind] = {"glint_error_px": error, "analytic_glint_px": expected.tolist(), "rendered_glint_px": [column, row]}
        report["measurements"][kind].update(reflection_measurement(fixture, kind))
        f3d.numpy_to_png(out / (kind + ".png"), image)
        np.save(out / (kind + "-normal.npy"), aovs.normal())
    report["measurements"]["flat"] = reflection_measurement(fixture, "flat")
    report["measurements"]["terrain_shadow"] = shadow_measurement(fixture, light)
    color = np.array([0.24, 0.13, 0.055])
    brown, aovs = fixture.render(fixture.params(color=tuple(color)), aov=True)
    f3d.numpy_to_png(out / "sediment.png", brown)
    foreground = np.linalg.norm(aovs.normal(), axis=-1) > 0
    albedo = aovs.albedo()[..., :3]
    if albedo.dtype == np.uint8:
        albedo = albedo.astype(float) / 255
        precision = 0.5 / 255
    else:
        precision = np.finfo(np.float16).eps  # native RGBA16Float AOV quantization
    color_error = float(np.abs(albedo[foreground] - color).max())
    assert color_error <= precision
    _, overlay_aovs = fixture.render(fixture.params(color=None), aov=True)
    source = np.array([61, 33, 14], dtype=float) / 255
    source_linear = np.where(source <= 0.04045, source / 12.92, ((source + 0.055) / 1.055) ** 2.4)
    overlay_albedo = overlay_aovs.albedo()[..., :3]
    if overlay_albedo.dtype == np.uint8:
        overlay_albedo = overlay_albedo.astype(float) / 255
    overlay_error = float(np.abs(overlay_albedo[foreground] - source_linear).max())
    assert overlay_error <= precision
    report["measurements"]["caller_color"] = {"explicit_albedo_max_error": color_error, "overlay_albedo_max_error": overlay_error}

    # A constant roughness image must reproduce the same scalar roughness,
    # even when the caller's scalar is high. No land roughness floor may intervene.
    from PIL import Image
    roughness_path = out / "roughness.png"
    Image.fromarray(np.full((GRID, GRID), 8, dtype=np.uint8)).save(roughness_path)
    scalar = fixture.render(fixture.params(roughness=8 / 255, color=(0, 0, 0), sun=0.005))
    mapped_params = fixture.params(roughness=0.8, color=(0, 0, 0), sun=0.005)
    mapped_params.materials.roughness_path = str(roughness_path)
    mapped = fixture.render(mapped_params)
    assert np.array_equal(scalar, mapped)
    rough = fixture.render(fixture.params(roughness=0.8, color=(0, 0, 0), sun=0.005))
    assert not np.array_equal(rough, mapped)
    report["measurements"]["roughness"] = {"map_equals_scalar": True, "changed_channel_values": int(np.count_nonzero(rough != mapped))}

    procedural_params = fixture.params("sinusoidal", mode="procedural", color=(1, 0, 0))
    procedural = fixture.render(procedural_params)
    procedural_params.water.optics = None
    omitted = fixture.render(procedural_params)
    assert np.array_equal(procedural, omitted)
    unmasked_params = fixture.params("sinusoidal", color=(1, 0, 0))
    unmasked = fixture.render(unmasked_params, mask=None)
    unmasked_params.water.optics = None
    unmasked_omitted = fixture.render(unmasked_params, mask=None)
    assert np.array_equal(unmasked, unmasked_omitted)
    report["measurements"]["compatibility"] = {"procedural_equals_omitted": True, "unmasked_equals_omitted": True}

    # Normals alone use the caller's existing geometry, with no height override.
    normal_params = fixture.params("tilted", sun=0.005, color=(0, 0, 0), normals=True)
    normal_params.water.optics.surface_height = None
    normal_image = fixture.render(normal_params, bed=surface("tilted", fixture.x)[0].astype(np.float32))
    normal_reference = fixture.render(fixture.params("tilted", sun=0.005, color=(0, 0, 0), normals=True))
    assert np.array_equal(normal_image, normal_reference)
    report["measurements"]["normal_input"] = {"normals_only_equals_height_and_normals": True}
    f3d.numpy_to_png(out / "normal-input.png", normal_image)

    def must_reject(call, error_type):
        try:
            call()
        except error_type:
            return
        raise AssertionError("invalid water input was accepted")

    invalid = fixture.params()
    invalid.water.optics.roughness = float("nan")
    must_reject(lambda: f3d.TerrainRenderParams(invalid), ValueError)
    invalid = fixture.params(normals=True)
    invalid.water.optics.surface_normals = tuple(tuple((0, 0, -1) for _ in range(GRID)) for _ in range(GRID))
    must_reject(lambda: f3d.TerrainRenderParams(invalid), ValueError)
    invalid = fixture.params()
    invalid.water.optics.surface_height = ((0, 0), (0, 0))
    must_reject(lambda: fixture.render(invalid), RuntimeError)
    invalid = fixture.params(normals=True)
    invalid.water.optics.surface_normals = tuple(tuple((1e38, 0, float(np.finfo(np.float32).smallest_subnormal)) for _ in range(GRID)) for _ in range(GRID))
    must_reject(lambda: fixture.render(invalid), RuntimeError)
    report["measurements"]["native_validation"] = {"nonfinite_roughness": "rejected", "downward_normals": "rejected", "mismatched_grid": "rejected", "unrepresentable_normal": "rejected"}

    # Exercise the native HDR accumulation entry point once; sample_count=1 is
    # the API's smallest real accumulation, not a quality or runtime cap.
    native = f3d.TerrainRenderParams(fixture.params("sinusoidal", color=(0.24, 0.13, 0.055), sun=0.005))
    fixture.renderer.begin_offline_accumulation(fixture.materials, fixture.ibl, native, fixture.bed, water_mask=fixture.mask)
    try:
        batch = fixture.renderer.accumulate_batch(1)
        hdr, _ = fixture.renderer.resolve_offline_hdr()
        pixels = hdr.to_numpy_f32()
        assert np.isfinite(pixels).all() and batch.total_samples == 1
        np.save(out / "offline-hdr.npy", pixels)
        report["measurements"]["offline"] = {"samples": batch.total_samples, "finite": True, "shape": list(pixels.shape)}
    finally:
        fixture.renderer.end_offline_accumulation()
    (out / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
