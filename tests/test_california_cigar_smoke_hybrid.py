from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL.Image")


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "california_cigar_smoke_demo.py"


def load_module():
    python_dir = REPO_ROOT / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))
    spec = importlib.util.spec_from_file_location("california_cigar_smoke_demo", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_hybrid_sources_stay_within_one_fire_complex() -> None:
    module = load_module()
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), (180, 140), total_frames=90, seed=17)

    assert len(sources) >= 45
    xs = np.asarray([source.x for source in sources], dtype=np.float32)
    ys = np.asarray([source.y for source in sources], dtype=np.float32)
    strengths = np.asarray([source.strength for source in sources], dtype=np.float32)
    radii = np.asarray([source.radius_px for source in sources], dtype=np.float32)
    fire_x = 0.36 * (180 - 1)
    fire_y = 0.62 * (140 - 1)
    distance = np.hypot(xs - fire_x, ys - fire_y)

    assert float(np.ptp(xs)) < 45.0
    assert float(np.ptp(ys)) < 50.0
    assert float(np.max(distance)) < 28.0
    assert float(np.max(strengths)) > float(np.min(strengths))
    assert float(np.percentile(radii, 75.0)) > 2.0
    assert any(source.start_frame > 0 for source in sources)
    assert sum(source.start_frame > 0 for source in sources) > len(sources) // 2
    assert min(source.burst_period_frames for source in sources) >= 30.0
    assert max(source.burst_period_frames for source in sources) <= 82.0
    assert any(
        module._source_burst_envelope(source, 0) < 0.20
        and module._source_burst_envelope(source, 15) > 1.0
        for source in sources
    )


def test_hybrid_density_advects_from_single_event_and_tracks_age() -> None:
    module = load_module()
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), (180, 140), total_frames=80, seed=23)
    simulator = module.HybridSmokeSimulator((180, 140), sources, seed=23)

    early = None
    state = None
    for frame in range(72):
        state = simulator.step(frame)
        if frame == 12:
            early = state.density.copy()
    assert early is not None
    assert state is not None

    later = state.density
    assert float(later.sum()) > float(early.sum()) * 1.05
    assert np.count_nonzero(later > 0.002) > np.count_nonzero(early > 0.002)
    assert np.count_nonzero(later > 0.002) > int(later.size * 0.35)
    assert np.count_nonzero(later > 0.002) < int(later.size * 0.80)

    yy, xx = np.mgrid[0 : later.shape[0], 0 : later.shape[1]].astype(np.float64)
    early_mass = np.asarray(early, dtype=np.float64)
    later_mass = np.asarray(later, dtype=np.float64)
    early_x = float(np.sum(xx * early_mass) / np.sum(early_mass))
    early_y = float(np.sum(yy * early_mass) / np.sum(early_mass))
    later_x = float(np.sum(xx * later_mass) / np.sum(later_mass))
    later_y = float(np.sum(yy * later_mass) / np.sum(later_mass))
    assert later_x > early_x + 8.0
    assert later_y < early_y - 2.0

    age = np.divide(
        state.age_mass,
        state.density,
        out=np.zeros_like(state.density),
        where=state.density > 1.0e-5,
    )
    active_age = age[state.density > 0.005]
    assert float(np.percentile(active_age, 75.0)) > 1.5
    assert float(np.percentile(active_age, 90.0)) > 10.0

    assert state.layer_density is not None
    assert state.layer_age_mass is not None
    assert len(state.layer_density) == module.HYBRID_SMOKE_LAYER_COUNT
    layer_density = np.stack(state.layer_density, axis=0)
    assert np.allclose(state.density, np.clip(np.sum(layer_density, axis=0), 0.0, 6.0))
    layer_centroids = []
    for layer in state.layer_density:
        mass = np.asarray(layer, dtype=np.float64)
        layer_centroids.append(
            (
                float(np.sum(xx * mass) / np.sum(mass)),
                float(np.sum(yy * mass) / np.sum(mass)),
            )
        )
    assert float(np.ptp([centroid[0] for centroid in layer_centroids])) > 2.0
    assert float(np.ptp([centroid[1] for centroid in layer_centroids])) > 6.0


def test_hybrid_smoke_rgba_is_filamentary_and_translucent() -> None:
    module = load_module()
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), (200, 156), total_frames=96, seed=31)
    simulator = module.HybridSmokeSimulator((200, 156), sources, seed=31)
    state = simulator.state()
    for frame in range(86):
        state = simulator.step(frame)

    rgba = module.hybrid_smoke_rgba(state, 85, seed=31)
    alpha = rgba[..., 3]

    assert rgba.shape == (156, 200, 4)
    assert rgba.dtype == np.uint8
    assert int(alpha.max()) <= module.HYBRID_SMOKE_MAX_ALPHA
    assert int(alpha.max()) >= 125
    assert np.count_nonzero(alpha) > int(alpha.size * 0.35)
    assert np.count_nonzero(alpha) < int(alpha.size * 0.72)
    assert np.count_nonzero(alpha > 80) > int(alpha.size * 0.015)
    assert np.count_nonzero(alpha > 120) > int(alpha.size * 0.0005)

    gradient_y, gradient_x = np.gradient(alpha.astype(np.float32) / 255.0)
    gradient = np.hypot(gradient_x, gradient_y)
    assert float(np.mean(gradient[alpha > 7])) > 0.009
    assert float(np.percentile(gradient[alpha > 7], 99.0)) < 0.135

    bands = [
        np.any((1 <= alpha) & (alpha <= 24)),
        np.any((25 <= alpha) & (alpha <= 62)),
        np.any((63 <= alpha) & (alpha <= 120)),
        np.any((121 <= alpha) & (alpha <= module.HYBRID_SMOKE_MAX_ALPHA)),
    ]
    assert sum(bool(item) for item in bands) >= 4

    thin_rgb = rgba[(alpha >= 1) & (alpha <= 24)][..., :3].astype(np.float32)
    dense_rgb = rgba[alpha > 120][..., :3].astype(np.float32)
    assert thin_rgb.shape[0] > 100
    assert dense_rgb.shape[0] > 20
    assert float(np.mean(thin_rgb[:, 2] - thin_rgb[:, 0])) > 12.0
    assert float(np.mean(dense_rgb)) > 195.0
    assert abs(float(np.mean(dense_rgb[:, 0] - dense_rgb[:, 2]))) < 16.0


def test_hybrid_residual_haze_persists_as_soft_aged_layer() -> None:
    module = load_module()
    source = module.HybridSmokeSource(
        x=44.0,
        y=46.0,
        strength=1.25,
        radius_px=5.2,
        start_frame=0,
        end_frame=8,
        seed=71,
        burst_period_frames=34.0,
        burst_phase_frames=0.0,
        burst_duty=0.55,
        heat=1.1,
        smoke_rate=1.2,
        altitude_bias=0.30,
    )
    simulator = module.HybridSmokeSimulator((128, 96), [source], seed=71)
    state = simulator.state()
    for frame in range(96):
        state = simulator.step(frame)

    assert state.residual_haze is not None
    haze = state.residual_haze
    assert float(haze.max()) > 0.01
    assert np.count_nonzero(haze > 0.002) > int(haze.size * 0.08)

    rgba = module._residual_haze_rgba(haze, 96, seed=71)
    alpha = rgba[..., 3]
    assert int(alpha.max()) <= module.HYBRID_SMOKE_RESIDUAL_HAZE_MAX_ALPHA
    assert int(alpha.max()) > 4
    assert np.count_nonzero(alpha > 0) > int(alpha.size * 0.05)

    gradient_y, gradient_x = np.gradient(alpha.astype(np.float32) / 255.0)
    active_gradient = np.hypot(gradient_x, gradient_y)[alpha > 0]
    assert active_gradient.size > 0
    assert float(np.percentile(active_gradient, 99.0)) < 0.080


def test_atmospheric_composite_has_visible_smoke_lift() -> None:
    module = load_module()
    base = module.Image.new("RGBA", (24, 24), (34, 37, 38, 255))
    smoke = module.Image.new("RGBA", (24, 24), (214, 218, 214, 104))

    out = np.asarray(module.composite_atmospheric_smoke(base, smoke).convert("RGB"), dtype=np.float32)
    before = np.asarray(base.convert("RGB"), dtype=np.float32)

    assert float(np.mean(out - before)) > 40.0


def test_warp_map_layer_uses_premultiplied_alpha_edges() -> None:
    module = load_module()
    rgba = np.zeros((24, 24, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[7:17, 7:17, :3] = 174
    rgba[7:17, 7:17, 3] = 176
    plate = module.TerrainPlate(
        image=module.Image.new("RGBA", (48, 48), (0, 0, 0, 255)),
        quad=[(0.0, 0.0), (47.0, 0.0), (47.0, 47.0), (0.0, 47.0)],
        fire_xy=(24.0, 24.0),
        fire_uv=(0.5, 0.5),
        texture_size=(24, 24),
    )

    warped = np.asarray(module.warp_map_layer_to_plate(rgba, plate, (48, 48)).convert("RGBA"), dtype=np.int16)
    edge = (warped[..., 3] > 2) & (warped[..., 3] < 150)

    assert int(np.count_nonzero(edge)) > 20
    assert float(np.percentile(warped[..., 0][edge] - warped[..., 2][edge], 95.0)) < 18.0


def test_main_smoke_compositor_keeps_atmospheric_blanket_with_physical_detail() -> None:
    module = load_module()
    atmospheric = np.zeros((42, 64, 4), dtype=np.uint8)
    physical = np.zeros_like(atmospheric)
    atmospheric[10:34, 6:58, :3] = (190, 195, 190)
    atmospheric[10:34, 6:58, 3] = 72
    physical[18:27, 18:44, :3] = (222, 220, 208)
    physical[18:27, 18:44, 3] = 132

    combined = module.composite_main_smoke_maps(atmospheric, physical)
    alpha = combined[..., 3]

    assert combined.shape == atmospheric.shape
    assert int(alpha.max()) <= module.HYBRID_SMOKE_MAX_ALPHA
    assert np.count_nonzero(alpha > 0) > np.count_nonzero(physical[..., 3] > 0) * 3
    assert int(alpha[12, 10]) > 20
    assert int(alpha[22, 28]) > int(physical[22, 28, 3] * 0.80)


def test_hybrid_smoke_frames_are_temporally_coherent_not_redrawn_wisps() -> None:
    module = load_module()
    map_size = (144, 112)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=100, seed=73)
    simulator = module.HybridSmokeSimulator(map_size, sources, seed=73)

    previous = None
    current = None
    for frame in range(66):
        state = simulator.step(frame)
        rgba = module.hybrid_smoke_rgba(state, frame, seed=73)
        if frame == 64:
            previous = rgba[..., 3].astype(np.float32) / 255.0
        if frame == 65:
            current = rgba[..., 3].astype(np.float32) / 255.0

    assert previous is not None
    assert current is not None
    active = (previous > 0.02) | (current > 0.02)
    assert np.count_nonzero(active) > int(previous.size * 0.25)
    prev_active = previous[active]
    curr_active = current[active]
    corr = np.corrcoef(prev_active, curr_active)[0, 1]
    mean_delta = float(np.mean(np.abs(curr_active - prev_active)))
    assert float(corr) > 0.86
    assert mean_delta > 0.0012
    assert mean_delta < 0.050


def test_smoke_composite_keeps_orange_sources_visible_under_veil() -> None:
    module = load_module()
    base = module.Image.new("RGBA", (72, 48), (28, 31, 32, 255))
    draw = module.ImageDraw.Draw(base, "RGBA")
    draw.ellipse((26, 18, 46, 31), fill=(255, 104, 18, 230))
    draw.ellipse((32, 21, 40, 28), fill=(255, 228, 116, 255))
    smoke = module.Image.new("RGBA", (72, 48), (214, 218, 214, 118))

    out = np.asarray(module.composite_atmospheric_smoke(base, smoke).convert("RGB"), dtype=np.int16)
    center = out[21:29, 30:42]
    surrounding = out[8:16, 8:20]

    assert int(center[..., 0].max()) > int(surrounding[..., 0].max()) + 35
    assert float(np.mean(center[..., 0] - center[..., 2])) > 34.0
    assert float(np.mean(center[..., 1] - center[..., 2])) > 18.0


def test_enhance_terrain_texture_grades_to_charcoal() -> None:
    module = load_module()
    texture = module.Image.fromarray(np.full((32, 32, 3), 180, dtype=np.uint8), mode="RGB")
    dem = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)

    charcoal = np.asarray(module.enhance_terrain_texture(texture, dem).convert("RGB"), dtype=np.float32)
    luma = charcoal @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    channel_spread = np.mean(np.abs(charcoal[..., 0] - charcoal[..., 2]))

    assert float(np.mean(luma)) < 65.0
    assert float(np.max(luma)) < 128.0
    assert float(channel_spread) < 12.0


def test_volume_detail_defaults_to_2p5d_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    monkeypatch.setattr(sys, "argv", ["california_cigar_smoke_demo.py"])

    args = module.parse_args()

    assert bool(args.volume_detail) is False
    assert bool(args.physical_smoke) is True
    assert args.physical_smoke_backend == "auto"

    monkeypatch.setattr(sys, "argv", ["california_cigar_smoke_demo.py", "--volume-detail"])
    assert bool(module.parse_args().volume_detail) is True

    monkeypatch.setattr(sys, "argv", ["california_cigar_smoke_demo.py", "--no-physical-smoke"])
    assert bool(module.parse_args().physical_smoke) is False

    monkeypatch.setattr(sys, "argv", ["california_cigar_smoke_demo.py", "--physical-smoke-backend", "numpy"])
    assert module.parse_args().physical_smoke_backend == "numpy"


def test_physical_smoke_main_effect_renders_projected_volume() -> None:
    module = load_module()
    map_size = (72, 54)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=48, seed=41)
    effect = module.make_physical_main_smoke(
        sources,
        map_size,
        dims=(44, 14, 34),
        render_size=(64, 48),
        max_sources=6,
        substeps=1,
        seed=41,
    )
    assert effect is not None
    assert effect.backend == ("native" if module._native_physical_smoke_available() else "numpy")

    for frame in range(10):
        module.step_physical_main_smoke(effect, frame)
    rgba = module.render_physical_main_smoke(effect, 10)
    alpha = rgba[..., 3]
    report = effect.domain.physics_report()

    assert rgba.shape == (54, 72, 4)
    assert rgba.dtype == np.uint8
    assert float(report["mass"]) > 0.0
    assert int(alpha.max()) > 10
    assert int(alpha.max()) <= module.PHYSICAL_SMOKE_MAX_ALPHA
    assert np.count_nonzero(alpha > 0) > 20


def test_python_raymarch_uses_density_color_and_source_glow() -> None:
    module = load_module()
    domain = module.NumpyPhysicalSmokeDomain((18, 9, 16), sparse_threshold=1.0e-6)
    z, y, x = np.mgrid[0:16, 0:9, 0:18].astype(np.float32)
    dense_core = np.exp(-(((x - 8.5) / 4.5) ** 2 + ((z - 7.5) / 3.5) ** 2 + ((y - 4.0) / 2.6) ** 2))
    hot_source = np.exp(-(((x - 5.0) / 1.7) ** 2 + ((z - 6.0) / 1.5) ** 2 + ((y - 3.0) / 1.3) ** 2))
    density = (0.74 * dense_core + 0.32 * hot_source).astype(np.float32)
    temperature = (1.8 * hot_source).astype(np.float32)
    emission = (2.2 * hot_source).astype(np.float32)
    soot = (0.22 * dense_core).astype(np.float32)
    domain.set_density(density)
    domain.set_temperature(temperature)
    domain.set_emission(emission)
    domain.set_soot(soot)
    settings = module.PhysicalSmokeRenderSettings3D(
        density_scale=1.22,
        extinction=1.48,
        exposure=1.24,
        fire_glow=1.15,
        thin_color=(0.45, 0.51, 0.58),
        dense_color=(0.90, 0.88, 0.78),
    )

    with_glow = module._python_projected_volume_raymarch(domain, (72, 54), 7, 91, settings)
    domain.set_temperature(np.zeros_like(temperature))
    domain.set_emission(np.zeros_like(emission))
    without_glow = module._python_projected_volume_raymarch(domain, (72, 54), 7, 91, settings)

    assert with_glow.shape == (54, 72, 4)
    assert int(with_glow[..., 3].max()) > 20
    assert int(np.count_nonzero(with_glow[..., 3] > 0)) > 80
    warm_with = with_glow[..., 0].astype(np.int16) - with_glow[..., 2].astype(np.int16)
    warm_without = without_glow[..., 0].astype(np.int16) - without_glow[..., 2].astype(np.int16)
    warm_delta = warm_with - warm_without
    assert int(warm_delta.max()) > 8
    assert int(np.count_nonzero(warm_delta > 8)) > 20
    assert int(with_glow[..., 0].max()) > int(without_glow[..., 0].max())


def test_python_volume_light_transmittance_self_shadows_density() -> None:
    module = load_module()
    density = np.zeros((18, 10, 22), dtype=np.float32)
    soot = np.zeros_like(density)
    particle_age = np.zeros_like(density)
    density[6:12, 4:8, 7:13] = 1.15
    soot[6:12, 4:8, 7:13] = 0.18
    settings = module.PhysicalSmokeRenderSettings3D(
        density_scale=1.3,
        extinction=1.6,
        shadow_steps=24,
        shadow_step_size=0.7,
    )

    light_trans = module._python_volume_light_transmittance(
        density,
        soot,
        4,
        settings.density_scale,
        settings.extinction,
        settings.soot_absorption,
        settings,
    )

    assert light_trans.shape == (18, 22)
    assert float(light_trans[9, 10]) < 0.45
    assert float(light_trans[2, 2]) > 0.92


def test_python_shadow_grid_self_shadows_full_volume() -> None:
    module = load_module()
    density = np.zeros((18, 10, 22), dtype=np.float32)
    soot = np.zeros_like(density)
    particle_age = np.zeros_like(density)
    density[6:12, 4:8, 7:13] = 1.15
    soot[6:12, 4:8, 7:13] = 0.18
    settings = module.PhysicalSmokeRenderSettings3D(
        density_scale=1.3,
        extinction=1.6,
        shadow_steps=24,
        shadow_step_size=0.7,
    )
    light_dir = np.asarray(module.PHYSICAL_SMOKE_SUN_DIRECTION, dtype=np.float32)
    light_dir /= max(float(np.linalg.norm(light_dir)), 1.0e-6)

    shadow = module._python_volume_shadow_grid(
        density,
        soot,
        particle_age,
        light_dir,
        settings.density_scale,
        settings.extinction,
        settings.soot_absorption,
        settings,
    )
    aged_shadow = module._python_volume_shadow_grid(
        density,
        soot,
        np.full_like(density, 24.0),
        light_dir,
        settings.density_scale,
        settings.extinction,
        settings.soot_absorption,
        settings,
    )

    assert shadow.shape == density.shape
    assert float(shadow[6:12, 4:8, 7:13].min()) < 0.45
    assert float(aged_shadow[6:12, 4:8, 7:13].min()) > float(shadow[6:12, 4:8, 7:13].min())
    assert float(shadow[1, 1, 1]) > 0.92


def test_physical_smoke_uses_streamers_cells_and_history() -> None:
    module = load_module()
    map_size = (72, 54)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=64, seed=47)
    effect = module.make_physical_main_smoke(
        sources,
        map_size,
        dims=(44, 14, 34),
        render_size=(64, 48),
        max_sources=6,
        substeps=1,
        backend="numpy",
        seed=47,
    )
    assert effect is not None

    emitters = module._physical_emitters_for_frame(effect, 12)
    radii = np.asarray([emitter.radius for emitter in emitters], dtype=np.float32)
    assert len(emitters) > len(effect.sources)
    assert float(np.percentile(radii, 20.0)) < float(np.percentile(radii, 80.0)) * 0.62

    for frame in range(16):
        module.step_physical_main_smoke(effect, frame)
    first = module.render_physical_main_smoke(effect, 16)
    assert len(effect.history) == 1
    for frame in range(17, 26):
        module.step_physical_main_smoke(effect, frame)
    second = module.render_physical_main_smoke(effect, 26)

    assert len(effect.history) >= 2
    assert np.count_nonzero(second[..., 3] > 0) >= np.count_nonzero(first[..., 3] > 0) * 0.80
    assert int(second[..., 3].max()) <= module.PHYSICAL_SMOKE_MAX_ALPHA
    active_alpha = second[..., 3][second[..., 3] > 0]
    assert float(np.std(active_alpha)) > 3.0


def test_numpy_physical_smoke_advects_particle_age_with_density() -> None:
    module = load_module()
    domain = module.NumpyPhysicalSmokeDomain((16, 8, 12), sparse_threshold=1.0e-6)
    density = np.zeros((12, 8, 16), dtype=np.float32)
    density[5:8, 3:5, 3:6] = 1.0
    velocity = np.zeros(density.shape + (3,), dtype=np.float32)
    velocity[..., 0] = 1.0
    domain.set_density(density)
    domain.set_velocity(velocity)
    domain.particle_age[...] = np.where(density > 0.0, 8.0, -1.0).astype(np.float32)
    settings = module.PhysicalSmokeStepSettings3D(
        dt=1.0,
        density_decay=0.0,
        temperature_decay=0.0,
        velocity_damping=0.0,
        diffusion=0.0,
        buoyancy=0.0,
        vorticity=0.0,
        pressure_iterations=1,
        turbulence_strength=0.0,
        wind=(0.0, 0.0, 0.0),
    )

    domain.step(settings, [])
    after_density = domain.to_density_numpy()
    after_age = domain.to_particle_age_numpy()
    peak = np.unravel_index(int(np.argmax(after_density)), after_density.shape)

    assert peak[2] > 3
    assert float(after_age[peak]) > 8.5


def test_physical_main_smoke_default_integrates_density_field_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    map_size = (64, 48)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=40, seed=49)
    effect = module.make_physical_main_smoke(
        sources,
        map_size,
        dims=(36, 12, 28),
        render_size=(48, 36),
        max_sources=5,
        substeps=1,
        backend="numpy",
        seed=49,
    )
    assert effect is not None

    observed = {
        "_postprocess_physical_smoke_rgba": 0,
        "_physical_volume_structure_rgba": 0,
        "_physical_structure_enhancement_rgba": 0,
        "_physical_history_rgba": 0,
        "_temporal_blend_physical_smoke": 0,
    }
    for name in observed:
        original = getattr(module, name)

        def wrapped(*args, _name=name, _original=original, **kwargs):
            observed[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapped)

    for frame in range(8):
        module.step_physical_main_smoke(effect, frame)
    rgba = module.render_physical_main_smoke(effect, 8)

    assert rgba.shape == (map_size[1], map_size[0], 4)
    assert int(rgba[..., 3].max()) > 0
    assert all(count >= 1 for count in observed.values())


def test_numpy_physical_solver_vorticity_confinement_adds_curl_force() -> None:
    module = load_module()
    domain = module.NumpyPhysicalSmokeDomain((18, 10, 16), sparse_threshold=1.0e-6)
    z, y, x = np.mgrid[0:16, 0:10, 0:18].astype(np.float32)
    density = np.exp(-(((x - 8.5) / 5.2) ** 2 + ((z - 7.5) / 4.4) ** 2 + ((y - 4.5) / 3.0) ** 2)).astype(np.float32)
    swirl = np.exp(-(((x - 8.5) / 4.0) ** 2 + ((z - 7.5) / 3.4) ** 2 + ((y - 4.5) / 2.6) ** 2)).astype(np.float32)
    velocity = np.zeros(density.shape + (3,), dtype=np.float32)
    velocity[..., 0] = -(z - 7.5) * 0.08 * swirl
    velocity[..., 2] = (x - 8.5) * 0.08 * swirl
    domain.set_density(density)
    domain.set_velocity(velocity)

    before = domain.to_velocity_numpy()
    settings = module.PhysicalSmokeStepSettings3D(
        dt=0.22,
        buoyancy=0.0,
        wind=(0.0, 0.0, 0.0),
        turbulence_strength=0.0,
        velocity_damping=0.0,
        vorticity=1.35,
    )
    domain._apply_forces(settings)
    after = domain.to_velocity_numpy()

    delta = np.linalg.norm(after - before, axis=-1)
    assert float(delta.max()) > 0.0005
    assert int(np.count_nonzero(delta > 0.0001)) > 20


def test_physical_main_smoke_mechanisms_are_active_end_to_end() -> None:
    module = load_module()
    map_size = (84, 64)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=72, seed=57)
    effect = module.make_physical_main_smoke(
        sources,
        map_size,
        dims=(38, 12, 30),
        render_size=(56, 44),
        max_sources=6,
        substeps=1,
        backend="numpy",
        seed=57,
    )
    assert effect is not None

    centroids = []
    for frame in range(18):
        module.step_physical_main_smoke(effect, frame)
        density = effect.domain.to_density_numpy()
        if float(np.sum(density)) > 0.0:
            z, _y, x = np.mgrid[0 : density.shape[0], 0 : density.shape[1], 0 : density.shape[2]].astype(np.float32)
            mass = density.astype(np.float64)
            centroids.append(
                (
                    float(np.sum(x * mass) / np.sum(mass)),
                    float(np.sum(z * mass) / np.sum(mass)),
                )
            )

    report = effect.domain.physics_report()
    assert float(report["mass"]) > 0.0
    assert len(centroids) >= 3
    assert abs(centroids[-1][0] - centroids[0][0]) + abs(centroids[-1][1] - centroids[0][1]) > 0.18

    fields = module._physical_volume_lane_fields(effect, 18)
    assert fields is not None
    assert float(fields["support"].max()) > 0.05
    assert float(fields["lane"].max()) > 0.20
    assert float(fields["curl"].max()) > 0.05
    assert float(fields["shear"].max()) > 0.05

    first = module.render_physical_main_smoke(effect, 18)
    assert len(effect.history) == 1
    for frame in range(19, 29):
        module.step_physical_main_smoke(effect, frame)
    second = module.render_physical_main_smoke(effect, 29)
    assert len(effect.history) >= 2
    assert int(second[..., 3].max()) <= module.PHYSICAL_SMOKE_MAX_ALPHA
    assert np.count_nonzero(second[..., 3] > 0) >= np.count_nonzero(first[..., 3] > 0) * 0.70
    warm_smoke = second[..., 0].astype(np.int16) - second[..., 2].astype(np.int16)
    assert int(warm_smoke.max()) > 30
    assert int(np.count_nonzero((warm_smoke > 10) & (second[..., 3] > 0))) > len(effect.sources)

    fire = module.hybrid_fire_sources_rgba(effect.sources, 18, map_size, glow_only=True, bloom_scale=1.2)
    assert fire.shape == (map_size[1], map_size[0], 4)
    assert int(fire[..., 0].max()) > int(fire[..., 2].max()) + 100
    assert int(np.count_nonzero(fire[..., 3] > 0)) > len(effect.sources)


def test_projected_smoke_depth_cues_create_shadow_and_rim_light() -> None:
    module = load_module()
    alpha = np.zeros((48, 64), dtype=np.float32)
    alpha[16:31, 18:42] = 0.58
    alpha[20:27, 26:34] = 0.92

    shadow, rim, depth = module._projected_smoke_depth_cues(alpha, 12, 2020)

    assert shadow.shape == alpha.shape
    assert rim.shape == alpha.shape
    assert depth.shape == alpha.shape
    assert float(shadow.max()) > 0.12
    assert float(depth.max()) > 0.20
    assert int(np.count_nonzero(rim > 0.01)) > 0


def test_physical_structure_enhancement_adds_visible_ribbons() -> None:
    module = load_module()
    map_size = (80, 60)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=64, seed=51)
    effect = module.make_physical_main_smoke(
        sources,
        map_size,
        dims=(36, 12, 28),
        render_size=(48, 36),
        max_sources=6,
        substeps=1,
        backend="numpy",
        seed=51,
    )
    assert effect is not None

    base_alpha = np.zeros((map_size[1], map_size[0]), dtype=np.float32)
    base_alpha[24:42, 22:58] = 0.16
    ribbons = module._physical_structure_enhancement_rgba(effect, 20, base_alpha)
    ribbon_alpha = ribbons[..., 3]

    assert ribbons.shape == (map_size[1], map_size[0], 4)
    assert int(ribbon_alpha.max()) > 16
    assert int(np.count_nonzero(ribbon_alpha > 0)) > 80
    assert float(np.std(ribbon_alpha[ribbon_alpha > 0])) > 4.0


def test_physical_volume_structure_uses_density_and_velocity_features() -> None:
    module = load_module()
    map_size = (80, 60)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=64, seed=53)
    effect = module.make_physical_main_smoke(
        sources,
        map_size,
        dims=(24, 10, 20),
        render_size=(40, 30),
        max_sources=4,
        substeps=1,
        backend="numpy",
        seed=53,
    )
    assert effect is not None

    z, y, x = np.mgrid[0:20, 0:10, 0:24].astype(np.float32)
    ridge_a = np.exp(-(((x - 7.0) / 3.8) ** 2 + ((z - 7.0) / 2.5) ** 2 + ((y - 4.2) / 2.2) ** 2))
    ridge_b = 0.76 * np.exp(-(((x - 14.0) / 4.8) ** 2 + ((z - 13.0) / 3.2) ** 2 + ((y - 5.4) / 2.6) ** 2))
    density = (ridge_a + ridge_b).astype(np.float32)
    velocity = np.zeros(density.shape + (3,), dtype=np.float32)
    velocity[..., 0] = (z - 10.0) * 0.025
    velocity[..., 2] = -(x - 12.0) * 0.025
    effect.domain.set_density(np.ascontiguousarray(density, dtype=np.float32))
    effect.domain.set_velocity(np.ascontiguousarray(velocity, dtype=np.float32))

    fields = module._physical_volume_lane_fields(effect, 18)
    assert fields is not None
    for key in ("support", "ridges", "lane", "voids", "gain", "curl", "shear"):
        assert fields[key].shape == (map_size[1], map_size[0])
        assert fields[key].dtype == np.float32
    assert float(fields["lane"].max()) > 0.35
    assert float(fields["voids"].max()) > 0.08
    assert float(fields["curl"].max()) > 0.20

    rgba = module._physical_volume_structure_rgba(effect, 18, fields)
    alpha = rgba[..., 3]

    assert rgba.shape == (map_size[1], map_size[0], 4)
    assert int(alpha.max()) > 10
    assert int(np.count_nonzero(alpha > 0)) > 120
    assert float(np.std(alpha[alpha > 0])) > 3.0

    raw = np.zeros((map_size[1], map_size[0], 4), dtype=np.uint8)
    raw[..., :3] = 180
    raw[..., 3] = np.clip(np.round(fields["support"] * 190.0), 0, 190).astype(np.uint8)
    without_fields = module._postprocess_physical_smoke_rgba(raw, 18, 53, map_size, None)
    with_fields = module._postprocess_physical_smoke_rgba(raw, 18, 53, map_size, fields)
    void_mask = (fields["voids"] > 0.08) & (fields["lane"] < 0.42) & (fields["ridges"] < 0.42)
    assert int(np.count_nonzero(void_mask)) > 20
    assert float(np.mean(with_fields[..., 3][void_mask])) < float(np.mean(without_fields[..., 3][void_mask])) * 0.92


def test_physical_smoke_numpy_backend_remains_available() -> None:
    module = load_module()
    map_size = (60, 44)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=36, seed=43)
    effect = module.make_physical_main_smoke(
        sources,
        map_size,
        dims=(36, 12, 28),
        render_size=(48, 36),
        max_sources=5,
        substeps=1,
        backend="numpy",
        seed=43,
    )
    assert effect is not None
    assert effect.backend == "numpy"

    for frame in range(8):
        module.step_physical_main_smoke(effect, frame)
    rgba = module.render_physical_main_smoke(effect, 8)

    assert rgba.shape == (44, 60, 4)
    assert int(rgba[..., 3].max()) > 10


def test_hybrid_lifecycle_fades_old_smoke() -> None:
    module = load_module()
    alpha = module._hybrid_lifecycle_alpha(np.asarray([6.0, 80.0, 205.0], dtype=np.float32))

    assert float(alpha[1]) > float(alpha[0]) * 0.65
    assert float(alpha[2]) < float(alpha[1]) * 0.20


def test_hrrr_smoke_guidance_url_and_raster_conversion() -> None:
    module = load_module()
    url = module.hrrr_smoke_image_url("2020081618", 3, base_url="https://example.test/hrrr")
    assert url.endswith("/for_web/hrrr_ncep_smoke_jet/2020081618/full/trc1_full_int_f003.png")

    image = module.Image.new("RGB", (80, 60), (245, 246, 246))
    draw = module.ImageDraw.Draw(image)
    draw.rectangle((12, 22, 70, 34), fill=(190, 112, 64))
    draw.rectangle((38, 16, 62, 42), fill=(118, 128, 208))

    density = module._hrrr_smoke_image_to_density(image, (40, 30))

    assert density.shape == (30, 40)
    assert density.dtype == np.float32
    assert float(density.max()) > 0.35
    assert np.count_nonzero(density > 0.05) > 20
