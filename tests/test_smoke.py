from __future__ import annotations

import numpy as np

from forge3d import smoke


def _seed_density() -> np.ndarray:
    density = np.zeros((12, 10, 14), dtype=np.float32)
    density[5:8, 4:7, 3:6] = 1.0
    return density


def test_smoke_domain_from_density_exports_required_fields() -> None:
    domain = smoke.domain_from_density(_seed_density(), voxel_size=(2.0, 3.0, 4.0))

    report = domain.memory_report()
    density = np.asarray(domain.to_density_numpy())
    velocity = np.asarray(domain.to_velocity_numpy())

    assert density.shape == (12, 10, 14)
    assert velocity.shape == (12, 10, 14, 3)
    assert report["voxel_count"] == 12 * 10 * 14
    assert report["dense_bytes"] >= report["voxel_count"] * 10 * 4
    assert report["active_bricks"] >= 1
    assert domain.sample_density((8.0, 15.0, 24.0)) > 0.0


def test_smoke_step_advects_with_velocity_and_preserves_mass() -> None:
    domain = smoke.domain_from_density(_seed_density())
    velocity = np.zeros((12, 10, 14, 3), dtype=np.float32)
    velocity[..., 0] = 1.0
    domain.set_velocity(velocity)

    before = np.asarray(domain.to_density_numpy())
    before_mass = float(before.sum())
    before_x = _center_of_mass_x(before)
    settings = smoke.SmokeStepSettings(
        dt=1.0,
        density_decay=0.0,
        temperature_decay=0.0,
        buoyancy=0.0,
        vorticity=0.0,
        diffusion=0.0,
        pressure_iterations=1,
    )
    domain.step(settings)
    after = np.asarray(domain.to_density_numpy())

    assert _center_of_mass_x(after) > before_x
    assert abs(float(after.sum()) - before_mass) / before_mass < 0.05


def test_smoke_emitter_and_render_rgba_are_nonblank() -> None:
    domain = smoke.SmokeDomain((16, 16, 16))
    emitter = smoke.SmokeEmitter(
        center=(8.0, 6.0, 8.0),
        radius=3.0,
        density_rate=3.0,
        temperature_rate=2.0,
    )
    domain.add_emitter(emitter, 1.0)
    domain.step(smoke.SmokeStepSettings(dt=0.1), [])

    rgba = np.asarray(
        domain.render_rgba(
            32,
            24,
            camera_pos=(8.0, 8.0, -18.0),
            target=(8.0, 8.0, 8.0),
        )
    )

    assert rgba.shape == (24, 32, 4)
    assert rgba.dtype == np.uint8
    assert int(rgba[..., 3].max()) > 0


def test_smoke_npz_roundtrip_preserves_metadata(tmp_path) -> None:
    density = _seed_density()
    velocity = np.zeros(density.shape + (3,), dtype=np.float32)
    velocity[..., 0] = 2.0
    cube = smoke.cube_from_arrays(
        density,
        velocity=velocity,
        voxel_size=(10.0, 20.0, 100.0),
        origin=(-120.0, 30.0, 0.0),
        vertical_levels=(0.0, 100.0, 300.0),
        source="unit-test",
    )
    path = tmp_path / "smoke_volume.npz"
    smoke.save_npz_volume(path, cube)

    loaded = smoke.load_npz_volume(path)

    assert loaded.density.shape == density.shape
    assert loaded.velocity is not None
    assert loaded.velocity.shape == velocity.shape
    assert loaded.voxel_size == (10.0, 20.0, 100.0)
    assert loaded.origin == (-120.0, 30.0, 0.0)
    assert loaded.vertical_levels == (0.0, 100.0, 300.0)


def test_smoke_capability_report_documents_requirement_surface() -> None:
    report = smoke.capability_report()

    assert "density" in report["representation"]
    assert "velocity" in report["representation"]
    assert "pressure" in report["representation"]
    assert "pressure_projection" in report["simulation"]
    assert "xarray/netcdf/grib_optional" in report["importers"]
    assert "beer_lambert_extinction" in report["renderer"]
    assert "henyey_greenstein_phase" in report["renderer"]
    assert "projected_3d_raymarch_rgba" in report["renderer"]


def _center_of_mass_x(density: np.ndarray) -> float:
    zz, yy, xx = np.indices(density.shape, dtype=np.float64)
    del zz, yy
    mass = float(density.sum())
    assert mass > 0.0
    return float((xx * density).sum() / mass)
