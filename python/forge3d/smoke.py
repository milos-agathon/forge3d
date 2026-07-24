"""Physical smoke volumes, simulation stepping, and volume import helpers."""

from __future__ import annotations

import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ._native import get_native_module
from .certificate import _captured_cpu_render

_native = get_native_module()
# PyO3 0.21 ABI3 extension classes are not safe to construct under Python 3.14
# in this repository build; keep smoke on the Python fallback there.
_PY314_NATIVE_SMOKE_UNSAFE = sys.version_info >= (3, 14)
_HAS_NATIVE_SMOKE = bool(
    not _PY314_NATIVE_SMOKE_UNSAFE
    and _native is not None
    and all(
        hasattr(_native, name)
        for name in (
            "SmokeDomain",
            "SmokeEmitter",
            "SmokeStepSettings",
            "SmokeRenderSettings",
        )
    )
)


@dataclass
class AtmosphericSmokeCube:
    """Geospatial or model-derived 3D smoke cube ready for a smoke domain."""

    density: np.ndarray
    velocity: np.ndarray | None = None
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    vertical_levels: tuple[float, ...] = ()
    times: tuple[str, ...] = ()
    crs: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        self.density = _as_density(self.density)
        if self.velocity is not None:
            self.velocity = _as_velocity(self.velocity, self.density.shape)
        self.voxel_size = _tuple3_float(self.voxel_size, "voxel_size")
        self.origin = _tuple3_float(self.origin, "origin")

    def to_domain(self) -> Any:
        domain = domain_from_density(self.density, voxel_size=self.voxel_size, origin=self.origin)
        if self.velocity is not None:
            domain.set_velocity(np.ascontiguousarray(self.velocity, dtype=np.float32))
        return domain


if _HAS_NATIVE_SMOKE:
    SmokeDomain = _native.SmokeDomain
    SmokeEmitter = _native.SmokeEmitter
    SmokeStepSettings = _native.SmokeStepSettings
    SmokeRenderSettings = _native.SmokeRenderSettings
else:

    @dataclass
    class SmokeEmitter:
        center: tuple[float, float, float] = (0.0, 0.0, 0.0)
        radius: float = 1.0
        density_rate: float = 1.0
        temperature_rate: float = 1.0
        fuel_rate: float = 0.0
        soot_rate: float = 0.2
        humidity_rate: float = 0.0
        emission_rate: float = 1.0
        velocity: tuple[float, float, float] = (0.0, 1.0, 0.0)
        start_time: float = 0.0
        end_time: float = np.finfo(np.float32).max

        def __post_init__(self) -> None:
            self.center = _tuple3_float(self.center, "center")
            self.velocity = _tuple3_float(self.velocity, "velocity")
            if self.radius <= 0.0:
                raise ValueError("radius must be > 0")
            if self.end_time < self.start_time:
                raise ValueError("end_time must be >= start_time")

    @dataclass
    class SmokeStepSettings:
        dt: float = 1.0 / 30.0
        density_decay: float = 0.015
        temperature_decay: float = 0.08
        velocity_damping: float = 0.01
        diffusion: float = 0.0005
        buoyancy: float = 0.7
        vorticity: float = 0.12
        pressure_iterations: int = 20
        turbulence_strength: float = 0.0
        turbulence_seed: int = 0
        mac_cormack: bool = False
        mass_conservation: bool = True
        terrain_collision: bool = True
        boundary_damping: float = 0.0
        wind: tuple[float, float, float] = (0.0, 0.0, 0.0)

        def __post_init__(self) -> None:
            if self.dt <= 0.0:
                raise ValueError("dt must be > 0")
            self.wind = _tuple3_float(self.wind, "wind")

    @dataclass
    class SmokeRenderSettings:
        density_scale: float = 1.0
        extinction: float = 2.6
        scattering: float = 0.85
        absorption: float = 0.45
        phase_g: float = 0.24
        step_size: float = 0.0
        max_steps: int = 256
        self_shadow: bool = True
        shadow_steps: int = 20
        shadow_step_size: float = 0.0
        jitter_strength: float = 0.5
        exposure: float = 1.0
        thin_color: tuple[float, float, float] = (0.50, 0.54, 0.58)
        dense_color: tuple[float, float, float] = (0.93, 0.91, 0.82)
        soot_absorption: float = 0.22
        fire_glow: float = 0.35

    class SmokeDomain:
        """NumPy reference fallback mirroring the native SmokeDomain API."""

        def __init__(
            self,
            dims: tuple[int, int, int],
            voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
            origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
            brick_size: tuple[int, int, int] = (16, 16, 16),
            sparse_threshold: float = 1.0e-5,
        ) -> None:
            self.dims = tuple(int(v) for v in dims)
            if len(self.dims) != 3 or min(self.dims) < 2:
                raise ValueError("dims must be (x, y, z), each >= 2")
            self.voxel_size = _tuple3_float(voxel_size, "voxel_size")
            self.origin = _tuple3_float(origin, "origin")
            self.brick_size = tuple(int(v) for v in brick_size)
            self.sparse_threshold = float(sparse_threshold)
            shape = (self.dims[2], self.dims[1], self.dims[0])
            self.density = np.zeros(shape, dtype=np.float32)
            self.velocity = np.zeros(shape + (3,), dtype=np.float32)
            self.temperature = np.zeros(shape, dtype=np.float32)
            self.pressure = np.zeros(shape, dtype=np.float32)
            self.fuel = np.zeros(shape, dtype=np.float32)
            self.soot = np.zeros(shape, dtype=np.float32)
            self.humidity = np.zeros(shape, dtype=np.float32)
            self.particle_age = np.full(shape, -1.0, dtype=np.float32)
            self.emission_rate = np.zeros(shape, dtype=np.float32)
            self.time_seconds = 0.0
            self.frame_index = 0

        @classmethod
        def from_density(
            cls,
            density: np.ndarray,
            voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
            origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        ) -> "SmokeDomain":
            density = _as_density(density)
            domain = cls((density.shape[2], density.shape[1], density.shape[0]), voxel_size, origin)
            domain.set_density(density)
            return domain

        def set_density(self, density: np.ndarray) -> None:
            density = _as_density(density)
            if density.shape != self.density.shape:
                raise ValueError(f"density shape must be {self.density.shape}")
            self.density[...] = density
            self.particle_age[...] = np.where(density > self.sparse_threshold, 0.0, -1.0)

        def set_velocity(self, velocity: np.ndarray) -> None:
            velocity = _as_velocity(velocity, self.density.shape)
            self.velocity[...] = velocity

        def set_temperature(self, temperature: np.ndarray) -> None:
            temperature = _as_density(temperature)
            if temperature.shape != self.temperature.shape:
                raise ValueError(f"temperature shape must be {self.temperature.shape}")
            self.temperature[...] = temperature

        def set_soot(self, soot: np.ndarray) -> None:
            soot = _as_density(soot)
            if soot.shape != self.soot.shape:
                raise ValueError(f"soot shape must be {self.soot.shape}")
            self.soot[...] = soot

        def set_emission(self, emission: np.ndarray) -> None:
            emission = _as_density(emission)
            if emission.shape != self.emission_rate.shape:
                raise ValueError(f"emission shape must be {self.emission_rate.shape}")
            self.emission_rate[...] = emission

        def add_emitter(self, emitter: SmokeEmitter, dt: float) -> None:
            zz, yy, xx = np.indices(self.density.shape, dtype=np.float32)
            wx = self.origin[0] + (xx + 0.5) * self.voxel_size[0]
            wy = self.origin[1] + (yy + 0.5) * self.voxel_size[1]
            wz = self.origin[2] + (zz + 0.5) * self.voxel_size[2]
            dist = np.sqrt(
                (wx - emitter.center[0]) ** 2
                + (wy - emitter.center[1]) ** 2
                + (wz - emitter.center[2]) ** 2
            )
            falloff = np.clip(1.0 - dist / emitter.radius, 0.0, 1.0).astype(np.float32)
            amount = float(dt) * falloff
            self.density += emitter.density_rate * amount
            self.temperature += emitter.temperature_rate * amount
            self.fuel += emitter.fuel_rate * amount
            self.soot += emitter.soot_rate * amount
            self.humidity += emitter.humidity_rate * amount
            self.emission_rate += emitter.emission_rate * falloff
            for i, value in enumerate(emitter.velocity):
                self.velocity[..., i] += value * amount
            self.particle_age = np.where(falloff > 0.0, 0.0, self.particle_age)

        def step(
            self,
            settings: SmokeStepSettings,
            emitters: Sequence[SmokeEmitter] | None = None,
        ) -> None:
            self.emission_rate.fill(0.0)
            for emitter in emitters or ():
                if emitter.start_time <= self.time_seconds <= emitter.end_time:
                    self.add_emitter(emitter, settings.dt)
            self.velocity[..., 0] += settings.wind[0] * settings.dt
            self.velocity[..., 1] += (settings.wind[1] + self.temperature * settings.buoyancy) * settings.dt
            self.velocity[..., 2] += settings.wind[2] * settings.dt
            self.velocity *= np.exp(-settings.velocity_damping * settings.dt)

            zz, yy, xx = np.indices(self.density.shape, dtype=np.float32)
            old_velocity = self.velocity.copy()
            back_x = xx - old_velocity[..., 0] * settings.dt / max(self.voxel_size[0], 1.0e-6)
            back_y = yy - old_velocity[..., 1] * settings.dt / max(self.voxel_size[1], 1.0e-6)
            back_z = zz - old_velocity[..., 2] * settings.dt / max(self.voxel_size[2], 1.0e-6)
            for component in range(3):
                self.velocity[..., component] = _smoke_sample_volume(
                    old_velocity[..., component],
                    back_x,
                    back_y,
                    back_z,
                )

            old_mass = float(self.density.sum())
            advect_velocity = self.velocity.copy()
            advected_age = _smoke_advect_scalar(
                self.particle_age,
                advect_velocity,
                xx,
                yy,
                zz,
                settings.dt,
                self.voxel_size,
                mac_cormack=settings.mac_cormack,
                min_value=-1.0,
            )
            self.density = _smoke_advect_scalar(
                self.density,
                advect_velocity,
                xx,
                yy,
                zz,
                settings.dt,
                self.voxel_size,
                mac_cormack=settings.mac_cormack,
            )
            self.temperature = _smoke_advect_scalar(
                self.temperature,
                advect_velocity,
                xx,
                yy,
                zz,
                settings.dt,
                self.voxel_size,
                mac_cormack=settings.mac_cormack,
            )
            self.fuel = _smoke_advect_scalar(
                self.fuel,
                advect_velocity,
                xx,
                yy,
                zz,
                settings.dt,
                self.voxel_size,
                mac_cormack=settings.mac_cormack,
            )
            self.soot = _smoke_advect_scalar(
                self.soot,
                advect_velocity,
                xx,
                yy,
                zz,
                settings.dt,
                self.voxel_size,
                mac_cormack=settings.mac_cormack,
            )
            self.humidity = _smoke_advect_scalar(
                self.humidity,
                advect_velocity,
                xx,
                yy,
                zz,
                settings.dt,
                self.voxel_size,
                mac_cormack=settings.mac_cormack,
            )
            if settings.diffusion > 0.0:
                mix = float(np.clip(settings.diffusion * settings.dt * 32.0, 0.0, 0.25))
                self.density = _smoke_diffuse_scalar(self.density, mix)
                self.temperature = _smoke_diffuse_scalar(self.temperature, mix * 0.8)
                self.fuel = _smoke_diffuse_scalar(self.fuel, mix * 0.8)
                self.soot = _smoke_diffuse_scalar(self.soot, mix * 0.9)
                self.humidity = _smoke_diffuse_scalar(self.humidity, mix)
            if settings.mass_conservation and old_mass > 0.0:
                new_mass = float(self.density.sum())
                if new_mass > 0.0:
                    self.density *= old_mass / new_mass
            age_decay = _smoke_smoothstep(7.0, 36.0, np.clip(self.particle_age, 0.0, None))
            self.density *= np.exp(-settings.density_decay * settings.dt * (1.0 + 3.0 * age_decay)).astype(np.float32)
            self.temperature *= np.exp(-settings.temperature_decay * settings.dt)
            self.soot *= np.exp(-settings.density_decay * settings.dt * (0.42 + 1.15 * age_decay)).astype(np.float32)
            self.particle_age = np.where(
                self.density > self.sparse_threshold,
                np.maximum(advected_age, 0.0) + settings.dt,
                -1.0,
            )
            self.time_seconds += settings.dt
            self.frame_index += 1

        def to_density_numpy(self) -> np.ndarray:
            return np.array(self.density, copy=True)

        def to_velocity_numpy(self) -> np.ndarray:
            return np.array(self.velocity, copy=True)

        def to_temperature_numpy(self) -> np.ndarray:
            return np.array(self.temperature, copy=True)

        def to_soot_numpy(self) -> np.ndarray:
            return np.array(self.soot, copy=True)

        def to_emission_numpy(self) -> np.ndarray:
            return np.array(self.emission_rate, copy=True)

        def to_particle_age_numpy(self) -> np.ndarray:
            return np.array(self.particle_age, copy=True)

        def sample_density(self, position: tuple[float, float, float]) -> float:
            gx = int(np.clip((position[0] - self.origin[0]) / self.voxel_size[0], 0, self.dims[0] - 1))
            gy = int(np.clip((position[1] - self.origin[1]) / self.voxel_size[1], 0, self.dims[1] - 1))
            gz = int(np.clip((position[2] - self.origin[2]) / self.voxel_size[2], 0, self.dims[2] - 1))
            return float(self.density[gz, gy, gx])

        def memory_report(self) -> dict[str, Any]:
            voxel_count = int(np.prod(self.dims))
            dense_bytes = voxel_count * 10 * 4
            active = int(np.count_nonzero(self.density > self.sparse_threshold))
            return {
                "voxel_count": voxel_count,
                "dense_bytes": dense_bytes,
                "active_bricks": int(active > 0),
                "total_bricks": 1,
                "sparse_bytes_estimate": min(dense_bytes, active * 10 * 4),
                "utilization": float(active > 0),
                "time_seconds": self.time_seconds,
                "frame_index": self.frame_index,
            }

        def physics_report(self) -> dict[str, Any]:
            return {
                "mass": float(self.density.sum()),
                "max_density": float(self.density.max(initial=0.0)),
                "divergence_l2": float(np.sqrt(np.mean(self.velocity**2))),
                "time_seconds": self.time_seconds,
                "frame_index": self.frame_index,
            }

        def render_rgba(
            self,
            width: int,
            height: int,
            camera_pos: tuple[float, float, float],
            target: tuple[float, float, float],
            up: tuple[float, float, float] = (0.0, 1.0, 0.0),
            fovy_deg: float = 45.0,
            sun_direction: tuple[float, float, float] = (0.4, 0.8, -0.2),
            settings: SmokeRenderSettings | None = None,
            certificate: bool | str | Path = False,
            cache: str | Path | None = None,
        ) -> np.ndarray:
            del camera_pos, target, up, fovy_deg
            settings = settings or SmokeRenderSettings()
            return self.render_projection_rgba(
                width,
                height,
                view_direction=(0.0, -1.0, 0.0),
                sun_direction=sun_direction,
                settings=settings,
                certificate=certificate,
                cache=cache,
            )

        @_captured_cpu_render(
            "python.smoke.render_projection_rgba", "smoke.cpu_projection", draw_calls=1
        )
        def render_projection_rgba(
            self,
            width: int,
            height: int,
            view_direction: tuple[float, float, float] = (0.0, -1.0, 0.0),
            sun_direction: tuple[float, float, float] = (0.4, 0.8, -0.2),
            settings: SmokeRenderSettings | None = None,
            certificate: bool | str | Path = False,
            cache: str | Path | None = None,
        ) -> np.ndarray:
            _ = cache
            from . import _degradation

            _degradation.record(
                "cpu_fallback",
                "smoke.render",
                "native smoke bindings are unavailable; NumPy CPU raymarching was used",
            )
            settings = settings or SmokeRenderSettings()
            density = np.clip(self.density, 0.0, None).astype(np.float32, copy=False)
            soot_volume = np.clip(self.soot, 0.0, None).astype(np.float32, copy=False)
            view_dir = np.asarray(_tuple3_float(view_direction, "view_direction"), dtype=np.float32)
            view_dir /= max(float(np.linalg.norm(view_dir)), 1.0e-6)
            sun_dir = np.asarray(_tuple3_float(sun_direction, "sun_direction"), dtype=np.float32)
            sun_dir /= max(float(np.linalg.norm(sun_dir)), 1.0e-6)
            phase = _smoke_henyey_greenstein(float(np.dot(view_dir, sun_dir)), settings.phase_g)
            depth, altitude_count, width_in = density.shape
            out_w = max(1, int(width))
            out_h = max(1, int(height))
            pixel_x, pixel_z = np.meshgrid(
                np.linspace(0.0, max(float(width_in - 1), 0.0), out_w, dtype=np.float32),
                np.linspace(0.0, max(float(depth - 1), 0.0), out_h, dtype=np.float32),
            )
            step_size = float(settings.step_size)
            if step_size <= 0.0:
                diagonal = math.sqrt(width_in * width_in + altitude_count * altitude_count + depth * depth)
                step_size = max(diagonal / max(float(settings.max_steps), 1.0), 0.35)
            max_steps = int(max(1, settings.max_steps))
            transmittance = np.ones((out_h, out_w), dtype=np.float32)
            rgb_accum = np.zeros((out_h, out_w, 3), dtype=np.float32)
            thin = np.asarray(settings.thin_color, dtype=np.float32)
            dense = np.asarray(settings.dense_color, dtype=np.float32)
            scatter_albedo = float(np.clip(settings.scattering / max(settings.scattering + settings.absorption, 1.0e-5), 0.02, 0.98))
            plane_y = np.full_like(pixel_x, (altitude_count - 1) * 0.5, dtype=np.float32)
            diagonal = math.sqrt(max(float(width_in - 1), 1.0) ** 2 + max(float(altitude_count - 1), 1.0) ** 2 + max(float(depth - 1), 1.0) ** 2)
            origin_x = (pixel_x - float(view_dir[0]) * (diagonal + step_size)).astype(np.float32)
            origin_y = (plane_y - float(view_dir[1]) * (diagonal + step_size)).astype(np.float32)
            origin_z = (pixel_z - float(view_dir[2]) * (diagonal + step_size)).astype(np.float32)
            t_enter, t_exit, valid = _smoke_ray_box_intersection(
                origin_x,
                origin_y,
                origin_z,
                view_dir,
                (max(float(width_in - 1), 0.0), max(float(altitude_count - 1), 0.0), max(float(depth - 1), 0.0)),
            )
            jitter = np.sin(pixel_x * 12.9898 + pixel_z * 78.233 + float(self.frame_index) * 37.719)
            jitter = (jitter * 43758.5453 - np.floor(jitter * 43758.5453)).astype(np.float32)
            t_current = (np.maximum(t_enter, 0.0) + jitter * float(np.clip(settings.jitter_strength, 0.0, 1.0)) * step_size).astype(np.float32)
            shadow_volume = _smoke_shadow_grid(density, soot_volume, self.particle_age, sun_dir, settings) if settings.self_shadow else None
            sun_radiance = np.array([1.0, 0.96, 0.84], dtype=np.float32) * 11.5
            sky_base = np.array([0.42, 0.50, 0.62], dtype=np.float32)
            bounce_base = np.array([0.54, 0.50, 0.44], dtype=np.float32)
            old_blue = np.array([0.40, 0.43, 0.47], dtype=np.float32)
            pale_core = np.array([0.64, 0.67, 0.67], dtype=np.float32)
            aged_blue = np.array([0.34, 0.37, 0.41], dtype=np.float32)
            glow_color = np.array([3.40, 0.62, 0.03], dtype=np.float32)
            rgb_flat = rgb_accum.reshape(-1, 3)
            trans_flat = transmittance.reshape(-1)
            t_flat = t_current.reshape(-1)
            for _step_index in range(max_steps):
                active = valid & (t_current <= t_exit) & (transmittance > 0.008)
                if not bool(np.any(active)):
                    break
                active_index = np.flatnonzero(active)
                t = t_current[active]
                segment_length = np.minimum(step_size, t_exit[active] - t).astype(np.float32)
                segment_length = np.clip(segment_length, 0.0, step_size)
                sample_x = origin_x[active] + float(view_dir[0]) * t
                sample_y = origin_y[active] + float(view_dir[1]) * t
                sample_z = origin_z[active] + float(view_dir[2]) * t
                layer = _smoke_sample_volume(density, sample_x, sample_y, sample_z)
                occupied = (layer > 1.0e-5) & (segment_length > 1.0e-5)
                t_flat[active_index] = t + step_size
                if not bool(np.any(occupied)):
                    continue
                occupied_index = active_index[occupied]
                sample_x = sample_x[occupied]
                sample_y = sample_y[occupied]
                sample_z = sample_z[occupied]
                segment_length = segment_length[occupied]
                layer = layer[occupied]
                soot_layer = _smoke_sample_volume(soot_volume, sample_x, sample_y, sample_z)
                age_layer = _smoke_sample_volume(self.particle_age, sample_x, sample_y, sample_z)
                age_t = _smoke_smoothstep(1.6, 17.0, age_layer)
                altitude = np.clip(sample_y / max(float(altitude_count - 1), 1.0), 0.0, 1.0).astype(np.float32)
                sigma = np.clip(
                    layer
                    * settings.density_scale
                    * (1.0 + 0.26 * altitude)
                    * (1.0 + soot_layer * settings.soot_absorption * 0.85),
                    0.0,
                    8.0,
                )
                concentration_gate = 0.50 + 0.50 * _smoke_smoothstep(0.045, 0.34, layer)
                sigma *= 1.0 - 0.58 * age_t
                sigma *= concentration_gate
                sigma_t = sigma * settings.extinction
                segment_transmittance = np.exp(-sigma_t * segment_length).astype(np.float32)
                segment_weight = np.divide(
                    1.0 - segment_transmittance,
                    sigma_t,
                    out=segment_length.copy(),
                    where=sigma_t > 1.0e-6,
                )
                dense_t = _smoke_smoothstep(0.05, 1.40, sigma)
                pale = thin * (1.0 - dense_t[..., None]) + pale_core * dense_t[..., None]
                layer_rgb = old_blue * (1.0 - dense_t[..., None]) + pale * dense_t[..., None]
                core = (
                    _smoke_smoothstep(0.42, 1.57, sigma)
                    * (1.0 - 0.24 * np.clip(soot_layer, 0.0, 1.0))
                    * (1.0 - 0.68 * age_t)
                )
                layer_rgb = layer_rgb * (1.0 - core[..., None]) + dense * core[..., None]
                layer_rgb = layer_rgb * (1.0 - 0.42 * age_t[..., None]) + aged_blue * (0.42 * age_t[..., None])
                if shadow_volume is not None:
                    light_trans = _smoke_sample_volume(shadow_volume, sample_x, sample_y, sample_z)
                else:
                    light_trans = np.ones_like(layer, dtype=np.float32)
                sigma_s = sigma_t * scatter_albedo
                sky_radiance = sky_base * (
                    0.20 + 0.16 * (1.0 - light_trans)
                )[..., None]
                ground_bounce = bounce_base * (0.045 * (1.0 - altitude))[..., None]
                direct = layer_rgb * sigma_s[..., None] * sun_radiance * (phase * light_trans[..., None])
                multiple = layer_rgb * sigma_s[..., None] * (sky_radiance + ground_bounce)
                temp = np.clip(_smoke_sample_volume(self.temperature, sample_x, sample_y, sample_z), 0.0, None)
                emit = np.clip(_smoke_sample_volume(self.emission_rate, sample_x, sample_y, sample_z), 0.0, None)
                fresh_heat = temp * (1.0 - age_t) * (1.0 - age_t)
                glow_strength = np.clip((fresh_heat * 0.10 + emit * 1.18) * settings.fire_glow, 0.0, 5.0)
                glow = glow_color * glow_strength[..., None]
                rgb_flat[occupied_index] += (direct + multiple + glow) * segment_weight[..., None] * trans_flat[occupied_index][..., None]
                trans_flat[occupied_index] *= segment_transmittance
            alpha = np.clip(1.0 - transmittance, 0.0, 1.0)
            rgb = np.divide(rgb_accum, alpha[..., None], out=np.zeros_like(rgb_accum), where=alpha[..., None] > 1.0e-6)
            rgb = rgb / (1.0 + rgb)
            out = np.empty((out_h, out_w, 4), dtype=np.uint8)
            out[..., :3] = np.clip(rgb * settings.exposure * 255.0, 0, 255).astype(np.uint8)
            out[..., 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
            return out


def native_smoke_available() -> bool:
    return _HAS_NATIVE_SMOKE


def domain_from_density(
    density: np.ndarray,
    *,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Any:
    density = _as_density(density)
    return SmokeDomain.from_density(density, voxel_size=voxel_size, origin=origin)


def cube_from_arrays(
    density: np.ndarray,
    *,
    velocity: np.ndarray | None = None,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertical_levels: Sequence[float] = (),
    times: Sequence[str] = (),
    crs: str | None = None,
    source: str | None = None,
) -> AtmosphericSmokeCube:
    return AtmosphericSmokeCube(
        density=density,
        velocity=velocity,
        voxel_size=voxel_size,
        origin=origin,
        vertical_levels=tuple(float(v) for v in vertical_levels),
        times=tuple(str(v) for v in times),
        crs=crs,
        source=source,
    )


def load_npz_volume(
    path: str | Path,
    *,
    density_key: str = "density",
    velocity_key: str = "velocity",
) -> AtmosphericSmokeCube:
    data = np.load(path, allow_pickle=False)
    density = data[density_key]
    velocity = data[velocity_key] if velocity_key in data.files else None
    voxel_size = tuple(float(v) for v in data["voxel_size"]) if "voxel_size" in data.files else (1.0, 1.0, 1.0)
    origin = tuple(float(v) for v in data["origin"]) if "origin" in data.files else (0.0, 0.0, 0.0)
    levels = tuple(float(v) for v in data["vertical_levels"]) if "vertical_levels" in data.files else ()
    return cube_from_arrays(
        density,
        velocity=velocity,
        voxel_size=voxel_size,  # type: ignore[arg-type]
        origin=origin,  # type: ignore[arg-type]
        vertical_levels=levels,
        source=str(path),
    )


def save_npz_volume(path: str | Path, domain_or_cube: Any) -> None:
    if isinstance(domain_or_cube, AtmosphericSmokeCube):
        density = domain_or_cube.density
        velocity = domain_or_cube.velocity
        voxel_size = domain_or_cube.voxel_size
        origin = domain_or_cube.origin
        levels = np.asarray(domain_or_cube.vertical_levels, dtype=np.float32)
    else:
        density = np.asarray(domain_or_cube.to_density_numpy(), dtype=np.float32)
        velocity = np.asarray(domain_or_cube.to_velocity_numpy(), dtype=np.float32)
        voxel_size = getattr(domain_or_cube, "voxel_size", (1.0, 1.0, 1.0))
        origin = getattr(domain_or_cube, "origin", (0.0, 0.0, 0.0))
        levels = np.asarray((), dtype=np.float32)
    kwargs: dict[str, Any] = {
        "density": density,
        "voxel_size": np.asarray(voxel_size, dtype=np.float32),
        "origin": np.asarray(origin, dtype=np.float32),
        "vertical_levels": levels,
    }
    if velocity is not None:
        kwargs["velocity"] = velocity
    np.savez_compressed(path, **kwargs)


def load_xarray_volume(
    source: Any,
    *,
    density_var: str = "density",
    wind_vars: tuple[str, str, str] | None = ("u", "v", "w"),
    time_index: int | None = 0,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> AtmosphericSmokeCube:
    """Load a 3D smoke cube from an xarray Dataset or NetCDF/GRIB path.

    GRIB support depends on the user's xarray/cfgrib installation; forge3d keeps
    this dependency optional and validates the resulting arrays at the boundary.
    """

    xr = __import__("xarray")
    ds = xr.open_dataset(source) if isinstance(source, (str, Path)) else source
    density = _extract_zyx(ds[density_var], time_index=time_index)
    velocity = None
    if wind_vars is not None and all(name in ds for name in wind_vars):
        velocity = np.stack(
            [_extract_zyx(ds[name], time_index=time_index) for name in wind_vars],
            axis=-1,
        )
    levels = _levels_from_dataset(ds)
    return cube_from_arrays(
        density,
        velocity=velocity,
        voxel_size=voxel_size,
        origin=origin,
        vertical_levels=levels,
        source=str(source) if isinstance(source, (str, Path)) else None,
    )


def interpolate_density_frames(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    a = _as_density(a)
    b = _as_density(b)
    if a.shape != b.shape:
        raise ValueError("density frames must have matching shapes")
    return np.ascontiguousarray(a * (1.0 - alpha) + b * alpha, dtype=np.float32)


def capability_report() -> dict[str, Any]:
    return {
        "native_backend": _HAS_NATIVE_SMOKE,
        "representation": [
            "density",
            "velocity",
            "temperature",
            "pressure",
            "fuel",
            "soot",
            "humidity",
            "particle_age",
            "emission_rate",
            "sparse_brick_memory_report",
        ],
        "simulation": [
            "semi_lagrangian_advection",
            "maccormack_scalar_correction",
            "buoyancy",
            "vorticity_confinement",
            "pressure_projection",
            "diffusion_decay",
            "emitters",
            "terrain_floor_boundary",
            "deterministic_turbulence",
        ],
        "importers": ["numpy", "npz", "xarray/netcdf/grib_optional"],
        "renderer": [
            "beer_lambert_extinction",
            "single_scattering",
            "henyey_greenstein_phase",
            "self_shadowing",
            "blue_noise_style_jitter",
            "projected_3d_raymarch_rgba",
            "soot_and_temperature_color",
            "transparent_rgba_export",
        ],
    }


def _smoke_henyey_greenstein(cos_theta: float, g: float) -> float:
    g = float(np.clip(g, -0.99, 0.99))
    denom = max(1.0 + g * g - 2.0 * g * float(np.clip(cos_theta, -1.0, 1.0)), 1.0e-4)
    return float((1.0 - g * g) / (4.0 * math.pi * denom**1.5))


def _smoke_smoothstep(edge0: float, edge1: float, value: np.ndarray | float) -> np.ndarray:
    denom = max(float(edge1) - float(edge0), 1.0e-6)
    t = np.clip((np.asarray(value, dtype=np.float32) - float(edge0)) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smoke_sample_volume(field: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    depth, height, width = src.shape
    valid = (x >= 0.0) & (x <= width - 1.0) & (y >= 0.0) & (y <= height - 1.0) & (z >= 0.0) & (z <= depth - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y1, 0, height - 1)
    z0c = np.clip(z0, 0, depth - 1)
    z1c = np.clip(z1, 0, depth - 1)
    wx = x - x0.astype(np.float32)
    wy = y - y0.astype(np.float32)
    wz = z - z0.astype(np.float32)
    c000 = src[z0c, y0c, x0c]
    c100 = src[z0c, y0c, x1c]
    c010 = src[z0c, y1c, x0c]
    c110 = src[z0c, y1c, x1c]
    c001 = src[z1c, y0c, x0c]
    c101 = src[z1c, y0c, x1c]
    c011 = src[z1c, y1c, x0c]
    c111 = src[z1c, y1c, x1c]
    c00 = c000 * (1.0 - wx) + c100 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx
    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    return np.where(valid, c0 * (1.0 - wz) + c1 * wz, 0.0).astype(np.float32)


def _smoke_advect_scalar(
    field: np.ndarray,
    velocity: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    dt: float,
    voxel_size: tuple[float, float, float],
    *,
    mac_cormack: bool = False,
    min_value: float = 0.0,
) -> np.ndarray:
    back_x = grid_x - velocity[..., 0] * float(dt) / max(float(voxel_size[0]), 1.0e-6)
    back_y = grid_y - velocity[..., 1] * float(dt) / max(float(voxel_size[1]), 1.0e-6)
    back_z = grid_z - velocity[..., 2] * float(dt) / max(float(voxel_size[2]), 1.0e-6)
    predicted = _smoke_sample_volume(field, back_x, back_y, back_z)
    if not mac_cormack:
        return np.clip(predicted, float(min_value), None).astype(np.float32)

    back_vx = _smoke_sample_volume(velocity[..., 0], back_x, back_y, back_z)
    back_vy = _smoke_sample_volume(velocity[..., 1], back_x, back_y, back_z)
    back_vz = _smoke_sample_volume(velocity[..., 2], back_x, back_y, back_z)
    fwd_x = back_x + back_vx * float(dt) / max(float(voxel_size[0]), 1.0e-6)
    fwd_y = back_y + back_vy * float(dt) / max(float(voxel_size[1]), 1.0e-6)
    fwd_z = back_z + back_vz * float(dt) / max(float(voxel_size[2]), 1.0e-6)
    recovered = _smoke_sample_volume(predicted, fwd_x, fwd_y, fwd_z)
    candidate = predicted + (np.asarray(field, dtype=np.float32) - recovered) * np.float32(0.5)
    lo, hi = _smoke_local_min_max_volume(field, back_x, back_y, back_z)
    corrected = np.clip(candidate, lo, hi)
    return np.clip(corrected, float(min_value), None).astype(np.float32)


def _smoke_local_min_max_volume(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(field, dtype=np.float32)
    depth, height, width = src.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y1, 0, height - 1)
    z0c = np.clip(z0, 0, depth - 1)
    z1c = np.clip(z1, 0, depth - 1)
    corners = (
        src[z0c, y0c, x0c],
        src[z0c, y0c, x1c],
        src[z0c, y1c, x0c],
        src[z0c, y1c, x1c],
        src[z1c, y0c, x0c],
        src[z1c, y0c, x1c],
        src[z1c, y1c, x0c],
        src[z1c, y1c, x1c],
    )
    return np.minimum.reduce(corners).astype(np.float32), np.maximum.reduce(corners).astype(np.float32)


def _smoke_light_transmittance(
    density: np.ndarray,
    soot: np.ndarray,
    layer_index: int,
    sun_dir: np.ndarray,
    settings: Any,
) -> np.ndarray:
    depth, _altitude_count, width = density.shape
    z_grid, x_grid = np.mgrid[0:depth, 0:width].astype(np.float32)
    y_grid = np.full((depth, width), float(layer_index), dtype=np.float32)
    step_size = float(getattr(settings, "shadow_step_size", 0.0))
    if step_size <= 0.0:
        step_size = 1.0
    steps = int(max(1, getattr(settings, "shadow_steps", 20)))
    optical_depth = np.zeros((depth, width), dtype=np.float32)
    for step_index in range(1, steps + 1):
        distance = float(step_index) * step_size
        sample_x = x_grid + float(sun_dir[0]) * distance
        sample_y = y_grid + float(sun_dir[1]) * distance
        sample_z = z_grid + float(sun_dir[2]) * distance
        sample_density = _smoke_sample_volume(density, sample_x, sample_y, sample_z)
        sample_soot = _smoke_sample_volume(soot, sample_x, sample_y, sample_z)
        optical_depth += (
            sample_density
            * float(settings.density_scale)
            * float(settings.extinction)
            * (1.0 + sample_soot * float(settings.soot_absorption) * 0.85)
            * step_size
        ).astype(np.float32)
    return np.exp(-np.clip(optical_depth, 0.0, 9.0)).astype(np.float32)


def _smoke_shadow_grid(
    density: np.ndarray,
    soot: np.ndarray,
    particle_age: np.ndarray | None,
    sun_dir: np.ndarray,
    settings: Any,
) -> np.ndarray:
    depth, altitude_count, width = density.shape
    z_grid, y_grid, x_grid = np.mgrid[0:depth, 0:altitude_count, 0:width].astype(np.float32)
    step_size = float(getattr(settings, "shadow_step_size", 0.0))
    if step_size <= 0.0:
        step_size = 1.0
    steps = int(max(1, getattr(settings, "shadow_steps", 20)))
    optical_depth = np.zeros_like(density, dtype=np.float32)
    for step_index in range(1, steps + 1):
        distance = float(step_index) * step_size
        sample_x = x_grid + float(sun_dir[0]) * distance
        sample_y = y_grid + float(sun_dir[1]) * distance
        sample_z = z_grid + float(sun_dir[2]) * distance
        sample_density = _smoke_sample_volume(density, sample_x, sample_y, sample_z)
        sample_soot = _smoke_sample_volume(soot, sample_x, sample_y, sample_z)
        if particle_age is not None:
            sample_age = _smoke_sample_volume(particle_age, sample_x, sample_y, sample_z)
            age_t = _smoke_smoothstep(1.6, 17.0, sample_age)
        else:
            age_t = np.zeros_like(sample_density, dtype=np.float32)
        concentration_gate = 0.50 + 0.50 * _smoke_smoothstep(0.045, 0.34, sample_density)
        optical_depth += (
            sample_density
            * float(settings.density_scale)
            * (1.0 - 0.58 * age_t)
            * concentration_gate
            * float(settings.extinction)
            * (1.0 + sample_soot * float(settings.soot_absorption) * 0.85)
            * step_size
        ).astype(np.float32)
    return np.exp(-np.clip(optical_depth, 0.0, 9.0)).astype(np.float32)


def _smoke_ray_box_intersection(
    origin_x: np.ndarray,
    origin_y: np.ndarray,
    origin_z: np.ndarray,
    ray_dir: np.ndarray,
    bounds_max: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def axis_interval(origin: np.ndarray, direction: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
        if abs(direction) < 1.0e-6:
            inside = (origin >= 0.0) & (origin <= upper)
            near = np.where(inside, -np.inf, 1.0).astype(np.float32)
            far = np.where(inside, np.inf, 0.0).astype(np.float32)
            return near, far
        t0 = (0.0 - origin) / direction
        t1 = (upper - origin) / direction
        return np.minimum(t0, t1).astype(np.float32), np.maximum(t0, t1).astype(np.float32)

    x_near, x_far = axis_interval(origin_x, float(ray_dir[0]), bounds_max[0])
    y_near, y_far = axis_interval(origin_y, float(ray_dir[1]), bounds_max[1])
    z_near, z_far = axis_interval(origin_z, float(ray_dir[2]), bounds_max[2])
    t_enter = np.maximum(np.maximum(x_near, y_near), z_near).astype(np.float32)
    t_exit = np.minimum(np.minimum(x_far, y_far), z_far).astype(np.float32)
    valid = t_exit >= np.maximum(t_enter, 0.0)
    return t_enter, t_exit, valid


def _smoke_diffuse_scalar(field: np.ndarray, mix: float) -> np.ndarray:
    amount = float(np.clip(mix, 0.0, 0.5))
    if amount <= 0.0:
        return field.astype(np.float32, copy=True)
    padded = np.pad(np.asarray(field, dtype=np.float32), 1, mode="edge")
    neighbors = (
        padded[1:-1, 1:-1, :-2]
        + padded[1:-1, 1:-1, 2:]
        + padded[1:-1, :-2, 1:-1]
        + padded[1:-1, 2:, 1:-1]
        + padded[:-2, 1:-1, 1:-1]
        + padded[2:, 1:-1, 1:-1]
    ) / 6.0
    return np.clip(np.asarray(field, dtype=np.float32) * (1.0 - amount) + neighbors * amount, 0.0, None).astype(np.float32)


def _as_density(value: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(value, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("density must be a 3D array shaped (z, y, x)")
    if not np.isfinite(arr).all():
        raise ValueError("density contains non-finite values")
    return arr


def _as_velocity(value: np.ndarray, density_shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.ascontiguousarray(value, dtype=np.float32)
    if arr.shape != density_shape + (3,):
        raise ValueError(f"velocity must be shaped {density_shape + (3,)}")
    if not np.isfinite(arr).all():
        raise ValueError("velocity contains non-finite values")
    return arr


def _tuple3_float(value: Iterable[float], name: str) -> tuple[float, float, float]:
    items = tuple(float(v) for v in value)
    if len(items) != 3:
        raise ValueError(f"{name} must contain 3 values")
    if not np.isfinite(np.asarray(items, dtype=np.float32)).all():
        raise ValueError(f"{name} values must be finite")
    return items  # type: ignore[return-value]


def _extract_zyx(data_array: Any, *, time_index: int | None) -> np.ndarray:
    arr = data_array
    if time_index is not None:
        for dim in getattr(arr, "dims", ()):
            if str(dim).lower() in {"time", "valid_time"}:
                arr = arr.isel({dim: time_index})
                break
    squeeze_dims = [dim for dim, size in zip(arr.dims, arr.shape) if size == 1]
    if squeeze_dims:
        arr = arr.squeeze(squeeze_dims)
    preferred = _preferred_zyx_order(tuple(str(dim) for dim in arr.dims))
    if preferred is not None:
        arr = arr.transpose(*preferred)
    return _as_density(np.asarray(arr.values, dtype=np.float32))


def _preferred_zyx_order(dims: Sequence[str]) -> tuple[str, str, str] | None:
    z_names = ("z", "level", "lev", "height", "altitude", "isobaricinhpa")
    y_names = ("y", "lat", "latitude")
    x_names = ("x", "lon", "longitude")
    by_lower = {str(dim).lower(): str(dim) for dim in dims}
    z = next((by_lower[name] for name in z_names if name in by_lower), None)
    y = next((by_lower[name] for name in y_names if name in by_lower), None)
    x = next((by_lower[name] for name in x_names if name in by_lower), None)
    if z and y and x:
        return (z, y, x)
    return None


def _levels_from_dataset(ds: Any) -> tuple[float, ...]:
    for name in ("z", "level", "lev", "height", "altitude", "isobaricInhPa"):
        if name in ds.coords:
            values = np.asarray(ds.coords[name].values, dtype=np.float32).ravel()
            return tuple(float(v) for v in values)
    return ()


__all__ = [
    "AtmosphericSmokeCube",
    "SmokeDomain",
    "SmokeEmitter",
    "SmokeRenderSettings",
    "SmokeStepSettings",
    "capability_report",
    "cube_from_arrays",
    "domain_from_density",
    "interpolate_density_frames",
    "load_npz_volume",
    "load_xarray_volume",
    "native_smoke_available",
    "save_npz_volume",
]
