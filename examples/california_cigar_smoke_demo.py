#!/usr/bin/env python3
"""August Complex hybrid smoke demo using forge3d.smoke.

The terrain and fire event come from the cached California wildfire example
assets. The main smoke pass is a persistent 3D density field with advective
flow, diffusion, decay, self-shadowed scattering, source emission, and
projected volume ray marching.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if PYTHON_DIR.exists() and str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

try:
    import forge3d.smoke as f3d_smoke
except ModuleNotFoundError:
    f3d_smoke = None

CACHE = ROOT / "examples" / ".cache" / "california_wildfire_smoke"
DEM_PATH = CACHE / "california_osm_r165475_terrarium_z8_max900.tif"
OVERLAY_PATH = CACHE / "california_osm_r165475_terrarium_z8_max900_dark_relief_overlay.png"
META_PATH = CACHE / "california_osm_r165475_terrarium_z8_max900.json"
OUT_DIR = ROOT / "examples" / "out" / "california_cigar_smoke"
DEFAULT_OUTPUT = OUT_DIR / "august_complex_cigar_smoke_8s.mp4"
DEFAULT_PREVIEW = OUT_DIR / "august_complex_cigar_smoke_8s.preview.png"

WEB_MERCATOR_LIMIT = 20037508.342789244
FPS = 30
DURATION_SECONDS = 8
WIDTH = 960
HEIGHT = 540
HYBRID_SMOKE_WIDTH = 520
HYBRID_SMOKE_HEIGHT = 408
HYBRID_SMOKE_MAX_AGE_FRAMES = 306.0
HYBRID_SMOKE_MAX_ALPHA = 168
HYBRID_SMOKE_SEED = 2020
HYBRID_SMOKE_LAYER_COUNT = 3
HYBRID_SMOKE_LAYER_WEIGHTS = (0.56, 0.30, 0.14)
HYBRID_SMOKE_RENDER_LAYER_ALPHA = (0.74, 0.58, 0.44)
HYBRID_SMOKE_RESIDUAL_HAZE_MAX_ALPHA = 42
PHYSICAL_SMOKE_DIMS = (84, 22, 66)
PHYSICAL_SMOKE_RENDER_SIZE = (284, 216)
PHYSICAL_SMOKE_MAX_ALPHA = 154
PHYSICAL_SMOKE_MAX_SOURCES = 12
PHYSICAL_SMOKE_HISTORY_STRIDE = 6
PHYSICAL_SMOKE_HISTORY_MAX_AGE_FRAMES = 156
PHYSICAL_SMOKE_HISTORY_MAX_LAYERS = 12
PHYSICAL_SMOKE_HISTORY_ALPHA_SCALE = 0.30
PHYSICAL_SMOKE_STRUCTURE_MAX_ALPHA = 110
PHYSICAL_SMOKE_VOLUME_STRUCTURE_MAX_ALPHA = 86
PHYSICAL_SMOKE_VIEW_DIRECTION = (0.42, -0.68, 0.60)
PHYSICAL_SMOKE_PARALLAX_SCALE = 1.55
PHYSICAL_SMOKE_SUN_DIRECTION = (0.34, 0.82, -0.22)
HRRR_SMOKE_BASE_URL = "https://rapidrefresh.noaa.gov/hrrr/HRRRsmoke"
HRRR_SMOKE_OLD_BASE_URL = "https://rapidrefresh.noaa.gov/hrrr/HRRRsmokeold"
HRRR_SMOKE_DATASET_KEY = "hrrr_ncep_smoke_jet"
HRRR_SMOKE_BASE_URLS = (HRRR_SMOKE_BASE_URL, HRRR_SMOKE_OLD_BASE_URL)
HRRR_SMOKE_RAW_BASE_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
HRRR_SMOKE_RAW_FIELD = "COLMD"
HRRR_SMOKE_RUNTIME = "2026060318"
HRRR_SMOKE_PLOT_TYPE = "trc1_full_int"
HRRR_SMOKE_FORECAST_HOURS = tuple(range(0, 19))
HRRR_SMOKE_GUIDANCE_STRENGTH = 0.22
HRRR_SMOKE_PANEL_CROP_FRAC = (0.014, 0.128, 0.994, 0.870)
HRRR_SMOKE_CA_SUBSET_FRAC = (0.055, 0.300, 0.295, 0.690)
_SMOKE_TEXTURE_CACHE: dict[tuple[tuple[int, int], int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_PIXEL_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


@dataclass(frozen=True)
class FireEvent:
    name: str
    lon: float
    lat: float
    date: str
    final_area_ha: float
    source: str


@dataclass(frozen=True)
class TerrainPlate:
    image: Image.Image
    quad: list[tuple[float, float]]
    fire_xy: tuple[float, float]
    fire_uv: tuple[float, float]
    texture_size: tuple[int, int]


@dataclass(frozen=True)
class HybridSmokeSource:
    x: float
    y: float
    strength: float
    radius_px: float
    start_frame: int
    end_frame: int
    seed: int
    burst_period_frames: float
    burst_phase_frames: float
    burst_duty: float
    heat: float = 1.0
    smoke_rate: float = 1.0
    altitude_bias: float = 0.0


@dataclass
class HybridSmokeState:
    density: np.ndarray
    age_mass: np.ndarray
    layer_density: tuple[np.ndarray, ...] | None = None
    layer_age_mass: tuple[np.ndarray, ...] | None = None
    residual_haze: np.ndarray | None = None


@dataclass(frozen=True)
class HrrrSmokeGuidance:
    frames: tuple[np.ndarray, ...]
    runtime: str
    plot_type: str
    source_label: str


@dataclass
class PhysicalSmokeMainEffect:
    domain: Any
    step_settings: Any
    render_settings: Any
    sources: list[HybridSmokeSource]
    map_size: tuple[int, int]
    dims: tuple[int, int, int]
    render_size: tuple[int, int]
    substeps: int
    backend: str
    seed: int = HYBRID_SMOKE_SEED
    history: list[tuple[int, np.ndarray]] = field(default_factory=list)
    previous_render_frame: int | None = None
    previous_render_rgba: np.ndarray | None = None


@dataclass(frozen=True)
class PhysicalSmokeEmitter3D:
    center: tuple[float, float, float]
    radius: float
    density_rate: float
    temperature_rate: float
    soot_rate: float
    humidity_rate: float
    emission_rate: float
    velocity: tuple[float, float, float]


@dataclass(frozen=True)
class PhysicalSmokeStepSettings3D:
    dt: float = 0.18
    density_decay: float = 0.010
    temperature_decay: float = 0.18
    velocity_damping: float = 0.046
    diffusion: float = 0.00125
    buoyancy: float = 0.018
    vorticity: float = 0.30
    pressure_iterations: int = 14
    turbulence_strength: float = 0.54
    turbulence_seed: int = HYBRID_SMOKE_SEED
    terrain_collision: bool = True
    boundary_damping: float = 0.035
    wind: tuple[float, float, float] = (0.115, 0.0, -0.044)


@dataclass(frozen=True)
class PhysicalSmokeRenderSettings3D:
    density_scale: float = 1.08
    extinction: float = 1.32
    soot_absorption: float = 0.22
    exposure: float = 1.12
    scattering: float = 1.00
    absorption: float = 0.26
    phase_g: float = 0.48
    step_size: float = 0.72
    max_steps: int = 128
    self_shadow: bool = True
    shadow_steps: int = 18
    shadow_step_size: float = 1.15
    jitter_strength: float = 0.35
    thin_color: tuple[float, float, float] = (0.50, 0.54, 0.58)
    dense_color: tuple[float, float, float] = (0.93, 0.91, 0.82)
    fire_glow: float = 0.38


class NumpyPhysicalSmokeDomain:
    def __init__(
        self,
        dims: tuple[int, int, int],
        sparse_threshold: float = 1.0e-6,
    ) -> None:
        nx, ny, nz = (max(2, int(v)) for v in dims)
        self.dims = (nx, ny, nz)
        self.sparse_threshold = float(sparse_threshold)
        shape = (nz, ny, nx)
        self.density = np.zeros(shape, dtype=np.float32)
        self.velocity = np.zeros(shape + (3,), dtype=np.float32)
        self.temperature = np.zeros(shape, dtype=np.float32)
        self.soot = np.zeros(shape, dtype=np.float32)
        self.humidity = np.zeros(shape, dtype=np.float32)
        self.particle_age = np.full(shape, -1.0, dtype=np.float32)
        self.emission_rate = np.zeros(shape, dtype=np.float32)
        self.pressure = np.zeros(shape, dtype=np.float32)
        self.time_seconds = 0.0
        self.frame_index = 0
        self._grid_x, self._grid_y, self._grid_z = _volume_grids(shape)

    def add_emitter(self, emitter: PhysicalSmokeEmitter3D, dt: float) -> None:
        nx, ny, nz = self.dims
        cx, cy, cz = emitter.center
        radius = max(float(emitter.radius), 1.0e-4)
        x0 = max(0, int(math.floor(cx - radius * 2.2)))
        x1 = min(nx, int(math.ceil(cx + radius * 2.2)))
        y0 = max(0, int(math.floor(cy - radius * 1.8)))
        y1 = min(ny, int(math.ceil(cy + radius * 2.4)))
        z0 = max(0, int(math.floor(cz - radius * 2.2)))
        z1 = min(nz, int(math.ceil(cz + radius * 2.2)))
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            return
        zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1].astype(np.float32)
        dx = xx - np.float32(cx)
        dy = (yy - np.float32(cy)) * 1.18
        dz = zz - np.float32(cz)
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        falloff = np.clip(1.0 - _smoothstep(0.0, radius, dist), 0.0, 1.0).astype(np.float32)
        if not np.any(falloff > 0.0):
            return
        amount = float(dt) * falloff
        target = np.s_[z0:z1, y0:y1, x0:x1]
        self.density[target] += emitter.density_rate * amount
        self.temperature[target] += emitter.temperature_rate * amount
        self.soot[target] += emitter.soot_rate * amount
        self.humidity[target] += emitter.humidity_rate * amount
        self.emission_rate[target] += emitter.emission_rate * falloff
        for component, value in enumerate(emitter.velocity):
            self.velocity[target + (component,)] += float(value) * amount
        self.particle_age[target] = np.where(falloff > 0.0, 0.0, self.particle_age[target])

    def step(self, settings: PhysicalSmokeStepSettings3D, emitters: list[PhysicalSmokeEmitter3D]) -> None:
        dt = float(settings.dt)
        self.emission_rate.fill(0.0)
        for emitter in emitters:
            self.add_emitter(emitter, dt)

        self._apply_forces(settings)
        velocity_before = self.velocity.copy()
        for component in range(3):
            self.velocity[..., component] = _advect_volume_scalar(
                velocity_before[..., component],
                velocity_before,
                self._grid_x,
                self._grid_y,
                self._grid_z,
                dt,
            )
        self._project_velocity(settings.pressure_iterations)
        self._apply_boundaries(settings)
        self._apply_lane_advection_shear(settings)

        advect_velocity = self.velocity.copy()
        advected_age = _advect_volume_scalar(
            self.particle_age,
            advect_velocity,
            self._grid_x,
            self._grid_y,
            self._grid_z,
            dt,
            mac_cormack=True,
            min_value=-1.0,
        )
        self.density = _advect_volume_scalar(
            self.density,
            advect_velocity,
            self._grid_x,
            self._grid_y,
            self._grid_z,
            dt,
            mac_cormack=True,
        )
        self.temperature = _advect_volume_scalar(
            self.temperature,
            advect_velocity,
            self._grid_x,
            self._grid_y,
            self._grid_z,
            dt,
            mac_cormack=True,
        )
        self.soot = _advect_volume_scalar(
            self.soot,
            advect_velocity,
            self._grid_x,
            self._grid_y,
            self._grid_z,
            dt,
            mac_cormack=True,
        )
        self.humidity = _advect_volume_scalar(
            self.humidity,
            advect_velocity,
            self._grid_x,
            self._grid_y,
            self._grid_z,
            dt,
            mac_cormack=True,
        )
        self.particle_age = np.where(
            self.density > self.sparse_threshold,
            np.maximum(advected_age, 0.0),
            -1.0,
        ).astype(np.float32)
        self._apply_subgrid_density_eddies(settings)
        self._diffuse_and_decay(settings)
        self._project_velocity(max(1, settings.pressure_iterations // 2))
        self._apply_boundaries(settings)
        self.time_seconds += dt
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

    def set_density(self, density: np.ndarray) -> None:
        arr = np.asarray(density, dtype=np.float32)
        if arr.shape != self.density.shape:
            raise ValueError(f"density shape must be {self.density.shape}, got {arr.shape}")
        self.density = np.ascontiguousarray(np.clip(arr, 0.0, None), dtype=np.float32)

    def set_velocity(self, velocity: np.ndarray) -> None:
        arr = np.asarray(velocity, dtype=np.float32)
        if arr.shape != self.velocity.shape:
            raise ValueError(f"velocity shape must be {self.velocity.shape}, got {arr.shape}")
        self.velocity = np.ascontiguousarray(arr, dtype=np.float32)

    def set_temperature(self, temperature: np.ndarray) -> None:
        arr = np.asarray(temperature, dtype=np.float32)
        if arr.shape != self.temperature.shape:
            raise ValueError(f"temperature shape must be {self.temperature.shape}, got {arr.shape}")
        self.temperature = np.ascontiguousarray(np.clip(arr, 0.0, None), dtype=np.float32)

    def set_soot(self, soot: np.ndarray) -> None:
        arr = np.asarray(soot, dtype=np.float32)
        if arr.shape != self.soot.shape:
            raise ValueError(f"soot shape must be {self.soot.shape}, got {arr.shape}")
        self.soot = np.ascontiguousarray(np.clip(arr, 0.0, None), dtype=np.float32)

    def set_emission(self, emission: np.ndarray) -> None:
        arr = np.asarray(emission, dtype=np.float32)
        if arr.shape != self.emission_rate.shape:
            raise ValueError(f"emission shape must be {self.emission_rate.shape}, got {arr.shape}")
        self.emission_rate = np.ascontiguousarray(np.clip(arr, 0.0, None), dtype=np.float32)

    def physics_report(self) -> dict[str, float]:
        return {
            "mass": float(np.sum(self.density)),
            "max_density": float(np.max(self.density, initial=0.0)),
            "divergence_l2": float(np.sqrt(np.mean(_volume_divergence(self.velocity) ** 2))),
            "time_seconds": float(self.time_seconds),
            "frame_index": float(self.frame_index),
        }

    def _apply_forces(self, settings: PhysicalSmokeStepSettings3D) -> None:
        dt = float(settings.dt)
        wind = np.asarray(settings.wind, dtype=np.float32)
        self.velocity[..., 0] += wind[0] * dt
        self.velocity[..., 1] += (wind[1] + self.temperature * settings.buoyancy) * dt
        self.velocity[..., 2] += wind[2] * dt
        if settings.turbulence_strength > 0.0:
            texture = _advected_smoke_texture(
                (self.density.shape[0], self.density.shape[2]),
                int(self.frame_index),
                int(settings.turbulence_seed) + 9401,
            )
            grad_z, grad_x = np.gradient(texture)
            amp = settings.turbulence_strength * dt
            altitude = np.linspace(0.55, 1.15, self.density.shape[1], dtype=np.float32)[None, :, None]
            self.velocity[..., 0] += grad_z[:, None, :] * amp * 3.4 * altitude
            self.velocity[..., 2] -= grad_x[:, None, :] * amp * 3.4 * altitude
            self.velocity[..., 1] += (texture[:, None, :] - 0.5) * amp * 0.08
            wind_len = max(float(math.hypot(wind[0], wind[2])), 1.0e-6)
            wind_x = float(wind[0]) / wind_len
            wind_z = float(wind[2]) / wind_len
            cross_x = -wind_z
            cross_z = wind_x
            x = self._grid_x.astype(np.float32)
            z = self._grid_z.astype(np.float32)
            along = x * wind_x + z * wind_z
            cross_coord = x * cross_x + z * cross_z
            lane_phase = along * 0.34 + cross_coord * 0.72 + float(self.frame_index) * 0.075
            lane_phase += float(settings.turbulence_seed) * 0.0031
            lane = np.sin(lane_phase).astype(np.float32)
            lane += 0.45 * np.sin(lane_phase * 0.53 + z * 0.29).astype(np.float32)
            lane *= _smoothstep(self.sparse_threshold, max(self.sparse_threshold * 80.0, 0.018), self.density)
            self.velocity[..., 0] += cross_x * lane * amp * 0.82 * altitude
            self.velocity[..., 2] += cross_z * lane * amp * 0.82 * altitude
            speed_lanes = (0.5 + 0.5 * np.cos(lane_phase * 0.41 + x * 0.18)).astype(np.float32)
            self.velocity[..., 0] += wind_x * speed_lanes * amp * 0.30 * altitude
            self.velocity[..., 2] += wind_z * speed_lanes * amp * 0.30 * altitude
            altitude_coord = np.linspace(0.0, 1.0, self.density.shape[1], dtype=np.float32)[None, :, None]
            shear = (altitude_coord - 0.42) * amp * 1.35
            self.velocity[..., 0] += cross_x * shear
            self.velocity[..., 2] += cross_z * shear
        self._apply_vorticity_confinement(settings)
        damping = math.exp(-settings.velocity_damping * dt)
        self.velocity *= np.float32(damping)

    def _apply_lane_advection_shear(self, settings: PhysicalSmokeStepSettings3D) -> None:
        if settings.turbulence_strength <= 0.0 or not np.any(self.density > self.sparse_threshold):
            return
        wind = np.asarray(settings.wind, dtype=np.float32)
        wind_len = max(float(math.hypot(wind[0], wind[2])), 1.0e-6)
        wind_x = float(wind[0]) / wind_len
        wind_z = float(wind[2]) / wind_len
        cross_x = -wind_z
        cross_z = wind_x
        x = self._grid_x.astype(np.float32)
        y = self._grid_y.astype(np.float32)
        z = self._grid_z.astype(np.float32)
        along = x * wind_x + z * wind_z
        cross_coord = x * cross_x + z * cross_z
        amp = np.float32(settings.turbulence_strength * settings.dt)
        active = _smoothstep(self.sparse_threshold, max(self.sparse_threshold * 60.0, 0.012), self.density)
        lane_phase = along * 0.23 + cross_coord * 0.49 + float(self.frame_index) * 0.105
        lane_phase += float(settings.turbulence_seed) * 0.0027
        lane_force = np.sin(lane_phase).astype(np.float32)
        lane_force += 0.58 * np.sin(lane_phase * 0.41 + y * 0.74).astype(np.float32)
        altitude = y / max(float(self.density.shape[1] - 1), 1.0)
        altitude_shear = (altitude - 0.44) * 0.95
        force = (lane_force * 2.75 + altitude_shear * 1.45) * active * amp
        self.velocity[..., 0] += cross_x * force
        self.velocity[..., 2] += cross_z * force
        slab_phase = along * 0.17 - cross_coord * 0.31 + y * 1.12 + float(self.frame_index) * 0.043
        slab_phase += float(settings.turbulence_seed) * 0.0021
        slab_lane = np.sin(slab_phase).astype(np.float32)
        slab_lane += 0.42 * np.sin(slab_phase * 0.53 + along * 0.09).astype(np.float32)
        slab_split = ((altitude - 0.50) * 2.55 + slab_lane * 0.58) * active * amp
        self.velocity[..., 0] += cross_x * slab_split * 1.90
        self.velocity[..., 2] += cross_z * slab_split * 1.90
        speed_split = np.sin(slab_phase * 0.39 + y * 0.67).astype(np.float32) * active * amp
        self.velocity[..., 0] += wind_x * speed_split * 0.52
        self.velocity[..., 2] += wind_z * speed_split * 0.52
        mass = np.clip(self.density, 0.0, None)
        total = float(np.sum(mass))
        if total <= 1.0e-6:
            return
        cx = float(np.sum(x * mass) / total)
        cz = float(np.sum(z * mass) / total)
        altitude_gain = 0.55 + 0.75 * altitude
        for eddy_index, (distance, radius, side, strength) in enumerate(
            (
                (5.5, 5.4, 1.0, 1.85),
                (11.5, 7.6, -1.0, 1.58),
                (19.0, 10.2, 1.0, 1.30),
                (28.0, 13.0, -1.0, 1.05),
            )
        ):
            phase = float(self.frame_index) * (0.035 + eddy_index * 0.006) + float(settings.turbulence_seed) * 0.0013
            center_x = cx + wind_x * distance + cross_x * side * radius * (0.40 + 0.20 * math.sin(phase))
            center_z = cz + wind_z * distance + cross_z * side * radius * (0.40 + 0.20 * math.cos(phase))
            dx = x - np.float32(center_x)
            dz = z - np.float32(center_z)
            r2 = dx * dx + dz * dz
            envelope = np.exp(-r2 / np.float32(2.0 * radius * radius)).astype(np.float32) * active
            inv_r = 1.0 / np.sqrt(r2 + np.float32(1.0))
            spin = side * strength * amp * envelope * altitude_gain
            self.velocity[..., 0] += (-dz * inv_r) * spin
            self.velocity[..., 2] += (dx * inv_r) * spin

    def _apply_subgrid_density_eddies(self, settings: PhysicalSmokeStepSettings3D) -> None:
        if settings.turbulence_strength <= 0.0 or not np.any(self.density > self.sparse_threshold):
            return
        texture = _advected_smoke_texture(
            (self.density.shape[0], self.density.shape[2]),
            int(self.frame_index),
            int(settings.turbulence_seed) + 17417,
        )
        broad = _pil_blur_float(texture, 5.0)
        x = self._grid_x.astype(np.float32)
        y = self._grid_y.astype(np.float32)
        z = self._grid_z.astype(np.float32)
        phase = x * 0.18 + z * 0.27 + y * 0.72 + broad[:, None, :] * 3.4 + float(self.frame_index) * 0.046
        phase += float(settings.turbulence_seed) * 0.0019
        ribbons = 0.5 + 0.5 * np.sin(phase).astype(np.float32)
        sheets = 0.5 + 0.5 * np.sin(phase * 0.47 - z * 0.16 + y * 0.51).astype(np.float32)
        active = _smoothstep(self.sparse_threshold, max(self.sparse_threshold * 90.0, 0.018), self.density)
        voids = _smoothstep(0.45, 0.84, 1.0 - ribbons) * _smoothstep(0.34, 0.76, 1.0 - sheets) * active
        ridges = _smoothstep(0.62, 0.94, ribbons) * _smoothstep(0.48, 0.90, sheets) * active
        age_t = _smoothstep(2.0, 28.0, np.clip(self.particle_age, 0.0, None))
        void_strength = 0.62 + 0.32 * age_t
        ridge_strength = 0.075 - 0.045 * age_t
        gain = 1.0 - void_strength * voids + ridge_strength * ridges
        wind = np.asarray(settings.wind, dtype=np.float32)
        wind_len = max(float(math.hypot(wind[0], wind[2])), 1.0e-6)
        wind_x = float(wind[0]) / wind_len
        wind_z = float(wind[2]) / wind_len
        cross_x = -wind_z
        cross_z = wind_x
        along = x * wind_x + z * wind_z
        cross_coord = x * cross_x + z * cross_z
        channel_phase = (
            along * 0.115
            + cross_coord * 0.52
            + broad[:, None, :] * 5.4
            + np.sin(y * 0.62 + along * 0.035) * 0.85
            + float(self.frame_index) * 0.033
            + float(settings.turbulence_seed) * 0.0023
        )
        channel_wave = 0.5 + 0.5 * np.sin(channel_phase).astype(np.float32)
        channel_wave += 0.28 * np.sin(channel_phase * 0.47 - cross_coord * 0.19 + y * 0.34).astype(np.float32)
        entrainment = _smoothstep(0.58, 1.06, channel_wave)
        lateral_slots = _smoothstep(0.50, 0.94, 1.0 - (0.62 * ribbons + 0.38 * sheets))
        core_protect = 1.0 - 0.56 * _smoothstep(0.72, 1.75, self.density)
        aged_sheet = (0.28 + 0.72 * age_t) * active * core_protect
        clear_air = np.clip(entrainment * (0.54 + 0.46 * lateral_slots) * aged_sheet, 0.0, 1.0)
        channel_void = np.clip(
            _smoothstep(0.42, 0.86, 1.0 - channel_wave)
            * (0.55 + 0.45 * lateral_slots)
            * active
            * (0.42 + 0.58 * age_t)
            * core_protect,
            0.0,
            1.0,
        )
        gain *= 1.0 - (0.024 + 0.055 * age_t) * clear_air
        gain *= 1.0 - (0.045 + 0.070 * age_t) * channel_void
        self.density = np.clip(self.density * gain.astype(np.float32), 0.0, 8.0).astype(np.float32)
        self.humidity = np.clip(
            self.humidity
            * (
                1.0
                - (0.15 + 0.10 * age_t) * voids
                - (0.024 + 0.055 * age_t) * clear_air
                - (0.045 + 0.070 * age_t) * channel_void
            ),
            0.0,
            None,
        ).astype(np.float32)

    def _apply_vorticity_confinement(self, settings: PhysicalSmokeStepSettings3D) -> None:
        strength = float(settings.vorticity)
        if strength <= 0.0 or not np.any(self.density > self.sparse_threshold):
            return
        u = self.velocity[..., 0]
        v = self.velocity[..., 1]
        w = self.velocity[..., 2]
        du_dz, du_dy, du_dx = np.gradient(u)
        dv_dz, dv_dy, dv_dx = np.gradient(v)
        dw_dz, dw_dy, dw_dx = np.gradient(w)
        omega_x = dw_dy - dv_dz
        omega_y = du_dz - dw_dx
        omega_z = dv_dx - du_dy
        omega_mag = np.sqrt(omega_x * omega_x + omega_y * omega_y + omega_z * omega_z)
        if not np.any(omega_mag > 1.0e-7):
            return
        dmag_dz, dmag_dy, dmag_dx = np.gradient(omega_mag)
        norm = np.sqrt(dmag_dx * dmag_dx + dmag_dy * dmag_dy + dmag_dz * dmag_dz) + 1.0e-6
        nx = dmag_dx / norm
        ny = dmag_dy / norm
        nz = dmag_dz / norm
        force_x = ny * omega_z - nz * omega_y
        force_y = nz * omega_x - nx * omega_z
        force_z = nx * omega_y - ny * omega_x
        density_gate = _smoothstep(self.sparse_threshold, max(self.sparse_threshold * 40.0, 0.02), self.density)
        amp = np.float32(strength * settings.dt * 0.36)
        self.velocity[..., 0] += force_x.astype(np.float32) * amp * density_gate
        self.velocity[..., 1] += force_y.astype(np.float32) * amp * density_gate
        self.velocity[..., 2] += force_z.astype(np.float32) * amp * density_gate

    def _diffuse_and_decay(self, settings: PhysicalSmokeStepSettings3D) -> None:
        mix = float(np.clip(settings.diffusion * settings.dt * 64.0, 0.0, 0.28))
        if mix > 0.0:
            self.density = _diffuse_volume(self.density, mix)
            self.temperature = _diffuse_volume(self.temperature, mix * 0.8)
            self.soot = _diffuse_volume(self.soot, mix * 0.9)
            self.humidity = _diffuse_volume(self.humidity, mix)
        age_decay = _smoothstep(7.0, 36.0, np.clip(self.particle_age, 0.0, None))
        self.density *= np.exp(-settings.density_decay * settings.dt * (1.0 + 3.0 * age_decay)).astype(np.float32)
        self.temperature *= np.float32(math.exp(-settings.temperature_decay * settings.dt))
        self.soot *= np.exp(-settings.density_decay * settings.dt * (0.42 + 1.15 * age_decay)).astype(np.float32)
        active = self.density > self.sparse_threshold
        self.particle_age = np.where(active, np.maximum(self.particle_age, 0.0) + settings.dt, -1.0).astype(np.float32)
        self.density = np.clip(self.density, 0.0, 8.0).astype(np.float32)

    def _project_velocity(self, iterations: int) -> None:
        divergence = _volume_divergence(self.velocity)
        pressure = np.zeros_like(divergence, dtype=np.float32)
        for _ in range(max(1, int(iterations))):
            padded = np.pad(pressure, 1, mode="edge")
            pressure = (
                padded[1:-1, 1:-1, :-2]
                + padded[1:-1, 1:-1, 2:]
                + padded[1:-1, :-2, 1:-1]
                + padded[1:-1, 2:, 1:-1]
                + padded[:-2, 1:-1, 1:-1]
                + padded[2:, 1:-1, 1:-1]
                - divergence
            ) / 6.0
        grad_z, grad_y, grad_x = np.gradient(pressure)
        self.velocity[..., 0] -= grad_x.astype(np.float32)
        self.velocity[..., 1] -= grad_y.astype(np.float32)
        self.velocity[..., 2] -= grad_z.astype(np.float32)
        self.pressure = pressure

    def _apply_boundaries(self, settings: PhysicalSmokeStepSettings3D) -> None:
        self.velocity[:, :, 0, 0] = 0.0
        self.velocity[:, :, -1, 0] = 0.0
        self.velocity[:, 0, :, 1] = 0.0
        self.velocity[:, -1, :, 1] = 0.0
        self.velocity[0, :, :, 2] = 0.0
        self.velocity[-1, :, :, 2] = 0.0
        self.density[:, :, 0] *= 0.58
        self.density[:, :, -1] *= 0.58
        self.density[0, :, :] *= 0.58
        self.density[-1, :, :] *= 0.58
        if self.density.shape[2] > 3:
            self.density[:, :, 1] *= 0.78
            self.density[:, :, -2] *= 0.78
        if self.density.shape[0] > 3:
            self.density[1, :, :] *= 0.78
            self.density[-2, :, :] *= 0.78
        if settings.terrain_collision:
            keep = 1.0 - settings.boundary_damping
            self.density[:, 0, :] *= keep
            self.temperature[:, 0, :] *= keep


AUGUST_COMPLEX = FireEvent(
    name="August Complex",
    lon=-122.9,
    lat=39.7,
    date="2020-08-16",
    final_area_ha=417_898.0,
    source="Reference metadata: examples/california_wildfire_smoke_video.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an 8-second August Complex hybrid smoke demo.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--size", type=int, nargs=2, default=(WIDTH, HEIGHT), metavar=("W", "H"))
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--duration", type=float, default=DURATION_SECONDS)
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--hybrid-smoke-width", type=int, default=HYBRID_SMOKE_WIDTH)
    parser.add_argument("--hybrid-smoke-height", type=int, default=HYBRID_SMOKE_HEIGHT)
    parser.add_argument("--smoke-width", type=int, default=360)
    parser.add_argument("--smoke-height", type=int, default=210)
    parser.add_argument("--physical-smoke-dims", type=int, nargs=3, default=PHYSICAL_SMOKE_DIMS, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--physical-render-size",
        type=int,
        nargs=2,
        default=PHYSICAL_SMOKE_RENDER_SIZE,
        metavar=("W", "H"),
    )
    parser.add_argument("--physical-max-sources", type=int, default=PHYSICAL_SMOKE_MAX_SOURCES)
    parser.add_argument("--physical-substeps", type=int, default=1)
    parser.add_argument(
        "--physical-smoke-backend",
        choices=("auto", "native", "numpy"),
        default="auto",
        help="3D smoke backend for the main effect; auto prefers native forge3d.smoke.",
    )
    parser.add_argument("--physical-smoke", action="store_true", dest="physical_smoke")
    parser.add_argument("--no-physical-smoke", action="store_false", dest="physical_smoke")
    parser.add_argument("--hrrr-smoke-dir", type=Path, default=CACHE / "hrrr_smoke")
    parser.add_argument("--hrrr-runtime", default=HRRR_SMOKE_RUNTIME)
    parser.add_argument("--hrrr-plot-type", default=HRRR_SMOKE_PLOT_TYPE)
    parser.add_argument("--hrrr-base-url", default=HRRR_SMOKE_BASE_URL)
    parser.add_argument("--fetch-hrrr-smoke", action="store_true")
    parser.add_argument("--volume-detail", action="store_true", dest="volume_detail")
    parser.add_argument("--no-volume-detail", action="store_false", dest="volume_detail")
    parser.set_defaults(volume_detail=False, physical_smoke=True)
    return parser.parse_args()


def lonlat_to_web_mercator(lon: float, lat: float) -> tuple[float, float]:
    clipped_lat = float(np.clip(lat, -85.05112878, 85.05112878))
    x = WEB_MERCATOR_LIMIT * lon / 180.0
    y = WEB_MERCATOR_LIMIT * math.log(math.tan((90.0 + clipped_lat) * math.pi / 360.0)) / math.pi
    return x, y


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def fire_pixel(bounds: tuple[float, float, float, float], image_size: tuple[int, int]) -> tuple[float, float]:
    west, south, east, north = bounds
    mx, my = lonlat_to_web_mercator(AUGUST_COMPLEX.lon, AUGUST_COMPLEX.lat)
    width, height = image_size
    x = (mx - west) / max(east - west, 1e-6) * (width - 1)
    y = (north - my) / max(north - south, 1e-6) * (height - 1)
    return x, y


def crop_fire_extent() -> tuple[Image.Image, np.ndarray, tuple[float, float]]:
    if not (DEM_PATH.exists() and OVERLAY_PATH.exists() and META_PATH.exists()):
        raise RuntimeError(
            "Cached California terrain assets are missing. Run "
            "examples/california_wildfire_smoke_video.py once to populate examples/.cache."
        )
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    bounds = tuple(float(v) for v in meta["bounds_mercator"])
    overlay = Image.open(OVERLAY_PATH).convert("RGBA")
    dem = np.asarray(Image.open(DEM_PATH), dtype=np.float32)
    fire_x, fire_y = fire_pixel(bounds, overlay.size)

    crop_w = min(300, overlay.width)
    crop_h = min(230, overlay.height)
    left = int(np.clip(round(fire_x - crop_w * 0.35), 0, overlay.width - crop_w))
    top = int(np.clip(round(fire_y - crop_h * 0.60), 0, overlay.height - crop_h))
    box = (left, top, left + crop_w, top + crop_h)
    crop_overlay = overlay.crop(box)
    crop_dem = dem[top : top + crop_h, left : left + crop_w]
    fire_in_crop = (fire_x - left, fire_y - top)
    return enhance_terrain_texture(crop_overlay, crop_dem), crop_dem, fire_in_crop


def terrain_crop_mercator_bounds() -> tuple[float, float, float, float]:
    if not (OVERLAY_PATH.exists() and META_PATH.exists()):
        raise RuntimeError(
            "Cached California terrain metadata is missing. Run "
            "examples/california_wildfire_smoke_video.py once to populate examples/.cache."
        )
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    west, south, east, north = (float(v) for v in meta["bounds_mercator"])
    with Image.open(OVERLAY_PATH) as overlay:
        overlay_size = overlay.size
    fire_x, fire_y = fire_pixel((west, south, east, north), overlay_size)
    width, height = overlay_size
    crop_w = min(300, width)
    crop_h = min(230, height)
    left = int(np.clip(round(fire_x - crop_w * 0.35), 0, width - crop_w))
    top = int(np.clip(round(fire_y - crop_h * 0.60), 0, height - crop_h))
    x0 = west + left / max(width - 1, 1) * (east - west)
    x1 = west + (left + crop_w - 1) / max(width - 1, 1) * (east - west)
    y_top = north - top / max(height - 1, 1) * (north - south)
    y_bottom = north - (top + crop_h - 1) / max(height - 1, 1) * (north - south)
    return float(x0), float(y_bottom), float(x1), float(y_top)


def enhance_terrain_texture(texture: Image.Image, dem: np.ndarray) -> Image.Image:
    rgb = np.asarray(texture.convert("RGB"), dtype=np.float32) / 255.0
    valid = np.isfinite(dem)
    terrain = np.where(valid, dem, np.nanmedian(dem[valid]) if np.any(valid) else 0.0)
    lo, hi = np.percentile(terrain[valid], [2, 98]) if np.any(valid) else (0.0, 1.0)
    norm = np.clip((terrain - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    gy, gx = np.gradient(norm)
    shade = np.clip(0.48 - gx * 2.15 - gy * 1.30 + norm * 0.36, 0.20, 1.16)
    luma = np.sum(rgb * np.array([0.299, 0.587, 0.114], dtype=np.float32), axis=2)
    coal = np.array([0.030, 0.034, 0.035], dtype=np.float32)
    charcoal = np.array([0.100, 0.104, 0.098], dtype=np.float32)
    ash = np.array([0.310, 0.300, 0.270], dtype=np.float32)
    grade = (
        coal * (1.0 - norm[..., None])
        + charcoal * (1.0 - np.abs(norm[..., None] - 0.52) * 1.42).clip(0.0, 1.0)
        + ash * (norm[..., None] ** 1.85)
    )
    relief_detail = np.clip(0.76 + luma * 0.26, 0.72, 1.02)
    out = np.clip(grade * shade[..., None] * relief_detail[..., None] * 0.88, 0.0, 1.0)
    return Image.fromarray(np.round(out * 255.0).astype(np.uint8), mode="RGB").convert("RGBA")


def perspective_coeffs(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> list[float]:
    matrix = []
    vector = []
    for (x, y), (u, v) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        vector.extend([u, v])
    return np.linalg.solve(np.asarray(matrix, dtype=np.float64), np.asarray(vector)).tolist()


def terrain_plate(width: int, height: int) -> TerrainPlate:
    texture, _dem, fire_crop = crop_fire_extent()
    sky = Image.new("RGBA", (width, height), (11, 17, 22, 255))
    arr = np.array(sky, dtype=np.uint8, copy=True)
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    arr[..., 0] = np.round(14 + 18 * (1.0 - yy)).astype(np.uint8)
    arr[..., 1] = np.round(20 + 28 * (1.0 - yy)).astype(np.uint8)
    arr[..., 2] = np.round(25 + 36 * (1.0 - yy)).astype(np.uint8)
    base = Image.fromarray(arr, mode="RGBA")

    dst = [
        (width * 0.09, height * 0.27),
        (width * 0.91, height * 0.15),
        (width * 0.96, height * 0.88),
        (width * 0.06, height * 0.95),
    ]
    draw = ImageDraw.Draw(base, "RGBA")
    draw.polygon([dst[3], dst[2], (width * 0.94, height * 0.98), (width * 0.07, height * 1.04)], fill=(4, 8, 10, 220))
    coeffs = perspective_coeffs(
        [(0, 0), (texture.width, 0), (texture.width, texture.height), (0, texture.height)],
        dst,
    )
    warped = texture.transform((width, height), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
    base.alpha_composite(warped)
    fire_uv = (fire_crop[0] / texture.width, fire_crop[1] / texture.height)
    fire_screen = bilinear_quad_point(dst, *fire_uv)
    return TerrainPlate(
        image=base,
        quad=dst,
        fire_xy=fire_screen,
        fire_uv=fire_uv,
        texture_size=texture.size,
    )


def bilinear_quad_point(quad: list[tuple[float, float]], u: float, v: float) -> tuple[float, float]:
    tl, tr, br, bl = [np.asarray(p, dtype=np.float32) for p in quad]
    top = tl * (1.0 - u) + tr * u
    bottom = bl * (1.0 - u) + br * u
    point = top * (1.0 - v) + bottom * v
    return float(point[0]), float(point[1])


def _smoothstep(edge0: float, edge1: float, value: np.ndarray | float) -> np.ndarray:
    denom = max(float(edge1) - float(edge0), 1.0e-6)
    t = np.clip((np.asarray(value, dtype=np.float32) - float(edge0)) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _pixel_grids(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = (int(shape[0]), int(shape[1]))
    key = (height, width)
    cached = _PIXEL_GRID_CACHE.get(key)
    if cached is None:
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        cached = (x, y)
        _PIXEL_GRID_CACHE[key] = cached
    return cached


def _box_blur_3x3(field: np.ndarray, passes: int = 1) -> np.ndarray:
    out = np.asarray(field, dtype=np.float32)
    for _ in range(max(0, int(passes))):
        padded = np.pad(out, 1, mode="edge")
        center = padded[1:-1, 1:-1]
        axial = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        )
        diagonal = (
            padded[:-2, :-2]
            + padded[:-2, 2:]
            + padded[2:, :-2]
            + padded[2:, 2:]
        )
        out = center * 0.42 + axial * 0.1175 + diagonal * 0.0275
    return out.astype(np.float32, copy=False)


def _pil_blur_float(field: np.ndarray, radius: float) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float32)
    peak = max(float(np.max(arr)), 1.0e-6)
    image = Image.fromarray(np.clip(arr / peak * 255.0, 0, 255).astype(np.uint8))
    blurred = image.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return (np.asarray(blurred, dtype=np.float32) / 255.0 * peak).astype(np.float32)


def _bilinear_sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    height, width = src.shape
    valid = (x >= 0.0) & (x <= width - 1.0) & (y >= 0.0) & (y <= height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y1, 0, height - 1)
    wx = x - x0.astype(np.float32)
    wy = y - y0.astype(np.float32)
    top = src[y0c, x0c] * (1.0 - wx) + src[y0c, x1c] * wx
    bottom = src[y1c, x0c] * (1.0 - wx) + src[y1c, x1c] * wx
    sampled = top * (1.0 - wy) + bottom * wy
    return np.where(valid, sampled, 0.0).astype(np.float32)


def _resample_float_field(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError("field must be 2D")
    target_h, target_w = int(shape[0]), int(shape[1])
    if src.shape == (target_h, target_w):
        return src.astype(np.float32, copy=True)
    x, y = _pixel_grids((target_h, target_w))
    src_h, src_w = src.shape
    sample_x = x / max(float(target_w - 1), 1.0) * max(float(src_w - 1), 0.0)
    sample_y = y / max(float(target_h - 1), 1.0) * max(float(src_h - 1), 0.0)
    return _bilinear_sample(src, sample_x, sample_y)


def _bilinear_sample_wrapped(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    height, width = src.shape
    xw = np.mod(x, max(width, 1)).astype(np.float32)
    yw = np.mod(y, max(height, 1)).astype(np.float32)
    xf = np.floor(xw)
    yf = np.floor(yw)
    x0 = np.mod(xf.astype(np.int64), width)
    y0 = np.mod(yf.astype(np.int64), height)
    x1 = (x0 + 1) % width
    y1 = (y0 + 1) % height
    wx = xw - xf.astype(np.float32)
    wy = yw - yf.astype(np.float32)
    top = src[y0, x0] * (1.0 - wx) + src[y0, x1] * wx
    bottom = src[y1, x0] * (1.0 - wx) + src[y1, x1] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def _volume_grids(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z, y, x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(np.float32)
    return x, y, z


def _trilinear_sample_volume(field: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
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


def _advect_volume_scalar(
    field: np.ndarray,
    velocity: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    dt: float,
    *,
    mac_cormack: bool = False,
    min_value: float = 0.0,
) -> np.ndarray:
    back_x = grid_x - velocity[..., 0] * np.float32(dt)
    back_y = grid_y - velocity[..., 1] * np.float32(dt)
    back_z = grid_z - velocity[..., 2] * np.float32(dt)
    predicted = _trilinear_sample_volume(field, back_x, back_y, back_z)
    if not mac_cormack:
        return np.clip(predicted, float(min_value), None).astype(np.float32)

    back_vx = _trilinear_sample_volume(velocity[..., 0], back_x, back_y, back_z)
    back_vy = _trilinear_sample_volume(velocity[..., 1], back_x, back_y, back_z)
    back_vz = _trilinear_sample_volume(velocity[..., 2], back_x, back_y, back_z)
    fwd_x = back_x + back_vx * np.float32(dt)
    fwd_y = back_y + back_vy * np.float32(dt)
    fwd_z = back_z + back_vz * np.float32(dt)
    recovered = _trilinear_sample_volume(predicted, fwd_x, fwd_y, fwd_z)
    candidate = predicted + (np.asarray(field, dtype=np.float32) - recovered) * np.float32(0.5)
    lo, hi = _local_min_max_volume(field, back_x, back_y, back_z)
    corrected = np.clip(candidate, lo, hi)
    return np.clip(corrected, float(min_value), None).astype(np.float32)


def _local_min_max_volume(field: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    lo = np.minimum.reduce(corners)
    hi = np.maximum.reduce(corners)
    return lo.astype(np.float32), hi.astype(np.float32)


def _diffuse_volume(field: np.ndarray, mix: float) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    padded = np.pad(src, 1, mode="edge")
    axial = (
        padded[1:-1, 1:-1, :-2]
        + padded[1:-1, 1:-1, 2:]
        + padded[1:-1, :-2, 1:-1]
        + padded[1:-1, 2:, 1:-1]
        + padded[:-2, 1:-1, 1:-1]
        + padded[2:, 1:-1, 1:-1]
    ) / 6.0
    return (src * (1.0 - mix) + axial * mix).astype(np.float32)


def _volume_divergence(velocity: np.ndarray) -> np.ndarray:
    vel = np.asarray(velocity, dtype=np.float32)
    div = np.zeros(vel.shape[:3], dtype=np.float32)
    div[:, :, 1:-1] += (vel[:, :, 2:, 0] - vel[:, :, :-2, 0]) * 0.5
    div[:, 1:-1, :] += (vel[:, 2:, :, 1] - vel[:, :-2, :, 1]) * 0.5
    div[1:-1, :, :] += (vel[2:, :, :, 2] - vel[:-2, :, :, 2]) * 0.5
    return div


def _smooth_noise_field(shape: tuple[int, int], seed: int, cell_px: float, blur_radius: float) -> np.ndarray:
    height, width = shape
    cell = max(float(cell_px), 2.0)
    small_w = max(4, int(math.ceil(width / cell)) + 3)
    small_h = max(4, int(math.ceil(height / cell)) + 3)
    rng = np.random.default_rng(int(seed))
    raw = np.round(rng.random((small_h, small_w), dtype=np.float32) * 255.0).astype(np.uint8)
    image = Image.fromarray(raw).resize((width, height), Image.Resampling.BICUBIC)
    if blur_radius > 0.0:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    noise = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip(noise, 0.0, 1.0).astype(np.float32)


def hrrr_smoke_image_url(
    runtime: str,
    forecast_hour: int,
    *,
    base_url: str = HRRR_SMOKE_BASE_URL,
    plot_type: str = HRRR_SMOKE_PLOT_TYPE,
    dataset_key: str = HRRR_SMOKE_DATASET_KEY,
) -> str:
    root = str(base_url).rstrip("/")
    return (
        f"{root}/for_web/{str(dataset_key)}/{str(runtime)}/full/"
        f"{str(plot_type)}_f{int(forecast_hour):03d}.png"
    )


def _hrrr_smoke_display_url(
    runtime: str,
    forecast_hour: int,
    *,
    base_url: str,
    plot_type: str,
    dataset_key: str = HRRR_SMOKE_DATASET_KEY,
) -> str:
    root = str(base_url).rstrip("/")
    query = urllib.parse.urlencode(
        {
            "keys": str(dataset_key),
            "runtime": str(runtime),
            "plot_type": str(plot_type),
            "fcst": f"{int(forecast_hour):03d}",
            "time_inc": "60",
        }
    )
    return f"{root}/displayMapUpdated.cgi?{query}"


class _HrrrSmokeImageParser(HTMLParser):
    def __init__(self, runtime: str, plot_type: str) -> None:
        super().__init__()
        self.runtime = str(runtime)
        self.plot_type = str(plot_type)
        self.src: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.src is not None or tag.lower() != "img":
            return
        attr_map = {key.lower(): value for key, value in attrs if value is not None}
        src = attr_map.get("src", "")
        if (
            ".png" in src
            and "for_web/" in src
            and self.runtime in src
            and f"{self.plot_type}_f" in src
        ):
            self.src = src


def _hrrr_smoke_base_url_candidates(base_url: str) -> tuple[str, ...]:
    candidates: list[str] = []
    for item in (str(base_url), *HRRR_SMOKE_BASE_URLS):
        normalized = item.rstrip("/")
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return tuple(candidates)


def _hrrr_smoke_cache_path(
    cache_dir: Path,
    runtime: str,
    forecast_hour: int,
    plot_type: str,
) -> Path:
    return Path(cache_dir) / str(runtime) / f"{str(plot_type)}_f{int(forecast_hour):03d}.png"


def _hrrr_raw_grib_url(
    runtime: str,
    forecast_hour: int,
    *,
    base_url: str = HRRR_SMOKE_RAW_BASE_URL,
) -> str:
    runtime_text = str(runtime)
    date = runtime_text[:8]
    cycle = runtime_text[8:10]
    root = str(base_url).rstrip("/")
    return f"{root}/hrrr.{date}/conus/hrrr.t{cycle}z.wrfsfcf{int(forecast_hour):02d}.grib2"


def _hrrr_raw_idx_url(
    runtime: str,
    forecast_hour: int,
    *,
    base_url: str = HRRR_SMOKE_RAW_BASE_URL,
) -> str:
    return f"{_hrrr_raw_grib_url(runtime, forecast_hour, base_url=base_url)}.idx"


def _hrrr_raw_cache_path(
    cache_dir: Path,
    runtime: str,
    forecast_hour: int,
    field: str = HRRR_SMOKE_RAW_FIELD,
) -> Path:
    return Path(cache_dir) / str(runtime) / f"{str(field).lower()}_f{int(forecast_hour):03d}.grib2"


def _parse_grib_index_range(index_text: str, field: str) -> tuple[int, int] | None:
    rows: list[tuple[int, int, str]] = []
    for line in index_text.splitlines():
        parts = line.split(":")
        if len(parts) < 5:
            continue
        try:
            message_number = int(parts[0])
            offset = int(parts[1])
        except ValueError:
            continue
        rows.append((message_number, offset, parts[3]))
    for idx, (_message_number, offset, name) in enumerate(rows):
        if name != field:
            continue
        if idx + 1 >= len(rows):
            return None
        return offset, rows[idx + 1][1] - 1
    return None


def _fetch_url_payload(url: str, extra_headers: dict[str, str] | None = None) -> tuple[bytes, str, str]:
    headers = {"User-Agent": "forge3d-california-cigar-smoke/1.0"}
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(
        str(url),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=20.0) as response:
        payload = response.read()
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
    return payload, final_url, content_type


def _is_png_payload(payload: bytes, content_type: str = "") -> bool:
    return (
        len(payload) > 1024
        and payload.startswith(b"\x89PNG\r\n\x1a\n")
        and ("png" in content_type.lower() or not content_type)
    )


def _download_hrrr_smoke_png(
    url: str,
    dest: Path,
    *,
    display_url: str | None = None,
    runtime: str,
    plot_type: str,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload, _, content_type = _fetch_url_payload(url)
        if _is_png_payload(payload, content_type):
            dest.write_bytes(payload)
            return
    except urllib.error.HTTPError:
        pass

    if display_url is not None:
        display_payload, final_display_url, _ = _fetch_url_payload(display_url)
        parser = _HrrrSmokeImageParser(runtime, plot_type)
        parser.feed(display_payload.decode("utf-8", errors="ignore"))
        if parser.src:
            image_url = urllib.parse.urljoin(final_display_url, parser.src)
            payload, _, content_type = _fetch_url_payload(image_url)
            if _is_png_payload(payload, content_type):
                dest.write_bytes(payload)
                return

    raise RuntimeError(f"HRRR-Smoke response was not an available PNG: {url}")


def _download_hrrr_raw_smoke_grib(
    runtime: str,
    forecast_hour: int,
    dest: Path,
    *,
    field: str = HRRR_SMOKE_RAW_FIELD,
    base_url: str = HRRR_SMOKE_RAW_BASE_URL,
) -> None:
    index_payload, _, _ = _fetch_url_payload(_hrrr_raw_idx_url(runtime, forecast_hour, base_url=base_url))
    byte_range = _parse_grib_index_range(index_payload.decode("utf-8", errors="replace"), field)
    if byte_range is None:
        raise RuntimeError(f"Raw HRRR field {field} was not found with a bounded byte range")
    start, end = byte_range
    payload, _, _ = _fetch_url_payload(
        _hrrr_raw_grib_url(runtime, forecast_hour, base_url=base_url),
        extra_headers={"Range": f"bytes={start}-{end}"},
    )
    if not payload.startswith(b"GRIB") or len(payload) < 1024:
        raise RuntimeError(f"Raw HRRR field {field} response was not a GRIB2 message")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)


def _hrrr_raw_smoke_grib_to_density(grib_path: Path, map_size: tuple[int, int]) -> np.ndarray:
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.transform import from_bounds
        from rasterio.warp import reproject
    except (ImportError, AttributeError):
        return np.zeros((int(map_size[1]), int(map_size[0])), dtype=np.float32)

    width, height = map(int, map_size)
    dst = np.zeros((height, width), dtype=np.float32)
    dst_transform = from_bounds(*terrain_crop_mercator_bounds(), width, height)
    try:
        with rasterio.open(grib_path) as src:
            source = src.read(1).astype(np.float32, copy=False)
            source = np.where(np.isfinite(source), np.clip(source, 0.0, None), 0.0).astype(np.float32)
            reproject(
                source=source,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:3857",
                src_nodata=0.0,
                dst_nodata=0.0,
                resampling=Resampling.bilinear,
            )
    except Exception as exc:
        print(f"Skipping unreadable raw HRRR-Smoke frame {grib_path}: {exc}")
        return np.zeros((height, width), dtype=np.float32)

    field = np.clip(dst, 0.0, None).astype(np.float32)
    positive = field[field > 0.0]
    if positive.size == 0 or float(np.max(positive)) < 1.0e-10:
        return np.zeros((height, width), dtype=np.float32)
    lo = float(np.percentile(positive, 45.0))
    hi = float(np.percentile(positive, 99.4))
    density = np.clip((field - lo) / max(hi - lo, 1.0e-10), 0.0, 1.0)
    density = density**0.72
    return _pil_blur_float(density, max(0.45, min(width, height) / 520.0)).astype(np.float32)


def _hrrr_smoke_image_to_density(image: Image.Image, map_size: tuple[int, int]) -> np.ndarray:
    width, height = map(int, map_size)
    source_image = image.convert("RGB")
    src_w, src_h = source_image.size
    if src_w >= 500 and src_h >= 400:
        px0, py0, px1, py1 = HRRR_SMOKE_PANEL_CROP_FRAC
        panel_box = (
            int(round(src_w * px0)),
            int(round(src_h * py0)),
            int(round(src_w * px1)),
            int(round(src_h * py1)),
        )
        panel = source_image.crop(panel_box)
        pw, ph = panel.size
        sx0, sy0, sx1, sy1 = HRRR_SMOKE_CA_SUBSET_FRAC
        source_image = panel.crop(
            (
                int(round(pw * sx0)),
                int(round(ph * sy0)),
                int(round(pw * sx1)),
                int(round(ph * sy1)),
            )
        )

    rgb = np.asarray(source_image, dtype=np.float32) / 255.0
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return np.zeros((height, width), dtype=np.float32)

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    luma = r * 0.299 + g * 0.587 + b * 0.114
    saturation = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    warm_smoke = np.clip((r * 0.60 + g * 0.42) - b * 0.48, 0.0, 1.0)
    cool_smoke = np.clip(b * 0.85 - r * 0.22 - g * 0.16, 0.0, 1.0)
    pale_haze = np.clip(luma - 0.56, 0.0, 1.0) * np.clip(0.34 - saturation, 0.0, 0.34) * 0.36
    signal = saturation * 0.78 + warm_smoke * 0.22 + cool_smoke * 0.18 + pale_haze
    signal *= _smoothstep(0.045, 0.13, saturation)
    signal *= 1.0 - 0.86 * _smoothstep(0.90, 0.99, luma)
    signal *= _smoothstep(0.16, 0.30, luma)
    if src_w >= 500 and src_h >= 400:
        hh, ww = signal.shape
        border = max(2, int(round(min(ww, hh) * 0.018)))
        signal[:border, :] = 0.0
        signal[-border:, :] = 0.0
        signal[:, :border] = 0.0
        signal[:, -border:] = 0.0
    signal = np.clip(signal, 0.0, 1.0).astype(np.float32)

    source = Image.fromarray(np.round(signal * 255.0).astype(np.uint8), mode="L")
    resized = source.resize((width, height), Image.Resampling.BICUBIC)
    density = np.asarray(resized, dtype=np.float32) / 255.0
    density = _pil_blur_float(density, max(1.0, min(width, height) / 180.0))
    positive = density[density > 0.0]
    if positive.size == 0 or float(np.max(positive)) < 1.0e-5:
        return np.zeros((height, width), dtype=np.float32)
    lo = float(np.percentile(positive, 58.0))
    hi = float(np.percentile(positive, 99.0))
    density = np.clip((density - lo) / max(hi - lo, 1.0e-5), 0.0, 1.0)
    return _pil_blur_float(density, max(0.7, min(width, height) / 260.0)).astype(np.float32)


def _load_hrrr_raw_smoke_guidance(
    cache_dir: Path,
    map_size: tuple[int, int],
    *,
    runtime: str = HRRR_SMOKE_RUNTIME,
    forecast_hours: tuple[int, ...] = HRRR_SMOKE_FORECAST_HOURS,
    fetch: bool = False,
    raw_base_url: str = HRRR_SMOKE_RAW_BASE_URL,
    field: str = HRRR_SMOKE_RAW_FIELD,
) -> HrrrSmokeGuidance | None:
    frames: list[np.ndarray] = []
    for hour in forecast_hours:
        path = _hrrr_raw_cache_path(Path(cache_dir), runtime, int(hour), field)
        if fetch and not path.exists():
            try:
                _download_hrrr_raw_smoke_grib(
                    runtime,
                    int(hour),
                    path,
                    field=field,
                    base_url=raw_base_url,
                )
            except (OSError, RuntimeError, urllib.error.URLError) as exc:
                print(f"Skipping unavailable raw HRRR-Smoke frame {runtime} f{int(hour):03d}: {exc}")
                continue
        if not path.exists():
            continue
        density = _hrrr_raw_smoke_grib_to_density(path, map_size)
        if np.any(density > 0.0):
            frames.append(density.astype(np.float32, copy=False))

    if not frames:
        return None
    label = f"raw HRRR-Smoke {runtime} {field} ({len(frames)} cached GRIB frames)"
    return HrrrSmokeGuidance(tuple(frames), str(runtime), str(field), label)


def load_hrrr_smoke_guidance(
    cache_dir: Path,
    map_size: tuple[int, int],
    *,
    runtime: str = HRRR_SMOKE_RUNTIME,
    plot_type: str = HRRR_SMOKE_PLOT_TYPE,
    forecast_hours: tuple[int, ...] = HRRR_SMOKE_FORECAST_HOURS,
    base_url: str = HRRR_SMOKE_BASE_URL,
    fetch: bool = False,
    prefer_raw: bool = True,
    raw_base_url: str = HRRR_SMOKE_RAW_BASE_URL,
) -> HrrrSmokeGuidance | None:
    if prefer_raw:
        raw_guidance = _load_hrrr_raw_smoke_guidance(
            cache_dir,
            map_size,
            runtime=runtime,
            forecast_hours=forecast_hours,
            fetch=fetch,
            raw_base_url=raw_base_url,
        )
        if raw_guidance is not None:
            return raw_guidance

    frames: list[np.ndarray] = []
    for hour in forecast_hours:
        path = _hrrr_smoke_cache_path(Path(cache_dir), runtime, int(hour), plot_type)
        if fetch and not path.exists():
            errors: list[str] = []
            for candidate_base_url in _hrrr_smoke_base_url_candidates(base_url):
                try:
                    _download_hrrr_smoke_png(
                        hrrr_smoke_image_url(
                            runtime,
                            int(hour),
                            base_url=candidate_base_url,
                            plot_type=plot_type,
                        ),
                        path,
                        display_url=_hrrr_smoke_display_url(
                            runtime,
                            int(hour),
                            base_url=candidate_base_url,
                            plot_type=plot_type,
                        ),
                        runtime=runtime,
                        plot_type=plot_type,
                    )
                    break
                except (OSError, RuntimeError, urllib.error.URLError) as exc:
                    errors.append(f"{candidate_base_url}: {exc}")
            if not path.exists():
                suffix = f": {'; '.join(errors[-2:])}" if errors else ""
                print(
                    "Skipping unavailable HRRR-Smoke frame "
                    f"{runtime} f{int(hour):03d}{suffix}"
                )
                continue
        if not path.exists():
            continue
        try:
            density = _hrrr_smoke_image_to_density(Image.open(path), map_size)
        except OSError as exc:
            print(f"Skipping unreadable HRRR-Smoke frame {path}: {exc}")
            continue
        if np.any(density > 0.0):
            frames.append(density.astype(np.float32, copy=False))

    if not frames:
        return None
    label = f"HRRR-Smoke {runtime} {plot_type} ({len(frames)} cached frames)"
    return HrrrSmokeGuidance(tuple(frames), str(runtime), str(plot_type), label)


def _advected_smoke_texture(shape: tuple[int, int], frame_index: int, seed: int) -> np.ndarray:
    height, width = shape
    x, y = _pixel_grids(shape)
    scale = min(width, height) / 408.0
    t = float(frame_index)
    cache_key = (shape, int(seed))
    cached = _SMOKE_TEXTURE_CACHE.get(cache_key)
    if cached is None:
        cached = (
            _smooth_noise_field(shape, int(seed) + 301, 112.0 * scale, 10.5 * scale),
            _smooth_noise_field(shape, int(seed) + 907, 46.0 * scale, 4.8 * scale),
            _smooth_noise_field(shape, int(seed) + 1709, 18.0 * scale, 1.7 * scale),
        )
        _SMOKE_TEXTURE_CACHE[cache_key] = cached
    broad, medium, fine = cached

    drift_x = 1.18 * scale * t
    drift_y = -0.43 * scale * t
    curl_x = (medium - 0.5) * 22.0 * scale + (broad - 0.5) * 62.0 * scale
    curl_y = (broad - 0.5) * 38.0 * scale - (medium - 0.5) * 12.0 * scale
    broad_s = _bilinear_sample_wrapped(broad, x - drift_x * 0.38 + curl_x * 0.15, y - drift_y * 0.38 + curl_y * 0.15)
    medium_s = _bilinear_sample_wrapped(medium, x - drift_x + curl_x * 0.34, y - drift_y + curl_y * 0.34)
    fine_s = _bilinear_sample_wrapped(fine, x - drift_x * 1.46 + curl_x * 0.72, y - drift_y * 1.46 + curl_y * 0.72)
    texture = 0.50 * broad_s + 0.34 * medium_s + 0.16 * fine_s
    return np.clip(texture, 0.0, 1.0).astype(np.float32)


def _hybrid_border_fade(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x, y = _pixel_grids(shape)
    distance = np.minimum.reduce((x, y, width - 1.0 - x, height - 1.0 - y))
    margin = max(6.0, min(width, height) * 0.045)
    return _smoothstep(0.0, margin, distance).astype(np.float32)


def _hybrid_lifecycle_alpha(age_frames: np.ndarray) -> np.ndarray:
    age = np.asarray(age_frames, dtype=np.float32)
    birth = 0.12 + 0.88 * _smoothstep(0.0, 6.0, age)
    mature = 1.0 - 0.20 * _smoothstep(30.0, 100.0, age)
    fade_end = min(HYBRID_SMOKE_MAX_AGE_FRAMES - 28.0, 236.0)
    old_fade = 1.0 - _smoothstep(145.0, fade_end, age) ** 0.75
    return np.clip(birth * mature * old_fade, 0.0, 1.0).astype(np.float32)


def _procedural_hrrr_smoke_guidance(
    shape: tuple[int, int],
    frame_index: int,
    seed: int,
    source_xy: tuple[float, float],
) -> np.ndarray:
    height, width = shape
    x, y = _pixel_grids(shape)
    scale = min(width, height) / 408.0
    sx, sy = source_xy
    t = float(frame_index)
    wind = np.array([1.0, -0.42], dtype=np.float32)
    wind /= max(float(np.linalg.norm(wind)), 1.0e-6)
    cross_dir = np.array([-wind[1], wind[0]], dtype=np.float32)
    dx = x - np.float32(sx)
    dy = y - np.float32(sy)
    along = dx * wind[0] + dy * wind[1]
    cross = dx * cross_dir[0] + dy * cross_dir[1]
    along_shift = along - t * 0.54 * scale
    downwind = np.maximum(along_shift, 0.0)
    range_fade = 1.0 - _smoothstep(width * 0.68, width * 1.02, downwind)

    wave = np.sin(along_shift / max(46.0 * scale, 1.0) + t * 0.013 + seed * 0.003)
    shear_wave = np.sin(along_shift / max(92.0 * scale, 1.0) - t * 0.006 + seed * 0.007)
    lane_center = (28.0 * wave + 13.0 * shear_wave) * scale
    lane_width = (10.0 + 0.055 * downwind) * scale
    primary_lane = (
        np.exp(-((cross - lane_center) ** 2) / (2.0 * lane_width * lane_width + 1.0e-6))
        * _smoothstep(-20.0 * scale, 48.0 * scale, along_shift)
        * np.exp(-downwind / max(520.0 * scale, 1.0))
        * range_fade
    )

    secondary_center = lane_center - (34.0 + 18.0 * np.sin(t * 0.011 + seed * 0.013)) * scale
    secondary_width = (17.0 + 0.080 * downwind) * scale
    secondary_lane = (
        np.exp(-((cross - secondary_center) ** 2) / (2.0 * secondary_width * secondary_width + 1.0e-6))
        * _smoothstep(16.0 * scale, 96.0 * scale, along_shift)
        * np.exp(-downwind / max(620.0 * scale, 1.0))
        * range_fade
    )

    front_center = (76.0 + 0.24 * t) * scale
    leading_band = (
        np.exp(-((along_shift - front_center) ** 2) / (2.0 * (48.0 * scale) ** 2 + 1.0e-6))
        * np.exp(-((cross - lane_center * 0.38) ** 2) / (2.0 * (72.0 * scale) ** 2 + 1.0e-6))
        * _smoothstep(-10.0 * scale, 70.0 * scale, along_shift)
        * range_fade
    )

    hook_x = sx + wind[0] * (82.0 + t * 0.31) * scale
    hook_y = sy + wind[1] * (82.0 + t * 0.31) * scale
    hx = x - np.float32(hook_x)
    hy = y - np.float32(hook_y)
    radius = np.hypot(hx, hy)
    theta = np.arctan2(hy, hx)
    spiral_radius = (46.0 + 0.12 * t + 8.0 * np.sin(theta * 2.0 - t * 0.019)) * scale
    hook = np.exp(-((radius - spiral_radius) ** 2) / (2.0 * (12.5 * scale) ** 2 + 1.0e-6))
    hook *= np.clip(0.48 + 0.52 * np.cos(theta - 0.017 * t + seed * 0.002), 0.0, 1.0)
    hook *= _smoothstep(18.0 * scale, 175.0 * scale, along) * range_fade

    eddy_x = sx + wind[0] * (185.0 + 0.18 * t) * scale + cross_dir[0] * 46.0 * scale
    eddy_y = sy + wind[1] * (185.0 + 0.18 * t) * scale + cross_dir[1] * 46.0 * scale
    er = np.hypot(x - np.float32(eddy_x), y - np.float32(eddy_y))
    et = np.arctan2(y - np.float32(eddy_y), x - np.float32(eddy_x))
    eddy = np.exp(-((er - 64.0 * scale) ** 2) / (2.0 * (24.0 * scale) ** 2 + 1.0e-6))
    eddy *= np.clip(0.42 + 0.58 * np.cos(et + t * 0.010 - seed * 0.005), 0.0, 1.0)
    eddy *= _smoothstep(70.0 * scale, 260.0 * scale, along_shift) * range_fade

    veil_axis = cross + 0.30 * along_shift - 22.0 * scale * np.sin(t * 0.009 + seed * 0.011)
    aged_veil = np.exp(-(veil_axis * veil_axis) / (2.0 * (118.0 * scale) ** 2 + 1.0e-6))
    aged_veil *= _smoothstep(-44.0 * scale, 125.0 * scale, along_shift)
    aged_veil *= 1.0 - _smoothstep(width * 0.74, width * 1.08, downwind)

    northern_sheet_center = -height * 0.08 + 42.0 * scale * np.sin(t * 0.006 + seed * 0.017)
    synoptic_sheet = np.exp(-((cross - northern_sheet_center) ** 2) / (2.0 * (152.0 * scale) ** 2 + 1.0e-6))
    synoptic_sheet *= _smoothstep(95.0 * scale, 300.0 * scale, along_shift)
    synoptic_sheet *= 1.0 - _smoothstep(width * 0.78, width * 1.10, downwind)

    texture = _advected_smoke_texture(shape, frame_index, int(seed) + 809)
    broad_texture = _pil_blur_float(texture, 7.0 * scale)
    striations = 0.5 + 0.5 * np.sin(
        cross / max(8.5 * scale, 1.0)
        + along_shift / max(44.0 * scale, 1.0)
        + t * 0.021
        + seed * 0.019
    )
    base = (
        0.42 * primary_lane
        + 0.23 * secondary_lane
        + 0.17 * leading_band
        + 0.18 * hook
        + 0.12 * eddy
        + 0.15 * aged_veil
        + 0.08 * synoptic_sheet
    )
    holes = _smoothstep(0.58, 0.90, 1.0 - texture) * _smoothstep(0.05, 0.42, base)
    breakup = np.clip(0.52 + 0.58 * texture + 0.20 * striations + 0.16 * (broad_texture - 0.5), 0.12, 1.26)
    guidance = base * breakup * (1.0 - 0.34 * holes)
    guidance = _pil_blur_float(np.clip(guidance, 0.0, 1.0), max(0.7, 1.05 * scale))
    return np.clip(guidance, 0.0, 1.0).astype(np.float32)


def make_hybrid_smoke_sources(
    fire_uv: tuple[float, float],
    map_size: tuple[int, int],
    total_frames: int = 120,
    seed: int = HYBRID_SMOKE_SEED,
) -> list[HybridSmokeSource]:
    width, height = map(int, map_size)
    if width <= 8 or height <= 8:
        raise ValueError("hybrid smoke map must be larger than 8x8")
    rng = np.random.default_rng(int(seed))
    fire_x = float(fire_uv[0]) * (width - 1)
    fire_y = float(fire_uv[1]) * (height - 1)
    scale = min(width, height) / 408.0
    wind = np.array([1.0, -0.38], dtype=np.float32)
    wind /= max(float(np.linalg.norm(wind)), 1.0e-6)
    cross = np.array([-wind[1], wind[0]], dtype=np.float32)
    clusters = (
        (0.0, 0.0, 24, 1.22, 1.00),
        (-20.0, 20.0, 10, 0.92, 0.94),
        (20.0, -17.0, 9, 0.82, 0.90),
        (-15.0, -31.0, 8, 0.68, 0.86),
        (31.0, 13.0, 7, 0.58, 0.80),
    )
    sources: list[HybridSmokeSource] = []
    source_index = 0
    for cluster_index, (dx, dy, count, cluster_strength, spread) in enumerate(clusters):
        center = np.array([fire_x + dx * scale, fire_y + dy * scale], dtype=np.float32)
        for _ in range(count):
            along = rng.normal(0.0, 10.0 * spread * scale)
            lateral = rng.normal(0.0, 20.0 * spread * scale)
            jitter = wind * along + cross * lateral
            x = float(np.clip(center[0] + jitter[0], 3.0, width - 4.0))
            y = float(np.clip(center[1] + jitter[1], 3.0, height - 4.0))
            start_limit = max(1, min(30, int(total_frames * 0.24)))
            start_frame = 0 if source_index < 6 else int(rng.integers(0, start_limit))
            radius = float(rng.uniform(4.2, 9.4) * scale * (0.92 + 0.20 * cluster_strength))
            strength = float(cluster_strength * rng.uniform(0.50, 1.08))
            burst_period = float(rng.uniform(32.0, 78.0))
            altitude_bias = float(np.clip(0.18 + cluster_index * 0.13 + rng.normal(0.0, 0.12), -0.18, 0.70))
            sources.append(
                HybridSmokeSource(
                    x=x,
                    y=y,
                    strength=strength,
                    radius_px=max(1.8, radius),
                    start_frame=start_frame,
                    end_frame=int(total_frames) + 300,
                    seed=int(seed + source_index * 101 + cluster_index * 17),
                    burst_period_frames=burst_period,
                    burst_phase_frames=float(rng.uniform(0.0, burst_period)),
                    burst_duty=float(rng.uniform(0.32, 0.58)),
                    heat=float(np.clip(0.74 + 0.48 * cluster_strength + rng.normal(0.0, 0.10), 0.42, 1.55)),
                    smoke_rate=float(np.clip(0.72 + 0.38 * cluster_strength + rng.normal(0.0, 0.08), 0.44, 1.46)),
                    altitude_bias=altitude_bias,
                )
            )
            source_index += 1
    return sources


def _hybrid_layer_altitude(layer_index: int) -> float:
    if HYBRID_SMOKE_LAYER_COUNT <= 1:
        return 0.0
    return float(np.clip(int(layer_index), 0, HYBRID_SMOKE_LAYER_COUNT - 1)) / float(HYBRID_SMOKE_LAYER_COUNT - 1)


def _hybrid_layer_wind_vector(layer_index: int) -> np.ndarray:
    altitude = _hybrid_layer_altitude(layer_index)
    direction = np.array(
        [
            1.0 + 0.12 * altitude,
            -0.34 - 0.24 * altitude + 0.05 * math.sin(2.1 + layer_index),
        ],
        dtype=np.float32,
    )
    direction /= max(float(np.linalg.norm(direction)), 1.0e-6)
    return direction


def _hybrid_wind_field(
    frame_index: float,
    shape: tuple[int, int],
    seed: int,
    layer_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    x, y = _pixel_grids(shape)
    xn = x / max(float(width - 1), 1.0)
    yn = y / max(float(height - 1), 1.0)
    scale = min(width, height) / 408.0
    altitude = _hybrid_layer_altitude(layer_index)
    phase = (int(seed) % 8191) / 8191.0 * math.tau + layer_index * 1.73
    t = float(frame_index)
    wind = _hybrid_layer_wind_vector(layer_index)
    speed = (2.48 + 0.62 * altitude) * scale

    u = np.full(shape, wind[0] * speed, dtype=np.float32)
    v = np.full(shape, wind[1] * speed, dtype=np.float32)
    u += (0.38 * scale * (1.0 + 0.22 * altitude) * np.sin(math.tau * (0.32 * yn + 0.0020 * t) + phase)).astype(np.float32)
    v += (0.30 * scale * (1.0 + 0.18 * altitude) * np.sin(math.tau * (0.36 * xn - 0.28 * yn + 0.0018 * t) + phase * 0.7)).astype(np.float32)

    stream = (
        np.sin(math.tau * ((0.62 + 0.08 * altitude) * xn + (0.34 - 0.05 * altitude) * yn) + 0.010 * t + phase)
        * np.sin(math.tau * ((0.46 - 0.06 * altitude) * yn - (0.27 + 0.04 * altitude) * xn) - 0.008 * t + phase * 1.61)
    ).astype(np.float32)
    dpsi_dy, dpsi_dx = np.gradient(stream)
    u += (4.60 + 0.55 * altitude) * scale * dpsi_dy.astype(np.float32)
    v += -(4.05 + 0.45 * altitude) * scale * dpsi_dx.astype(np.float32)

    synoptic = (
        np.sin(math.tau * (0.18 * xn - 0.31 * yn) + 0.004 * t + phase * 0.33)
        + np.cos(math.tau * (0.27 * xn + 0.20 * yn) - 0.003 * t + phase * 1.21)
    ).astype(np.float32)
    u += (0.36 * scale * (1.0 + 0.35 * altitude) * synoptic).astype(np.float32)
    v += (0.16 * scale * (1.0 + 0.28 * altitude) * np.sin(math.tau * (0.22 * xn + 0.24 * yn) - 0.006 * t + phase)).astype(np.float32)
    lane_texture = _pil_blur_float(
        _advected_smoke_texture(shape, int(round(t * 0.72)), seed + 6011 + layer_index * 811),
        max(3.4, 8.5 * scale),
    )
    lane_gy, lane_gx = np.gradient(lane_texture)
    lane_amp = (26.0 + 11.0 * altitude) * scale
    u += (lane_gy * lane_amp).astype(np.float32)
    v += (-lane_gx * lane_amp * 0.82).astype(np.float32)
    u = np.clip(u, -0.48 * scale, (4.20 + 0.70 * altitude) * scale)
    v = np.clip(v, -(1.76 + 0.38 * altitude) * scale, 1.02 * scale)
    return u.astype(np.float32), v.astype(np.float32)


def _hybrid_crosswind_spread(
    field: np.ndarray,
    amount: float = 0.16,
    wind: np.ndarray | None = None,
) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    if amount <= 0.0 or not np.any(src > 0.0):
        return src.copy()
    x, y = _pixel_grids(src.shape)
    scale = min(src.shape) / 408.0
    if wind is None:
        wind = _hybrid_layer_wind_vector(0)
    cross = np.array([-float(wind[1]), float(wind[0])], dtype=np.float32)
    cross /= max(float(np.linalg.norm(cross)), 1.0e-6)
    out = src * (1.0 - amount * 0.58)
    for distance, weight in ((3.0, 0.22), (7.2, 0.16), (13.5, 0.09), (22.0, 0.045)):
        dx = cross[0] * distance * scale
        dy = cross[1] * distance * scale
        shifted_a = _bilinear_sample(src, x - dx, y - dy)
        shifted_b = _bilinear_sample(src, x + dx, y + dy)
        out += amount * weight * (shifted_a + shifted_b)
    return np.clip(out, 0.0, 6.0).astype(np.float32)


def _hybrid_downwind_stream(
    field: np.ndarray,
    amount: float = 0.18,
    wind: np.ndarray | None = None,
) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    if amount <= 0.0 or not np.any(src > 0.0):
        return src.copy()
    x, y = _pixel_grids(src.shape)
    scale = min(src.shape) / 408.0
    if wind is None:
        wind = _hybrid_layer_wind_vector(0)
    out = src * (1.0 - amount * 0.50)
    for distance, weight in ((5.0, 0.26), (13.0, 0.18), (26.0, 0.10), (45.0, 0.05)):
        dx = wind[0] * distance * scale
        dy = wind[1] * distance * scale
        out += amount * weight * _bilinear_sample(src, x - dx, y - dy)
    return np.clip(out, 0.0, 6.0).astype(np.float32)


def _source_burst_envelope(source: HybridSmokeSource, frame_index: int) -> float:
    period = max(float(source.burst_period_frames), 1.0)
    phase = ((float(frame_index) + float(source.burst_phase_frames)) % period) / period
    attack = float(_smoothstep(0.0, 0.10, phase))
    release = 1.0 - float(_smoothstep(source.burst_duty, min(source.burst_duty + 0.24, 1.0), phase))
    ember = attack * release
    surge = 0.5 + 0.5 * math.sin(frame_index * 0.37 + source.seed * 0.029)
    return float(np.clip(0.06 + 1.62 * ember + 0.24 * surge * ember, 0.04, 1.92))


def _source_layer_weight(
    source: HybridSmokeSource,
    layer_index: int,
    layer_count: int = HYBRID_SMOKE_LAYER_COUNT,
) -> float:
    count = max(int(layer_count), 1)
    if count == 1:
        return 1.0
    altitudes = np.linspace(0.0, 1.0, count, dtype=np.float32)
    base_positions = np.linspace(0.0, 1.0, len(HYBRID_SMOKE_LAYER_WEIGHTS), dtype=np.float32)
    base = np.interp(altitudes, base_positions, np.asarray(HYBRID_SMOKE_LAYER_WEIGHTS, dtype=np.float32))
    center = np.clip(0.24 + float(source.altitude_bias) * 0.62, 0.06, 0.92)
    sigma = 0.36 + 0.08 * np.clip(float(source.heat) - 1.0, 0.0, 1.0)
    plume_lift = np.exp(-((altitudes - center) ** 2) / (2.0 * sigma * sigma + 1.0e-6))
    weights = base * (0.52 + plume_lift)
    total = max(float(np.sum(weights)), 1.0e-6)
    return float(weights[int(np.clip(layer_index, 0, count - 1))] / total)


def _inject_hybrid_sources(
    density: np.ndarray,
    age_mass: np.ndarray,
    sources: list[HybridSmokeSource],
    frame_index: int,
    layer_index: int = 0,
    layer_count: int = HYBRID_SMOKE_LAYER_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    out_density = np.asarray(density, dtype=np.float32).copy()
    out_age_mass = np.asarray(age_mass, dtype=np.float32).copy()
    height, width = out_density.shape
    altitude = _hybrid_layer_altitude(layer_index)
    wind = _hybrid_layer_wind_vector(layer_index)
    cross_dir = np.array([-wind[1], wind[0]], dtype=np.float32)

    for source_index, source in enumerate(sources):
        if frame_index < source.start_frame or frame_index > source.end_frame:
            continue
        layer_weight = _source_layer_weight(source, layer_index, layer_count)
        if layer_weight <= 0.012:
            continue
        radius = max(float(source.radius_px) * (1.0 + 0.30 * altitude), 1.0)
        tail = radius * (20.0 + 12.0 * altitude)
        pad = int(math.ceil(tail + radius * 7.0))
        x0 = max(0, int(math.floor(source.x - pad)))
        x1 = min(width, int(math.ceil(source.x + pad + tail)))
        y0 = max(0, int(math.floor(source.y - pad - tail * 0.70)))
        y1 = min(height, int(math.ceil(source.y + pad + tail * 0.20)))
        if x0 >= x1 or y0 >= y1:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx = xx - np.float32(source.x)
        dy = yy - np.float32(source.y)
        along = dx * wind[0] + dy * wind[1]
        cross = dx * cross_dir[0] + dy * cross_dir[1]
        along_frac = np.clip(along / max(tail, 1.0), 0.0, 1.0)
        core = np.exp(-(dx * dx + dy * dy) / (2.0 * (radius * 1.02) ** 2))
        tail_gate = _smoothstep(-radius * 0.35, radius * 1.20, along) * (along <= tail)
        curl_offset = radius * (
            2.8 * np.sin(along / max(radius * 8.0, 1.0) + source.seed * 0.017 + frame_index * 0.025)
            + 4.5 * along_frac * np.sin(along / max(radius * 15.0, 1.0) + source.seed * 0.009)
            + 2.0 * (along_frac ** 0.7) * np.sin(along / max(radius * 28.0, 1.0) + source.seed * 0.005 + frame_index * 0.012)
        )
        tail_width = radius * (1.15 + 4.2 * along_frac**0.75)
        plume_tail = (
            np.exp(-np.maximum(along, 0.0) / max(radius * 17.5, 1.0))
            * np.exp(-((cross - curl_offset) ** 2) / (2.0 * tail_width * tail_width + 1.0e-6))
            * tail_gate
        )
        veil_center = tail * 0.46
        veil_width = radius * 9.6
        broad_veil = (
            np.exp(-((along - veil_center) ** 2) / (2.0 * (tail * 0.52) ** 2 + 1.0e-6))
            * np.exp(-(cross * cross) / (2.0 * veil_width * veil_width + 1.0e-6))
            * tail_gate
        )
        sheet_offset = tail * (0.18 + 0.11 * math.sin(source.seed * 0.013))
        regional_sheet = (
            np.exp(-((along - sheet_offset) ** 2) / (2.0 * (tail * 0.82) ** 2 + 1.0e-6))
            * np.exp(-(cross * cross) / (2.0 * (radius * 15.5) ** 2 + 1.0e-6))
            * tail_gate
        )
        streamer_width = radius * (0.92 + 2.18 * along_frac)
        streamer_offset_a = radius * (
            2.0 * math.sin(source.seed * 0.019)
            + 5.1 * along_frac
            + 2.0 * math.sin(frame_index * 0.031 + source.seed * 0.007)
        )
        streamer_offset_b = -radius * (
            1.8 * math.cos(source.seed * 0.017)
            + 4.0 * along_frac
            + 1.6 * math.sin(frame_index * 0.027 + source.seed * 0.011)
        )
        streamer_a = (
            np.exp(-np.maximum(along, 0.0) / max(radius * 24.0, 1.0))
            * np.exp(-((cross - streamer_offset_a) ** 2) / (2.0 * streamer_width * streamer_width + 1.0e-6))
            * tail_gate
        )
        streamer_b = (
            np.exp(-np.maximum(along, 0.0) / max(radius * 21.0, 1.0))
            * np.exp(-((cross - streamer_offset_b) ** 2) / (2.0 * (streamer_width * 0.88) ** 2 + 1.0e-6))
            * tail_gate
        )
        pulse = 0.88 + 0.12 * math.sin(frame_index * 0.19 + source.seed * 0.017 + source_index)
        burst = _source_burst_envelope(source, frame_index)
        phase = source.seed * 0.031 + frame_index * 0.045
        lane = np.sin(cross / max(radius * 0.78, 1.0) + along / max(radius * 5.2, 1.0) + phase)
        fine_lane = np.sin(cross / max(radius * 0.34, 1.0) - along / max(radius * 3.4, 1.0) + phase * 1.73)
        cellular = np.sin(along / max(radius * 7.4, 1.0) + cross / max(radius * 2.25, 1.0) + phase * 0.62)
        filament_gain = np.clip(0.92 + 0.14 * lane + 0.06 * fine_lane, 0.62, 1.24)
        hole_cut = 1.0 - 0.14 * _smoothstep(0.16, 0.90, cellular) * along_frac
        addition = source.strength * source.smoke_rate * layer_weight * pulse * (
            0.016 * (1.0 - 0.28 * altitude) * core
            + 0.062 * (1.0 - 0.02 * altitude) * plume_tail
            + 0.0058 * (1.0 + 2.20 * altitude) * broad_veil
            + 0.0018 * (1.0 + 3.10 * altitude) * regional_sheet
            + 0.0220 * (1.0 + 0.48 * altitude) * (streamer_a + streamer_b)
        )
        addition = (addition * burst * filament_gain * hole_cut).astype(np.float32)
        out_density[y0:y1, x0:x1] += addition
        plume_age = np.clip(
            6.5 + 14.0 * altitude + np.maximum(along, 0.0) / max(radius * (0.95 + 0.42 * altitude), 1.0),
            0.0,
            HYBRID_SMOKE_MAX_AGE_FRAMES * 0.72,
        ).astype(np.float32)
        out_age_mass[y0:y1, x0:x1] += addition * plume_age

    return out_density, out_age_mass


class HybridSmokeSimulator:
    def __init__(
        self,
        map_size: tuple[int, int],
        sources: list[HybridSmokeSource],
        seed: int = HYBRID_SMOKE_SEED,
        hrrr_guidance: HrrrSmokeGuidance | None = None,
        guidance_cadence_frames: float = 12.0,
    ) -> None:
        width, height = map(int, map_size)
        if width <= 8 or height <= 8:
            raise ValueError("hybrid smoke map must be larger than 8x8")
        self.map_size = (width, height)
        self.sources = list(sources)
        self.seed = int(seed)
        self.hrrr_guidance = hrrr_guidance
        self.guidance_cadence_frames = max(float(guidance_cadence_frames), 1.0)
        self.density = np.zeros((height, width), dtype=np.float32)
        self.age_mass = np.zeros((height, width), dtype=np.float32)
        self.layer_density = np.zeros((HYBRID_SMOKE_LAYER_COUNT, height, width), dtype=np.float32)
        self.layer_age_mass = np.zeros((HYBRID_SMOKE_LAYER_COUNT, height, width), dtype=np.float32)
        self.residual_haze = np.zeros((height, width), dtype=np.float32)
        self.previous_density = self.density.copy()
        self.previous_age_mass = self.age_mass.copy()
        self.previous_layer_density = self.layer_density.copy()
        self.previous_layer_age_mass = self.layer_age_mass.copy()
        self.previous_residual_haze = self.residual_haze.copy()
        self.frame_index = 0
        self._border = _hybrid_border_fade(self.density.shape)
        if self.sources:
            weights = np.asarray([max(source.strength, 0.01) for source in self.sources], dtype=np.float32)
            xs = np.asarray([source.x for source in self.sources], dtype=np.float32)
            ys = np.asarray([source.y for source in self.sources], dtype=np.float32)
            total = max(float(np.sum(weights)), 1.0e-6)
            self._source_xy = (float(np.sum(xs * weights) / total), float(np.sum(ys * weights) / total))
        else:
            self._source_xy = (width * 0.5, height * 0.5)

    def _hrrr_guidance_density(self, frame_index: int) -> np.ndarray:
        guidance = self.hrrr_guidance
        if guidance is None or not guidance.frames:
            return _procedural_hrrr_smoke_guidance(self.density.shape, frame_index, self.seed, self._source_xy)
        if len(guidance.frames) == 1:
            return guidance.frames[0].astype(np.float32, copy=False)
        position = np.clip(float(frame_index) / self.guidance_cadence_frames, 0.0, float(len(guidance.frames) - 1))
        lo = int(math.floor(position))
        hi = min(lo + 1, len(guidance.frames) - 1)
        frac = np.float32(position - lo)
        return (
            guidance.frames[lo].astype(np.float32, copy=False) * (1.0 - frac)
            + guidance.frames[hi].astype(np.float32, copy=False) * frac
        ).astype(np.float32)

    def _set_layers(self, layer_density: np.ndarray, layer_age_mass: np.ndarray) -> None:
        layer_density = np.clip(np.asarray(layer_density, dtype=np.float32), 0.0, 6.0)
        layer_age_mass = np.clip(np.asarray(layer_age_mass, dtype=np.float32), 0.0, None)
        layer_density = np.where(layer_density >= 1.0e-7, layer_density, 0.0).astype(np.float32)
        layer_age_mass = np.where(layer_density > 0.0, layer_age_mass, 0.0).astype(np.float32)
        self.layer_density = layer_density
        self.layer_age_mass = layer_age_mass
        self.density = np.clip(np.sum(layer_density, axis=0), 0.0, 6.0).astype(np.float32)
        self.age_mass = np.clip(np.sum(layer_age_mass, axis=0), 0.0, None).astype(np.float32)
        self.age_mass = np.where(self.density > 0.0, self.age_mass, 0.0).astype(np.float32)

    def step(self, frame_index: int | None = None) -> HybridSmokeState:
        frame = self.frame_index if frame_index is None else int(frame_index)
        x, y = _pixel_grids(self.density.shape)
        self.previous_density = self.density.copy()
        self.previous_age_mass = self.age_mass.copy()
        self.previous_layer_density = self.layer_density.copy()
        self.previous_layer_age_mass = self.layer_age_mass.copy()
        self.previous_residual_haze = self.residual_haze.copy()

        next_density_layers: list[np.ndarray] = []
        next_age_layers: list[np.ndarray] = []
        for layer_index in range(HYBRID_SMOKE_LAYER_COUNT):
            altitude = _hybrid_layer_altitude(layer_index)
            wind = _hybrid_layer_wind_vector(layer_index)
            u, v = _hybrid_wind_field(frame, self.density.shape, self.seed, layer_index=layer_index)
            density = _bilinear_sample(self.layer_density[layer_index], x - u, y - v)
            age_mass = _bilinear_sample(self.layer_age_mass[layer_index], x - u, y - v)
            age = np.divide(age_mass, density, out=np.zeros_like(density), where=density > 1.0e-7)
            age = np.where(density > 1.0e-7, age + 1.0, 0.0).astype(np.float32)

            old_smoke_decay = 0.028 + 0.015 * altitude
            base_decay = 0.991 - 0.004 * altitude
            age_decay = 1.0 - old_smoke_decay * _smoothstep(150.0, HYBRID_SMOKE_MAX_AGE_FRAMES + 24.0, age)
            density = np.clip(density * base_decay * age_decay, 0.0, 6.0)
            age_mass = density * age

            diffusion_mix = 0.108 + 0.082 * altitude
            blurred_density = _box_blur_3x3(density, passes=1)
            blurred_age_mass = _box_blur_3x3(age_mass, passes=1)
            density = density * (1.0 - diffusion_mix) + blurred_density * diffusion_mix
            age_mass = age_mass * (1.0 - diffusion_mix) + blurred_age_mass * diffusion_mix
            density = _hybrid_downwind_stream(density, amount=0.150 + 0.064 * altitude, wind=wind)
            age_mass = _hybrid_downwind_stream(age_mass, amount=0.150 + 0.064 * altitude, wind=wind)
            density = _hybrid_crosswind_spread(density, amount=0.050 + 0.055 * altitude, wind=wind)
            age_mass = _hybrid_crosswind_spread(age_mass, amount=0.050 + 0.055 * altitude, wind=wind)
            local_age = np.divide(age_mass, density, out=np.zeros_like(density), where=density > 1.0e-7)
            age_shear = (0.34 + 0.20 * altitude) * _smoothstep(10.0, 118.0, local_age)
            if np.any(age_shear > 0.001):
                density = _bilinear_sample(density, x - wind[0] * age_shear, y - wind[1] * age_shear)
                age_mass = _bilinear_sample(age_mass, x - wind[0] * age_shear, y - wind[1] * age_shear)
            density, age_mass = _inject_hybrid_sources(
                density,
                age_mass,
                self.sources,
                frame,
                layer_index=layer_index,
                layer_count=HYBRID_SMOKE_LAYER_COUNT,
            )
            next_density_layers.append(density)
            next_age_layers.append(age_mass)

        layer_density = np.stack(next_density_layers, axis=0).astype(np.float32)
        layer_age_mass = np.stack(next_age_layers, axis=0).astype(np.float32)
        aggregate_density = np.clip(np.sum(layer_density, axis=0), 0.0, 6.0).astype(np.float32)

        guidance = self._hrrr_guidance_density(frame)
        broad_guidance = _pil_blur_float(guidance, max(2.0, min(self.density.shape) / 78.0))
        source_proximity = _smoothstep(0.002, 0.064, aggregate_density)
        regional_presence = _smoothstep(0.020, 0.46, broad_guidance)
        if self.hrrr_guidance is None:
            source_gate = 0.14 + 0.86 * source_proximity
            guided_veil = (0.0048 * guidance + 0.0022 * broad_guidance) * source_gate * regional_presence
            density_gain = 0.980 + 0.045 * guidance * source_gate
        else:
            guided_veil = (0.024 * guidance + 0.015 * broad_guidance) * (0.24 + 0.76 * source_proximity)
            density_gain = 0.948 + HRRR_SMOKE_GUIDANCE_STRENGTH * guidance
        age_hint = HYBRID_SMOKE_MAX_AGE_FRAMES * (0.44 + 0.24 * _smoothstep(0.0, 1.0, broad_guidance))
        guidance_layer_weights = np.asarray((0.18, 0.39, 0.43), dtype=np.float32)
        guidance_layer_weights /= max(float(np.sum(guidance_layer_weights)), 1.0e-6)
        for layer_index in range(HYBRID_SMOKE_LAYER_COUNT):
            altitude = _hybrid_layer_altitude(layer_index)
            layer_guided_veil = guided_veil * guidance_layer_weights[layer_index]
            layer_density[layer_index] = np.clip(
                layer_density[layer_index] * density_gain + layer_guided_veil,
                0.0,
                6.0,
            )
            layer_age_mass[layer_index] = (
                layer_age_mass[layer_index] * density_gain
                + layer_guided_veil * age_hint * (0.92 + 0.24 * altitude)
            )

        layer_density = np.clip(layer_density * self._border[None, :, :], 0.0, 6.0).astype(np.float32)
        layer_age_mass = np.clip(layer_age_mass * self._border[None, :, :], 0.0, None).astype(np.float32)
        self._set_layers(layer_density, layer_age_mass)
        self._update_residual_haze(frame)
        self.frame_index = frame + 1
        return self.state()

    def _update_residual_haze(self, frame_index: int) -> None:
        x, y = _pixel_grids(self.density.shape)
        layer_index = max(0, HYBRID_SMOKE_LAYER_COUNT - 1)
        wind = _hybrid_layer_wind_vector(layer_index)
        u, v = _hybrid_wind_field(
            float(frame_index) * 0.56 + 31.0,
            self.density.shape,
            self.seed + 9973,
            layer_index=layer_index,
        )
        advected = _bilinear_sample(self.residual_haze, x - u * 0.42, y - v * 0.42)
        advected = _hybrid_crosswind_spread(advected, amount=0.040, wind=wind)

        age = np.divide(self.age_mass, self.density, out=np.zeros_like(self.density), where=self.density > 1.0e-7)
        old_smoke = self.density * _smoothstep(28.0, HYBRID_SMOKE_MAX_AGE_FRAMES * 0.68, age)
        high_slab = self.layer_density[layer_index] if self.layer_density.size else self.density
        haze_feed = np.clip(old_smoke * 0.0125 + high_slab * 0.0050 + self.density * 0.0028, 0.0, 1.0)
        broad_feed = _pil_blur_float(haze_feed, max(5.5, min(self.density.shape) / 42.0))
        regional_feed = _pil_blur_float(haze_feed, max(12.0, min(self.density.shape) / 16.0)) * 0.22
        texture = _pil_blur_float(
            _advected_smoke_texture(self.density.shape, frame_index, self.seed + 12829),
            max(7.0, min(self.density.shape) / 38.0),
        )
        injected = (broad_feed + regional_feed) * np.clip(0.66 + 0.34 * texture, 0.42, 1.06)
        residual = advected * 0.993 + injected
        residual = _hybrid_downwind_stream(residual, amount=0.055, wind=wind)
        residual = _pil_blur_float(np.clip(residual, 0.0, 1.15), 1.65)
        self.residual_haze = np.clip(residual * self._border, 0.0, 1.15).astype(np.float32)

    def interpolated_state(self, alpha: float = 1.0) -> HybridSmokeState:
        t = float(np.clip(alpha, 0.0, 1.0))
        layer_density = self.previous_layer_density * (1.0 - t) + self.layer_density * t
        layer_age_mass = self.previous_layer_age_mass * (1.0 - t) + self.layer_age_mass * t
        residual_haze = self.previous_residual_haze * (1.0 - t) + self.residual_haze * t
        density = np.clip(np.sum(layer_density, axis=0), 0.0, 6.0).astype(np.float32)
        age_mass = np.clip(np.sum(layer_age_mass, axis=0), 0.0, None).astype(np.float32)
        age_mass = np.where(density > 0.0, age_mass, 0.0).astype(np.float32)
        return HybridSmokeState(
            density=density.copy(),
            age_mass=age_mass.copy(),
            layer_density=tuple(layer.copy() for layer in layer_density),
            layer_age_mass=tuple(layer.copy() for layer in layer_age_mass),
            residual_haze=residual_haze.astype(np.float32, copy=True),
        )

    def state(self) -> HybridSmokeState:
        return HybridSmokeState(
            density=self.density.copy(),
            age_mass=self.age_mass.copy(),
            layer_density=tuple(layer.copy() for layer in self.layer_density),
            layer_age_mass=tuple(layer.copy() for layer in self.layer_age_mass),
            residual_haze=self.residual_haze.copy(),
        )


def _hybrid_smoke_field_rgba(
    state: HybridSmokeState,
    frame_index: int,
    seed: int = HYBRID_SMOKE_SEED,
    alpha_multiplier: float = 1.0,
    color_bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    density = np.clip(np.asarray(state.density, dtype=np.float32), 0.0, None)
    age_mass = np.asarray(state.age_mass, dtype=np.float32)
    if density.ndim != 2 or age_mass.shape != density.shape:
        raise ValueError("density and age_mass must be matching 2D arrays")

    height, width = density.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    if not np.any(density > 0.0):
        return rgba

    fine = _pil_blur_float(density, 1.8)
    medium = _pil_blur_float(density, 6.5)
    broad = _pil_blur_float(density, 18.0)
    sheet = np.clip(density * 0.50 + fine * 0.30 + medium * 0.15 + broad * 0.05, 0.0, None)
    positive = sheet[sheet > 0.0]
    scale_value = max(float(np.percentile(positive, 98.8)) if positive.size else 1.0, 1.0e-5)
    norm = np.clip(sheet / (scale_value * 1.08), 0.0, 1.74)

    age = np.divide(age_mass, density, out=np.zeros_like(density), where=density > 1.0e-7)
    age = _pil_blur_float(age * (density > 0.0), 1.8)
    age_frac = np.clip(age / max(HYBRID_SMOKE_MAX_AGE_FRAMES, 1.0), 0.0, 1.0)
    age_alpha = _hybrid_lifecycle_alpha(age)
    coverage = _smoothstep(0.038, 0.275, sheet)

    texture = _advected_smoke_texture(density.shape, frame_index, seed)
    broad_texture = _pil_blur_float(texture, 7.8)
    detail_texture = texture - _pil_blur_float(texture, 2.80)
    x, y = _pixel_grids(density.shape)
    flow_scale = min(width, height) / 408.0
    wind = np.array([1.0, -0.42], dtype=np.float32)
    wind /= max(float(np.linalg.norm(wind)), 1.0e-6)
    cross_dir = np.array([-wind[1], wind[0]], dtype=np.float32)
    flow_along = x * wind[0] + y * wind[1]
    flow_cross = x * cross_dir[0] + y * cross_dir[1]
    streaks = 0.5 + 0.5 * np.sin(
        flow_cross / max(10.5 * flow_scale, 1.0)
        + flow_along / max(64.0 * flow_scale, 1.0)
        + frame_index * 0.019
        + seed * 0.011
    )
    ridge_density = np.clip(fine - medium * 0.52, 0.0, None)
    ridge_positive = ridge_density[ridge_density > 0.0]
    ridge_scale = max(float(np.percentile(ridge_positive, 96.5)) if ridge_positive.size else 1.0, 1.0e-5)
    ridge_norm = np.clip(ridge_density / ridge_scale, 0.0, 1.45)
    holes = _smoothstep(0.12, 0.70, 1.0 - broad_texture) * (1.0 - 0.22 * _smoothstep(0.25, 1.15, ridge_norm))
    holes = np.maximum(holes, 0.28 * _smoothstep(0.74, 0.99, streaks) * _smoothstep(0.05, 0.44, norm))
    edge_weight = _smoothstep(0.030, 0.50, norm) * (1.0 - 0.20 * _smoothstep(1.02, 1.72, norm))
    texture_gain = 0.92 + 0.18 * (broad_texture - 0.5) + 0.08 * detail_texture + 0.05 * (streaks - 0.5)
    texture_gain *= 1.0 - 0.28 * holes * edge_weight
    texture_gain = np.clip(texture_gain, 0.56, 1.16)
    filament_mask = 0.86 + 0.16 * _smoothstep(-0.020, 0.092, detail_texture) + 0.08 * ridge_norm
    filament_mask *= 1.0 - 0.20 * holes * edge_weight
    filament_mask = np.clip(filament_mask, 0.66, 1.16)

    alpha_shape = _smoothstep(0.012, 1.3, norm) ** 0.90
    source_core = _smoothstep(0.98, 1.62, norm) ** 1.16
    detail_density = np.clip(density - medium * 0.62, 0.0, None)
    detail_positive = detail_density[detail_density > 0.0]
    detail_scale = max(float(np.percentile(detail_positive, 97.0)) if detail_positive.size else 1.0, 1.0e-5)
    detail_norm = np.clip(detail_density / detail_scale, 0.0, 1.55)
    fresh_band = _smoothstep(0.08, 0.90, detail_norm) * (1.0 - _smoothstep(82.0, 170.0, age))
    source_core = np.maximum(source_core, fresh_band * 0.48)
    haze_floor = 10.8 * coverage * (1.0 - _smoothstep(0.34, 1.08, norm))
    age_visibility = np.clip(0.34 + 0.66 * age_alpha + 0.12 * source_core, 0.0, 1.0)
    alpha = (
        (238.0 * alpha_shape + 172.0 * source_core + 20.0 * ridge_norm + haze_floor * 3.10)
        * coverage
        * age_visibility
        * texture_gain
        * filament_mask
    )
    alpha += 52.0 * fresh_band * filament_mask * np.clip(0.78 + 0.18 * texture, 0.62, 1.04)
    alpha *= 0.52 + 0.50 * _smoothstep(0.52, 1.42, norm)
    alpha *= 1.0 - 0.10 * holes * edge_weight * (1.0 - 0.30 * source_core)
    alpha *= _smoothstep(0.004, 0.070, broad + medium * 0.20)
    alpha *= float(alpha_multiplier)
    alpha = _pil_blur_float(alpha.astype(np.float32), 1.1)
    alpha = np.clip(alpha, 0.0, HYBRID_SMOKE_MAX_ALPHA)
    alpha = np.where(alpha >= 2.0, alpha, 0.0)

    density_t = _smoothstep(0.026, 1.20, norm)
    old_blue = np.array([96.0, 108.0, 126.0], dtype=np.float32)
    thin_gray = np.array([158.0, 164.0, 168.0], dtype=np.float32)
    milky = np.array([242.0, 238.0, 224.0], dtype=np.float32)
    age_t = _smoothstep(60.0, HYBRID_SMOKE_MAX_AGE_FRAMES * 0.85, age)
    base_rgb = old_blue * (1.0 - density_t[..., None]) + thin_gray * density_t[..., None]
    base_rgb = base_rgb * (1.0 - age_t[..., None] * 0.32) + old_blue * (age_t[..., None] * 0.32)
    # Charcoal blend for very old smoke
    charcoal = np.array([72.0, 78.0, 86.0], dtype=np.float32)
    charcoal_t = _smoothstep(0.65, 0.92, age_t)
    base_rgb = base_rgb * (1.0 - charcoal_t[..., None] * 0.45) + charcoal * (charcoal_t[..., None] * 0.45)
    source_mix = np.clip(source_core * 0.46 + fresh_band * 0.42 + ridge_norm * 0.10, 0.0, 0.84)
    base_rgb = base_rgb * (1.0 - source_mix[..., None]) + milky * source_mix[..., None]
    fresh = (1.0 - _smoothstep(16.0, 74.0, age)) * _smoothstep(0.18, 1.18, norm)
    base_rgb += fresh[..., None] * np.array([12.0, 11.0, 6.0], dtype=np.float32)
    base_rgb += (broad_texture[..., None] - 0.5) * 9.0 * coverage[..., None]
    base_rgb += np.asarray(color_bias, dtype=np.float32)[None, None, :]
    rgb = np.clip(base_rgb, 0.0, 245.0)
    visible = alpha > 0.0
    rgba[..., 0] = np.where(visible, rgb[..., 0], 0.0).astype(np.uint8)
    rgba[..., 1] = np.where(visible, rgb[..., 1], 0.0).astype(np.uint8)
    rgba[..., 2] = np.where(visible, rgb[..., 2], 0.0).astype(np.uint8)
    rgba[..., 3] = alpha.astype(np.uint8)
    return rgba


def _offset_rgba_layer(rgba: np.ndarray, dx: float, dy: float) -> np.ndarray:
    src = Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA")
    dst = Image.new("RGBA", src.size, (0, 0, 0, 0))
    ox = int(round(dx))
    oy = int(round(dy))
    width, height = src.size
    src_x0 = max(0, -ox)
    src_y0 = max(0, -oy)
    dst_x0 = max(0, ox)
    dst_y0 = max(0, oy)
    copy_w = min(width - src_x0, width - dst_x0)
    copy_h = min(height - src_y0, height - dst_y0)
    if copy_w > 0 and copy_h > 0:
        crop = src.crop((src_x0, src_y0, src_x0 + copy_w, src_y0 + copy_h))
        dst.alpha_composite(crop, (dst_x0, dst_y0))
    return np.asarray(dst, dtype=np.uint8)


def _premultiplied_over(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    bottom_f = np.asarray(bottom, dtype=np.float32) / 255.0
    top_f = np.asarray(top, dtype=np.float32) / 255.0
    bottom_a = bottom_f[..., 3:4]
    top_a = top_f[..., 3:4]
    out_a = top_a + bottom_a * (1.0 - top_a)
    out_rgb_premul = top_f[..., :3] * top_a + bottom_f[..., :3] * bottom_a * (1.0 - top_a)
    out_rgb = np.divide(out_rgb_premul, out_a, out=np.zeros_like(out_rgb_premul), where=out_a > 1.0e-6)
    out = np.zeros_like(bottom_f)
    out[..., :3] = out_rgb
    out[..., 3:4] = out_a
    out[..., 3] = np.minimum(out[..., 3], HYBRID_SMOKE_MAX_ALPHA / 255.0)
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def composite_main_smoke_maps(
    atmospheric_rgba: np.ndarray,
    physical_rgba: np.ndarray | None,
    *,
    atmospheric_alpha: float = 0.42,
    physical_alpha: float = 0.92,
) -> np.ndarray:
    blanket = _scale_rgba_alpha(atmospheric_rgba, atmospheric_alpha)
    if physical_rgba is None:
        return blanket
    detail = _scale_rgba_alpha(physical_rgba, physical_alpha)
    combined = _premultiplied_over(blanket, detail)
    combined[..., 3] = np.minimum(combined[..., 3], HYBRID_SMOKE_MAX_ALPHA).astype(np.uint8)
    return combined


def _residual_haze_rgba(
    residual_haze: np.ndarray | None,
    frame_index: int,
    seed: int = HYBRID_SMOKE_SEED,
) -> np.ndarray:
    haze = np.clip(np.asarray(residual_haze, dtype=np.float32), 0.0, None)
    if haze.ndim != 2:
        raise ValueError("residual haze must be a 2D array")
    height, width = haze.shape
    out = np.zeros((height, width, 4), dtype=np.uint8)
    if not np.any(haze > 1.0e-6):
        return out

    scale = min(width, height) / 408.0
    soft = _pil_blur_float(haze, max(2.4, 4.8 * scale))
    broad = _pil_blur_float(haze, max(8.0, 16.0 * scale))
    sheet = np.clip(soft * 0.64 + broad * 0.44, 0.0, None)
    positive = sheet[sheet > 0.0]
    scale_value = max(float(np.percentile(positive, 99.3)) if positive.size else 1.0, 1.0e-5)
    norm = np.clip(sheet / (scale_value * 1.18), 0.0, 1.0)

    texture = _pil_blur_float(
        _advected_smoke_texture(haze.shape, frame_index, seed + 23011),
        max(7.0, 12.5 * scale),
    )
    alpha_f = (_smoothstep(0.012, 0.88, norm) ** 0.86) * np.clip(0.70 + 0.25 * (texture - 0.5), 0.50, 0.94)
    alpha = _pil_blur_float(alpha_f * HYBRID_SMOKE_RESIDUAL_HAZE_MAX_ALPHA, max(0.65, 1.15 * scale))
    alpha = np.clip(alpha, 0.0, HYBRID_SMOKE_RESIDUAL_HAZE_MAX_ALPHA)
    alpha = np.where(alpha >= 2.0, alpha, 0.0)

    thin = np.array([94.0, 109.0, 126.0], dtype=np.float32)
    flat = np.array([139.0, 149.0, 153.0], dtype=np.float32)
    density_t = _smoothstep(0.035, 0.82, norm)
    rgb = thin * (1.0 - density_t[..., None]) + flat * density_t[..., None]
    rgb *= 0.86 + 0.08 * texture[..., None]
    out[..., :3] = np.where(alpha[..., None] > 0.0, np.clip(np.round(rgb), 0, 220), 0).astype(np.uint8)
    out[..., 3] = alpha.astype(np.uint8)
    return out


def hybrid_smoke_rgba(
    state: HybridSmokeState,
    frame_index: int,
    seed: int = HYBRID_SMOKE_SEED,
) -> np.ndarray:
    layers = state.layer_density
    layer_ages = state.layer_age_mass
    if not layers or not layer_ages or len(layers) != len(layer_ages):
        return _hybrid_smoke_field_rgba(state, frame_index, seed)

    layer_count = min(len(layers), len(layer_ages), HYBRID_SMOKE_LAYER_COUNT)
    density_shape = np.asarray(layers[0]).shape
    combined = np.zeros((density_shape[0], density_shape[1], 4), dtype=np.uint8)
    if state.residual_haze is not None:
        combined = _premultiplied_over(combined, _residual_haze_rgba(state.residual_haze, frame_index, seed))
    flow_scale = min(density_shape) / 408.0
    for layer_index in range(layer_count):
        altitude = _hybrid_layer_altitude(layer_index)
        color_bias = (
            7.0 - 13.0 * altitude,
            6.0 - 9.0 * altitude,
            -3.0 + 14.0 * altitude,
        )
        layer_state = HybridSmokeState(
            density=np.asarray(layers[layer_index], dtype=np.float32),
            age_mass=np.asarray(layer_ages[layer_index], dtype=np.float32),
        )
        layer_rgba = _hybrid_smoke_field_rgba(
            layer_state,
            frame_index,
            seed + 503 * layer_index,
            alpha_multiplier=HYBRID_SMOKE_RENDER_LAYER_ALPHA[layer_index],
            color_bias=color_bias,
        )
        wind = _hybrid_layer_wind_vector(layer_index)
        cross = np.array([-wind[1], wind[0]], dtype=np.float32)
        dx = (altitude - 0.35) * 3.8 * flow_scale + cross[0] * 2.3 * altitude * flow_scale
        dy = -altitude * 5.2 * flow_scale + cross[1] * 1.8 * (altitude - 0.5) * flow_scale
        combined = _premultiplied_over(combined, _offset_rgba_layer(layer_rgba, dx, dy))

    aggregate_boost = _hybrid_smoke_field_rgba(
        HybridSmokeState(density=state.density, age_mass=state.age_mass),
        frame_index,
        seed + 1709,
        alpha_multiplier=0.18,
        color_bias=(-4.0, -2.0, 4.0),
    )
    combined = _premultiplied_over(combined, aggregate_boost)
    combined[..., 3] = np.minimum(combined[..., 3], HYBRID_SMOKE_MAX_ALPHA).astype(np.uint8)
    return combined


def hybrid_fire_sources_rgba(
    sources: list[HybridSmokeSource],
    frame_index: int,
    map_size: tuple[int, int],
    *,
    glow_only: bool = False,
    bloom_scale: float = 1.0,
    core_alpha_scale: float = 1.0,
) -> np.ndarray:
    width, height = map(int, map_size)
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    halo = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    wide_halo = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    halo_draw = ImageDraw.Draw(halo, "RGBA")
    wide_draw = ImageDraw.Draw(wide_halo, "RGBA")
    for idx, source in enumerate(sources):
        if frame_index < source.start_frame or frame_index > source.end_frame:
            continue
        pulse = 0.72 + 0.28 * math.sin(frame_index * 0.44 + source.seed * 0.031 + idx)
        radius = max(0.75, source.radius_px * 0.22)
        halo_radius = radius * (3.4 + 2.1 * float(bloom_scale)) * max(0.70, float(bloom_scale))
        alpha = int(np.clip((70.0 + 74.0 * source.strength * source.heat) * pulse, 28, 205))
        wide_radius = halo_radius * (1.9 + 0.30 * float(bloom_scale))
        wide_draw.ellipse(
            (source.x - wide_radius, source.y - wide_radius, source.x + wide_radius, source.y + wide_radius),
            fill=(255, 98, 24, int(alpha * (0.048 + 0.038 * float(bloom_scale)))),
        )
        halo_draw.ellipse(
            (source.x - halo_radius, source.y - halo_radius, source.x + halo_radius, source.y + halo_radius),
            fill=(255, 108, 32, int(alpha * (0.13 + 0.08 * float(bloom_scale)))),
        )
        if glow_only:
            continue
        core_alpha = int(np.clip(alpha * float(core_alpha_scale), 0, 255))
        draw.ellipse(
            (source.x - radius, source.y - radius, source.x + radius, source.y + radius),
            fill=(255, 118, 28, core_alpha),
        )
        core = max(0.45, radius * 0.36)
        draw.ellipse(
            (source.x - core, source.y - core, source.x + core, source.y + core),
            fill=(255, 226, 104, min(255, core_alpha + 42)),
        )
    wide_halo = wide_halo.filter(
        ImageFilter.GaussianBlur(radius=max(3.0, min(width, height) / 75.0 * max(0.8, float(bloom_scale))))
    )
    halo = halo.filter(ImageFilter.GaussianBlur(radius=max(1.5, min(width, height) / 160.0 * max(0.8, float(bloom_scale)))))
    wide_halo.alpha_composite(halo)
    halo = wide_halo
    if not glow_only:
        halo.alpha_composite(layer)
    return np.asarray(halo, dtype=np.uint8)


def _premultiply_rgba_uint8(rgba: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgba, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise ValueError("RGBA array must have shape (H, W, 4)")
    out = arr.copy()
    out[..., :3] *= out[..., 3:4] / 255.0
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _unpremultiply_rgba_uint8(rgba: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgba, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise ValueError("RGBA array must have shape (H, W, 4)")
    alpha = arr[..., 3:4] / 255.0
    out = arr.copy()
    out[..., :3] = np.divide(out[..., :3], alpha, out=np.zeros_like(out[..., :3]), where=alpha > 1.0e-5)
    out[..., :3] = np.where(arr[..., 3:4] > 0.5, out[..., :3], 0.0)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def warp_map_layer_to_plate(rgba: np.ndarray, plate: TerrainPlate, size: tuple[int, int]) -> Image.Image:
    source = Image.fromarray(_premultiply_rgba_uint8(rgba), mode="RGBA")
    src_w, src_h = source.size
    coeffs = perspective_coeffs(
        [(0, 0), (src_w, 0), (src_w, src_h), (0, src_h)],
        plate.quad,
    )
    warped = source.transform(size, Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
    warped = warped.filter(ImageFilter.GaussianBlur(radius=max(0.16, size[0] / 4200.0)))
    return Image.fromarray(_unpremultiply_rgba_uint8(np.asarray(warped, dtype=np.uint8)), mode="RGBA")


def composite_atmospheric_smoke(base: Image.Image, smoke_layer: Image.Image) -> Image.Image:
    base_arr = np.asarray(base.convert("RGBA"), dtype=np.float32)
    smoke_arr = np.asarray(smoke_layer.convert("RGBA"), dtype=np.float32)
    alpha = smoke_arr[..., 3:4] / 255.0
    optical = np.clip(alpha * 0.98, 0.0, 1.0) ** 0.90
    veil = smoke_arr[..., :3] / 255.0
    terrain = base_arr[..., :3] / 255.0
    warm_signal = np.clip((terrain[..., 0:1] - terrain[..., 2:3]) * 1.55 + (terrain[..., 1:2] - terrain[..., 2:3]) * 0.38, 0.0, 1.0)
    source_transmission = 1.0 - 0.34 * _smoothstep(0.10, 0.48, warm_signal)
    transmittance = np.exp(-0.72 * optical * source_transmission)
    premul_smoke = veil * optical
    backscatter = np.array([0.65, 0.67, 0.66], dtype=np.float32)[None, None, :] * (0.17 * optical)
    glow_through = terrain * warm_signal * optical * 0.18
    lifted = terrain * transmittance + premul_smoke * 0.92 + backscatter + glow_through
    base_arr[..., :3] = np.clip(lifted * 255.0, 0.0, 255.0)
    base_arr[..., 3] = 255.0
    return Image.fromarray(base_arr.astype(np.uint8), mode="RGBA")


def make_smoke_domain() -> tuple[object, object, object, dict[str, object]]:
    if f3d_smoke is None:
        raise RuntimeError("forge3d.smoke is required for volume detail; rerun with --no-volume-detail")
    domain = f3d_smoke.SmokeDomain((96, 48, 52), voxel_size=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    emitter = f3d_smoke.SmokeEmitter(
        center=(18.0, 8.0, 24.0),
        radius=3.2,
        density_rate=2.25,
        temperature_rate=1.12,
        soot_rate=0.12,
        emission_rate=2.3,
        velocity=(7.4, 0.20, -0.36),
    )
    step = f3d_smoke.SmokeStepSettings(
        dt=0.10,
        density_decay=0.014,
        temperature_decay=0.22,
        velocity_damping=0.026,
        diffusion=0.0012,
        buoyancy=0.035,
        vorticity=0.32,
        pressure_iterations=14,
        turbulence_strength=0.76,
        turbulence_seed=2020,
        wind=(2.20, 0.00, -0.32),
    )
    render = f3d_smoke.SmokeRenderSettings(
        density_scale=1.34,
        extinction=1.62,
        scattering=1.02,
        absorption=0.28,
        phase_g=0.48,
        max_steps=144,
        self_shadow=True,
        shadow_steps=28,
        shadow_step_size=0.90,
        jitter_strength=0.45,
        thin_color=(0.48, 0.54, 0.61),
        dense_color=(0.93, 0.91, 0.82),
        soot_absorption=0.32,
        fire_glow=0.34,
    )
    camera = {
        "camera_pos": (38.0, 23.0, -104.0),
        "target": (46.0, 16.0, 24.0),
        "up": (0.0, 1.0, 0.0),
        "fovy_deg": 33.0,
        "sun_direction": (0.35, 0.78, -0.18),
        "source_world": (18.0, 8.0, 24.0),
    }
    return domain, emitter, step, {"render": render, **camera}


def project_smoke_source(camera: dict[str, object], width: int, height: int) -> tuple[float, float]:
    eye = np.asarray(camera["camera_pos"], dtype=np.float32)
    target = np.asarray(camera["target"], dtype=np.float32)
    up = np.asarray(camera["up"], dtype=np.float32)
    point = np.asarray(camera["source_world"], dtype=np.float32)
    forward = target - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-6)
    up /= max(float(np.linalg.norm(up)), 1e-6)
    side = np.cross(forward, up)
    side /= max(float(np.linalg.norm(side)), 1e-6)
    cam_up = np.cross(side, forward)
    rel = point - eye
    depth = max(float(np.dot(rel, forward)), 1e-3)
    focal = 1.0 / math.tan(math.radians(float(camera["fovy_deg"])) * 0.5)
    aspect = width / max(float(height), 1.0)
    sx = (float(np.dot(rel, side)) * focal / aspect / depth) * 0.5 + 0.5
    sy = 0.5 - (float(np.dot(rel, cam_up)) * focal / depth) * 0.5
    return sx * width, sy * height


def draw_fire(frame: Image.Image, xy: tuple[float, float], progress: float) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    x, y = xy
    pulse = 0.5 + 0.5 * math.sin(progress * math.tau * 6.0)
    scale = frame.width / WIDTH
    for radius, alpha in ((38, 42), (20, 84), (8, 205)):
        radius *= scale
        color = (255, int(116 + 70 * pulse), 18, int(alpha))
        draw.ellipse((x - radius, y - radius * 0.55, x + radius, y + radius * 0.55), fill=color)
    draw.ellipse((x - 4 * scale, y - 3 * scale, x + 4 * scale, y + 3 * scale), fill=(255, 232, 126, 245))


def draw_labels(frame: Image.Image) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    scale = min(frame.width / WIDTH, frame.height / HEIGHT)
    pad = max(18, int(round(38 * scale)))
    text_font = load_font(max(12, int(round(19 * scale))))
    small_font = load_font(max(10, int(round(15 * scale))))
    area_font = load_font(max(24, int(round(48 * scale))), bold=True)
    lines = ("Data: CAL FIRE perimeter, cached California DEM", "August Complex, 2020")
    for idx, line in enumerate(lines):
        y = pad + idx * max(18, int(round(25 * scale)))
        draw.text((pad + 1, y + 1), line, font=small_font, fill=(0, 0, 0, 180))
        draw.text((pad, y), line, font=small_font, fill=(234, 239, 235, 230))

    bottom = frame.height - pad - max(74, int(round(112 * scale)))
    draw.text((pad + 2, bottom + 2), "Area Burned:", font=text_font, fill=(0, 0, 0, 190))
    draw.text((pad, bottom), "Area Burned:", font=text_font, fill=(236, 239, 235, 235))
    value = f"{AUGUST_COMPLEX.final_area_ha/1000:.0f} k ha"
    value_y = bottom + max(24, int(round(38 * scale)))
    draw.text((pad + 2, value_y + 2), value, font=area_font, fill=(0, 0, 0, 200))
    draw.text((pad, value_y), value, font=area_font, fill=(248, 250, 246, 245))


def composite_volume_detail(
    base: Image.Image,
    smoke_rgba: np.ndarray,
    fire_xy: tuple[float, float],
    source_xy: tuple[float, float],
) -> Image.Image:
    smoke_arr = np.asarray(smoke_rgba, dtype=np.float32).copy()
    alpha = smoke_arr[..., 3:4] / 255.0
    smoke_lift = np.array([76.0, 77.0, 74.0], dtype=np.float32)
    smoke_arr[..., :3] = np.clip(smoke_arr[..., :3] * 1.26 + smoke_lift * alpha, 0.0, 255.0)
    smoke_arr[..., 3] = np.clip(smoke_arr[..., 3] * 0.54, 0.0, 116.0)
    smoke_image = Image.fromarray(smoke_arr.astype(np.uint8), mode="RGBA").filter(ImageFilter.GaussianBlur(radius=1.15))
    smoke_scale = 1.04
    scaled_size = (int(round(smoke_image.width * smoke_scale)), int(round(smoke_image.height * smoke_scale)))
    smoke_image = smoke_image.resize(scaled_size, Image.Resampling.BICUBIC)
    smoke_image = smoke_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    scaled_source_xy = (source_xy[0] * smoke_scale, source_xy[1] * smoke_scale)
    flipped_source_x = smoke_image.width - scaled_source_xy[0]
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    offset = (
        int(round(fire_xy[0] - flipped_source_x)),
        int(round(fire_xy[1] - scaled_source_xy[1] - base.height * 0.030)),
    )
    layer.alpha_composite(smoke_image, offset)
    out = base.copy()
    out.alpha_composite(layer)
    return out


def _scale_rgba_alpha(rgba: np.ndarray, scale: float) -> np.ndarray:
    out = np.asarray(rgba, dtype=np.uint8).copy()
    out[..., 3] = np.clip(out[..., 3].astype(np.float32) * float(scale), 0.0, 255.0).astype(np.uint8)
    return out


def _select_physical_sources(
    sources: list[HybridSmokeSource],
    max_sources: int,
) -> list[HybridSmokeSource]:
    if max_sources <= 0:
        return []
    ranked = sorted(
        enumerate(sources),
        key=lambda item: (
            item[1].strength * item[1].smoke_rate * (1.0 + 0.20 * item[1].heat),
            -item[0],
        ),
        reverse=True,
    )
    selected_indexes: list[int] = []
    for idx, _source in ranked[: max(0, max_sources - 6)]:
        selected_indexes.append(idx)
    selected_indexes.extend(range(min(6, len(sources))))
    seen: set[int] = set()
    ordered: list[HybridSmokeSource] = []
    for idx in sorted(selected_indexes):
        if idx in seen or idx >= len(sources):
            continue
        seen.add(idx)
        ordered.append(sources[idx])
    return ordered[:max_sources]


def _native_physical_smoke_available() -> bool:
    if f3d_smoke is None:
        return False
    native_available = getattr(f3d_smoke, "native_smoke_available", lambda: False)
    if not bool(native_available()):
        return False
    required = ("SmokeDomain", "SmokeEmitter", "SmokeStepSettings", "SmokeRenderSettings")
    if not all(hasattr(f3d_smoke, name) for name in required):
        return False
    return bool(hasattr(f3d_smoke.SmokeDomain, "render_projection_rgba"))


def _physical_backend(requested: str) -> str:
    backend = str(requested).lower()
    if backend not in {"auto", "native", "numpy"}:
        raise ValueError("physical smoke backend must be 'auto', 'native', or 'numpy'")
    if backend == "numpy":
        return "numpy"
    if _native_physical_smoke_available():
        return "native"
    if backend == "native":
        raise RuntimeError("native physical smoke backend requested but forge3d.smoke native projection is unavailable")
    return "numpy"


def _make_physical_domain(dims: tuple[int, int, int], backend: str) -> object:
    if backend == "native":
        assert f3d_smoke is not None
        return f3d_smoke.SmokeDomain(dims, sparse_threshold=1.0e-6)
    return NumpyPhysicalSmokeDomain(dims, sparse_threshold=1.0e-6)


def _make_physical_step_settings(backend: str, substep_count: int, seed: int) -> object:
    kwargs = {
        "dt": 0.22 / substep_count,
        "density_decay": 0.0038,
        "temperature_decay": 0.20,
        "velocity_damping": 0.018,
        "diffusion": 0.0018,
        "buoyancy": 0.0045,
        "vorticity": 0.88,
        "pressure_iterations": 14,
        "turbulence_strength": 1.20,
        "turbulence_seed": int(seed),
        "terrain_collision": True,
        "boundary_damping": 0.026,
        "wind": (0.278, 0.000, -0.106),
    }
    if backend == "native":
        assert f3d_smoke is not None
        return f3d_smoke.SmokeStepSettings(
            **kwargs,
            mac_cormack=True,
            mass_conservation=False,
        )
    return PhysicalSmokeStepSettings3D(**kwargs)


def _make_physical_render_settings(backend: str) -> object:
    if backend == "native":
        assert f3d_smoke is not None
        return f3d_smoke.SmokeRenderSettings(
            density_scale=1.32,
            extinction=1.30,
            scattering=1.24,
            absorption=0.26,
            phase_g=0.24,
            step_size=0.62,
            max_steps=150,
            self_shadow=True,
            shadow_steps=32,
            shadow_step_size=0.82,
            jitter_strength=0.36,
            exposure=1.62,
            thin_color=(0.50, 0.54, 0.58),
            dense_color=(0.93, 0.91, 0.82),
            soot_absorption=0.18,
            fire_glow=0.24,
        )
    return PhysicalSmokeRenderSettings3D(
        density_scale=1.32,
        extinction=1.30,
        exposure=1.62,
        scattering=1.24,
        phase_g=0.24,
        soot_absorption=0.18,
        fire_glow=0.24,
    )


def make_physical_main_smoke(
    sources: list[HybridSmokeSource],
    map_size: tuple[int, int],
    *,
    dims: tuple[int, int, int] = PHYSICAL_SMOKE_DIMS,
    render_size: tuple[int, int] = PHYSICAL_SMOKE_RENDER_SIZE,
    max_sources: int = PHYSICAL_SMOKE_MAX_SOURCES,
    substeps: int = 1,
    backend: str = "auto",
    seed: int = HYBRID_SMOKE_SEED,
) -> PhysicalSmokeMainEffect | None:
    nx, ny, nz = (max(8, int(v)) for v in dims)
    render_w, render_h = (max(16, int(v)) for v in render_size)
    substep_count = max(1, int(substeps))
    resolved_backend = _physical_backend(backend)
    domain = _make_physical_domain((nx, ny, nz), resolved_backend)
    step = _make_physical_step_settings(resolved_backend, substep_count, seed)
    render = _make_physical_render_settings(resolved_backend)
    selected_sources = _select_physical_sources(sources, max_sources=max_sources)
    if not selected_sources:
        return None
    return PhysicalSmokeMainEffect(
        domain=domain,
        step_settings=step,
        render_settings=render,
        sources=selected_sources,
        map_size=tuple(map(int, map_size)),
        dims=(nx, ny, nz),
        render_size=(render_w, render_h),
        substeps=substep_count,
        backend=resolved_backend,
        seed=int(seed),
    )


def _physical_emitters_for_frame(effect: PhysicalSmokeMainEffect, frame_index: int) -> list[PhysicalSmokeEmitter3D]:
    nx, ny, nz = effect.dims
    map_w, map_h = effect.map_size
    emitters: list[PhysicalSmokeEmitter3D] = []
    wind = _hybrid_layer_wind_vector(1)
    cross = np.array([-wind[1], wind[0]], dtype=np.float32)
    for source_index, source in enumerate(effect.sources):
        if frame_index < source.start_frame or frame_index > source.end_frame:
            continue
        burst = _source_burst_envelope(source, frame_index)
        if burst < 0.08:
            continue
        sx = float(source.x) / max(float(map_w - 1), 1.0) * float(nx - 1)
        sz = float(source.y) / max(float(map_h - 1), 1.0) * float(nz - 1)
        altitude = float(np.clip(0.18 + 0.34 * source.altitude_bias + 0.06 * source.heat, 0.12, 0.62))
        sy = 1.8 + altitude * float(ny - 4)
        scale_x = float(nx) / max(float(map_w), 1.0)
        scale_z = float(nz) / max(float(map_h), 1.0)
        radius = max(1.28, float(source.radius_px) * math.sqrt(scale_x * scale_z) * (0.82 + 0.18 * source.heat))
        flicker = 0.82 + 0.18 * math.sin(frame_index * 0.31 + source.seed * 0.023 + source_index)
        lateral = (
            0.72 * math.sin(frame_index * 0.041 + source.seed * 0.017)
            + 0.28 * math.sin(frame_index * 0.013 + source_index * 1.41)
        )
        velocity = (
            float(wind[0] * (1.92 + 0.34 * source.heat) + cross[0] * lateral),
            float(0.006 + 0.014 * source.heat),
            float(wind[1] * (1.92 + 0.32 * source.heat) + cross[1] * lateral),
        )
        density_rate = float(0.96 * source.strength * source.smoke_rate * burst * flicker)
        temperature_rate = float(0.18 * source.heat * burst)
        soot_rate = float(0.018 + 0.010 * source.heat)
        emitters.append(
            PhysicalSmokeEmitter3D(
                center=(sx, sy, sz),
                radius=radius,
                density_rate=density_rate,
                temperature_rate=temperature_rate,
                soot_rate=soot_rate,
                humidity_rate=0.10,
                emission_rate=0.55 * source.heat,
                velocity=velocity,
            )
        )
        if source_index < 6 and burst > 0.18:
            streamer_nodes = (
                (0.48, 0.18, 0.28, 0.18),
                (0.96, 0.17, 0.24, -0.30),
                (1.58, 0.22, 0.20, 0.46),
                (2.34, 0.32, 0.13, -0.54),
            )
            for streamer_index, (distance, radius_scale, rate_scale, bend_sign) in enumerate(streamer_nodes):
                phase = frame_index * (0.061 + streamer_index * 0.017) + source.seed * 0.013
                bend = bend_sign + 0.34 * math.sin(phase * 0.79 + streamer_index)
                lateral_stream = bend * radius * (0.72 + 0.14 * streamer_index)
                vertical_stream = math.cos(phase * 0.83 + 0.7) * (0.12 + 0.03 * streamer_index)
                stream_center = (
                    float(np.clip(sx + wind[0] * radius * distance + cross[0] * lateral_stream, 1.0, nx - 2.0)),
                    float(np.clip(sy + vertical_stream + 0.10 * streamer_index, 1.0, ny - 2.0)),
                    float(np.clip(sz + wind[1] * radius * distance + cross[1] * lateral_stream, 1.0, nz - 2.0)),
                )
                curl_push = 0.11 * math.cos(phase * 1.17 + source_index)
                stream_velocity = (
                    float(velocity[0] * (1.28 + 0.05 * streamer_index) + cross[0] * curl_push),
                    float(velocity[1] * (0.36 + 0.04 * streamer_index)),
                    float(velocity[2] * (1.28 + 0.05 * streamer_index) + cross[1] * curl_push),
                )
                emitters.append(
                    PhysicalSmokeEmitter3D(
                        center=stream_center,
                        radius=max(0.38, radius * radius_scale),
                        density_rate=density_rate * rate_scale,
                        temperature_rate=temperature_rate * (0.34 - 0.06 * streamer_index),
                        soot_rate=soot_rate * 0.55,
                        humidity_rate=0.10,
                        emission_rate=0.28 * source.heat,
                        velocity=stream_velocity,
                    )
                )
            if source_index < 6 and burst > 0.24:
                hook_phase = frame_index * 0.027 + source.seed * 0.019 + source_index * 1.71
                hook_side = -1.0 if math.sin(hook_phase) < 0.0 else 1.0
                for hook_index in range(3):
                    arc_t = (hook_index + 1.0) / 3.0
                    arc_angle = hook_phase + hook_side * (0.55 + 1.10 * arc_t)
                    arc_distance = radius * (1.35 + 0.62 * hook_index)
                    lateral_arc = math.sin(arc_angle) * radius * (0.88 + 0.32 * hook_index)
                    arc_center = (
                        float(np.clip(sx + wind[0] * arc_distance + cross[0] * lateral_arc, 1.0, nx - 2.0)),
                    float(np.clip(sy + 0.10 * hook_index + 0.10 * math.cos(arc_angle), 1.0, ny - 2.0)),
                        float(np.clip(sz + wind[1] * arc_distance + cross[1] * lateral_arc, 1.0, nz - 2.0)),
                    )
                    curl_velocity = 0.13 * hook_side * math.cos(arc_angle)
                    emitters.append(
                        PhysicalSmokeEmitter3D(
                            center=arc_center,
                            radius=max(0.48, radius * (0.34 + 0.08 * hook_index)),
                            density_rate=density_rate * (0.18 - 0.045 * hook_index),
                            temperature_rate=temperature_rate * (0.17 - 0.038 * hook_index),
                            soot_rate=soot_rate * 0.34,
                            humidity_rate=0.12,
                            emission_rate=0.16 * source.heat,
                            velocity=(
                                float(velocity[0] * (0.98 - 0.06 * hook_index) + cross[0] * curl_velocity),
                                float(velocity[1] * 0.20),
                                float(velocity[2] * (0.98 - 0.06 * hook_index) + cross[1] * curl_velocity),
                            ),
                        )
                    )
    return emitters


def _native_physical_emitter(emitter: PhysicalSmokeEmitter3D) -> object:
    assert f3d_smoke is not None
    return f3d_smoke.SmokeEmitter(
        center=emitter.center,
        radius=emitter.radius,
        density_rate=emitter.density_rate,
        temperature_rate=emitter.temperature_rate,
        fuel_rate=0.0,
        soot_rate=emitter.soot_rate,
        humidity_rate=emitter.humidity_rate,
        emission_rate=emitter.emission_rate,
        velocity=emitter.velocity,
    )


def step_physical_main_smoke(effect: PhysicalSmokeMainEffect, frame_index: int) -> None:
    for _substep in range(max(1, effect.substeps)):
        emitters = _physical_emitters_for_frame(effect, frame_index)
        if effect.backend == "native":
            emitters = [_native_physical_emitter(emitter) for emitter in emitters]  # type: ignore[assignment]
        effect.domain.step(effect.step_settings, emitters)


def _henyey_greenstein_py(cos_theta: float, g: float) -> float:
    g = float(np.clip(g, -0.99, 0.99))
    denom = max(1.0 + g * g - 2.0 * g * float(np.clip(cos_theta, -1.0, 1.0)), 1.0e-4)
    return float((1.0 - g * g) / (4.0 * math.pi * denom**1.5))


def _physical_projection_view_direction() -> tuple[float, float, float]:
    view = np.asarray(PHYSICAL_SMOKE_VIEW_DIRECTION, dtype=np.float32)
    if view.shape != (3,):
        view = np.asarray((0.42, -0.68, 0.60), dtype=np.float32)
    parallax = max(0.25, float(PHYSICAL_SMOKE_PARALLAX_SCALE))
    view = np.array([view[0] * parallax, view[1], view[2] * parallax], dtype=np.float32)
    norm = max(float(np.linalg.norm(view)), 1.0e-6)
    view /= norm
    return float(view[0]), float(view[1]), float(view[2])


def _python_volume_light_transmittance(
    density: np.ndarray,
    soot: np.ndarray,
    layer_index: int,
    density_scale: float,
    extinction: float,
    soot_absorption: float,
    settings: object | None,
) -> np.ndarray:
    depth, altitude_count, width = density.shape
    z_grid, x_grid = np.mgrid[0:depth, 0:width].astype(np.float32)
    light_dir = np.asarray((0.34, 0.82, -0.22), dtype=np.float32)
    light_dir /= max(float(np.linalg.norm(light_dir)), 1.0e-6)
    shadow_steps = int(max(1, getattr(settings, "shadow_steps", 18)))
    shadow_step_size = float(getattr(settings, "shadow_step_size", 1.15))
    if shadow_step_size <= 0.0:
        shadow_step_size = 1.15
    optical_depth = np.zeros((depth, width), dtype=np.float32)
    base_y = np.full((depth, width), float(layer_index), dtype=np.float32)
    for step_index in range(1, shadow_steps + 1):
        distance = float(step_index) * shadow_step_size
        sample_x = x_grid + light_dir[0] * distance
        sample_y = base_y + light_dir[1] * distance
        sample_z = z_grid + light_dir[2] * distance
        sample_density = _trilinear_sample_volume(density, sample_x, sample_y, sample_z)
        sample_soot = _trilinear_sample_volume(soot, sample_x, sample_y, sample_z)
        optical_depth += (
            sample_density
            * density_scale
            * extinction
            * (1.0 + sample_soot * soot_absorption * 0.85)
            * shadow_step_size
        ).astype(np.float32)
    return np.exp(-np.clip(optical_depth, 0.0, 9.0)).astype(np.float32)


def _python_volume_shadow_grid(
    density: np.ndarray,
    soot: np.ndarray,
    particle_age: np.ndarray | None,
    light_dir: np.ndarray,
    density_scale: float,
    extinction: float,
    soot_absorption: float,
    settings: object | None,
) -> np.ndarray:
    depth, altitude_count, width = density.shape
    grid_x, grid_y, grid_z = _volume_grids(density.shape)
    shadow_steps = int(max(1, getattr(settings, "shadow_steps", 18)))
    shadow_step_size = float(getattr(settings, "shadow_step_size", 1.15))
    if shadow_step_size <= 0.0:
        shadow_step_size = 1.15
    optical_depth = np.zeros_like(density, dtype=np.float32)
    for step_index in range(1, shadow_steps + 1):
        distance = float(step_index) * shadow_step_size
        sample_x = grid_x + float(light_dir[0]) * distance
        sample_y = grid_y + float(light_dir[1]) * distance
        sample_z = grid_z + float(light_dir[2]) * distance
        sample_density = _trilinear_sample_volume(density, sample_x, sample_y, sample_z)
        sample_soot = _trilinear_sample_volume(soot, sample_x, sample_y, sample_z)
        if particle_age is not None:
            sample_age = _trilinear_sample_volume(particle_age, sample_x, sample_y, sample_z)
            age_t = _smoothstep(1.6, 17.0, sample_age)
        else:
            age_t = np.zeros_like(sample_density, dtype=np.float32)
        concentration_gate = 0.50 + 0.50 * _smoothstep(0.045, 0.34, sample_density)
        optical_depth += (
            sample_density
            * density_scale
            * (1.0 - 0.58 * age_t)
            * concentration_gate
            * extinction
            * (1.0 + sample_soot * soot_absorption * 0.85)
            * shadow_step_size
        ).astype(np.float32)
    return np.exp(-np.clip(optical_depth, 0.0, 9.0)).astype(np.float32)


def _python_ray_box_intersection(
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


def _python_projected_volume_raymarch(
    domain: object,
    render_size: tuple[int, int],
    frame_index: int,
    seed: int,
    settings: object | None = None,
) -> np.ndarray:
    density = np.clip(np.asarray(domain.to_density_numpy(), dtype=np.float32), 0.0, None)
    if density.ndim != 3:
        raise ValueError("physical smoke density must be a 3D array")

    def _optional_volume(name: str, *, clip_nonnegative: bool = True) -> np.ndarray:
        getter = getattr(domain, name, None)
        if getter is None:
            return np.zeros_like(density, dtype=np.float32)
        try:
            arr = np.asarray(getter(), dtype=np.float32)
        except Exception:
            return np.zeros_like(density, dtype=np.float32)
        if arr.shape != density.shape:
            return np.zeros_like(density, dtype=np.float32)
        if clip_nonnegative:
            arr = np.clip(arr, 0.0, None)
        return arr.astype(np.float32)

    temperature = _optional_volume("to_temperature_numpy")
    soot = _optional_volume("to_soot_numpy")
    emission = _optional_volume("to_emission_numpy")
    particle_age = _optional_volume("to_particle_age_numpy", clip_nonnegative=False)
    density_scale = float(getattr(settings, "density_scale", 1.18))
    extinction = float(getattr(settings, "extinction", 1.42))
    soot_absorption = float(getattr(settings, "soot_absorption", 0.34))
    exposure = float(getattr(settings, "exposure", 1.08))
    scattering = float(getattr(settings, "scattering", 0.92))
    absorption = float(getattr(settings, "absorption", 0.32))
    thin_color = np.asarray(getattr(settings, "thin_color", (0.43, 0.49, 0.56)), dtype=np.float32)
    dense_color = np.asarray(getattr(settings, "dense_color", (0.88, 0.87, 0.80)), dtype=np.float32)
    if thin_color.shape != (3,):
        thin_color = np.array([0.43, 0.49, 0.56], dtype=np.float32)
    if dense_color.shape != (3,):
        dense_color = np.array([0.88, 0.87, 0.80], dtype=np.float32)
    fire_glow = float(getattr(settings, "fire_glow", 0.26))
    phase_g = float(getattr(settings, "phase_g", 0.48))
    self_shadow = bool(getattr(settings, "self_shadow", True))
    view_dir = np.asarray(_physical_projection_view_direction(), dtype=np.float32)
    view_dir /= max(float(np.linalg.norm(view_dir)), 1.0e-6)
    light_dir = np.asarray(PHYSICAL_SMOKE_SUN_DIRECTION, dtype=np.float32)
    light_dir /= max(float(np.linalg.norm(light_dir)), 1.0e-6)
    phase = _henyey_greenstein_py(float(np.dot(view_dir, light_dir)), phase_g)

    depth, altitude_count, width = density.shape
    render_w, render_h = (max(1, int(render_size[0])), max(1, int(render_size[1])))
    step_size = float(getattr(settings, "step_size", 0.72))
    if step_size <= 0.0:
        diagonal = math.sqrt(width * width + altitude_count * altitude_count + depth * depth)
        step_size = max(diagonal / max(float(getattr(settings, "max_steps", 128)), 1.0), 0.35)
    max_steps = int(max(1, getattr(settings, "max_steps", 128)))
    jitter_strength = float(np.clip(getattr(settings, "jitter_strength", 0.35), 0.0, 1.0))
    transmittance = np.ones((render_h, render_w), dtype=np.float32)
    rgb_accum = np.zeros((render_h, render_w, 3), dtype=np.float32)
    texture = _advected_smoke_texture((render_h, render_w), frame_index, seed + 3109)
    pixel_x, pixel_z = _pixel_grids((render_h, render_w))
    plane_x = pixel_x / max(float(render_w - 1), 1.0) * max(float(width - 1), 0.0)
    plane_z = pixel_z / max(float(render_h - 1), 1.0) * max(float(depth - 1), 0.0)
    plane_y = np.full_like(plane_x, (altitude_count - 1) * 0.5, dtype=np.float32)
    diagonal = math.sqrt(max(float(width - 1), 1.0) ** 2 + max(float(altitude_count - 1), 1.0) ** 2 + max(float(depth - 1), 1.0) ** 2)
    origin_x = (plane_x - float(view_dir[0]) * (diagonal + step_size)).astype(np.float32)
    origin_y = (plane_y - float(view_dir[1]) * (diagonal + step_size)).astype(np.float32)
    origin_z = (plane_z - float(view_dir[2]) * (diagonal + step_size)).astype(np.float32)
    t_enter, t_exit, valid = _python_ray_box_intersection(
        origin_x,
        origin_y,
        origin_z,
        view_dir,
        (max(float(width - 1), 0.0), max(float(altitude_count - 1), 0.0), max(float(depth - 1), 0.0)),
    )
    jitter = _bilinear_sample_wrapped(texture, pixel_x + seed * 0.017, pixel_z - frame_index * 0.023)
    t_current = (np.maximum(t_enter, 0.0) + jitter * jitter_strength * step_size).astype(np.float32)
    shadow_volume = (
        _python_volume_shadow_grid(density, soot, particle_age, light_dir, density_scale, extinction, soot_absorption, settings)
        if self_shadow
        else None
    )
    scatter_albedo = np.clip(scattering / max(scattering + absorption, 1.0e-5), 0.02, 0.98)
    sun_radiance = np.array([1.0, 0.96, 0.84], dtype=np.float32) * 11.5
    sky_base = np.array([0.52, 0.60, 0.72], dtype=np.float32)
    bounce_base = np.array([0.58, 0.54, 0.48], dtype=np.float32)
    old_blue = np.array([0.46, 0.49, 0.52], dtype=np.float32)
    aged_blue = np.array([0.36, 0.39, 0.43], dtype=np.float32)
    pale_core = np.array([0.70, 0.73, 0.72], dtype=np.float32)
    glow_color = np.array([3.40, 0.62, 0.03], dtype=np.float32)
    warm_color = np.array([0.95, 0.58, 0.24], dtype=np.float32)
    rgb_flat = rgb_accum.reshape(-1, 3)
    trans_flat = transmittance.reshape(-1)
    t_flat = t_current.reshape(-1)
    pixel_x_flat = pixel_x.reshape(-1)
    pixel_z_flat = pixel_z.reshape(-1)

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
        layer = _trilinear_sample_volume(density, sample_x, sample_y, sample_z)
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
        temp_layer = _trilinear_sample_volume(temperature, sample_x, sample_y, sample_z)
        soot_layer = _trilinear_sample_volume(soot, sample_x, sample_y, sample_z)
        emission_layer = _trilinear_sample_volume(emission, sample_x, sample_y, sample_z)
        age_layer = _trilinear_sample_volume(particle_age, sample_x, sample_y, sample_z)
        altitude = np.clip(sample_y / max(float(altitude_count - 1), 1.0), 0.0, 1.0).astype(np.float32)
        age_t = _smoothstep(1.6, 17.0, age_layer)
        sigma = np.clip(
            layer * density_scale * (1.05 + 0.34 * altitude) * (1.0 + soot_layer * soot_absorption),
            0.0,
            4.5,
        )
        concentration_gate = 0.50 + 0.50 * _smoothstep(0.045, 0.34, layer)
        sigma *= 1.0 - 0.58 * age_t
        sigma *= concentration_gate
        sigma_t = np.clip(sigma * extinction, 0.0, 8.0).astype(np.float32)
        optical_depth = sigma_t * segment_length
        segment_transmittance = np.exp(-optical_depth).astype(np.float32)
        segment_weight = np.divide(
            1.0 - segment_transmittance,
            sigma_t,
            out=segment_length.copy(),
            where=sigma_t > 1.0e-6,
        )
        segment_weight *= 0.80 + 0.20 * _bilinear_sample_wrapped(
            texture,
            pixel_x_flat[occupied_index] + altitude * 6.5,
            pixel_z_flat[occupied_index] - altitude * 4.0,
        )
        dense_t = _smoothstep(0.05, 1.40, sigma)
        thin = thin_color * (1.0 - dense_t[..., None]) + pale_core * dense_t[..., None]
        layer_rgb = old_blue * (1.0 - dense_t[..., None]) + thin * dense_t[..., None]
        core = (
            _smoothstep(0.58, 1.62, sigma)
            * (1.0 - 0.26 * _smoothstep(0.08, 0.42, soot_layer))
            * (1.0 - 0.68 * age_t)
        )
        layer_rgb = layer_rgb * (1.0 - core[..., None]) + dense_color * core[..., None]
        layer_rgb = layer_rgb * (1.0 - 0.42 * age_t[..., None]) + aged_blue * (0.42 * age_t[..., None])
        warm = _smoothstep(0.18, 1.10, temp_layer) * (1.0 - age_t)
        layer_rgb = layer_rgb * (1.0 - warm[..., None] * 0.035) + warm_color * (warm[..., None] * 0.035)
        if shadow_volume is not None:
            light_trans = _trilinear_sample_volume(shadow_volume, sample_x, sample_y, sample_z)
        else:
            light_trans = np.ones_like(layer, dtype=np.float32)
        sigma_s = sigma_t * scatter_albedo
        sky_radiance = sky_base * (
            0.36 + 0.26 * (1.0 - light_trans)
        )[..., None]
        ground_bounce = bounce_base * (0.070 * (1.0 - altitude))[..., None]
        fresh_heat = temp_layer * (1.0 - age_t) * (1.0 - age_t)
        glow_strength = np.clip((fresh_heat * 0.10 + emission_layer * 1.18) * fire_glow, 0.0, 5.0)
        glow = glow_color * glow_strength[..., None]
        direct = layer_rgb * sigma_s[..., None] * sun_radiance * (phase * light_trans[..., None])
        multiple = layer_rgb * sigma_s[..., None] * (sky_radiance + ground_bounce)
        source_radiance = direct + multiple + glow
        rgb_flat[occupied_index] += source_radiance * segment_weight[..., None] * trans_flat[occupied_index][..., None]
        trans_flat[occupied_index] *= segment_transmittance
    alpha = np.clip(1.0 - transmittance, 0.0, 1.0)
    rgb = np.divide(rgb_accum, alpha[..., None], out=np.zeros_like(rgb_accum), where=alpha[..., None] > 1.0e-6)
    rgb = rgb / (1.0 + rgb)
    out = np.zeros((render_h, render_w, 4), dtype=np.uint8)
    out[..., :3] = np.clip(np.round(rgb * exposure * 255.0), 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)
    return out


def _render_projected_physical_volume(effect: PhysicalSmokeMainEffect, frame_index: int) -> np.ndarray:
    render_w, render_h = effect.render_size
    if hasattr(effect.domain, "render_projection_rgba"):
        return np.asarray(
            effect.domain.render_projection_rgba(
                render_w,
                render_h,
                view_direction=_physical_projection_view_direction(),
                sun_direction=PHYSICAL_SMOKE_SUN_DIRECTION,
                settings=effect.render_settings,
            ),
            dtype=np.uint8,
        )
    return _python_projected_volume_raymarch(
        effect.domain,
        effect.render_size,
        frame_index,
        effect.seed,
        effect.render_settings,
    )


def _curl_warp_scalar(field: np.ndarray, frame_index: int, seed: int, strength: float) -> np.ndarray:
    src = np.asarray(field, dtype=np.float32)
    if src.ndim != 2 or not np.any(src > 0.0):
        return src.astype(np.float32, copy=True)
    shape = src.shape
    scale = min(shape) / 408.0
    x, y = _pixel_grids(shape)
    curl = _pil_blur_float(_advected_smoke_texture(shape, frame_index, seed), 4.8)
    gy, gx = np.gradient(curl)
    amp = float(strength) * max(scale, 0.12)
    dx = gy * amp + np.sin(y / max(24.0 * scale, 1.0) + curl * 4.8 + frame_index * 0.011) * 1.7 * scale
    dy = -gx * amp + np.sin(x / max(34.0 * scale, 1.0) - curl * 3.9 + frame_index * 0.008) * 1.1 * scale
    return _bilinear_sample(src, x - dx, y - dy)


def _directional_accumulation(
    field: np.ndarray,
    direction: tuple[float, float],
    *,
    steps: int,
    step_px: float,
    falloff: float,
) -> np.ndarray:
    src = np.clip(np.asarray(field, dtype=np.float32), 0.0, None)
    if src.ndim != 2 or not np.any(src > 0.0):
        return np.zeros_like(src, dtype=np.float32)
    dx, dy = float(direction[0]), float(direction[1])
    length = math.hypot(dx, dy)
    if length < 1.0e-6:
        return np.zeros_like(src, dtype=np.float32)
    dx /= length
    dy /= length
    x, y = _pixel_grids(src.shape)
    accum = np.zeros_like(src, dtype=np.float32)
    weight_sum = 0.0
    for idx in range(1, max(1, int(steps)) + 1):
        weight = math.exp(-float(idx) / max(float(falloff), 1.0e-3))
        offset = float(idx) * float(step_px)
        accum += _bilinear_sample(src, x - dx * offset, y - dy * offset) * weight
        weight_sum += weight
    return accum / max(weight_sum, 1.0e-6)


def _projected_smoke_depth_cues(
    alpha: np.ndarray,
    frame_index: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    density = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    if density.ndim != 2 or not np.any(density > 0.0):
        zeros = np.zeros_like(density, dtype=np.float32)
        return zeros, zeros, zeros
    scale = min(density.shape) / 408.0
    soft_density = _pil_blur_float(density, 2.0)
    sun_shadow = _directional_accumulation(
        soft_density,
        (0.58, -0.34),
        steps=12,
        step_px=max(2.4 * scale, 0.8),
        falloff=5.4,
    )
    view_depth = _directional_accumulation(
        soft_density,
        (-0.18, 0.92),
        steps=8,
        step_px=max(3.4 * scale, 1.0),
        falloff=4.0,
    )
    shadow_texture = _pil_blur_float(_advected_smoke_texture(density.shape, frame_index, seed + 13007), 5.4)
    shadow = np.clip((sun_shadow * 0.82 + view_depth * 0.30) * (0.78 + 0.44 * shadow_texture), 0.0, 1.0)
    gy, gx = np.gradient(_pil_blur_float(density, 1.15))
    sun_edge = np.clip(gx * 0.58 - gy * 0.34, 0.0, 1.0)
    rim = _pil_blur_float(sun_edge, 1.2) * _smoothstep(0.018, 0.36, density) * (1.0 - _smoothstep(0.58, 0.95, density))
    optical_depth = np.clip(_pil_blur_float(density, 5.2) * 0.62 + shadow * 0.54, 0.0, 1.0)
    return shadow.astype(np.float32), rim.astype(np.float32), optical_depth.astype(np.float32)


def _active_source_centroid(effect: PhysicalSmokeMainEffect, frame_index: int) -> tuple[float, float, float]:
    weights: list[float] = []
    coords: list[tuple[float, float]] = []
    for source in effect.sources:
        if frame_index < source.start_frame or frame_index > source.end_frame:
            continue
        burst = _source_burst_envelope(source, frame_index)
        if burst <= 0.05:
            continue
        weights.append(float(source.strength * source.smoke_rate * (0.35 + burst) * (1.0 + 0.15 * source.heat)))
        coords.append((float(source.x), float(source.y)))
    if not weights:
        if effect.sources:
            coords = [(float(source.x), float(source.y)) for source in effect.sources[: min(6, len(effect.sources))]]
            weights = [max(float(source.strength), 0.05) for source in effect.sources[: len(coords)]]
        else:
            return effect.map_size[0] * 0.36, effect.map_size[1] * 0.62, 0.0
    weights_arr = np.asarray(weights, dtype=np.float32)
    coords_arr = np.asarray(coords, dtype=np.float32)
    total = max(float(np.sum(weights_arr)), 1.0e-6)
    centroid = np.sum(coords_arr * weights_arr[:, None], axis=0) / total
    return float(centroid[0]), float(centroid[1]), float(np.clip(np.mean(weights_arr), 0.0, 8.0))


def _angle_difference(a: np.ndarray, b: float) -> np.ndarray:
    return np.arctan2(np.sin(a - float(b)), np.cos(a - float(b))).astype(np.float32)


def _physical_structure_enhancement_rgba(
    effect: PhysicalSmokeMainEffect,
    frame_index: int,
    base_alpha: np.ndarray,
) -> np.ndarray:
    alpha = np.clip(np.asarray(base_alpha, dtype=np.float32), 0.0, 1.0)
    height, width = alpha.shape
    x, y = _pixel_grids((height, width))
    wind = _hybrid_layer_wind_vector(1).astype(np.float32)
    wind_len = max(float(np.linalg.norm(wind)), 1.0e-6)
    wind = wind / wind_len
    cross = np.array([-wind[1], wind[0]], dtype=np.float32)
    cx, cy, active_strength = _active_source_centroid(effect, frame_index)
    rel_x = x - np.float32(cx)
    rel_y = y - np.float32(cy)
    along = rel_x * wind[0] + rel_y * wind[1]
    cross_coord = rel_x * cross[0] + rel_y * cross[1]
    scale = min(height, width) / 408.0
    source_envelope = _smoothstep(0.0, 42.0 * scale, along) * (1.0 - _smoothstep(340.0 * scale, 515.0 * scale, along))
    base_support = np.clip(_smoothstep(0.008, 0.18, alpha) + 0.70 * source_envelope, 0.0, 1.0)
    broad_texture = _pil_blur_float(_advected_smoke_texture((height, width), frame_index, effect.seed + 25031), 6.6)
    fine_texture = _pil_blur_float(_advected_smoke_texture((height, width), frame_index, effect.seed + 26033), 2.0)

    streamers = np.zeros_like(alpha, dtype=np.float32)
    for lane_idx, lane_offset in enumerate((-22.0, -9.0, 6.0, 19.0)):
        phase = frame_index * (0.020 + lane_idx * 0.004) + effect.seed * 0.003 + lane_idx * 1.37
        center = (
            lane_offset * scale
            + np.sin(along / max((36.0 + lane_idx * 9.0) * scale, 1.0) + phase) * (10.0 + lane_idx * 2.4) * scale
            + (broad_texture - 0.5) * 16.0 * scale
        )
        width_px = (3.6 + lane_idx * 0.7) * scale + np.clip(along, 0.0, 260.0 * scale) * 0.035
        ridge = np.exp(-((cross_coord - center) ** 2) / (2.0 * np.maximum(width_px, 1.0) ** 2))
        lane_gate = _smoothstep(4.0 * scale, 44.0 * scale, along) * (1.0 - _smoothstep(210.0 * scale, 360.0 * scale, along))
        streamers += ridge * lane_gate * (0.86 + 0.34 * (fine_texture - 0.5))

    hooks = np.zeros_like(alpha, dtype=np.float32)
    for hook_idx, distance in enumerate((96.0, 158.0, 235.0, 315.0)):
        side = -1.0 if hook_idx % 2 else 1.0
        phase = frame_index * (0.010 + hook_idx * 0.002) + effect.seed * 0.002 + hook_idx * 1.11
        center_x = np.float32(cx) + wind[0] * distance * scale + cross[0] * side * (28.0 + hook_idx * 10.0) * scale
        center_y = np.float32(cy) + wind[1] * distance * scale + cross[1] * side * (28.0 + hook_idx * 10.0) * scale
        hx = (x - center_x) * wind[0] + (y - center_y) * wind[1]
        hy = (x - center_x) * cross[0] + (y - center_y) * cross[1]
        radius = (34.0 + hook_idx * 13.0) * scale
        width_px = (6.0 + hook_idx * 1.8) * scale
        ring = np.exp(-((np.sqrt(hx * hx + hy * hy) - radius) ** 2) / (2.0 * max(width_px, 1.0) ** 2))
        theta = np.arctan2(hy * side, hx + radius * 0.28)
        target_angle = 0.45 + 0.42 * math.sin(phase)
        arc = np.exp(-(_angle_difference(theta, target_angle) ** 2) / (2.0 * 0.78 * 0.78))
        break_mask = 0.64 + 0.58 * broad_texture - 0.24 * fine_texture
        hooks += ring * arc * break_mask * (0.72 - hook_idx * 0.07)

    fan = np.zeros_like(alpha, dtype=np.float32)
    for fan_idx, fan_offset in enumerate((-1.0, 0.0, 1.0)):
        center = fan_offset * 58.0 * scale + np.sin(along / max(115.0 * scale, 1.0) + fan_idx) * 18.0 * scale
        width_px = 20.0 * scale + np.clip(along, 0.0, 390.0 * scale) * 0.13
        band = np.exp(-((cross_coord - center) ** 2) / (2.0 * np.maximum(width_px, 1.0) ** 2))
        gate = _smoothstep(120.0 * scale, 210.0 * scale, along) * (1.0 - _smoothstep(390.0 * scale, 545.0 * scale, along))
        fan += band * gate * (0.35 + 0.55 * broad_texture)

    structure = np.clip(streamers * 0.70 + hooks * 0.92 + fan * 0.04, 0.0, 1.0)
    structure *= base_support * np.clip(0.80 + 0.72 * (broad_texture - 0.5) + 0.34 * (fine_texture - 0.5), 0.38, 1.32)
    voids = _smoothstep(0.58, 0.92, 1.0 - broad_texture) * _smoothstep(0.04, 0.62, structure)
    structure *= 1.0 - 0.38 * voids
    structure = _pil_blur_float(np.clip(structure, 0.0, 1.0), 1.15)
    if not np.any(structure > 0.002):
        return np.zeros((height, width, 4), dtype=np.uint8)

    age_t = _smoothstep(120.0 * scale, 450.0 * scale, along)
    dense_t = _smoothstep(0.22, 0.84, structure + alpha * 0.30)
    thin = np.array([140.0, 154.0, 166.0], dtype=np.float32)
    mid = np.array([180.0, 187.0, 184.0], dtype=np.float32)
    milky = np.array([238.0, 237.0, 224.0], dtype=np.float32)
    rgb = thin * (1.0 - dense_t[..., None]) + mid * dense_t[..., None]
    core = _smoothstep(0.44, 0.92, structure) * (1.0 - 0.45 * age_t)
    rgb = rgb * (1.0 - core[..., None]) + milky * core[..., None]
    aged_blue = np.array([86.0, 99.0, 116.0], dtype=np.float32)
    rgb = rgb * (1.0 - 0.26 * age_t[..., None]) + aged_blue * (0.26 * age_t[..., None])
    shadow, rim, depth = _projected_smoke_depth_cues(structure, frame_index, effect.seed + 27143)
    rgb *= (1.0 - np.clip(0.14 * shadow + 0.08 * depth, 0.0, 0.26))[..., None]
    rgb += rim[..., None] * np.array([28.0, 27.0, 22.0], dtype=np.float32)

    alpha_u8 = np.clip(np.round(structure * PHYSICAL_SMOKE_STRUCTURE_MAX_ALPHA * 0.82), 0, PHYSICAL_SMOKE_STRUCTURE_MAX_ALPHA)
    out = np.zeros((height, width, 4), dtype=np.uint8)
    out[..., :3] = np.clip(np.round(rgb), 0, 245).astype(np.uint8)
    out[..., 3] = np.where(alpha_u8 >= 3, alpha_u8, 0).astype(np.uint8)
    return out


def _domain_density_velocity_numpy(effect: PhysicalSmokeMainEffect) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        density = np.asarray(effect.domain.to_density_numpy(), dtype=np.float32)
    except Exception:
        return np.zeros((0, 0, 0), dtype=np.float32), None
    velocity = None
    if hasattr(effect.domain, "to_velocity_numpy"):
        try:
            velocity = np.asarray(effect.domain.to_velocity_numpy(), dtype=np.float32)
        except Exception:
            velocity = None
    elif hasattr(effect.domain, "velocity"):
        try:
            velocity = np.asarray(effect.domain.velocity, dtype=np.float32)
        except Exception:
            velocity = None
    return density, velocity


def _physical_volume_lane_fields(effect: PhysicalSmokeMainEffect, frame_index: int) -> dict[str, np.ndarray] | None:
    density, velocity = _domain_density_velocity_numpy(effect)
    map_w, map_h = effect.map_size
    if density.ndim != 3 or density.size == 0 or not np.any(density > 1.0e-6):
        return None

    d = np.clip(density, 0.0, None)
    column = np.sum(d, axis=1)
    peak = max(float(np.percentile(column[column > 0.0], 98.5)) if np.any(column > 0.0) else 0.0, 1.0e-6)
    column_t = 1.0 - np.exp(-column / peak * 1.75)
    support = _resample_float_field(column_t, (map_h, map_w))
    support = _pil_blur_float(np.clip(support, 0.0, 1.0), 0.55)

    altitude_idx = np.linspace(0.0, 1.0, d.shape[1], dtype=np.float32)[None, :, None]
    altitude_mass = np.sum(d * altitude_idx, axis=1)
    altitude = np.divide(altitude_mass, np.maximum(column, 1.0e-6), out=np.zeros_like(column), where=column > 1.0e-6)
    altitude_map = _resample_float_field(altitude, (map_h, map_w))

    fine = _pil_blur_float(support, 0.75)
    broad = _pil_blur_float(support, 7.0)
    density_ridges = np.clip(fine - broad * 0.72, 0.0, 1.0)
    ridge_positive = density_ridges[density_ridges > 0.0]
    ridge_scale = max(float(np.percentile(ridge_positive, 96.0)) if ridge_positive.size else 0.0, 1.0e-5)
    density_ridges = np.clip(density_ridges / ridge_scale, 0.0, 1.0)
    gy, gx = np.gradient(_pil_blur_float(support, 1.2))
    density_edge = np.clip(np.sqrt(gx * gx + gy * gy) * 6.0, 0.0, 1.0)

    curl_map = np.zeros_like(support, dtype=np.float32)
    shear_map = np.zeros_like(support, dtype=np.float32)
    speed_map = np.zeros_like(support, dtype=np.float32)
    if velocity is not None and velocity.shape == density.shape + (3,):
        weights = np.maximum(d, 1.0e-5)
        weight_sum = np.sum(weights, axis=1)
        vx = np.sum(velocity[..., 0] * weights, axis=1) / np.maximum(weight_sum, 1.0e-6)
        vz = np.sum(velocity[..., 2] * weights, axis=1) / np.maximum(weight_sum, 1.0e-6)
        vx_map = _resample_float_field(vx, (map_h, map_w))
        vz_map = _resample_float_field(vz, (map_h, map_w))
        dvx_dy, dvx_dx = np.gradient(_pil_blur_float(vx_map, 1.0))
        dvz_dy, dvz_dx = np.gradient(_pil_blur_float(vz_map, 1.0))
        curl = dvz_dx - dvx_dy
        shear = np.sqrt((dvx_dx - dvz_dy) ** 2 + (dvx_dy + dvz_dx) ** 2)
        curl_scale = max(float(np.percentile(np.abs(curl), 96.0)), 1.0e-5)
        shear_scale = max(float(np.percentile(shear, 96.0)), 1.0e-5)
        curl_map = np.clip(np.abs(curl) / curl_scale, 0.0, 1.0)
        shear_map = np.clip(shear / shear_scale, 0.0, 1.0)
        curl_map = _pil_blur_float(curl_map, 0.85)
        shear_map = _pil_blur_float(shear_map, 1.2)
        speed = np.sqrt(vx_map * vx_map + vz_map * vz_map)
        speed_scale = max(float(np.percentile(speed, 96.0)), 1.0e-5)
        speed_map = _pil_blur_float(np.clip(speed / speed_scale, 0.0, 1.0), 1.4)

    texture = _pil_blur_float(_advected_smoke_texture((map_h, map_w), frame_index, effect.seed + 31057), 3.2)
    lane_texture = _pil_blur_float(_advected_smoke_texture((map_h, map_w), frame_index, effect.seed + 31891), 8.8)
    x, y = _pixel_grids((map_h, map_w))
    wind = _hybrid_layer_wind_vector(1)
    along = x * wind[0] + y * wind[1]
    cross = x * (-wind[1]) + y * wind[0]
    scale = min(map_h, map_w) / 408.0
    filament_phase = (
        along / max(48.0 * scale, 1.0)
        - cross / max(29.0 * scale, 1.0)
        + lane_texture * 5.7
        + curl_map * 2.4
        + frame_index * 0.010
    )
    filament_gate = _pil_blur_float(_smoothstep(0.54, 0.98, 0.5 + 0.5 * np.sin(filament_phase)), 1.2)
    rollup_phase = (
        along / max(34.0 * scale, 1.0)
        + np.sin(cross / max(24.0 * scale, 1.0) + lane_texture * 3.6) * 2.2
        + curl_map * 2.7
        - frame_index * 0.013
    )
    rollup_gate = _pil_blur_float(_smoothstep(0.56, 0.98, 0.5 + 0.5 * np.sin(rollup_phase)), 1.45)
    lane_field = np.clip(
        filament_gate * 0.44
        + rollup_gate * 0.24
        + density_ridges * 0.34
        + curl_map * 0.18
        + shear_map * 0.14
        + speed_map * 0.10,
        0.0,
        1.0,
    )
    sheet_support = _smoothstep(0.055, 0.58, support)
    weak_lane = 1.0 - _smoothstep(0.26, 0.76, lane_field + density_ridges * 0.28 + density_edge * 0.16)
    texture_voids = _smoothstep(0.54, 0.92, 1.0 - lane_texture)
    ribbon_gaps = _smoothstep(0.48, 0.90, 1.0 - filament_gate) * _smoothstep(0.42, 0.88, 1.0 - rollup_gate)
    sheet_voids = np.maximum(texture_voids * weak_lane, ribbon_gaps * (1.0 - 0.34 * density_ridges))
    sheet_voids = _pil_blur_float(np.clip(sheet_voids * sheet_support, 0.0, 1.0), 1.25)
    lane_gain = np.clip(
        0.66
        + 0.42 * lane_field
        + 0.24 * density_ridges
        + 0.12 * density_edge
        + 0.12 * curl_map
        + 0.08 * shear_map
        - 0.42 * sheet_voids,
        0.28,
        1.24,
    )
    return {
        "support": support.astype(np.float32),
        "altitude": altitude_map.astype(np.float32),
        "ridges": density_ridges.astype(np.float32),
        "edge": density_edge.astype(np.float32),
        "curl": curl_map.astype(np.float32),
        "shear": shear_map.astype(np.float32),
        "speed": speed_map.astype(np.float32),
        "texture": texture.astype(np.float32),
        "lane_texture": lane_texture.astype(np.float32),
        "filaments": filament_gate.astype(np.float32),
        "rollups": rollup_gate.astype(np.float32),
        "lane": lane_field.astype(np.float32),
        "voids": sheet_voids.astype(np.float32),
        "gain": lane_gain.astype(np.float32),
    }


def _physical_lane_field(
    volume_fields: dict[str, np.ndarray] | None,
    key: str,
    shape: tuple[int, int],
) -> np.ndarray:
    if volume_fields is None or key not in volume_fields:
        return np.zeros(shape, dtype=np.float32)
    arr = np.asarray(volume_fields[key], dtype=np.float32)
    if arr.ndim != 2:
        return np.zeros(shape, dtype=np.float32)
    if arr.shape != shape:
        arr = _resample_float_field(arr, shape)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _physical_volume_structure_rgba(
    effect: PhysicalSmokeMainEffect,
    frame_index: int,
    volume_fields: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    map_w, map_h = effect.map_size
    fields = volume_fields if volume_fields is not None else _physical_volume_lane_fields(effect, frame_index)
    if fields is None:
        return np.zeros((map_h, map_w, 4), dtype=np.uint8)

    support = _physical_lane_field(fields, "support", (map_h, map_w))
    altitude_map = _physical_lane_field(fields, "altitude", (map_h, map_w))
    density_ridges = _physical_lane_field(fields, "ridges", (map_h, map_w))
    density_edge = _physical_lane_field(fields, "edge", (map_h, map_w))
    curl_map = _physical_lane_field(fields, "curl", (map_h, map_w))
    shear_map = _physical_lane_field(fields, "shear", (map_h, map_w))
    lane_field = _physical_lane_field(fields, "lane", (map_h, map_w))
    sheet_voids = _physical_lane_field(fields, "voids", (map_h, map_w))
    texture = _physical_lane_field(fields, "texture", (map_h, map_w))

    physical_structure = (
        density_ridges * 0.60
        + density_edge * 0.20
        + curl_map * 0.30
        + shear_map * 0.22
        + lane_field * 0.34
    )
    physical_structure *= _smoothstep(0.020, 0.42, support)
    physical_structure *= np.clip(0.72 + 0.52 * (texture - 0.5) + 0.38 * altitude_map, 0.34, 1.20)
    physical_structure *= 1.0 - 0.36 * sheet_voids * _smoothstep(0.05, 0.68, physical_structure)
    physical_structure = _pil_blur_float(np.clip(physical_structure, 0.0, 1.0), 0.95)
    if not np.any(physical_structure > 0.003):
        return np.zeros((map_h, map_w, 4), dtype=np.uint8)

    aged = _smoothstep(0.18, 0.88, support) * (1.0 - _smoothstep(0.18, 0.72, density_ridges))
    dense_t = _smoothstep(0.18, 0.86, physical_structure + support * 0.16)
    thin = np.array([116.0, 132.0, 148.0], dtype=np.float32)
    mid = np.array([160.0, 170.0, 171.0], dtype=np.float32)
    bright = np.array([226.0, 226.0, 214.0], dtype=np.float32)
    rgb = thin * (1.0 - dense_t[..., None]) + mid * dense_t[..., None]
    bright_core = _smoothstep(0.44, 0.92, density_ridges + lane_field * 0.24) * (1.0 - 0.45 * aged)
    rgb = rgb * (1.0 - bright_core[..., None]) + bright * bright_core[..., None]
    aged_blue = np.array([78.0, 92.0, 110.0], dtype=np.float32)
    rgb = rgb * (1.0 - 0.24 * aged[..., None]) + aged_blue * (0.24 * aged[..., None])
    shadow, rim, depth = _projected_smoke_depth_cues(physical_structure, frame_index, effect.seed + 32213)
    rgb *= (1.0 - np.clip(0.18 * shadow + 0.10 * depth, 0.0, 0.32))[..., None]
    rgb += rim[..., None] * np.array([20.0, 20.0, 17.0], dtype=np.float32)

    out = np.zeros((map_h, map_w, 4), dtype=np.uint8)
    alpha_u8 = np.clip(
        np.round(physical_structure * PHYSICAL_SMOKE_VOLUME_STRUCTURE_MAX_ALPHA),
        0,
        PHYSICAL_SMOKE_VOLUME_STRUCTURE_MAX_ALPHA,
    )
    out[..., :3] = np.clip(np.round(rgb), 0, 242).astype(np.uint8)
    out[..., 3] = np.where(alpha_u8 >= 3, alpha_u8, 0).astype(np.uint8)
    return out


def _physical_history_layer_rgba(
    rgba: np.ndarray,
    age_frames: int,
    frame_index: int,
    seed: int,
) -> np.ndarray:
    arr = np.asarray(rgba, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 4 or age_frames <= 0:
        return np.zeros_like(arr, dtype=np.uint8)
    alpha = arr[..., 3] / 255.0
    if not np.any(alpha > 0.0):
        return np.zeros_like(arr, dtype=np.uint8)

    height, width = alpha.shape
    shape = (height, width)
    scale = min(shape) / 408.0
    x, y = _pixel_grids(shape)
    wind = _hybrid_layer_wind_vector(1)
    cross = np.array([-wind[1], wind[0]], dtype=np.float32)
    history_noise = _pil_blur_float(_advected_smoke_texture(shape, frame_index - age_frames // 2, seed + 11903), 8.0)
    gy, gx = np.gradient(history_noise)
    age = float(age_frames)
    drift = age * 0.34 * max(scale, 0.16)
    sway = math.sin(frame_index * 0.013 + seed * 0.007 + age * 0.041) * age * 0.050 * max(scale, 0.16)
    curl_amp = min(age, 72.0) * 5.2 * max(scale, 0.16)
    sample_x = x - wind[0] * drift - cross[0] * sway - gy * curl_amp
    sample_y = y - wind[1] * drift - cross[1] * sway + gx * curl_amp

    warped = np.zeros_like(arr, dtype=np.float32)
    for channel in range(4):
        warped[..., channel] = _bilinear_sample(arr[..., channel], sample_x, sample_y)
    blur_radius = min(3.6, 0.026 * age)
    if blur_radius > 0.25:
        for channel in range(4):
            warped[..., channel] = _pil_blur_float(warped[..., channel], blur_radius)

    age_t = float(_smoothstep(0.0, PHYSICAL_SMOKE_HISTORY_MAX_AGE_FRAMES, age))
    fade = math.exp(-age / 68.0) * (1.0 - float(_smoothstep(72.0, PHYSICAL_SMOKE_HISTORY_MAX_AGE_FRAMES, age)) * 0.72)
    alpha_f = np.clip(
        warped[..., 3] / 255.0 * fade * PHYSICAL_SMOKE_HISTORY_ALPHA_SCALE,
        0.0,
        PHYSICAL_SMOKE_MAX_ALPHA / 255.0,
    )
    along_coord = x * wind[0] + y * wind[1]
    cross_coord = x * (-wind[1]) + y * wind[0]
    generation_texture = _pil_blur_float(
        _advected_smoke_texture(shape, frame_index - age_frames, seed + 15131),
        9.5 + age_t * 4.0,
    )
    ribbon_phase = (
        along_coord / max((50.0 + age * 0.20) * scale, 1.0)
        - cross_coord / max((24.0 + age * 0.11) * scale, 1.0)
        + generation_texture * 5.1
        + frame_index * 0.006
    )
    aged_ribbons = _pil_blur_float(0.5 + 0.5 * np.sin(ribbon_phase), 1.8 + age_t * 1.25)
    generation_phase = (
        along_coord / max((74.0 + age * 0.18) * scale, 1.0)
        + cross_coord / max((31.0 + age * 0.10) * scale, 1.0)
        + generation_texture * 4.2
        - frame_index * 0.004
    )
    generation_bands = _pil_blur_float(0.5 + 0.5 * np.sin(generation_phase), 3.4 + age_t * 1.6)
    aged_voids = (
        _smoothstep(0.54, 0.90, 1.0 - generation_texture)
        * _smoothstep(0.46, 0.88, 1.0 - aged_ribbons)
        * _smoothstep(0.04, 0.46, alpha_f)
    )
    alpha_f *= np.clip(
        0.47 + 0.47 * generation_texture + 0.34 * (aged_ribbons - 0.5) + 0.22 * (generation_bands - 0.5),
        0.14,
        1.16,
    )
    alpha_f *= 1.0 - (0.42 + 0.30 * age_t) * aged_voids
    alpha_f = _pil_blur_float(alpha_f, 0.65 + age_t * 1.25)
    aged_blue = np.array([82.0, 95.0, 110.0], dtype=np.float32)
    flat_gray = np.array([135.0, 143.0, 148.0], dtype=np.float32)
    target_rgb = aged_blue * age_t + flat_gray * (1.0 - age_t)
    rgb = warped[..., :3] * (1.0 - 0.58 * age_t) + target_rgb * (0.58 * age_t)
    history_shadow = _directional_accumulation(
        _pil_blur_float(alpha_f, 1.8 + age_t),
        (0.58, -0.34),
        steps=5,
        step_px=max(3.5 * scale, 0.9),
        falloff=3.2,
    )
    history_rim = np.clip(np.gradient(_pil_blur_float(alpha_f, 1.0))[1] * 0.70, 0.0, 1.0)
    rgb *= (0.94 - 0.10 * age_t) * (1.0 - 0.22 * history_shadow[..., None])
    rgb += history_rim[..., None] * np.array([14.0, 15.0, 14.0], dtype=np.float32) * (1.0 - 0.35 * age_t)

    out = np.zeros_like(arr, dtype=np.uint8)
    out[..., :3] = np.clip(np.round(rgb), 0, 235).astype(np.uint8)
    out[..., 3] = np.clip(np.round(alpha_f * 255.0), 0, PHYSICAL_SMOKE_MAX_ALPHA).astype(np.uint8)
    out[..., 3] = np.where(out[..., 3] >= 2, out[..., 3], 0).astype(np.uint8)
    return out


def _physical_history_rgba(effect: PhysicalSmokeMainEffect, frame_index: int) -> np.ndarray:
    height, width = effect.map_size[1], effect.map_size[0]
    combined = np.zeros((height, width, 4), dtype=np.uint8)
    for history_frame, history_rgba in effect.history:
        age = int(frame_index) - int(history_frame)
        if age <= 0 or age > PHYSICAL_SMOKE_HISTORY_MAX_AGE_FRAMES:
            continue
        layer = _physical_history_layer_rgba(history_rgba, age, frame_index, effect.seed)
        combined = _premultiplied_over(combined, layer)
        combined[..., 3] = np.minimum(combined[..., 3], PHYSICAL_SMOKE_MAX_ALPHA).astype(np.uint8)
    return combined


def _physical_source_glow_rgba(effect: PhysicalSmokeMainEffect, frame_index: int) -> np.ndarray:
    glow = hybrid_fire_sources_rgba(
        effect.sources,
        frame_index,
        effect.map_size,
        glow_only=False,
        bloom_scale=0.92,
        core_alpha_scale=0.38,
    )
    return _scale_rgba_alpha(glow, 0.58)


def _physical_temporal_reproject_rgba(
    rgba: np.ndarray,
    age_frames: int,
    frame_index: int,
    seed: int,
) -> np.ndarray:
    arr = np.asarray(rgba, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 4 or age_frames <= 0:
        return np.zeros_like(arr, dtype=np.uint8)
    alpha = arr[..., 3] / 255.0
    if not np.any(alpha > 0.0):
        return np.zeros_like(arr, dtype=np.uint8)

    height, width = alpha.shape
    shape = (height, width)
    scale = min(shape) / 408.0
    x, y = _pixel_grids(shape)
    wind = _hybrid_layer_wind_vector(1)
    cross = np.array([-wind[1], wind[0]], dtype=np.float32)
    lane_noise = _pil_blur_float(_advected_smoke_texture(shape, frame_index, seed + 21101), 6.2)
    gy, gx = np.gradient(lane_noise)
    age = float(age_frames)
    drift = age * 0.48 * max(scale, 0.16)
    curl_amp = min(age, 4.0) * 4.2 * max(scale, 0.16)
    sway = math.sin(frame_index * 0.019 + seed * 0.004) * age * 0.035 * max(scale, 0.16)
    sample_x = x - wind[0] * drift - cross[0] * sway - gy * curl_amp
    sample_y = y - wind[1] * drift - cross[1] * sway + gx * curl_amp

    warped = np.zeros_like(arr, dtype=np.float32)
    for channel in range(4):
        warped[..., channel] = _bilinear_sample(arr[..., channel], sample_x, sample_y)
    for channel in range(4):
        warped[..., channel] = _pil_blur_float(warped[..., channel], 0.35)
    warped[..., 3] *= math.exp(-age / 4.5)
    out = np.clip(np.round(warped), 0, 255).astype(np.uint8)
    out[..., 3] = np.minimum(out[..., 3], PHYSICAL_SMOKE_MAX_ALPHA).astype(np.uint8)
    return out


def _temporal_blend_physical_smoke(
    effect: PhysicalSmokeMainEffect,
    current: np.ndarray,
    frame_index: int,
) -> np.ndarray:
    previous = effect.previous_render_rgba
    previous_frame = effect.previous_render_frame
    if previous is None or previous_frame is None:
        return current
    age = int(frame_index) - int(previous_frame)
    if age <= 0 or age > 4 or previous.shape != current.shape:
        return current
    reprojected = _physical_temporal_reproject_rgba(previous, age, frame_index, effect.seed)
    if not np.any(reprojected[..., 3] > 0):
        return current
    temporal_weight = 0.24 * (1.0 - float(_smoothstep(1.0, 4.0, age)))
    under = _scale_rgba_alpha(reprojected, temporal_weight)
    blended = _premultiplied_over(under, current)
    blended[..., 3] = np.minimum(blended[..., 3], PHYSICAL_SMOKE_MAX_ALPHA).astype(np.uint8)
    return blended


def _record_physical_history(effect: PhysicalSmokeMainEffect, frame_index: int, rgba: np.ndarray) -> None:
    if effect.history and int(effect.history[-1][0]) == int(frame_index):
        return
    if effect.history and int(frame_index) - int(effect.history[-1][0]) < PHYSICAL_SMOKE_HISTORY_STRIDE:
        return
    stored = np.asarray(rgba, dtype=np.uint8).copy()
    effect.history.append((int(frame_index), stored))
    min_frame = int(frame_index) - PHYSICAL_SMOKE_HISTORY_MAX_AGE_FRAMES
    effect.history[:] = [
        (history_frame, history_rgba)
        for history_frame, history_rgba in effect.history
        if history_frame >= min_frame
    ][-PHYSICAL_SMOKE_HISTORY_MAX_LAYERS:]


def _postprocess_physical_smoke_rgba(
    rgba: np.ndarray,
    frame_index: int,
    seed: int,
    map_size: tuple[int, int],
    volume_fields: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA")
    if image.size != tuple(map(int, map_size)):
        image = image.resize(tuple(map(int, map_size)), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32)
    alpha = np.clip(arr[..., 3] / 255.0, 0.0, 1.0)
    if not np.any(alpha > 0.0):
        return np.zeros((int(map_size[1]), int(map_size[0]), 4), dtype=np.uint8)
    alpha = np.clip(_curl_warp_scalar(alpha, frame_index, seed + 7109, 390.0), 0.0, 1.0)
    alpha = np.clip(_curl_warp_scalar(alpha, frame_index, seed + 9719, 150.0) * 0.84 + alpha * 0.16, 0.0, 1.0)
    fine = _pil_blur_float(alpha, 0.85)
    mid = _pil_blur_float(alpha, 2.8)
    broad = _pil_blur_float(alpha, 7.2)
    texture = _advected_smoke_texture(alpha.shape, frame_index, seed + 4517)
    low_texture = _pil_blur_float(texture, 8.6)
    cell_texture = _pil_blur_float(_advected_smoke_texture(alpha.shape, frame_index, seed + 8123), 13.0)
    branch_texture = _pil_blur_float(_advected_smoke_texture(alpha.shape, frame_index, seed + 19037), 5.8)
    x, y = _pixel_grids(alpha.shape)
    scale = min(alpha.shape) / 408.0
    wind = _hybrid_layer_wind_vector(1)
    cross_coord = x * (-wind[1]) + y * wind[0]
    along_coord = x * wind[0] + y * wind[1]
    xnorm = x / max(float(alpha.shape[1] - 1), 1.0)
    fresh_zone = 1.0 - _smoothstep(0.20, 0.53, xnorm)
    aged_zone = _smoothstep(0.32, 0.98, xnorm)
    lane_phase = (
        cross_coord / max(9.5 * scale, 1.0)
        + along_coord / max(74.0 * scale, 1.0)
        + low_texture * 4.9
        + branch_texture * 2.2
        + frame_index * 0.022
    )
    streamer_wave = 0.5 + 0.5 * np.sin(lane_phase)
    streamers = _pil_blur_float(_smoothstep(0.48, 0.98, streamer_wave), 1.35)
    wide_streamers = _pil_blur_float(streamer_wave, 2.4)
    hook_wave = (
        0.5
        + 0.5
        * np.sin(
            cross_coord / max(18.0 * scale, 1.0)
            - along_coord / max(46.0 * scale, 1.0)
            + texture * 3.8
            + branch_texture * 2.6
            + frame_index * 0.014
        )
    )
    hooks = _pil_blur_float(_smoothstep(0.55, 0.98, hook_wave), 1.75)
    loop_phase = (
        along_coord / max(42.0 * scale, 1.0)
        + np.sin(cross_coord / max(28.0 * scale, 1.0) + branch_texture * 3.1) * 2.1
        - frame_index * 0.013
    )
    loops = _pil_blur_float(_smoothstep(0.54, 0.96, 0.5 + 0.5 * np.sin(loop_phase)), 3.0)
    cell_phase = (
        along_coord / max(48.0 * scale, 1.0)
        - cross_coord / max(34.0 * scale, 1.0)
        + cell_texture * 5.6
        + frame_index * 0.010
    )
    plume_cells = _pil_blur_float(0.5 + 0.5 * np.sin(cell_phase), 3.2)
    fan_phase = (
        cross_coord / max(30.0 * scale, 1.0)
        + along_coord / max(130.0 * scale, 1.0)
        + low_texture * 3.0
        - frame_index * 0.006
    )
    fan_bands = _pil_blur_float(0.5 + 0.5 * np.sin(fan_phase), 4.2)
    rollup_phase = (
        along_coord / max(36.0 * scale, 1.0)
        + np.sin(cross_coord / max(21.0 * scale, 1.0) + branch_texture * 3.8) * 2.4
        + low_texture * 3.2
        - frame_index * 0.012
    )
    rollup_ridges = _pil_blur_float(_smoothstep(0.57, 0.98, 0.5 + 0.5 * np.sin(rollup_phase)), 1.95)
    streaks = 0.5 + 0.5 * np.sin(
        x / max(alpha.shape[1] / 24.0, 1.0)
        + y / max(alpha.shape[0] / 6.0, 1.0)
        + frame_index * 0.016
        + seed * 0.011
    )
    veil = _pil_blur_float(alpha, 15.5)
    fresh_alpha = np.clip(fine * (0.60 + 0.36 * streamers) + mid * 0.085 + broad * 0.012, 0.0, 1.0)
    aged_alpha = np.clip(fine * 0.46 + mid * 0.18 + broad * 0.040 + veil * 0.0060, 0.0, 1.0)
    shaped_alpha = np.clip(fresh_alpha * fresh_zone + aged_alpha * (1.0 - fresh_zone), 0.0, 1.0)
    shaped_alpha = np.clip(
        shaped_alpha
        + rollup_ridges * _smoothstep(0.045, 0.58, alpha) * (0.020 + 0.046 * aged_zone)
        + hooks * streamers * _smoothstep(0.050, 0.62, alpha) * (0.026 + 0.034 * fresh_zone),
        0.0,
        1.0,
    )
    shaped_alpha *= np.clip(
        0.61
        + 0.30 * (low_texture - 0.5)
        + 0.26 * (streamers - 0.5) * (0.86 + 0.54 * fresh_zone)
        + 0.17 * (hooks - 0.5)
        + 0.14 * (loops - 0.5)
        + 0.08 * (plume_cells - 0.5)
        + 0.07 * (fan_bands - 0.5) * aged_zone
        + 0.08 * (streaks - 0.5),
        0.34,
        1.18,
    )
    shaped_alpha = _pil_blur_float(shaped_alpha, 0.92)
    shaped_alpha = np.clip(shaped_alpha, 0.0, 1.0) ** 0.94
    holes = _smoothstep(0.54, 0.94, 1.0 - low_texture) * _smoothstep(0.04, 0.58, shaped_alpha)
    lane_gaps = _smoothstep(0.50, 0.86, 1.0 - wide_streamers) * _smoothstep(0.05, 0.64, shaped_alpha)
    fresh_lane_gaps = fresh_zone * _smoothstep(0.54, 0.92, 1.0 - streamers) * _smoothstep(0.04, 0.58, shaped_alpha)
    cell_voids = _smoothstep(0.48, 0.84, 1.0 - plume_cells) * _smoothstep(0.07, 0.70, shaped_alpha)
    loop_voids = _smoothstep(0.52, 0.88, 1.0 - loops) * aged_zone * _smoothstep(0.08, 0.64, shaped_alpha)
    fan_voids = _smoothstep(0.50, 0.84, 1.0 - fan_bands) * aged_zone * _smoothstep(0.06, 0.58, shaped_alpha)
    branch_voids = _smoothstep(0.56, 0.88, 1.0 - branch_texture) * _smoothstep(0.08, 0.62, shaped_alpha)
    sheet_split = (
        _smoothstep(0.44, 0.82, 1.0 - rollup_ridges)
        * _smoothstep(0.38, 0.92, 1.0 - hooks)
        * aged_zone
        * _smoothstep(0.06, 0.66, shaped_alpha)
    )
    edge_band = _smoothstep(0.015, 0.24, shaped_alpha) * (1.0 - _smoothstep(0.42, 0.78, shaped_alpha))
    edge_breakup = _smoothstep(0.50, 0.88, 1.0 - low_texture) * edge_band

    volume_support = _physical_lane_field(volume_fields, "support", alpha.shape)
    volume_lane = _physical_lane_field(volume_fields, "lane", alpha.shape)
    volume_ridges = _physical_lane_field(volume_fields, "ridges", alpha.shape)
    volume_curl = _physical_lane_field(volume_fields, "curl", alpha.shape)
    volume_shear = _physical_lane_field(volume_fields, "shear", alpha.shape)
    volume_voids_base = _physical_lane_field(volume_fields, "voids", alpha.shape)
    volume_gain = _physical_lane_field(volume_fields, "gain", alpha.shape)
    volume_voids = np.zeros_like(shaped_alpha, dtype=np.float32)
    volume_lane_recovery = np.zeros_like(shaped_alpha, dtype=np.float32)
    if np.any(volume_support > 0.001):
        volume_sheet = _smoothstep(0.055, 0.58, volume_support) * _smoothstep(0.045, 0.64, shaped_alpha)
        weak_solver_lane = 1.0 - _smoothstep(0.32, 0.84, volume_lane + volume_ridges * 0.30 + volume_curl * 0.16)
        lane_channel_gaps = _smoothstep(0.48, 0.90, 1.0 - volume_lane) * (1.0 - 0.38 * volume_ridges)
        volume_voids = np.clip(
            np.maximum(volume_voids_base, lane_channel_gaps * weak_solver_lane) * volume_sheet,
            0.0,
            1.0,
        )
        volume_voids = _pil_blur_float(volume_voids, 0.95)
        solver_carve = volume_voids * np.clip(0.56 + 0.44 * weak_solver_lane, 0.0, 1.0)
        solver_gain = np.clip(
            0.62
            + 0.24 * (volume_gain - 0.66)
            + 0.20 * volume_lane
            + 0.13 * volume_ridges
            + 0.08 * (volume_curl + volume_shear)
            - 0.62 * solver_carve,
            0.32,
            1.22,
        )
        shaped_alpha *= solver_gain
        shaped_alpha *= 1.0 - (0.42 + 0.42 * aged_zone) * solver_carve
        volume_lane_recovery = (
            volume_support
            * _smoothstep(0.030, 0.62, alpha)
            * (
                0.038 * volume_ridges
                + 0.026 * volume_lane
                + 0.016 * volume_curl
                + 0.012 * volume_shear
            )
            * (1.0 - 0.92 * volume_voids)
        )

    shaped_alpha *= (
        1.0
        - 0.12 * holes
        - 0.17 * lane_gaps
        - 0.22 * fresh_lane_gaps
        - 0.38 * cell_voids
        - 0.30 * loop_voids
        - 0.24 * fan_voids
        - 0.17 * branch_voids
        - 0.34 * sheet_split
        - 0.18 * edge_breakup
        - 0.36 * volume_voids
    )
    ribbon_recovery = (
        rollup_ridges * _smoothstep(0.045, 0.58, alpha) * (0.038 + 0.058 * aged_zone)
        + hooks * streamers * _smoothstep(0.050, 0.62, alpha) * (0.034 + 0.044 * fresh_zone)
        + volume_lane_recovery
    )
    shaped_alpha = np.clip(shaped_alpha + ribbon_recovery * (1.0 - 0.35 * holes), 0.0, 1.0)
    downwind_tail = 1.0 - 0.66 * _smoothstep(0.80, 1.0, xnorm) * (0.62 + 0.38 * (1.0 - low_texture))
    shaped_alpha *= np.clip(downwind_tail, 0.16, 1.0)
    shaped_alpha *= _hybrid_border_fade(alpha.shape) ** 1.45
    shaped_alpha = np.clip(shaped_alpha, 0.0, 1.0)
    shadow, rim_light, optical_depth = _projected_smoke_depth_cues(shaped_alpha, frame_index, seed + 32029)
    age_proxy = aged_zone * _smoothstep(0.035, 0.58, shaped_alpha)
    dense_t = _smoothstep(0.060, 0.54, shaped_alpha + optical_depth * 0.18)
    source_core = (
        _smoothstep(0.38, 0.86, fine + rollup_ridges * 0.10 + streamers * 0.08)
        * (0.72 * fresh_zone + 0.24 * (1.0 - age_proxy))
    )
    old_blue = np.array([100.0, 114.0, 130.0], dtype=np.float32)
    thin_gray = np.array([160.0, 169.0, 172.0], dtype=np.float32)
    milky = np.array([238.0, 236.0, 224.0], dtype=np.float32)
    rgb = old_blue * (1.0 - dense_t[..., None]) + thin_gray * dense_t[..., None]
    aged_blue = np.array([88.0, 101.0, 115.0], dtype=np.float32)
    rgb = rgb * (1.0 - 0.30 * age_proxy[..., None]) + aged_blue * (0.30 * age_proxy[..., None])
    rgb = rgb * (1.0 - source_core[..., None]) + milky * source_core[..., None]
    shadow_color = np.array([70.0, 82.0, 95.0], dtype=np.float32)
    shadow_mix = np.clip(0.34 * shadow + 0.16 * optical_depth, 0.0, 0.48)
    rgb = rgb * (1.0 - shadow_mix[..., None]) + shadow_color * shadow_mix[..., None]
    rgb += rim_light[..., None] * np.array([24.0, 23.0, 18.0], dtype=np.float32)
    rgb += (low_texture[..., None] - 0.5) * 7.0 + (branch_texture[..., None] - 0.5) * 5.0
    out = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
    out[..., :3] = np.clip(np.round(rgb), 0, 242).astype(np.uint8)
    out[..., 3] = np.clip(np.round(shaped_alpha * PHYSICAL_SMOKE_MAX_ALPHA), 0, PHYSICAL_SMOKE_MAX_ALPHA).astype(np.uint8)
    out[..., 3] = np.where(out[..., 3] >= 3, out[..., 3], 0).astype(np.uint8)
    return out


def _raymarched_volume_to_map_rgba(rgba: np.ndarray, map_size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA")
    if image.size != tuple(map(int, map_size)):
        arr = np.asarray(image, dtype=np.float32)
        arr[..., :3] *= arr[..., 3:4] / 255.0
        image = Image.fromarray(np.clip(np.round(arr), 0, 255).astype(np.uint8), mode="RGBA").resize(
            tuple(map(int, map_size)),
            Image.Resampling.BICUBIC,
        )
        arr = np.asarray(image, dtype=np.float32)
        alpha = arr[..., 3:4]
        arr[..., :3] = np.divide(
            arr[..., :3],
            alpha / 255.0,
            out=np.zeros_like(arr[..., :3]),
            where=alpha > 1.0,
        )
    else:
        arr = np.asarray(image, dtype=np.float32)

    arr[..., :3] *= arr[..., 3:4] / 255.0
    reconstruction_radius = max(0.62, min(int(map_size[0]), int(map_size[1])) / 340.0)
    arr = np.asarray(
        Image.fromarray(np.clip(np.round(arr), 0, 255).astype(np.uint8), mode="RGBA").filter(
            ImageFilter.GaussianBlur(radius=float(reconstruction_radius))
        ),
        dtype=np.float32,
    )
    alpha = arr[..., 3:4]
    arr[..., :3] = np.divide(
        arr[..., :3],
        alpha / 255.0,
        out=np.zeros_like(arr[..., :3]),
        where=alpha > 1.0,
    )
    max_alpha = PHYSICAL_SMOKE_MAX_ALPHA / 255.0
    alpha_f = np.clip(arr[..., 3] / 255.0, 0.0, max_alpha)
    alpha_f *= _hybrid_border_fade(alpha_f.shape) ** 1.65
    x, _y = _pixel_grids(alpha_f.shape)
    right_fade = 1.0 - 0.86 * _smoothstep(alpha_f.shape[1] * 0.76, alpha_f.shape[1] - 1.0, x)
    alpha_f *= np.clip(right_fade, 0.0, 1.0)
    arr[..., 3] = np.round(alpha_f * 255.0)
    arr[..., :3] = np.where(arr[..., 3:4] > 0.0, arr[..., :3], 0.0)
    out = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    out[..., 3] = np.where(out[..., 3] >= 2, out[..., 3], 0).astype(np.uint8)
    return out


def render_physical_main_smoke(effect: PhysicalSmokeMainEffect, frame_index: int) -> np.ndarray:
    raw = _render_projected_physical_volume(effect, frame_index)
    current = _raymarched_volume_to_map_rgba(raw, effect.map_size)
    volume_fields = _physical_volume_lane_fields(effect, frame_index)
    current = _postprocess_physical_smoke_rgba(
        current,
        frame_index,
        effect.seed,
        effect.map_size,
        volume_fields,
    )
    volume_structure = _physical_volume_structure_rgba(effect, frame_index, volume_fields)
    if np.any(volume_structure[..., 3] > 0):
        current = _premultiplied_over(current, volume_structure)
        current[..., 3] = np.minimum(current[..., 3], PHYSICAL_SMOKE_MAX_ALPHA).astype(np.uint8)
    source_structure = _physical_structure_enhancement_rgba(
        effect,
        frame_index,
        current[..., 3].astype(np.float32) / 255.0,
    )
    if np.any(source_structure[..., 3] > 0):
        current = _premultiplied_over(current, source_structure)
        current[..., 3] = np.minimum(current[..., 3], PHYSICAL_SMOKE_MAX_ALPHA).astype(np.uint8)
    fresh_smoke = np.asarray(current, dtype=np.uint8).copy()
    history = _physical_history_rgba(effect, frame_index)
    if np.any(history[..., 3] > 0):
        current = _premultiplied_over(history, current)
        current[..., 3] = np.minimum(current[..., 3], PHYSICAL_SMOKE_MAX_ALPHA).astype(np.uint8)
    current = _temporal_blend_physical_smoke(effect, current, frame_index)
    under = _physical_source_glow_rgba(effect, frame_index)
    if np.any(under[..., 3] > 0):
        current = _premultiplied_over(under, current)
        current[..., 3] = np.minimum(current[..., 3], PHYSICAL_SMOKE_MAX_ALPHA).astype(np.uint8)
    effect.previous_render_frame = int(frame_index)
    effect.previous_render_rgba = fresh_smoke
    _record_physical_history(effect, frame_index, fresh_smoke)
    return current


def render_video(args: argparse.Namespace) -> None:
    width, height = map(int, args.size)
    fps = int(args.fps)
    frames = max(1, int(round(float(args.duration) * fps)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plate = terrain_plate(width, height)
    terrain = plate.image
    map_size = (max(16, int(args.hybrid_smoke_width)), max(16, int(args.hybrid_smoke_height)))
    warmup_frames = max(0, int(round(float(args.warmup_seconds) * fps)))
    sim_frames = warmup_frames + frames
    hybrid_sources = make_hybrid_smoke_sources(plate.fire_uv, map_size, total_frames=sim_frames)
    hrrr_guidance = load_hrrr_smoke_guidance(
        Path(args.hrrr_smoke_dir),
        map_size,
        runtime=str(args.hrrr_runtime),
        plot_type=str(args.hrrr_plot_type),
        base_url=str(args.hrrr_base_url),
        fetch=bool(args.fetch_hrrr_smoke),
    )
    if hrrr_guidance is None:
        print(
            "No cached HRRR-Smoke guidance frames found for "
            f"{args.hrrr_runtime}; using deterministic HRRR-style guidance."
        )
    else:
        print(f"Using {hrrr_guidance.source_label}")
    cadence = max(1.0, sim_frames / max(len(hrrr_guidance.frames) - 1, 1)) if hrrr_guidance else 12.0
    hybrid_sim = HybridSmokeSimulator(
        map_size,
        hybrid_sources,
        hrrr_guidance=hrrr_guidance,
        guidance_cadence_frames=cadence,
    )
    physical_effect = None
    if bool(args.physical_smoke):
        physical_effect = make_physical_main_smoke(
            hybrid_sources,
            map_size,
            dims=tuple(map(int, args.physical_smoke_dims)),
            render_size=tuple(map(int, args.physical_render_size)),
            max_sources=int(args.physical_max_sources),
            substeps=int(args.physical_substeps),
            backend=str(args.physical_smoke_backend),
        )
        if physical_effect is None:
            print("Physical 3D smoke unavailable; using layered 2.5D smoke as the main pass.")
        else:
            print(
                "Using physical 3D smoke main pass "
                f"backend={physical_effect.backend}, "
                f"dims={physical_effect.dims}, render={physical_effect.render_size}, "
                f"sources={len(physical_effect.sources)}"
            )

    domain = emitter = step = camera = render_settings = source_xy = None
    smoke_w, smoke_h = int(args.smoke_width), int(args.smoke_height)
    if bool(args.volume_detail):
        domain, emitter, step, camera = make_smoke_domain()
        source_xy = project_smoke_source(camera, smoke_w, smoke_h)
        render_settings = camera["render"]

    for warmup_idx in range(warmup_frames):
        hybrid_sim.step(warmup_idx)
        if physical_effect is not None:
            step_physical_main_smoke(physical_effect, warmup_idx)
        if domain is not None and step is not None and emitter is not None:
            domain.step(step, [emitter])
    preview_frame = None

    with tempfile.TemporaryDirectory(prefix="august_complex_cigar_frames_", dir=args.output.parent) as tmpdir:
        frames_dir = Path(tmpdir)
        for frame_idx in range(frames):
            sim_frame = warmup_frames + frame_idx
            hybrid_sim.step(sim_frame)
            state = hybrid_sim.interpolated_state(0.82)
            fire_bloom_map = hybrid_fire_sources_rgba(
                hybrid_sources,
                sim_frame,
                map_size,
                glow_only=True,
                bloom_scale=1.85,
            )
            fire_bloom_layer = warp_map_layer_to_plate(fire_bloom_map, plate, (width, height))
            terrain_with_bloom = terrain.copy()
            terrain_with_bloom.alpha_composite(fire_bloom_layer)
            if physical_effect is not None:
                step_physical_main_smoke(physical_effect, sim_frame)
                physical_smoke_map = render_physical_main_smoke(physical_effect, sim_frame)
                smoke_map = composite_main_smoke_maps(
                    hybrid_smoke_rgba(state, sim_frame),
                    physical_smoke_map,
                )
            else:
                smoke_map = hybrid_smoke_rgba(state, sim_frame)
            smoke_layer = warp_map_layer_to_plate(smoke_map, plate, (width, height))
            frame = composite_atmospheric_smoke(terrain_with_bloom, smoke_layer)

            if (
                domain is not None
                and step is not None
                and emitter is not None
                and camera is not None
                and render_settings is not None
                and source_xy is not None
            ):
                domain.step(step, [emitter])
                smoke_rgba = np.asarray(
                    domain.render_rgba(
                        smoke_w,
                        smoke_h,
                        camera_pos=camera["camera_pos"],
                        target=camera["target"],
                        up=camera["up"],
                        fovy_deg=camera["fovy_deg"],
                        sun_direction=camera["sun_direction"],
                        settings=render_settings,
                    )
                )
                frame = composite_volume_detail(frame, smoke_rgba, plate.fire_xy, source_xy)

            fire_map = hybrid_fire_sources_rgba(
                hybrid_sources,
                sim_frame,
                map_size,
                bloom_scale=0.52,
                core_alpha_scale=0.82,
            )
            fire_layer = warp_map_layer_to_plate(fire_map, plate, (width, height))
            frame.alpha_composite(fire_layer)
            draw_labels(frame)
            if frame_idx == frames // 2:
                preview_frame = frame.copy()
            frame.save(frames_dir / f"frame_{frame_idx:04d}.png")

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to encode the MP4.")
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "17",
            "-preset",
            "medium",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
            str(args.output),
        ]
        subprocess.run(cmd, check=True)

    (preview_frame or terrain).save(args.preview)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.preview}")


def main() -> None:
    render_video(parse_args())


if __name__ == "__main__":
    main()
