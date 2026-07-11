from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass
class AtmosphericSmokeCube:
    density: np.ndarray
    velocity: np.ndarray | None = ...
    voxel_size: tuple[float, float, float] = ...
    origin: tuple[float, float, float] = ...
    vertical_levels: tuple[float, ...] = ...
    times: tuple[str, ...] = ...
    crs: str | None = ...
    source: str | None = ...
    def to_domain(self) -> Any: ...


class SmokeEmitter:
    def __init__(
        self,
        center: tuple[float, float, float] = ...,
        radius: float = ...,
        density_rate: float = ...,
        temperature_rate: float = ...,
        fuel_rate: float = ...,
        soot_rate: float = ...,
        humidity_rate: float = ...,
        emission_rate: float = ...,
        velocity: tuple[float, float, float] = ...,
        start_time: float = ...,
        end_time: float = ...,
    ) -> None: ...


class SmokeStepSettings:
    def __init__(
        self,
        dt: float = ...,
        density_decay: float = ...,
        temperature_decay: float = ...,
        velocity_damping: float = ...,
        diffusion: float = ...,
        buoyancy: float = ...,
        vorticity: float = ...,
        pressure_iterations: int = ...,
        turbulence_strength: float = ...,
        turbulence_seed: int = ...,
        mac_cormack: bool = ...,
        mass_conservation: bool = ...,
        terrain_collision: bool = ...,
        boundary_damping: float = ...,
        wind: tuple[float, float, float] = ...,
    ) -> None: ...


class SmokeRenderSettings:
    def __init__(
        self,
        density_scale: float = ...,
        extinction: float = ...,
        scattering: float = ...,
        absorption: float = ...,
        phase_g: float = ...,
        step_size: float = ...,
        max_steps: int = ...,
        self_shadow: bool = ...,
        shadow_steps: int = ...,
        shadow_step_size: float = ...,
        jitter_strength: float = ...,
        exposure: float = ...,
        thin_color: tuple[float, float, float] = ...,
        dense_color: tuple[float, float, float] = ...,
        soot_absorption: float = ...,
        fire_glow: float = ...,
    ) -> None: ...


class SmokeDomain:
    dims: tuple[int, int, int]
    voxel_size: tuple[float, float, float]
    origin: tuple[float, float, float]
    time_seconds: float
    frame_index: int
    def __init__(
        self,
        dims: tuple[int, int, int],
        voxel_size: tuple[float, float, float] = ...,
        origin: tuple[float, float, float] = ...,
        brick_size: tuple[int, int, int] = ...,
        sparse_threshold: float = ...,
    ) -> None: ...
    @classmethod
    def from_density(
        cls,
        density: np.ndarray,
        voxel_size: tuple[float, float, float] = ...,
        origin: tuple[float, float, float] = ...,
    ) -> SmokeDomain: ...
    def set_density(self, density: np.ndarray) -> None: ...
    def set_velocity(self, velocity: np.ndarray) -> None: ...
    def add_emitter(self, emitter: SmokeEmitter, dt: float) -> None: ...
    def step(
        self,
        settings: SmokeStepSettings,
        emitters: Sequence[SmokeEmitter] | None = ...,
    ) -> None: ...
    def to_density_numpy(self) -> np.ndarray: ...
    def to_velocity_numpy(self) -> np.ndarray: ...
    def to_particle_age_numpy(self) -> np.ndarray: ...
    def sample_density(self, position: tuple[float, float, float]) -> float: ...
    def sample_temperature(self, position: tuple[float, float, float]) -> float: ...
    def memory_report(self) -> dict[str, Any]: ...
    def physics_report(self) -> dict[str, Any]: ...
    def render_rgba(
        self,
        width: int,
        height: int,
        camera_pos: tuple[float, float, float],
        target: tuple[float, float, float],
        up: tuple[float, float, float] = ...,
        fovy_deg: float = ...,
        sun_direction: tuple[float, float, float] = ...,
        settings: SmokeRenderSettings | None = ...,
        certificate: bool | str | Path = ...,
    ) -> np.ndarray: ...
    def render_projection_rgba(
        self,
        width: int,
        height: int,
        view_direction: tuple[float, float, float] = ...,
        sun_direction: tuple[float, float, float] = ...,
        settings: SmokeRenderSettings | None = ...,
        certificate: bool | str | Path = ...,
    ) -> np.ndarray: ...


def native_smoke_available() -> bool: ...
def domain_from_density(
    density: np.ndarray,
    *,
    voxel_size: tuple[float, float, float] = ...,
    origin: tuple[float, float, float] = ...,
) -> Any: ...
def cube_from_arrays(
    density: np.ndarray,
    *,
    velocity: np.ndarray | None = ...,
    voxel_size: tuple[float, float, float] = ...,
    origin: tuple[float, float, float] = ...,
    vertical_levels: Sequence[float] = ...,
    times: Sequence[str] = ...,
    crs: str | None = ...,
    source: str | None = ...,
) -> AtmosphericSmokeCube: ...
def load_npz_volume(
    path: str | Path,
    *,
    density_key: str = ...,
    velocity_key: str = ...,
) -> AtmosphericSmokeCube: ...
def save_npz_volume(path: str | Path, domain_or_cube: Any) -> None: ...
def load_xarray_volume(
    source: Any,
    *,
    density_var: str = ...,
    wind_vars: tuple[str, str, str] | None = ...,
    time_index: int | None = ...,
    voxel_size: tuple[float, float, float] = ...,
    origin: tuple[float, float, float] = ...,
) -> AtmosphericSmokeCube: ...
def interpolate_density_frames(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray: ...
def capability_report() -> dict[str, Any]: ...
