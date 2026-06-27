use std::fmt;

pub const DEFAULT_BRICK_AXIS: usize = 16;
pub const MAX_CPU_VOXELS: usize = 256 * 256 * 256;
const FIELD_COUNT_F32: usize = 10;

#[derive(Debug, Clone)]
pub struct SmokeDomainConfig {
    pub dims: [usize; 3],
    pub voxel_size: [f32; 3],
    pub origin: [f32; 3],
    pub brick_size: [usize; 3],
    pub sparse_threshold: f32,
}

impl SmokeDomainConfig {
    pub fn new(dims: [usize; 3], voxel_size: [f32; 3], origin: [f32; 3]) -> Result<Self, String> {
        let config = Self {
            dims,
            voxel_size,
            origin,
            brick_size: [DEFAULT_BRICK_AXIS; 3],
            sparse_threshold: 1.0e-5,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        for (axis, value) in self.dims.iter().enumerate() {
            if *value < 2 {
                return Err(format!("dims[{axis}] must be >= 2"));
            }
        }
        if self.voxel_count() > MAX_CPU_VOXELS {
            return Err(format!(
                "smoke domain has {} voxels, exceeding CPU reference limit {}",
                self.voxel_count(),
                MAX_CPU_VOXELS
            ));
        }
        for (axis, value) in self.voxel_size.iter().enumerate() {
            if !value.is_finite() || *value <= 0.0 {
                return Err(format!("voxel_size[{axis}] must be finite and > 0"));
            }
        }
        for (axis, value) in self.origin.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!("origin[{axis}] must be finite"));
            }
        }
        for (axis, value) in self.brick_size.iter().enumerate() {
            if *value == 0 {
                return Err(format!("brick_size[{axis}] must be >= 1"));
            }
        }
        if !self.sparse_threshold.is_finite() || self.sparse_threshold < 0.0 {
            return Err("sparse_threshold must be finite and >= 0".to_string());
        }
        Ok(())
    }

    pub fn voxel_count(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }
}

#[derive(Debug, Clone)]
pub struct SmokeEmitter {
    pub center: [f32; 3],
    pub radius: f32,
    pub density_rate: f32,
    pub temperature_rate: f32,
    pub fuel_rate: f32,
    pub soot_rate: f32,
    pub humidity_rate: f32,
    pub emission_rate: f32,
    pub velocity: [f32; 3],
    pub start_time: f32,
    pub end_time: f32,
}

impl Default for SmokeEmitter {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
            density_rate: 1.0,
            temperature_rate: 1.0,
            fuel_rate: 0.0,
            soot_rate: 0.2,
            humidity_rate: 0.0,
            emission_rate: 1.0,
            velocity: [0.0, 1.0, 0.0],
            start_time: 0.0,
            end_time: f32::MAX,
        }
    }
}

impl SmokeEmitter {
    pub fn validate(&self) -> Result<(), String> {
        for (axis, value) in self.center.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!("center[{axis}] must be finite"));
            }
        }
        if !self.radius.is_finite() || self.radius <= 0.0 {
            return Err("radius must be finite and > 0".to_string());
        }
        for (name, value) in [
            ("density_rate", self.density_rate),
            ("temperature_rate", self.temperature_rate),
            ("fuel_rate", self.fuel_rate),
            ("soot_rate", self.soot_rate),
            ("humidity_rate", self.humidity_rate),
            ("emission_rate", self.emission_rate),
            ("start_time", self.start_time),
            ("end_time", self.end_time),
        ] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite"));
            }
        }
        for (axis, value) in self.velocity.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!("velocity[{axis}] must be finite"));
            }
        }
        if self.end_time < self.start_time {
            return Err("end_time must be >= start_time".to_string());
        }
        Ok(())
    }

    pub fn is_active(&self, time_seconds: f32) -> bool {
        time_seconds >= self.start_time && time_seconds <= self.end_time
    }
}

#[derive(Debug, Clone)]
pub struct SmokeStepSettings {
    pub dt: f32,
    pub density_decay: f32,
    pub temperature_decay: f32,
    pub velocity_damping: f32,
    pub diffusion: f32,
    pub buoyancy: f32,
    pub vorticity: f32,
    pub pressure_iterations: u32,
    pub turbulence_strength: f32,
    pub turbulence_seed: u32,
    pub mac_cormack: bool,
    pub mass_conservation: bool,
    pub terrain_collision: bool,
    pub boundary_damping: f32,
    pub wind: [f32; 3],
}

impl Default for SmokeStepSettings {
    fn default() -> Self {
        Self {
            dt: 1.0 / 30.0,
            density_decay: 0.015,
            temperature_decay: 0.08,
            velocity_damping: 0.01,
            diffusion: 0.0005,
            buoyancy: 0.7,
            vorticity: 0.12,
            pressure_iterations: 20,
            turbulence_strength: 0.0,
            turbulence_seed: 0,
            mac_cormack: false,
            mass_conservation: true,
            terrain_collision: true,
            boundary_damping: 0.0,
            wind: [0.0, 0.0, 0.0],
        }
    }
}

impl SmokeStepSettings {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("dt", self.dt),
            ("density_decay", self.density_decay),
            ("temperature_decay", self.temperature_decay),
            ("velocity_damping", self.velocity_damping),
            ("diffusion", self.diffusion),
            ("buoyancy", self.buoyancy),
            ("vorticity", self.vorticity),
            ("turbulence_strength", self.turbulence_strength),
            ("boundary_damping", self.boundary_damping),
        ] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite"));
            }
        }
        if self.dt <= 0.0 {
            return Err("dt must be > 0".to_string());
        }
        if self.density_decay < 0.0
            || self.temperature_decay < 0.0
            || self.velocity_damping < 0.0
            || self.diffusion < 0.0
            || self.vorticity < 0.0
            || self.turbulence_strength < 0.0
        {
            return Err(
                "decay, damping, diffusion, vorticity, and turbulence must be >= 0".to_string(),
            );
        }
        if !(0.0..=1.0).contains(&self.boundary_damping) {
            return Err("boundary_damping must be in [0, 1]".to_string());
        }
        for (axis, value) in self.wind.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!("wind[{axis}] must be finite"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SmokeRenderSettings {
    pub density_scale: f32,
    pub extinction: f32,
    pub scattering: f32,
    pub absorption: f32,
    pub phase_g: f32,
    pub step_size: f32,
    pub max_steps: u32,
    pub self_shadow: bool,
    pub shadow_steps: u32,
    pub shadow_step_size: f32,
    pub jitter_strength: f32,
    pub exposure: f32,
    pub thin_color: [f32; 3],
    pub dense_color: [f32; 3],
    pub soot_absorption: f32,
    pub fire_glow: f32,
}

impl Default for SmokeRenderSettings {
    fn default() -> Self {
        Self {
            density_scale: 1.0,
            extinction: 2.6,
            scattering: 0.85,
            absorption: 0.45,
            phase_g: 0.24,
            step_size: 0.0,
            max_steps: 256,
            self_shadow: true,
            shadow_steps: 20,
            shadow_step_size: 0.0,
            jitter_strength: 0.5,
            exposure: 1.0,
            thin_color: [0.50, 0.54, 0.58],
            dense_color: [0.93, 0.91, 0.82],
            soot_absorption: 0.22,
            fire_glow: 0.35,
        }
    }
}

impl SmokeRenderSettings {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("density_scale", self.density_scale),
            ("extinction", self.extinction),
            ("scattering", self.scattering),
            ("absorption", self.absorption),
            ("phase_g", self.phase_g),
            ("step_size", self.step_size),
            ("shadow_step_size", self.shadow_step_size),
            ("jitter_strength", self.jitter_strength),
            ("exposure", self.exposure),
            ("soot_absorption", self.soot_absorption),
            ("fire_glow", self.fire_glow),
        ] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite"));
            }
        }
        if self.density_scale < 0.0 || self.extinction < 0.0 || self.scattering < 0.0 {
            return Err("density_scale, extinction, and scattering must be >= 0".to_string());
        }
        if self.absorption < 0.0 || self.soot_absorption < 0.0 || self.fire_glow < 0.0 {
            return Err("absorption, soot_absorption, and fire_glow must be >= 0".to_string());
        }
        if !(-0.99..=0.99).contains(&self.phase_g) {
            return Err("phase_g must be in [-0.99, 0.99]".to_string());
        }
        if self.step_size < 0.0 || self.shadow_step_size < 0.0 {
            return Err("step sizes must be >= 0".to_string());
        }
        if self.max_steps == 0 || self.shadow_steps == 0 {
            return Err("max_steps and shadow_steps must be >= 1".to_string());
        }
        if !(0.0..=1.0).contains(&self.jitter_strength) {
            return Err("jitter_strength must be in [0, 1]".to_string());
        }
        for (name, color) in [
            ("thin_color", self.thin_color),
            ("dense_color", self.dense_color),
        ] {
            for (axis, value) in color.iter().enumerate() {
                if !value.is_finite() || *value < 0.0 {
                    return Err(format!("{name}[{axis}] must be finite and >= 0"));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct SmokeMemoryReport {
    pub voxel_count: usize,
    pub dense_bytes: usize,
    pub active_bricks: usize,
    pub total_bricks: usize,
    pub sparse_bytes_estimate: usize,
    pub utilization: f32,
    pub time_seconds: f32,
    pub frame_index: u64,
}

#[derive(Clone)]
pub struct SmokeVolume {
    pub config: SmokeDomainConfig,
    pub density: Vec<f32>,
    pub velocity: Vec<f32>,
    pub temperature: Vec<f32>,
    pub pressure: Vec<f32>,
    pub fuel: Vec<f32>,
    pub soot: Vec<f32>,
    pub humidity: Vec<f32>,
    pub particle_age: Vec<f32>,
    pub emission_rate: Vec<f32>,
    pub time_seconds: f32,
    pub frame_index: u64,
}

impl SmokeVolume {
    pub fn new(config: SmokeDomainConfig) -> Result<Self, String> {
        config.validate()?;
        let voxel_count = config.voxel_count();
        Ok(Self {
            config,
            density: vec![0.0; voxel_count],
            velocity: vec![0.0; voxel_count * 3],
            temperature: vec![0.0; voxel_count],
            pressure: vec![0.0; voxel_count],
            fuel: vec![0.0; voxel_count],
            soot: vec![0.0; voxel_count],
            humidity: vec![0.0; voxel_count],
            particle_age: vec![-1.0; voxel_count],
            emission_rate: vec![0.0; voxel_count],
            time_seconds: 0.0,
            frame_index: 0,
        })
    }

    pub fn dims(&self) -> [usize; 3] {
        self.config.dims
    }

    pub fn voxel_count(&self) -> usize {
        self.config.voxel_count()
    }

    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.config.dims[1] + y) * self.config.dims[0] + x
    }

    pub fn world_position(&self, x: usize, y: usize, z: usize) -> [f32; 3] {
        [
            self.config.origin[0] + (x as f32 + 0.5) * self.config.voxel_size[0],
            self.config.origin[1] + (y as f32 + 0.5) * self.config.voxel_size[1],
            self.config.origin[2] + (z as f32 + 0.5) * self.config.voxel_size[2],
        ]
    }

    pub fn grid_coord_from_world(&self, position: [f32; 3]) -> [f32; 3] {
        [
            (position[0] - self.config.origin[0]) / self.config.voxel_size[0] - 0.5,
            (position[1] - self.config.origin[1]) / self.config.voxel_size[1] - 0.5,
            (position[2] - self.config.origin[2]) / self.config.voxel_size[2] - 0.5,
        ]
    }

    pub fn bounds_min(&self) -> [f32; 3] {
        self.config.origin
    }

    pub fn bounds_max(&self) -> [f32; 3] {
        [
            self.config.origin[0] + self.config.dims[0] as f32 * self.config.voxel_size[0],
            self.config.origin[1] + self.config.dims[1] as f32 * self.config.voxel_size[1],
            self.config.origin[2] + self.config.dims[2] as f32 * self.config.voxel_size[2],
        ]
    }

    pub fn mass(&self) -> f32 {
        self.density.iter().copied().sum()
    }

    pub fn max_density(&self) -> f32 {
        self.density.iter().copied().fold(0.0, f32::max)
    }

    pub fn dense_bytes(&self) -> usize {
        self.voxel_count() * FIELD_COUNT_F32 * std::mem::size_of::<f32>()
    }

    pub fn active_brick_count(&self) -> usize {
        let [nx, ny, nz] = self.config.dims;
        let [bx, by, bz] = self.config.brick_size;
        let bricks_x = nx.div_ceil(bx);
        let bricks_y = ny.div_ceil(by);
        let bricks_z = nz.div_ceil(bz);
        let threshold = self.config.sparse_threshold;
        let mut active = 0;

        for brick_z in 0..bricks_z {
            for brick_y in 0..bricks_y {
                for brick_x in 0..bricks_x {
                    let x0 = brick_x * bx;
                    let y0 = brick_y * by;
                    let z0 = brick_z * bz;
                    let x1 = (x0 + bx).min(nx);
                    let y1 = (y0 + by).min(ny);
                    let z1 = (z0 + bz).min(nz);
                    let mut brick_active = false;
                    'cells: for z in z0..z1 {
                        for y in y0..y1 {
                            for x in x0..x1 {
                                let idx = self.index(x, y, z);
                                if self.density[idx].abs() > threshold
                                    || self.temperature[idx].abs() > threshold
                                    || self.fuel[idx].abs() > threshold
                                    || self.soot[idx].abs() > threshold
                                    || self.emission_rate[idx].abs() > threshold
                                {
                                    brick_active = true;
                                    break 'cells;
                                }
                            }
                        }
                    }
                    if brick_active {
                        active += 1;
                    }
                }
            }
        }
        active
    }

    pub fn memory_report(&self) -> SmokeMemoryReport {
        let [nx, ny, nz] = self.config.dims;
        let [bx, by, bz] = self.config.brick_size;
        let total_bricks = nx.div_ceil(bx) * ny.div_ceil(by) * nz.div_ceil(bz);
        let active_bricks = self.active_brick_count();
        let brick_voxels = bx * by * bz;
        let sparse_bytes_estimate =
            active_bricks * (brick_voxels * FIELD_COUNT_F32 * std::mem::size_of::<f32>() + 64);
        let dense_bytes = self.dense_bytes();
        let utilization = if total_bricks > 0 {
            active_bricks as f32 / total_bricks as f32
        } else {
            0.0
        };

        SmokeMemoryReport {
            voxel_count: self.voxel_count(),
            dense_bytes,
            active_bricks,
            total_bricks,
            sparse_bytes_estimate: sparse_bytes_estimate.min(dense_bytes),
            utilization,
            time_seconds: self.time_seconds,
            frame_index: self.frame_index,
        }
    }

    pub fn set_density(&mut self, density: Vec<f32>) -> Result<(), String> {
        if density.len() != self.voxel_count() {
            return Err(format!(
                "density length {} does not match voxel_count {}",
                density.len(),
                self.voxel_count()
            ));
        }
        if density.iter().any(|v| !v.is_finite()) {
            return Err("density contains non-finite values".to_string());
        }
        self.density = density;
        for (age, d) in self.particle_age.iter_mut().zip(self.density.iter()) {
            *age = if *d > self.config.sparse_threshold {
                0.0
            } else {
                -1.0
            };
        }
        Ok(())
    }

    pub fn set_velocity(&mut self, velocity: Vec<f32>) -> Result<(), String> {
        if velocity.len() != self.voxel_count() * 3 {
            return Err(format!(
                "velocity length {} does not match voxel_count*3 {}",
                velocity.len(),
                self.voxel_count() * 3
            ));
        }
        if velocity.iter().any(|v| !v.is_finite()) {
            return Err("velocity contains non-finite values".to_string());
        }
        self.velocity = velocity;
        Ok(())
    }
}

impl fmt::Debug for SmokeVolume {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SmokeVolume")
            .field("dims", &self.config.dims)
            .field("voxel_size", &self.config.voxel_size)
            .field("origin", &self.config.origin)
            .field("time_seconds", &self.time_seconds)
            .field("frame_index", &self.frame_index)
            .field("mass", &self.mass())
            .finish()
    }
}
