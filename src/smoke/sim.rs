use glam::Vec3;

use super::sampling::{index, sample_scalar, sample_vector, sample_vector_component, smoothstep};
use super::types::{SmokeEmitter, SmokeStepSettings, SmokeVolume};

impl SmokeVolume {
    pub fn add_emitter(&mut self, emitter: &SmokeEmitter, dt: f32) -> Result<(), String> {
        emitter.validate()?;
        if !dt.is_finite() || dt <= 0.0 {
            return Err("dt must be finite and > 0".to_string());
        }

        let [nx, ny, nz] = self.config.dims;
        let radius = emitter.radius.max(1.0e-6);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = self.index(x, y, z);
                    let p = self.world_position(x, y, z);
                    let d = Vec3::from_array(p).distance(Vec3::from_array(emitter.center));
                    if d > radius {
                        continue;
                    }
                    let falloff = 1.0 - smoothstep(0.0, radius, d);
                    let amount = dt * falloff;
                    self.density[idx] =
                        (self.density[idx] + emitter.density_rate * amount).max(0.0);
                    self.temperature[idx] =
                        (self.temperature[idx] + emitter.temperature_rate * amount).max(0.0);
                    self.fuel[idx] = (self.fuel[idx] + emitter.fuel_rate * amount).max(0.0);
                    self.soot[idx] = (self.soot[idx] + emitter.soot_rate * amount).max(0.0);
                    self.humidity[idx] =
                        (self.humidity[idx] + emitter.humidity_rate * amount).max(0.0);
                    self.emission_rate[idx] += emitter.emission_rate * falloff;
                    self.particle_age[idx] = 0.0;

                    for component in 0..3 {
                        let vi = idx * 3 + component;
                        self.velocity[vi] += emitter.velocity[component] * amount;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn step(
        &mut self,
        settings: &SmokeStepSettings,
        emitters: &[SmokeEmitter],
    ) -> Result<(), String> {
        settings.validate()?;
        for emitter in emitters {
            emitter.validate()?;
        }

        self.emission_rate.fill(0.0);
        for emitter in emitters {
            if emitter.is_active(self.time_seconds) {
                self.add_emitter(emitter, settings.dt)?;
            }
        }

        self.apply_forces(settings);
        let velocity_before_advection = self.velocity.clone();
        self.velocity = advect_vector(
            &velocity_before_advection,
            &velocity_before_advection,
            self.config.dims,
            self.config.voxel_size,
            settings.dt,
        );
        diffuse_vector(
            &mut self.velocity,
            self.config.dims,
            settings.diffusion,
            settings.dt,
        );
        if settings.vorticity > 0.0 {
            self.apply_vorticity_confinement(settings.vorticity, settings.dt);
        }
        self.project(settings.pressure_iterations.max(1));
        self.apply_boundary_conditions(settings);
        self.apply_lane_advection_shear(settings);

        let density_mass_before = self.mass();
        self.density = advect_scalar(
            &self.density,
            &self.velocity,
            self.config.dims,
            self.config.voxel_size,
            settings.dt,
            settings.mac_cormack,
        );
        if settings.mass_conservation {
            scale_to_mass(&mut self.density, density_mass_before);
        }
        self.temperature = advect_scalar(
            &self.temperature,
            &self.velocity,
            self.config.dims,
            self.config.voxel_size,
            settings.dt,
            settings.mac_cormack,
        );
        self.fuel = advect_scalar(
            &self.fuel,
            &self.velocity,
            self.config.dims,
            self.config.voxel_size,
            settings.dt,
            settings.mac_cormack,
        );
        self.soot = advect_scalar(
            &self.soot,
            &self.velocity,
            self.config.dims,
            self.config.voxel_size,
            settings.dt,
            settings.mac_cormack,
        );
        self.humidity = advect_scalar(
            &self.humidity,
            &self.velocity,
            self.config.dims,
            self.config.voxel_size,
            settings.dt,
            settings.mac_cormack,
        );

        self.apply_subgrid_density_eddies(settings);
        self.apply_scalar_diffusion(settings.diffusion, settings.dt);
        self.apply_decay_and_age(settings);
        self.project((settings.pressure_iterations / 2).max(1));
        self.apply_boundary_conditions(settings);
        self.time_seconds += settings.dt;
        self.frame_index += 1;
        Ok(())
    }

    pub fn divergence_l2(&self) -> f32 {
        let divergence =
            compute_divergence(&self.velocity, self.config.dims, self.config.voxel_size);
        let sum_sq = divergence.iter().map(|v| v * v).sum::<f32>();
        (sum_sq / divergence.len().max(1) as f32).sqrt()
    }

    pub fn sample_density_world(&self, position: [f32; 3]) -> f32 {
        let p = self.grid_coord_from_world(position);
        sample_scalar(&self.density, self.config.dims, p)
    }

    pub fn sample_temperature_world(&self, position: [f32; 3]) -> f32 {
        let p = self.grid_coord_from_world(position);
        sample_scalar(&self.temperature, self.config.dims, p)
    }

    fn apply_forces(&mut self, settings: &SmokeStepSettings) {
        let [nx, ny, nz] = self.config.dims;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = self.index(x, y, z);
                    let vi = idx * 3;
                    self.velocity[vi] += settings.wind[0] * settings.dt;
                    self.velocity[vi + 1] += (settings.wind[1]
                        + self.temperature[idx] * settings.buoyancy)
                        * settings.dt;
                    self.velocity[vi + 2] += settings.wind[2] * settings.dt;
                    if settings.velocity_damping > 0.0 {
                        let damping = (-settings.velocity_damping * settings.dt).exp();
                        self.velocity[vi] *= damping;
                        self.velocity[vi + 1] *= damping;
                        self.velocity[vi + 2] *= damping;
                    }
                    if settings.turbulence_strength > 0.0 {
                        let xf = x as f32 / (nx.saturating_sub(1)).max(1) as f32;
                        let yf = y as f32 / (ny.saturating_sub(1)).max(1) as f32;
                        let zf = z as f32 / (nz.saturating_sub(1)).max(1) as f32;
                        let seed_phase = settings.turbulence_seed as f32 * 0.000_137;
                        let t = self.time_seconds;
                        let amp = settings.turbulence_strength * settings.dt;
                        let altitude_gain = (0.45 + 0.75 * yf).clamp(0.35, 1.20);
                        let lane_a = (xf * 9.6 + zf * 4.2 + yf * 1.6 + t * 0.52 + seed_phase).sin();
                        let lane_b =
                            (zf * 7.4 - xf * 5.1 + yf * 2.7 - t * 0.37 + seed_phase * 1.7).cos();
                        let roll = ((xf + zf) * 3.9 - yf * 5.2 + t * 0.29 + seed_phase * 0.6).sin();
                        self.velocity[vi] += (0.62 * lane_a + 0.28 * roll) * amp * altitude_gain;
                        self.velocity[vi + 1] += (0.08 * lane_b - 0.05 * roll) * amp;
                        self.velocity[vi + 2] +=
                            (-0.56 * lane_b + 0.26 * lane_a) * amp * altitude_gain;
                        let wind_len = (settings.wind[0] * settings.wind[0]
                            + settings.wind[2] * settings.wind[2])
                            .sqrt();
                        if wind_len > 1.0e-6 {
                            let wind_x = settings.wind[0] / wind_len;
                            let wind_z = settings.wind[2] / wind_len;
                            let cross_x = -wind_z;
                            let cross_z = wind_x;
                            let along = x as f32 * wind_x + z as f32 * wind_z;
                            let cross_coord = x as f32 * cross_x + z as f32 * cross_z;
                            let lane_phase =
                                along * 0.34 + cross_coord * 0.72 + t * 0.34 + seed_phase * 11.0;
                            let lane_force = lane_phase.sin()
                                + 0.45 * (lane_phase * 0.53 + z as f32 * 0.29).sin();
                            let speed_lane =
                                0.5 + 0.5 * (lane_phase * 0.41 + x as f32 * 0.18).cos();
                            self.velocity[vi] += cross_x * lane_force * amp * 0.82 * altitude_gain
                                + wind_x * speed_lane * amp * 0.30 * altitude_gain;
                            self.velocity[vi + 2] +=
                                cross_z * lane_force * amp * 0.82 * altitude_gain
                                    + wind_z * speed_lane * amp * 0.30 * altitude_gain;
                            let shear = (yf - 0.42) * amp * 1.35;
                            self.velocity[vi] += cross_x * shear;
                            self.velocity[vi + 2] += cross_z * shear;
                        }
                    }
                }
            }
        }
    }

    fn apply_scalar_diffusion(&mut self, diffusion: f32, dt: f32) {
        if diffusion <= 0.0 {
            return;
        }
        diffuse_scalar_in_place(&mut self.density, self.config.dims, diffusion, dt);
        diffuse_scalar_in_place(&mut self.temperature, self.config.dims, diffusion, dt);
        diffuse_scalar_in_place(&mut self.fuel, self.config.dims, diffusion, dt);
        diffuse_scalar_in_place(&mut self.soot, self.config.dims, diffusion, dt);
        diffuse_scalar_in_place(&mut self.humidity, self.config.dims, diffusion, dt);
    }

    fn apply_decay_and_age(&mut self, settings: &SmokeStepSettings) {
        let temperature_decay = (-settings.temperature_decay * settings.dt).exp();
        for idx in 0..self.voxel_count() {
            let age_t = smoothstep(7.0, 36.0, self.particle_age[idx].max(0.0));
            let density_decay = (-settings.density_decay * settings.dt * (1.0 + 3.0 * age_t)).exp();
            let soot_decay = (-settings.density_decay * settings.dt * (0.42 + 1.15 * age_t)).exp();
            self.density[idx] *= density_decay;
            self.temperature[idx] *= temperature_decay;
            self.fuel[idx] *= density_decay;
            self.soot[idx] *= soot_decay;
            if self.density[idx] > self.config.sparse_threshold {
                self.particle_age[idx] = if self.particle_age[idx] < 0.0 {
                    0.0
                } else {
                    self.particle_age[idx] + settings.dt
                };
            } else {
                self.particle_age[idx] = -1.0;
            }
        }
    }

    fn project(&mut self, iterations: u32) {
        let dims = self.config.dims;
        let divergence = compute_divergence(&self.velocity, dims, self.config.voxel_size);
        self.pressure.fill(0.0);
        let mut next = self.pressure.clone();

        for _ in 0..iterations {
            for z in 1..dims[2] - 1 {
                for y in 1..dims[1] - 1 {
                    for x in 1..dims[0] - 1 {
                        let idx = index(dims, x, y, z);
                        let sum = self.pressure[index(dims, x - 1, y, z)]
                            + self.pressure[index(dims, x + 1, y, z)]
                            + self.pressure[index(dims, x, y - 1, z)]
                            + self.pressure[index(dims, x, y + 1, z)]
                            + self.pressure[index(dims, x, y, z - 1)]
                            + self.pressure[index(dims, x, y, z + 1)];
                        next[idx] = (sum - divergence[idx]) / 6.0;
                    }
                }
            }
            std::mem::swap(&mut self.pressure, &mut next);
        }

        for z in 1..dims[2] - 1 {
            for y in 1..dims[1] - 1 {
                for x in 1..dims[0] - 1 {
                    let idx = index(dims, x, y, z);
                    let grad_x = (self.pressure[index(dims, x + 1, y, z)]
                        - self.pressure[index(dims, x - 1, y, z)])
                        / (2.0 * self.config.voxel_size[0]);
                    let grad_y = (self.pressure[index(dims, x, y + 1, z)]
                        - self.pressure[index(dims, x, y - 1, z)])
                        / (2.0 * self.config.voxel_size[1]);
                    let grad_z = (self.pressure[index(dims, x, y, z + 1)]
                        - self.pressure[index(dims, x, y, z - 1)])
                        / (2.0 * self.config.voxel_size[2]);
                    let vi = idx * 3;
                    self.velocity[vi] -= grad_x;
                    self.velocity[vi + 1] -= grad_y;
                    self.velocity[vi + 2] -= grad_z;
                }
            }
        }
    }

    fn apply_lane_advection_shear(&mut self, settings: &SmokeStepSettings) {
        if settings.turbulence_strength <= 0.0 {
            return;
        }
        let [nx, ny, nz] = self.config.dims;
        let wind_len =
            (settings.wind[0] * settings.wind[0] + settings.wind[2] * settings.wind[2]).sqrt();
        if wind_len <= 1.0e-6 {
            return;
        }
        let wind_x = settings.wind[0] / wind_len;
        let wind_z = settings.wind[2] / wind_len;
        let cross_x = -wind_z;
        let cross_z = wind_x;
        let amp = settings.turbulence_strength * settings.dt;
        let seed_phase = settings.turbulence_seed as f32 * 0.0027;
        let mut total_mass = 0.0f32;
        let mut centroid_x = 0.0f32;
        let mut centroid_z = 0.0f32;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = self.index(x, y, z);
                    let mass = self.density[idx].max(0.0);
                    total_mass += mass;
                    centroid_x += x as f32 * mass;
                    centroid_z += z as f32 * mass;
                }
            }
        }
        if total_mass > 1.0e-6 {
            centroid_x /= total_mass;
            centroid_z /= total_mass;
        }
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = self.index(x, y, z);
                    let active = smoothstep(
                        self.config.sparse_threshold,
                        (self.config.sparse_threshold * 60.0).max(0.012),
                        self.density[idx],
                    );
                    if active <= 0.0 {
                        continue;
                    }
                    let along = x as f32 * wind_x + z as f32 * wind_z;
                    let cross_coord = x as f32 * cross_x + z as f32 * cross_z;
                    let lane_phase = along * 0.23
                        + cross_coord * 0.49
                        + self.frame_index as f32 * 0.105
                        + seed_phase;
                    let lane_force =
                        lane_phase.sin() + 0.58 * (lane_phase * 0.41 + y as f32 * 0.74).sin();
                    let altitude = y as f32 / (ny.saturating_sub(1)).max(1) as f32;
                    let altitude_shear = (altitude - 0.44) * 0.95;
                    let force = (lane_force * 2.75 + altitude_shear * 1.45) * active * amp;
                    let vi = idx * 3;
                    self.velocity[vi] += cross_x * force;
                    self.velocity[vi + 2] += cross_z * force;
                    let slab_phase = along * 0.17 - cross_coord * 0.31
                        + y as f32 * 1.12
                        + self.frame_index as f32 * 0.043
                        + settings.turbulence_seed as f32 * 0.0021;
                    let slab_lane =
                        slab_phase.sin() + 0.42 * (slab_phase * 0.53 + along * 0.09).sin();
                    let slab_split = ((altitude - 0.50) * 2.55 + slab_lane * 0.58) * active * amp;
                    self.velocity[vi] += cross_x * slab_split * 1.90;
                    self.velocity[vi + 2] += cross_z * slab_split * 1.90;
                    let speed_split = (slab_phase * 0.39 + y as f32 * 0.67).sin() * active * amp;
                    self.velocity[vi] += wind_x * speed_split * 0.52;
                    self.velocity[vi + 2] += wind_z * speed_split * 0.52;
                    if total_mass > 1.0e-6 {
                        let eddies = [
                            (5.5f32, 5.4f32, 1.0f32, 1.85f32),
                            (11.5f32, 7.6f32, -1.0f32, 1.58f32),
                            (19.0f32, 10.2f32, 1.0f32, 1.30f32),
                            (28.0f32, 13.0f32, -1.0f32, 1.05f32),
                        ];
                        let altitude_gain = 0.55 + 0.75 * altitude;
                        for (eddy_index, (distance, radius, side, strength)) in
                            eddies.iter().enumerate()
                        {
                            let phase = self.frame_index as f32
                                * (0.035 + eddy_index as f32 * 0.006)
                                + settings.turbulence_seed as f32 * 0.0013;
                            let center_x = centroid_x
                                + wind_x * *distance
                                + cross_x * *side * *radius * (0.40 + 0.20 * phase.sin());
                            let center_z = centroid_z
                                + wind_z * *distance
                                + cross_z * *side * *radius * (0.40 + 0.20 * phase.cos());
                            let dx = x as f32 - center_x;
                            let dz = z as f32 - center_z;
                            let r2 = dx * dx + dz * dz;
                            let envelope = (-r2 / (2.0 * *radius * *radius)).exp() * active;
                            let inv_r = 1.0 / (r2 + 1.0).sqrt();
                            let spin = *side * *strength * amp * envelope * altitude_gain;
                            self.velocity[vi] += -dz * inv_r * spin;
                            self.velocity[vi + 2] += dx * inv_r * spin;
                        }
                    }
                }
            }
        }
    }

    fn apply_subgrid_density_eddies(&mut self, settings: &SmokeStepSettings) {
        if settings.turbulence_strength <= 0.0 {
            return;
        }
        let [nx, ny, nz] = self.config.dims;
        let wind_len =
            (settings.wind[0] * settings.wind[0] + settings.wind[2] * settings.wind[2]).sqrt();
        let wind_len = wind_len.max(1.0e-6);
        let wind_x = settings.wind[0] / wind_len;
        let wind_z = settings.wind[2] / wind_len;
        let cross_x = -wind_z;
        let cross_z = wind_x;
        let seed_phase = settings.turbulence_seed as f32 * 0.0019;
        let t = self.frame_index as f32 * 0.046;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = self.index(x, y, z);
                    let active = smoothstep(
                        self.config.sparse_threshold,
                        (self.config.sparse_threshold * 90.0).max(0.018),
                        self.density[idx],
                    );
                    if active <= 0.0 {
                        continue;
                    }
                    let xf = x as f32;
                    let yf = y as f32;
                    let zf = z as f32;
                    let phase = xf * 0.18
                        + zf * 0.27
                        + yf * 0.72
                        + (xf * 0.043 + zf * 0.071 + t + seed_phase).sin() * 1.7
                        + t
                        + seed_phase;
                    let ribbons = 0.5 + 0.5 * phase.sin();
                    let sheets = 0.5 + 0.5 * (phase * 0.47 - zf * 0.16 + yf * 0.51).sin();
                    let voids = smoothstep(0.45, 0.84, 1.0 - ribbons)
                        * smoothstep(0.34, 0.76, 1.0 - sheets)
                        * active;
                    let ridges =
                        smoothstep(0.62, 0.94, ribbons) * smoothstep(0.48, 0.90, sheets) * active;
                    let age_t = smoothstep(2.0, 28.0, self.particle_age[idx].max(0.0));
                    let void_strength = 0.62 + 0.32 * age_t;
                    let ridge_strength = 0.075 - 0.045 * age_t;
                    let along = xf * wind_x + zf * wind_z;
                    let cross_coord = xf * cross_x + zf * cross_z;
                    let broad = 0.5 + 0.5 * (xf * 0.043 + zf * 0.071 + t + seed_phase).sin();
                    let channel_phase = along * 0.115
                        + cross_coord * 0.52
                        + broad * 5.4
                        + (yf * 0.62 + along * 0.035).sin() * 0.85
                        + self.frame_index as f32 * 0.033
                        + settings.turbulence_seed as f32 * 0.0023;
                    let channel_wave = 0.5
                        + 0.5 * channel_phase.sin()
                        + 0.28 * (channel_phase * 0.47 - cross_coord * 0.19 + yf * 0.34).sin();
                    let entrainment = smoothstep(0.58, 1.06, channel_wave);
                    let lateral_slots =
                        smoothstep(0.50, 0.94, 1.0 - (0.62 * ribbons + 0.38 * sheets));
                    let core_protect = 1.0 - 0.56 * smoothstep(0.72, 1.75, self.density[idx]);
                    let aged_sheet = (0.28 + 0.72 * age_t) * active * core_protect;
                    let clear_air =
                        (entrainment * (0.54 + 0.46 * lateral_slots) * aged_sheet).clamp(0.0, 1.0);
                    let channel_void = (smoothstep(0.42, 0.86, 1.0 - channel_wave)
                        * (0.55 + 0.45 * lateral_slots)
                        * active
                        * (0.42 + 0.58 * age_t)
                        * core_protect)
                        .clamp(0.0, 1.0);
                    let gain = (1.0 - void_strength * voids + ridge_strength * ridges)
                        * (1.0 - (0.024 + 0.055 * age_t) * clear_air)
                        * (1.0 - (0.045 + 0.070 * age_t) * channel_void);
                    self.density[idx] = (self.density[idx] * gain).clamp(0.0, 8.0);
                    self.humidity[idx] = (self.humidity[idx]
                        * (1.0
                            - (0.15 + 0.10 * age_t) * voids
                            - (0.024 + 0.055 * age_t) * clear_air
                            - (0.045 + 0.070 * age_t) * channel_void))
                        .max(0.0);
                }
            }
        }
    }

    fn apply_boundary_conditions(&mut self, settings: &SmokeStepSettings) {
        let [nx, ny, nz] = self.config.dims;
        let keep = 1.0 - settings.boundary_damping;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = self.index(x, y, z);
                    let vi = idx * 3;
                    if x == 0 || x == nx - 1 {
                        self.velocity[vi] = 0.0;
                        self.density[idx] *= 0.58;
                        self.temperature[idx] *= 0.70;
                    } else if x == 1 || x == nx - 2 {
                        self.density[idx] *= 0.78;
                        self.temperature[idx] *= 0.86;
                    }
                    if y == 0 || y == ny - 1 {
                        self.velocity[vi + 1] = 0.0;
                    }
                    if z == 0 || z == nz - 1 {
                        self.velocity[vi + 2] = 0.0;
                        self.density[idx] *= 0.58;
                        self.temperature[idx] *= 0.70;
                    } else if z == 1 || z == nz - 2 {
                        self.density[idx] *= 0.78;
                        self.temperature[idx] *= 0.86;
                    }
                    if settings.terrain_collision && y == 0 {
                        self.density[idx] *= keep;
                        self.temperature[idx] *= keep;
                    }
                }
            }
        }
    }

    fn apply_vorticity_confinement(&mut self, strength: f32, dt: f32) {
        let dims = self.config.dims;
        let mut curl = vec![[0.0; 3]; self.voxel_count()];
        let mut mag = vec![0.0; self.voxel_count()];
        for z in 1..dims[2] - 1 {
            for y in 1..dims[1] - 1 {
                for x in 1..dims[0] - 1 {
                    let c = curl_at(&self.velocity, dims, self.config.voxel_size, x, y, z);
                    let idx = index(dims, x, y, z);
                    curl[idx] = c;
                    mag[idx] = Vec3::from_array(c).length();
                }
            }
        }
        for z in 2..dims[2] - 2 {
            for y in 2..dims[1] - 2 {
                for x in 2..dims[0] - 2 {
                    let idx = index(dims, x, y, z);
                    let grad = Vec3::new(
                        mag[index(dims, x + 1, y, z)] - mag[index(dims, x - 1, y, z)],
                        mag[index(dims, x, y + 1, z)] - mag[index(dims, x, y - 1, z)],
                        mag[index(dims, x, y, z + 1)] - mag[index(dims, x, y, z - 1)],
                    );
                    let n = if grad.length_squared() > 1.0e-12 {
                        grad.normalize()
                    } else {
                        Vec3::ZERO
                    };
                    let force = n.cross(Vec3::from_array(curl[idx])) * strength * dt;
                    let vi = idx * 3;
                    self.velocity[vi] += force.x;
                    self.velocity[vi + 1] += force.y;
                    self.velocity[vi + 2] += force.z;
                }
            }
        }
    }
}

fn advect_scalar(
    old: &[f32],
    velocity: &[f32],
    dims: [usize; 3],
    voxel_size: [f32; 3],
    dt: f32,
    mac_cormack: bool,
) -> Vec<f32> {
    let mut predicted = vec![0.0; old.len()];
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let idx = index(dims, x, y, z);
                let p = [x as f32, y as f32, z as f32];
                let back = backtrace(p, sample_vector(velocity, dims, p), voxel_size, dt);
                predicted[idx] = sample_scalar(old, dims, back).max(0.0);
            }
        }
    }

    if !mac_cormack {
        return predicted;
    }

    let mut corrected = predicted.clone();
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let idx = index(dims, x, y, z);
                let p = [x as f32, y as f32, z as f32];
                let back = backtrace(p, sample_vector(velocity, dims, p), voxel_size, dt);
                let fwd = forwardtrace(back, sample_vector(velocity, dims, back), voxel_size, dt);
                let recovered = sample_scalar(&predicted, dims, fwd);
                let candidate = predicted[idx] + 0.5 * (old[idx] - recovered);
                let (lo, hi) = local_min_max(old, dims, back);
                corrected[idx] = candidate.clamp(lo, hi).max(0.0);
            }
        }
    }
    corrected
}

fn advect_vector(
    old: &[f32],
    velocity: &[f32],
    dims: [usize; 3],
    voxel_size: [f32; 3],
    dt: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; old.len()];
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let idx = index(dims, x, y, z);
                let p = [x as f32, y as f32, z as f32];
                let back = backtrace(p, sample_vector(velocity, dims, p), voxel_size, dt);
                for component in 0..3 {
                    out[idx * 3 + component] = sample_vector_component(old, dims, back, component);
                }
            }
        }
    }
    out
}

fn backtrace(p: [f32; 3], v: [f32; 3], voxel_size: [f32; 3], dt: f32) -> [f32; 3] {
    [
        p[0] - v[0] * dt / voxel_size[0],
        p[1] - v[1] * dt / voxel_size[1],
        p[2] - v[2] * dt / voxel_size[2],
    ]
}

fn forwardtrace(p: [f32; 3], v: [f32; 3], voxel_size: [f32; 3], dt: f32) -> [f32; 3] {
    [
        p[0] + v[0] * dt / voxel_size[0],
        p[1] + v[1] * dt / voxel_size[1],
        p[2] + v[2] * dt / voxel_size[2],
    ]
}

fn local_min_max(field: &[f32], dims: [usize; 3], p: [f32; 3]) -> (f32, f32) {
    let x0 = p[0].floor().clamp(0.0, (dims[0] - 1) as f32) as usize;
    let y0 = p[1].floor().clamp(0.0, (dims[1] - 1) as f32) as usize;
    let z0 = p[2].floor().clamp(0.0, (dims[2] - 1) as f32) as usize;
    let x1 = (x0 + 1).min(dims[0] - 1);
    let y1 = (y0 + 1).min(dims[1] - 1);
    let z1 = (z0 + 1).min(dims[2] - 1);
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for z in z0..=z1 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                let v = field[index(dims, x, y, z)];
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
    }
    (lo, hi)
}

fn scale_to_mass(field: &mut [f32], target_mass: f32) {
    if target_mass <= 0.0 {
        return;
    }
    let mass = field.iter().sum::<f32>();
    if mass > 1.0e-12 {
        let scale = target_mass / mass;
        for value in field {
            *value *= scale;
        }
    }
}

fn diffuse_scalar_in_place(field: &mut Vec<f32>, dims: [usize; 3], rate: f32, dt: f32) {
    if rate <= 0.0 {
        return;
    }
    let old = field.clone();
    let alpha = rate * dt;
    for z in 1..dims[2] - 1 {
        for y in 1..dims[1] - 1 {
            for x in 1..dims[0] - 1 {
                let idx = index(dims, x, y, z);
                let sum = old[index(dims, x - 1, y, z)]
                    + old[index(dims, x + 1, y, z)]
                    + old[index(dims, x, y - 1, z)]
                    + old[index(dims, x, y + 1, z)]
                    + old[index(dims, x, y, z - 1)]
                    + old[index(dims, x, y, z + 1)];
                field[idx] = (old[idx] + alpha * sum) / (1.0 + 6.0 * alpha);
            }
        }
    }
}

fn diffuse_vector(field: &mut Vec<f32>, dims: [usize; 3], rate: f32, dt: f32) {
    if rate <= 0.0 {
        return;
    }
    for component in 0..3 {
        let mut scalar = vec![0.0; dims[0] * dims[1] * dims[2]];
        for idx in 0..scalar.len() {
            scalar[idx] = field[idx * 3 + component];
        }
        diffuse_scalar_in_place(&mut scalar, dims, rate, dt);
        for idx in 0..scalar.len() {
            field[idx * 3 + component] = scalar[idx];
        }
    }
}

fn compute_divergence(velocity: &[f32], dims: [usize; 3], voxel_size: [f32; 3]) -> Vec<f32> {
    let mut divergence = vec![0.0; dims[0] * dims[1] * dims[2]];
    for z in 1..dims[2] - 1 {
        for y in 1..dims[1] - 1 {
            for x in 1..dims[0] - 1 {
                let idx = index(dims, x, y, z);
                let du = (velocity[index(dims, x + 1, y, z) * 3]
                    - velocity[index(dims, x - 1, y, z) * 3])
                    / (2.0 * voxel_size[0]);
                let dv = (velocity[index(dims, x, y + 1, z) * 3 + 1]
                    - velocity[index(dims, x, y - 1, z) * 3 + 1])
                    / (2.0 * voxel_size[1]);
                let dw = (velocity[index(dims, x, y, z + 1) * 3 + 2]
                    - velocity[index(dims, x, y, z - 1) * 3 + 2])
                    / (2.0 * voxel_size[2]);
                divergence[idx] = du + dv + dw;
            }
        }
    }
    divergence
}

fn curl_at(
    velocity: &[f32],
    dims: [usize; 3],
    voxel_size: [f32; 3],
    x: usize,
    y: usize,
    z: usize,
) -> [f32; 3] {
    let read = |x: usize, y: usize, z: usize, component: usize| -> f32 {
        velocity[index(dims, x, y, z) * 3 + component]
    };
    let dw_dy = (read(x, y + 1, z, 2) - read(x, y - 1, z, 2)) / (2.0 * voxel_size[1]);
    let dv_dz = (read(x, y, z + 1, 1) - read(x, y, z - 1, 1)) / (2.0 * voxel_size[2]);
    let du_dz = (read(x, y, z + 1, 0) - read(x, y, z - 1, 0)) / (2.0 * voxel_size[2]);
    let dw_dx = (read(x + 1, y, z, 2) - read(x - 1, y, z, 2)) / (2.0 * voxel_size[0]);
    let dv_dx = (read(x + 1, y, z, 1) - read(x - 1, y, z, 1)) / (2.0 * voxel_size[0]);
    let du_dy = (read(x, y + 1, z, 0) - read(x, y - 1, z, 0)) / (2.0 * voxel_size[1]);
    [dw_dy - dv_dz, du_dz - dw_dx, dv_dx - du_dy]
}

#[cfg(test)]
mod tests {
    use crate::smoke::{SmokeDomainConfig, SmokeEmitter, SmokeStepSettings, SmokeVolume};

    fn small_volume() -> SmokeVolume {
        SmokeVolume::new(SmokeDomainConfig::new([16, 16, 16], [1.0; 3], [0.0; 3]).unwrap()).unwrap()
    }

    #[test]
    fn emitter_adds_required_fields() {
        let mut volume = small_volume();
        let emitter = SmokeEmitter {
            center: [8.0, 8.0, 8.0],
            radius: 3.0,
            density_rate: 2.0,
            temperature_rate: 4.0,
            fuel_rate: 1.0,
            ..SmokeEmitter::default()
        };
        volume.add_emitter(&emitter, 0.5).unwrap();
        assert!(volume.mass() > 0.0);
        assert!(volume.temperature.iter().copied().fold(0.0, f32::max) > 0.0);
        assert!(volume.fuel.iter().copied().fold(0.0, f32::max) > 0.0);
        assert!(volume.emission_rate.iter().copied().fold(0.0, f32::max) > 0.0);
    }

    #[test]
    fn smoke_advects_with_wind_and_preserves_mass() {
        let mut volume = small_volume();
        let emitter = SmokeEmitter {
            center: [5.0, 8.0, 8.0],
            radius: 2.0,
            density_rate: 5.0,
            ..SmokeEmitter::default()
        };
        volume.add_emitter(&emitter, 1.0).unwrap();
        for idx in 0..volume.voxel_count() {
            volume.velocity[idx * 3] = 1.0;
        }
        let before_mass = volume.mass();
        let before = center_of_mass_x(&volume);
        let settings = SmokeStepSettings {
            dt: 1.0,
            density_decay: 0.0,
            temperature_decay: 0.0,
            buoyancy: 0.0,
            vorticity: 0.0,
            diffusion: 0.0,
            pressure_iterations: 1,
            ..SmokeStepSettings::default()
        };
        volume.step(&settings, &[]).unwrap();
        assert!(center_of_mass_x(&volume) > before);
        assert!((volume.mass() - before_mass).abs() / before_mass < 0.02);
    }

    #[test]
    fn buoyant_plume_rises() {
        let mut volume = small_volume();
        let emitter = SmokeEmitter {
            center: [8.0, 4.0, 8.0],
            radius: 2.0,
            density_rate: 4.0,
            temperature_rate: 5.0,
            ..SmokeEmitter::default()
        };
        volume.add_emitter(&emitter, 1.0).unwrap();
        let before = center_of_mass_y(&volume);
        let settings = SmokeStepSettings {
            dt: 0.5,
            density_decay: 0.0,
            temperature_decay: 0.0,
            buoyancy: 1.5,
            vorticity: 0.0,
            diffusion: 0.0,
            pressure_iterations: 8,
            ..SmokeStepSettings::default()
        };
        for _ in 0..4 {
            volume.step(&settings, &[]).unwrap();
        }
        assert!(center_of_mass_y(&volume) > before);
    }

    #[test]
    fn pressure_projection_reduces_divergence() {
        let mut volume = small_volume();
        for z in 1..15 {
            for y in 1..15 {
                for x in 1..15 {
                    let idx = volume.index(x, y, z);
                    volume.velocity[idx * 3] = x as f32 * 0.03;
                    volume.velocity[idx * 3 + 1] = y as f32 * -0.02;
                }
            }
        }
        let before = volume.divergence_l2();
        let settings = SmokeStepSettings {
            dt: 0.1,
            density_decay: 0.0,
            temperature_decay: 0.0,
            buoyancy: 0.0,
            vorticity: 0.0,
            diffusion: 0.0,
            pressure_iterations: 30,
            ..SmokeStepSettings::default()
        };
        volume.step(&settings, &[]).unwrap();
        assert!(volume.divergence_l2() < before);
    }

    fn center_of_mass_x(volume: &SmokeVolume) -> f32 {
        center_of_mass_axis(volume, 0)
    }

    fn center_of_mass_y(volume: &SmokeVolume) -> f32 {
        center_of_mass_axis(volume, 1)
    }

    fn center_of_mass_axis(volume: &SmokeVolume, axis: usize) -> f32 {
        let mut weighted = 0.0;
        let mut mass = 0.0;
        let [nx, ny, nz] = volume.config.dims;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = volume.index(x, y, z);
                    let p = [x as f32, y as f32, z as f32];
                    weighted += p[axis] * volume.density[idx];
                    mass += volume.density[idx];
                }
            }
        }
        weighted / mass.max(1.0e-9)
    }
}
