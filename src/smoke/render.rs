use glam::Vec3;

use super::sampling::{hash01, lerp, sample_scalar};
use super::types::{SmokeRenderSettings, SmokeVolume};

impl SmokeVolume {
    pub fn raymarch_rgba(
        &self,
        width: u32,
        height: u32,
        camera_pos: [f32; 3],
        target: [f32; 3],
        up: [f32; 3],
        fovy_deg: f32,
        sun_direction: [f32; 3],
        settings: &SmokeRenderSettings,
    ) -> Result<Vec<u8>, String> {
        settings.validate()?;
        if width == 0 || height == 0 {
            return Err("width and height must be >= 1".to_string());
        }
        if !fovy_deg.is_finite() || fovy_deg <= 0.0 || fovy_deg >= 179.0 {
            return Err("fovy_deg must be finite and in (0, 179)".to_string());
        }

        let eye = Vec3::from_array(camera_pos);
        let target = Vec3::from_array(target);
        let forward = (target - eye).normalize_or_zero();
        if forward.length_squared() < 1.0e-12 {
            return Err("camera_pos and target must not be equal".to_string());
        }
        let up = Vec3::from_array(up).normalize_or_zero();
        if up.length_squared() < 1.0e-12 {
            return Err("up vector must not be zero".to_string());
        }
        let right = forward.cross(up).normalize_or_zero();
        let camera_up = right.cross(forward).normalize_or_zero();
        let sun_dir = Vec3::from_array(sun_direction).normalize_or_zero();
        if sun_dir.length_squared() < 1.0e-12 {
            return Err("sun_direction must not be zero".to_string());
        }

        let min_step = self
            .config
            .voxel_size
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min)
            .max(1.0e-4);
        let step_size = if settings.step_size > 0.0 {
            settings.step_size
        } else {
            min_step * 0.75
        };
        let shadow_step = if settings.shadow_step_size > 0.0 {
            settings.shadow_step_size
        } else {
            step_size * 2.0
        };
        let tan_half_fov = (fovy_deg.to_radians() * 0.5).tan();
        let aspect = width as f32 / height as f32;
        let bounds_min = Vec3::from_array(self.bounds_min());
        let bounds_max = Vec3::from_array(self.bounds_max());
        let mut out = vec![0u8; width as usize * height as usize * 4];

        for y in 0..height {
            for x in 0..width {
                let px = ((x as f32 + 0.5) / width as f32 * 2.0 - 1.0) * aspect * tan_half_fov;
                let py = (1.0 - (y as f32 + 0.5) / height as f32 * 2.0) * tan_half_fov;
                let ray_dir = (forward + right * px + camera_up * py).normalize();
                let Some((mut t0, t1)) = ray_box_intersection(eye, ray_dir, bounds_min, bounds_max)
                else {
                    continue;
                };
                t0 = t0.max(0.0);
                let jitter_seed = x
                    .wrapping_mul(73856093)
                    .wrapping_add(y.wrapping_mul(19349663))
                    .wrapping_add(self.frame_index as u32);
                let dst = ((y * width + x) * 4) as usize;
                let rgba = self.march_ray_rgba(
                    eye,
                    ray_dir,
                    t0,
                    t1,
                    jitter_seed,
                    step_size,
                    shadow_step,
                    sun_dir,
                    settings,
                );
                out[dst..dst + 4].copy_from_slice(&rgba);
            }
        }
        Ok(out)
    }

    pub fn raymarch_projection_rgba(
        &self,
        width: u32,
        height: u32,
        view_direction: [f32; 3],
        sun_direction: [f32; 3],
        settings: &SmokeRenderSettings,
    ) -> Result<Vec<u8>, String> {
        settings.validate()?;
        if width == 0 || height == 0 {
            return Err("width and height must be >= 1".to_string());
        }
        let ray_dir = Vec3::from_array(view_direction).normalize_or_zero();
        if ray_dir.length_squared() < 1.0e-12 {
            return Err("view_direction must not be zero".to_string());
        }
        let sun_dir = Vec3::from_array(sun_direction).normalize_or_zero();
        if sun_dir.length_squared() < 1.0e-12 {
            return Err("sun_direction must not be zero".to_string());
        }

        let min_step = self
            .config
            .voxel_size
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min)
            .max(1.0e-4);
        let step_size = if settings.step_size > 0.0 {
            settings.step_size
        } else {
            min_step * 0.75
        };
        let shadow_step = if settings.shadow_step_size > 0.0 {
            settings.shadow_step_size
        } else {
            step_size * 2.0
        };
        let bounds_min = Vec3::from_array(self.bounds_min());
        let bounds_max = Vec3::from_array(self.bounds_max());
        let diagonal = (bounds_max - bounds_min).length().max(step_size * 2.0);
        let mut out = vec![0u8; width as usize * height as usize * 4];

        for py in 0..height {
            let fz = (py as f32 + 0.5) / height as f32;
            let z = lerp(bounds_min.z, bounds_max.z, fz);
            for px in 0..width {
                let fx = (px as f32 + 0.5) / width as f32;
                let x = lerp(bounds_min.x, bounds_max.x, fx);
                let plane_point = Vec3::new(x, (bounds_min.y + bounds_max.y) * 0.5, z);
                let origin = plane_point - ray_dir * diagonal;
                let Some((mut t0, t1)) =
                    ray_box_intersection(origin, ray_dir, bounds_min, bounds_max)
                else {
                    continue;
                };
                t0 = t0.max(0.0);
                let jitter_seed = px
                    .wrapping_mul(73856093)
                    .wrapping_add(py.wrapping_mul(19349663))
                    .wrapping_add(self.frame_index as u32)
                    .wrapping_add(0x9e3779b9);
                let rgba = self.march_ray_rgba(
                    origin,
                    ray_dir,
                    t0,
                    t1,
                    jitter_seed,
                    step_size,
                    shadow_step,
                    sun_dir,
                    settings,
                );
                let dst = ((py * width + px) * 4) as usize;
                out[dst..dst + 4].copy_from_slice(&rgba);
            }
        }
        Ok(out)
    }

    fn sample_render_fields(&self, position: [f32; 3]) -> RenderSample {
        let p = self.grid_coord_from_world(position);
        RenderSample {
            density: sample_scalar(&self.density, self.config.dims, p),
            temperature: sample_scalar(&self.temperature, self.config.dims, p),
            soot: sample_scalar(&self.soot, self.config.dims, p),
            humidity: sample_scalar(&self.humidity, self.config.dims, p),
            emission: sample_scalar(&self.emission_rate, self.config.dims, p),
            age: sample_scalar(&self.particle_age, self.config.dims, p).max(0.0),
        }
    }

    fn march_ray_rgba(
        &self,
        origin: Vec3,
        ray_dir: Vec3,
        t0: f32,
        t1: f32,
        jitter_seed: u32,
        step_size: f32,
        shadow_step: f32,
        sun_dir: Vec3,
        settings: &SmokeRenderSettings,
    ) -> [u8; 4] {
        let jitter = (hash01(jitter_seed) - 0.5) * settings.jitter_strength * step_size;
        let mut t = (t0 + jitter).max(0.0);
        let mut transmittance = 1.0f32;
        let mut rgb = Vec3::ZERO;
        let mut steps = 0u32;

        while t < t1 && steps < settings.max_steps && transmittance > 0.01 {
            let p = origin + ray_dir * t;
            let sample = self.sample_render_fields(p.to_array());
            let age_t = render_smoothstep(1.6, 17.0, sample.age);
            let concentration_gate = 0.50 + 0.50 * render_smoothstep(0.045, 0.34, sample.density);
            let density = (sample.density
                * settings.density_scale
                * (1.0 - 0.58 * age_t)
                * concentration_gate)
                .max(0.0);
            if density > 1.0e-5 {
                let sigma_t = density
                    * settings.extinction
                    * (1.0 + sample.soot * settings.soot_absorption * 0.85);
                let segment_transmittance = (-sigma_t * step_size).exp().clamp(0.0, 1.0);
                let segment_weight = if sigma_t > 1.0e-6 {
                    (1.0 - segment_transmittance) / sigma_t
                } else {
                    step_size
                };
                let light_trans = if settings.self_shadow {
                    self.sun_transmittance(p, sun_dir, shadow_step, settings.shadow_steps, settings)
                } else {
                    1.0
                };
                let cos_theta = ray_dir.dot(sun_dir).clamp(-1.0, 1.0);
                let phase = henyey_greenstein(cos_theta, settings.phase_g);
                let smoke_color = smoke_color(sample, settings);
                let scatter_albedo = (settings.scattering
                    / (settings.scattering + settings.absorption + sample.soot * 0.55 + 1.0e-5))
                    .clamp(0.02, 0.98);
                let sigma_s = sigma_t * scatter_albedo;
                let sun_radiance = Vec3::new(1.0, 0.96, 0.84) * 11.5;
                let sky_radiance = Vec3::new(0.52, 0.60, 0.72)
                    * (0.36 + 0.26 * (1.0 - light_trans).clamp(0.0, 1.0))
                    * (1.0 - sample.soot * 0.32).clamp(0.50, 1.0);
                let ground_bounce = Vec3::new(0.58, 0.54, 0.48)
                    * 0.070
                    * (1.0 - p.y / self.bounds_max()[1].max(1.0)).clamp(0.0, 1.0);
                let powder = (1.0 - (-sigma_t * step_size * 2.2).exp()).clamp(0.0, 1.0);
                let multiple_scatter = smoke_color
                    * sigma_s
                    * (sky_radiance
                        + ground_bounce
                        + Vec3::splat(powder * 0.055 * light_trans.sqrt()));
                let direct_scatter = smoke_color * sigma_s * sun_radiance * phase * light_trans;
                let freshness = (1.0 - sample.age / 17.0).clamp(0.0, 1.0);
                let fresh_heat = sample.temperature * freshness * freshness;
                let source_emission = Vec3::new(1.0, 0.30, 0.055)
                    * ((fresh_heat * 0.10 + sample.emission * 1.18) * settings.fire_glow)
                        .clamp(0.0, 5.0);
                let source_radiance = direct_scatter + multiple_scatter + source_emission;
                rgb += source_radiance * segment_weight * transmittance;
                transmittance *= segment_transmittance;
            }
            t += step_size;
            steps += 1;
        }

        let alpha = (1.0 - transmittance).clamp(0.0, 1.0);
        let straight = if alpha > 1.0e-5 { rgb / alpha } else { rgb };
        let mapped = tone_map(straight * settings.exposure);
        [
            to_u8(mapped.x),
            to_u8(mapped.y),
            to_u8(mapped.z),
            to_u8(alpha),
        ]
    }

    fn sun_transmittance(
        &self,
        start: Vec3,
        sun_dir: Vec3,
        step_size: f32,
        steps: u32,
        settings: &SmokeRenderSettings,
    ) -> f32 {
        let bounds_min = Vec3::from_array(self.bounds_min());
        let bounds_max = Vec3::from_array(self.bounds_max());
        let Some((mut t0, t1)) =
            ray_box_intersection(start + sun_dir * step_size, sun_dir, bounds_min, bounds_max)
        else {
            return 1.0;
        };
        t0 = t0.max(0.0);
        let mut optical_depth = 0.0;
        for i in 0..steps {
            let t = t0 + (i as f32 + 0.5) * step_size;
            if t > t1 {
                break;
            }
            let p = start + sun_dir * (step_size + t);
            let sample = self.sample_render_fields(p.to_array());
            let age_t = render_smoothstep(1.6, 17.0, sample.age);
            let concentration_gate = 0.50 + 0.50 * render_smoothstep(0.045, 0.34, sample.density);
            optical_depth += sample.density
                * settings.density_scale
                * (1.0 - 0.58 * age_t)
                * concentration_gate
                * settings.extinction
                * (1.0 + sample.soot * settings.soot_absorption)
                * step_size;
            if optical_depth > 8.0 {
                break;
            }
        }
        (-optical_depth).exp().clamp(0.0, 1.0)
    }
}

#[derive(Clone, Copy)]
struct RenderSample {
    density: f32,
    temperature: f32,
    soot: f32,
    humidity: f32,
    emission: f32,
    age: f32,
}

fn smoke_color(sample: RenderSample, settings: &SmokeRenderSettings) -> Vec3 {
    let body = (sample.density * 1.45 + sample.soot * 1.35).clamp(0.0, 1.0);
    let thin = Vec3::from_array(settings.thin_color);
    let dense = Vec3::from_array(settings.dense_color);
    let mut color = mix_vec3(thin, dense, body);
    let aged = (sample.age / 9.0).clamp(0.0, 1.0);
    color = mix_vec3(color, Vec3::new(0.36, 0.39, 0.43), aged * 0.42);
    let humidity_milk = sample.humidity.clamp(0.0, 1.0) * (0.18 + 0.42 * body);
    color = mix_vec3(
        color,
        Vec3::new(0.93, 0.92, 0.84),
        humidity_milk.clamp(0.0, 0.38),
    );
    let freshness = (1.0 - sample.age / 17.0).clamp(0.0, 1.0);
    let heat = (sample.temperature * 0.12 * freshness).clamp(0.0, 1.0);
    color = mix_vec3(color, Vec3::new(0.95, 0.62, 0.28), heat * 0.07);
    color
}

fn ray_box_intersection(origin: Vec3, dir: Vec3, min: Vec3, max: Vec3) -> Option<(f32, f32)> {
    let inv = Vec3::new(
        if dir.x.abs() > 1.0e-12 {
            1.0 / dir.x
        } else {
            f32::INFINITY
        },
        if dir.y.abs() > 1.0e-12 {
            1.0 / dir.y
        } else {
            f32::INFINITY
        },
        if dir.z.abs() > 1.0e-12 {
            1.0 / dir.z
        } else {
            f32::INFINITY
        },
    );
    let t0 = (min - origin) * inv;
    let t1 = (max - origin) * inv;
    let tmin = t0.min(t1);
    let tmax = t0.max(t1);
    let near = tmin.x.max(tmin.y).max(tmin.z);
    let far = tmax.x.min(tmax.y).min(tmax.z);
    if far >= near.max(0.0) {
        Some((near, far))
    } else {
        None
    }
}

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = (1.0 + g2 - 2.0 * g * cos_theta).max(1.0e-4);
    (1.0 - g2) / (4.0 * std::f32::consts::PI * denom.powf(1.5))
}

fn tone_map(color: Vec3) -> Vec3 {
    color / (Vec3::ONE + color)
}

fn render_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1.0e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mix_vec3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    Vec3::new(lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t))
}

fn to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smoke::sampling::index;
    use crate::smoke::{SmokeDomainConfig, SmokeEmitter, SmokeStepSettings};

    #[test]
    fn raymarch_returns_nonblank_smoke_layer() {
        let mut volume =
            SmokeVolume::new(SmokeDomainConfig::new([16, 16, 16], [1.0; 3], [0.0; 3]).unwrap())
                .unwrap();
        let emitter = SmokeEmitter {
            center: [8.0, 8.0, 8.0],
            radius: 4.0,
            density_rate: 4.0,
            temperature_rate: 1.0,
            ..SmokeEmitter::default()
        };
        volume.add_emitter(&emitter, 1.0).unwrap();
        volume
            .step(
                &SmokeStepSettings {
                    dt: 0.1,
                    density_decay: 0.0,
                    temperature_decay: 0.0,
                    ..SmokeStepSettings::default()
                },
                &[],
            )
            .unwrap();
        let rgba = volume
            .raymarch_rgba(
                32,
                32,
                [8.0, 8.0, -18.0],
                [8.0, 8.0, 8.0],
                [0.0, 1.0, 0.0],
                45.0,
                [0.4, 0.8, -0.2],
                &SmokeRenderSettings::default(),
            )
            .unwrap();
        assert_eq!(rgba.len(), 32 * 32 * 4);
        assert!(rgba.iter().skip(3).step_by(4).any(|alpha| *alpha > 0));
    }

    #[test]
    fn projected_raymarch_returns_map_aligned_smoke_layer() {
        let mut volume =
            SmokeVolume::new(SmokeDomainConfig::new([18, 12, 14], [1.0; 3], [0.0; 3]).unwrap())
                .unwrap();
        let emitter = SmokeEmitter {
            center: [8.0, 4.0, 7.0],
            radius: 3.5,
            density_rate: 5.0,
            temperature_rate: 0.8,
            ..SmokeEmitter::default()
        };
        volume.add_emitter(&emitter, 1.0).unwrap();

        let rgba = volume
            .raymarch_projection_rgba(
                36,
                28,
                [0.0, -1.0, 0.0],
                [0.4, 0.8, -0.2],
                &SmokeRenderSettings::default(),
            )
            .unwrap();

        assert_eq!(rgba.len(), 36 * 28 * 4);
        assert!(rgba.iter().skip(3).step_by(4).any(|alpha| *alpha > 0));
    }

    #[test]
    fn sun_transmittance_tracks_volume_self_shadowing() {
        let mut volume =
            SmokeVolume::new(SmokeDomainConfig::new([24, 12, 12], [1.0; 3], [0.0; 3]).unwrap())
                .unwrap();
        let dims = volume.config.dims;
        for z in 3..9 {
            for y in 3..9 {
                for x in 8..14 {
                    let idx = index(dims, x, y, z);
                    volume.density[idx] = 1.2;
                    volume.soot[idx] = 0.18;
                }
            }
        }
        let settings = SmokeRenderSettings {
            density_scale: 1.4,
            extinction: 1.8,
            shadow_steps: 48,
            shadow_step_size: 0.5,
            ..SmokeRenderSettings::default()
        };

        let lit = volume.sun_transmittance(
            Vec3::new(15.0, 6.0, 6.0),
            Vec3::new(1.0, 0.0, 0.0),
            0.5,
            settings.shadow_steps,
            &settings,
        );
        let occluded = volume.sun_transmittance(
            Vec3::new(15.0, 6.0, 6.0),
            Vec3::new(-1.0, 0.0, 0.0),
            0.5,
            settings.shadow_steps,
            &settings,
        );

        assert!(lit > 0.95);
        assert!(occluded < 0.35);
    }

    #[test]
    fn raymarch_emission_adds_warm_source_radiance() {
        let mut volume =
            SmokeVolume::new(SmokeDomainConfig::new([16, 16, 16], [1.0; 3], [0.0; 3]).unwrap())
                .unwrap();
        let dims = volume.config.dims;
        for z in 5..11 {
            for y in 5..11 {
                for x in 5..11 {
                    let idx = index(dims, x, y, z);
                    volume.density[idx] = 0.55;
                    volume.temperature[idx] = 0.85;
                    volume.emission_rate[idx] = 1.4;
                }
            }
        }
        let settings = SmokeRenderSettings {
            density_scale: 1.2,
            extinction: 1.25,
            fire_glow: 1.25,
            exposure: 1.15,
            ..SmokeRenderSettings::default()
        };
        let with_emission = volume
            .raymarch_rgba(
                32,
                32,
                [8.0, 8.0, -18.0],
                [8.0, 8.0, 8.0],
                [0.0, 1.0, 0.0],
                45.0,
                [0.3, 0.8, -0.2],
                &settings,
            )
            .unwrap();
        for value in &mut volume.temperature {
            *value = 0.0;
        }
        for value in &mut volume.emission_rate {
            *value = 0.0;
        }
        let without_emission = volume
            .raymarch_rgba(
                32,
                32,
                [8.0, 8.0, -18.0],
                [8.0, 8.0, 8.0],
                [0.0, 1.0, 0.0],
                45.0,
                [0.3, 0.8, -0.2],
                &settings,
            )
            .unwrap();

        let warm_peak = with_emission
            .chunks_exact(4)
            .map(|px| px[0] as i16 - px[2] as i16)
            .max()
            .unwrap();
        let no_emission_warm_peak = without_emission
            .chunks_exact(4)
            .map(|px| px[0] as i16 - px[2] as i16)
            .max()
            .unwrap();
        let red_peak = with_emission.chunks_exact(4).map(|px| px[0]).max().unwrap();
        let no_emission_red_peak = without_emission
            .chunks_exact(4)
            .map(|px| px[0])
            .max()
            .unwrap();

        assert!(warm_peak > no_emission_warm_peak + 12);
        assert!(red_peak > no_emission_red_peak);
    }
}
