use std::f32::consts::PI;

use crate::formats::hdr::HdrImage;
use crate::terrain::probes::types::{
    ProbeError, ProbePlacement, ReflectionProbe, ReflectionProbeMip, ReflectionProbeSet,
    REFLECTION_PROBE_FACE_COUNT,
};

const TWO_PI: f32 = PI * 2.0;

#[derive(Clone, Copy, Debug)]
pub enum ReflectionAlbedoMode {
    Material,
    Colormap,
    Mix,
}

#[derive(Clone, Copy, Debug)]
pub enum ReflectionOverlayBlend {
    Alpha,
    Add,
    Multiply,
    Replace,
}

#[derive(Clone, Debug)]
pub struct ReflectionOverlay {
    pub domain: (f32, f32),
    pub strength: f32,
    pub offset: f32,
    pub blend_mode: ReflectionOverlayBlend,
    pub stops: Vec<(f32, [f32; 3])>,
}

#[derive(Clone, Debug)]
pub struct ReflectionTerrainMaterial {
    pub albedo_mode: ReflectionAlbedoMode,
    pub colormap_strength: f32,
    pub raw_height_range: (f32, f32),
    pub overlay: Option<ReflectionOverlay>,
    pub grass_color: [f32; 3],
    pub dirt_color: [f32; 3],
    pub rock_color: [f32; 3],
    pub snow_color: [f32; 3],
    pub snow_enabled: bool,
    pub snow_altitude_min: f32,
    pub snow_altitude_blend: f32,
    pub snow_slope_max_deg: f32,
    pub snow_slope_blend_deg: f32,
    pub snow_aspect_influence: f32,
    pub rock_enabled: bool,
    pub rock_slope_min_deg: f32,
    pub rock_slope_blend_deg: f32,
    pub wetness_enabled: bool,
    pub wetness_strength: f32,
    pub wetness_slope_influence: f32,
}

#[derive(Clone, Copy)]
pub struct ReflectionCaptureLighting<'a> {
    pub env_image: Option<&'a HdrImage>,
    pub env_intensity: f32,
    pub env_rotation_rad: f32,
    pub light_dir: [f32; 3],
    pub light_color: [f32; 3],
    pub light_intensity: f32,
}

pub struct HeightfieldReflectionBaker<'a> {
    pub heightfield: &'a [f32],
    pub height_dims: (u32, u32),
    pub terrain_span: [f32; 2],
    pub z_scale: f32,
    pub resolution: u32,
    pub prefilter_sample_count: u32,
    pub trace_steps: u32,
    pub trace_refine_steps: u32,
    pub max_trace_distance: f32,
    pub material: ReflectionTerrainMaterial,
    pub lighting: ReflectionCaptureLighting<'a>,
}

#[derive(Clone, Copy)]
struct SurfaceHit {
    position_ws: [f32; 3],
    normal_ws: [f32; 3],
    raw_height: f32,
}

impl<'a> HeightfieldReflectionBaker<'a> {
    fn sample_texel_raw(&self, x: u32, y: u32) -> Option<f32> {
        let value = self.heightfield[(y * self.height_dims.0 + x) as usize];
        value.is_finite().then_some(value)
    }

    fn sample_height_raw(&self, world_x: f32, world_y: f32) -> Option<f32> {
        let (w, h) = self.height_dims;
        if w == 0 || h == 0 || self.heightfield.is_empty() {
            return None;
        }
        if w == 1 || h == 1 {
            return self.heightfield[0]
                .is_finite()
                .then_some(self.heightfield[0]);
        }

        let u = (world_x / self.terrain_span[0]) + 0.5;
        let v = (world_y / self.terrain_span[1]) + 0.5;
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return None;
        }

        let fx = u * (w - 1) as f32;
        let fy = v * (h - 1) as f32;
        let x0 = fx.floor().clamp(0.0, (w - 1) as f32) as u32;
        let y0 = fy.floor().clamp(0.0, (h - 1) as f32) as u32;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let samples = [
            ((1.0 - tx) * (1.0 - ty), self.sample_texel_raw(x0, y0)),
            (tx * (1.0 - ty), self.sample_texel_raw(x1, y0)),
            ((1.0 - tx) * ty, self.sample_texel_raw(x0, y1)),
            (tx * ty, self.sample_texel_raw(x1, y1)),
        ];

        let mut sum = 0.0;
        let mut weight = 0.0;
        for (wgt, sample) in samples {
            if let Some(value) = sample {
                sum += value * wgt;
                weight += wgt;
            }
        }
        (weight > 0.0).then_some(sum / weight)
    }

    fn sample_height_scaled(&self, world_x: f32, world_y: f32) -> Option<f32> {
        self.sample_height_raw(world_x, world_y)
            .map(|height| height * self.z_scale)
    }

    fn sample_normal_ws(&self, world_x: f32, world_y: f32) -> [f32; 3] {
        let step_x = if self.height_dims.0 > 1 {
            self.terrain_span[0] / (self.height_dims.0 - 1) as f32
        } else {
            1.0
        };
        let step_y = if self.height_dims.1 > 1 {
            self.terrain_span[1] / (self.height_dims.1 - 1) as f32
        } else {
            1.0
        };

        let left = self
            .sample_height_scaled(world_x - step_x, world_y)
            .or_else(|| self.sample_height_scaled(world_x, world_y))
            .unwrap_or(0.0);
        let right = self
            .sample_height_scaled(world_x + step_x, world_y)
            .or_else(|| self.sample_height_scaled(world_x, world_y))
            .unwrap_or(0.0);
        let down = self
            .sample_height_scaled(world_x, world_y - step_y)
            .or_else(|| self.sample_height_scaled(world_x, world_y))
            .unwrap_or(0.0);
        let up = self
            .sample_height_scaled(world_x, world_y + step_y)
            .or_else(|| self.sample_height_scaled(world_x, world_y))
            .unwrap_or(0.0);

        normalize3([
            -(right - left) / (2.0 * step_x.max(1e-4)),
            -(up - down) / (2.0 * step_y.max(1e-4)),
            1.0,
        ])
    }

    fn trace_surface(&self, origin: [f32; 3], direction: [f32; 3]) -> Option<SurfaceHit> {
        let dir = normalize3(direction);
        let step_count = self.trace_steps.max(8);
        let step_size = self.max_trace_distance.max(1e-3) / step_count as f32;
        let mut prev_t = 0.0f32;

        for step in 1..=step_count {
            let t = step as f32 * step_size;
            let sample_pos = [
                origin[0] + dir[0] * t,
                origin[1] + dir[1] * t,
                origin[2] + dir[2] * t,
            ];
            let raw_height = match self.sample_height_raw(sample_pos[0], sample_pos[1]) {
                Some(value) => value,
                None => {
                    prev_t = t;
                    continue;
                }
            };
            let terrain_z = raw_height * self.z_scale;
            if terrain_z >= sample_pos[2] {
                let mut low = prev_t;
                let mut high = t;
                for _ in 0..self.trace_refine_steps {
                    let mid = 0.5 * (low + high);
                    let mid_pos = [
                        origin[0] + dir[0] * mid,
                        origin[1] + dir[1] * mid,
                        origin[2] + dir[2] * mid,
                    ];
                    if let Some(mid_height) = self.sample_height_scaled(mid_pos[0], mid_pos[1]) {
                        if mid_height >= mid_pos[2] {
                            high = mid;
                        } else {
                            low = mid;
                        }
                    }
                }
                let hit_pos = [origin[0] + dir[0] * high, origin[1] + dir[1] * high, 0.0];
                let raw_height = self
                    .sample_height_raw(hit_pos[0], hit_pos[1])
                    .unwrap_or(raw_height);
                let scaled_height = raw_height * self.z_scale;
                return Some(SurfaceHit {
                    position_ws: [hit_pos[0], hit_pos[1], scaled_height],
                    normal_ws: self.sample_normal_ws(hit_pos[0], hit_pos[1]),
                    raw_height,
                });
            }
            prev_t = t;
        }
        None
    }

    fn shadow_visibility(&self, hit: &SurfaceHit) -> f32 {
        let shadow_origin = add3(hit.position_ws, mul3(hit.normal_ws, 0.05));
        if self
            .trace_surface(shadow_origin, self.lighting.light_dir)
            .is_some()
        {
            0.25
        } else {
            1.0
        }
    }

    fn sample_overlay_color(&self, raw_height: f32) -> Option<[f32; 3]> {
        let overlay = self.material.overlay.as_ref()?;
        if overlay.stops.is_empty() {
            return None;
        }
        let domain_min = overlay.domain.0;
        let domain_max = overlay.domain.1.max(domain_min + 1e-6);
        let value = (raw_height + overlay.offset).clamp(domain_min, domain_max);
        if value <= overlay.stops[0].0 {
            return Some(overlay.stops[0].1);
        }
        if let Some((_, color)) = overlay.stops.last() {
            if value >= overlay.stops.last().map(|(x, _)| *x).unwrap_or(value) {
                return Some(*color);
            }
        }
        for window in overlay.stops.windows(2) {
            let (v0, c0) = window[0];
            let (v1, c1) = window[1];
            if value >= v0 && value <= v1 {
                let t = if v1 > v0 {
                    (value - v0) / (v1 - v0)
                } else {
                    0.0
                };
                return Some(lerp3(c0, c1, t));
            }
        }
        overlay.stops.last().map(|(_, color)| *color)
    }

    fn resolve_material_albedo(&self, hit: &SurfaceHit) -> [f32; 3] {
        let mut material_albedo = self.material.grass_color;
        let slope = hit.normal_ws[2].clamp(-1.0, 1.0).acos();
        let aspect = hit.normal_ws[1].atan2(hit.normal_ws[0]);

        if self.material.rock_enabled {
            let slope_min = self.material.rock_slope_min_deg.to_radians();
            let slope_blend = self.material.rock_slope_blend_deg.to_radians().max(1e-3);
            let rock_weight = ((slope - slope_min) / slope_blend).clamp(0.0, 1.0);
            material_albedo = lerp3(material_albedo, self.material.rock_color, rock_weight);
        }

        if self.material.snow_enabled {
            let alt_factor = ((hit.position_ws[2] - self.material.snow_altitude_min)
                / self.material.snow_altitude_blend.max(1e-3))
            .clamp(0.0, 1.0);
            let slope_max = self.material.snow_slope_max_deg.to_radians();
            let slope_blend = self.material.snow_slope_blend_deg.to_radians().max(1e-3);
            let slope_factor =
                1.0 - ((slope - slope_max + slope_blend) / slope_blend).clamp(0.0, 1.0);
            let aspect_factor = lerp(
                1.0,
                0.5 + 0.5 * aspect.cos(),
                self.material.snow_aspect_influence.clamp(0.0, 1.0),
            );
            let snow_weight = (alt_factor * slope_factor * aspect_factor).clamp(0.0, 1.0);
            material_albedo = lerp3(material_albedo, self.material.snow_color, snow_weight);
        }

        if self.material.wetness_enabled {
            let flat_factor = 1.0 - (slope / (PI * 0.25)).clamp(0.0, 1.0);
            let wetness = (flat_factor * self.material.wetness_slope_influence).clamp(0.0, 1.0);
            let wet_darkening = 1.0 - wetness * self.material.wetness_strength.clamp(0.0, 1.0);
            let dirt_mix = lerp3(self.material.dirt_color, material_albedo, wet_darkening);
            material_albedo = lerp3(material_albedo, dirt_mix, wetness);
        }

        match self.material.albedo_mode {
            ReflectionAlbedoMode::Material => material_albedo,
            ReflectionAlbedoMode::Colormap => self
                .sample_overlay_color(hit.raw_height)
                .unwrap_or(material_albedo),
            ReflectionAlbedoMode::Mix => {
                if let Some(overlay_color) = self.sample_overlay_color(hit.raw_height) {
                    apply_overlay_blend(
                        material_albedo,
                        overlay_color,
                        self.material
                            .overlay
                            .as_ref()
                            .map(|overlay| overlay.blend_mode),
                        self.material.colormap_strength,
                    )
                } else {
                    material_albedo
                }
            }
        }
    }

    fn sample_environment(&self, direction: [f32; 3]) -> [f32; 3] {
        let dir = normalize3(direction);
        let (sin_theta, cos_theta) = self.lighting.env_rotation_rad.sin_cos();
        let rotated = [
            dir[0] * cos_theta + dir[2] * sin_theta,
            dir[1],
            -dir[0] * sin_theta + dir[2] * cos_theta,
        ];

        if let Some(hdr) = self.lighting.env_image {
            let u = (rotated[2].atan2(rotated[0]) / TWO_PI + 0.5).rem_euclid(1.0);
            let v = (rotated[1].clamp(-1.0, 1.0).acos() / PI).clamp(0.0, 1.0);
            return mul3(
                sample_hdr_equirect(hdr, u, v),
                self.lighting.env_intensity.max(0.0),
            );
        }

        let sky_mix = rotated[1].mul_add(0.5, 0.5).clamp(0.0, 1.0);
        lerp3(
            [0.12, 0.14, 0.18],
            [0.58, 0.74, 1.0],
            sky_mix * self.lighting.env_intensity.max(0.0),
        )
    }

    fn shade_hit(&self, hit: &SurfaceHit) -> [f32; 3] {
        let albedo = self.resolve_material_albedo(hit);
        let ambient = mul3(self.sample_environment(hit.normal_ws), 0.45);
        let shadow = self.shadow_visibility(hit);
        let n_dot_l = dot3(hit.normal_ws, normalize3(self.lighting.light_dir)).max(0.0);
        let direct_strength = self.lighting.light_intensity.max(0.0) * n_dot_l * shadow;
        let direct = mul3_components(mul3(albedo, direct_strength), self.lighting.light_color);
        add3(mul3(albedo, 0.18), add3(ambient, direct))
    }

    fn capture_direction(&self, origin: [f32; 3], direction: [f32; 3]) -> [f32; 4] {
        if let Some(hit) = self.trace_surface(origin, direction) {
            let shaded = self.shade_hit(&hit);
            [shaded[0], shaded[1], shaded[2], 1.0]
        } else {
            let env = self.sample_environment(direction);
            [env[0], env[1], env[2], 1.0]
        }
    }

    fn bake_base_mip(&self, origin: [f32; 3]) -> ReflectionProbeMip {
        let size = self.resolution.max(4);
        let mut texels =
            Vec::with_capacity((REFLECTION_PROBE_FACE_COUNT as u32 * size * size) as usize);
        for face in 0..REFLECTION_PROBE_FACE_COUNT {
            for y in 0..size {
                for x in 0..size {
                    let uv = [
                        ((x as f32 + 0.5) / size as f32) * 2.0 - 1.0,
                        ((y as f32 + 0.5) / size as f32) * 2.0 - 1.0,
                    ];
                    let dir = cube_face_direction(face, uv[0], uv[1]);
                    texels.push(self.capture_direction(origin, dir));
                }
            }
        }
        ReflectionProbeMip { size, texels }
    }

    fn prefilter_mip(
        &self,
        base_mip: &ReflectionProbeMip,
        mip_index: u32,
        mip_count: u32,
    ) -> ReflectionProbeMip {
        let size = (self.resolution >> mip_index).max(1);
        let roughness = if mip_count > 1 {
            (mip_index as f32 / (mip_count - 1) as f32).sqrt()
        } else {
            0.0
        };
        let sample_count = self.prefilter_sample_count.max(1);
        let mut texels =
            Vec::with_capacity((REFLECTION_PROBE_FACE_COUNT as u32 * size * size) as usize);

        for face in 0..REFLECTION_PROBE_FACE_COUNT {
            for y in 0..size {
                for x in 0..size {
                    let uv = [
                        ((x as f32 + 0.5) / size as f32) * 2.0 - 1.0,
                        ((y as f32 + 0.5) / size as f32) * 2.0 - 1.0,
                    ];
                    let normal = cube_face_direction(face, uv[0], uv[1]);
                    if roughness <= 1e-3 {
                        texels.push(sample_cubemap(base_mip, normal));
                        continue;
                    }

                    let mut accum = [0.0; 3];
                    let mut total_weight = 0.0;
                    for sample_index in 0..sample_count {
                        let xi = hammersley_2d(sample_index, sample_count);
                        let half_dir = importance_sample_ggx(xi, normal, roughness);
                        let light_dir = reflect3(neg3(normal), half_dir);
                        let n_dot_l = dot3(normal, light_dir).max(0.0);
                        if n_dot_l > 0.0 {
                            let color = sample_cubemap(base_mip, light_dir);
                            accum = add3(accum, mul3([color[0], color[1], color[2]], n_dot_l));
                            total_weight += n_dot_l;
                        }
                    }
                    let filtered = if total_weight > 0.0 {
                        mul3(accum, 1.0 / total_weight)
                    } else {
                        let color = sample_cubemap(base_mip, normal);
                        [color[0], color[1], color[2]]
                    };
                    texels.push([filtered[0], filtered[1], filtered[2], 1.0]);
                }
            }
        }

        ReflectionProbeMip { size, texels }
    }

    fn bake_probe(&self, origin: [f32; 3]) -> ReflectionProbe {
        let mip_count = self.resolution.max(1).ilog2() + 1;
        let base_mip = self.bake_base_mip(origin);
        let mut average = [0.0; 3];
        for texel in &base_mip.texels {
            average[0] += texel[0];
            average[1] += texel[1];
            average[2] += texel[2];
        }
        let inv_texel_count = 1.0 / base_mip.texels.len().max(1) as f32;
        average = mul3(average, inv_texel_count);

        let mut mips = Vec::with_capacity(mip_count as usize);
        mips.push(base_mip.clone());
        for mip_index in 1..mip_count {
            mips.push(self.prefilter_mip(&base_mip, mip_index, mip_count));
        }

        ReflectionProbe {
            position_ws: origin,
            average,
            mips,
        }
    }

    pub fn bake(&self, placement: &ProbePlacement) -> Result<ReflectionProbeSet, ProbeError> {
        if self.resolution == 0 {
            return Err(ProbeError::BakeFailed(
                "reflection probe resolution must be > 0".to_string(),
            ));
        }
        let probes = placement
            .positions_ws
            .iter()
            .map(|position| self.bake_probe(*position))
            .collect();
        Ok(ReflectionProbeSet {
            resolution: self.resolution,
            mip_level_count: self.resolution.max(1).ilog2() + 1,
            probes,
        })
    }
}

fn sample_hdr_equirect(hdr: &HdrImage, u: f32, v: f32) -> [f32; 3] {
    let width = hdr.width.max(1) as usize;
    let height = hdr.height.max(1) as usize;
    let x = u.rem_euclid(1.0) * (width as f32 - 1.0);
    let y = v.clamp(0.0, 1.0) * (height as f32 - 1.0);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1) % width;
    let y1 = (y0 + 1).min(height - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;

    let sample = |sx: usize, sy: usize| {
        let index = (sy * width + sx) * 3;
        [hdr.data[index], hdr.data[index + 1], hdr.data[index + 2]]
    };

    let c00 = sample(x0, y0);
    let c10 = sample(x1, y0);
    let c01 = sample(x0, y1);
    let c11 = sample(x1, y1);
    lerp3(lerp3(c00, c10, tx), lerp3(c01, c11, tx), ty)
}

fn sample_cubemap(mip: &ReflectionProbeMip, direction: [f32; 3]) -> [f32; 4] {
    let dir = normalize3(direction);
    let (face, u, v) = direction_to_face_uv(dir);
    let size = mip.size.max(1);
    let x = u.clamp(0.0, 1.0) * (size as f32 - 1.0);
    let y = v.clamp(0.0, 1.0) * (size as f32 - 1.0);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(size - 1);
    let y1 = (y0 + 1).min(size - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let face_size = size as usize * size as usize;
    let face_offset = face * face_size;

    let sample =
        |sx: u32, sy: u32| -> [f32; 4] { mip.texels[face_offset + (sy * size + sx) as usize] };
    let c00 = sample(x0, y0);
    let c10 = sample(x1, y0);
    let c01 = sample(x0, y1);
    let c11 = sample(x1, y1);
    lerp4(lerp4(c00, c10, tx), lerp4(c01, c11, tx), ty)
}

fn cube_face_direction(face_index: usize, u: f32, v: f32) -> [f32; 3] {
    normalize3(match face_index {
        0 => [1.0, -v, -u],
        1 => [-1.0, -v, u],
        2 => [u, 1.0, v],
        3 => [u, -1.0, -v],
        4 => [u, -v, 1.0],
        5 => [-u, -v, -1.0],
        _ => [1.0, 0.0, 0.0],
    })
}

fn direction_to_face_uv(direction: [f32; 3]) -> (usize, f32, f32) {
    let ax = direction[0].abs();
    let ay = direction[1].abs();
    let az = direction[2].abs();
    let (face, uc, vc, ma) = if ax >= ay && ax >= az {
        if direction[0] > 0.0 {
            (0usize, -direction[2], -direction[1], ax)
        } else {
            (1usize, direction[2], -direction[1], ax)
        }
    } else if ay >= ax && ay >= az {
        if direction[1] > 0.0 {
            (2usize, direction[0], direction[2], ay)
        } else {
            (3usize, direction[0], -direction[2], ay)
        }
    } else if direction[2] > 0.0 {
        (4usize, direction[0], -direction[1], az)
    } else {
        (5usize, -direction[0], -direction[1], az)
    };
    (face, 0.5 * (uc / ma + 1.0), 0.5 * (vc / ma + 1.0))
}

fn hammersley_2d(i: u32, n: u32) -> [f32; 2] {
    let mut bits = i;
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    [i as f32 / n.max(1) as f32, bits as f32 * 2.328_306_4e-10]
}

fn importance_sample_ggx(xi: [f32; 2], normal: [f32; 3], roughness: f32) -> [f32; 3] {
    let a = roughness * roughness;
    let phi = TWO_PI * xi[0];
    let cos_theta = ((1.0 - xi[1]) / (1.0 + (a * a - 1.0) * xi[1])).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

    let h = [phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta];
    let up = if normal[2].abs() < 0.999 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let tangent = normalize3(cross3(up, normal));
    let bitangent = cross3(normal, tangent);
    normalize3(add3(
        add3(mul3(tangent, h[0]), mul3(bitangent, h[1])),
        mul3(normal, h[2]),
    ))
}

fn apply_overlay_blend(
    base: [f32; 3],
    overlay: [f32; 3],
    blend_mode: Option<ReflectionOverlayBlend>,
    strength: f32,
) -> [f32; 3] {
    let amount = strength.clamp(0.0, 1.0);
    match blend_mode.unwrap_or(ReflectionOverlayBlend::Alpha) {
        ReflectionOverlayBlend::Alpha => lerp3(base, overlay, amount),
        ReflectionOverlayBlend::Add => add3(base, mul3(overlay, amount)),
        ReflectionOverlayBlend::Multiply => lerp3(base, mul3_components(base, overlay), amount),
        ReflectionOverlayBlend::Replace => lerp3(base, overlay, amount),
    }
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = dot3(v, v).sqrt().max(1e-6);
    [v[0] / len, v[1] / len, v[2] / len]
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn neg3(v: [f32; 3]) -> [f32; 3] {
    [-v[0], -v[1], -v[2]]
}

fn mul3(v: [f32; 3], scalar: f32) -> [f32; 3] {
    [v[0] * scalar, v[1] * scalar, v[2] * scalar]
}

fn mul3_components(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
        lerp(a[3], b[3], t),
    ]
}

fn reflect3(i: [f32; 3], n: [f32; 3]) -> [f32; 3] {
    add3(i, mul3(n, -2.0 * dot3(n, i)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::probes::{ProbeGridDesc, ProbePlacement};

    fn flat_heightfield(dim: u32) -> Vec<f32> {
        vec![0.0; (dim * dim) as usize]
    }

    fn simple_material() -> ReflectionTerrainMaterial {
        ReflectionTerrainMaterial {
            albedo_mode: ReflectionAlbedoMode::Material,
            colormap_strength: 1.0,
            raw_height_range: (0.0, 1.0),
            overlay: None,
            grass_color: [0.2, 0.35, 0.12],
            dirt_color: [0.35, 0.24, 0.12],
            rock_color: [0.38, 0.35, 0.32],
            snow_color: [0.95, 0.96, 1.0],
            snow_enabled: false,
            snow_altitude_min: 1.0,
            snow_altitude_blend: 1.0,
            snow_slope_max_deg: 45.0,
            snow_slope_blend_deg: 10.0,
            snow_aspect_influence: 0.3,
            rock_enabled: false,
            rock_slope_min_deg: 45.0,
            rock_slope_blend_deg: 10.0,
            wetness_enabled: false,
            wetness_strength: 0.0,
            wetness_slope_influence: 0.0,
        }
    }

    #[test]
    fn test_reflection_probe_bake_deterministic() {
        let grid = ProbeGridDesc {
            origin: [0.0, 0.0],
            spacing: [100.0, 100.0],
            dims: [1, 1],
            height_offset: 5.0,
            influence_radius: 0.0,
        };
        let placement = ProbePlacement::new(grid, vec![[0.0, 0.0, 5.0]]);
        let baker = HeightfieldReflectionBaker {
            heightfield: &flat_heightfield(32),
            height_dims: (32, 32),
            terrain_span: [100.0, 100.0],
            z_scale: 1.0,
            resolution: 8,
            prefilter_sample_count: 16,
            trace_steps: 96,
            trace_refine_steps: 4,
            max_trace_distance: 150.0,
            material: simple_material(),
            lighting: ReflectionCaptureLighting {
                env_image: None,
                env_intensity: 1.0,
                env_rotation_rad: 0.0,
                light_dir: normalize3([0.5, 0.25, 0.8]),
                light_color: [1.0, 0.96, 0.92],
                light_intensity: 1.0,
            },
        };
        let a = baker.bake(&placement).unwrap();
        let b = baker.bake(&placement).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_reflection_probe_generates_mips() {
        let grid = ProbeGridDesc {
            origin: [0.0, 0.0],
            spacing: [100.0, 100.0],
            dims: [1, 1],
            height_offset: 5.0,
            influence_radius: 0.0,
        };
        let placement = ProbePlacement::new(grid, vec![[0.0, 0.0, 5.0]]);
        let baker = HeightfieldReflectionBaker {
            heightfield: &flat_heightfield(16),
            height_dims: (16, 16),
            terrain_span: [100.0, 100.0],
            z_scale: 1.0,
            resolution: 8,
            prefilter_sample_count: 8,
            trace_steps: 64,
            trace_refine_steps: 2,
            max_trace_distance: 150.0,
            material: simple_material(),
            lighting: ReflectionCaptureLighting {
                env_image: None,
                env_intensity: 1.0,
                env_rotation_rad: 0.0,
                light_dir: normalize3([0.25, 0.1, 0.95]),
                light_color: [1.0, 1.0, 1.0],
                light_intensity: 1.0,
            },
        };
        let baked = baker.bake(&placement).unwrap();
        assert_eq!(baked.mip_level_count, 4);
        assert_eq!(baked.probes[0].mips.len(), 4);
        assert_eq!(baked.probes[0].mips[0].size, 8);
        assert_eq!(baked.probes[0].mips[3].size, 1);
    }
}
