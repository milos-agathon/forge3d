use crate::terrain::probes::baker::ProbeBaker;
#[cfg(test)]
use crate::terrain::probes::types::ProbeGridDesc;
use crate::terrain::probes::types::{ProbeError, ProbeIrradianceSet, ProbePlacement, SHL2};

pub struct HeightfieldAnalyticalBaker {
    pub heightfield: Vec<f32>,
    pub height_dims: (u32, u32),
    pub terrain_span: [f32; 2],
    pub sky_color: [f32; 3],
    pub sky_intensity: f32,
    pub ray_count: u32,
    pub max_trace_distance: f32,
}

impl HeightfieldAnalyticalBaker {
    fn sample_height(&self, world_x: f32, world_y: f32) -> Option<f32> {
        let (w, h) = self.height_dims;
        if w == 0 || h == 0 || self.heightfield.is_empty() {
            return None;
        }
        if w == 1 || h == 1 {
            let value = self.heightfield[0];
            return value.is_finite().then_some(value);
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
            ((1.0 - tx) * (1.0 - ty), self.sample_texel(x0, y0)),
            (tx * (1.0 - ty), self.sample_texel(x1, y0)),
            ((1.0 - tx) * ty, self.sample_texel(x0, y1)),
            (tx * ty, self.sample_texel(x1, y1)),
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

    fn sample_texel(&self, x: u32, y: u32) -> Option<f32> {
        let value = self.heightfield[(y * self.height_dims.0 + x) as usize];
        value.is_finite().then_some(value)
    }

    fn hemisphere_directions(count: u32) -> Vec<[f32; 3]> {
        let sample_count = count.max(1);
        let golden_angle = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
        (0..sample_count)
            .map(|i| {
                let t = (i as f32 + 0.5) / sample_count as f32;
                let z = t;
                let radius = (1.0 - z * z).max(0.0).sqrt();
                let theta = golden_angle * i as f32;
                [radius * theta.cos(), radius * theta.sin(), z]
            })
            .collect()
    }

    fn sh_basis(direction: [f32; 3]) -> [f32; 9] {
        let [x, y, z] = direction;
        [
            0.282095,
            0.488603 * y,
            0.488603 * z,
            0.488603 * x,
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3.0 * z * z - 1.0),
            1.092548 * x * z,
            0.546274 * (x * x - y * y),
        ]
    }

    fn ray_occluded(&self, origin: [f32; 3], direction: [f32; 3]) -> bool {
        let step_count = 96u32;
        let step_size = self.max_trace_distance.max(1e-3) / step_count as f32;
        for step in 1..=step_count {
            let t = step as f32 * step_size;
            let sample_pos = [
                origin[0] + direction[0] * t,
                origin[1] + direction[1] * t,
                origin[2] + direction[2] * t,
            ];
            if let Some(height) = self.sample_height(sample_pos[0], sample_pos[1]) {
                if height > sample_pos[2] {
                    return true;
                }
            }
        }
        false
    }

    fn bake_probe(&self, origin: [f32; 3], directions: &[[f32; 3]]) -> SHL2 {
        let mut sh = SHL2::default();
        let ray_count = directions.len().max(1) as f32;
        let solid_angle = 2.0 * std::f32::consts::PI / ray_count;

        for direction in directions {
            let cos_theta = direction[2].max(0.0);
            if cos_theta <= 0.0 || self.ray_occluded(origin, *direction) {
                continue;
            }
            let basis = Self::sh_basis(*direction);
            for (basis_index, basis_value) in basis.into_iter().enumerate() {
                let weight = basis_value * cos_theta * solid_angle * self.sky_intensity;
                sh.coeffs[basis_index][0] += self.sky_color[0] * weight;
                sh.coeffs[basis_index][1] += self.sky_color[1] * weight;
                sh.coeffs[basis_index][2] += self.sky_color[2] * weight;
            }
        }

        sh
    }
}

impl ProbeBaker for HeightfieldAnalyticalBaker {
    fn bake(&self, placement: &ProbePlacement) -> Result<ProbeIrradianceSet, ProbeError> {
        let directions = Self::hemisphere_directions(self.ray_count.max(1));
        let probes = placement
            .positions_ws
            .iter()
            .map(|position| self.bake_probe(*position, &directions))
            .collect();
        Ok(ProbeIrradianceSet { probes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_heightfield(dim: u32) -> Vec<f32> {
        vec![0.0; (dim * dim) as usize]
    }

    fn test_grid(dims: [u32; 2]) -> ProbeGridDesc {
        ProbeGridDesc {
            origin: [-50.0, -50.0],
            spacing: [
                100.0 / (dims[0].max(2) - 1) as f32,
                100.0 / (dims[1].max(2) - 1) as f32,
            ],
            dims,
            height_offset: 5.0,
            influence_radius: 0.0,
        }
    }

    #[test]
    fn test_probe_bake_deterministic() {
        let dim = 64u32;
        let grid = test_grid([4, 4]);
        let baker = HeightfieldAnalyticalBaker {
            heightfield: flat_heightfield(dim),
            height_dims: (dim, dim),
            terrain_span: [100.0, 100.0],
            sky_color: [0.6, 0.75, 1.0],
            sky_intensity: 1.0,
            ray_count: 64,
            max_trace_distance: 50.0,
        };
        let positions: Vec<[f32; 3]> = (0..16)
            .map(|i| {
                let col = (i % 4) as f32;
                let row = (i / 4) as f32;
                [
                    grid.origin[0] + grid.spacing[0] * col,
                    grid.origin[1] + grid.spacing[1] * row,
                    5.0,
                ]
            })
            .collect();
        let placement = ProbePlacement::new(grid, positions);

        let result1 = baker.bake(&placement).unwrap();
        let result2 = baker.bake(&placement).unwrap();

        for (probe_idx, (left, right)) in
            result1.probes.iter().zip(result2.probes.iter()).enumerate()
        {
            for coeff_idx in 0..9 {
                for channel in 0..3 {
                    assert_eq!(
                        left.coeffs[coeff_idx][channel], right.coeffs[coeff_idx][channel],
                        "Non-deterministic at probe {probe_idx}, coeff [{coeff_idx}][{channel}]"
                    );
                }
            }
        }
    }

    #[test]
    fn test_flat_terrain_unoccluded() {
        let dim = 32u32;
        let grid = test_grid([2, 2]);
        let baker = HeightfieldAnalyticalBaker {
            heightfield: flat_heightfield(dim),
            height_dims: (dim, dim),
            terrain_span: [100.0, 100.0],
            sky_color: [1.0, 1.0, 1.0],
            sky_intensity: 1.0,
            ray_count: 64,
            max_trace_distance: 50.0,
        };
        let positions = vec![
            [-50.0, -50.0, 5.0],
            [50.0, -50.0, 5.0],
            [-50.0, 50.0, 5.0],
            [50.0, 50.0, 5.0],
        ];
        let placement = ProbePlacement::new(grid, positions);
        let result = baker.bake(&placement).unwrap();

        for (probe_idx, probe) in result.probes.iter().enumerate() {
            assert!(
                probe.coeffs[0][0] > 0.0,
                "Probe {probe_idx} L0 R should be > 0, got {}",
                probe.coeffs[0][0]
            );
        }
    }

    #[test]
    fn test_nodata_heightfield_no_nan() {
        let dim = 32u32;
        let mut hf = flat_heightfield(dim);
        hf[0] = f32::NAN;
        hf[1] = f32::INFINITY;
        hf[2] = f32::NEG_INFINITY;

        let grid = test_grid([2, 2]);
        let baker = HeightfieldAnalyticalBaker {
            heightfield: hf,
            height_dims: (dim, dim),
            terrain_span: [100.0, 100.0],
            sky_color: [1.0, 1.0, 1.0],
            sky_intensity: 1.0,
            ray_count: 32,
            max_trace_distance: 50.0,
        };
        let positions = vec![
            [-50.0, -50.0, 5.0],
            [50.0, -50.0, 5.0],
            [-50.0, 50.0, 5.0],
            [50.0, 50.0, 5.0],
        ];
        let placement = ProbePlacement::new(grid, positions);
        let result = baker.bake(&placement).unwrap();

        for (probe_idx, probe) in result.probes.iter().enumerate() {
            for coeff_idx in 0..9 {
                for channel in 0..3 {
                    assert!(
                        !probe.coeffs[coeff_idx][channel].is_nan(),
                        "NaN at probe {probe_idx}, coeff [{coeff_idx}][{channel}]"
                    );
                    assert!(
                        probe.coeffs[coeff_idx][channel].is_finite(),
                        "Infinite at probe {probe_idx}, coeff [{coeff_idx}][{channel}]"
                    );
                }
            }
        }
    }
}
