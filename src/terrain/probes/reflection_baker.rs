use crate::terrain::probes::types::{
    ProbeError, ProbePlacement, ReflectionProbe, ReflectionProbeSet,
};

const FACE_NORMALS: [[f32; 3]; 6] = [
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
];

pub struct HeightfieldReflectionBaker {
    pub heightfield: Vec<f32>,
    pub height_dims: (u32, u32),
    pub terrain_span: [f32; 2],
    pub sky_color: [f32; 3],
    pub sky_intensity: f32,
    pub ground_color: [f32; 3],
    pub ray_count: u32,
    pub max_trace_distance: f32,
}

impl HeightfieldReflectionBaker {
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

    fn cube_face_direction(face_index: usize, u: f32, v: f32) -> [f32; 3] {
        let dir = match face_index {
            0 => [1.0, v, -u],
            1 => [-1.0, v, u],
            2 => [u, 1.0, -v],
            3 => [u, -1.0, v],
            4 => [u, v, 1.0],
            5 => [-u, v, -1.0],
            _ => FACE_NORMALS[0],
        };
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        [
            dir[0] / len.max(1e-6),
            dir[1] / len.max(1e-6),
            dir[2] / len.max(1e-6),
        ]
    }

    fn sample_face_directions(&self, face_index: usize) -> Vec<[f32; 3]> {
        let side = (self.ray_count.max(1) as f32).sqrt().ceil() as u32;
        let mut directions = Vec::with_capacity((side * side) as usize);
        for y in 0..side {
            for x in 0..side {
                let u = ((x as f32 + 0.5) / side as f32) * 2.0 - 1.0;
                let v = ((y as f32 + 0.5) / side as f32) * 2.0 - 1.0;
                directions.push(Self::cube_face_direction(face_index, u, v));
            }
        }
        directions
    }

    fn trace_distance(&self, origin: [f32; 3], direction: [f32; 3]) -> Option<f32> {
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
                    return Some(t);
                }
            }
        }
        None
    }

    fn sample_direction_color(&self, origin: [f32; 3], direction: [f32; 3]) -> [f32; 3] {
        if let Some(distance) = self.trace_distance(origin, direction) {
            let proximity = 1.0 - (distance / self.max_trace_distance.max(1e-3)).clamp(0.0, 1.0);
            let terrain_weight = 0.4 + 0.6 * proximity;
            return [
                self.ground_color[0] * terrain_weight,
                self.ground_color[1] * terrain_weight,
                self.ground_color[2] * terrain_weight,
            ];
        }

        if direction[2] > 0.0 {
            [
                self.sky_color[0] * self.sky_intensity,
                self.sky_color[1] * self.sky_intensity,
                self.sky_color[2] * self.sky_intensity,
            ]
        } else {
            [
                self.ground_color[0] * 0.25,
                self.ground_color[1] * 0.25,
                self.ground_color[2] * 0.25,
            ]
        }
    }

    fn bake_probe(&self, origin: [f32; 3]) -> ReflectionProbe {
        let mut probe = ReflectionProbe::default();
        for face_index in 0..FACE_NORMALS.len() {
            let directions = self.sample_face_directions(face_index);
            let sample_count = directions.len().max(1) as f32;
            let mut accum = [0.0; 3];
            for direction in directions {
                let color = self.sample_direction_color(origin, direction);
                accum[0] += color[0];
                accum[1] += color[1];
                accum[2] += color[2];
            }
            probe.faces[face_index] = [
                accum[0] / sample_count,
                accum[1] / sample_count,
                accum[2] / sample_count,
            ];
        }

        for face in &probe.faces {
            probe.average[0] += face[0];
            probe.average[1] += face[1];
            probe.average[2] += face[2];
        }
        probe.average[0] /= FACE_NORMALS.len() as f32;
        probe.average[1] /= FACE_NORMALS.len() as f32;
        probe.average[2] /= FACE_NORMALS.len() as f32;
        probe
    }

    pub fn bake(&self, placement: &ProbePlacement) -> Result<ReflectionProbeSet, ProbeError> {
        let probes = placement
            .positions_ws
            .iter()
            .map(|position| self.bake_probe(*position))
            .collect();
        Ok(ReflectionProbeSet { probes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::probes::{ProbeGridDesc, ProbePlacement};

    fn flat_heightfield(dim: u32) -> Vec<f32> {
        vec![0.0; (dim * dim) as usize]
    }

    #[test]
    fn test_reflection_probe_bake_deterministic() {
        let grid = ProbeGridDesc {
            origin: [-50.0, -50.0],
            spacing: [100.0, 100.0],
            dims: [1, 1],
            height_offset: 3.0,
            influence_radius: 0.0,
        };
        let placement = ProbePlacement::new(grid, vec![[0.0, 0.0, 3.0]]);
        let baker = HeightfieldReflectionBaker {
            heightfield: flat_heightfield(32),
            height_dims: (32, 32),
            terrain_span: [100.0, 100.0],
            sky_color: [0.6, 0.75, 1.0],
            sky_intensity: 1.0,
            ground_color: [0.2, 0.16, 0.12],
            ray_count: 16,
            max_trace_distance: 100.0,
        };
        let result_a = baker.bake(&placement).unwrap();
        let result_b = baker.bake(&placement).unwrap();
        assert_eq!(result_a, result_b);
    }

    #[test]
    fn test_reflection_probe_up_face_brighter_than_down_face() {
        let grid = ProbeGridDesc {
            origin: [-50.0, -50.0],
            spacing: [100.0, 100.0],
            dims: [1, 1],
            height_offset: 5.0,
            influence_radius: 0.0,
        };
        let placement = ProbePlacement::new(grid, vec![[0.0, 0.0, 5.0]]);
        let baker = HeightfieldReflectionBaker {
            heightfield: flat_heightfield(32),
            height_dims: (32, 32),
            terrain_span: [100.0, 100.0],
            sky_color: [0.6, 0.75, 1.0],
            sky_intensity: 1.0,
            ground_color: [0.2, 0.16, 0.12],
            ray_count: 16,
            max_trace_distance: 100.0,
        };
        let result = baker.bake(&placement).unwrap();
        let up = result.probes[0].faces[4];
        let down = result.probes[0].faces[5];
        let up_luma = 0.2126 * up[0] + 0.7152 * up[1] + 0.0722 * up[2];
        let down_luma = 0.2126 * down[0] + 0.7152 * down[1] + 0.0722 * down[2];
        assert!(up_luma > down_luma);
    }
}
