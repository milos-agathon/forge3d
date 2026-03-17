use super::*;

impl CloudRenderer {
    pub(super) fn build_noise_data(resolution: u32) -> (Vec<u8>, u32) {
        let padded_row = Self::align_to(resolution, 256);
        let mut data = vec![0u8; padded_row as usize * resolution as usize * resolution as usize];
        for z in 0..resolution {
            for y in 0..resolution {
                let row_offset = ((z * resolution + y) * padded_row) as usize;
                for x in 0..resolution {
                    let f = (x as f32 * 0.125 + y as f32 * 0.175 + z as f32 * 0.215).sin();
                    let g = (x as f32 * 0.05 + y as f32 * 0.09 + z as f32 * 0.07).cos();
                    let value = ((f + g) * 0.25 + 0.5).clamp(0.0, 1.0);
                    data[row_offset + x as usize] = (value * 255.0) as u8;
                }
            }
        }
        (data, padded_row)
    }

    pub(super) fn build_shape_data(size: u32) -> (Vec<u8>, u32) {
        let padded_row = Self::align_to(size, 256);
        let mut data = vec![0u8; padded_row as usize * size as usize];
        let half = size as f32 / 2.0;
        for y in 0..size {
            let dy = (y as f32 - half) / half;
            for x in 0..size {
                let dx = (x as f32 - half) / half;
                let dist = (dx * dx + dy * dy).sqrt();
                let softness = 1.0 - dist.powf(1.5);
                let noise = (x as f32 * 0.1).sin() * 0.1 + (y as f32 * 0.12).cos() * 0.1;
                let value = (softness + noise).clamp(0.0, 1.0);
                let offset = (y * padded_row + x) as usize;
                data[offset] = (value * 255.0) as u8;
            }
        }
        (data, padded_row)
    }

    pub(super) fn align_to(value: u32, alignment: u32) -> u32 {
        ((value + alignment - 1) / alignment) * alignment
    }

    pub(super) fn float_to_u8(value: f32) -> u8 {
        ((value.max(0.0).min(1.0)) * 255.0 + 0.5) as u8
    }
}
