use bytemuck::{Pod, Zeroable};

use crate::terrain::probes::types::{ProbeIrradianceSet, SHL2};

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ProbeGridUniformsGpu {
    pub grid_origin: [f32; 4],
    pub grid_params: [f32; 4],
    pub blend_params: [f32; 4],
}

impl ProbeGridUniformsGpu {
    pub fn disabled() -> Self {
        Self {
            grid_origin: [0.0, 0.0, 0.0, 0.0],
            grid_params: [1.0, 1.0, 1.0, 1.0],
            blend_params: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuProbeData {
    pub sh_r_01: [f32; 4],
    pub sh_r_23: [f32; 4],
    pub sh_r_4: [f32; 4],
    pub sh_g_01: [f32; 4],
    pub sh_g_23: [f32; 4],
    pub sh_g_4: [f32; 4],
    pub sh_b_01: [f32; 4],
    pub sh_b_23: [f32; 4],
    pub sh_b_4: [f32; 4],
}

impl GpuProbeData {
    pub fn zeroed() -> Self {
        <Self as Zeroable>::zeroed()
    }

    pub fn from_sh(sh: &SHL2) -> Self {
        let c = &sh.coeffs;
        Self {
            sh_r_01: [c[0][0], c[1][0], c[2][0], c[3][0]],
            sh_r_23: [c[4][0], c[5][0], c[6][0], c[7][0]],
            sh_r_4: [c[8][0], 0.0, 0.0, 0.0],
            sh_g_01: [c[0][1], c[1][1], c[2][1], c[3][1]],
            sh_g_23: [c[4][1], c[5][1], c[6][1], c[7][1]],
            sh_g_4: [c[8][1], 0.0, 0.0, 0.0],
            sh_b_01: [c[0][2], c[1][2], c[2][2], c[3][2]],
            sh_b_23: [c[4][2], c[5][2], c[6][2], c[7][2]],
            sh_b_4: [c[8][2], 0.0, 0.0, 0.0],
        }
    }

    pub fn to_sh(&self) -> SHL2 {
        let mut coeffs = [[0.0f32; 3]; 9];
        let r = [self.sh_r_01, self.sh_r_23, self.sh_r_4];
        let g = [self.sh_g_01, self.sh_g_23, self.sh_g_4];
        let b = [self.sh_b_01, self.sh_b_23, self.sh_b_4];
        for i in 0..9 {
            let block = i / 4;
            let lane = i % 4;
            coeffs[i][0] = r[block][lane];
            coeffs[i][1] = g[block][lane];
            coeffs[i][2] = b[block][lane];
        }
        SHL2 { coeffs }
    }
}

pub fn pack_probes_for_upload(set: &ProbeIrradianceSet) -> Vec<GpuProbeData> {
    set.probes.iter().map(GpuProbeData::from_sh).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_gpu_layout_size() {
        assert_eq!(std::mem::size_of::<GpuProbeData>(), 144);
        assert_eq!(std::mem::size_of::<ProbeGridUniformsGpu>(), 48);
    }

    #[test]
    fn test_zeroed_gpu_probe_data() {
        let z = GpuProbeData::zeroed();
        let bytes = bytemuck::bytes_of(&z);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_sh_packing_roundtrip() {
        let mut sh = SHL2 {
            coeffs: [[0.0; 3]; 9],
        };
        for (i, coeff) in sh.coeffs.iter_mut().enumerate() {
            *coeff = [i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3];
        }

        let gpu = GpuProbeData::from_sh(&sh);
        let roundtrip = gpu.to_sh();
        for i in 0..9 {
            for c in 0..3 {
                assert!(
                    (sh.coeffs[i][c] - roundtrip.coeffs[i][c]).abs() < 1e-6,
                    "Mismatch at coeff [{i}][{c}]"
                );
            }
        }
    }
}
