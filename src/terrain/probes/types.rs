/// Grid descriptor describing where probes live in world space.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeGridDesc {
    pub origin: [f32; 2],
    pub spacing: [f32; 2],
    pub dims: [u32; 2],
    pub height_offset: f32,
    pub influence_radius: f32,
}

/// World-space probe placement resolved from the grid descriptor.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbePlacement {
    pub grid: ProbeGridDesc,
    pub positions_ws: Vec<[f32; 3]>,
}

impl ProbePlacement {
    pub fn new(grid: ProbeGridDesc, positions_ws: Vec<[f32; 3]>) -> Self {
        assert_eq!(
            positions_ws.len(),
            (grid.dims[0] * grid.dims[1]) as usize,
            "positions_ws.len() must equal dims[0] * dims[1]"
        );
        Self { grid, positions_ws }
    }
}

/// SH L2 coefficients: 9 basis functions x RGB.
#[derive(Clone, Debug, PartialEq)]
pub struct SHL2 {
    pub coeffs: [[f32; 3]; 9],
}

impl Default for SHL2 {
    fn default() -> Self {
        Self {
            coeffs: [[0.0; 3]; 9],
        }
    }
}

/// One irradiance payload per resolved probe.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeIrradianceSet {
    pub probes: Vec<SHL2>,
}

pub const REFLECTION_PROBE_FACE_COUNT: usize = 6;

/// One cubemap mip payload for a reflection probe.
#[derive(Clone, Debug, PartialEq)]
pub struct ReflectionProbeMip {
    pub size: u32,
    /// Face-major RGBA texels: 6 * size * size entries.
    pub texels: Vec<[f32; 4]>,
}

impl ReflectionProbeMip {
    pub fn face_texels(&self, face_index: usize) -> &[[f32; 4]] {
        let face_len = (self.size * self.size) as usize;
        let start = face_index * face_len;
        let end = start + face_len;
        &self.texels[start..end]
    }
}

/// Low-resolution local reflection cubemap data for one placed probe.
#[derive(Clone, Debug, PartialEq)]
pub struct ReflectionProbe {
    pub position_ws: [f32; 3],
    pub average: [f32; 3],
    pub mips: Vec<ReflectionProbeMip>,
}

/// One reflection cubemap payload per resolved probe.
#[derive(Clone, Debug, PartialEq)]
pub struct ReflectionProbeSet {
    pub resolution: u32,
    pub mip_level_count: u32,
    pub probes: Vec<ReflectionProbe>,
}

/// Probe system errors.
#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    #[error("Probe bake failed: {0}")]
    BakeFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_placement_invariant() {
        let grid = ProbeGridDesc {
            origin: [0.0, 0.0],
            spacing: [10.0, 10.0],
            dims: [4, 4],
            height_offset: 5.0,
            influence_radius: 0.0,
        };
        let positions: Vec<[f32; 3]> = (0..16).map(|i| [i as f32, 0.0, 0.0]).collect();
        let placement = ProbePlacement::new(grid, positions);
        assert_eq!(placement.positions_ws.len(), 16);
    }

    #[test]
    #[should_panic(expected = "positions_ws.len() must equal dims[0] * dims[1]")]
    fn test_probe_placement_invariant_fails_on_mismatch() {
        let grid = ProbeGridDesc {
            origin: [0.0, 0.0],
            spacing: [10.0, 10.0],
            dims: [4, 4],
            height_offset: 5.0,
            influence_radius: 0.0,
        };
        let positions: Vec<[f32; 3]> = (0..10).map(|i| [i as f32, 0.0, 0.0]).collect();
        ProbePlacement::new(grid, positions);
    }
}
