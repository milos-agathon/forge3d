pub const DEFAULT_LIGHT_DIR: [f32; 3] = [0.408_248_28, 0.408_248_28, 0.816_496_55];

#[derive(Clone, Copy, Debug, Default)]
pub struct BrdfTileOverrides {
    pub light_dir: Option<[f32; 3]>,
    pub debug_kind: Option<u32>,
}

pub(super) fn normalize_light_dir(dir: [f32; 3]) -> Option<[f32; 3]> {
    let len_sq = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
    if len_sq <= 1e-8 {
        return None;
    }
    let inv_len = len_sq.sqrt().recip();
    Some([dir[0] * inv_len, dir[1] * inv_len, dir[2] * inv_len])
}
