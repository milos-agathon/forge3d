use super::*;

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct WaterReflectionUniforms {
    pub(super) reflection_view_proj: [[f32; 4]; 4],
    pub(super) water_plane: [f32; 4],
    pub(super) reflection_params: [f32; 4],
    pub(super) camera_world_pos: [f32; 4],
    pub(super) enable_flags: [f32; 4],
}

impl WaterReflectionUniforms {
    pub(super) fn disabled() -> Self {
        Self {
            reflection_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            water_plane: [0.0, 1.0, 0.0, 0.0],
            reflection_params: [0.8, 5.0, 0.02, 0.3],
            camera_world_pos: [0.0, 0.0, 0.0, 1.0],
            enable_flags: [0.0, 0.0, 0.5, 0.0],
        }
    }

    pub(super) fn enabled_main_pass(
        reflection_view_proj: [[f32; 4]; 4],
        water_plane_height: f32,
        camera_pos: [f32; 3],
        intensity: f32,
        fresnel_power: f32,
        wave_strength: f32,
        shore_atten_width: f32,
        resolution_scale: f32,
    ) -> Self {
        Self {
            reflection_view_proj,
            water_plane: [0.0, 0.0, 1.0, -water_plane_height],
            reflection_params: [intensity, fresnel_power, wave_strength, shore_atten_width],
            camera_world_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 1.0],
            enable_flags: [1.0, 0.0, resolution_scale, 0.0],
        }
    }

    pub(super) fn for_reflection_pass(water_plane_height: f32) -> Self {
        Self {
            reflection_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            water_plane: [0.0, 0.0, 1.0, -water_plane_height],
            reflection_params: [0.0, 0.0, 0.0, 0.0],
            camera_world_pos: [0.0, 0.0, 0.0, 1.0],
            enable_flags: [0.0, 1.0, 0.0, 0.0],
        }
    }
}

pub(super) fn compute_mirrored_view_matrix(
    view_matrix: [[f32; 4]; 4],
    plane_height: f32,
) -> [[f32; 4]; 4] {
    let reflect = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 2.0 * plane_height],
        [0.0, 0.0, 0.0, 1.0],
    ];
    mul_mat4(view_matrix, reflect)
}

pub(super) fn mul_mat4(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}
