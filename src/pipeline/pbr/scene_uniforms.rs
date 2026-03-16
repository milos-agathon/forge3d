use super::*;

/// Scene uniforms for PBR pipeline (model, view, projection, normal matrices)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PbrSceneUniforms {
    pub model_matrix: [[f32; 4]; 4],
    pub view_matrix: [[f32; 4]; 4],
    pub projection_matrix: [[f32; 4]; 4],
    pub normal_matrix: [[f32; 4]; 4],
}

impl PbrSceneUniforms {
    /// Construct uniforms from model, view, and projection matrices.
    pub fn from_matrices(model: Mat4, view: Mat4, projection: Mat4) -> Self {
        let normal_matrix = compute_normal_matrix(model);
        Self {
            model_matrix: model.to_cols_array_2d(),
            view_matrix: view.to_cols_array_2d(),
            projection_matrix: projection.to_cols_array_2d(),
            normal_matrix: normal_matrix.to_cols_array_2d(),
        }
    }
}

impl Default for PbrSceneUniforms {
    fn default() -> Self {
        Self::from_matrices(Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY)
    }
}

fn compute_normal_matrix(model: Mat4) -> Mat4 {
    let determinant = model.determinant();
    if determinant.abs() < 1e-6 {
        Mat4::IDENTITY
    } else {
        model.inverse().transpose()
    }
}
