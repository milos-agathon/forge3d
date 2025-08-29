//! Transform utilities for T/R/S operations and model matrix composition
//! 
//! Provides helper functions for translation, rotation, scaling, and matrix composition
//! following right-handed coordinate conventions.

use pyo3::prelude::*;
use pyo3::Bound;
use numpy::PyArray2;
use glam::{Mat4, Vec3, Quat};

/// Create a translation matrix
#[pyfunction]
#[pyo3(text_signature = "(tx, ty, tz)")]
pub fn translate<'py>(
    py: Python<'py>,
    tx: f32,
    ty: f32,
    tz: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let translation = Vec3::new(tx, ty, tz);
    let matrix = Mat4::from_translation(translation);
    mat4_to_numpy(py, matrix)
}

/// Create a rotation matrix around X axis (in degrees)
#[pyfunction]
#[pyo3(text_signature = "(degrees)")]
pub fn rotate_x<'py>(
    py: Python<'py>,
    degrees: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let radians = degrees.to_radians();
    let matrix = Mat4::from_rotation_x(radians);
    mat4_to_numpy(py, matrix)
}

/// Create a rotation matrix around Y axis (in degrees)
#[pyfunction]
#[pyo3(text_signature = "(degrees)")]
pub fn rotate_y<'py>(
    py: Python<'py>,
    degrees: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let radians = degrees.to_radians();
    let matrix = Mat4::from_rotation_y(radians);
    mat4_to_numpy(py, matrix)
}

/// Create a rotation matrix around Z axis (in degrees)
#[pyfunction]
#[pyo3(text_signature = "(degrees)")]
pub fn rotate_z<'py>(
    py: Python<'py>,
    degrees: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let radians = degrees.to_radians();
    let matrix = Mat4::from_rotation_z(radians);
    mat4_to_numpy(py, matrix)
}

/// Create a scale matrix
#[pyfunction]
#[pyo3(text_signature = "(sx, sy, sz)")]
pub fn scale<'py>(
    py: Python<'py>,
    sx: f32,
    sy: f32,
    sz: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let scale_vec = Vec3::new(sx, sy, sz);
    let matrix = Mat4::from_scale(scale_vec);
    mat4_to_numpy(py, matrix)
}

/// Create a uniform scale matrix
#[pyfunction]
#[pyo3(text_signature = "(s)")]
pub fn scale_uniform<'py>(
    py: Python<'py>,
    s: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let matrix = Mat4::from_scale(Vec3::splat(s));
    mat4_to_numpy(py, matrix)
}

/// Compose a model matrix from separate T, R, S components
/// Applies transformations in T * R * S order (scale first, then rotate, then translate)
#[pyfunction]
#[pyo3(text_signature = "(translation, rotation_degrees, scale)")]
pub fn compose_trs<'py>(
    py: Python<'py>,
    translation: (f32, f32, f32),
    rotation_degrees: (f32, f32, f32), // (x, y, z) rotations in degrees
    scale: (f32, f32, f32),
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let t = Vec3::new(translation.0, translation.1, translation.2);
    let s = Vec3::new(scale.0, scale.1, scale.2);
    
    // Create rotation quaternion from Euler angles (in degrees)
    let rx = Quat::from_rotation_x(rotation_degrees.0.to_radians());
    let ry = Quat::from_rotation_y(rotation_degrees.1.to_radians());
    let rz = Quat::from_rotation_z(rotation_degrees.2.to_radians());
    
    // Apply rotations in Z * Y * X order (standard Euler order)
    let rotation = rz * ry * rx;
    
    // Compose T * R * S matrix
    let matrix = Mat4::from_scale_rotation_translation(s, rotation, t);
    mat4_to_numpy(py, matrix)
}

/// Create a look-at transformation matrix (for objects, not cameras)
/// This makes an object look towards a target from its position
#[pyfunction]
#[pyo3(text_signature = "(position, target, up)")]
pub fn look_at_transform<'py>(
    py: Python<'py>,
    position: (f32, f32, f32),
    target: (f32, f32, f32),
    up: (f32, f32, f32),
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let pos = Vec3::new(position.0, position.1, position.2);
    let tgt = Vec3::new(target.0, target.1, target.2);
    let up_vec = Vec3::new(up.0, up.1, up.2);
    
    // Compute forward direction (object looks towards target)
    let forward = (tgt - pos).normalize();
    
    // Compute right direction
    let right = forward.cross(up_vec).normalize();
    
    // Recompute up to ensure orthogonality
    let corrected_up = right.cross(forward);
    
    // Create rotation matrix from basis vectors
    // Note: This creates an object-to-world transform (opposite of camera look-at)
    let rotation = Mat4::from_cols(
        right.extend(0.0),
        corrected_up.extend(0.0),
        (-forward).extend(0.0), // Negative because object looks along +Z in local space
        Vec3::ZERO.extend(1.0),
    );
    
    let translation = Mat4::from_translation(pos);
    let matrix = translation * rotation;
    
    mat4_to_numpy(py, matrix)
}

/// Multiply two 4x4 matrices (matrix multiplication: left * right)
#[pyfunction]
#[pyo3(text_signature = "(left, right)")]
pub fn multiply_matrices<'py>(
    py: Python<'py>,
    left: &pyo3::types::PyAny,
    right: &pyo3::types::PyAny,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // Convert input arrays to glam matrices
    let left_mat = numpy_to_mat4(left)?;
    let right_mat = numpy_to_mat4(right)?;
    
    // Multiply matrices
    let result = left_mat * right_mat;
    
    mat4_to_numpy(py, result)
}

/// Compute the inverse of a 4x4 matrix
#[pyfunction]
#[pyo3(text_signature = "(matrix)")]
pub fn invert_matrix<'py>(
    py: Python<'py>,
    matrix: &pyo3::types::PyAny,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let mat = numpy_to_mat4(matrix)?;
    let inverse = mat.inverse();
    mat4_to_numpy(py, inverse)
}

/// Compute the 3x3 normal matrix from a 4x4 model matrix
/// This is the inverse transpose of the upper-left 3x3 submatrix
/// Used for transforming normals when the model matrix has non-uniform scaling
pub fn normal_matrix3x3(model_matrix: Mat4) -> glam::Mat3 {
    // Extract the upper-left 3x3 rotation/scale part
    let mat3 = glam::Mat3::from_mat4(model_matrix);
    
    // Compute inverse transpose for proper normal transformation
    mat3.inverse().transpose()
}

/// Compute the 3x3 normal matrix from a 4x4 model matrix (Python interface)
#[pyfunction]
#[pyo3(text_signature = "(model_matrix)")]
pub fn compute_normal_matrix<'py>(
    py: Python<'py>,
    model_matrix: &pyo3::types::PyAny,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let model_mat = numpy_to_mat4(model_matrix)?;
    let normal_mat3 = normal_matrix3x3(model_mat);
    
    // Convert 3x3 matrix to numpy (extend to 4x4 for consistency)
    let normal_mat4 = Mat4::from_mat3(normal_mat3);
    mat4_to_numpy(py, normal_mat4)
}

// Helper functions for matrix conversion

/// Convert a Mat4 to a NumPy array with shape (4,4) and dtype float32, C-contiguous
fn mat4_to_numpy<'py>(py: Python<'py>, mat: Mat4) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // glam Mat4 is column-major, but we want to return it as a (4,4) array
    // where the indexing matches mathematical conventions
    let data = mat.to_cols_array_2d();
    
    // Create a flattened array in row-major order for NumPy
    let flat: Vec<f32> = (0..4).flat_map(|row| {
        (0..4).map(move |col| data[col][row])
    }).collect();
    
    let array = PyArray2::from_vec2_bound(py, &vec![
        flat[0..4].to_vec(),
        flat[4..8].to_vec(), 
        flat[8..12].to_vec(),
        flat[12..16].to_vec(),
    ])?;
    
    Ok(array)
}

/// Convert a NumPy array to a Mat4
fn numpy_to_mat4(array: &pyo3::types::PyAny) -> PyResult<Mat4> {
    use numpy::{PyArrayMethods, PyReadonlyArray2};
    use numpy::PyUntypedArrayMethods;
    
    // Try to convert to readonly array
    let array_f32: PyReadonlyArray2<f32> = array.extract()?;
    
    // Check shape
    if array_f32.shape() != [4, 4] {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            format!("Expected (4,4) matrix, got shape {:?}", array_f32.shape())
        ));
    }
    
    // Check contiguity
    if !array_f32.is_contiguous() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Matrix must be C-contiguous"
        ));
    }
    
    // Extract data (NumPy is row-major, glam expects column-major)
    let data = array_f32.as_slice()?;
    let mut cols_array = [0.0f32; 16];
    
    // Convert from row-major to column-major
    for row in 0..4 {
        for col in 0..4 {
            cols_array[col * 4 + row] = data[row * 4 + col];
        }
    }
    
    Ok(Mat4::from_cols_array(&cols_array))
}