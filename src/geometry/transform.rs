use super::{GeometryError, MeshBuffers};
use glam::{Mat3, Vec3};

const EPSILON: f32 = 1e-8;

fn bounding_box(positions: &[[f32; 3]]) -> Option<(Vec3, Vec3)> {
    if positions.is_empty() {
        return None;
    }

    let mut min = Vec3::from_array(positions[0]);
    let mut max = min;

    for p in positions.iter().copied().skip(1) {
        let v = Vec3::from_array(p);
        min = min.min(v);
        max = max.max(v);
    }

    Some((min, max))
}

fn translate_mesh(mesh: &mut MeshBuffers, offset: Vec3) {
    if offset.abs_diff_eq(Vec3::ZERO, 0.0) {
        return;
    }

    for position in &mut mesh.positions {
        let mut v = Vec3::from_array(*position);
        v += offset;
        *position = v.to_array();
    }
}

fn apply_linear(mesh: &mut MeshBuffers, linear: Mat3) -> Result<bool, GeometryError> {
    let det = linear.determinant();
    if det.abs() < EPSILON {
        return Err(GeometryError::new(
            "transform matrix must be non-singular for mesh operations",
        ));
    }

    for position in &mut mesh.positions {
        let v = Vec3::from_array(*position);
        let transformed = linear * v;
        *position = transformed.to_array();
    }

    if !mesh.normals.is_empty() {
        let normal_matrix = linear.inverse().transpose();
        for normal in &mut mesh.normals {
            let v = Vec3::from_array(*normal);
            let transformed = normal_matrix * v;
            let normalized = transformed.normalize_or_zero();
            *normal = normalized.to_array();
        }
    }

    Ok(det.is_sign_negative())
}

fn flip_winding(indices: &mut Vec<u32>) {
    for tri in indices.chunks_exact_mut(3) {
        tri.swap(1, 2);
    }
}

fn axis_index(axis: usize) -> Result<usize, GeometryError> {
    if axis < 3 {
        Ok(axis)
    } else {
        Err(GeometryError::new(
            "axis index must be 0 (x), 1 (y), or 2 (z)",
        ))
    }
}

pub fn center_to_target(mesh: &mut MeshBuffers, target: Vec3) -> Result<Vec3, GeometryError> {
    let (min, max) = bounding_box(&mesh.positions)
        .ok_or_else(|| GeometryError::new("mesh has no positions to center"))?;
    let center = (min + max) * 0.5;
    let translation = target - center;
    translate_mesh(mesh, translation);
    Ok(center)
}

pub fn scale_about_pivot(
    mesh: &mut MeshBuffers,
    scale: Vec3,
    pivot: Vec3,
) -> Result<bool, GeometryError> {
    if (scale.x.abs() < EPSILON) || (scale.y.abs() < EPSILON) || (scale.z.abs() < EPSILON) {
        return Err(GeometryError::new(
            "scale components must be non-zero for mesh scaling",
        ));
    }

    if pivot != Vec3::ZERO {
        translate_mesh(mesh, -pivot);
    }

    let linear = Mat3::from_diagonal(scale);
    let flipped = apply_linear(mesh, linear)?;

    if pivot != Vec3::ZERO {
        translate_mesh(mesh, pivot);
    }

    if flipped {
        flip_winding(&mut mesh.indices);
    }

    Ok(flipped)
}

pub fn flip_axis(mesh: &mut MeshBuffers, axis: usize) -> Result<bool, GeometryError> {
    let axis = axis_index(axis)?;
    let mut scale = Vec3::ONE;
    scale[axis] = -scale[axis];
    scale_about_pivot(mesh, scale, Vec3::ZERO)
}

pub fn swap_axes(
    mesh: &mut MeshBuffers,
    axis_a: usize,
    axis_b: usize,
) -> Result<bool, GeometryError> {
    let a = axis_index(axis_a)?;
    let b = axis_index(axis_b)?;
    if a == b {
        return Ok(false);
    }

    for position in &mut mesh.positions {
        position.swap(a, b);
    }

    for normal in &mut mesh.normals {
        normal.swap(a, b);
    }

    // Swapping two axes is an odd permutation (determinant -1), so winding flips.
    flip_winding(&mut mesh.indices);
    Ok(true)
}

pub fn compute_bounds(mesh: &MeshBuffers) -> Option<(Vec3, Vec3)> {
    bounding_box(&mesh.positions)
}
