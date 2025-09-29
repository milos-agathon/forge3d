//! UV Unwrap helpers (F10): planar and spherical projections.

use crate::geometry::MeshBuffers;
#[cfg(feature = "extension-module")]
use pyo3::{prelude::*, types::PyDict};
#[cfg(feature = "extension-module")]
use crate::geometry::{mesh_from_python, mesh_to_python};

fn compute_bounds(positions: &[[f32; 3]]) -> Option<([f32; 3], [f32; 3])> {
    if positions.is_empty() { return None; }
    let mut min = positions[0];
    let mut max = positions[0];
    for p in positions.iter() {
        for a in 0..3 {
            if p[a] < min[a] { min[a] = p[a]; }
            if p[a] > max[a] { max[a] = p[a]; }
        }
    }
    Some((min, max))
}

fn planar_unwrap(mesh: &mut MeshBuffers, axis: usize) {
    // axis: dropped axis (0=X -> use YZ, 1=Y -> use XZ, 2=Z -> use XY)
    let (u_idx, v_idx) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let Some((min, max)) = compute_bounds(&mesh.positions) else {
        mesh.uvs.clear();
        return;
    };
    let du = (max[u_idx] - min[u_idx]).abs().max(1e-8);
    let dv = (max[v_idx] - min[v_idx]).abs().max(1e-8);
    mesh.uvs = mesh
        .positions
        .iter()
        .map(|p| [ (p[u_idx] - min[u_idx]) / du, (p[v_idx] - min[v_idx]) / dv ])
        .collect();
}

fn spherical_unwrap(mesh: &mut MeshBuffers) {
    if mesh.positions.is_empty() {
        mesh.uvs.clear();
        return;
    }
    // Center at bbox center; radius from max extent
    let (min, max) = compute_bounds(&mesh.positions).unwrap();
    let center = [ (min[0]+max[0])*0.5, (min[1]+max[1])*0.5, (min[2]+max[2])*0.5 ];
    let rx = (max[0]-min[0]) * 0.5;
    let ry = (max[1]-min[1]) * 0.5;
    let rz = (max[2]-min[2]) * 0.5;
    let r = rx.max(ry).max(rz).max(1e-8);
    mesh.uvs = mesh.positions.iter().map(|p| {
        let x = (p[0] - center[0]) / r;
        let y = (p[1] - center[1]) / r;
        let z = (p[2] - center[2]) / r;
        let theta = y.clamp(-1.0, 1.0).asin(); // [-pi/2, pi/2]
        let phi = z.atan2(x); // [-pi, pi]
        let u = (phi + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
        let v = (theta + std::f32::consts::FRAC_PI_2) / std::f32::consts::PI; // [0,1]
        [u, v]
    }).collect();
}

#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn uv_planar_unwrap_py(py: Python<'_>, mesh: &Bound<'_, PyDict>, axis: Option<usize>) -> PyResult<PyObject> {
    let mut m = mesh_from_python(mesh)?;
    let ax = axis.unwrap_or(2).min(2);
    planar_unwrap(&mut m, ax);
    mesh_to_python(py, &m)
}

#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn uv_spherical_unwrap_py(py: Python<'_>, mesh: &Bound<'_, PyDict>) -> PyResult<PyObject> {
    let mut m = mesh_from_python(mesh)?;
    spherical_unwrap(&mut m);
    mesh_to_python(py, &m)
}
